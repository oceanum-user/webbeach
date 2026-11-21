// SPDX-License-Identifier: LGPL-3.0-or-later
// XBeach WebGPU solver — derived from XBeach (c) Deltares (LGPL v2.1+).
// Copyright (c) 2026 Oceanum. See LICENSE.
/**
 * Wave boundary + reduction kernels.
 * Stationary params BC: boundaryconditions.F90:404-421 (ee(1,j,:)=e01·taper).
 * Reductions: dtw/CFL minima and stationary convergence errors via atomics on
 * bit-ordered u32 (positive IEEE-754 floats preserve order as u32).
 */

import { preamble, WAVE_UNIFORMS_WGSL, type GridConst } from "./common.ts";
import { BUFFER_DECLS_WGSL, fieldAccessorsWGSL } from "../fields.ts";

function header(g: GridConst): string {
  return preamble(g) + WAVE_UNIFORMS_WGSL + BUFFER_DECLS_WGSL + fieldAccessorsWGSL();
}

// red layout: [0]=dtwMin/dtMin global, [1]=herrMax, [2]=terrMax, [3]=spare,
//             [8 .. 8+NX] = per-row dtw (stationary solver)
export const RED_DECL = /* wgsl */ `
@group(0) @binding(8) var<storage, read_write> red: array<atomic<u32>>;
fn fbits(x: f32) -> u32 { return bitcast<u32>(max(x, 0.0)); }
fn bitsf(b: u32) -> f32 { return bitcast<f32>(b); }
`;

/**
 * Fill offshore row ee(0,j,:) with e01 (in scratch[64..64+NTHETA)) applying
 * taper factor scratch[4]; also mirrors "if newstatbc: ee=0" host-side.
 * Writes into the CURRENT read buffer (a3) so the next step sees it — matches
 * Fortran wave_bc running before wave().
 */
export function statBcKernel(g: GridConst): string {
  return header(g) + /* wgsl */ `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let kk = i32(gid.x);
  if (kk >= M1 * NTHETA) { return; }
  let j = kk % M1;
  let it = kk / M1;
  let e01 = scratch[64 + it];
  var taper = 1.0;
  if (scratch[22] > 1e-30) { taper = min(TT() / scratch[22], 1.0); }
  a3[PA_ee * NCELL * NTHETA + at3(0, j, it)] = e01 * taper;
}`;
}

/**
 * Per-row dtw CFL init (wave_stationary_directions.F90:257-266):
 *   dtw(i) = 0.5·min( dsu(i-1:i+1,:)/|cgx(i-1:i+1,:,:)|,
 *                     dnv(i,:)/|cgy(i,:,:)|, dtheta/|ctheta(i,:,:)| )
 * Cell (ii,j) contributes its dsu/cgx term to rows ii-1, ii, ii+1 and its
 * dnv/cgy + dtheta/ctheta terms to row ii only. Result in red[8+i].
 */
export function dtwReduceKernel(g: GridConst): string {
  return header(g) + RED_DECL + /* wgsl */ `
const N3: i32 = NCELL * NTHETA;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let cell = i32(gid.x);
  if (cell >= NCELL) { return; }
  let i = cell % N1;
  let j = cell / N1;
  if (j < JMIN_EE || j > JMAX_EE) { return; }
  var dtx = 1e30;
  var dtrest = 1e30;
  for (var it = 0; it < NTHETA; it++) {
    let cgx = abs(v3[V3_cgx * N3 + at3(i, j, it)]);
    let cgy = abs(v3[V3_cgy * N3 + at3(i, j, it)]);
    let ct = abs(v3[V3_ctheta * N3 + at3(i, j, it)]);
    dtx = min(dtx, 0.5 * C2(C2_dsu, cell) / max(1e-6, cgx));
    if (NY > 0) { dtrest = min(dtrest, 0.5 * C2(C2_dnv, cell) / max(1e-6, cgy)); }
    dtrest = min(dtrest, 0.5 * U.dtheta / max(1e-6, ct));
  }
  // dsu/cgx term spreads to i-1, i, i+1 (Fortran window i-1:i+1 per row)
  let bx = fbits(dtx);
  if (i >= 1) { atomicMin(&red[8 + i - 1], bx); }
  atomicMin(&red[8 + i], min(bx, fbits(dtrest)));
  if (i <= NX - 1) { atomicMin(&red[8 + i + 1], bx); }
  atomicMin(&red[0], min(bx, fbits(dtrest)));
}`;
}

/**
 * Per-row sink limit (wave_stationary_directions.F90:489-497):
 *   dtw(i) = min(dtw(i), 0.5·ee/dd) over j,theta with ee>0, dd>0.
 */
export function dtwSinkKernel(g: GridConst): string {
  return header(g) + RED_DECL + /* wgsl */ `
const N3: i32 = NCELL * NTHETA;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let cell = i32(gid.x);
  if (cell >= NCELL) { return; }
  let i = cell % N1;
  let j = cell / N1;
  if (j < JMIN_EE || j > JMAX_EE) { return; }
  if (S2(S2_wete, cell) != 1.0) { return; }
  var lim = 1e30;
  let E = max(S2(S2_E, cell), 0.00001);
  let Dsum = S2(S2_D, cell);
  let Dfr = S2(S2_Df, cell);
  for (var it = 0; it < NTHETA; it++) {
    let e = a3[PA_ee * N3 + at3(i, j, it)];
    let dd = e * (Dsum + Dfr) / E;
    if (e > 0.0 && dd > 0.0) { lim = min(lim, 0.5 * e / dd); }
  }
  atomicMin(&red[8 + i], fbits(lim));
}`;
}

/** Convergence error: max|H-Hprev|, theta error, and roller max|RH-RHprev|
 * (RH in wave-height units). prevs in scratch at 1024(+NCELL, +2*NCELL). */
export function errReduceKernel(g: GridConst): string {
  return header(g) + RED_DECL + /* wgsl */ `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let cell = i32(gid.x);
  if (cell >= NCELL) { return; }
  let i = cell % N1;
  let j = cell / N1;
  let H = S2(S2_H, cell);
  let tm = S2(S2_thetamean, cell);
  let RH = sqrt(S2(S2_R, cell) / U.rhog8);
  let hp = scratch[1024 + cell];
  let tp = scratch[1024 + NCELL + cell];
  let rp = scratch[1024 + 2 * NCELL + cell];
  scratch[1024 + cell] = H;
  scratch[1024 + NCELL + cell] = tm;
  scratch[1024 + 2 * NCELL + cell] = RH;
  if (i < IMIN_EE || i > IMAX_EE || j < JMIN_EE || j > JMAX_EE) { return; }
  if (S2(S2_wete, cell) != 1.0) { return; }
  atomicMax(&red[1], fbits(abs(hp - H)));
  let terr = min(abs(tp - (tm - 2.0 * U.px)), min(abs(tp - tm), abs(tp - (tm + 2.0 * U.px))));
  atomicMax(&red[2], fbits(terr));
  atomicMax(&red[3], fbits(abs(rp - RH)));
}`;
}

/**
 * Bichromatic surfbeat BC (boundaryconditions.F90:426-437):
 * ee(0,j,:) = e01·½(1+cos(2π(t/Tlong − (sinθ0·(yz−yzg11)+cosθ0·(xz−xzg11))/Llong)))·taper
 * bound long wave: bi = −(2cg/c−½)(em−ei)/(cg²−g·hh)/ρ;
 * ui = cg·bi/ht·cos(θ0−alfaz)·lwave·(order−1).
 * scratch: [4]=taperfac, [10]=Llong, [11]=Tlong, [12]=lwave·(order−1),
 * [64..64+NTHETA)=e01. One thread per j.
 */
export function bichromBcKernel(g: GridConst): string {
  return header(g) + /* wgsl */ `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let j = i32(gid.x);
  if (j >= M1) { return; }
  var taper = 1.0;
  if (scratch[22] > 1e-30) { taper = min(TT() / scratch[22], 1.0); }
  let Llong = scratch[10];
  let Tlong = scratch[11];
  let cell = at(0, j);
  let xzg11 = C2(C2_xz, at(0, 0));
  let yzg11 = C2(C2_yz, at(0, 0));
  let phase = TT() / Tlong -
    (sin(U.theta0) * (C2(C2_yz, cell) - yzg11) + cos(U.theta0) * (C2(C2_xz, cell) - xzg11)) /
      Llong;
  var ei = 0.0;
  var e01sum = 0.0;
  for (var it = 0; it < NTHETA; it++) {
    let e01 = scratch[64 + it];
    let e = e01 * 0.5 * (1.0 + cos(2.0 * U.px * phase)) * taper;
    a3[PA_ee * NCELL * NTHETA + at3(0, j, it)] = e;
    ei += e;
    e01sum += 0.5 * e01;
  }
  ei = ei * U.dtheta;
  let em = e01sum * U.dtheta * taper;
  let cg = S2(S2_cg, cell);
  let c = S2(S2_c, cell);
  let hh = S2(S2_hh, cell);
  let bi = -(2.0 * cg / c - 0.5) * (em - ei) / (cg * cg - U.g * hh) / U.rho;
  let ht = S2(S2_zs0, cell) - S2(S2_zb, cell);
  S2set(S2_ui, cell, cg * bi / ht * cos(U.theta0 - C2(C2_alfaz, cell)) * scratch[12]);
  S2set(S2_vi, cell, 0.0);
}`;
}

/**
 * Generic wave BC row apply (spectral reuse path): host uploads
 * eeRow at scratch[BCROW .. BCROW+M1*NTHETA) (j-major), uiRow at
 * scratch[BCROW+M1*NTHETA ..+M1), viRow following. Writes a3 + ui/vi.
 * BCROW = 1024 + 3*NCELL (after the errReduce prev-state blocks).
 */
export function bcRowApplyKernel(g: GridConst): string {
  return header(g) + /* wgsl */ `
const BCROW: i32 = 1024 + 3 * NCELL;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let kk = i32(gid.x);
  let nee = M1 * NTHETA;
  if (kk < nee) {
    let j = kk % M1;
    let it = kk / M1;
    a3[PA_ee * NCELL * NTHETA + at3(0, j, it)] = scratch[BCROW + j * NTHETA + it];
    return;
  }
  let j = kk - nee;
  if (j < M1) {
    S2set(S2_ui, at(0, j), scratch[BCROW + nee + j]);
    S2set(S2_vi, at(0, j), scratch[BCROW + nee + M1 + j]);
  }
}`;
}

/**
 * OCE-252: the single_dir stationary boundary spread — ee_s(0,j,:) from the
 * Es table held in bnd at scratch[27] ((ny+1)*ntheta_s, j fastest), written
 * into BOTH ping-pong buffers of the ee_s set so the sweep's row 0 never
 * moves. Not tapered: Fortran sets ee_s(1,:,:) once from the spectrum
 * (set_stationary_spectrum) and never scales it. Compiled with the
 * directions GridConst (NTHETA = ntheta_s) and dispatched on the ee_s set.
 */
export function dirBcApplyKernel(g: GridConst): string {
  return header(g) + /* wgsl */ `
@group(0) @binding(9) var<storage, read> bnd: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let kk = i32(gid.x);
  if (kk >= M1 * NTHETA) { return; }
  let j = kk % M1;
  let it = kk / M1;
  let esoff = i32(scratch[27]);
  let v = bnd[esoff + it * M1 + j];
  a3[PA_ee * NCELL * NTHETA + at3(0, j, it)] = v;
  b3[PA_ee * NCELL * NTHETA + at3(0, j, it)] = v;
}`;
}

/**
 * Mean-accumulator slot count (readMeans layout; the WATERLINE block follows
 * at MEANS + NMEAN_SLOTS*NCELL). Shared with model.ts so the scratch
 * allocation and readback offsets cannot drift from the kernels.
 */
export const NMEAN_SLOTS = 16;

/** waterline accumulator slots per cell (bc.ts waterlineAccKernel) */
export const NWL_SLOTS = 4;
/** the wet threshold the waterline gauge walks on (m) — B2 §7; the
 * grid-resolution sensitivity is noted there. One named value shared by
 * the kernel and every host that interprets its statistics. */
export const WATERLINE_DWET_M = 0.05;

/**
 * Mean accumulators (varoutput-mean semantics): acc += dt·f for
 * [zs-czs, (zs-czs)^2 (czs = max(zb,0)), H^2, u, v, ue, ve, urms^2, D, dt, Qb, DR, R] at scratch offset
 * MEANS. Host divides by the dt slot at mean-output times ("mean" of
 * H/urms = rms). Slots 10-12 (Qb, roller dissipation, roller energy Er)
 * appended for the surf-zone delineation products (varoutput-mean parity:
 * Fortran means.F90). Slot 13 (E4) is a running MAX, not dt-weighted:
 * varianceupdate.F90 max2d = max(max2d, tvar), seeded 0 on window reset
 * (clearBuffer; |u| >= 0 so 0 is the max identity).
 */
export function meanAccKernel(g: GridConst): string {
  return header(g) + /* wgsl */ `
const BCROW: i32 = 1024 + 3 * NCELL;
const MEANS: i32 = BCROW + M1 * NTHETA + 2 * M1;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let kk = i32(gid.x);
  if (kk >= NCELL) { return; }
  let dt = DT();
  // zs accumulates as the DEVIATION from the static per-cell level it
  // sits near: zb on land (dry zs ≈ zb reaches tens of metres), 0 in
  // water (zs ≈ tide). Raw f32 running sums of zs² lose ~0.5 m² to
  // rounding wherever |zs| is tens of metres (found by the mission parity
  // harness on a 66 m-relief survey: zs_var read 46% nRMSE vs f64
  // Fortran; and zs - zb alone just moves the loss into deep water).
  // var is shift-invariant; hosts add the same max(zb, 0) back to the mean.
  let dzs = S2(S2_zs, kk) - max(S2(S2_zb, kk), 0.0);
  scratch[MEANS + 0 * NCELL + kk] += dt * dzs;
  scratch[MEANS + 1 * NCELL + kk] += dt * dzs * dzs;
  scratch[MEANS + 2 * NCELL + kk] += dt * S2(S2_H, kk) * S2(S2_H, kk);
  scratch[MEANS + 3 * NCELL + kk] += dt * S2(S2_u, kk);
  scratch[MEANS + 4 * NCELL + kk] += dt * S2(S2_v, kk);
  scratch[MEANS + 5 * NCELL + kk] += dt * S2(S2_ue, kk);
  scratch[MEANS + 6 * NCELL + kk] += dt * S2(S2_ve, kk);
  scratch[MEANS + 7 * NCELL + kk] += dt * S2(S2_urms, kk) * S2(S2_urms, kk);
  scratch[MEANS + 8 * NCELL + kk] += dt * S2(S2_D, kk);
  scratch[MEANS + 9 * NCELL + kk] += dt; // actual accumulated window length
  scratch[MEANS + 10 * NCELL + kk] += dt * S2(S2_Qb, kk);
  scratch[MEANS + 11 * NCELL + kk] += dt * S2(S2_DR, kk);
  scratch[MEANS + 12 * NCELL + kk] += dt * S2(S2_R, kk);
  // E4: running max of instantaneous GLM |u| at the cell centre. XBeach's
  // velocity-magnitude diagnostics default to GLM (variables.def
  // vmagu/vmagv; the Eulerian ones are the separate vmageu/vmagev), so
  // GLM u/v, not ue/ve. One thread per cell -> plain max is race-free; a
  // dt=0 overshoot step freezes the fields, so the max stays an identity.
  let umag = length(vec2<f32>(S2(S2_u, kk), S2(S2_v, kk)));
  scratch[MEANS + 13 * NCELL + kk] = max(scratch[MEANS + 13 * NCELL + kk], umag);
  // OCE-243: time-mean short-wave energy flux vector (W/m, grid frame),
  // from the integrate kernel's Σ ee·cg dθ — the incident run-up term
  // reads its onshore component at a shore point's sample station
  scratch[MEANS + 14 * NCELL + kk] += dt * S2(S2_Px, kk);
  scratch[MEANS + 15 * NCELL + kk] += dt * S2(S2_Py, kk);
}`;
}

/**
 * Waterline gauge (E5, OCE-243): one thread per CELL. Whenever the cell is
 * wet (hh > d_wet, wetz) with a dry 4-neighbour it IS the instantaneous
 * waterline, and the water level there is accumulated: [sum dt*dz,
 * sum dt*dz^2, max dz, sum dt] with dz = zs - zb, the level above the
 * cell's own bed (a waterline cell holds a thin lens, so dz stays small
 * and f32 keeps its square — the same cancellation guard as meanAcc's zs
 * slots), at offset WL = MEANS + NMEAN_SLOTS*NCELL, stride NCELL per
 * quantity. Hosts add zb back per cell and recombine cells into a shore
 * point's set-up, swash and maximum in f64.
 *
 * It replaces the per-row gauge, which walked each grid row from the land
 * side to the first wet cell: a row crossing a cusp or a spit gauged the
 * pocket or the bay behind the beach and the host attributed that to the
 * ocean waterline (Newport, 2026-10-15: 93 of 263 rows). Per cell there
 * is no direction to assume — every shore the grid resolves gauges itself,
 * islands and lee sides included. Grid-boundary cells are left out.
 */
export function waterlineAccKernel(g: GridConst): string {
  return header(g) + /* wgsl */ `
const BCROW: i32 = 1024 + 3 * NCELL;
const MEANS: i32 = BCROW + M1 * NTHETA + 2 * M1;
const WL: i32 = MEANS + ${NMEAN_SLOTS} * NCELL;
const DWET: f32 = ${WATERLINE_DWET_M};
fn wetAt(kk: i32) -> bool { return S2(S2_hh, kk) > DWET && S2(S2_wetz, kk) == 1.0; }
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let kk = i32(gid.x);
  if (kk >= NCELL) { return; }
  let i = kk % N1;
  let j = kk / N1;
  if (i <= 0 || j <= 0 || i >= N1 - 1 || j >= M1 - 1) { return; }
  if (!wetAt(kk)) { return; }
  let dryNb = !wetAt(kk - 1) || !wetAt(kk + 1) || !wetAt(kk - N1) || !wetAt(kk + N1);
  if (!dryNb) { return; }
  let dt = DT();
  let dz = S2(S2_zs, kk) - S2(S2_zb, kk);
  scratch[WL + 0 * NCELL + kk] += dt * dz;
  scratch[WL + 1 * NCELL + kk] += dt * dz * dz;
  scratch[WL + 2 * NCELL + kk] = max(scratch[WL + 2 * NCELL + kk], dz);
  scratch[WL + 3 * NCELL + kk] += dt;
}`;
}


/**
 * GPU-resident timestep (gpudt mode): consumes the CFL reductions
 * (red[0]=flow minimum, red[3]=dtheta/max|ctheta|), applies timestep.F90
 * logic (CFL*0.5, refraction criterion, chop so an integer number of steps
 * lands on tnext), advances t, publishes dt/t in scratch[16/17], and resets
 * the reduction slots for the next step. Past tnext, dt=0: every physics
 * kernel becomes an exact identity, so hosts may overshoot safely.
 * scratch: [23]=tnext, [24]=CFL, [25]=theta-criterion flag.
 */
export function dtStepKernel(g: GridConst): string {
  return header(g) + RED_DECL + /* wgsl */ `
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 1u, 0u, 0u);
  if (gid.x > 0u) { return; }
  let t = scratch[17];
  let tnext = scratch[23];
  let CFL = scratch[24];
  if (t >= tnext - 1e-6) {
    scratch[16] = 0.0;
  } else {
    var dt = bitsf(atomicLoad(&red[0])) * CFL * 0.5;
    if (scratch[25] > 0.5) {
      dt = min(dt, CFL * bitsf(atomicLoad(&red[3])));
    }
    if (tnext > t) {
      let n = ceil((tnext - t) / dt);
      dt = (tnext - t) / n;
    }
    var tn = t + dt;
    if (tn >= tnext - 1e-6) {
      dt = dt - (tn - tnext);
      tn = tnext;
    }
    scratch[16] = dt;
    scratch[17] = tn;
  }
  atomicStore(&red[0], 0x7f7fffffu);
  atomicStore(&red[3], 0x7f7fffffu);
}`;
}

/**
 * GPU-resident spectral wave BC (reuse semantics of
 * boundaryconditions.F90:594-664): interpolates the E_series/q_series record
 * tables (uploaded once per condition into the bnd buffer, binding 9) at
 * t=TT() and writes the boundary row + ui/vi. Layout of bnd:
 *   [ E records: nrec * (m1*ntheta), Fortran order j fastest within theta ]
 *   [ q records: nrec * (m1*4) ]
 * scratch: [19]=dtbc, [20]=nrec, [21]=q offset (floats), [22]=taper.
 */
export function spectralBcKernel(g: GridConst): string {
  return header(g) + /* wgsl */ `
@group(0) @binding(9) var<storage, read> bnd: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let kk = i32(gid.x);
  let nee = M1 * NTHETA;
  let dtbc = scratch[19];
  let nrec = i32(scratch[20]);
  let qoff = i32(scratch[21]);
  let t = TT();
  var taper = 1.0;
  if (scratch[22] > 1e-30) { taper = min(t / scratch[22], 1.0); }
  let newIdx = i32(floor(t / dtbc));
  let r1 = min(newIdx, nrec - 1);
  let r2 = min(newIdx + 1, nrec - 1);
  let tnew = f32(newIdx + 1) * dtbc;
  let w2 = (dtbc - (tnew - t)) / dtbc;
  let w1 = (tnew - t) / dtbc;
  if (kk < nee) {
    let j = kk % M1;
    let it = kk / M1;
    let src = it * M1 + j;
    let v = w2 * bnd[r2 * nee + src] + w1 * bnd[r1 * nee + src];
    a3[PA_ee * NCELL * NTHETA + at3(0, j, it)] = v * taper;
    return;
  }
  let j = kk - nee;
  if (j < M1) {
    let recQ = M1 * 4;
    let qx = w2 * bnd[qoff + r2 * recQ + 0 * M1 + j] +
      w1 * bnd[qoff + r1 * recQ + 0 * M1 + j];
    let qy = w2 * bnd[qoff + r2 * recQ + 1 * M1 + j] +
      w1 * bnd[qoff + r1 * recQ + 1 * M1 + j];
    let cell = at(0, j);
    let ht = S2(S2_zs0, cell) - S2(S2_zb, cell);
    let ca = cos(C2(C2_alfau, cell));
    let sa = sin(C2(C2_alfau, cell));
    S2set(S2_ui, cell, (qx * ca + qy * sa) / ht * taper);
    S2set(S2_vi, cell, (-qx * sa + qy * ca) / ht * taper);
  }
}`;
}
