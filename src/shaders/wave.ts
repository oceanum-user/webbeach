// SPDX-License-Identifier: LGPL-3.0-or-later
// XBeach WebGPU solver — derived from XBeach (c) Deltares (LGPL v2.1+).
// Copyright (c) 2026 Oceanum. See LICENSE.
/**
 * Wave module WGSL kernels — port of:
 *   wave_timestep.F90 (wave)              — hhw/hstokes prep
 *   wave_functions.F90 (dispersion, slope2D, compute_wave_direction_velocities,
 *                       advec*ho, compute_wave_forces, compute_stokes_drift,
 *                       breakerdelay)
 *   wave_instationary.F90                 — surfbeat step
 *   wave_stationary_directions.F90        — stationary (whole-grid relaxation,
 *                                           same discrete fixed point; see
 *                                           docs/deviations.md)
 *   roelvink.F90                          — breaker formulations
 *   wetcells.F90                          — wet/dry masks
 *
 * All kernels share bind group layout: U, c2, s2, c3, a3, b3, v3, scratch.
 */

import { preamble, WAVE_UNIFORMS_WGSL, type GridConst } from "./common.ts";
import { BUFFER_DECLS_WGSL, fieldAccessorsWGSL } from "../fields.ts";

function header(g: GridConst): string {
  return preamble(g) + WAVE_UNIFORMS_WGSL + BUFFER_DECLS_WGSL + fieldAccessorsWGSL() +
    /* wgsl */ `
// OCE-252: the wave depth this kernel set works on — hhw, or the
// wavint-smoothed hhws when compiled for the single_dir directions sweep
// (hhwlocal in wave_stationary_directions.F90:171-180)
fn HW(k: i32) -> f32 {
  if (DIRS == 1) { return S2(S2_hhws, k); }
  return S2(S2_hhw, k);
}
`;
}

const RED_RO = /* wgsl */ `
@group(0) @binding(8) var<storage, read_write> red: array<atomic<u32>>;
fn rowDtw(i: i32) -> f32 { return bitcast<f32>(atomicLoad(&red[8 + i])); }
`;

/** wetcells.F90: hum/hvm/wetu/wetv/wetz + base wete. */
export function wetcellsA(g: GridConst): string {
  return header(g) + /* wgsl */ `
const NUMEPS: f32 = 1.2e-7; // epsilon(0.) in f32

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let kk = i32(gid.x);
  if (kk >= NCELL) { return; }
  let i = kk % N1;
  let j = kk / N1;
  let ip = min(i + 1, NX);       // Fortran min(nx,i)+1 with 1-based -> clamp
  let jp = min(j + 1, NY);
  let hh = S2(S2_hh, kk);
  // wetcells.F90:66
  let hum = max(0.5 * (hh + S2(S2_hh, at(ip, j))), U.eps);
  S2set(S2_hum, kk, hum);
  // :75
  let hu = S2(S2_hu, kk);
  var wetu = 0.0;
  if (hu > U.eps + NUMEPS && hum > U.eps + NUMEPS) { wetu = 1.0; }
  S2set(S2_wetu, kk, wetu);
  // :90
  let hvm = max(0.5 * (hh + S2(S2_hh, at(i, jp))), U.eps);
  S2set(S2_hvm, kk, hvm);
  let hv = S2(S2_hv, kk);
  var wetv = 0.0;
  if (hv > U.eps + NUMEPS && hvm > U.eps + NUMEPS) { wetv = 1.0; }
  S2set(S2_wetv, kk, wetv);
  // :108
  var wetz = 0.0;
  if (hh > U.eps + NUMEPS) { wetz = 1.0; }
  S2set(S2_wetz, kk, wetz);
  // :117 base wete (dilation in pass B)
  var wete = 0.0;
  if (hh + U.deltaH * S2(S2_H, kk) > U.eps || wetz == 1.0) { wete = 1.0; }
  S2set(S2_wete, kk, wete);
}`;
}

/** wetcells.F90:122-143 — dilate wete by upwind distance using wetz/wetu/wetv. */
export function wetcellsB(g: GridConst): string {
  return header(g) + /* wgsl */ `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let kk = i32(gid.x);
  if (kk >= NCELL) { return; }
  if (S2(S2_wete, kk) != 0.0) { return; }
  let i = kk % N1;
  let j = kk / N1;
  var updist = 2;
  if (U.scheme == 1.0) { updist = 1; }
  var w = 0.0;
  for (var d = -updist; d <= updist; d++) {
    w = max(w, S2(S2_wetz, at(clamp(i + d, 0, NX), j)));
    w = max(w, S2(S2_wetz, at(i, clamp(j + d, 0, NY))));
  }
  w = max(w, S2(S2_wetu, at(max(i - 1, 0), j)));
  w = max(w, S2(S2_wetu, at(min(i, NX), j)));
  w = max(w, S2(S2_wetv, at(i, max(j - 1, 0))));
  w = max(w, S2(S2_wetv, at(i, min(j, NY))));
  S2set(S2_wete, kk, w);
}`;
}

/** wave_timestep.F90:56-73 — hhw and hstokes. */
export function wavePrep(g: GridConst): string {
  return header(g) + /* wgsl */ `
struct Extra { deltahmin: f32, oldhmin: f32, hmin: f32, pad: f32 };

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let kk = i32(gid.x);
  if (kk >= NCELL) { return; }
  let hh = S2(S2_hh, kk);
  let H = S2(S2_H, kk);
  if (U.deltaH > 0.0) {
    S2set(S2_hhw, kk, max(hh + U.deltaH * H, U.eps));
  } else {
    S2set(S2_hhw, kk, max(hh, U.eps));
  }
  // oldhmin handling: scratch[0] = oldhmin flag, scratch[1] = hmin
  if (scratch[0] == 1.0) {
    S2set(S2_hstokes, kk, max(hh, scratch[1]));
  } else {
    let gam = H / hh;
    if (gam > 1.0) {
      S2set(S2_hstokes, kk, scratch[2] * (gam - 1.0) * H + hh); // scratch[2]=deltahmin
    } else {
      S2set(S2_hstokes, kk, hh);
    }
  }
}`;
}

/**
 * wave_functions.F90:685-798 (dispersion) + 925-1102 (wave_dispersion, no-wci
 * branch). Depth argument choice handled by host (writes chosen depth into
 * scratch slot? no — uses hhw directly; single_dir/wci variants later).
 * sigm/sigt updates handled host-side via uniform sigmval (2π/Trep).
 */
export function dispersionKernel(g: GridConst): string {
  return header(g) + /* wgsl */ `
const APHI: f32 = ${1 / ((1 + Math.sqrt(5)) / 2 + 1)};
const BPHI: f32 = ${(1 + Math.sqrt(5)) / 2 / ((1 + Math.sqrt(5)) / 2 + 1)};

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let kk = i32(gid.x);
  if (kk >= NCELL) { return; }
  let wete = S2(S2_wete, kk);
  let h = HW(kk); // hhws for the directions sweep (wave_dispersion switch 1)
  S2set(S2_sigm, kk, U.sigmval);
  if (wete == 1.0) {
    let L0 = U.g * U.Trep * U.Trep / (2.0 * U.px);
    // Root of sigma^2 = g k tanh(kh) — same root as Fortran's golden-ratio
    // iteratedispersion (wave_functions.F90:800-837) but solved by Newton for
    // f32 accuracy (see docs/deviations.md#dispersion).
    var L1 = S2(S2_L1, kk);
    if (L1 <= 0.0) { L1 = L0; }
    let sig = U.sigmval;
    var k = 2.0 * U.px / L1; // warm start
    for (var itn = 0; itn < 8; itn++) {
      let kh = min(k * h, 20.0);
      let th = tanh(kh);
      let ch = cosh(min(kh, 10.0));
      let f = sig * sig - U.g * k * th;
      let df = -U.g * (th + kh / (ch * ch));
      let dk = f / df;
      k = k - dk;
      if (abs(dk) < 1e-7 * k) { break; }
    }
    L1 = 2.0 * U.px / k;
    S2set(S2_L1, kk, L1);
    let c = U.sigmval / k;
    let kh = min(k * h, 10.0);
    let n = 0.5 + kh / sinh(2.0 * kh);
    S2set(S2_k, kk, k);
    S2set(S2_c, kk, c);
    S2set(S2_nn, kk, n);
    S2set(S2_cg, kk, c * n);
  } else {
    S2set(S2_k, kk, 25.0);
    S2set(S2_c, kk, sqrt(U.g * U.eps));
    S2set(S2_nn, kk, 1.0);
    S2set(S2_cg, kk, sqrt(U.g * U.eps));
    S2set(S2_L1, kk, U.eps);
  }
}`;
}

/**
 * slope2D on hhw (wave_functions.F90:52-112) + sinh2kh + thetamean pre-update
 * (wave_instationary.F90:140-165, snells==0 branch energy-weighted mean).
 * statMode: slope limiting to |0.1| (wave_stationary_directions.F90:162).
 */
export function slopesKernel(g: GridConst): string {
  return header(g) + /* wgsl */ `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let kk = i32(gid.x);
  if (kk >= NCELL) { return; }
  let i = kk % N1;
  let j = kk / N1;
  let wete = S2(S2_wete, kk);
  var dhdx = 0.0;
  var dhdy = 0.0;
  if (wete == 1.0) {
    // u-gradients (h = hhw here; hhwlocal without wci)
    if (i >= 1 && i <= NX - 1) {
      dhdx = (HW(at(i + 1, j)) - HW(at(i - 1, j))) /
        (C2(C2_dsu, kk) + C2(C2_dsu, at(i - 1, j)));
    } else if (i == 0) {
      dhdx = (HW(at(1, j)) - HW(at(0, j))) / C2(C2_dsu, kk);
    } else {
      dhdx = (HW(at(NX, j)) - HW(at(NX - 1, j))) / C2(C2_dsu, at(NX - 1, j));
    }
    if (NY > 0) {
      if (j >= 1 && j <= NY - 1) {
        dhdy = (HW(at(i, j + 1)) - HW(at(i, j - 1))) /
          (C2(C2_dnv, kk) + C2(C2_dnv, at(i, j - 1)));
      } else if (j == 0) {
        dhdy = (HW(at(i, 1)) - HW(at(i, 0))) / C2(C2_dnv, kk);
      } else {
        dhdy = (HW(at(i, NY)) - HW(at(i, NY - 1))) / C2(C2_dnv, at(i, NY - 1));
      }
    }
  }
  if (U.statMode == 1.0) {
    dhdx = sign(dhdx) * min(abs(dhdx), 0.1);
    dhdy = sign(dhdy) * min(abs(dhdy), 0.1);
  }
  S2set(S2_dhdx, kk, dhdx);
  S2set(S2_dhdy, kk, dhdy);
  // sinh2kh (wave_instationary.F90:180-184)
  let hloc = HW(kk);
  let k = S2(S2_k, kk);
  var s2kh = 3000.0;
  if (2.0 * hloc * k <= 3000.0) { s2kh = sinh(min(2.0 * k * hloc, 10.0)); }
  if (U.statMode == 1.0 && wete != 1.0) { s2kh = 3000.0; }
  S2set(S2_sinh2kh, kk, s2kh);
  // thetamean pre-update, energy-weighted (snells=0, single_dir=0); the
  // directions sweep recomputes it from ee_s/thet_s (callTypeDirections)
  if ((U.snells == 0.0 && U.singledir == 0.0) || DIRS == 1) {
    var se = 0.0;
    var set_ = 0.0;
    for (var it = 0; it < NTHETA; it++) {
      let e = PA(PA_ee, at3(i, j, it));
      se += e;
      set_ += e * C3(C3_thet, at3(i, j, it));
    }
    if (U.statMode == 0.0 || wete == 1.0) {
      S2set(S2_thetamean, kk, (set_ / f32(NTHETA)) / (max(se, 0.00001) / f32(NTHETA)));
    }
  } else if (U.snells == 1.0) {
    // Snell (wave_instationary.F90:146-164): c offshore reference cell (0,0)
    let coff = S2(S2_c, at(0, 0));
    let alfaz = C2(C2_alfaz, kk);
    var tm = U.theta0;
    if (U.refraction == 1.0) {
      let alfaz00 = C2(C2_alfaz, at(0, 0));
      tm = asin(clamp(sin(U.theta0 - alfaz00) * S2(S2_c, kk) / coff, -1.0, 1.0)) + alfaz00;
    }
    S2set(S2_thetamean, kk, tm);
  }
}`;
}

/**
 * compute_wave_direction_velocities (wave_functions.F90:1104-1226), no-wci.
 * ctheta limited to 0.5·pi/Trep. For snells==1/single_dir the host must have
 * costh/sinth(:,:,1) refreshed (kernel below handles snells via thetamean).
 */
export function dirvelKernel(g: GridConst): string {
  return header(g) + /* wgsl */ `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let kk = i32(gid.x);
  if (kk >= NCELL * NTHETA) { return; }
  let cell = kk % NCELL;
  let it = kk / NCELL;
  let i = cell % N1;
  let j = cell / N1;
  var cs: f32;
  var sn: f32;
  if ((U.snells == 1.0 || U.singledir == 1.0) && NTHETA == 1 && DIRS == 0) {
    // the single instationary bin follows the stationary model's mean
    // direction (wave_instationary.F90:140-142)
    let rel = S2(S2_thetamean, cell) - C2(C2_alfaz, cell);
    cs = cos(rel);
    sn = sin(rel);
  } else {
    cs = C3(C3_costh, at3(i, j, it));
    sn = C3(C3_sinth, at3(i, j, it));
  }
  if (S2(S2_wete, cell) == 1.0) {
    let cg = S2(S2_cg, cell);
    let c = S2(S2_c, cell);
    v3[V3_cgx * NCELL * NTHETA + kk] = cg * cs;
    v3[V3_cgy * NCELL * NTHETA + kk] = cg * sn;
    v3[V3_cx * NCELL * NTHETA + kk] = c * cs;
    v3[V3_cy * NCELL * NTHETA + kk] = c * sn;
    var ct = S2(S2_sigm, cell) / S2(S2_sinh2kh, cell) *
      (S2(S2_dhdx, cell) * sn - S2(S2_dhdy, cell) * cs);
    ct = sign(ct) * min(abs(ct), 0.5 * U.px / U.Trep);
    v3[V3_ctheta * NCELL * NTHETA + kk] = ct;
  } else {
    v3[V3_cgx * NCELL * NTHETA + kk] = 0.0;
    v3[V3_cgy * NCELL * NTHETA + kk] = 0.0;
    v3[V3_cx * NCELL * NTHETA + kk] = 0.0;
    v3[V3_cy * NCELL * NTHETA + kk] = 0.0;
    v3[V3_ctheta * NCELL * NTHETA + kk] = 0.0;
  }
}`;
}

/**
 * The wave Euler step: advection (x, y, theta) of action, back-transform,
 * gammax limiter, breaker dissipation, bed friction, roller balance.
 * Port of wave_instationary.F90:189-352 and (statMode) the equivalent
 * operations of wave_stationary_directions.F90:286-528.
 *
 * One thread per cell; theta loop internal. Reads eeA/rrA (+ neighbours),
 * writes eeB/rrB and per-cell 2D fields. QbNew must hold the advected Qb for
 * roelvink_daly (separate kernel) before this runs.
 *
 * Fortran-order note: E,H used by dissipation are the post-advection,
 * post-gammax values of this cell — all computable locally.
 */
export function waveStepKernel(g: GridConst): string {
  return header(g) + RED_RO + /* wgsl */ `
const N3: i32 = NCELL * NTHETA;

fn actionAt(i: i32, j: i32, it: i32) -> f32 {
  // ee -> action. instationary: /sigt(==sigm no-wci); stationary: /sigm
  return PA(PA_ee, at3(i, j, it)) / S2(S2_sigm, at(i, j));
}

// advecxho flux at u-face fi (between cells fi and fi+1), Fortran fluxx(fi+1)
fn xflux(fi: i32, j: i32, it: i32) -> f32 {
  if (fi < 0 || fi >= NX) { return 0.0; }
  if (S2(S2_wete, at(fi, j)) != 1.0) { return 0.0; }
  let cgxu = 0.5 * (v3[V3_cgx * N3 + at3(fi + 1, j, it)] + v3[V3_cgx * N3 + at3(fi, j, it)]);
  let dnu = C2(C2_dnu, at(fi, j));
  if (U.scheme == 1.0) {
    if (cgxu > 0.0) { return actionAt(fi, j, it) * cgxu * dnu; }
    return actionAt(fi + 1, j, it) * cgxu * dnu;
  }
  // upwind_2 / warmbeam (advecxho SCHEME_UPWIND_2 branch, wave_functions.F90:175-231)
  if (fi >= 1 && fi <= NX - 2) {
    if (cgxu > 0.0) {
      let dsu_m = C2(C2_dsu, at(fi - 1, j));
      let dsu_0 = C2(C2_dsu, at(fi, j));
      var eupw = ((dsu_m + 0.5 * dsu_0) * actionAt(fi, j, it) -
        0.5 * dsu_0 * actionAt(fi - 1, j, it)) / dsu_m;
      if (eupw < 0.0) { eupw = actionAt(fi, j, it); }
      return eupw * cgxu * dnu;
    } else {
      // stationary window uses first order here for the updated row's right
      // face under negative flow (advecxho called with nx=3 window)
      if (U.statMode == 1.0) { return actionAt(fi + 1, j, it) * cgxu * dnu; }
      let dsu_p = C2(C2_dsu, at(fi + 1, j));
      let dsu_0 = C2(C2_dsu, at(fi, j));
      var eupw = ((dsu_p + 0.5 * dsu_0) * actionAt(fi + 1, j, it) -
        0.5 * dsu_0 * actionAt(fi + 2, j, it)) / dsu_p;
      if (eupw < 0.0) { eupw = actionAt(fi + 1, j, it); }
      return eupw * cgxu * dnu;
    }
  } else if (fi == 0) {
    if (cgxu > 0.0) { return actionAt(0, j, it) * cgxu * dnu; }
    let dsu_p = C2(C2_dsu, at(1, j));
    let dsu_0 = C2(C2_dsu, at(0, j));
    var eupw = ((dsu_p + 0.5 * dsu_0) * actionAt(1, j, it) -
      0.5 * dsu_0 * actionAt(2, j, it)) / dsu_p;
    if (eupw < 0.0) { eupw = actionAt(1, j, it); }
    return eupw * cgxu * dnu;
  } else { // fi == NX-1
    if (cgxu > 0.0) {
      let dsu_m = C2(C2_dsu, at(NX - 2, j));
      let dsu_0 = C2(C2_dsu, at(NX - 1, j));
      var eupw = ((dsu_m + 0.5 * dsu_0) * actionAt(NX - 1, j, it) -
        0.5 * dsu_0 * actionAt(NX - 2, j, it)) / dsu_m;
      if (eupw < 0.0) { eupw = actionAt(NX - 1, j, it); }
      return eupw * cgxu * dnu;
    }
    return actionAt(NX, j, it) * cgxu * dnu;
  }
}

// advecyho flux at v-face fj (between cells fj and fj+1), Fortran fluxy(fj+1)
// stationary solver forces upwind_1 in y (wave_stationary_directions.F90:108)
fn yflux(i: i32, fj: i32, it: i32) -> f32 {
  if (fj < 0 || fj >= NY) { return 0.0; }
  if (S2(S2_wete, at(i, fj)) != 1.0) { return 0.0; }
  let cgyv = 0.5 * (v3[V3_cgy * N3 + at3(i, fj + 1, it)] + v3[V3_cgy * N3 + at3(i, fj, it)]);
  let dsv = C2(C2_dsv, at(i, fj));
  if (U.scheme == 1.0 || U.statMode == 1.0) {
    if (cgyv > 0.0) { return actionAt(i, fj, it) * cgyv * dsv; }
    return actionAt(i, fj + 1, it) * cgyv * dsv;
  }
  if (fj >= 1 && fj <= NY - 2) {
    if (cgyv > 0.0) {
      let dnv_m = C2(C2_dnv, at(i, fj - 1));
      let dnv_0 = C2(C2_dnv, at(i, fj));
      var eupw = ((dnv_m + 0.5 * dnv_0) * actionAt(i, fj, it) -
        0.5 * dnv_0 * actionAt(i, fj - 1, it)) / dnv_m;
      if (eupw < 0.0) { eupw = actionAt(i, fj, it); }
      return eupw * cgyv * dsv;
    } else {
      let dnv_p = C2(C2_dnv, at(i, fj + 1));
      let dnv_0 = C2(C2_dnv, at(i, fj));
      var eupw = ((dnv_p + 0.5 * dnv_0) * actionAt(i, fj + 1, it) -
        0.5 * dnv_0 * actionAt(i, fj + 2, it)) / dnv_p;
      if (eupw < 0.0) { eupw = actionAt(i, fj + 1, it); }
      return eupw * cgyv * dsv;
    }
  } else if (fj == 0) {
    if (cgyv > 0.0) { return actionAt(i, 0, it) * cgyv * dsv; }
    let dnv_p = C2(C2_dnv, at(i, 1));
    let dnv_0 = C2(C2_dnv, at(i, 0));
    var eupw = ((dnv_p + 0.5 * dnv_0) * actionAt(i, 1, it) -
      0.5 * dnv_0 * actionAt(i, 2, it)) / dnv_p;
    if (eupw < 0.0) { eupw = actionAt(i, 1, it); }
    return eupw * cgyv * dsv;
  } else { // fj == NY-1
    if (cgyv > 0.0) {
      let dnv_m = C2(C2_dnv, at(i, NY - 2));
      let dnv_0 = C2(C2_dnv, at(i, NY - 1));
      var eupw = ((dnv_m + 0.5 * dnv_0) * actionAt(i, NY - 1, it) -
        0.5 * dnv_0 * actionAt(i, NY - 2, it)) / dnv_m;
      if (eupw < 0.0) { eupw = actionAt(i, NY - 1, it); }
      return eupw * cgyv * dsv;
    }
    return actionAt(i, NY, it) * cgyv * dsv;
  }
}

// advecthetaho flux between bins it and it+1 (Fortran fluxtheta(it+1))
fn tflux(i: i32, j: i32, ft: i32) -> f32 {
  if (ft < 0 || ft >= NTHETA - 1) { return 0.0; }
  let ctb = 0.5 * (v3[V3_ctheta * N3 + at3(i, j, ft + 1)] + v3[V3_ctheta * N3 + at3(i, j, ft)]);
  if (U.scheme == 1.0) {
    if (ctb > 0.0) { return actionAt(i, j, ft) * ctb; }
    return actionAt(i, j, ft + 1) * ctb;
  }
  if (ft >= 1 && ft <= NTHETA - 3) {
    if (ctb > 0.0) {
      var eupw = 1.5 * actionAt(i, j, ft) - 0.5 * actionAt(i, j, ft - 1);
      if (eupw < 0.0) { eupw = actionAt(i, j, ft); }
      return eupw * ctb;
    } else {
      var eupw = 1.5 * actionAt(i, j, ft + 1) - 0.5 * actionAt(i, j, ft + 2);
      if (eupw < 0.0) { eupw = actionAt(i, j, ft + 1); }
      return eupw * ctb;
    }
  } else if (ft == 0) {
    if (ctb > 0.0) { return actionAt(i, j, 0) * ctb; }
    var eupw = 1.5 * actionAt(i, j, 1) - 0.5 * actionAt(i, j, 2);
    if (eupw < 0.0) { eupw = actionAt(i, j, 1); }
    return eupw * ctb;
  } else { // ft == NTHETA-2
    if (ctb > 0.0) {
      var eupw = 1.5 * actionAt(i, j, ft) - 0.5 * actionAt(i, j, ft - 1);
      if (eupw < 0.0) { eupw = actionAt(i, j, ft); }
      return eupw * ctb;
    }
    return actionAt(i, j, ft + 1) * ctb;
  }
}

// roller advection uses cx/cy/ctheta on rr (no action transform)
fn rrAt(i: i32, j: i32, it: i32) -> f32 { return PA(PA_rr, at3(i, j, it)); }

fn xfluxR(fi: i32, j: i32, it: i32) -> f32 {
  if (fi < 0 || fi >= NX) { return 0.0; }
  if (S2(S2_wete, at(fi, j)) != 1.0) { return 0.0; }
  let cgxu = 0.5 * (v3[V3_cx * N3 + at3(fi + 1, j, it)] + v3[V3_cx * N3 + at3(fi, j, it)]);
  let dnu = C2(C2_dnu, at(fi, j));
  if (U.scheme == 1.0) {
    if (cgxu > 0.0) { return rrAt(fi, j, it) * cgxu * dnu; }
    return rrAt(fi + 1, j, it) * cgxu * dnu;
  }
  if (fi >= 1 && fi <= NX - 2) {
    if (cgxu > 0.0) {
      let dsu_m = C2(C2_dsu, at(fi - 1, j));
      let dsu_0 = C2(C2_dsu, at(fi, j));
      var eupw = ((dsu_m + 0.5 * dsu_0) * rrAt(fi, j, it) -
        0.5 * dsu_0 * rrAt(fi - 1, j, it)) / dsu_m;
      if (eupw < 0.0) { eupw = rrAt(fi, j, it); }
      return eupw * cgxu * dnu;
    } else {
      if (U.statMode == 1.0) { return rrAt(fi + 1, j, it) * cgxu * dnu; }
      let dsu_p = C2(C2_dsu, at(fi + 1, j));
      let dsu_0 = C2(C2_dsu, at(fi, j));
      var eupw = ((dsu_p + 0.5 * dsu_0) * rrAt(fi + 1, j, it) -
        0.5 * dsu_0 * rrAt(fi + 2, j, it)) / dsu_p;
      if (eupw < 0.0) { eupw = rrAt(fi + 1, j, it); }
      return eupw * cgxu * dnu;
    }
  } else if (fi == 0) {
    if (cgxu > 0.0) { return rrAt(0, j, it) * cgxu * dnu; }
    let dsu_p = C2(C2_dsu, at(1, j));
    let dsu_0 = C2(C2_dsu, at(0, j));
    var eupw = ((dsu_p + 0.5 * dsu_0) * rrAt(1, j, it) - 0.5 * dsu_0 * rrAt(2, j, it)) / dsu_p;
    if (eupw < 0.0) { eupw = rrAt(1, j, it); }
    return eupw * cgxu * dnu;
  } else {
    if (cgxu > 0.0) {
      let dsu_m = C2(C2_dsu, at(NX - 2, j));
      let dsu_0 = C2(C2_dsu, at(NX - 1, j));
      var eupw = ((dsu_m + 0.5 * dsu_0) * rrAt(NX - 1, j, it) -
        0.5 * dsu_0 * rrAt(NX - 2, j, it)) / dsu_m;
      if (eupw < 0.0) { eupw = rrAt(NX - 1, j, it); }
      return eupw * cgxu * dnu;
    }
    return rrAt(NX, j, it) * cgxu * dnu;
  }
}

fn yfluxR(i: i32, fj: i32, it: i32) -> f32 {
  if (fj < 0 || fj >= NY) { return 0.0; }
  if (S2(S2_wete, at(i, fj)) != 1.0) { return 0.0; }
  let cgyv = 0.5 * (v3[V3_cy * N3 + at3(i, fj + 1, it)] + v3[V3_cy * N3 + at3(i, fj, it)]);
  let dsv = C2(C2_dsv, at(i, fj));
  if (U.scheme == 1.0 || U.statMode == 1.0) {
    if (cgyv > 0.0) { return rrAt(i, fj, it) * cgyv * dsv; }
    return rrAt(i, fj + 1, it) * cgyv * dsv;
  }
  if (fj >= 1 && fj <= NY - 2) {
    if (cgyv > 0.0) {
      let dnv_m = C2(C2_dnv, at(i, fj - 1));
      let dnv_0 = C2(C2_dnv, at(i, fj));
      var eupw = ((dnv_m + 0.5 * dnv_0) * rrAt(i, fj, it) -
        0.5 * dnv_0 * rrAt(i, fj - 1, it)) / dnv_m;
      if (eupw < 0.0) { eupw = rrAt(i, fj, it); }
      return eupw * cgyv * dsv;
    } else {
      let dnv_p = C2(C2_dnv, at(i, fj + 1));
      let dnv_0 = C2(C2_dnv, at(i, fj));
      var eupw = ((dnv_p + 0.5 * dnv_0) * rrAt(i, fj + 1, it) -
        0.5 * dnv_0 * rrAt(i, fj + 2, it)) / dnv_p;
      if (eupw < 0.0) { eupw = rrAt(i, fj + 1, it); }
      return eupw * cgyv * dsv;
    }
  } else if (fj == 0) {
    if (cgyv > 0.0) { return rrAt(i, 0, it) * cgyv * dsv; }
    let dnv_p = C2(C2_dnv, at(i, 1));
    let dnv_0 = C2(C2_dnv, at(i, 0));
    var eupw = ((dnv_p + 0.5 * dnv_0) * rrAt(i, 1, it) - 0.5 * dnv_0 * rrAt(i, 2, it)) / dnv_p;
    if (eupw < 0.0) { eupw = rrAt(i, 1, it); }
    return eupw * cgyv * dsv;
  } else {
    if (cgyv > 0.0) {
      let dnv_m = C2(C2_dnv, at(i, NY - 2));
      let dnv_0 = C2(C2_dnv, at(i, NY - 1));
      var eupw = ((dnv_m + 0.5 * dnv_0) * rrAt(i, NY - 1, it) -
        0.5 * dnv_0 * rrAt(i, NY - 2, it)) / dnv_m;
      if (eupw < 0.0) { eupw = rrAt(i, NY - 1, it); }
      return eupw * cgyv * dsv;
    }
    return rrAt(i, NY, it) * cgyv * dsv;
  }
}

fn tfluxR(i: i32, j: i32, ft: i32) -> f32 {
  if (ft < 0 || ft >= NTHETA - 1) { return 0.0; }
  let ctb = 0.5 * (v3[V3_ctheta * N3 + at3(i, j, ft + 1)] + v3[V3_ctheta * N3 + at3(i, j, ft)]);
  if (U.scheme == 1.0) {
    if (ctb > 0.0) { return rrAt(i, j, ft) * ctb; }
    return rrAt(i, j, ft + 1) * ctb;
  }
  if (ft >= 1 && ft <= NTHETA - 3) {
    if (ctb > 0.0) {
      var eupw = 1.5 * rrAt(i, j, ft) - 0.5 * rrAt(i, j, ft - 1);
      if (eupw < 0.0) { eupw = rrAt(i, j, ft); }
      return eupw * ctb;
    } else {
      var eupw = 1.5 * rrAt(i, j, ft + 1) - 0.5 * rrAt(i, j, ft + 2);
      if (eupw < 0.0) { eupw = rrAt(i, j, ft + 1); }
      return eupw * ctb;
    }
  } else if (ft == 0) {
    if (ctb > 0.0) { return rrAt(i, j, 0) * ctb; }
    var eupw = 1.5 * rrAt(i, j, 1) - 0.5 * rrAt(i, j, 2);
    if (eupw < 0.0) { eupw = rrAt(i, j, 1); }
    return eupw * ctb;
  } else {
    if (ctb > 0.0) {
      var eupw = 1.5 * rrAt(i, j, ft) - 0.5 * rrAt(i, j, ft - 1);
      if (eupw < 0.0) { eupw = rrAt(i, j, ft); }
      return eupw * ctb;
    }
    return rrAt(i, j, ft + 1) * ctb;
  }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let cell = i32(gid.x);
  if (cell >= NCELL) { return; }
  let i = cell % N1;
  let j = cell / N1;
  // stationary: per-row dtw (Fortran row CFL + sink limits); else model dt
  let dt = select(DT(), min(rowDtw(i), 1e20), U.statMode == 1.0);
  let wete = S2(S2_wete, cell);
  var interior = i >= IMIN_EE && i <= IMAX_EE && j >= JMIN_EE && j <= JMAX_EE;
  // red-black sweep parity (stationary relaxation): U.spare in {0,1}; -1 = all.
  // Skipped cells copy state through but must not clobber derived fields.
  let paritySkip = U.spare >= 0.0 && f32(i % 2) != U.spare;
  if (paritySkip) { interior = false; }

  var ee1: array<f32, NTHETA>;
  var rr1: array<f32, NTHETA>;

  // ---- advection Euler update (wave_instationary.F90:199-220) ----
  for (var it = 0; it < NTHETA; it++) {
    var e = PA(PA_ee, at3(i, j, it));
    var r = PA(PA_rr, at3(i, j, it));
    if (wete == 1.0 && interior) {
      let dsdnzi = C2(C2_dsdnzi, cell);
      var xadv = (xflux(i, j, it) - xflux(i - 1, j, it)) * dsdnzi;
      var yadv = 0.0;
      var tadv = 0.0;
      var xradv = (xfluxR(i, j, it) - xfluxR(i - 1, j, it)) * dsdnzi;
      var yradv = 0.0;
      var tradv = 0.0;
      if (NY > 0) {
        yadv = (yflux(i, j, it) - yflux(i, j - 1, it)) * dsdnzi;
        yradv = (yfluxR(i, j, it) - yfluxR(i, j - 1, it)) * dsdnzi;
      }
      if (U.refraction == 1.0 && NTHETA > 1) {
        tadv = (tflux(i, j, it) - tflux(i, j, it - 1)) / U.dtheta;
        tradv = (tfluxR(i, j, it) - tfluxR(i, j, it - 1)) / U.dtheta;
      }
      // warmbeam anti-diffusive correction (advecxho/yho WARMBEAM tail)
      if (U.scheme == 4.0 && U.statMode == 0.0) {
        let cgx0 = v3[V3_cgx * N3 + at3(i, j, it)];
        xadv = xadv -
          ((actionAt(i + 1, j, it) - actionAt(i, j, it)) / C2(C2_dsu, cell) -
            (actionAt(i, j, it) - actionAt(i - 1, j, it)) / C2(C2_dsu, at(i - 1, j))) /
            C2(C2_dsz, cell) * dt / 2.0 * cgx0 * cgx0;
        let cx0 = v3[V3_cx * N3 + at3(i, j, it)];
        xradv = xradv -
          ((rrAt(i + 1, j, it) - rrAt(i, j, it)) / C2(C2_dsu, cell) -
            (rrAt(i, j, it) - rrAt(i - 1, j, it)) / C2(C2_dsu, at(i - 1, j))) /
            C2(C2_dsz, cell) * dt / 2.0 * cx0 * cx0;
        if (NY > 0) {
          let cgy0 = v3[V3_cgy * N3 + at3(i, j, it)];
          yadv = yadv -
            ((actionAt(i, j + 1, it) - actionAt(i, j, it)) / C2(C2_dnv, cell) -
              (actionAt(i, j, it) - actionAt(i, j - 1, it)) / C2(C2_dnv, at(i, j - 1))) /
              C2(C2_dnz, cell) * dt / 2.0 * cgy0 * cgy0;
          let cy0 = v3[V3_cy * N3 + at3(i, j, it)];
          yradv = yradv -
            ((rrAt(i, j + 1, it) - rrAt(i, j, it)) / C2(C2_dnv, cell) -
              (rrAt(i, j, it) - rrAt(i, j - 1, it)) / C2(C2_dnv, at(i, j - 1))) /
              C2(C2_dnz, cell) * dt / 2.0 * cy0 * cy0;
        }
      }
      let a = actionAt(i, j, it) - dt * (xadv + yadv + tadv);
      e = max(a * S2(S2_sigm, cell), 0.0);
      r = max(r - dt * (xradv + yradv + tradv), 0.0);
    } else if (wete != 1.0) {
      e = 0.0;
      r = 0.0;
    }
    ee1[it] = e;
    rr1[it] = r;
  }

  // ---- E, H + gammax limiter (wave_instationary.F90:222-249) ----
  var E = 0.0;
  for (var it = 0; it < NTHETA; it++) { E += ee1[it]; }
  E = E * U.dtheta;
  if (wete != 1.0) { E = 0.0; }
  var H = sqrt(E / U.rhog8);
  // gammax reference depth: instationary uses hh, stationary uses hhw
  let gamref = select(S2(S2_hh, cell), HW(cell), U.statMode == 1.0);
  if (U.statMode == 0.0) {
    if (H > U.gammax * gamref && wete == 1.0 && interior) {
      let fac = (H / (U.gammax * gamref));
      for (var it = 0; it < NTHETA; it++) { ee1[it] = ee1[it] / (fac * fac); }
      H = min(H, U.gammax * gamref);
      E = U.rhog8 * H * H;
    }
  } else {
    // stationary: multiplicative (gammafac_i)^2 on ee then re-integrate
    let gfi = (U.gammax * gamref) / max(H, U.eps / 1000.0);
    if (wete == 1.0 && gfi < 1.0) {
      for (var it = 0; it < NTHETA; it++) { ee1[it] = ee1[it] * gfi * gfi; }
      E = 0.0;
      for (var it = 0; it < NTHETA; it++) { E += ee1[it]; }
      E = E * U.dtheta;
      H = sqrt(E / U.rhog8);
    }
  }

  // ---- breaker dissipation (roelvink.F90) ----
  var D = 0.0;
  var Qb = 0.0;
  let hhw = S2(S2_hhw, cell);
  let hh = S2(S2_hh, cell);
  let kw = S2(S2_k, cell);
  if (U.breakmodel == 1.0 || U.breakmodel == 3.0 || U.breakmodel == 4.0) {
    // roelvink 1/2/daly
    let kmr = clamp(kw, 0.01, 100.0);
    if (U.breakmodel != 4.0) {
      let arg = -pow(H / (U.gamma * hhw), U.ngam);
      Qb = min(1.0 - exp(max(arg, -100.0)), 1.0);
    } else {
      // daly: advected Qb in QbNew (instationary); stationary keeps prior Qb
      Qb = select(S2(S2_QbNew, cell), S2(S2_Qb, cell), U.statMode == 1.0);
      if (wete == 1.0) {
        if (H > U.gamma * hhw) { Qb = 1.0; }
        else if (H < U.gamma2 * hhw) { Qb = 0.0; }
      }
    }
    D = Qb * 2.0 * U.alpha * E;
    if (U.breakmodel == 1.0) {
      D = D / U.Trep;
    } else {
      D = D / U.Trep * H / hh;
    }
  } else if (U.breakmodel == 2.0) {
    // baldock
    let f = 1.0 / U.Trep;
    let kh = kw * hhw;
    let gamma = U.gamma;
    let Hb = tanh(gamma * kh / 0.88) * (0.88 / max(kw, 1e-10));
    let R = Hb / max(H, 0.00001);
    Qb = exp(-R * R);
    D = 0.25 * U.alpha * f * U.rho * U.g * (Hb * Hb + H * H) * Qb;
  } else if (U.breakmodel == 5.0) {
    // janssen-battjes
    let f = 1.0 / U.Trep;
    let kh = kw * hhw;
    let Hb = tanh(U.gamma * kh / 0.88) * (0.88 / kw);
    let R = Hb / max(H, 0.00001);
    let sqp = sqrt(U.px);
    // erf via Abramowitz-Stegun 7.1.26 (max err 1.5e-7, fine for f32)
    let t = 1.0 / (1.0 + 0.3275911 * R);
    let erfR = 1.0 -
      (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t +
        0.254829592) * t * exp(-R * R);
    Qb = 1.0 + 4.0 / (3.0 * sqp) * (R * R * R + 1.5 * R) * exp(-R * R) - erfR;
    D = 3.0 * sqp / 16.0 * U.alpha * f * U.rho * U.g * H * H * H / hh * Qb;
  }
  // stationary solver zeroes D on dry cells via row form; keep parity
  if (U.statMode == 1.0 && wete != 1.0) { D = 0.0; }
  // F6: Fortran evaluates Qb/D on the INTERIOR only (roelvink.F90 writes
  // imin_ee:imax_ee, jmin_ee:jmax_ee); boundary rows/columns keep
  // Qb = D = 0 forever. With break=roelvink_daly this is LOAD-BEARING,
  // not cosmetic: evaluating the criterion at the offshore column lets a
  // big group flip Qb(0) to 1 there, and the advected Qb then SUSTAINS
  // breaking through the first interior columns via daly's hysteresis
  // band — the measured offshore over-breaking strip (Qb_mean ~6x the
  // reference on big near-45deg-oblique hours, MISSION_PARITY.md).
  if (!interior) {
    D = 0.0;
    Qb = 0.0;
  }
  if (!paritySkip) {
    S2set(S2_Qb, cell, Qb);
    S2set(S2_D, cell, D);
  }

  // ---- bed friction dissipation ----
  var Df = 0.0;
  let fwv = C2(C2_fw, cell);
  if (U.statMode == 0.0) {
    // wave_instationary.F90:282-286 (coef 2/(3 pi)); zeroed when hh > fwcutoff
    let uorb = U.px * H / U.Trep / sinh(min(max(kw, 0.01) * hhw, 10.0));
    Df = 2.0 / (3.0 * U.px) * U.rho * fwv * uorb * uorb * uorb;
    if (hh > U.fwcutoff) { Df = 0.0; }
  } else {
    // wave_stationary_directions.F90:427-432 (coef 0.28), gated fw>0 & depth
    // (hhwlocal: hhws in the directions sweep)
    let hwl = HW(cell);
    if (fwv > 0.0 && wete == 1.0 && hwl <= U.fwcutoff) {
      let uorb = U.px * H / U.Trep / sinh(min(max(kw, 0.01) * hwl, 10.0));
      Df = 0.28 * U.rho * fwv * uorb * uorb * uorb;
    }
  }
  if (!paritySkip) { S2set(S2_Df, cell, Df); }

  // ---- distribute dissipation, roller balance (:288-352 / :438-528) ----
  let Emax = max(E, 0.00001);
  for (var it = 0; it < NTHETA; it++) {
    var e = ee1[it];
    var r = rr1[it];
    var drr = 0.0;
    if (wete == 1.0 && interior) {
      let dder = e * D / Emax;
      let dd = dder + e * Df / Emax;
      e = e - dt * dd;
      // the directions sweep carries no roller (callTypeDirections)
      if (U.roller == 1.0 && DIRS == 0) {
        let cxv = v3[V3_cx * N3 + at3(i, j, it)];
        let cyv = v3[V3_cy * N3 + at3(i, j, it)];
        let BR = select(U.beta, S2(S2_BR, cell), U.statMode == 0.0);
        drr = 2.0 * U.g * BR * max(r, 0.0) / sqrt(cxv * cxv + cyv * cyv);
        r = r + dt * (dder - drr);
      } else {
        r = 0.0;
        drr = 0.0;
      }
      e = max(e, 0.0);
      r = max(r, 0.0);
    } else if (wete != 1.0) {
      e = 0.0;
      r = 0.0;
      drr = 0.0;
    }
    if (S2(S2_wetz, cell) == 0.0) { r = 0.0; drr = 0.0; }
    PBset(PA_ee, at3(i, j, it), e);
    PBset(PA_rr, at3(i, j, it), r);
    // store drr sum for DR
    rr1[it] = drr;
  }
  if (!paritySkip) {
    var DRsum = 0.0;
    for (var it = 0; it < NTHETA; it++) { DRsum += rr1[it]; }
    S2set(S2_DR, cell, DRsum * U.dtheta);
  }
}`;
}

/**
 * Boundary copies after wave step (wave_instationary.F90:354-425):
 * bay boundary ee(nx+1)=ee(nx); lateral neumann rows.
 * Writes in-place on B (post-step) buffer.
 */
export function waveBcKernel(g: GridConst): string {
  return header(g) + /* wgsl */ `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let kk = i32(gid.x);
  // threads: one per (row j, theta) for bay copy + one per (col i, theta) for lateral
  let nbay = M1 * NTHETA;
  if (kk < nbay) {
    let j = kk % M1;
    let it = kk / M1;
    b3[PA_ee * NCELL * NTHETA + at3(NX, j, it)] = b3[PA_ee * NCELL * NTHETA + at3(NX - 1, j, it)];
    b3[PA_rr * NCELL * NTHETA + at3(NX, j, it)] = b3[PA_rr * NCELL * NTHETA + at3(NX - 1, j, it)];
    return;
  }
  let kk2 = kk - nbay;
  let nlat = N1 * NTHETA;
  if (NY > 0 && (U.statMode == 1.0 || (TT() > 0.0 && U.lateralwave == 0.0)) && kk2 < 2 * nlat) {
    let side = kk2 / nlat;      // 0 = j=0 edge, 1 = j=NY edge
    let r = kk2 % nlat;
    let i = r % N1;
    let it = r / N1;
    if (side == 0) {
      b3[PA_ee * NCELL * NTHETA + at3(i, 0, it)] = b3[PA_ee * NCELL * NTHETA + at3(i, 1, it)];
      b3[PA_rr * NCELL * NTHETA + at3(i, 0, it)] = b3[PA_rr * NCELL * NTHETA + at3(i, 1, it)];
    } else {
      b3[PA_ee * NCELL * NTHETA + at3(i, NY, it)] =
        b3[PA_ee * NCELL * NTHETA + at3(i, NY - 1, it)];
      b3[PA_rr * NCELL * NTHETA + at3(i, NY, it)] =
        b3[PA_rr * NCELL * NTHETA + at3(i, NY - 1, it)];
    }
  }
}`;
}

/**
 * Integrate E,R,DR,H; roller gammax; thetamean recompute; urms.
 * (wave_instationary.F90:427-472). Reads the post-BC ping buffer B.
 */
export function integrateKernel(g: GridConst): string {
  return header(g) + /* wgsl */ `
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
  let interior = i >= IMIN_EE && i <= IMAX_EE && j >= JMIN_EE && j <= JMAX_EE;
  var E = 0.0;
  var R = 0.0;
  // OCE-243: the energy flux vector Σ ee·cg dθ — what the incident
  // run-up term reads at a shore point's sample station (bc.ts meanAcc
  // keeps its time-mean); cgx/cgy are this step's group velocities per
  // bin, current included, exactly as the advection uses them
  var Px = 0.0;
  var Py = 0.0;
  for (var it = 0; it < NTHETA; it++) {
    let k3 = at3(i, j, it);
    let e = b3[PA_ee * N3 + k3];
    E += e;
    R += b3[PA_rr * N3 + k3];
    Px += e * v3[V3_cgx * N3 + k3];
    Py += e * v3[V3_cgy * N3 + k3];
  }
  E = E * U.dtheta;
  R = R * U.dtheta;
  Px = Px * U.dtheta;
  Py = Py * U.dtheta;
  var H = sqrt(E / U.rhog8);
  let wete = S2(S2_wete, cell);
  // roller gammax correction (rollergammax=1), instationary :438-458
  if (U.rollergammax == 1.0) {
    let hhw = S2(S2_hhw, cell);
    let RH = sqrt(R / U.rhog8);
    if (U.statMode == 0.0) {
      if (RH > U.gammax * hhw && wete == 1.0 && interior) {
        let fac = RH / (U.gammax * hhw);
        for (var it = 0; it < NTHETA; it++) {
          let k3 = PA_rr * N3 + at3(i, j, it);
          b3[k3] = b3[k3] / (fac * fac);
        }
        R = min(R, U.gammax * hhw);
      }
    } else {
      let gfi = (U.gammax * hhw) / max(RH, U.eps / 1000.0);
      if (wete == 1.0 && gfi < 1.0) {
        for (var it = 0; it < NTHETA; it++) {
          let k3 = PA_rr * N3 + at3(i, j, it);
          b3[k3] = b3[k3] * gfi * gfi;
        }
        R = RH * gfi;
      }
    }
  }
  S2set(S2_E, cell, E);
  S2set(S2_R, cell, R);
  S2set(S2_H, cell, H);
  S2set(S2_Px, cell, Px);
  S2set(S2_Py, cell, Py);
  if (U.statMode == 1.0 && wete != 1.0) {
    S2set(S2_E, cell, 0.0);
    S2set(S2_H, cell, 0.0);
    S2set(S2_Px, cell, 0.0);
    S2set(S2_Py, cell, 0.0);
  }
  // thetamean (instationary :460-466; stationary :557-571; the directions
  // sweep always, from ee_s/thet_s)
  if ((U.snells == 0.0 && U.singledir == 0.0) || DIRS == 1) {
    var se = 0.0;
    var set_ = 0.0;
    for (var it = 0; it < NTHETA; it++) {
      let e = b3[PA_ee * N3 + at3(i, j, it)];
      se += e;
      set_ += e * C3(C3_thet, at3(i, j, it));
    }
    if (wete == 1.0) {
      S2set(S2_thetamean, cell, (set_ / f32(NTHETA)) / (max(se, 0.00001) / f32(NTHETA)));
    } else if (U.statMode == 1.0 && i > 0) {
      // stationary: forward-copy on dry (:574)
      S2set(S2_thetamean, cell, S2(S2_thetamean, at(i - 1, j)));
    }
  }
  // urms (instationary :434-435, :472)
  let kw = S2(S2_k, cell);
  let uorb = U.px * H / U.Trep / sinh(min(max(kw, 0.01) * HW(cell), 10.0));
  S2set(S2_urms, cell, uorb / sqrt(2.0));
}`;
}

/**
 * OCE-252: update_means_wave_flow (wave_functions.F90:13-30) for single_dir —
 * the exponentially smoothed depth the directions model sees,
 *   hhws = max(f·hhw + (1−f)·hhws, eps),  f = dt / (2·wavint),
 * every step after wavePrep (hhw). wavint sits in scratch[26].
 */
export function hhwsUpdateKernel(g: GridConst): string {
  return header(g) + /* wgsl */ `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let kk = i32(gid.x);
  if (kk >= NCELL) { return; }
  let f = DT() / (2.0 * scratch[26]);
  S2set(S2_hhws, kk, max(f * S2(S2_hhw, kk) + (1.0 - f) * S2(S2_hhws, kk), U.eps));
}`;
}

/**
 * Radiation stresses + forces (wave_functions.F90:1228-1304).
 * Pass 1: Sxx, Syy, Sxy per cell.
 */
export function radStressKernel(g: GridConst): string {
  return header(g) + /* wgsl */ `
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
  let n = S2(S2_nn, cell);
  var sxx = 0.0;
  var syy = 0.0;
  var sxy = 0.0;
  var sumee = 0.0;
  for (var it = 0; it < NTHETA; it++) {
    let e = b3[PA_ee * N3 + at3(i, j, it)];
    let r = b3[PA_rr * N3 + at3(i, j, it)];
    var cs: f32;
    var sn: f32;
    if ((U.snells == 1.0 || U.singledir == 1.0) && NTHETA == 1) {
      let rel = S2(S2_thetamean, cell) - C2(C2_alfaz, cell);
      cs = cos(rel);
      sn = sin(rel);
    } else {
      cs = C3(C3_costh, at3(i, j, it));
      sn = C3(C3_sinth, at3(i, j, it));
    }
    sxx += n * (1.0 + cs * cs) * e + cs * cs * r;
    syy += n * (1.0 + sn * sn) * e + sn * sn * r;
    sxy += n * sn * cs * e + sn * cs * r;
    sumee += e;
  }
  S2set(S2_Sxx, cell, (sxx - 0.5 * sumee) * U.dtheta);
  S2set(S2_Syy, cell, (syy - 0.5 * sumee) * U.dtheta);
  S2set(S2_Sxy, cell, sxy * U.dtheta);
}`;
}

/** Pass 2: Fx, Fy from stress gradients (wave_functions.F90:1251-1302). */
export function forceKernel(g: GridConst): string {
  return header(g) + /* wgsl */ `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let cell = i32(gid.x);
  if (cell >= NCELL) { return; }
  let i = cell % N1;
  let j = cell / N1;
  var Fx = 0.0;
  var Fy = 0.0;
  if (NY > 0) {
    if (j >= 1 && j <= NY - 1 && i <= NX - 1) {
      Fx = -(S2(S2_Sxx, at(i + 1, j)) - S2(S2_Sxx, cell)) / C2(C2_dsu, cell) -
        (S2(S2_Sxy, at(i, j + 1)) + S2(S2_Sxy, at(i + 1, j + 1)) -
          S2(S2_Sxy, at(i, j - 1)) - S2(S2_Sxy, at(i + 1, j - 1))) /
          (C2(C2_dnv, at(i, j - 1)) + C2(C2_dnv, cell) +
            C2(C2_dnv, at(i + 1, j - 1)) + C2(C2_dnv, at(i + 1, j)));
    }
    if (j <= NY - 1 && i >= 1 && i <= NX - 1) {
      Fy = -(S2(S2_Syy, at(i, j + 1)) - S2(S2_Syy, cell)) / C2(C2_dnv, cell) -
        (S2(S2_Sxy, at(i + 1, j)) + S2(S2_Sxy, at(i + 1, j + 1)) -
          S2(S2_Sxy, at(i - 1, j)) - S2(S2_Sxy, at(i - 1, j + 1))) /
          (C2(C2_dsu, at(i - 1, j)) + C2(C2_dsu, cell) +
            C2(C2_dsu, at(i - 1, j + 1)) + C2(C2_dsu, at(i, j + 1)));
    }
  } else {
    if (i <= NX - 1) {
      Fx = -(S2(S2_Sxx, at(i + 1, 0)) - S2(S2_Sxx, cell)) / C2(C2_dsu, cell);
    }
    if (i >= 1 && i <= NX - 1) {
      Fy = -(S2(S2_Sxy, at(i + 1, 0)) - S2(S2_Sxy, at(i - 1, 0))) /
        (C2(C2_dsu, at(i - 1, 0)) + C2(C2_dsu, cell));
    }
  }
  S2set(S2_Fx, cell, Fx);
  S2set(S2_Fy, cell, Fy);
}`;
}

/** Edge fixes for forces (Fy(1,:)=Fy(2,:); Fx zero at bay; lateral copies). */
export function forceBcKernel(g: GridConst): string {
  return header(g) + /* wgsl */ `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let kk = i32(gid.x);
  if (kk < M1) {
    // rows: istop Fy(1,:)=Fy(2,:); isbot Fx,Fy(nx+1)=0
    let j = kk;
    S2set(S2_Fy, at(0, j), S2(S2_Fy, at(1, j)));
    S2set(S2_Fx, at(NX, j), 0.0);
    S2set(S2_Fy, at(NX, j), 0.0);
    return;
  }
  let i = kk - M1;
  if (NY > 0 && i < N1) {
    S2set(S2_Fx, at(i, 0), S2(S2_Fx, at(i, 1)));
    S2set(S2_Fx, at(i, NY), S2(S2_Fx, at(i, NY - 1)));
  }
}`;
}

/**
 * Stokes drift + breaker delay (wave_functions.F90:1306-1358 + 839-923).
 * breakerdelay's upstream scan is done in-thread (1D cross-shore assumption,
 * as in Fortran).
 */
export function stokesKernel(g: GridConst): string {
  return header(g) + /* wgsl */ `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let cell = i32(gid.x);
  if (cell >= NCELL) { return; }
  let wete = S2(S2_wete, cell);
  var ustw = 0.0;
  var ustr = 0.0;
  if (wete == 1.0) {
    let c = S2(S2_c, cell);
    let hst = S2(S2_hstokes, cell);
    ustw = S2(S2_E, cell) / c / U.rho / hst;
    ustr = 2.0 * S2(S2_R, cell) / c / U.rho / hst;
    let rel = S2(S2_thetamean, cell) - C2(C2_alfaz, cell);
    S2set(S2_uwf, cell, ustw * cos(rel));
    S2set(S2_vwf, cell, ustw * sin(rel));
  } else {
    S2set(S2_uwf, cell, 0.0);
    S2set(S2_vwf, cell, 0.0);
  }
  S2set(S2_ustr, cell, ustr);
  // scratch[3] = breakerdelay flag; usd computed in separate pass if on
  if (scratch[3] <= 0.0) {
    S2set(S2_ust, cell, ustw + ustr);
  } else {
    S2set(S2_usd, cell, ustw); // stash ustw; breakerDelay pass adds usd
  }
}`;
}

/** breakerdelay (wave_functions.F90:839-923): weighted upstream average of ustr. */
export function breakerDelayKernel(g: GridConst): string {
  return header(g) + /* wgsl */ `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let cell = i32(gid.x);
  if (cell >= NCELL) { return; }
  let i = cell % N1;
  let j = cell / N1;
  // Fortran: jy loop j1..max(1,ny); j1=2 for 2D; row 0 handled by copy below
  var usd = S2(S2_ustr, cell);
  if (i == 0) {
    usd = S2(S2_ustr, cell);
  } else if (S2(S2_wete, cell) == 1.0) {
    let Lbr = sqrt(U.g * S2(S2_hhw, cell)) * U.Trep * scratch[3];
    var nbr = 0;
    var ii = i - 1;
    // xz distance along row (use sdist as proxy for |xz(i)-xz(jx)| on rectilinear)
    loop {
      if (ii <= 0) { break; }
      if (abs(C2(C2_sdist, at(ii, j)) - C2(C2_sdist, cell)) > Lbr) { break; }
      nbr = nbr + 1;
      ii = ii - 1;
    }
    if (nbr > 1) {
      var num = 0.0;
      var den = 0.0;
      for (var w = 1; w <= nbr + 1; w++) {
        let src = i - nbr + w - 1;
        num += f32(w) * S2(S2_ustr, at(clamp(src, 0, NX), j));
        den += f32(w);
      }
      usd = num / den;
    } else {
      usd = S2(S2_ustr, cell);
    }
  }
  // combine: ust = ustw (stashed in usd field by stokesKernel) + usd
  let ustw = S2(S2_usd, cell);
  S2set(S2_ust, cell, ustw + usd);
}`;
}

/** ust boundary copies (compute_stokes_drift tail). */
export function stokesBcKernel(g: GridConst): string {
  return header(g) + /* wgsl */ `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let kk = i32(gid.x);
  if (kk < M1) {
    S2set(S2_ust, at(0, kk), S2(S2_ust, at(1, kk)));
    return;
  }
  let i = kk - M1;
  if (NY > 0 && i < N1) {
    S2set(S2_ust, at(i, 0), S2(S2_ust, at(i, 1)));
    S2set(S2_ust, at(i, NY), S2(S2_ust, at(i, NY - 1)));
  }
}`;
}

/**
 * Roelvink-Daly Qb advection (wave_instationary.F90:260-277):
 * cgxm = c cos(thetamean-alfaz), advecqx/qy first-order upwind; QbNew out.
 */
export function qbAdvectKernel(g: GridConst): string {
  return header(g) + /* wgsl */ `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let cell = i32(gid.x);
  if (cell >= NCELL) { return; }
  let i = cell % N1;
  let j = cell / N1;
  var qb = S2(S2_Qb, cell);
  // interior ranges: advecqx uses old1Dadvection switch; new: imin_ee..imax_ee
  let i1 = select(IMIN_EE, 1, U.oldRD1D == 1.0);
  let i2 = select(IMAX_EE, NX - 1, U.oldRD1D == 1.0);
  let j1 = select(JMIN_EE, 1, U.oldRD1D == 1.0);
  let j2 = select(JMAX_EE, NY - 1, U.oldRD1D == 1.0);
  var adv = 0.0;
  if (i >= i1 && i <= i2 && j >= j1 && j <= j2 && S2(S2_wete, cell) == 1.0) {
    let rel = S2(S2_thetamean, cell) - C2(C2_alfaz, cell);
    let cgxm = S2(S2_c, cell) * cos(rel);
    if (cgxm > 0.0) {
      adv += cgxm * (qb - S2(S2_Qb, at(i - 1, j))) / C2(C2_dsu, at(i - 1, j));
    } else if (cgxm < 0.0) {
      adv += cgxm * (S2(S2_Qb, at(i + 1, j)) - qb) / C2(C2_dsu, cell);
    } else {
      adv += cgxm * (S2(S2_Qb, at(i + 1, j)) - S2(S2_Qb, at(i - 1, j))) /
        (C2(C2_dsu, cell) + C2(C2_dsu, at(i - 1, j)));
    }
    if (NY > 0) {
      let cgym = S2(S2_c, cell) * sin(rel);
      if (cgym > 0.0) {
        adv += cgym * (qb - S2(S2_Qb, at(i, j - 1))) / C2(C2_dnv, at(i, j - 1));
      } else if (cgym < 0.0) {
        adv += cgym * (S2(S2_Qb, at(i, j + 1)) - qb) / C2(C2_dnv, cell);
      } else {
        adv += cgym * (S2(S2_Qb, at(i, j + 1)) - S2(S2_Qb, at(i, j - 1))) /
          (C2(C2_dnv, cell) + C2(C2_dnv, at(i, j - 1)));
      }
    }
  }
  var out = qb - DT() * adv;
  if (S2(S2_wete, cell) == 0.0) { out = 0.0; }
  out = clamp(out, 0.0, 1.0);
  S2set(S2_QbNew, cell, out);
}`;
}
