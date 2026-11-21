// SPDX-License-Identifier: LGPL-3.0-or-later
// XBeach WebGPU solver — derived from XBeach (c) Deltares (LGPL v2.1+).
// Copyright (c) 2026 Oceanum. See LICENSE.
/**
 * Field registry: packs all model fields into a small number of storage
 * buffers (WebGPU limits storage buffers per stage), with generated WGSL
 * accessor code.
 *
 * Buffers:
 *  C2  — static 2D fields (grid metrics, angles)          [read-only]
 *  S2  — dynamic 2D fields (state)                        [read-write]
 *  C3  — static 3D fields (thet, costh, sinth)            [read-only]
 *  A3/B3 — ping-pong 3D state (ee, rr)                    [read-write]
 *  V3  — 3D velocity fields (cgx, cgy, cx, cy, ctheta)    [read-write]
 */

export const C2_FIELDS = [
  "dsu", "dsz", "dsv", "dsc", "dnu", "dnz", "dnv", "dnc",
  "dsdnui", "dsdnvi", "dsdnzi", "alfaz", "alfau", "alfav", "sdist", "ndist",
  "xz", "yz", "fw", "bedfriccoef_f",
] as const;

export const S2_FIELDS = [
  // bathy + water levels
  "zb", "zs", "zs0", "zs1", "hh", "hhw", "hstokes", "hold",
  // wet/dry
  "wete", "wetz", "wetu", "wetv",
  // depths at u/v points
  "hu", "hv", "hum", "hvm",
  // flow velocities
  "uu", "vv", "u", "v", "ue", "ve", "ueu", "vev", "vu", "uv",
  "vmagu", "vmageu", "vmagv", "vmagev", "qx", "qy", "qmag",
  // flow slopes + terms
  "dzsdx", "dzsdy", "dzsdt", "ududx", "vdudy", "udvdx", "vdvdy",
  "viscu", "viscv", "nuh", "cfu", "cfv", "taubx", "tauby",
  // wave integrated fields
  "E", "H", "R", "DR", "D", "Df", "Qb", "QbNew", "urms",
  "thetamean", "sigm", "k", "c", "cg", "nn", "L1",
  // radiation stress + forces
  "Sxx", "Syy", "Sxy", "Fx", "Fy",
  // Stokes / roller transport
  "ust", "ustr", "usd", "uwf", "vwf", "BR",
  // wave BC helpers
  "ui", "vi", "umean", "vmean", "dhdx", "dhdy", "sinh2kh",
  // smoothed depths (single_dir / wci)
  "hhws", "hhwcins", "uws", "vws", "uwcins", "vwcins",
  // wind
  "windsu", "windnv",
  // extremes
  "maxzs", "minzs",
  // OCE-243: short-wave energy flux vector Σ_θ ee·cg (W/m, grid frame) —
  // the directional energy actually heading somewhere, for the shoreline
  // run-up product's incident term (time-mean in meanAcc slots 14/15)
  "Px", "Py",
] as const;

export const C3_FIELDS = ["thet", "costh", "sinth"] as const;
export const P3_FIELDS = ["ee", "rr"] as const; // ping-pong pair A3/B3
export const V3_FIELDS = ["cgx", "cgy", "cx", "cy", "ctheta"] as const;

export type C2Field = typeof C2_FIELDS[number];
export type S2Field = typeof S2_FIELDS[number];
export type C3Field = typeof C3_FIELDS[number];
export type P3Field = typeof P3_FIELDS[number];
export type V3Field = typeof V3_FIELDS[number];

/** Generate WGSL constants + accessor functions for the packed layout. */
export function fieldAccessorsWGSL(): string {
  const lines: string[] = [];
  const gen = (
    fields: readonly string[],
    prefix: string,
    buf: string,
    dim3: boolean,
    writable: boolean,
  ) => {
    fields.forEach((f, i) => lines.push(`const ${prefix}_${f}: i32 = ${i};`));
    const stride = dim3 ? "NCELL * NTHETA" : "NCELL";
    lines.push(`fn ${prefix}(f: i32, k: i32) -> f32 { return ${buf}[f * (${stride}) + k]; }`);
    if (writable) {
      lines.push(
        `fn ${prefix}set(f: i32, k: i32, v: f32) { ${buf}[f * (${stride}) + k] = v; }`,
      );
    }
  };
  gen(C2_FIELDS, "C2", "c2", false, false);
  gen(S2_FIELDS, "S2", "s2", false, true);
  gen(C3_FIELDS, "C3", "c3", true, false);
  gen(P3_FIELDS, "PA", "a3", true, true); // read side of ping-pong
  gen(P3_FIELDS, "PB", "b3", true, true); // write side
  gen(V3_FIELDS, "V3", "v3", true, true);
  return lines.join("\n");
}

export const BUFFER_DECLS_WGSL = /* wgsl */ `
@group(0) @binding(1) var<storage, read>       c2: array<f32>;
@group(0) @binding(2) var<storage, read_write> s2: array<f32>;
@group(0) @binding(3) var<storage, read>       c3: array<f32>;
@group(0) @binding(4) var<storage, read_write> a3: array<f32>;
@group(0) @binding(5) var<storage, read_write> b3: array<f32>;
@group(0) @binding(6) var<storage, read_write> v3: array<f32>;
@group(0) @binding(7) var<storage, read_write> scratch: array<f32>;
// GPU-resident time state: dt/t live in scratch (written per step by the
// dtStep kernel in gpudt mode, or uploaded by the host in sync mode)
fn DT() -> f32 { return scratch[16]; }
fn TT() -> f32 { return scratch[17]; }
`;

export function sizes(nx: number, ny: number, ntheta: number) {
  const ncell = (nx + 1) * (ny + 1);
  return {
    ncell,
    c2: C2_FIELDS.length * ncell,
    s2: S2_FIELDS.length * ncell,
    c3: C3_FIELDS.length * ncell * ntheta,
    p3: P3_FIELDS.length * ncell * ntheta,
    v3: V3_FIELDS.length * ncell * ntheta,
  };
}

export function c2Offset(f: C2Field, ncell: number): number {
  return C2_FIELDS.indexOf(f) * ncell;
}
export function s2Offset(f: S2Field, ncell: number): number {
  return S2_FIELDS.indexOf(f) * ncell;
}
export function c3Offset(f: C3Field, ncell: number, ntheta: number): number {
  return C3_FIELDS.indexOf(f) * ncell * ntheta;
}
export function p3Offset(f: P3Field, ncell: number, ntheta: number): number {
  return P3_FIELDS.indexOf(f) * ncell * ntheta;
}
