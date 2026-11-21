// SPDX-License-Identifier: LGPL-3.0-or-later
// XBeach WebGPU solver — derived from XBeach (c) Deltares (LGPL v2.1+).
// Copyright (c) 2026 Oceanum. See LICENSE.
/**
 * Shared WGSL preamble. Grid dimensions are baked in as constants at pipeline
 * build time (per-run specialization: nx, ny, ntheta are fixed for a run).
 *
 * Index conventions (0-based):
 *   i = 0..nx   (cross-shore, Fortran 1..nx+1)
 *   j = 0..ny   (alongshore,  Fortran 1..ny+1)
 *   cell id     k  = i + j*(nx+1)
 *   3D id       k3 = i + j*(nx+1) + it*(nx+1)*(ny+1)   (theta-major slices)
 *
 * Fortran loop-range equivalents (spaceparams.F90 ranges_init, serial):
 *   imin_ee..imax_ee = 2..nx      -> 1..nx-1      (0-based)
 *   jmin_ee..jmax_ee = 2..ny      -> 1..ny-1  (ny>0) else 0..0
 *   imin_uu..imax_uu = 2..nx-1    -> 1..nx-2
 *   jmin_uu..jmax_uu = 2..ny      -> 1..ny-1  (ny>0) else 0..0
 *   imin_vv..imax_vv = 2..nx      -> 1..nx-1
 *   jmin_vv..jmax_vv = 2..ny-1    -> 1..ny-2  (ny>0) else 0..0
 *   imin_zs..imax_zs = 2..nx      -> 1..nx-1
 *   jmin_zs..jmax_zs = 2..ny      -> 1..ny-1  (ny>0) else 0..0
 */

export interface GridConst {
  nx: number;
  ny: number;
  ntheta: number;
  /** OCE-252: compile the kernel for the single_dir stationary-directions
   * sweep — NTHETA is then ntheta_s, the 3-D buffers bound are the ee_s set,
   * and depths come from the smoothed hhws (callTypeDirections,
   * wave_stationary_directions.F90). */
  dirs?: boolean;
}

export function preamble(g: GridConst): string {
  const n1 = g.nx + 1;
  const m1 = g.ny + 1;
  return /* wgsl */ `
const NX: i32 = ${g.nx};
const NY: i32 = ${g.ny};
const NTHETA: i32 = ${g.ntheta};
const DIRS: i32 = ${g.dirs ? 1 : 0};   // 1: single_dir directions sweep (OCE-252)
const N1: i32 = ${n1};          // points in s-direction
const M1: i32 = ${m1};          // points in n-direction
const NCELL: i32 = ${n1 * m1};
const PI: f32 = 3.14159265358979323846;

// at() is i + j*N1 written as a dot product on purpose. With the plain form,
// Apple's Metal compiler on Intel gen9 GPUs (Iris Plus 645, macOS 15; Chrome
// and Deno alike) miscompiles some neighbour lookups: at(i - 1, j) came out
// as -1 in dtFlow (robust-access clamp -> dsu read as 0.02 -> dt 500x too
// small, "timestep collapsed at t=1 s" on every bathymetry) and continuity
// updated zs wrongly in ~1000 cells. Same value; do not "simplify" it back.
fn at(i: i32, j: i32) -> i32 { return dot(vec2<i32>(i, j), vec2<i32>(1, N1)); }
fn at3(i: i32, j: i32, it: i32) -> i32 { return i + j * N1 + it * NCELL; }

// Fortran-equivalent interior ranges (serial, non-MPI)
const IMIN_EE: i32 = 1;
const IMAX_EE: i32 = NX - 1;      // inclusive
const JMIN_EE: i32 = ${g.ny > 0 ? 1 : 0};
const JMAX_EE: i32 = ${g.ny > 0 ? g.ny - 1 : 0};
const IMIN_UU: i32 = 1;
const IMAX_UU: i32 = NX - 2;
const JMIN_UU: i32 = ${g.ny > 0 ? 1 : 0};
const JMAX_UU: i32 = ${g.ny > 0 ? g.ny - 1 : 0};
const IMIN_VV: i32 = 1;
const IMAX_VV: i32 = NX - 1;
const JMIN_VV: i32 = ${g.ny > 0 ? 1 : 0};
const JMAX_VV: i32 = ${g.ny > 0 ? Math.max(g.ny - 2, 0) : 0};
const IMIN_ZS: i32 = 1;
const IMAX_ZS: i32 = NX - 1;
const JMIN_ZS: i32 = ${g.ny > 0 ? 1 : 0};
const JMAX_ZS: i32 = ${g.ny > 0 ? g.ny - 1 : 0};
`;
}

/**
 * Uniform parameter block shared by wave kernels. Keep in sync with
 * WaveUniforms in model.ts (std140-like: pack scalars as array of f32).
 */
export const WAVE_UNIFORMS_WGSL = /* wgsl */ `
struct Uni {
  dt: f32,          // current time step
  Trep: f32,
  sigmval: f32,     // 2 pi / Trep
  g: f32,
  rho: f32,
  rhog8: f32,
  px: f32,
  eps: f32,
  gamma: f32,
  gamma2: f32,
  ngam: f32,        // par%n (breaking power)
  alpha: f32,
  gammax: f32,
  Trepinv: f32,
  fwcutoff: f32,
  beta: f32,        // roller slope par%beta
  t: f32,
  wci: f32,
  dtheta: f32,
  scheme: f32,       // 1 upwind_1, 3 upwind_2, 4 warmbeam
  breakmodel: f32,   // BREAK_* constant
  roller: f32,
  refraction: f32,
  deltaH: f32,       // par%delta
  rollergammax: f32,
  snells: f32,
  singledir: f32,
  theta0: f32,
  lateralwave: f32,  // 0 neumann, 1 wavecrest, 2 cyclic
  oldRD1D: f32,
  // stationary-solver controls
  dtw: f32,
  rowI: f32,         // current row for stationary sweep (-1 => full grid)
  statMode: f32,     // 1 during stationary iteration (use sigm for action)
  spare: f32,
};
@group(0) @binding(0) var<uniform> U: Uni;
`;
