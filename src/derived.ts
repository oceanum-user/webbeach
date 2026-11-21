// SPDX-License-Identifier: LGPL-3.0-or-later
// XBeach WebGPU solver — derived from XBeach (c) Deltares (LGPL v2.1+).
// Copyright (c) 2026 Oceanum. See LICENSE.
/**
 * Derived surf-zone descriptors for the surf-zone assessment application
 * (docs/output_variables.md). Host-side, computed from model fields along a
 * cross-shore transect (fixed j).
 */

export interface SurfZoneDescriptors {
  breakerLines: number[]; // i-indices of breaking-line maxima (offshore->shore)
  nBreakingLines: number;
  breakerHeight: number; // Hrms at outermost breaking line [m]
  breakerDepth: number; // water depth there [m]
  breakerType: "spilling" | "plunging" | "collapsing" | "none";
  surfZoneWidth: number; // outermost breaking line -> waterline [m]
  waterlineIndex: number;
  xi0: number; // surf similarity parameter
}

export interface TransectInput {
  H: Float32Array | Float64Array; // Hrms along transect
  D: Float32Array | Float64Array; // breaker dissipation
  Qb: Float32Array | Float64Array;
  hh: Float32Array | Float64Array;
  wetz: Float32Array | Float64Array;
  zb: Float32Array | Float64Array;
  sdist: Float32Array | Float64Array; // cross-shore distance [m]
  Trep: number;
  g?: number;
  qbThreshold?: number; // default 0.2
}

export function surfZoneDescriptors(tr: TransectInput): SurfZoneDescriptors {
  const g = tr.g ?? 9.81;
  const qbThr = tr.qbThreshold ?? 0.2;
  const n = tr.H.length;

  // waterline: last wet cell from offshore
  let waterline = n - 1;
  for (let i = 0; i < n; i++) {
    if (tr.wetz[i] === 0) {
      waterline = i;
      break;
    }
  }

  // breaking lines: local maxima of D where Qb > threshold, separated by
  // regions where Qb drops below threshold/2 (distinct breaker zones)
  const lines: number[] = [];
  let inZone = false;
  let zoneMaxD = -Infinity;
  let zoneMaxI = -1;
  for (let i = 0; i < waterline; i++) {
    const active = tr.Qb[i] > qbThr && tr.D[i] > 0;
    if (active) {
      if (!inZone) {
        inZone = true;
        zoneMaxD = -Infinity;
      }
      if (tr.D[i] > zoneMaxD) {
        zoneMaxD = tr.D[i];
        zoneMaxI = i;
      }
    } else if (inZone && tr.Qb[i] < qbThr / 2) {
      lines.push(zoneMaxI);
      inZone = false;
    }
  }
  if (inZone) lines.push(zoneMaxI);

  if (lines.length === 0) {
    return {
      breakerLines: [],
      nBreakingLines: 0,
      breakerHeight: 0,
      breakerDepth: 0,
      breakerType: "none",
      surfZoneWidth: 0,
      waterlineIndex: waterline,
      xi0: 0,
    };
  }

  const outer = lines[0];
  const Hb = tr.H[outer];
  const hb = tr.hh[outer];
  const width = Math.abs(tr.sdist[waterline] - tr.sdist[outer]);

  // surf similarity: foreshore slope between breaker line and waterline
  const dzb = tr.zb[waterline] - tr.zb[outer];
  const dxs = Math.max(Math.abs(tr.sdist[waterline] - tr.sdist[outer]), 1e-6);
  const tanb = Math.abs(dzb / dxs);
  const L0 = (g * tr.Trep * tr.Trep) / (2 * Math.PI);
  const xi0 = tanb / Math.sqrt(Math.max(Hb / L0, 1e-12));
  const type = xi0 < 0.4 ? "spilling" : xi0 < 2.0 ? "plunging" : "collapsing";

  return {
    breakerLines: lines,
    nBreakingLines: lines.length,
    breakerHeight: Hb,
    breakerDepth: hb,
    breakerType: type,
    surfZoneWidth: width,
    waterlineIndex: waterline,
    xi0,
  };
}

/** Hs and Hmax from Hrms (Rayleigh statistics). */
export function waveHeightStats(Hrms: number): { Hs: number; Hmax: number } {
  return { Hs: Math.SQRT2 * Hrms, Hmax: 1.86 * Hrms };
}
