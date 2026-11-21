// SPDX-License-Identifier: LGPL-3.0-or-later
// XBeach WebGPU solver — derived from XBeach (c) Deltares (LGPL v2.1+).
// Copyright (c) 2026 Oceanum. See LICENSE.
/**
 * SWAN ASCII 2-D spectrum reader for the headless CLI (OCE-288): a
 * wbctype=swan case without pre-generated bcf tables gets its boundary
 * tables from the TS wavemaker, as the app's worker does from an in-memory
 * SpectrumTable. Mirrors read_swan_file (waveparamsnew.F90) in what it
 * accepts: AFREQ/NDIR axes, one quantity (VaDens in m2/Hz/degr, or EnDens
 * in J/m2/Hz/degr divided by rho g), the FIRST data block only. Directions
 * are taken as nautical coming-from in the grid frame (alfa = 0), which is
 * what the calling application is expected to write — the
 * dthetaS_XB offset is applied by the caller.
 */

import type { SpectrumTable } from "../spectrum.ts";

const RHO_G = 1025 * 9.81;

export function readSwanTable(text: string, dthetaDeg = 0): SpectrumTable {
  const lines = text.split(/\r?\n/);
  let pos = 0;
  const next = (): string | null => {
    while (pos < lines.length) {
      const l = lines[pos++].trim();
      if (l.length && !l.startsWith("$")) return l;
    }
    return null;
  };
  const num = (what: string): number => {
    const l = next();
    const v = l === null ? NaN : parseFloat(l.split(/\s+/)[0]);
    if (!Number.isFinite(v)) throw new Error(`SWAN file: cannot read ${what}`);
    return v;
  };
  const numbers = (n: number, what: string): Float64Array => {
    const out = new Float64Array(n);
    let k = 0;
    while (k < n) {
      const l = next();
      if (l === null) throw new Error(`SWAN file ended inside ${what}`);
      for (const tok of l.split(/\s+/)) {
        if (k < n) out[k++] = parseFloat(tok);
      }
    }
    return out;
  };
  const head = next();
  if (head === null || !head.toUpperCase().startsWith("SWAN")) {
    throw new Error("not a SWAN ASCII spectral file (no SWAN header)");
  }
  let freq: Float64Array | null = null, dir: Float64Array | null = null;
  let hasTime = false, enDens = false, exception = NaN;
  for (;;) {
    const l = next();
    if (l === null) throw new Error("SWAN file ended before QUANT");
    const kw = l.split(/\s+/)[0].toUpperCase();
    if (kw === "TIME") { hasTime = true; num("time coding"); }
    else if (kw === "LONLAT" || kw === "LOCATIONS") {
      const n = num("number of locations");
      if (n !== 1) throw new Error(`SWAN file has ${n} locations; one is supported`);
      next();
    } else if (kw === "RFREQ") throw new Error("SWAN file uses relative frequencies (RFREQ)");
    else if (kw === "AFREQ") freq = numbers(num("number of frequencies"), "AFREQ");
    else if (kw === "NDIR" || kw === "CDIR") dir = numbers(num("number of directions"), "NDIR");
    else if (kw === "QUANT") {
      if (num("number of quantities") !== 1) throw new Error("SWAN file: one quantity expected");
      const name = next() ?? "";
      if (name.includes("EnDens")) enDens = true;
      else if (!name.includes("VaDens")) throw new Error(`SWAN quantity "${name}" unsupported`);
      next(); // unit
      exception = num("exception value");
      break;
    }
  }
  if (!freq || !dir) throw new Error("SWAN file lacks AFREQ or NDIR");
  let l = next();
  if (l === null) throw new Error("SWAN file has no data block");
  if (/^\d{8}\.\d{6}/.test(l)) {
    if (!hasTime) throw new Error("SWAN date line without TIME section");
    l = next();
  }
  const nf = freq.length, nd = dir.length;
  const efth = new Float64Array(nf * nd);
  const kw = (l ?? "").split(/\s+/)[0].toUpperCase();
  if (kw === "FACTOR") {
    const factor = num("FACTOR value");
    const raw = numbers(nf * nd, "the spectral block");
    const scale = factor * (enDens ? 1 / RHO_G : 1);
    for (let i = 0; i < raw.length; i++) {
      const v = raw[i];
      efth[i] = (Number.isFinite(exception) && (v === exception || v * factor === exception))
        ? 0 : v * scale;
    }
  } else if (kw !== "NODATA" && kw !== "ZERO") {
    throw new Error(`SWAN file: unexpected data line "${l}"`);
  }
  const dirDeg = Float64Array.from(dir, (d) => d + dthetaDeg);
  return { freq, dirDeg, efth };
}
