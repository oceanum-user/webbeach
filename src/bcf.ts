// SPDX-License-Identifier: LGPL-3.0-or-later
// XBeach WebGPU solver — derived from XBeach (c) Deltares (LGPL v2.1+).
// Copyright (c) 2026 Oceanum. See LICENSE.
/**
 * Reader for XBeach surfbeat boundary-condition files (reuse path):
 *   ebcflist.bcf / qbcflist.bcf  — text: endtime rt dtbc Trep theta0 Hrms file
 *   E_series*.bcf — float64 records of (ny+1)*ntheta values per dtbc step
 *   q_series*.bcf — float64 records of (ny+1)*4 values per dtbc step
 * Reading matches boundaryconditions.F90:477-664 (linear time interpolation,
 * ui/vi from fluxes rotated by alfau, divided by ht=zs0-zb, taper factor).
 */

// BcCondition lives in wavemaker.ts (shared with the generated-table shape);
// the type-only import keeps this Deno-only reader out of the solver bundle.
import type { BcCondition } from "./wavemaker.ts";
export type { BcCondition };

export async function readBcfList(dir: string, ny: number, ntheta: number,
  nthetaS = 0): Promise<BcCondition[]> {
  const elist = (await Deno.readTextFile(`${dir}/ebcflist.bcf`)).trim().split(/\r?\n/);
  const qlist = (await Deno.readTextFile(`${dir}/qbcflist.bcf`)).trim().split(/\r?\n/);
  // OCE-252: a single_dir run also wrote esbcflist.bcf — one record per
  // condition of the stationary boundary spread ee_s(1,:,:), (ny+1)*ntheta_s
  // float64, j fastest (waveparamsnew.F90:3104-3110, read back at
  // boundaryconditions.F90:505-510)
  const eslist = nthetaS > 0
    ? await Deno.readTextFile(`${dir}/esbcflist.bcf`).then(
      (t) => t.trim().split(/\r?\n/), () => null)
    : null;
  const out: BcCondition[] = [];
  for (let li = 0; li < elist.length; li++) {
    const e = elist[li].trim().split(/\s+/);
    const q = qlist[li].trim().split(/\s+/);
    const eFile = e[6], qFile = q[6];
    const eBytes = await Deno.readFile(`${dir}/${eFile}`);
    const qBytes = await Deno.readFile(`${dir}/${qFile}`);
    const eData = new Float64Array(eBytes.buffer, 0, eBytes.byteLength / 8);
    const qData = new Float64Array(qBytes.buffer, 0, qBytes.byteLength / 8);
    const recE = (ny + 1) * ntheta;
    let esData: Float64Array | undefined;
    if (eslist) {
      const es = eslist[li].trim().split(/\s+/);
      const esBytes = await Deno.readFile(`${dir}/${es[6]}`);
      esData = new Float64Array(esBytes.buffer, 0, (ny + 1) * nthetaS);
    }
    out.push({
      esData,
      endtime: parseFloat(e[0]),
      rt: parseFloat(e[1]),
      dtbc: parseFloat(e[2]),
      Trep: parseFloat(e[3]),
      theta0: parseFloat(e[4]),
      Hrms: parseFloat(e[5]),
      eFile,
      qFile,
      eData,
      qData,
      nrec: Math.floor(eData.length / recE),
    });
  }
  return out;
}

/**
 * Interpolate boundary arrays at time t within a condition.
 * Fortran record semantics: record r (1-based) holds the signal at time
 * (r-1)*dtbc; interpolation between floor(t/dtbc)+1 and +2.
 * E_series record layout: Fortran array (ny+1, ntheta) column-major
 * -> j fastest. q_series: (ny+1, 4) -> j fastest.
 */
export function interpBc(
  cond: BcCondition,
  t: number,
  ny: number,
  ntheta: number,
  taperfac: number,
  ht: Float64Array, // zs0 - zb at row 0, per j
  alfau0: Float64Array, // alfau at row 0, per j
): { eeRow: Float32Array; uiRow: Float32Array; viRow: Float32Array } {
  const m1 = ny + 1;
  const recE = m1 * ntheta;
  const recQ = m1 * 4;
  const newIdx = Math.floor(t / cond.dtbc); // 0-based record of "earlier"
  const r1 = Math.min(newIdx, cond.nrec - 1);
  const r2 = Math.min(newIdx + 1, cond.nrec - 1);
  const tnew = (newIdx + 1) * cond.dtbc;
  const w2 = (cond.dtbc - (tnew - t)) / cond.dtbc;
  const w1 = (tnew - t) / cond.dtbc;
  const eeRow = new Float32Array(recE); // j-major [j][theta] for bcRowApply
  for (let it = 0; it < ntheta; it++) {
    for (let j = 0; j < m1; j++) {
      const src = it * m1 + j; // fortran (j, itheta) column-major
      const v = w2 * cond.eData[r2 * recE + src] + w1 * cond.eData[r1 * recE + src];
      eeRow[j * ntheta + it] = v * taperfac;
    }
  }
  const uiRow = new Float32Array(m1), viRow = new Float32Array(m1);
  for (let j = 0; j < m1; j++) {
    const qx = w2 * cond.qData[r2 * recQ + 0 * m1 + j] + w1 * cond.qData[r1 * recQ + 0 * m1 + j];
    const qy = w2 * cond.qData[r2 * recQ + 1 * m1 + j] + w1 * cond.qData[r1 * recQ + 1 * m1 + j];
    const ca = Math.cos(alfau0[j]), sa = Math.sin(alfau0[j]);
    uiRow[j] = ((qx * ca + qy * sa) / ht[j]) * taperfac;
    viRow[j] = ((-qx * sa + qy * ca) / ht[j]) * taperfac;
  }
  return { eeRow, uiRow, viRow };
}
