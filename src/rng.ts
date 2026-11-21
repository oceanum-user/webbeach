// SPDX-License-Identifier: LGPL-3.0-or-later
// XBeach WebGPU solver — derived from XBeach (c) Deltares (LGPL v2.1+).
// Copyright (c) 2026 Oceanum. See LICENSE.
/**
 * Park-Miller "minimal standard" LCG with Schrage factorization, ported from
 * trunk/src/xbeachlibrary/math_tools.F90:1268-1305 (real*8 function random(j),
 * Park & Miller, CACM 31(10), 1988).
 *
 * Every intermediate stays below 2^31, so all arithmetic is exact in f64 and
 * the stream is bit-identical to the Fortran stream for the same seed. The
 * wavemaker draws one shared stream for phases and directions
 * (waveparamsnew.F90:1610-1662); the draw order is load-bearing.
 */

const A = 16807; // math_tools.F90:1285 parameter a
const M = 2147483647; // parameter m = 2^31-1
const Q = 127773; // parameter q = m / a (integer division)
const R = 2836; // parameter r = mod(m, a)

export class XbRandom {
  /** Fortran `save seed` with `data seed/1/` (math_tools.F90:1283-1284). */
  private state = 1;

  /**
   * random(j) with j /= 0 (math_tools.F90:1288-1294): re-seed with
   * mod(|j|, m), 0 replaced by 1, and return the FIRST draw of the new
   * series (the seeding call consumes one number — load-bearing).
   */
  seed(j: number): number {
    let s = Math.abs(Math.trunc(j)) % M;
    if (s === 0) s = 1;
    this.state = s;
    return this.next();
  }

  /**
   * random(0) (math_tools.F90:1295-1303): one Schrage step
   * test = a*lo - r*hi; seed = test (+m if test <= 0); returns seed/m.
   */
  next(): number {
    const hi = Math.floor(this.state / Q);
    const lo = this.state % Q;
    const test = A * lo - R * hi;
    this.state = test > 0 ? test : test + M;
    return this.state / M;
  }
}
