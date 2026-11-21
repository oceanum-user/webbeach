// SPDX-License-Identifier: LGPL-3.0-or-later
// XBeach WebGPU solver — derived from XBeach (c) Deltares (LGPL v2.1+).
// Copyright (c) 2026 Oceanum. See LICENSE.
/**
 * Tests for the radix-2 FFT and the Fortran-semantics Hilbert envelope
 * (trunk/src/xbeachlibrary/math_tools.F90:976-1023). The envelope oracle is
 * an independent O(n^2) direct-DFT replica of the Fortran algorithm,
 * including its zero-padding to the next power of two.
 */
import { assert, assertAlmostEquals, assertEquals, assertThrows } from "@std/assert";
import { fft, hilbertEnvelope, idftRealAdd } from "../src/fft.ts";

function mulberry(seed: number): () => number {
  // deterministic test noise
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

Deno.test("forward FFT of an impulse is flat", () => {
  const n = 8;
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  re[0] = 1;
  fft(re, im, false);
  for (let k = 0; k < n; k++) {
    assertAlmostEquals(re[k], 1, 1e-15);
    assertAlmostEquals(im[k], 0, 1e-15);
  }
});

Deno.test("forward FFT of a single tone puts N/2 in the two mirror bins", () => {
  const n = 16;
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  for (let t = 0; t < n; t++) re[t] = Math.cos((2 * Math.PI * 3 * t) / n);
  fft(re, im, false);
  for (let k = 0; k < n; k++) {
    const expected = k === 3 || k === n - 3 ? n / 2 : 0;
    assertAlmostEquals(re[k], expected, 1e-12, `re[${k}]`);
    assertAlmostEquals(im[k], 0, 1e-12, `im[${k}]`);
  }
});

Deno.test("random vector round-trips through forward+inverse within 1e-12", () => {
  const n = 128;
  const rand = mulberry(42);
  const re0 = Float64Array.from({ length: n }, () => rand() - 0.5);
  const im0 = Float64Array.from({ length: n }, () => rand() - 0.5);
  const re = Float64Array.from(re0);
  const im = Float64Array.from(im0);
  fft(re, im, false);
  fft(re, im, true);
  for (let t = 0; t < n; t++) {
    assertAlmostEquals(re[t] / n, re0[t], 1e-12);
    assertAlmostEquals(im[t] / n, im0[t], 1e-12);
  }
});

Deno.test("fft rejects non-power-of-two lengths", () => {
  assertThrows(() => fft(new Float64Array(6), new Float64Array(6), false));
});

/**
 * Independent replica of the Fortran hilbert (math_tools.F90:976-1023) using
 * a direct O(n^2) DFT: zero-pad to next power of two, unnormalized forward
 * DFT, h = [1, 2 x (n/2-1), 1, 0 x (n/2-1)], unnormalized inverse DFT / n,
 * truncate, magnitude.
 */
function hilbertOracle(x: Float64Array): Float64Array {
  const m = x.length;
  let n = 1;
  while (n < m) n <<= 1;
  const Xre = new Float64Array(n);
  const Xim = new Float64Array(n);
  for (let k = 0; k < n; k++) {
    for (let t = 0; t < m; t++) {
      const w = (2 * Math.PI * k * t) / n;
      Xre[k] += x[t] * Math.cos(w);
      Xim[k] -= x[t] * Math.sin(w);
    }
  }
  for (let k = 1; k < n / 2; k++) {
    Xre[k] *= 2;
    Xim[k] *= 2;
  }
  for (let k = n / 2 + 1; k < n; k++) {
    Xre[k] = 0;
    Xim[k] = 0;
  }
  const out = new Float64Array(m);
  for (let t = 0; t < m; t++) {
    let re = 0, im = 0;
    for (let k = 0; k < n; k++) {
      const w = (2 * Math.PI * k * t) / n;
      re += Xre[k] * Math.cos(w) - Xim[k] * Math.sin(w);
      im += Xre[k] * Math.sin(w) + Xim[k] * Math.cos(w);
    }
    out[t] = Math.hypot(re / n, im / n);
  }
  return out;
}

Deno.test("hilbertEnvelope of an on-bin cosine is exactly 1 (no padding case)", () => {
  const n = 256; // power of two: no padding, analytic signal is e^{i w t}
  const x = Float64Array.from({ length: n }, (_, t) => Math.cos((2 * Math.PI * 10 * t) / n));
  const env = hilbertEnvelope(x);
  for (let t = 0; t < n; t++) assertAlmostEquals(env[t], 1, 1e-12);
});

Deno.test("hilbertEnvelope recovers the AM modulation envelope", () => {
  const n = 512;
  const fc = 64, fm = 8; // carrier and modulation on exact bins
  const x = new Float64Array(n);
  for (let t = 0; t < n; t++) {
    x[t] = (1 + 0.5 * Math.cos((2 * Math.PI * fm * t) / n)) *
      Math.cos((2 * Math.PI * fc * t) / n);
  }
  const env = hilbertEnvelope(x);
  for (let t = 0; t < n; t++) {
    assertAlmostEquals(env[t], 1 + 0.5 * Math.cos((2 * Math.PI * fm * t) / n), 1e-10);
  }
});

Deno.test("hilbertEnvelope matches the padded-DFT oracle for non-pow2 length", () => {
  // length 300 pads to 512 exactly as Fortran does; the interior AND the edge
  // artefacts must match the oracle bit-for-bit up to fp noise.
  const m = 300;
  const rand = mulberry(7);
  const x = Float64Array.from({ length: m }, (_, t) => Math.cos(0.31 * t) + 0.3 * (rand() - 0.5));
  const env = hilbertEnvelope(x);
  const ora = hilbertOracle(x);
  assertEquals(env.length, m);
  for (let t = 0; t < m; t++) assertAlmostEquals(env[t], ora[t], 1e-10, `t=${t}`);
});

Deno.test("hilbertEnvelope interior of a padded cosine stays near 1", () => {
  // padding (300 -> 512 zeros) distorts only the record edges
  const m = 300;
  const x = Float64Array.from({ length: m }, (_, t) => Math.cos(0.5 * t));
  const env = hilbertEnvelope(x);
  for (let t = 50; t < m - 50; t++) {
    assert(Math.abs(env[t] - 1) < 0.1, `interior t=${t}: ${env[t]}`);
  }
});

// OCE-288: the wavemaker's synthesis transform. Any length, checked against
// the plain sum it replaces; accumulates into out, and a second call on the
// same length (cached plan and scratch) must not carry state over.
Deno.test("idftRealAdd equals the plain inverse DFT sum for any length", () => {
  for (const N of [2, 6, 64, 250, 1806, 3604]) {
    let seed = 12345 + N;
    const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648 - 0.5;
    for (let pass = 0; pass < 2; pass++) {
      const Xr = new Float64Array(N), Xi = new Float64Array(N);
      // sparse lower-half bins, as the wavemaker fills them
      for (let k = 0; k <= N / 2; k += 1 + (k % 3)) {
        Xr[k] = rnd();
        Xi[k] = rnd();
      }
      const out = new Float64Array(N).fill(0.25);
      idftRealAdd(out, N, Xr, Xi);
      let maxErr = 0, scale = 0;
      for (let n = 0; n < N; n += Math.max(1, Math.floor(N / 97))) {
        let ref = 0.25;
        for (let k = 0; k < N; k++) {
          if (Xr[k] === 0 && Xi[k] === 0) continue;
          const a = (2 * Math.PI * ((k * n) % N)) / N;
          ref += Xr[k] * Math.cos(a) - Xi[k] * Math.sin(a);
        }
        maxErr = Math.max(maxErr, Math.abs(out[n] - ref));
        scale = Math.max(scale, Math.abs(ref));
      }
      assert(maxErr <= 1e-11 * Math.max(1, scale), `N=${N} pass=${pass}: err ${maxErr}`);
    }
  }
});
