// SPDX-License-Identifier: LGPL-3.0-or-later
// XBeach WebGPU solver — derived from XBeach (c) Deltares (LGPL v2.1+).
// Copyright (c) 2026 Oceanum. See LICENSE.
/**
 * Radix-2 complex FFT and the Hilbert envelope used by the spectral
 * wavemaker.
 *
 * XBeach's Singleton mixed-radix fftn (trunk/src/xbeachlibrary/
 * math_tools.F90:300-968) is NOT ported. The wavemaker's synthesis IFFTs
 * have an arbitrary (even) length tslen; idftRealAdd evaluates them with
 * Bluestein's algorithm on the radix-2 transform, which gives the Fortran
 * plain-sum inverse DFT to rounding error for any length (OCE-288: the
 * direct sums it replaced cost O(K*tslen) per series and dominated the run
 * start-up). The transform inside hilbert (math_tools.F90:976-1023) is
 * zero-padded by Fortran to the next power of two — radix-2 native.
 *
 * Sign/scale conventions match fftn (math_tools.F90:330-332, 376-383) minus
 * the 1/sqrt(N) scale that callers always cancel: forward uses
 * e^(-2*pi*i*k*n/N), inverse uses e^(+2*pi*i*k*n/N), both unnormalized.
 */

/**
 * In-place unnormalized radix-2 FFT of length 2^n.
 * forward (inverse=false): X[k] = sum_n x[n] e^(-2 pi i k n / N)
 * inverse (inverse=true):  x[n] = sum_k X[k] e^(+2 pi i k n / N)
 * Twiddles are computed by direct trig per butterfly (no recurrence) to keep
 * rounding error near machine precision for the parity tolerances.
 */
export function fft(re: Float64Array, im: Float64Array, inverse: boolean): void {
  const n = re.length;
  if (n !== im.length) throw new Error("fft: re/im length mismatch");
  if (n === 0 || (n & (n - 1)) !== 0) {
    throw new Error(`fft: length ${n} is not a power of two`);
  }
  // bit-reversal permutation
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      const ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
  }
  const sgn = inverse ? 1 : -1;
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const ang = (sgn * 2 * Math.PI) / len;
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < half; k++) {
        const wr = Math.cos(ang * k);
        const wi = Math.sin(ang * k);
        const ur = re[i + k], ui = im[i + k];
        const vr = re[i + k + half] * wr - im[i + k + half] * wi;
        const vi = re[i + k + half] * wi + im[i + k + half] * wr;
        re[i + k] = ur + vr;
        im[i + k] = ui + vi;
        re[i + k + half] = ur - vr;
        im[i + k + half] = ui - vi;
      }
    }
  }
}

/**
 * Hilbert envelope |analytic signal| of a real series, exact Fortran
 * semantics (subroutine hilbert, math_tools.F90:976-1023): zero-pad to the
 * NEXT power of two of x.length, forward FFT, multiply the spectrum by
 * h = [1, 2, ..., 2, 1, 0, ..., 0] (h(1)=1, h(2:n/2)=2, h(n/2+1)=1, rest 0),
 * inverse FFT, truncate to x.length, return the magnitude.
 *
 * Note: Fortran picks the padded length via single-precision
 * p = log(real(m))/log(2.0), n = 2**ceiling(p) (math_tools.F90:987-994);
 * we compute the exact next power of two. For the common power-of-two ln
 * values the f32 rounding still yields the same n.
 */
export function hilbertEnvelope(x: Float64Array): Float64Array {
  const m = x.length;
  let n = 1;
  while (n < m) n <<= 1;
  const re = new Float64Array(n);
  re.set(x);
  const im = new Float64Array(n);
  fft(re, im, false);
  for (let k = 1; k < n / 2; k++) {
    re[k] *= 2;
    im[k] *= 2;
  }
  for (let k = n / 2 + 1; k < n; k++) {
    re[k] = 0;
    im[k] = 0;
  }
  fft(re, im, true);
  const out = new Float64Array(m);
  for (let i = 0; i < m; i++) out[i] = Math.hypot(re[i], im[i]) / n;
  return out;
}

// ------------------------------------------------- arbitrary-length inverse

/** Radix-2 tables for one length: twiddles and the bit-reversal map. */
interface Radix2Tables { c: Float64Array; s: Float64Array; rev: Uint32Array }
const tables = new Map<number, Radix2Tables>();

/**
 * In-place unnormalized radix-2 FFT with cached twiddles; same conventions
 * as fft. Used by the Bluestein transform, which calls it thousands of times
 * per boundary block on one length.
 */
function fftCached(re: Float64Array, im: Float64Array, inverse: boolean): void {
  const n = re.length;
  let t = tables.get(n);
  if (!t) {
    const c = new Float64Array(n / 2), s = new Float64Array(n / 2);
    for (let k = 0; k < n / 2; k++) {
      c[k] = Math.cos((2 * Math.PI * k) / n);
      s[k] = Math.sin((2 * Math.PI * k) / n);
    }
    const rev = new Uint32Array(n);
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      rev[i] = j;
    }
    t = { c, s, rev };
    tables.set(n, t);
  }
  const { c, s, rev } = t;
  for (let i = 1; i < n; i++) {
    const j = rev[i];
    if (i < j) {
      const tr = re[i];
      re[i] = re[j];
      re[j] = tr;
      const ti = im[i];
      im[i] = im[j];
      im[j] = ti;
    }
  }
  const sgn = inverse ? 1 : -1;
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1, step = n / len;
    for (let i = 0; i < n; i += len) {
      for (let k = 0; k < half; k++) {
        const wr = c[k * step], wi = sgn * s[k * step];
        const a = i + k, b = a + half;
        const vr = re[b] * wr - im[b] * wi;
        const vi = re[b] * wi + im[b] * wr;
        re[b] = re[a] - vr;
        im[b] = im[a] - vi;
        re[a] += vr;
        im[a] += vi;
      }
    }
  }
}

/** Bluestein plan for one series length N: chirp, its transform, scratch. */
interface ChirpPlan {
  M: number;
  wr: Float64Array; // chirp w[k] = e^(+i pi k^2 / N)
  wi: Float64Array;
  br: Float64Array; // FFT of the conjugate chirp, wrapped to length M
  bi: Float64Array;
  ar: Float64Array; // scratch
  ai: Float64Array;
}
const plans = new Map<number, ChirpPlan>();

function chirpPlan(N: number): ChirpPlan {
  let pl = plans.get(N);
  if (pl) return pl;
  let M = 1;
  while (M < 2 * N - 1) M <<= 1;
  const wr = new Float64Array(N), wi = new Float64Array(N);
  for (let k = 0; k < N; k++) {
    // k^2 reduced mod 2N keeps the phase exact for long series
    const a = (Math.PI * ((k * k) % (2 * N))) / N;
    wr[k] = Math.cos(a);
    wi[k] = Math.sin(a);
  }
  const br = new Float64Array(M), bi = new Float64Array(M);
  for (let k = 0; k < N; k++) {
    br[k] = wr[k];
    bi[k] = -wi[k];
    if (k > 0) {
      br[M - k] = wr[k];
      bi[M - k] = -wi[k];
    }
  }
  fftCached(br, bi, false);
  pl = { M, wr, wi, br, bi, ar: new Float64Array(M), ai: new Float64Array(M) };
  plans.set(N, pl);
  return pl;
}

/**
 * out[n] += Re sum_{k=0}^{N-1} X[k] e^(+2 pi i k n / N), n = 0..N-1, for any
 * N (Bluestein: k n = (k^2 + n^2 - (n-k)^2)/2 turns the sum into a
 * convolution with a chirp, done with power-of-two FFTs). Unnormalized, as
 * the Fortran plain-sum inverse DFT the wavemaker needs.
 */
export function idftRealAdd(
  out: Float64Array,
  N: number,
  Xr: Float64Array,
  Xi: Float64Array,
): void {
  const { M, wr, wi, br, bi, ar, ai } = chirpPlan(N);
  ar.fill(0);
  ai.fill(0);
  for (let k = 0; k < N; k++) {
    ar[k] = Xr[k] * wr[k] - Xi[k] * wi[k];
    ai[k] = Xr[k] * wi[k] + Xi[k] * wr[k];
  }
  fftCached(ar, ai, false);
  for (let k = 0; k < M; k++) {
    const r = ar[k] * br[k] - ai[k] * bi[k];
    ai[k] = ar[k] * bi[k] + ai[k] * br[k];
    ar[k] = r;
  }
  fftCached(ar, ai, true);
  for (let n = 0; n < N; n++) out[n] += (ar[n] * wr[n] - ai[n] * wi[n]) / M;
}
