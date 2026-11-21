// SPDX-License-Identifier: LGPL-3.0-or-later
// XBeach WebGPU solver — derived from XBeach (c) Deltares (LGPL v2.1+).
// Copyright (c) 2026 Oceanum. See LICENSE.
/**
 * Spectral wavemaker (surfbeat): generates the E_series / q_series boundary
 * condition tables from a 2-D variance-density spectrum, porting
 * trunk/src/xbeachlibrary/waveparamsnew.F90 subroutine spectral_wave_bc
 * (:98-335) and its helpers on the wbcversion=3, nonhspectrum=0,
 * swkhmin<=0 path. single_dir=1 (OCE-252): one instationary bin spanning the
 * window (ntheta=1, as initialize.F90 sizes it) plus the stationary boundary
 * spread Es (set_stationary_spectrum, :3077-3112).
 *
 * v1 scope (documented in docs/deviations.md): single spectrum location
 * (nspectrumloc=1 — Sfinterp is alongshore-constant, phases still vary per
 * boundary point) and a single condition block per run (lastwaveelevation
 * starts at zero); cyclicdiradjust, swkhmin>0, single_dir and the nonh paths
 * are out of scope.
 *
 * The synthesis inverse FFTs are the plain unnormalized inverse DFT sum:
 * Fortran computes ifft(Gn)/sqrt(N) then Re(.*N) (math_tools.F90 fftn
 * scaling, :330-332). They are evaluated for any (even) tslen by
 * fft.ts idftRealAdd, from the lower-half bins and their conjugate mirror
 * (lowerHalfIdft below). Components keep the Fortran frequency
 * inconsistency: the time oscillation uses Fourier bin Findex(i) (realized
 * frequency (Findex(i)-1)/rtin) while kgen/wgen stay on fgen (:2124-2148).
 */

import { XbRandom } from "./rng.ts";
import { hilbertEnvelope, idftRealAdd } from "./fft.ts";
import {
  interpUsingTrapezRule,
  buildSpecin,
  combineSpectra,
  interpolateSpectrum,
  iterateDispersion,
  linearInterp,
  NAINT,
} from "./spectrum.ts";
import type { SpectrumInput } from "./spectrum.ts";
import type { Params } from "./params.ts";
import type { Grid } from "./grid.ts";

export type { SpectrumInput, SpectrumTable } from "./spectrum.ts";

// ---------------------------------------------------------------- interfaces

/**
 * One boundary-condition block, in the exact shape the engine's spectral BC
 * path consumes (Model.setSpectralBnd / bcf.ts): eData holds nrec records of
 * (ny+1)*ntheta float64 (j fastest, then theta) in J/m^2/rad; qData holds
 * nrec records of (ny+1)*4 float64 ([qx|qy|qtot|eta], j fastest) in m^2/s.
 * nrec = tslenbc+2 with the last two records duplicating record tslenbc
 * ("IMDB bugfix", waveparamsnew.F90:2541,2547 / :3004,3008).
 */
export interface SpectralCondition {
  endtime: number;
  rt: number;
  dtbc: number;
  Trep: number;
  theta0: number; // radians, Cartesian world frame (= combspec dir0)
  Hrms: number; // hm0/sqrt(2)
  nrec: number;
  eData: Float64Array;
  qData: Float64Array;
  /** OCE-252 single_dir: the stationary boundary spread ee_s(1,:,:) in
   * J/m^2/rad, (ny+1)*ntheta_s, j fastest — the Fortran Es_series record
   * (set_stationary_spectrum); absent when single_dir=0 */
  esData?: Float64Array;
}

/** Boundary condition read back from .bcf files (reuse path, bcf.ts). */
export interface BcCondition extends SpectralCondition {
  eFile: string;
  qFile: string;
}

/**
 * Optional dump of the intermediate wave-train realization, for unit tests
 * and the Tier-1 parity tooling (genbcf --dump-components). Filled in place
 * when passed via MakeSpectralOpts.diag; arrays are the live internals (do
 * not mutate). thetagen holds the post-wrap/post-nspr values, i.e. what
 * generate_qbcf consumes (waveparamsnew.F90:2213-2284).
 */
export interface WavemakerDiag {
  h0?: number;
  Trep?: number;
  Hrms?: number;
  theta0?: number;
  K?: number;
  dfgen?: number;
  /** 0-based first Fourier bin = Findex(1)-1 (waveparamsnew.F90:2124-2128);
   * component i (0-based) oscillates at bin findex0+i. */
  findex0?: number;
  fgen?: Float64Array;
  phigen?: Float64Array;
  thetagen?: Float64Array;
  kgen?: Float64Array;
  A?: Float64Array;
  Sfinterp?: Float64Array;
  Sfinterpq?: Float64Array;
  Hm0interp?: number;
  WDindex?: Int32Array;
  timeAxis?: WaveTimeAxis;
  /** % of variance in components with WDindex==0 (waveparamsnew.F90:2288-2304). */
  perclost?: number;
}

export interface MakeSpectralOpts {
  /** Overrides the random=1 clock seed (test hook). Ignored for random=0,
   * which always seeds with 100 (waveparamsnew.F90:1615-1619). */
  seed?: number;
  /** When present, filled with the intermediate realization (see WavemakerDiag). */
  diag?: WavemakerDiag;
}

// ------------------------------------------------------------------- helpers

/**
 * frange port (waveparamsnew.F90:1759-1799): first/last frequency bins with
 * Sf > sprdthr*max(Sf) (strict) and f <= fmax. Fortran finds them with
 * first-occurrence maxloc idioms (:1787-1792) — ties resolve to the first
 * index. Returns 0-based indices.
 */
export function frange(
  f: Float64Array,
  Sf: Float64Array,
  sprdthr: number,
  fmax: number,
): { ind1: number; ind2: number } {
  let maxSf = -Infinity;
  for (let i = 0; i < Sf.length; i++) if (Sf[i] > maxSf) maxSf = Sf[i];
  if (!(maxSf > 0)) throw new Error("frange: spectrum has no energy");
  let first = -1, last = -1;
  for (let i = 0; i < Sf.length; i++) {
    if (Sf[i] > sprdthr * maxSf && !(f[i] > fmax)) {
      if (first < 0) first = i;
      last = i;
    }
  }
  return { ind1: first, ind2: last };
}

/** Internal time axis + tapers (generate_wave_time_axis output). */
export interface WaveTimeAxis {
  dtin: number;
  dtchanged: boolean;
  tslen: number;
  tslenbc: number;
  rtin: number;
  tin: Float64Array;
  taperw: Float64Array; // start ramp only (:1894-1898); no end taper on E
  taperf: Float64Array; // start ramp + mirrored end ramp, 0 from ind_end_taper
  indEndTaper: number; // 1-based ind_end_taper, as in Fortran (:1911)
}

/**
 * generate_wave_time_axis port (waveparamsnew.F90:1846-1911, surfbeat
 * branch): tslenbc = nint(rtbc/dtbc)+1; dtin = dtbc reduced to 0.5/fgen(K)
 * when larger (dtchanged); tslen = ceiling(1/dfgen/dtin)+1 rounded up to
 * even; tapers of length ntaper = nint(5*Trep/dtin) with tanh(5i/ntaper)
 * ramps; taperf additionally ramps down before, and is zeroed from, the
 * first index with tin >= rtbc.
 */
export function makeWaveTimeAxis(
  rtbc: number,
  dtbc: number,
  fgenK: number,
  trep: number,
  dfgen: number,
): WaveTimeAxis {
  const tslenbc = Math.round(rtbc / dtbc) + 1; // :1850
  let dtin = dtbc;
  let dtchanged = false;
  if (dtin > 0.5 / fgenK) { // :1856-1859
    dtin = 0.5 / fgenK;
    dtchanged = true;
  }
  let tslen = Math.ceil(1 / dfgen / dtin) + 1; // :1868
  if (tslen % 2 !== 0) tslen += 1; // :1869-1871
  const rtin = tslen * dtin; // :1874
  const tin = new Float64Array(tslen);
  for (let i = 0; i < tslen; i++) tin[i] = i * dtin;
  const taperf = new Float64Array(tslen).fill(1);
  const taperw = new Float64Array(tslen).fill(1);
  const ntaper = Math.round((5 * trep) / dtin); // :1892 (nint)
  for (let i = 1; i <= Math.min(ntaper, tslen); i++) {
    taperf[i - 1] = Math.tanh((5 * i) / ntaper);
    taperw[i - 1] = taperf[i - 1];
  }
  // end taper on taperf only (:1902-1911)
  let indend: number; // 1-based
  if (tin[tslen - 1] > rtbc) {
    indend = tslen;
    for (let i = 0; i < tslen; i++) {
      if (tin[i] >= rtbc) {
        indend = i + 1; // first masked index (minval(minloc(...)))
        break;
      }
    }
  } else {
    indend = tslen;
  }
  for (let i = 1; i <= Math.min(ntaper, indend); i++) {
    taperf[indend - i] = Math.min(taperf[indend - i], Math.tanh((5 * i) / ntaper));
  }
  for (let i = indend - 1; i < tslen; i++) taperf[i] = 0;
  return { dtin, dtchanged, tslen, tslenbc, rtin, tin, taperw, taperf, indEndTaper: indend };
}

/**
 * distribute_wave_train_directions port (waveparamsnew.F90:2167-2309,
 * nonhspec=false): wraps each direction by +-2pi into the model bin-edge
 * range where possible, then assigns 1-based bin ids on the EDGES
 * thetamin+(i-1)*dtheta — 0 below range, ntheta+1 above (both dropped from
 * E), exact top-edge equality goes to the top bin. nspr==1 snaps thetagen to
 * the bin CENTRES theta and clamps ids into [1,ntheta] (:2264-2284).
 * Mutates thetagen in place (wrap and nspr snap), as the Fortran does —
 * generate_qbcf consumes the mutated values.
 */
export function assignDirectionBins(
  thetagen: Float64Array,
  thetaminRad: number,
  dtheta: number,
  ntheta: number,
  theta: Float64Array,
  nspr: number,
): Int32Array {
  const K = thetagen.length;
  const edges = new Float64Array(ntheta + 1);
  for (let i = 0; i <= ntheta; i++) edges[i] = thetaminRad + i * dtheta;
  const emin = edges[0], emax = edges[ntheta];
  for (let i = 0; i < K; i++) {
    let tooSmall = thetagen[i] < emin;
    let tooLarge = thetagen[i] > emax;
    while (tooSmall) {
      thetagen[i] += 2 * Math.PI;
      tooSmall = thetagen[i] < emin;
    }
    while (tooLarge) {
      thetagen[i] -= 2 * Math.PI;
      tooLarge = thetagen[i] > emax;
    }
  }
  const WDindex = new Int32Array(K);
  for (let i = 0; i < K; i++) {
    const th = thetagen[i];
    if (th < emin) WDindex[i] = 0;
    else if (th > emax) WDindex[i] = ntheta + 1;
    else if (th === emax) WDindex[i] = ntheta;
    else {
      for (let ii = 1; ii <= ntheta; ii++) {
        if (th >= edges[ii - 1] && th < edges[ii]) WDindex[i] = ii;
      }
    }
  }
  if (nspr === 1) {
    for (let i = 0; i < K; i++) {
      if (WDindex[i] === 0) WDindex[i] = 1;
      else if (WDindex[i] > ntheta) WDindex[i] = ntheta;
      thetagen[i] = theta[WDindex[i] - 1];
    }
  }
  return WDindex;
}

/** Difference-interaction properties of one primary pair. */
export interface BoundPair {
  deltheta: number;
  k3: number;
  cg3: number;
  D: number;
}

/**
 * Bound-wave pair properties (generate_qbcf inner loop,
 * waveparamsnew.F90:2787-2823): deltheta = |th2-th1|+pi; difference wave
 * number k3 per Van Dongeren et al. 2003 eq. 19 (the +pi makes this the
 * difference vector); cg3 = 2*pi*deltaf/k3 limited to
 * nmax*sqrt(g/k3*tanh(k3*h0)) (:2799-2804); interaction coefficient D per
 * Okihiro et al. eq. 4a for surface elevation, with the LIMITED T2n=cg3*k3
 * in the resonance denominator (the shallow-water/nmax protection).
 */
export function boundPair(
  g: number,
  h0: number,
  nmax: number,
  w1: number,
  w2: number,
  k1: number,
  k2: number,
  th1: number,
  th2: number,
  deltaf: number,
): BoundPair {
  const deltheta = Math.abs(th2 - th1) + Math.PI;
  const cosdt = Math.cos(deltheta);
  const k3 = Math.sqrt(k1 * k1 + k2 * k2 + 2 * k1 * k2 * cosdt);
  let cg3 = (2 * Math.PI * deltaf) / k3;
  cg3 = Math.min(cg3, nmax * Math.sqrt((g / k3) * Math.tanh(k3 * h0)));
  const term1 = -w1 * w2;
  const term2 = -w1 + w2;
  const term2new = cg3 * k3;
  const chk1 = Math.cosh(k1 * h0);
  const chk2 = Math.cosh(k2 * h0);
  const D = (-g * k1 * k2 * cosdt) / 2 / term1 +
    (term2 * term2) / (g * 2) +
    ((g * term2) / ((g * k3 * Math.tanh(k3 * h0) - term2new * term2new) * term1)) *
      (term2 * ((term1 * term1) / g / g - k1 * k2 * cosdt) -
        0.5 * ((-w1 * k2 * k2) / (chk2 * chk2) + (w2 * k1 * k1) / (chk1 * chk1)));
  return { deltheta, k3, cg3, D };
}

/**
 * The Fortran plain-sum inverse DFT of a length-N spectrum given by its
 * lower-half bins, whose upper half is the flipped conjugate of bins
 * 1..N/2-1 (generate_ebcf :2393-2417, generate_qbcf :2915-2934). A bin
 * (0-based index b, complex G_b) contributes
 * f*(Re G_b cos(w n) - Im G_b sin(w n)) with f=2 for 0<b<N/2 and f=1 for
 * b=0 or b=N/2; bins with b>N/2 are dropped, as the Fortran conjugate-mirror
 * assignment overwrites that region of Gn. Bins are collected with add()
 * and the real series is accumulated into out by flush().
 */
class LowerHalfIdft {
  private readonly xr: Float64Array;
  private readonly xi: Float64Array;
  private any = false;
  constructor(private readonly N: number) {
    this.xr = new Float64Array(N);
    this.xi = new Float64Array(N);
  }
  add(b: number, gRe: number, gIm: number): void {
    if (b > this.N / 2) return;
    const factor = b === 0 || 2 * b === this.N ? 1 : 2;
    this.xr[b] += factor * gRe;
    this.xi[b] += factor * gIm;
    this.any = true;
  }
  flush(out: Float64Array): void {
    if (!this.any) return;
    idftRealAdd(out, this.N, this.xr, this.xi);
    this.xr.fill(0);
    this.xi.fill(0);
    this.any = false;
  }
}

// --------------------------------------------------------------- main entry

/**
 * Generate the spectral boundary-condition block(s) for a run. Port of the
 * spectral_wave_bc pipeline (waveparamsnew.F90:98-335): spectrum chain ->
 * wave-train components (random draws + dispersion) -> time axis -> variance
 * and amplitudes -> Fourier components with alongshore phase unrolling ->
 * directional binning -> generate_ebcf -> generate_qbcf. v1 returns a single
 * block covering [0, p.rt].
 */
export function makeSpectralConditions(
  spec: SpectrumInput,
  p: Params,
  grid: Grid,
  opts: MakeSpectralOpts = {},
): SpectralCondition[] {
  // v1 scope guards (see module header / docs/deviations.md)
  if (p.nspectrumloc !== 1) {
    throw new Error("wavemaker: nspectrumloc>1 (LOCLIST alongshore interpolation) not implemented");
  }
  if (p.wbcversion !== 3) {
    throw new Error("wavemaker: only wbcversion=3 is implemented");
  }

  const ny = grid.ny;
  const m1 = ny + 1;
  const ntheta = grid.ntheta;
  const row = (j: number) => j * (grid.nx + 1); // (1,j) of Fortran row 1

  // boundary-average depth (calculate_average_water_depth, :1524-1538)
  let hnum = 0, hden = 0;
  for (let j = 0; j < m1; j++) {
    hnum += (p.zs0 - grid.zb[row(j)]) * grid.dnc[row(j)];
    hden += grid.dnc[row(j)];
  }
  const h0 = hnum / hden;

  const rtbc = p.rt; // :402 (morfac time scaling out of scope)
  const dtbc = p.dtbc; // :403

  // --- spectrum chain (spec sections 1.1-1.3)
  const specin = buildSpecin(spec, h0, p.g);
  let fmax = 1; // "assume 1Hz as maximum frequency" (:165)
  for (let i = 0; i < specin.f.length; i++) fmax = Math.max(fmax, specin.f[i]);
  const tm01 = p.Tm01switch === 1;
  const specinterp = interpolateSpectrum(specin, fmax, p.trepfac, tm01);
  const comb = combineSpectra([specinterp], p.trepfac, tm01);
  const Trep = comb.trep; // :206
  const Hrms = comb.hm0 / Math.sqrt(2); // :207
  const theta0 = comb.dir0;

  // --- wave-train components (generate_wavetrain_components, :1544-1723)
  const fmaxRange = 2 * comb.f[comb.f.length - 1]; // :1575 (surfbeat no-op)
  const { ind1, ind2 } = frange(comb.f, comb.Sf, p.sprdthr, fmaxRange);
  let K = Math.ceil(rtbc * (comb.f[ind2] - comb.f[ind1]) + 1); // :1586
  K = Math.max(K, 200); // Kmin (:1588)
  const dfgen = (comb.f[ind2] - comb.f[ind1]) / (K - 1); // :1600
  const fgen = new Float64Array(K);
  for (let i = 0; i < K; i++) fgen[i] = comb.f[ind1] + i * dfgen;

  // draws — ONE stream, order load-bearing (:1610-1662): first number seeds
  // (random=0 -> seed 100, deterministic; random=1 -> clock seed, here
  // Date.now(); SYSTEM_CLOCK semantics differ so random=1 gives statistical
  // parity only), then 2K-1 sequential draws.
  const rng = new XbRandom();
  const randnums = new Float64Array(2 * K);
  randnums[0] = p.random === 1 ? rng.seed(opts.seed ?? Date.now()) : rng.seed(100);
  for (let i = 1; i < 2 * K; i++) randnums[i] = rng.next();

  const phigen = new Float64Array(K); // :1624
  for (let i = 0; i < K; i++) phigen[i] = randnums[i] * 2 * Math.PI;

  // random directions by inverse-CDF of the directional distribution at
  // fgen(i) (:1631-1662)
  const thetagen = new Float64Array(K);
  {
    const pdf = new Float64Array(NAINT);
    const cdf = new Float64Array(NAINT);
    const nf = comb.f.length;
    for (let i = 0; i < K; i++) {
      // per-angle linear interpolation of S in f, exact linear_interp
      // semantics on the shared frequency axis (:1633-1635)
      const l1 = comb.f[nf - 1] > comb.f[0];
      let jl = 0, ju = nf + 1;
      while (ju - jl > 1) {
        const jm = (ju + jl) >> 1;
        const l2 = fgen[i] > comb.f[jm - 1];
        if ((l1 && l2) || (!l1 && !l2)) jl = jm;
        else ju = jm;
      }
      let sum = 0;
      for (let ii = 0; ii < NAINT; ii++) {
        let v: number;
        if (jl <= 0) v = comb.S[0 * NAINT + ii];
        else if (jl >= nf) v = comb.S[(nf - 1) * NAINT + ii];
        else {
          const a = comb.f[jl], b = comb.f[jl - 1];
          const y0 = comb.S[(jl - 1) * NAINT + ii], y1 = comb.S[jl * NAINT + ii];
          const dyy = a === b ? 0 : (y1 - y0) / (a - b);
          v = y0 + (fgen[i] - b) * dyy;
        }
        pdf[ii] = v;
        sum += v;
      }
      for (let ii = 0; ii < NAINT; ii++) pdf[ii] /= sum; // :1640
      cdf[0] = pdf[0]; // :1645
      for (let ii = 1; ii < NAINT; ii++) {
        cdf[ii] = cdf[ii - 1] + (pdf[ii] + pdf[ii - 1]) / 2;
      }
      const draw = randnums[K + i];
      if (draw >= cdf[0]) {
        thetagen[i] = linearInterp(cdf, comb.ang, draw); // :1654
      } else {
        // wrap segment (0, ang(end)-2pi) -> (cdf(1), ang(1)) (:1656-1660)
        thetagen[i] = linearInterp(
          [0, cdf[0]],
          [comb.ang[NAINT - 1] - 2 * Math.PI, comb.ang[0]],
          draw,
        );
        if (thetagen[i] < 0) thetagen[i] += 2 * Math.PI;
      }
    }
  }

  // long-crested override (:223-226): nspectra==1 and scoeff>1000
  if (specin.scoeff > 1000) thetagen.fill(specin.dir0);

  // dispersion (:1668-1677) — golden-ratio solver, NOT the engine's Newton
  const kgen = new Float64Array(K);
  const wgen = new Float64Array(K);
  for (let i = 0; i < K; i++) {
    const L0 = (p.g * (1 / fgen[i]) ** 2) / 2 / Math.PI;
    let L = iterateDispersion(L0, L0, h0);
    if (L < 0) L = -L; // dead branch, kept for fidelity (:1671-1675)
    kgen[i] = (2 * Math.PI) / L;
    wgen[i] = 2 * Math.PI * fgen[i]; // :1680
  }

  // --- time axis and tapers (spec 3.1)
  const ta = makeWaveTimeAxis(rtbc, dtbc, fgen[K - 1], Trep, dfgen);
  const { tslen, tslenbc, dtin, dtchanged, tin } = ta;

  // --- variance -> amplitudes (spec 3.2), nspectrumloc=1: alongshore
  // constant (generate_wave_train_variance :1918-1972 and the non-interp
  // branch of generate_wave_train_properties_per_offshore_point :2071-2076)
  const SfRaw = new Float64Array(K);
  for (let i = 0; i < K; i++) SfRaw[i] = linearInterp(specinterp.f, specinterp.Sf, fgen[i]);
  let vsum = 0;
  for (let i = 0; i < K; i++) vsum += SfRaw[i];
  const hm0post = 4 * Math.sqrt(vsum * dfgen);
  const vscale = (specinterp.hm0 / hm0post) ** 2; // unconditional (:1958-1959)
  const Sfinterp = new Float64Array(K); // vargen
  const Sfinterpq = new Float64Array(K); // vargenq = min(scaled, raw) (:1965-1969)
  for (let i = 0; i < K; i++) {
    Sfinterp[i] = SfRaw[i] * vscale;
    Sfinterpq[i] = Math.min(Sfinterp[i], SfRaw[i]);
  }
  const A = new Float64Array(K); // :2074
  for (let i = 0; i < K; i++) A[i] = Math.sqrt(2 * Sfinterp[i] * dfgen);
  let hsum = 0;
  for (let i = 0; i < K; i++) hsum += Sfinterp[i];
  const Hm0interp = 4 * Math.sqrt(hsum * dfgen); // :2076

  // --- Fourier components (generate_wave_train_Fourier, :2105-2162):
  // CompFn(j,i) = A/2 e^(i phigen) e^(-i kgen (sin(th) dy + cos(th) dx)),
  // stored sparse per (boundary point, component); bin index
  // Findex(i)-1 = floor(fgen(1)/dfgen) + i (0-based i). Uses thetagen BEFORE
  // the +-2pi wrap / nspr snap of the direction binning (call order :241/:248).
  const findex0 = Math.floor(fgen[0] / dfgen); // :2124-2128
  const compRe = new Float64Array(m1 * K);
  const compIm = new Float64Array(m1 * K);
  const x0 = grid.xz[row(0)], y0 = grid.yz[row(0)];
  for (let j = 0; j < m1; j++) {
    const dxj = grid.xz[row(j)] - x0;
    const dyj = grid.yz[row(j)] - y0;
    for (let i = 0; i < K; i++) {
      const ph = phigen[i] -
        kgen[i] * (Math.sin(thetagen[i]) * dyj + Math.cos(thetagen[i]) * dxj);
      compRe[j * K + i] = (A[i] / 2) * Math.cos(ph);
      compIm[j * K + i] = (A[i] / 2) * Math.sin(ph);
    }
  }

  // --- directional binning (spec 3.3); mutates thetagen (wrap/nspr) which
  // generate_qbcf then consumes
  const WDindex = assignDirectionBins(
    thetagen,
    grid.thetamin,
    grid.dtheta,
    ntheta,
    grid.theta,
    p.nspr,
  );

  // energy outside the computational bins (:2288-2304): WDindex==0 counts as
  // lost; rescaling later re-injects the remainder (wbcScaleEnergy)
  let lostvar = 0, keptvar = 0;
  for (let i = 0; i < K; i++) {
    if (WDindex[i] === 0) lostvar += m1 * A[i] * A[i];
    else keptvar += m1 * A[i] * A[i];
  }
  const perclost = (100 * lostvar) / (lostvar + keptvar);
  if (perclost > 5) {
    console.warn(
      `wavemaker: ${perclost.toFixed(1)}% of wave energy falls outside the ` +
        `computational domain at the offshore boundary`,
    );
  }

  const idft = new LowerHalfIdft(tslen);
  const nrec = tslenbc + 2;
  const recE = m1 * ntheta;
  const recQ = m1 * 4;
  const eData = new Float64Array(nrec * recE);
  const qData = new Float64Array(nrec * recQ);

  // ============================ generate_ebcf (:2314-2561) ==================
  {
    // components per 1-based theta bin (:2366-2387)
    const bins: number[][] = [];
    for (let it = 0; it < ntheta; it++) bins.push([]);
    for (let i = 0; i < K; i++) {
      if (WDindex[i] >= 1 && WDindex[i] <= ntheta) bins[WDindex[i] - 1].push(i);
    }
    const zeta = new Float64Array(ntheta * tslen);
    const eta = new Float64Array(tslen);
    const E = new Float64Array(ntheta * tslen);
    const Erow = new Float64Array(tslenbc);
    for (let j = 0; j < m1; j++) {
      E.fill(0);
      if (p.bclwonly !== 1) { // :2341 (bclwonly==1 -> zero short-wave E)
        zeta.fill(0);
        eta.fill(0);
        for (let it = 0; it < ntheta; it++) {
          const zrow = zeta.subarray(it * tslen, (it + 1) * tslen);
          for (const i of bins[it]) idft.add(findex0 + i, compRe[j * K + i], compIm[j * K + i]);
          idft.flush(zrow);
          // startup blend with lastwaveelevation (zeros, single-block v1)
          // (:2417-2418)
          for (let n = 0; n < tslen; n++) zrow[n] *= ta.taperw[n];
          for (let n = 0; n < tslen; n++) eta[n] += zrow[n];
        }
        // Hilbert envelope of the full short-wave signal (:2436-2447)
        const Amp = hilbertEnvelope(eta);
        let s2 = 0;
        for (let n = 0; n < tslen; n++) s2 += eta[n] * eta[n];
        const stdeta = Math.sqrt(s2 / (tslen - 1)); // :2452 (1/(n-1))
        for (let it = 0; it < ntheta; it++) {
          if (bins[it].length === 0) continue; // Ampzeta stays 0 (:2462-2466)
          const zrow = zeta.subarray(it * tslen, (it + 1) * tslen);
          let z2 = 0;
          for (let n = 0; n < tslen; n++) z2 += zrow[n] * zrow[n];
          const ratio = Math.sqrt(z2 / (tslen - 1)) / stdeta; // :2455-2461
          const erow = E.subarray(it * tslen, (it + 1) * tslen);
          for (let n = 0; n < tslen; n++) {
            const az = Amp[n] * ratio;
            erow[n] = (0.5 * p.rho * p.g * az * az) / grid.dtheta; // :2482-2485
          }
        }
      }
      // wbcEvarreduce (:2489-2496)
      if (p.wbcEvarreduce < 1 - 1e-10) {
        for (let it = 0; it < ntheta; it++) {
          const erow = E.subarray(it * tslen, (it + 1) * tslen);
          let em = 0;
          for (let n = 0; n < tslen; n++) em += erow[n];
          em /= tslen;
          for (let n = 0; n < tslen; n++) {
            erow[n] = p.wbcEvarreduce * (erow[n] - em) + em;
          }
        }
      }
      // wbcScaleEnergy (:2499-2509): rescale the kept energy to the full Hm0
      if (p.wbcScaleEnergy === 1) {
        const nIBF = Math.round(((tslenbc - 1) * dtbc) / dtin);
        let esum = 0;
        for (let it = 0; it < ntheta; it++) {
          for (let n = 0; n < nIBF; n++) esum += E[it * tslen + n];
        }
        const emean = (esum * grid.dtheta) / nIBF;
        const target = (Hm0interp / Math.sqrt(2)) ** 2 * p.rhog8; // rhog8 = rho*g/8
        if (emean > 0) { // guard: Fortran divides 0/0 -> NaN when E is all zero
          const fac = target / emean;
          for (let n = 0; n < E.length; n++) E[n] *= fac;
        }
      }
      // write records (:2520-2551): j fastest then theta; tslenbc+2 records,
      // the last two clamped to record tslenbc; dtchanged interpolates to the
      // dtbc axis first
      for (let it = 0; it < ntheta; it++) {
        const erow = E.subarray(it * tslen, (it + 1) * tslen);
        let src: ArrayLike<number>;
        if (dtchanged) {
          for (let r = 0; r < tslenbc; r++) Erow[r] = linearInterp(tin, erow, r * dtbc);
          src = Erow;
        } else {
          src = erow;
        }
        for (let r = 0; r < nrec; r++) {
          eData[r * recE + it * m1 + j] = src[Math.min(r, tslenbc - 1)];
        }
      }
    }
  }

  // ============================ generate_qbcf (:2695-3026) ==================
  {
    // Triangular pair arrays, m = 1..K-1 (difference index), i = 0..K-1-m.
    // With nspectrumloc=1 the amplitudes Abnd are alongshore-constant
    // (Sfinterp does not vary with j), so the per-pair complex bases are
    // precomputed once; only the alongshore phase unroll depends on j.
    const nPairs = (K - 1) * K; // full (K-1) x K storage, as Fortran allocates
    const baseRe = new Float64Array(nPairs); // Abnd/2 * e^(-i dphi3)
    const baseIm = new Float64Array(nPairs);
    const facx = new Float64Array(nPairs); // cg3*cos(theta3)
    const facy = new Float64Array(nPairs); // cg3*sin(theta3)
    const facq = new Float64Array(nPairs); // cg3
    const KKx = new Float64Array(nPairs);
    const KKy = new Float64Array(nPairs);
    // arg(conj CompFn(1,i)) for the bound-wave phase (:2846-2850)
    const argConj = new Float64Array(K);
    for (let i = 0; i < K; i++) argConj[i] = Math.atan2(-compIm[i], compRe[i]);
    const SfQ = p.Sfold === 1 ? Sfinterpq : Sfinterp; // :2879-2887
    for (let m = 1; m < K; m++) {
      const deltaf = m * dfgen;
      const rowOff = (m - 1) * K;
      if (deltaf <= p.fcutoff) continue; // :2839 (whole row excluded)
      for (let i = 0; i < K - m; i++) {
        const i2 = i + m;
        // lower-limit exclusion, Herbers 1994 eq. 1 (:2833-2835)
        if (fgen[i] <= deltaf) continue;
        const bp = boundPair(
          p.g,
          h0,
          p.nmax,
          wgen[i],
          wgen[i2],
          kgen[i],
          kgen[i2],
          thetagen[i],
          thetagen[i2],
          deltaf,
        );
        const kkx = kgen[i2] * Math.cos(thetagen[i2]) - kgen[i] * Math.cos(thetagen[i]); // :2790
        const kky = kgen[i2] * Math.sin(thetagen[i2]) - kgen[i] * Math.sin(thetagen[i]); // :2791
        // Eforc = 2 D^2 Sf1 Sf2 dfgen; Abnd = sqrt(2 Eforc dfgen)*sign(D)
        // (:2879-2899; the pi phase term is absent, sign(D) carries it)
        const Eforc = 2 * bp.D * bp.D * SfQ[i] * SfQ[i2] * dfgen;
        const Abnd = Math.sqrt(2 * Eforc * dfgen) * (bp.D < 0 ? -1 : 1);
        const dphi3 = argConj[i2] - argConj[i]; // :2850
        const theta3 = Math.atan2(kky, kkx); // :2854
        const k = rowOff + i;
        baseRe[k] = (Abnd / 2) * Math.cos(-dphi3);
        baseIm[k] = (Abnd / 2) * Math.sin(-dphi3);
        facx[k] = bp.cg3 * Math.cos(theta3);
        facy[k] = bp.cg3 * Math.sin(theta3);
        facq[k] = bp.cg3;
        KKx[k] = kkx;
        KKy[k] = kky;
      }
    }

    const gRe = new Float64Array(4 * K); // per-iq bins, bin index m
    const gIm = new Float64Array(4 * K);
    const q = new Float64Array(4 * tslen);
    const qRow = new Float64Array(tslenbc);
    const nIBFq = Math.round(rtbc / dtin); // :2943
    for (let j = 0; j < m1; j++) {
      const dxj = grid.xz[row(j)] - x0;
      const dyj = grid.yz[row(j)] - y0;
      gRe.fill(0);
      gIm.fill(0);
      // assemble Gn(2:K) = sum_i Ftemp(m,i) per quantity, with the
      // alongshore unroll e^(-i(KKy dy + KKx dx)) (:2904-2917)
      for (let m = 1; m < K; m++) {
        const rowOff = (m - 1) * K;
        let sxr = 0, sxi = 0, syr = 0, syi = 0, sqr = 0, sqi = 0, ser = 0, sei = 0;
        for (let i = 0; i < K - m; i++) {
          const k = rowOff + i;
          const br = baseRe[k], bi = baseIm[k];
          if (br === 0 && bi === 0) continue;
          const ph = -(KKy[k] * dyj + KKx[k] * dxj);
          const cr = Math.cos(ph), ci = Math.sin(ph);
          const rr = br * cr - bi * ci;
          const ri = br * ci + bi * cr;
          sxr += rr * facx[k];
          sxi += ri * facx[k];
          syr += rr * facy[k];
          syi += ri * facy[k];
          sqr += rr * facq[k];
          sqi += ri * facq[k];
          ser += rr;
          sei += ri;
        }
        gRe[0 * K + m] = sxr;
        gIm[0 * K + m] = sxi;
        gRe[1 * K + m] = syr;
        gIm[1 * K + m] = syi;
        gRe[2 * K + m] = sqr;
        gIm[2 * K + m] = sqi;
        gRe[3 * K + m] = ser;
        gIm[3 * K + m] = sei;
      }
      // inverse transform (plain sum) and taperf window, both ends
      // (:2925-2934)
      q.fill(0);
      for (let iq = 0; iq < 4; iq++) {
        const qrow = q.subarray(iq * tslen, (iq + 1) * tslen);
        for (let m = 1; m < K; m++) {
          const re = gRe[iq * K + m], im = gIm[iq * K + m];
          if (re === 0 && im === 0) continue;
          idft.add(m, re, im);
        }
        idft.flush(qrow);
        for (let n = 0; n < tslen; n++) qrow[n] *= ta.taperf[n];
      }
      // wbcRemoveStokes (:2942-2950): remove the mean over the first
      // nint(rtbc/dtin) samples
      if (p.wbcRemoveStokes === 1) {
        for (let iq = 0; iq < 4; iq++) {
          const qrow = q.subarray(iq * tslen, (iq + 1) * tslen);
          let qm = 0;
          for (let n = 0; n < nIBFq; n++) qm += qrow[n];
          qm /= nIBFq;
          for (let n = 0; n < tslen; n++) qrow[n] -= qm;
        }
      }
      // wbcQvarreduce (:2952-2959)
      if (p.wbcQvarreduce < 1 - 1e-10) {
        for (let iq = 0; iq < 4; iq++) {
          const qrow = q.subarray(iq * tslen, (iq + 1) * tslen);
          let qm = 0;
          for (let n = 0; n < tslen; n++) qm += qrow[n];
          qm /= tslen;
          for (let n = 0; n < tslen; n++) {
            qrow[n] = p.wbcQvarreduce * (qrow[n] - qm) + qm;
          }
        }
      }
      // order==1: no second-order flux / elevation (:2974-2979)
      if (p.order === 1) {
        q.fill(0, 0 * tslen, 1 * tslen);
        q.fill(0, 3 * tslen, 4 * tslen);
      }
      // write records (:2984-3010): [qx|qy|qtot|eta], j fastest, tslenbc+2
      // records with the last two clamped. In the dtchanged branch we
      // interpolate ALL FOUR quarters — deliberate non-replication of the
      // Fortran 3-slice qinterp bug (eta quarter undefined there); the engine
      // never reads eta (spec flag 1, docs/deviations.md).
      for (let iq = 0; iq < 4; iq++) {
        const qrow = q.subarray(iq * tslen, (iq + 1) * tslen);
        let src: ArrayLike<number>;
        if (dtchanged) {
          for (let r = 0; r < tslenbc; r++) qRow[r] = linearInterp(tin, qrow, r * dtbc);
          src = qRow;
        } else {
          src = qrow;
        }
        for (let r = 0; r < nrec; r++) {
          qData[r * recQ + iq * m1 + j] = src[Math.min(r, tslenbc - 1)];
        }
      }
    }
  }

  if (opts.diag) {
    Object.assign(
      opts.diag,
      {
        h0,
        Trep,
        Hrms,
        theta0,
        K,
        dfgen,
        findex0,
        fgen,
        phigen,
        thetagen,
        kgen,
        A,
        Sfinterp,
        Sfinterpq,
        Hm0interp,
        WDindex,
        timeAxis: ta,
        perclost,
      } satisfies WavemakerDiag,
    );
  }

  // OCE-252 single_dir: set_stationary_spectrum (:3077-3112) — the combined
  // spectrum's directional distribution Sd(ang) bin-averaged onto the
  // stationary grid theta_s (interp_using_trapez_rule over a 2π cycle),
  // alongshore-constant, scaled to energy density by rho·g
  let esData: Float64Array | undefined;
  if (p.single_dir === 1) {
    const ns = grid.ntheta_s;
    const spread = interpUsingTrapezRule(comb.ang, comb.Sd, 2 * Math.PI, grid.theta_s, grid.dtheta_s);
    esData = new Float64Array(m1 * ns);
    for (let it = 0; it < ns; it++) {
      const v = spread[it] * p.rho * p.g;
      for (let j = 0; j < m1; j++) esData[it * m1 + j] = v;
    }
  }

  // admin-line equivalents (generate_admin_files, :3128-3146): endtime of the
  // single v1 block is rtbc (spectrumendtime starts at 0)
  return [{
    endtime: rtbc,
    rt: rtbc,
    dtbc,
    Trep,
    theta0,
    Hrms,
    nrec,
    eData,
    qData,
    ...(esData ? { esData } : {}),
  }];
}
