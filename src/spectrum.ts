// SPDX-License-Identifier: LGPL-3.0-or-later
// XBeach WebGPU solver — derived from XBeach (c) Deltares (LGPL v2.1+).
// Copyright (c) 2026 Oceanum. See LICENSE.
/**
 * Input-spectrum construction, interpolation to the internal 801x401 grid,
 * and combined-spectrum statistics for the spectral wavemaker (surfbeat).
 *
 * Ports the reader/interpolation half of trunk/src/xbeachlibrary/
 * waveparamsnew.F90 (spectral_wave_bc, wbcversion=3): read_jonswap_file
 * (:429-882), jonswapgk (:889-930), the SWAN-table angle/unit conventions of
 * read_swan_file (:936-1146), interpolate_spectrum (:1263-1385),
 * generate_combined_spectrum (:1441-1518) and tpDcalc (:1728-1753), plus the
 * interpolation kernels from interp.F90 and the dispersion solver from
 * wave_functions.F90.
 *
 * Internal 2D spectra are variance density in m^2/Hz/rad on angles in
 * radians, Cartesian, CCW from East, world frame (waveparamsnew.F90:604-611,
 * :1093-1122). The nautical -> Cartesian conversion (270 - theta) and the
 * per-degree -> per-radian scaling (x 180/pi) happen here, on the LGPL side.
 */

/** Standard internal grid sizes (waveparamsnew.F90:87-88). */
export const NFINT = 801;
export const NAINT = 401;

// ---------------------------------------------------------------- interfaces

/** App-facing 2D spectrum table (structurally accepts the app's Spectrum2D). */
export interface SpectrumTable {
  freq: Float64Array; // Hz, absolute, ascending
  dirDeg: Float64Array; // nautical coming-from, GRID frame (alfa=0), source order
  efth: Float64Array; // m^2/Hz/deg, [ifreq*ndir + idir]
}

export type SpectrumInput =
  | {
    kind: "jons";
    hm0: number;
    fp: number;
    gammajsp: number;
    s: number;
    mainangDeg: number; // nautical coming-from, degrees
    fnyq?: number; // default max(0.3, 3*fp) (waveparamsnew.F90:537)
    dfj?: number; // default fnyq/200 (:538)
    tma?: boolean;
  }
  | { kind: "table"; table: SpectrumTable };

/** Internal input spectrum, mirrors type(spectrum) (waveparamsnew.F90:22-40). */
export interface SpecIn {
  f: Float64Array; // Hz
  ang: Float64Array; // radians Cartesian; NOT wrapped (table path can exceed [0,2pi))
  S: Float64Array; // m^2/Hz/rad, [ifreq*nang + iang]
  Sf: Float64Array; // direction-integrated, mid-point/edge rule
  scoeff: number; // -1 when not imposed; >1000 flags long-crested
  hm0: number; // -1 when not imposed (table input)
  fp: number;
  dir0: number; // radians Cartesian (jons: mainang); 0 when unused
}

/** Spectrum on the standard 801x401 grid (interpolate_spectrum output). */
export interface SpecInterp {
  f: Float64Array; // (i)*df, i=0..800
  df: number;
  ang: Float64Array; // (i)*dang, dang=2pi/400
  dang: number;
  S: Float64Array; // [ifreq*NAINT + iang]
  Sf: Float64Array;
  Sd: Float64Array;
  hm0: number;
  fp: number;
  dir0naut: number; // nautical degrees (Fortran specinterp%dir0, :1376)
  trep: number;
  dirm: number; // nautical degrees
}

/** Averaged spectrum (generate_combined_spectrum output). */
export interface CombSpec {
  f: Float64Array;
  df: number;
  ang: Float64Array;
  dang: number;
  S: Float64Array;
  Sf: Float64Array;
  Sd: Float64Array;
  trep: number;
  dir0: number; // radians, Cartesian world frame (written to the .bcf lists)
  fp: number;
  hm0: number;
}

// ------------------------------------------------------- interpolation kernels

/**
 * linear_interp + binary_search port (trunk/src/xbeachlibrary/
 * interp.F90:105-158, :180-224): clamped linear interpolation, bracketing
 * index found with the Numerical Recipes bisection using strict comparisons
 * (ties resolve to the lower segment; out-of-range clamps to the end values).
 */
export function linearInterp(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  xx: number,
): number {
  const n = x.length;
  if (n <= 0) return 0;
  if (n === 1) return y[0];
  // binary_search (interp.F90:180-224), 1-based j in [0..n]
  const l1 = x[n - 1] > x[0];
  let jl = 0, ju = n + 1;
  while (ju - jl > 1) {
    const jm = (ju + jl) >> 1;
    const l2 = xx > x[jm - 1];
    if ((l1 && l2) || (!l1 && !l2)) jl = jm;
    else ju = jm;
  }
  const j = jl;
  if (j <= 0) return y[0];
  if (j >= n) return y[n - 1];
  const a = x[j], b = x[j - 1]; // x(j+1), x(j) in Fortran 1-based
  const dyy = a === b ? 0 : (y[j] - y[j - 1]) / (a - b);
  return y[j - 1] + (xx - x[j - 1]) * dyy;
}

/**
 * interp_in_cyclic_function port (interp.F90:1475-1551): pads the input to a
 * full cycle (gap bins zero-filled, circle closed with y(1)), then samples
 * with the Fortran left/right weighting — including the quirk that the
 * right-hand weight uses mod(xp, dx) rather than mod(xp - x(1), dx).
 * Requires xp >= 0 (always true for the 0..2pi target grid).
 */
export function interpCyclic(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  xcycle: number,
  xp: ArrayLike<number>,
): Float64Array {
  const n = x.length;
  const dx = x[1] - x[0];
  const icycle = Math.round(xcycle / dx); // nint
  const nc = icycle + 1;
  if (n > nc) throw new Error("interpCyclic: input covers more than one cycle");
  const xc = new Float64Array(nc);
  const yc = new Float64Array(nc);
  for (let i = 0; i < n; i++) {
    xc[i] = x[i];
    yc[i] = y[i];
  }
  if (n <= icycle) {
    for (let i = n; i < icycle; i++) {
      xc[i] = xc[i - 1] + dx;
      yc[i] = 0;
    }
    xc[icycle] = xc[icycle - 1] + dx;
    yc[icycle] = yc[0];
  }
  const np = xp.length;
  const yp = new Float64Array(np);
  for (let ip = 0; ip < np; ip++) {
    let ileft = Math.floor((xp[ip] - xc[0]) / dx) + 1; // 1-based
    while (ileft < 1) ileft += icycle;
    while (ileft > nc) ileft -= icycle;
    const yleft = yc[ileft - 1];
    let iright = ileft + 1;
    while (iright < 1) iright += icycle;
    while (iright > nc) iright -= icycle;
    const yright = yc[iright - 1];
    const facright = (xp[ip] % dx) / dx; // Fortran mod, xp >= 0
    yp[ip] = (1 - facright) * yleft + facright * yright;
  }
  return yp;
}

/**
 * trapezoidal_cyclic + interp_using_trapez_rule port (interp.F90:1413-1473),
 * OCE-252: bin-average a cyclic function y(x) (equidistant x) onto target
 * bin centres xtarg of width dxTarg — the integral over each target bin by
 * the trapezoid rule on the source grid's nodes inside it, divided by the
 * bin width. The single_dir wavemaker uses it to put the combined
 * spectrum's directional distribution Sd(ang) on the stationary direction
 * grid theta_s (set_stationary_spectrum, waveparamsnew.F90:3097). Fortran
 * takes the bin width from xtarg(2)-xtarg(1); passed explicitly here so a
 * one-bin grid is defined.
 */
export function interpUsingTrapezRule(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  xcycle: number,
  xtarg: ArrayLike<number>,
  dxTarg: number,
): Float64Array {
  const dx = x[1] - x[0];
  const out = new Float64Array(xtarg.length);
  for (let it = 0; it < xtarg.length; it++) {
    const x1 = xtarg[it] - 0.5 * dxTarg;
    const x2 = xtarg[it] + 0.5 * dxTarg;
    // trapezoidal_cyclic: nodes x1, the source grid points strictly inside, x2
    const np = Math.floor(x2 / dx) - (Math.floor(x1 / dx) + 1) + 2;
    const xp = new Float64Array(np);
    xp[0] = x1;
    for (let ip = 1; ip < np - 1; ip++) xp[ip] = (Math.floor(x1 / dx) + ip) * dx;
    xp[np - 1] = x2;
    const yp = interpCyclic(x, y, xcycle, xp);
    let integ = 0;
    for (let ip = 0; ip < np - 1; ip++) integ += 0.5 * (xp[ip + 1] - xp[ip]) * (yp[ip + 1] + yp[ip]);
    out[it] = integ / dxTarg;
  }
  return out;
}

/** First index of the maximum value (Fortran maxval(maxloc(v)) idiom — ties
 * resolve to the first occurrence, which is load-bearing in several places). */
function imaxFirst(v: ArrayLike<number>, lo = 0, hi = v.length): number {
  let im = lo;
  for (let i = lo + 1; i < hi; i++) if (v[i] > v[im]) im = i;
  return im;
}

// ------------------------------------------------------------------ dispersion

/**
 * iteratedispersion port (trunk/src/xbeachlibrary/wave_functions.F90:805-837):
 * golden-ratio fixed point L1 <- aphi*L1 + bphi*L0*tanh(2*pi*h/L1) with
 * absolute tolerance 1e-5 on |L2-L1| and max 150 iterations. The
 * early-terminated values are load-bearing for kgen parity — do NOT swap in
 * the engine's Newton solver (see docs/deviations.md dispersion entry).
 * Fortran flags non-convergence by returning -L1, but the loop guard makes
 * iter <= itermax always true, so that branch is dead; we always return L1.
 */
export function iterateDispersion(L0: number, Lestimate: number, h: number): number {
  const phi = (1 + Math.sqrt(5)) / 2;
  const aphi = 1 / (phi + 1);
  const bphi = phi / (phi + 1);
  let L1 = Lestimate;
  let err = Infinity;
  let iter = 0;
  while (err > 0.00001 && iter < 150) {
    iter++;
    const L2 = L0 * Math.tanh((2 * Math.PI * h) / L1);
    L1 = L1 * aphi + L2 * bphi;
    err = Math.abs(L2 - L1);
  }
  return L1;
}

// ------------------------------------------------------------ JONSWAP builder

/**
 * jonswapgk port (waveparamsnew.F90:889-930): unscaled JONSWAP shape
 * y = x^-5 exp(-1.25 x^-4) gamma^exp(-(x-1)^2/(2 sigma^2)) with
 * sigma = 0.07 for x<1 else 0.09, normalized to max 1.
 */
export function jonswapgk(x: ArrayLike<number>, gam: number): Float64Array {
  const y = new Float64Array(x.length);
  let ymax = -Infinity;
  for (let i = 0; i < x.length; i++) {
    let xa = Math.abs(x[i]);
    if (xa === 0) xa = 1e-20;
    const sigma = xa < 1 ? 0.07 : 0.09;
    const fac1 = Math.pow(xa, -5);
    const fac2 = Math.exp(-1.25 * Math.pow(xa, -4));
    const fac3 = Math.pow(gam, Math.exp(-((xa - 1) ** 2) / (2 * sigma * sigma)));
    y[i] = fac1 * fac2 * fac3;
    if (y[i] > ymax) ymax = y[i];
  }
  for (let i = 0; i < y.length; i++) y[i] /= ymax;
  return y;
}

/** Direction-integrated Sf with the Fortran mid-point/edge rule
 * (waveparamsnew.F90:857-869, identical blocks at :1130-1144, :1237-1251). */
function integrateSf(f: Float64Array, ang: Float64Array, S: Float64Array): Float64Array {
  const nf = f.length, nang = ang.length;
  const Sf = new Float64Array(nf);
  for (let i = 0; i < nf; i++) {
    let acc = 0;
    for (let ii = 0; ii < nang; ii++) {
      let w: number;
      if (ii === 0) w = Math.abs(ang[1] - ang[0]);
      else if (ii === nang - 1) w = Math.abs(ang[ii] - ang[ii - 1]);
      else w = Math.abs(ang[ii + 1] - ang[ii - 1]) / 2;
      acc += S[i * nang + ii] * w;
    }
    Sf[i] = acc;
  }
  return Sf;
}

/**
 * Build the internal input spectrum from a JONSWAP parameter set or a 2D
 * spectrum table (buildSpecin dispatch; h0 is the boundary-average depth,
 * used only by the TMA option; g defaults to 9.81 — pass par%g).
 */
export function buildSpecin(input: SpectrumInput, h0: number, g = 9.81): SpecIn {
  return input.kind === "jons" ? buildJons(input, h0, g) : buildTable(input.table);
}

/**
 * JONSWAP branch of read_jonswap_file (waveparamsnew.F90:585-696), single
 * partition (nmodal=1). Frequency grid f(i)=i*dfj with
 * nf = ceiling((fnyq-dfj)/dfj) (:589-596); 401 directions 0..2pi (:604-611);
 * Hm0 imposed by DISCRETE sum, not trapezoid (:641); half-angle cosine-power
 * spreading with nint(scoeff) (:655-666), discretely normalized (:669).
 */
function buildJons(
  p: Extract<SpectrumInput, { kind: "jons" }>,
  h0: number,
  g: number,
): SpecIn {
  const fnyq = p.fnyq ?? Math.max(0.3, 3 * p.fp); // :537
  const dfj = p.dfj ?? fnyq / 200; // :538
  const nf = Math.ceil((fnyq - dfj) / dfj); // :589
  const f = new Float64Array(nf);
  for (let i = 0; i < nf; i++) f[i] = (i + 1) * dfj; // :594-596
  const dang = (2 * Math.PI) / (NAINT - 1); // :610
  const ang = new Float64Array(NAINT);
  for (let i = 0; i < NAINT; i++) ang[i] = i * dang; // :611-613

  // unscaled shape on the relative frequency axis (:618-622)
  const x = new Float64Array(nf);
  for (let i = 0; i < nf; i++) x[i] = f[i] / p.fp;
  const y = jonswapgk(x, p.gammajsp);

  // optional TMA factor (:625-640)
  if (p.tma) {
    for (let ii = 0; ii < nf; ii++) {
      const L0 = (g * (1 / f[ii]) ** 2) / 2 / Math.PI;
      let L = iterateDispersion(L0, L0, h0);
      if (L < 0) L = -L; // dead branch, kept for fidelity
      const k = (2 * Math.PI) / L;
      const kh = k * h0;
      const n = 0.5 * (1 + kh * ((1 - Math.tanh(kh) ** 2) / Math.tanh(kh)));
      y[ii] *= (1 / (2 * n)) * Math.tanh(kh) ** 2;
    }
  }

  // impose Hm0 by discrete sum (:641)
  let ysum = 0;
  for (let i = 0; i < nf; i++) ysum += y[i];
  const yscale = (p.hm0 / (4 * Math.sqrt(ysum * dfj))) ** 2;
  for (let i = 0; i < nf; i++) y[i] *= yscale;

  // nautical deg -> Cartesian rad, wrapped to [0,2pi] (:644-652)
  let mainang = 1.5 * Math.PI - (p.mainangDeg * Math.PI) / 180;
  while (mainang > 2 * Math.PI || mainang < 0) {
    if (mainang > 2 * Math.PI) mainang -= 2 * Math.PI;
    else mainang += 2 * Math.PI;
  }

  // half-angle offsets, wrapped elementwise to [0,2pi] (:655-663)
  const tempdir = new Float64Array(NAINT);
  for (let i = 0; i < NAINT; i++) tempdir[i] = (ang[i] - mainang) / 2;
  let unwrapped = true;
  while (unwrapped) {
    unwrapped = false;
    for (let i = 0; i < NAINT; i++) {
      if (tempdir[i] > 2 * Math.PI) {
        tempdir[i] -= 2 * Math.PI;
        unwrapped = true;
      } else if (tempdir[i] < 0) {
        tempdir[i] += 2 * Math.PI;
        unwrapped = true;
      }
    }
  }

  // cosine-power spreading with integer exponent 2*nint(s) (:666), normalized
  // by its own discrete surface (:669)
  const Dd = new Float64Array(NAINT);
  const expo = 2 * Math.round(p.s);
  let dsum = 0;
  for (let i = 0; i < NAINT; i++) {
    Dd[i] = Math.pow(Math.cos(tempdir[i]), expo);
    dsum += Dd[i];
  }
  for (let i = 0; i < NAINT; i++) Dd[i] /= dsum * dang;

  // 2D variance density (:672-677)
  const S = new Float64Array(nf * NAINT);
  for (let i = 0; i < nf; i++) {
    for (let ii = 0; ii < NAINT; ii++) S[i * NAINT + ii] = y[i] * Dd[ii];
  }

  return {
    f,
    ang,
    S,
    Sf: integrateSf(f, ang, S),
    scoeff: p.s, // :693
    hm0: p.hm0, // :690
    fp: p.fp,
    dir0: mainang, // :692 (radians Cartesian)
  };
}

/**
 * Table branch, following the SWAN NDIR conventions of read_swan_file:
 * nautical coming-from deg -> Cartesian deg via 270-theta (:1030), flipped
 * ascending if needed together with the S columns (:1039-1042, flipv/flipa),
 * deg -> rad (:1045), per-degree -> per-radian S x 180/pi (:1122).
 * scoeff/hm0 set to -1 = "not imposed" (:1126-1128). Angles are NOT wrapped
 * into [0,2pi) — the cyclic direction interpolation handles that, exactly as
 * in Fortran.
 */
function buildTable(t: SpectrumTable): SpecIn {
  const nf = t.freq.length, nang = t.dirDeg.length;
  if (t.efth.length !== nf * nang) {
    throw new Error("SpectrumTable: efth length must be freq.length*dirDeg.length");
  }
  const f = Float64Array.from(t.freq);
  let angDeg = Array.from(t.dirDeg, (d) => 270 - d);
  let S = Float64Array.from(t.efth);
  if (nang >= 2 && angDeg[1] < angDeg[0]) {
    angDeg = angDeg.reverse();
    const Sf2 = new Float64Array(nf * nang);
    for (let i = 0; i < nf; i++) {
      for (let ii = 0; ii < nang; ii++) {
        Sf2[i * nang + ii] = S[i * nang + (nang - 1 - ii)];
      }
    }
    S = Sf2;
  }
  const ang = new Float64Array(nang);
  for (let i = 0; i < nang; i++) ang[i] = (angDeg[i] * Math.PI) / 180;
  for (let i = 0; i < S.length; i++) S[i] *= 180 / Math.PI;
  return {
    f,
    ang,
    S,
    Sf: integrateSf(f, ang, S),
    scoeff: -1,
    hm0: -1,
    fp: 0, // unused on the table path (only read when scoeff>1000)
    dir0: 0,
  };
}

// ----------------------------------------------------- standard-grid interp

/**
 * interpolate_spectrum port (waveparamsnew.F90:1263-1385). Target grid:
 * f(i)=(i-1)*df with df = fmax/(nfint-1) (:1292-1296) and
 * ang(i)=(i-1)*2pi/(naint-1) (:1297-1301); fmax = max(1 Hz, max input f)
 * belongs to the caller (:165,182). Direction rows are interpolated
 * cyclically (:1319-1327), frequencies linearly with zero outside the input
 * range (:1330-1338); the Sf-conservation pass (:1350-1369) — not correctHm0
 * — is what conserves Hm0/Sf, including the argmax-dump branch for all-zero
 * rows with nonzero input Sf.
 */
export function interpolateSpectrum(
  specin: SpecIn,
  fmax: number,
  trepfac = 0.01,
  tm01 = false,
): SpecInterp {
  const nfin = specin.f.length;
  const nangin = specin.ang.length;
  const df = fmax / (NFINT - 1);
  const dang = (2 * Math.PI) / (NAINT - 1);
  const f = new Float64Array(NFINT);
  for (let i = 0; i < NFINT; i++) f[i] = i * df;
  const ang = new Float64Array(NAINT);
  for (let i = 0; i < NAINT; i++) ang[i] = i * dang;

  // pre-interpolation hm0 (:1304-1317): imposed value if set, else forward-
  // difference double sum on the (possibly non-monotonic) input grid
  let hm0pre: number;
  if (specin.hm0 > 0) hm0pre = specin.hm0;
  else {
    let m0 = 0;
    for (let j = 0; j < nangin; j++) {
      for (let i = 0; i < nfin; i++) {
        const dfi = specin.f[Math.max(1, i)] - specin.f[Math.max(0, i - 1)];
        const dai = specin.ang[Math.max(1, j)] - specin.ang[Math.max(0, j - 1)];
        m0 += specin.S[i * nangin + j] * dfi * dai;
      }
    }
    hm0pre = 4 * Math.sqrt(m0);
  }

  // cyclic direction interpolation per input frequency row (:1319-1327)
  const Stemp = new Float64Array(nfin * NAINT);
  const rowIn = new Float64Array(nangin);
  for (let i = 0; i < nfin; i++) {
    for (let ii = 0; ii < nangin; ii++) rowIn[ii] = specin.S[i * nangin + ii];
    Stemp.set(interpCyclic(specin.ang, rowIn, 2 * Math.PI, ang), i * NAINT);
  }

  // frequency interpolation, zero outside the input range (:1330-1338)
  const S = new Float64Array(NFINT * NAINT);
  const col = new Float64Array(nfin);
  const fLo = specin.f[0], fHi = specin.f[nfin - 1];
  for (let j = 0; j < NAINT; j++) {
    for (let i = 0; i < nfin; i++) col[i] = Stemp[i * NAINT + j];
    for (let i = 0; i < NFINT; i++) {
      if (f[i] > fHi || f[i] < fLo) continue; // stays 0
      S[i * NAINT + j] = linearInterp(specin.f, col, f[i]);
    }
  }

  // 1D spectrum (:1348)
  const Sf = new Float64Array(NFINT);
  for (let i = 0; i < NFINT; i++) {
    let acc = 0;
    for (let j = 0; j < NAINT; j++) acc += S[i * NAINT + j];
    Sf[i] = acc * dang;
  }

  // Sf-conservation pass (:1350-1369)
  let fminIn = Infinity, fmaxIn = -Infinity;
  for (let i = 0; i < nfin; i++) {
    if (specin.f[i] < fminIn) fminIn = specin.f[i];
    if (specin.f[i] > fmaxIn) fmaxIn = specin.f[i];
  }
  for (let i = 0; i < NFINT; i++) {
    if (f[i] < fminIn || f[i] > fmaxIn) continue;
    const Sfnow = linearInterp(specin.f, specin.Sf, f[i]);
    if (Sf[i] > 0 && Sfnow > 0) {
      const factor = Sfnow / Sf[i];
      Sf[i] *= factor;
      for (let j = 0; j < NAINT; j++) S[i * NAINT + j] *= factor;
    } else if (Sf[i] === 0 && Sfnow >= 0) {
      Sf[i] = Sfnow;
      const d = imaxFirst(S, i * NAINT, (i + 1) * NAINT) - i * NAINT;
      S[i * NAINT + d] = Sfnow / dang;
    } else {
      Sf[i] = 0;
      for (let j = 0; j < NAINT; j++) S[i * NAINT + j] = 0;
    }
  }

  // statistics from the interpolated spectrum (:1371-1383)
  let m0 = 0;
  for (let i = 0; i < NFINT; i++) m0 += Sf[i];
  m0 *= df;
  const hm0 = 4 * Math.sqrt(m0);
  const Sd = new Float64Array(NAINT);
  for (let j = 0; j < NAINT; j++) {
    let acc = 0;
    for (let i = 0; i < NFINT; i++) acc += S[i * NAINT + j];
    Sd[j] = acc * df;
  }
  const dir0naut = 270 - (ang[imaxFirst(Sd)] * 180) / Math.PI;
  const fp = f[imaxFirst(Sf)];
  const trep = tpDcalc(Sf, f, trepfac, tm01);
  let sSum = 0, cSum = 0, sdSum = 0;
  for (let j = 0; j < NAINT; j++) {
    sSum += Math.sin(ang[j]) * Sd[j];
    cSum += Math.cos(ang[j]) * Sd[j];
    sdSum += Sd[j];
  }
  let dirm = 270 - (180 / Math.PI) * Math.atan2(sSum / sdSum, cSum / sdSum);
  dirm = dirm % 360; // Fortran mod keeps the dividend sign, like JS %

  // hm0pre is only used for the (commented-out) global rescale in Fortran;
  // returned hm0 is the post-conservation value, as in :1372-1373.
  void hm0pre;
  return { f, df, ang, dang, S, Sf, Sd, hm0, fp, dir0naut, trep, dirm };
}

/**
 * tpDcalc port (waveparamsnew.F90:1728-1753): representative period from the
 * bins with Sf >= trepfac*max(Sf); tm01 true -> Tm01 = sum(Sf)/sum(Sf*f),
 * else (default) Tm-1,0 = sum(Sf/max(f,0.001))/sum(Sf).
 */
export function tpDcalc(
  Sf: ArrayLike<number>,
  f: ArrayLike<number>,
  trepfac: number,
  tm01: boolean,
): number {
  let maxSf = -Infinity;
  for (let i = 0; i < Sf.length; i++) if (Sf[i] > maxSf) maxSf = Sf[i];
  const thr = trepfac * maxSf;
  let num = 0, den = 0;
  for (let i = 0; i < Sf.length; i++) {
    if (Sf[i] >= thr) {
      if (tm01) {
        num += Sf[i];
        den += Sf[i] * f[i];
      } else {
        num += Sf[i] / Math.max(f[i], 0.001);
        den += Sf[i];
      }
    }
  }
  return num / den;
}

// ------------------------------------------------------------ combined spectrum

/**
 * generate_combined_spectrum port (waveparamsnew.F90:1441-1518): plain
 * average of S and Sf over locations; trep recomputed from the averaged Sf
 * via tpDcalc; dir0 = Sd-weighted mean angle over the peak bin +-1 neighbour
 * with cyclic closure (:1479-1501), radians Cartesian; fp = Sf-weighted mean
 * frequency of the >95%-of-peak band (:1505-1512); hm0 = 4*sqrt(sum(Sf)*df).
 */
export function combineSpectra(
  specs: SpecInterp[],
  trepfac = 0.01,
  tm01 = false,
): CombSpec {
  const n = specs.length;
  if (n === 0) throw new Error("combineSpectra: no spectra");
  const ref = specs[0];
  const S = new Float64Array(NFINT * NAINT);
  const Sf = new Float64Array(NFINT);
  for (const sp of specs) {
    for (let k = 0; k < S.length; k++) S[k] += sp.S[k] / n;
    for (let i = 0; i < NFINT; i++) Sf[i] += sp.Sf[i] / n;
  }
  const trep = tpDcalc(Sf, ref.f, trepfac, tm01);

  const Sd = new Float64Array(NAINT);
  for (let j = 0; j < NAINT; j++) {
    let acc = 0;
    for (let i = 0; i < NFINT; i++) acc += S[i * NAINT + j];
    Sd[j] = acc * ref.df;
  }
  const iloc = imaxFirst(Sd); // first occurrence on ties (:1481)
  const ang = ref.ang;
  let peakSd: number[], peakang: number[];
  if (iloc > 0 && iloc < NAINT - 1) {
    peakSd = [Sd[iloc - 1], Sd[iloc], Sd[iloc + 1]];
    peakang = [ang[iloc - 1], ang[iloc], ang[iloc + 1]];
  } else if (iloc === 0) {
    peakSd = [Sd[NAINT - 1], Sd[0], Sd[1]];
    peakang = [ang[NAINT - 1] - 2 * Math.PI, ang[0], ang[1]];
  } else {
    peakSd = [Sd[NAINT - 2], Sd[NAINT - 1], Sd[0]];
    peakang = [ang[NAINT - 2], ang[NAINT - 1], ang[0] + 2 * Math.PI];
  }
  let dir0 = (peakSd[0] * peakang[0] + peakSd[1] * peakang[1] + peakSd[2] * peakang[2]) /
    (peakSd[0] + peakSd[1] + peakSd[2]);
  if (dir0 > 2 * Math.PI) dir0 -= 2 * Math.PI;
  else if (dir0 < 0) dir0 += 2 * Math.PI;

  // smoothed peak frequency (:1505-1512)
  let maxSf = -Infinity;
  for (let i = 0; i < NFINT; i++) if (Sf[i] > maxSf) maxSf = Sf[i];
  let fnum = 0, fden = 0;
  for (let i = 0; i < NFINT; i++) {
    if (Sf[i] > 0.95 * maxSf) {
      fnum += Sf[i] * ref.f[i];
      fden += Sf[i];
    }
  }
  const fp = fnum / fden;

  let m0 = 0;
  for (let i = 0; i < NFINT; i++) m0 += Sf[i];
  m0 *= ref.df;
  return {
    f: ref.f,
    df: ref.df,
    ang,
    dang: ref.dang,
    S,
    Sf,
    Sd,
    trep,
    dir0,
    fp,
    hm0: 4 * Math.sqrt(m0),
  };
}
