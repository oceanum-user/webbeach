// SPDX-License-Identifier: LGPL-3.0-or-later
// XBeach WebGPU solver — derived from XBeach (c) Deltares (LGPL v2.1+).
// Copyright (c) 2026 Oceanum. See LICENSE.
/**
 * Tests for the spectrum chain (trunk/src/xbeachlibrary/waveparamsnew.F90:
 * read_jonswap_file :585-696, jonswapgk :889-930, interpolate_spectrum
 * :1263-1385, generate_combined_spectrum :1441-1518, tpDcalc :1728-1753).
 * The whole-chain anchor compares against the Fortran-written ebcflist.bcf
 * line of verification/cases/surfbeat1d_jons (printed precision).
 */
import { assert, assertAlmostEquals, assertEquals } from "@std/assert";
import {
  buildSpecin,
  combineSpectra,
  interpolateSpectrum,
  jonswapgk,
  linearInterp,
  NAINT,
  type SpecIn,
  tpDcalc,
} from "../src/spectrum.ts";

// the fixture's jonswap.txt (verification/cases/surfbeat1d_jons)
const FIXTURE_JONS = {
  kind: "jons" as const,
  hm0: 1.0,
  fp: 0.125,
  gammajsp: 3.3,
  s: 20,
  mainangDeg: 270,
  fnyq: 0.5,
};

Deno.test("jonswapgk closed-form values and the sigma switch at x=1", () => {
  const gam = 3.3;
  const x = [0.5, 0.9, 1.0, 1.1, 1.5, 2.0];
  const y = jonswapgk(x, gam);
  // independent transcription of waveparamsnew.F90:906-928
  const raw = x.map((xi) => {
    const sigma = xi < 1 ? 0.07 : 0.09;
    return Math.pow(xi, -5) * Math.exp(-1.25 * Math.pow(xi, -4)) *
      Math.pow(gam, Math.exp(-((xi - 1) ** 2) / (2 * sigma * sigma)));
  });
  const ymax = Math.max(...raw);
  for (let i = 0; i < x.length; i++) {
    assertAlmostEquals(y[i], raw[i] / ymax, 1e-15, `x=${x[i]}`);
  }
  // peak is at x=1 for this range and normalization makes it 1
  assertEquals(y[2], 1);
  // sigma switch: x=0.9 uses 0.07 (narrower), x=1.1 uses 0.09 (broader), so
  // the shape is asymmetric around the peak
  assert(y[3] > y[1], "gamma bump must be wider on the x>1 side");
});

Deno.test("JONSWAP spreading uses the integer exponent 2*nint(s)", () => {
  // waveparamsnew.F90:666: Dd = cos(...)**(2*nint(scoeff))
  const a = buildSpecin({ ...FIXTURE_JONS, s: 10.4 }, 8);
  const b = buildSpecin({ ...FIXTURE_JONS, s: 10.0 }, 8);
  const c = buildSpecin({ ...FIXTURE_JONS, s: 10.6 }, 8);
  assertEquals(a.S, b.S, "nint(10.4) == 10, spectra identical");
  let differs = false;
  for (let i = 0; i < a.S.length; i++) {
    if (a.S[i] !== c.S[i]) {
      differs = true;
      break;
    }
  }
  assert(differs, "nint(10.6) == 11 must change the spreading");
});

Deno.test("JONSWAP Hm0 is imposed by discrete sum over the frequency grid", () => {
  // waveparamsnew.F90:641: y = (Hm0/(4 sqrt(sum(y) dfj)))^2 y. With the
  // equidistant direction grid every integration weight in the Sf rule is
  // dang, and Dd is normalized by sum(Dd)*dang, so Sf(i) == y(i) exactly and
  // 4*sqrt(sum(Sf)*dfj) recovers Hm0 to machine precision.
  const specin = buildSpecin(FIXTURE_JONS, 8);
  const dfj = FIXTURE_JONS.fnyq / 200;
  let s = 0;
  for (const v of specin.Sf) s += v;
  assertAlmostEquals(4 * Math.sqrt(s * dfj), FIXTURE_JONS.hm0, 1e-12);
  // frequency grid f(i) = i*dfj, nf = ceiling((fnyq-dfj)/dfj)
  assertEquals(specin.f.length, Math.ceil((FIXTURE_JONS.fnyq - dfj) / dfj));
  assertAlmostEquals(specin.f[0], dfj, 1e-15);
});

Deno.test("table input follows the SWAN NDIR conventions", () => {
  // read_swan_file: ang = 270-naut, flip to ascending with the S columns,
  // per-degree -> per-radian (x 180/pi); scoeff/hm0 = -1 (:1027-1128)
  const specin = buildSpecin({
    kind: "table",
    table: {
      freq: Float64Array.from([0.1]),
      dirDeg: Float64Array.from([260, 270, 280]), // naut -> cart [10, 0, -10]: descending
      efth: Float64Array.from([1, 2, 3]),
    },
  }, 8);
  const d2r = Math.PI / 180;
  assertAlmostEquals(specin.ang[0], -10 * d2r, 1e-15);
  assertAlmostEquals(specin.ang[1], 0, 1e-15);
  assertAlmostEquals(specin.ang[2], 10 * d2r, 1e-15);
  // S flipped with the angles and scaled by 180/pi
  assertAlmostEquals(specin.S[0], 3 * (180 / Math.PI), 1e-12);
  assertAlmostEquals(specin.S[1], 2 * (180 / Math.PI), 1e-12);
  assertAlmostEquals(specin.S[2], 1 * (180 / Math.PI), 1e-12);
  assertEquals(specin.scoeff, -1);
  assertEquals(specin.hm0, -1);
});

Deno.test("interpolation conserves Sf per frequency bin (waveparamsnew.F90:1350-1369)", () => {
  const specin = buildSpecin(FIXTURE_JONS, 8);
  let fmax = 1;
  for (const f of specin.f) fmax = Math.max(fmax, f);
  const si = interpolateSpectrum(specin, fmax);
  const fLo = specin.f[0], fHi = specin.f[specin.f.length - 1];
  let checked = 0;
  for (let i = 0; i < si.f.length; i++) {
    if (si.f[i] <= fLo || si.f[i] >= fHi) continue; // strictly inside
    const want = linearInterp(specin.f, specin.Sf, si.f[i]);
    const tol = Math.max(1e-18, 1e-12 * Math.abs(want));
    assert(Math.abs(si.Sf[i] - want) <= tol, `bin ${i}: ${si.Sf[i]} vs ${want}`);
    checked++;
  }
  assert(checked > 100, "conservation pass must cover the input range");
  // hm0 is preserved through the resampling to well under a percent
  assertAlmostEquals(si.hm0, FIXTURE_JONS.hm0, 1e-4);
});

Deno.test("all-zero direction rows with nonzero input Sf dump into the argmax bin", () => {
  // waveparamsnew.F90:1360-1363: Sf(i)==0 and Sfnow>=0 -> Sf(i)=Sfnow and
  // S(i, maxloc) = Sfnow/dang. Crafted input: S identically zero but Sf>0.
  const nangIn = 5;
  const dx = (2 * Math.PI) / 4;
  const specin: SpecIn = {
    f: Float64Array.from([0.1, 0.3]),
    ang: Float64Array.from({ length: nangIn }, (_, i) => i * dx),
    S: new Float64Array(2 * nangIn),
    Sf: Float64Array.from([0.5, 0.5]),
    scoeff: -1,
    hm0: -1,
    fp: 0,
    dir0: 0,
  };
  const si = interpolateSpectrum(specin, 1);
  const dang = (2 * Math.PI) / (NAINT - 1);
  for (let i = 0; i < si.f.length; i++) {
    if (si.f[i] <= 0.1 || si.f[i] >= 0.3) continue;
    assertAlmostEquals(si.Sf[i], 0.5, 1e-12, `Sf bin ${i}`);
    // dumped into the first (argmax of an all-zero row) direction bin
    assertAlmostEquals(si.S[i * NAINT], 0.5 / dang, 1e-9, `S dump bin ${i}`);
    for (let j = 1; j < NAINT; j++) assertEquals(si.S[i * NAINT + j], 0);
  }
});

Deno.test("tpDcalc: both switches, threshold at trepfac*max(Sf)", () => {
  // waveparamsnew.F90:1728-1753 (mask is Sf >= trepfac*max)
  const Sf = [0, 1, 2, 1];
  const f = [0.05, 0.1, 0.15, 0.2];
  // trepfac=0.5 -> threshold 1 -> bins 1,2,3 kept
  assertAlmostEquals(tpDcalc(Sf, f, 0.5, true), 4 / (1 * 0.1 + 2 * 0.15 + 1 * 0.2), 1e-15);
  assertAlmostEquals(tpDcalc(Sf, f, 0.5, false), (1 / 0.1 + 2 / 0.15 + 1 / 0.2) / 4, 1e-15);
  // f=0 bins are guarded by max(f, 0.001) in the Tm-1,0 branch
  const t = tpDcalc([1, 1], [0, 0.5], 0, false);
  assertAlmostEquals(t, (1 / 0.001 + 1 / 0.5) / 2, 1e-12);
});

Deno.test("whole-chain anchor: fixture jonswap.txt reproduces the ebcflist.bcf line", () => {
  // verification/cases/surfbeat1d_jons/ebcflist.bcf (Fortran-written):
  //   1200.000  1200.000  1.000  7.31773  0.005  0.70710  E_reuse.bcf
  // Trep/Hrms at f9.5/f11.5 print precision, maindir (rad) at f9.3.
  const specin = buildSpecin(FIXTURE_JONS, 8);
  let fmax = 1;
  for (const f of specin.f) fmax = Math.max(fmax, f);
  const si = interpolateSpectrum(specin, fmax, 0.01, false);
  const comb = combineSpectra([si], 0.01, false);
  assertAlmostEquals(comb.trep, 7.31773, 0.5e-5, "Trep");
  assertAlmostEquals(comb.hm0 / Math.sqrt(2), 0.70710, 0.5e-5, "Hrms");
  assertAlmostEquals(comb.dir0, 0.005, 0.5e-3, "theta0 (rad, Cartesian)");
});

Deno.test("combineSpectra averages locations and recomputes stats", () => {
  const specin = buildSpecin(FIXTURE_JONS, 8);
  let fmax = 1;
  for (const f of specin.f) fmax = Math.max(fmax, f);
  const si = interpolateSpectrum(specin, fmax);
  const comb1 = combineSpectra([si]);
  const comb2 = combineSpectra([si, si]); // averaging identical spectra is a no-op
  assertAlmostEquals(comb2.hm0, comb1.hm0, 1e-12);
  assertAlmostEquals(comb2.trep, comb1.trep, 1e-12);
  assertAlmostEquals(comb2.dir0, comb1.dir0, 1e-12);
  // fp is the Sf-weighted mean of the >95%-of-peak band (:1505-1512)
  assert(comb1.fp > 0.115 && comb1.fp < 0.135, `fp=${comb1.fp}`);
});
