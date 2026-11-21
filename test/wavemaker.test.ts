// SPDX-License-Identifier: LGPL-3.0-or-later
// XBeach WebGPU solver — derived from XBeach (c) Deltares (LGPL v2.1+).
// Copyright (c) 2026 Oceanum. See LICENSE.
/**
 * Tests for the spectral wavemaker (trunk/src/xbeachlibrary/waveparamsnew.F90
 * spectral_wave_bc pipeline, wbcversion=3 surfbeat path). Pure CPU — no GPU.
 *
 * The end-to-end cases run against verification/cases/surfbeat1d_jons: exact
 * scalar anchors from the Fortran-written ebcflist.bcf line, plus Tier-2
 * statistical invariants against the shipped (random=1) E_reuse/q_reuse
 * tables. Bit-level table parity (Tier 1) needs the random=0 Fortran fixture
 * and lives in the verification harness, not here.
 */
import { assert, assertAlmostEquals, assertEquals } from "@std/assert";
import { parseParamsTxt, resolveParams } from "../src/params.ts";
import { buildGrid, buildXY } from "../src/grid.ts";
import {
  assignDirectionBins,
  boundPair,
  frange,
  makeSpectralConditions,
  makeWaveTimeAxis,
  type SpectralCondition,
  type WavemakerDiag,
} from "../src/wavemaker.ts";
import { iterateDispersion } from "../src/spectrum.ts";

const CASE = new URL("../verification/cases/surfbeat1d_jons/", import.meta.url);
const FIXTURE_JONS = {
  kind: "jons" as const,
  hm0: 1.0,
  fp: 0.125,
  gammajsp: 3.3,
  s: 20,
  mainangDeg: 270,
  fnyq: 0.5,
};

async function fixtureSetup(): Promise<{
  p: ReturnType<typeof resolveParams>;
  grid: ReturnType<typeof buildGrid>;
}> {
  const p = resolveParams(parseParamsTxt(await Deno.readTextFile(new URL("params.txt", CASE))));
  p.random = 0; // deterministic draws (seed 100, waveparamsnew.F90:1618)
  const { x, y } = buildXY(p);
  // Constant depth for self-consistent tests only. NOT the case's effective
  // boundary depth: XBeach copies zb row 2 onto row 1 at init
  // (initialize.F90:248, mirrored in grid.ts), so the real surfbeat1d_jons
  // boundary depth is bed.dep column 2 = 7.9644 m, not column 1 = 8.0 m.
  // Any Tier-1 q-table comparison against Fortran output must use bed.dep
  // (a constant-8 fixture fails q parity by a uniform ~0.6 %).
  const zbIn = new Float64Array((p.nx + 1) * (p.ny + 1)).fill(8.0);
  return { p, grid: buildGrid(p, x, y, zbIn) };
}

function smallParams(extra: string): ReturnType<typeof resolveParams> {
  return resolveParams(parseParamsTxt(`
wavemodel = surfbeat
depfile = bed.dep
posdwn = 1
nx = 20
ny = 2
vardx = 0
dx = 2
dy = 2
xori = 0
yori = 0
alfa = 0
thetamin = -80
thetamax = 80
dtheta = 10
thetanaut = 0
single_dir = 0
wbctype = parametric
bcfile = jonswap.txt
random = 0
tstop = 300
zs0 = 0
front = abs_1d
back = abs_1d
${extra}
`));
}

// ------------------------------------------------------------------- frange

Deno.test("frange keeps bins with Sf > sprdthr*max and f <= fmax", () => {
  // waveparamsnew.F90:1759-1799 (strict >; first/last kept index)
  const f = Float64Array.from([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
  const Sf = Float64Array.from([0, 0.05, 1, 0.5, 0.04, 0]);
  const r1 = frange(f, Sf, 0.08, 10);
  assertEquals(r1, { ind1: 2, ind2: 3 });
  // exact threshold is excluded (strict >)
  const r2 = frange(f, Sf, 0.05, 10);
  assertEquals(r2, { ind1: 2, ind2: 3 });
  // fmax trims the tail
  const r3 = frange(f, Sf, 0.03, 0.35);
  assertEquals(r3, { ind1: 1, ind2: 2 });
});

// ---------------------------------------------------------------- time axis

Deno.test("wave time axis and tapers match the Fortran construction", () => {
  // generate_wave_time_axis (waveparamsnew.F90:1846-1911), hand-computed:
  // rtbc=10, dtbc=1, fgen(K)=0.4 (no dt change), Trep=0.6 -> ntaper=3,
  // dfgen=1/12 -> tslen=13 -> even 14, indend = first tin>=10 -> 11 (1-based)
  const ta = makeWaveTimeAxis(10, 1, 0.4, 0.6, 1 / 12);
  assertEquals(ta.tslenbc, 11);
  assertEquals(ta.dtchanged, false);
  assertEquals(ta.dtin, 1);
  assertEquals(ta.tslen, 14);
  assertAlmostEquals(ta.rtin, 14, 1e-15);
  assertEquals(ta.indEndTaper, 11);
  const t53 = Math.tanh(5 / 3), t103 = Math.tanh(10 / 3), t5 = Math.tanh(5);
  const wantW = [t53, t103, t5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
  const wantF = [t53, t103, t5, 1, 1, 1, 1, 1, t5, t103, 0, 0, 0, 0];
  for (let i = 0; i < 14; i++) {
    assertEquals(ta.taperw[i], wantW[i], `taperw[${i}]`);
    assertEquals(ta.taperf[i], wantF[i], `taperf[${i}]`);
  }
});

Deno.test("internal step drops to 0.5/fgen(K) when dtbc is too coarse", () => {
  // waveparamsnew.F90:1855-1859 (surfbeat branch)
  const ta = makeWaveTimeAxis(10, 1, 0.6, 0.6, 1 / 12);
  assertEquals(ta.dtchanged, true);
  assertAlmostEquals(ta.dtin, 0.5 / 0.6, 1e-15);
  assertEquals(ta.tslen, 16); // ceil(12/0.8333)+1 = 15 -> even 16
});

// --------------------------------------------------------------- dispersion

Deno.test("golden-ratio dispersion matches an independent replica bit-for-bit", () => {
  // wave_functions.F90:800-837 transcribed independently
  const replica = (L0: number, Lest: number, h: number): number => {
    const phi = (1 + Math.sqrt(5)) / 2;
    const aphi = 1 / (phi + 1), bphi = phi / (phi + 1);
    let L1 = Lest, err = Infinity, iter = 0;
    while (err > 0.00001 && iter < 150) {
      iter++;
      const L2 = L0 * Math.tanh((2 * Math.PI * h) / L1);
      L1 = L1 * aphi + L2 * bphi;
      err = Math.abs(L2 - L1);
    }
    return L1;
  };
  for (const f of [0.05, 0.09375, 0.125, 0.20125, 0.4]) {
    for (const h of [2, 8, 25]) {
      const L0 = (9.81 * (1 / f) ** 2) / 2 / Math.PI;
      assertEquals(iterateDispersion(L0, L0, h), replica(L0, L0, h), `f=${f} h=${h}`);
    }
  }
});

Deno.test("dispersion stops at the 1e-5 tolerance (early termination is load-bearing)", () => {
  // kgen parity depends on NOT converging further than Fortran does
  const f = 0.09375, h = 8;
  const L0 = (9.81 * (1 / f) ** 2) / 2 / Math.PI;
  const L = iterateDispersion(L0, L0, h);
  // fully-converged fixed point
  const phi = (1 + Math.sqrt(5)) / 2;
  const aphi = 1 / (phi + 1), bphi = phi / (phi + 1);
  let Linf = L0;
  for (let i = 0; i < 10000; i++) {
    const L2 = L0 * Math.tanh((2 * Math.PI * h) / Linf);
    const next = Linf * aphi + L2 * bphi;
    if (Math.abs(next - Linf) < 1e-13) break;
    Linf = next;
  }
  const diff = Math.abs(L - Linf);
  assert(diff > 1e-9, `converged too far (diff=${diff})`);
  assert(diff < 1e-2, `not converged enough (diff=${diff})`);
  // still physically consistent: g k tanh(kh) ~ omega^2
  const k = (2 * Math.PI) / L;
  const w2 = (2 * Math.PI * f) ** 2;
  assertAlmostEquals(9.81 * k * Math.tanh(k * h), w2, 1e-3 * w2);
});

// ------------------------------------------------------- direction binning

Deno.test("direction binning: edges, wrap by 2pi, top-edge catch, out-of-range", () => {
  // distribute_wave_train_directions (waveparamsnew.F90:2213-2262)
  const d2r = Math.PI / 180;
  const thetamin = -80 * d2r, dtheta = 10 * d2r, ntheta = 16;
  const theta = Float64Array.from(
    { length: ntheta },
    (_, i) => (thetamin + dtheta / 2 + i * dtheta) % (2 * Math.PI),
  );
  const emax = thetamin + ntheta * dtheta; // same float expression as the impl
  const gen = Float64Array.from([
    5 * d2r, // centre of 1-based bin 9
    emax, // exact top edge -> top bin (:2247-2249)
    -75 * d2r, // bin 1
    100 * d2r, // above range, wrap sends it below -> WDindex 0
    -100 * d2r, // below range, wrap sends it above -> WDindex ntheta+1
    350 * d2r, // wraps to -10 deg -> bin 8
  ]);
  const wd = assignDirectionBins(gen, thetamin, dtheta, ntheta, theta, 0);
  assertEquals(Array.from(wd), [9, 16, 1, 0, 17, 8]);
  // the wrap mutates thetagen in place (used later by generate_qbcf)
  assertAlmostEquals(gen[5], -10 * d2r, 1e-12);

  // nspr=1 snaps to bin centres and clamps into [1,ntheta] (:2264-2284)
  const gen2 = Float64Array.from([5 * d2r, 100 * d2r, -100 * d2r]);
  const wd2 = assignDirectionBins(gen2, thetamin, dtheta, ntheta, theta, 1);
  assertEquals(Array.from(wd2), [9, 1, 16]);
  assertEquals(gen2[0], theta[8]);
  assertEquals(gen2[1], theta[0]);
  assertEquals(gen2[2], theta[15]);
});

// ------------------------------------------------------------- bound pairs

Deno.test("Okihiro interaction coefficient matches a hand transcription", () => {
  // generate_qbcf (waveparamsnew.F90:2787-2823); independent transcription
  const okihiro = (
    g: number,
    h0: number,
    nmax: number,
    w1: number,
    w2: number,
    k1: number,
    k2: number,
    th1: number,
    th2: number,
    df: number,
  ) => {
    const dth = Math.abs(th2 - th1) + Math.PI;
    const k3 = Math.sqrt(k1 * k1 + k2 * k2 + 2 * k1 * k2 * Math.cos(dth));
    const cg3 = Math.min((2 * Math.PI * df) / k3, nmax * Math.sqrt((g / k3) * Math.tanh(k3 * h0)));
    const t1 = -w1 * w2, t2 = -w1 + w2, t2n = cg3 * k3;
    const ch1 = Math.cosh(k1 * h0), ch2 = Math.cosh(k2 * h0);
    const D = (-g * k1 * k2 * Math.cos(dth)) / 2 / t1 + (t2 * t2) / (2 * g) +
      ((g * t2) / ((g * k3 * Math.tanh(k3 * h0) - t2n * t2n) * t1)) *
        (t2 * ((t1 * t1) / g / g - k1 * k2 * Math.cos(dth)) -
          0.5 * ((-w1 * k2 * k2) / (ch2 * ch2) + (w2 * k1 * k1) / (ch1 * ch1)));
    return { k3, cg3, D };
  };
  const g = 9.81, h0 = 8, nmax = 0.8;
  // ordinary pair: limiter inactive
  {
    const w1 = 2 * Math.PI * 0.1, w2 = 2 * Math.PI * 0.105;
    const bp = boundPair(g, h0, nmax, w1, w2, 0.08, 0.0845, 0.1, -0.15, 0.005);
    const want = okihiro(g, h0, nmax, w1, w2, 0.08, 0.0845, 0.1, -0.15, 0.005);
    assertAlmostEquals(bp.k3, want.k3, 1e-15);
    assertAlmostEquals(bp.cg3, want.cg3, 1e-15);
    assertAlmostEquals(bp.D, want.D, Math.abs(want.D) * 1e-12);
    assert(bp.cg3 < nmax * Math.sqrt((g / bp.k3) * Math.tanh(bp.k3 * h0)), "limiter inactive");
  }
  // near-collinear pair with tiny k3: cg3 hits the nmax limiter and the
  // limited T2n=cg3*k3 enters the resonance denominator (:2799-2823)
  {
    const w1 = 2 * Math.PI * 0.1, w2 = 2 * Math.PI * 0.105;
    const bp = boundPair(g, h0, nmax, w1, w2, 0.08, 0.0801, 0, 0, 0.005);
    const want = okihiro(g, h0, nmax, w1, w2, 0.08, 0.0801, 0, 0, 0.005);
    const limit = nmax * Math.sqrt((g / bp.k3) * Math.tanh(bp.k3 * h0));
    assertEquals(bp.cg3, limit, "cg3 must be nmax-limited");
    assertAlmostEquals(bp.D, want.D, Math.abs(want.D) * 1e-12);
  }
});

// ------------------------------------------- end-to-end: fixture, ny=0 case

Deno.test("fixture chain: scalar anchors and component bookkeeping", async () => {
  const { p, grid } = await fixtureSetup();
  const diag: WavemakerDiag = {};
  const conds = makeSpectralConditions(FIXTURE_JONS, p, grid, { diag });
  assertEquals(conds.length, 1);
  const c = conds[0];

  // anchors from the Fortran-written ebcflist.bcf (printed precision)
  const eline = (await Deno.readTextFile(new URL("ebcflist.bcf", CASE))).trim().split(/\s+/);
  assertAlmostEquals(c.rt, parseFloat(eline[1]), 1e-9);
  assertAlmostEquals(c.dtbc, parseFloat(eline[2]), 1e-9);
  assertAlmostEquals(c.Trep, parseFloat(eline[3]), 0.5e-5);
  assertAlmostEquals(c.theta0, parseFloat(eline[4]), 0.5e-3);
  assertAlmostEquals(c.Hrms, parseFloat(eline[5]), 0.5e-5);

  // K = max(ceiling(rtbc*(f2-f1))+1, 200) = Kmin here; dfgen = (f2-f1)/(K-1)
  assertEquals(diag.K, 200);
  assertAlmostEquals(diag.fgen![0], 0.09375, 1e-12); // 75/800 Hz (sprdthr=0.08)
  assertAlmostEquals(diag.dfgen!, (0.20125 - 0.09375) / 199, 1e-15);
  // Findex(1) = floor(fgen(1)/dfgen) + 1 (waveparamsnew.F90:2124-2128)
  assertEquals(diag.findex0, 173);
  assertEquals(diag.findex0, Math.floor(diag.fgen![0] / diag.dfgen!));

  // time axis: dtbc=1 < 0.5/fgen(K)=2.48 -> unchanged; tslen even
  const ta = diag.timeAxis!;
  assertEquals(ta.dtchanged, false);
  assertEquals(ta.tslenbc, 1201);
  assertEquals(ta.tslen, 1854);
  assertEquals(ta.tslen % 2, 0);
  assertEquals(ta.indEndTaper, 1201);
  // all component bins stay below Nyquist of the internal axis
  assert(diag.findex0! + diag.K! - 1 < ta.tslen / 2);

  // record shapes: nrec = tslenbc+2, E stride (ny+1)*ntheta, q stride (ny+1)*4
  const m1 = p.ny + 1, ntheta = grid.ntheta;
  assertEquals(c.nrec, ta.tslenbc + 2);
  assertEquals(c.eData.length, c.nrec * m1 * ntheta);
  assertEquals(c.qData.length, c.nrec * m1 * 4);
});

Deno.test("fixture chain: Tier-2 invariants and comparison with the Fortran tables", async () => {
  const { p, grid } = await fixtureSetup();
  const diag: WavemakerDiag = {};
  const c = makeSpectralConditions(FIXTURE_JONS, p, grid, { diag })[0];
  const m1 = p.ny + 1, ntheta = grid.ntheta;
  const recE = m1 * ntheta, recQ = m1 * 4;
  const ta = diag.timeAxis!;

  // variance bookkeeping: sum A^2/2 = sum Sfinterp*dfgen = (Hm0interp/4)^2
  // exactly, by the unconditional (hm0/hm0post)^2 rescale (:1958-1959)
  let a2 = 0;
  for (const a of diag.A!) a2 += (a * a) / 2;
  assertAlmostEquals(a2, (diag.Hm0interp! / 4) ** 2, a2 * 1e-12);

  // no energy dropped outside the directional grid for this narrow sea
  assert(diag.perclost! < 5, `perclost=${diag.perclost}`);

  // energy-weighted mean direction within 2 degrees of theta0 (K=200 draws)
  let dnum = 0, dden = 0;
  for (let i = 0; i < diag.K!; i++) {
    dnum += diag.A![i] ** 2 * diag.thetagen![i];
    dden += diag.A![i] ** 2;
  }
  assert(
    Math.abs(dnum / dden - c.theta0) < (2 * Math.PI) / 180,
    `mean dir ${dnum / dden} vs theta0 ${c.theta0}`,
  );

  // wbcScaleEnergy=1: time-mean directional-integrated energy over the
  // boundary-file window equals rhog8*(Hm0interp/sqrt2)^2 (:2499-2509)
  const nIBF = Math.round(((ta.tslenbc - 1) * p.dtbc) / ta.dtin);
  let esum = 0;
  for (let r = 0; r < nIBF; r++) {
    for (let it = 0; it < ntheta; it++) esum += c.eData[r * recE + it * m1];
  }
  const emean = (esum * grid.dtheta) / nIBF;
  assertAlmostEquals(emean, (diag.Hm0interp! / Math.sqrt(2)) ** 2 * p.rhog8, emean * 1e-9);

  // wbcRemoveStokes=1: q means over the first nint(rtbc/dtin) samples vanish
  const nIBFq = Math.round(p.rt / ta.dtin);
  for (const slot of [0, 1, 2, 3]) {
    let qm = 0;
    for (let r = 0; r < nIBFq; r++) qm += c.qData[r * recQ + slot * m1];
    assert(Math.abs(qm / nIBFq) < 1e-12, `q slot ${slot} mean ${qm / nIBFq}`);
  }

  // IMDB record duplication: the last two records clamp to record tslenbc
  const last = (ta.tslenbc - 1) * recE;
  for (let k = 0; k < recE; k++) {
    assertEquals(c.eData[ta.tslenbc * recE + k], c.eData[last + k]);
    assertEquals(c.eData[(ta.tslenbc + 1) * recE + k], c.eData[last + k]);
  }
  const lastQ = (ta.tslenbc - 1) * recQ;
  for (let k = 0; k < recQ; k++) {
    assertEquals(c.qData[ta.tslenbc * recQ + k], c.qData[lastQ + k]);
    assertEquals(c.qData[(ta.tslenbc + 1) * recQ + k], c.qData[lastQ + k]);
  }

  // --- statistical comparison against the shipped Fortran tables (random=1;
  // our seed-100 realization differs, so only statistics can match)
  const eRef = new Float64Array((await Deno.readFile(new URL("E_reuse.bcf", CASE))).buffer);
  const qRef = new Float64Array((await Deno.readFile(new URL("q_reuse.bcf", CASE))).buffer);
  assertEquals(eRef.length, c.eData.length);
  assertEquals(qRef.length, c.qData.length);

  const nCmp = ta.tslenbc;
  const totGen = new Float64Array(nCmp), totRef = new Float64Array(nCmp);
  for (let r = 0; r < nCmp; r++) {
    let sg = 0, sr = 0;
    for (let it = 0; it < ntheta; it++) {
      sg += c.eData[r * recE + it * m1];
      sr += eRef[r * recE + it * m1];
    }
    totGen[r] = sg;
    totRef[r] = sr;
  }
  // mean energy level: fixed by wbcScaleEnergy in both models
  const mg = totGen.reduce((a, b) => a + b) / nCmp;
  const mr = totRef.reduce((a, b) => a + b) / nCmp;
  assert(Math.abs(mg - mr) / mr < 0.01, `mean E ${mg} vs ${mr}`);

  // band-integrated variance of the energy signal in the IG band (0-0.05 Hz):
  // the wave-group structure both models must share (<= 25% single seed)
  const igVar = (tot: Float64Array): number => {
    const n = tot.length;
    const mean = tot.reduce((a, b) => a + b) / n;
    const kmax = Math.floor(0.05 * n * p.dtbc);
    let v = 0;
    for (let k = 1; k <= kmax; k++) {
      let re = 0, im = 0;
      for (let t = 0; t < n; t++) {
        const w = (2 * Math.PI * k * t) / n;
        re += (tot[t] - mean) * Math.cos(w);
        im -= (tot[t] - mean) * Math.sin(w);
      }
      v += (2 * (re * re + im * im)) / (n * n);
    }
    return v;
  };
  const ratio = igVar(totGen) / igVar(totRef);
  assert(ratio > 0.75 && ratio < 1.25, `IG band variance ratio ${ratio}`);

  // bound-wave flux variance: same order as the Fortran realization. NOTE:
  // per-realization scatter is larger than the 15% the plan estimated (a
  // 10-seed ensemble spans roughly +-20% around its mean, and the Fortran
  // table is itself one clock-seeded draw), so this is a smoke bound; the
  // tight check is Tier-1 table parity against a random=0 Fortran fixture.
  const qvar = (qd: Float64Array): number => {
    let m = 0;
    for (let r = 0; r < nCmp; r++) m += qd[r * recQ];
    m /= nCmp;
    let v = 0;
    for (let r = 0; r < nCmp; r++) v += (qd[r * recQ] - m) ** 2;
    return v / nCmp;
  };
  const qratio = qvar(c.qData) / qvar(qRef);
  assert(qratio > 0.5 && qratio < 2, `qx variance ratio ${qratio}`);
});

// ----------------------------------------- 2D layout and dtchanged branches

Deno.test("2D long-crested sea pins the record layout and alongshore structure", () => {
  const p = smallParams("dtbc = 1");
  const { x, y } = buildXY(p);
  const zbIn = new Float64Array((p.nx + 1) * (p.ny + 1)).fill(8.0);
  const grid = buildGrid(p, x, y, zbIn);
  const diag: WavemakerDiag = {};
  // s=1024 (>1000) triggers the long-crested override thetagen=dir0
  // (waveparamsnew.F90:223-226); 265 degN -> 5 deg Cartesian, centre of the
  // 1-based bin 9 of the -80:10:80 grid
  const c: SpectralCondition = makeSpectralConditions(
    { ...FIXTURE_JONS, s: 1024, mainangDeg: 265 },
    p,
    grid,
    { diag },
  )[0];
  const m1 = p.ny + 1, ntheta = grid.ntheta, recE = m1 * ntheta, recQ = m1 * 4;
  const d2r = Math.PI / 180;
  for (const th of diag.thetagen!) assertAlmostEquals(th, 5 * d2r, 1e-12);
  for (const wd of diag.WDindex!) assertEquals(wd, 9);

  // E records: j fastest then theta — only the theta=8 (0-based) slot may be
  // nonzero, for every record and every j
  for (let r = 0; r < c.nrec; r++) {
    for (let it = 0; it < ntheta; it++) {
      for (let j = 0; j < m1; j++) {
        const v = c.eData[r * recE + it * m1 + j];
        if (it === 8) continue;
        assertEquals(v, 0, `record ${r} theta ${it} j ${j}`);
      }
    }
  }
  // oblique sea: the alongshore phase unroll must survive (no j-broadcast)
  const ev = [0, 1, 2].map((j) => c.eData[100 * recE + 8 * m1 + j]);
  assert(new Set(ev).size === 3, `E constant alongshore: ${ev}`);
  assert(ev.every((v) => v > 0));
  const qv = [0, 1, 2].map((j) => c.qData[100 * recQ + 0 * m1 + j]);
  assert(new Set(qv).size === 3, `qx constant alongshore: ${qv}`);
});

Deno.test("dtchanged branch resamples all four q quarters onto the dtbc axis", () => {
  // dtbc=2 with fgen(K)~0.32 forces dtin=0.5/fgen(K) (waveparamsnew.F90:1856);
  // the eta quarter is interpolated too — deliberate non-replication of the
  // Fortran 3-slice qinterp bug (docs/deviations.md; the engine never reads
  // eta from q records)
  const p = smallParams("dtbc = 2");
  const { x, y } = buildXY(p);
  const zbIn = new Float64Array((p.nx + 1) * (p.ny + 1)).fill(8.0);
  const grid = buildGrid(p, x, y, zbIn);
  const diag: WavemakerDiag = {};
  const c = makeSpectralConditions(
    { kind: "jons", hm0: 1.0, fp: 0.2, gammajsp: 3.3, s: 20, mainangDeg: 270, fnyq: 0.6 },
    p,
    grid,
    { diag },
  )[0];
  const ta = diag.timeAxis!;
  assertEquals(ta.dtchanged, true);
  assert(ta.dtin < p.dtbc);
  const m1 = p.ny + 1, recQ = m1 * 4;
  assertEquals(c.nrec, ta.tslenbc + 2);
  assertEquals(c.eData.length, c.nrec * m1 * grid.ntheta);
  assertEquals(c.qData.length, c.nrec * recQ);
  for (const v of c.eData) assert(Number.isFinite(v));
  for (const v of c.qData) assert(Number.isFinite(v));
  // eta quarter is populated (order=2 default keeps it, :2974-2979)
  let etaAbs = 0;
  for (let r = 0; r < c.nrec; r++) etaAbs += Math.abs(c.qData[r * recQ + 3 * m1]);
  assert(etaAbs > 0, "eta quarter must be written in the dtchanged branch");
});

Deno.test("order=1 zeroes the second-order flux and elevation quarters", () => {
  // waveparamsnew.F90:2974-2979
  const p = smallParams("dtbc = 1\norder = 1");
  const { x, y } = buildXY(p);
  const zbIn = new Float64Array((p.nx + 1) * (p.ny + 1)).fill(8.0);
  const grid = buildGrid(p, x, y, zbIn);
  const c = makeSpectralConditions(FIXTURE_JONS, p, grid)[0];
  const m1 = p.ny + 1, recQ = m1 * 4;
  let qxAbs = 0, qyAbs = 0, etaAbs = 0;
  for (let r = 0; r < c.nrec; r++) {
    for (let j = 0; j < m1; j++) {
      qxAbs += Math.abs(c.qData[r * recQ + 0 * m1 + j]);
      qyAbs += Math.abs(c.qData[r * recQ + 1 * m1 + j]);
      etaAbs += Math.abs(c.qData[r * recQ + 3 * m1 + j]);
    }
  }
  assertEquals(qxAbs, 0, "qx zeroed for order=1");
  assertEquals(etaAbs, 0, "eta zeroed for order=1");
  assert(qyAbs > 0, "qy is kept");
});
