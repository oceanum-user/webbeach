// SPDX-License-Identifier: LGPL-3.0-or-later
// XBeach WebGPU solver — derived from XBeach (c) Deltares (LGPL v2.1+).
// Copyright (c) 2026 Oceanum. See LICENSE.
/**
 * Tier-1 wavemaker parity gate: the TS spectral wavemaker
 * (makeSpectralConditions, random=0 -> seed 100) must reproduce the
 * Fortran-generated E_series/q_series tables of the committed fixture
 * verification/{cases,reference}/surfbeat1d_jons_r0 record-by-record.
 *
 * Tolerance 1e-6 relative to the reference table maximum — the measured
 * margin is ~4e-9 (docs/deviations.md "Measured parity"), ~2.5 orders below.
 * The case's REAL bed.dep is load-bearing: the effective boundary depth is
 * bed.dep column 2 after the zb row copy (initialize.F90:248, mirrored in
 * grid.ts) and a constant-depth stand-in fails q parity by ~0.6 %.
 * Pure CPU — runs in GitHub CI via `deno task test`.
 */
import { assert, assertAlmostEquals, assertEquals } from "@std/assert";
import { parseParamsTxt, resolveParams } from "../src/params.ts";
import { buildGrid, buildXY } from "../src/grid.ts";
import { readDepFile } from "../src/io.ts";
import { makeSpectralConditions, type SpectrumInput } from "../src/wavemaker.ts";
import { type BcCondition, readBcfList } from "../src/bcf.ts";

const CASE = new URL("../verification/cases/surfbeat1d_jons_r0/", import.meta.url);
const REF = new URL("../verification/reference/surfbeat1d_jons_r0/", import.meta.url);
const RELTOL = 1e-6;

interface Fixture {
  p: ReturnType<typeof resolveParams>;
  gen: ReturnType<typeof makeSpectralConditions>[0];
  ref: BcCondition;
}

let cached: Promise<Fixture> | null = null;

function fixture(): Promise<Fixture> {
  cached ??= (async () => {
    const p = resolveParams(
      parseParamsTxt(await Deno.readTextFile(new URL("params.txt", CASE))),
    );
    assertEquals(p.random, 0, "fixture must be the deterministic random=0 case");
    const dep = await readDepFile(new URL(p.depfile!, CASE).pathname, p.nx, p.ny);
    const { x, y } = buildXY(p);
    const grid = buildGrid(p, x, y, dep);
    // JONSWAP parameter file, same keyword subset run.ts reads
    // (waveparamsnew.F90:429-546 via readSpectrumInput)
    const kv = parseParamsTxt(await Deno.readTextFile(new URL(p.bcfile!, CASE)));
    const num = (k: string) => (k in kv ? parseFloat(kv[k]) : undefined);
    const hm0 = num("hm0");
    assert(hm0 !== undefined, `${p.bcfile}: Hm0 is required`);
    const tp = num("tp");
    const fpFile = num("fp");
    const spec: SpectrumInput = {
      kind: "jons",
      hm0,
      fp: tp !== undefined ? 1 / tp : fpFile !== undefined ? fpFile : 1 / 12.5,
      gammajsp: num("gammajsp") ?? 3.3,
      s: num("s") ?? 10,
      mainangDeg: num("mainang") ?? num("dir0") ?? 270,
      fnyq: num("fnyq"),
      dfj: num("dfj"),
      tma: (num("tma") ?? 0) === 1,
    };
    const conds = makeSpectralConditions(spec, p, grid);
    assertEquals(conds.length, 1);
    const refConds = await readBcfList(REF.pathname, p.ny, grid.ntheta);
    assertEquals(refConds.length, 1);
    return { p, gen: conds[0], ref: refConds[0] };
  })();
  return cached;
}

/** Max |gen-ref| per record, asserted against reltol * max|ref| overall. */
function compareTable(
  label: string,
  gen: Float64Array,
  ref: Float64Array,
  nrec: number,
  stride: (r: number, k: number) => number,
  recLen: number,
) {
  let maxRef = 0;
  for (let r = 0; r < nrec; r++) {
    for (let k = 0; k < recLen; k++) {
      maxRef = Math.max(maxRef, Math.abs(ref[stride(r, k)]));
    }
  }
  assert(maxRef > 0, `${label}: reference table is all zero`);
  let worst = 0, worstRec = -1;
  for (let r = 0; r < nrec; r++) {
    let recMax = 0;
    for (let k = 0; k < recLen; k++) {
      const i = stride(r, k);
      recMax = Math.max(recMax, Math.abs(gen[i] - ref[i]));
    }
    if (recMax > worst) {
      worst = recMax;
      worstRec = r;
    }
  }
  assert(
    worst <= RELTOL * maxRef,
    `${label}: max rel diff ${(worst / maxRef).toExponential(2)} at record ` +
      `${worstRec} exceeds ${RELTOL} (abs ${worst.toExponential(2)}, ` +
      `table max ${maxRef.toExponential(2)})`,
  );
}

Deno.test("r0 parity: scalar anchors match the Fortran ebcflist line", async () => {
  const { gen, ref } = await fixture();
  // ref values carry the Fortran list-file print precision (3-5 decimals)
  assertAlmostEquals(gen.endtime, ref.endtime, 1e-3);
  assertAlmostEquals(gen.rt, ref.rt, 1e-3);
  assertAlmostEquals(gen.dtbc, ref.dtbc, 1e-3);
  assertAlmostEquals(gen.Trep, ref.Trep, 1e-4);
  assertAlmostEquals(gen.theta0, ref.theta0, 5e-4);
  assertAlmostEquals(gen.Hrms, ref.Hrms, 1e-4);
  assertEquals(gen.nrec, ref.nrec);
  assertEquals(gen.eData.length, ref.eData.length);
  assertEquals(gen.qData.length, ref.qData.length);
});

Deno.test("r0 parity: E table matches Fortran record-by-record", async () => {
  const { p, gen, ref } = await fixture();
  const recE = gen.eData.length / gen.nrec;
  assertEquals(recE, (p.ny + 1) * 16); // ntheta = (80-(-80))/10
  compareTable("E", gen.eData, ref.eData, gen.nrec, (r, k) => r * recE + k, recE);
});

Deno.test("r0 parity: q table matches Fortran per column record-by-record", async () => {
  const { p, gen, ref } = await fixture();
  const m1 = p.ny + 1;
  const recQ = m1 * 4;
  assertEquals(gen.qData.length / gen.nrec, recQ);
  // per column so a small column (eta) is not masked by qtot's scale
  const cols = ["qx", "qy", "qtot", "eta"] as const;
  for (let c = 0; c < 4; c++) {
    compareTable(
      `q.${cols[c]}`,
      gen.qData,
      ref.qData,
      gen.nrec,
      (r, k) => r * recQ + c * m1 + k,
      m1,
    );
  }
});
