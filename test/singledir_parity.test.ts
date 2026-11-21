// SPDX-License-Identifier: LGPL-3.0-or-later
// XBeach WebGPU solver — derived from XBeach (c) Deltares (LGPL v2.1+).
// Copyright (c) 2026 Oceanum. See LICENSE.
/**
 * OCE-252 single_dir wavemaker parity: on the committed 2-D fixture
 * verification/{cases,reference}/rip2d_jons_sd1 (JONSWAP, random=0, single_dir=1,
 * dtheta_s=10) the TS wavemaker must reproduce the Fortran tables:
 *   - E_series with ONE direction bin spanning the window (ntheta=1),
 *   - q_series,
 *   - Es_series: the stationary boundary spread ee_s(1,:,:) on the 12-bin
 *     theta_s grid (set_stationary_spectrum / interp_using_trapez_rule).
 * Pure CPU — runs in CI via `deno task test`.
 */
import { assert, assertAlmostEquals, assertEquals } from "@std/assert";
import { parseParamsTxt, resolveParams } from "../src/params.ts";
import { buildGrid, buildXY } from "../src/grid.ts";
import { readDepFile } from "../src/io.ts";
import { makeSpectralConditions, type SpectrumInput } from "../src/wavemaker.ts";
import { interpUsingTrapezRule } from "../src/spectrum.ts";
import { type BcCondition, readBcfList } from "../src/bcf.ts";

const CASE = new URL("../verification/cases/rip2d_jons_sd1/", import.meta.url);
const REF = new URL("../verification/reference/rip2d_jons_sd1/", import.meta.url);
const RELTOL = 1e-6;

interface Fixture {
  p: ReturnType<typeof resolveParams>;
  grid: ReturnType<typeof buildGrid>;
  gen: ReturnType<typeof makeSpectralConditions>[0];
  ref: BcCondition;
}
let cached: Promise<Fixture> | null = null;
function fixture(): Promise<Fixture> {
  cached ??= (async () => {
    const p = resolveParams(parseParamsTxt(await Deno.readTextFile(new URL("params.txt", CASE))));
    assertEquals(p.single_dir, 1);
    assertEquals(p.random, 0);
    const dep = await readDepFile(new URL(p.depfile!, CASE).pathname, p.nx, p.ny);
    const { x, y } = buildXY(p);
    const grid = buildGrid(p, x, y, dep);
    assertEquals(grid.ntheta, 1, "single_dir: one instationary bin");
    assertEquals(grid.ntheta_s, 12, "(60-(-60))/10 stationary bins");
    const kv = parseParamsTxt(await Deno.readTextFile(new URL(p.bcfile!, CASE)));
    const num = (k: string) => (k in kv ? parseFloat(kv[k]) : undefined);
    const spec: SpectrumInput = {
      kind: "jons", hm0: num("hm0")!, fp: num("fp")!, gammajsp: num("gammajsp") ?? 3.3,
      s: num("s") ?? 10, mainangDeg: num("mainang") ?? 270, fnyq: num("fnyq"), dfj: num("dfj"),
      tma: false,
    };
    const conds = makeSpectralConditions(spec, p, grid);
    assertEquals(conds.length, 1);
    const refConds = await readBcfList(REF.pathname, p.ny, grid.ntheta, grid.ntheta_s);
    assertEquals(refConds.length, 1);
    return { p, grid, gen: conds[0], ref: refConds[0] };
  })();
  return cached;
}

function compareTable(label: string, gen: ArrayLike<number>, ref: ArrayLike<number>, n: number,
  idx: (k: number) => number = (k) => k) {
  let maxRef = 0, worst = 0, at = -1;
  for (let k = 0; k < n; k++) maxRef = Math.max(maxRef, Math.abs(ref[idx(k)]));
  assert(maxRef > 0, `${label}: reference all zero`);
  for (let k = 0; k < n; k++) {
    const d = Math.abs(gen[idx(k)] - ref[idx(k)]);
    if (d > worst) { worst = d; at = k; }
  }
  assert(worst <= RELTOL * maxRef,
    `${label}: max rel diff ${(worst / maxRef).toExponential(2)} at ${at} exceeds ${RELTOL}`);
}

Deno.test("single_dir parity: list-line anchors and one-bin table shapes", async () => {
  const { p, gen, ref } = await fixture();
  assertAlmostEquals(gen.Trep, ref.Trep, 1e-4);
  assertAlmostEquals(gen.theta0, ref.theta0, 5e-4);
  assertAlmostEquals(gen.Hrms, ref.Hrms, 1e-4);
  assertEquals(gen.nrec, ref.nrec);
  assertEquals(gen.eData.length, ref.eData.length);
  assertEquals(gen.eData.length / gen.nrec, p.ny + 1); // ntheta = 1
  assertEquals(gen.qData.length, ref.qData.length);
});

Deno.test("single_dir parity: E (one bin) and q tables match Fortran", async () => {
  const { p, gen, ref } = await fixture();
  compareTable("E", gen.eData, ref.eData, gen.eData.length);
  const m1 = p.ny + 1, recQ = m1 * 4;
  for (const [c, name] of ["qx", "qy", "qtot", "eta"].entries()) {
    compareTable(`q.${name}`, gen.qData, ref.qData, gen.nrec * m1,
      (k) => Math.floor(k / m1) * recQ + c * m1 + (k % m1));
  }
});

Deno.test("single_dir parity: Es stationary boundary spread matches Fortran", async () => {
  const { p, grid, gen, ref } = await fixture();
  assert(gen.esData, "generated condition carries esData");
  assert(ref.esData, "reference carries esData (esbcflist.bcf)");
  const n = (p.ny + 1) * grid.ntheta_s;
  assertEquals(gen.esData!.length, n);
  assertEquals(ref.esData!.length, n);
  compareTable("Es", gen.esData!, ref.esData!, n);
  // alongshore-constant, as set_stationary_spectrum writes it
  const m1 = p.ny + 1;
  for (let it = 0; it < grid.ntheta_s; it++) {
    for (let j = 1; j < m1; j++) assertEquals(gen.esData![it * m1 + j], gen.esData![it * m1]);
  }
  // the spread integrates to the full energy: sum(Es)·dtheta_s ≈ rhog8·Hrms²
  // in the window — a 280° JONSWAP with s=20 sits almost wholly inside ±60°
  let e = 0;
  for (let it = 0; it < grid.ntheta_s; it++) e += gen.esData![it * m1];
  const E0 = p.rhog8 * gen.Hrms * gen.Hrms;
  assert(e * grid.dtheta_s > 0.9 * E0 && e * grid.dtheta_s <= 1.02 * E0,
    `Es energy ${e * grid.dtheta_s} vs E0 ${E0}`);
});

Deno.test("interpUsingTrapezRule: bin averages of a cyclic function", () => {
  // y = 1 + cos(x) on 0..2π (400 steps, as the spectrum grid) averaged onto
  // 12 bins of 30° centred at 15°, 45°, …: the exact bin mean is
  // 1 + (sin(b) - sin(a))/(b - a)
  const n = 401, dx = 2 * Math.PI / 400;
  const x = new Float64Array(n), y = new Float64Array(n);
  for (let i = 0; i < n; i++) { x[i] = i * dx; y[i] = 1 + Math.cos(x[i]); }
  const w = Math.PI / 6;
  const xt = new Float64Array(12);
  for (let i = 0; i < 12; i++) xt[i] = (i + 0.5) * w;
  const out = interpUsingTrapezRule(x, y, 2 * Math.PI, xt, w);
  // 1e-2 of a range of 2: the Fortran scheme is reproduced quirks included
  // (trapezoidal_cyclic skips the last grid node inside a bin, and the cyclic
  // interpolation's mod(xp, dx) weight flips to ~1 on nodes that land a
  // rounding error below a grid point), which shifts the averages by up to
  // half a source step. The Es parity test above is the exact gate against
  // Fortran's own table; this one guards the shape and the cyclic wrap.
  for (let i = 0; i < 12; i++) {
    const a = i * w, b = (i + 1) * w;
    assertAlmostEquals(out[i], 1 + (Math.sin(b) - Math.sin(a)) / w, 1e-2);
  }
});
