// SPDX-License-Identifier: LGPL-3.0-or-later
// XBeach WebGPU solver — derived from XBeach (c) Deltares (LGPL v2.1+).
// Copyright (c) 2026 Oceanum. See LICENSE.
/**
 * Mean-accumulator layout guards: the scratch offsets readMeans/readRunup
 * compute host-side are tied to the kernels only through NMEAN_SLOTS, so
 * these tests pin the generated WGSL to that constant — adding a slot to
 * meanAccKernel without bumping NMEAN_SLOTS (or vice versa) must fail here,
 * not corrupt the RUNUP block at runtime.
 */
import { assert, assertEquals, assertMatch } from "@std/assert";
import { meanAccKernel, NMEAN_SLOTS, NWL_SLOTS, WATERLINE_DWET_M,
  waterlineAccKernel } from "../src/shaders/bc.ts";

const g = { nx: 8, ny: 4, ntheta: 3 };

Deno.test("meanAcc writes exactly NMEAN_SLOTS slots", () => {
  const wgsl = meanAccKernel(g);
  const slots = [...wgsl.matchAll(/scratch\[MEANS \+ (\d+) \* NCELL \+ kk\]/g)]
    .map((m) => Number(m[1]));
  assertEquals(Math.max(...slots), NMEAN_SLOTS - 1);
  // every slot 0..NMEAN_SLOTS-1 is written (no gap a host reader could misread)
  for (let s = 0; s < NMEAN_SLOTS; s++) assert(slots.includes(s), `slot ${s} missing`);
});

Deno.test("zs slots (0/1) accumulate the deviation from zb (f32 cancellation guard)", () => {
  const wgsl = meanAccKernel({ nx: 4, ny: 4, ntheta: 3 });
  assert(wgsl.includes("let dzs = S2(S2_zs, kk) - max(S2(S2_zb, kk), 0.0);"),
    "zs must accumulate about max(zb,0) — raw sums lose ~0.5 m² on high land, " +
    "and zs-zb loses it in deep water instead");
  assert(wgsl.includes("dt * dzs;") && wgsl.includes("dt * dzs * dzs;"));
});

Deno.test("Er slot (12) is the dt-weighted roller energy R", () => {
  assertMatch(
    meanAccKernel(g),
    /scratch\[MEANS \+ 12 \* NCELL \+ kk\] \+= dt \* S2\(S2_R, kk\)/,
  );
});

Deno.test("|u|max slot (13) is a plain max of GLM |u|, not dt-weighted", () => {
  const wgsl = meanAccKernel(g);
  assertMatch(
    wgsl,
    /scratch\[MEANS \+ 13 \* NCELL \+ kk\] = max\(scratch\[MEANS \+ 13 \* NCELL \+ kk\], umag\)/,
  );
  // GLM cell-centre velocities (XBeach vmag convention), not Eulerian ue/ve
  assertMatch(wgsl, /umag = length\(vec2<f32>\(S2\(S2_u, kk\), S2\(S2_v, kk\)\)\)/);
  const maxLine = wgsl.split("\n").find((l) => l.includes("13 * NCELL + kk] ="))!;
  assert(!maxLine.includes("dt"), "max slot must not be dt-weighted");
});

Deno.test("energy-flux slots (14/15) are the dt-weighted Px/Py of the integrate kernel", () => {
  const wgsl = meanAccKernel(g);
  assertMatch(wgsl, /scratch\[MEANS \+ 14 \* NCELL \+ kk\] \+= dt \* S2\(S2_Px, kk\)/);
  assertMatch(wgsl, /scratch\[MEANS \+ 15 \* NCELL \+ kk\] \+= dt \* S2\(S2_Py, kk\)/);
});

Deno.test("waterline block starts after the NMEAN_SLOTS mean slots, NWL_SLOTS per cell", () => {
  const wgsl = waterlineAccKernel(g);
  assert(wgsl.includes(`const WL: i32 = MEANS + ${NMEAN_SLOTS} * NCELL;`));
  const slots = [...wgsl.matchAll(/scratch\[WL \+ (\d+) \* NCELL \+ kk\]/g)].map((m) => Number(m[1]));
  assertEquals(Math.max(...slots), NWL_SLOTS - 1);
  for (let s = 0; s < NWL_SLOTS; s++) assert(slots.includes(s), `waterline slot ${s} missing`);
});

Deno.test("waterline gauge is per cell: wet with a dry 4-neighbour, level above the cell's bed", () => {
  const wgsl = waterlineAccKernel(g);
  // one thread per cell, not per row — no walk along the row
  assert(wgsl.includes("if (kk >= NCELL) { return; }"));
  assert(!/for \(var i = N1 - 1/.test(wgsl), "must not walk the row from the land side");
  assert(wgsl.includes(`const DWET: f32 = ${WATERLINE_DWET_M};`));
  assert(wgsl.includes("!wetAt(kk - 1) || !wetAt(kk + 1) || !wetAt(kk - N1) || !wetAt(kk + N1)"));
  // the level accumulates above zb (f32 cancellation guard, as meanAcc)
  assert(wgsl.includes("let dz = S2(S2_zs, kk) - S2(S2_zb, kk);"));
  assert(wgsl.includes("dt * dz;") && wgsl.includes("dt * dz * dz;"));
  // grid-boundary cells never gauge
  assert(wgsl.includes("if (i <= 0 || j <= 0 || i >= N1 - 1 || j >= M1 - 1) { return; }"));
});
