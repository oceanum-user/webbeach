// SPDX-License-Identifier: LGPL-3.0-or-later
// XBeach WebGPU solver — derived from XBeach (c) Deltares (LGPL v2.1+).
// Copyright (c) 2026 Oceanum. See LICENSE.
/**
 * SWAN ASCII 2-D spectrum reader (src/cli/swanfile.ts, OCE-288): the CLI's
 * route from a case directory's spectrum file to the TS wavemaker's table.
 * Checks the axes, the FACTOR scaling, the exception value, the EnDens
 * conversion, the first-block-only rule and the direction offset.
 */
import { assertAlmostEquals, assertEquals, assertThrows } from "@std/assert";
import { readSwanTable } from "../src/cli/swanfile.ts";

function swan(quant: "VaDens" | "EnDens", blocks: number[][][], factor = 1, exception = -99): string {
  const nf = blocks[0].length, nd = blocks[0][0].length;
  const unit = quant === "VaDens" ? "m2/Hz/degr" : "J/m2/Hz/degr";
  const lines = [
    "SWAN   1", "$ test", "TIME", "1", "LOCATIONS", "1", "0.0 0.0",
    "AFREQ", String(nf), ...Array.from({ length: nf }, (_, i) => (0.05 * (i + 1)).toFixed(4)),
    "NDIR", String(nd), ...Array.from({ length: nd }, (_, j) => (10 * j).toFixed(1)),
    "QUANT", "1", quant, unit, String(exception),
  ];
  blocks.forEach((b, t) => {
    lines.push(`20000101.${String(t).padStart(2, "0")}0000`, "FACTOR", factor.toExponential(6));
    for (const row of b) lines.push(row.map((v) => String(v)).join(" "));
  });
  return lines.join("\n") + "\n";
}

Deno.test("VaDens block: axes, FACTOR scaling and the exception value", () => {
  const b = [[1, 2, 3], [4, -99, 6]];
  const t = readSwanTable(swan("VaDens", [b], 0.5));
  assertEquals(Array.from(t.freq).map((v) => +v.toFixed(4)), [0.05, 0.1]);
  assertEquals(Array.from(t.dirDeg), [0, 10, 20]);
  assertEquals(t.efth.length, 6); // [ifreq*ndir + idir]
  assertAlmostEquals(t.efth[0 * 3 + 2], 1.5, 1e-12);
  assertAlmostEquals(t.efth[1 * 3 + 0], 2.0, 1e-12);
  assertEquals(t.efth[1 * 3 + 1], 0); // exception -> no energy
});

Deno.test("EnDens is divided by rho g; only the first time block is read", () => {
  const b1 = [[1025 * 9.81, 0], [0, 0]];
  const b2 = [[7, 7], [7, 7]];
  const t = readSwanTable(swan("EnDens", [b1, b2]));
  assertAlmostEquals(t.efth[0], 1.0, 1e-12);
  assertEquals(t.efth[3], 0);
});

Deno.test("the dthetaS_XB offset is added to every direction", () => {
  const t = readSwanTable(swan("VaDens", [[[1, 1], [1, 1]]]), 15);
  assertEquals(Array.from(t.dirDeg), [15, 25]);
});

Deno.test("a truncated file fails loudly", () => {
  const text = swan("VaDens", [[[1, 2], [3, 4]]]).split("\n").slice(0, -3).join("\n");
  assertThrows(() => readSwanTable(text));
});
