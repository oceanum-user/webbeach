// SPDX-License-Identifier: LGPL-3.0-or-later
// XBeach WebGPU solver — derived from XBeach (c) Deltares (LGPL v2.1+).
// Copyright (c) 2026 Oceanum. See LICENSE.
/**
 * Bit-exactness tests for the Park-Miller stream
 * (trunk/src/xbeachlibrary/math_tools.F90:1268-1305). Expected integer seeds
 * come from an independent BigInt oracle of seed_{n+1} = 16807*seed_n mod
 * (2^31-1); every intermediate of the Schrage form stays below 2^31, so the
 * f64 implementation must reproduce them exactly.
 */
import { assertEquals, assertNotEquals } from "@std/assert";
import { XbRandom } from "../src/rng.ts";

const M = 2147483647;

// First 10 states of the seed-100 series (BigInt oracle).
const SEED100_STATES = [
  1680700,
  330237489,
  1203733775,
  1857601685,
  594259709,
  1923970613,
  1512819812,
  1903683451,
  1996387951,
  1007791729,
];

Deno.test("seed 100 stream is bit-exact for the first 10 draws", () => {
  const r = new XbRandom();
  const draws = [r.seed(100)];
  for (let i = 1; i < 10; i++) draws.push(r.next());
  for (let i = 0; i < 10; i++) {
    assertEquals(draws[i], SEED100_STATES[i] / M, `draw ${i + 1}`);
  }
});

Deno.test("random(j) returns the FIRST draw of the re-seeded series", () => {
  // math_tools.F90:1288-1294: the seeding call consumes one number.
  const r = new XbRandom();
  assertEquals(r.seed(100), 1680700 / M);
  // and the next call continues the same series, not a fresh one
  assertEquals(r.next(), 330237489 / M);
});

Deno.test("seeding uses |j| (Fortran iabs)", () => {
  const a = new XbRandom();
  const b = new XbRandom();
  assertEquals(a.seed(-100), b.seed(100));
  assertEquals(a.next(), b.next());
});

Deno.test("seed multiple of m collapses to 1 (mod-m, zero replaced by 1)", () => {
  // math_tools.F90:1289-1293: seed = mod(|j|, m); if 0 then 1.
  const a = new XbRandom();
  const b = new XbRandom();
  assertEquals(a.seed(M), b.seed(1));
  assertEquals(a.next(), b.next());
});

Deno.test("Schrage negative branch adds m", () => {
  // state = q = 127773 -> hi = 1, lo = 0 -> test = -2836 <= 0 -> + m
  const r = new XbRandom();
  assertEquals(r.seed(127773), 2147480811 / M); // 2147480811 = m - 2836 (oracle)
});

Deno.test("default state matches the Fortran 'data seed/1/'", () => {
  // next() without seeding starts from state 1 (math_tools.F90:1283-1284)
  const r = new XbRandom();
  assertEquals(r.next(), 16807 / M);
  assertNotEquals(r.next(), 16807 / M);
});
