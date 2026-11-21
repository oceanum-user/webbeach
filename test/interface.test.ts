// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (c) 2026 Oceanum. See LICENSE.
/**
 * The public surface applications depend on (docs/INTERFACE.md). A name
 * dropped from the bundle entry or from Model fails here, so the contract
 * changes on purpose and with a version bump — never as a side effect.
 */
import { assert, assertEquals } from "@std/assert";
import * as solver from "../solver.ts";
import { NMEAN_SLOTS } from "../src/shaders/bc.ts";

const FUNCTIONS = [
  "parseParamsTxt", "resolveParams", "describeParams", "buildXY", "buildGrid",
  "makeSpectralConditions", "surfZoneDescriptors",
] as const;

const MODEL_METHODS = [
  "setSpectralBnd", "setBichromLlong", "solveDirections", "wetcells",
  "advanceTime", "writeUniforms", "dispatch", "applyBichromBc", "applyFlowBc",
  "waveInstationaryStep", "flowStep", "accumulateMeans", "gpuBatch",
  "readGpuT", "readField", "readFields", "readMeans", "readWaterline",
  "setTide", "setWind",
] as const;

Deno.test("bundle entry exports the documented functions", () => {
  const s = solver as unknown as Record<string, unknown>;
  for (const name of FUNCTIONS) {
    assertEquals(typeof s[name], "function", `missing export: ${name}`);
  }
});

Deno.test("Model keeps the documented members", () => {
  assertEquals(typeof solver.Model, "function");
  assertEquals(typeof solver.Model.create, "function", "missing Model.create");
  const proto = solver.Model.prototype as unknown as Record<string, unknown>;
  for (const name of MODEL_METHODS) {
    assertEquals(typeof proto[name], "function", `missing Model.${name}`);
  }
});

Deno.test("readMeans keeps 16 slots", () => {
  // A change here is a major version: hosts index the slots by number.
  assert(NMEAN_SLOTS === 16);
});
