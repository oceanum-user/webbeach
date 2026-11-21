// SPDX-License-Identifier: LGPL-3.0-or-later
// Copyright (c) 2026 Oceanum. See LICENSE.
//
// WebBeach — bundle entry point.
//
// This module is the public surface of the solver. `scripts/build.sh`
// bundles it into a standalone `solver.js`, an ES module an application
// loads at run time with a dynamic import(). Keeping the solver a separate,
// replaceable file is what lets a user rebuild it from src/ and swap it in
// (LGPL). docs/INTERFACE.md states the contract; test/interface.test.ts
// holds it.
export { describeParams, parseParamsTxt, resolveParams, type ParamRecord } from "./src/params.ts";
export {
  makeSpectralConditions,
  type SpectralCondition,
  type SpectrumInput,
  type SpectrumTable,
} from "./src/wavemaker.ts";
export { buildGrid, buildXY, type Grid } from "./src/grid.ts";
export { Model } from "./src/model.ts";
export { surfZoneDescriptors } from "./src/derived.ts";
export { type S2Field } from "./src/fields.ts";
