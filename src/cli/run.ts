// SPDX-License-Identifier: LGPL-3.0-or-later
// XBeach WebGPU solver — derived from XBeach (c) Deltares (LGPL v2.1+).
// Copyright (c) 2026 Oceanum. See LICENSE.
/**
 * Headless runner: deno task run <case-dir> [--out <dir>]
 * Reads params.txt (XBeach format), runs the WebGPU model, writes .dat
 * outputs compatible with the Fortran comparison layout.
 */

import { parseParamsTxt, resolveParams, WAVEMODEL } from "../params.ts";
import { buildGrid, buildXY } from "../grid.ts";
import { Model } from "../model.ts";
import { DatWriter, readDepFile } from "../io.ts";
import type { S2Field } from "../fields.ts";

const OUTPUT_MAP: Record<string, S2Field> = {
  H: "H", zs: "zs", zb: "zb", u: "u", v: "v", ue: "ue", ve: "ve",
  D: "D", DR: "DR", R: "R", E: "E", Qb: "Qb", Df: "Df",
  k: "k", c: "c", cg: "cg", thetamean: "thetamean", sigm: "sigm",
  urms: "urms", ust: "ust", Fx: "Fx", Fy: "Fy", Sxx: "Sxx", Syy: "Syy",
  Sxy: "Sxy", hh: "hh", uu: "uu", vv: "vv", qx: "qx", qy: "qy",
  wetz: "wetz", zs1: "zs1", ui: "ui", vi: "vi",
};

function parseGlobalVars(text: string): string[] {
  // multi-line format: "nglobalvar = N" followed by N lines of names
  const lines = text.split(/\r?\n/);
  const out: string[] = [];
  for (let li = 0; li < lines.length; li++) {
    const m = lines[li].match(/^\s*nglobalvar\s*=\s*(\d+)/i);
    if (m) {
      const n = parseInt(m[1]);
      for (let k = 1; k <= n && li + k < lines.length; k++) {
        const name = lines[li + k].trim();
        if (name && !name.includes("=")) out.push(name);
      }
      break;
    }
  }
  return out;
}

/**
 * Read the wave-spectrum input for TS table generation: wbctype=parametric
 * (single-partition JONSWAP parameter file, the read_jonswap_file keyword
 * subset of waveparamsnew.F90:429-546) or wbctype=swan (2-D spectrum file,
 * OCE-288, cli/swanfile.ts); vardens/jonstable case files still need
 * pre-generated bcf tables (reuse).
 * The app path is unaffected: it ships a SpectrumTable through the solver
 * API directly.
 */

/** The mean block as the Fortran trunk names it, from the accumulator
 * slots (shaders/bc.ts NMEAN_SLOTS; test/means.test.ts pins each slot).
 * One function for both output loops — the two copies it replaces had
 * drifted: neither wrote slot 12, the roller energy R, for a year
 * (OCE-247, E2), and no parity case could compare it. */
async function appendMeanBlock(writer: import("../io.ts").DatWriter, acc: Float32Array,
  ncell: number, tintm: number, grid: { zb: ArrayLike<number>; alfaz: ArrayLike<number> }) {
  const T = acc[9 * ncell] > 0 ? acc[9 * ncell] : tintm;
  const get = (f: number) => acc.subarray(f * ncell, (f + 1) * ncell);
  // slots 0/1 hold zs-max(zb,0) sums (f32 cancellation guard, meanAcc)
  const zsm = Float64Array.from(get(0), (v, kk) => v / T + Math.max(grid.zb[kk], 0));
  const zsv = Float64Array.from(get(1), (v, kk) => v / T - (get(0)[kk] / T) ** 2);
  const Hm = Float64Array.from(get(2), (v) => Math.sqrt(Math.max(v / T, 0)));
  const um0 = get(3), vm0 = get(4), uem0 = get(5), vem0 = get(6);
  const um = new Float64Array(ncell), vm = new Float64Array(ncell);
  const uem = new Float64Array(ncell), vem = new Float64Array(ncell);
  for (let kk = 0; kk < ncell; kk++) {
    const ca = Math.cos(grid.alfaz[kk]), sa = Math.sin(grid.alfaz[kk]);
    um[kk] = (um0[kk] * ca - vm0[kk] * sa) / T;
    vm[kk] = (um0[kk] * sa + vm0[kk] * ca) / T;
    uem[kk] = (uem0[kk] * ca - vem0[kk] * sa) / T;
    vem[kk] = (uem0[kk] * sa + vem0[kk] * ca) / T;
  }
  await writer.append("zs_mean", zsm);
  await writer.append("zs_var", zsv);
  await writer.append("H_mean", Hm);
  await writer.append("u_mean", um);
  await writer.append("v_mean", vm);
  await writer.append("ue_mean", uem);
  await writer.append("ve_mean", vem);
  await writer.append("urms_mean", Float64Array.from(get(7), (v) => Math.sqrt(Math.max(v / T, 0))));
  await writer.append("D_mean", Float64Array.from(get(8), (v) => v / T));
  await writer.append("Qb_mean", Float64Array.from(get(10), (v) => v / T));
  await writer.append("DR_mean", Float64Array.from(get(11), (v) => v / T));
  // OCE-247 (E2): slot 12 is the dt-weighted roller energy R
  await writer.append("R_mean", Float64Array.from(get(12), (v) => v / T));
}
async function readSpectrumInput(
  p: ReturnType<typeof resolveParams>,
  caseDir: string,
): Promise<import("../wavemaker.ts").SpectrumInput> {
  const { WBCTYPE } = await import("../params.ts");
  if (p.wbctype === WBCTYPE.swan) {
    // OCE-288: a 2-D spectrum file (the app's tools write one per hour) feeds
    // the TS wavemaker's table path exactly as the worker's in-memory table
    const { readSwanTable } = await import("./swanfile.ts");
    const table = readSwanTable(await Deno.readTextFile(`${caseDir}/${p.bcfile}`), p.dthetaS_XB);
    return { kind: "table", table };
  }
  if (p.wbctype !== WBCTYPE.parametric) {
    throw new Error(
      "no ebcflist.bcf in case dir; TS spectral-table generation currently " +
        "supports wbctype=parametric (JONSWAP file) and wbctype=swan (2-D " +
        "spectrum file) — provide Fortran-generated bcf files (reuse " +
        "semantics) for vardens/jonstable",
    );
  }
  const kv = parseParamsTxt(await Deno.readTextFile(`${caseDir}/${p.bcfile}`));
  const num = (k: string) => (k in kv ? parseFloat(kv[k]) : undefined);
  const hm0 = num("hm0");
  if (hm0 === undefined) throw new Error(`${p.bcfile}: Hm0 is required`);
  // Tp takes precedence over fp, default Tp=12.5 s (waveparamsnew.F90:489-501)
  const tp = num("tp");
  const fpFile = num("fp");
  const fp = tp !== undefined ? 1 / tp : fpFile !== undefined ? fpFile : 1 / 12.5;
  return {
    kind: "jons",
    hm0,
    fp,
    gammajsp: num("gammajsp") ?? 3.3,
    s: num("s") ?? 10,
    mainangDeg: num("mainang") ?? num("dir0") ?? 270,
    fnyq: num("fnyq"),
    dfj: num("dfj"),
    tma: (num("tma") ?? 0) === 1,
  };
}

/**
 * Surfbeat (instationary) time loop. Wave BCs:
 *  - bichromatic (wbctype=params + Tlong): analytic GPU kernel
 *  - spectral (parametric/jonstable/swan/vardens/reuse): reads the
 *    Fortran-generated E_series/q_series files from the case dir when they
 *    exist (reuse semantics — every verification case keeps identical
 *    boundary forcing); when no bcf files are present, generates the tables
 *    with the TS wavemaker port (E1, wavemaker.ts).
 */
async function runSurfbeat(
  // deno-lint-ignore no-explicit-any
  model: any,
  p: ReturnType<typeof resolveParams>,
  grid: ReturnType<typeof buildGrid>,
  caseDir: string,
  outputTimes: number[],
  writeOutputs: () => Promise<void>,
  writer?: import("../io.ts").DatWriter,
  gpudt = false,
) {
  const { WBCTYPE } = await import("../params.ts");
  const isBichrom = p.wbctype === WBCTYPE.params && p.Tlong !== null;
  let conds: import("../bcf.ts").BcCondition[] = [];
  let interpBcFn: typeof import("../bcf.ts").interpBc | null = null;
  const m1 = p.ny + 1;
  const ht0 = new Float64Array(m1), alfau0 = new Float64Array(m1);
  if (isBichrom) {
    await model.setBichromLlong();
  } else {
    const bcf = await import("../bcf.ts");
    // reuse-when-present: pre-generated tables keep verification forcing
    // byte-identical. Generation triggers only when the LIST file is absent
    // — a list that references missing series files stays a loud error.
    const haveList = await Deno.stat(`${caseDir}/ebcflist.bcf`)
      .then(() => true, () => false);
    if (haveList) {
      // OCE-252: a single_dir case also needs the Es table (esbcflist.bcf)
      conds = await bcf.readBcfList(caseDir, p.ny, grid.ntheta,
        p.single_dir === 1 ? grid.ntheta_s : 0);
      console.log(`read ${conds.length} bc condition(s), nrec=${conds[0]?.nrec}` +
        (conds[0]?.esData ? `, Es spread ${grid.ntheta_s} bins` : ""));
    } else {
      // no pre-generated tables in the case dir: generate them (E1)
      const spec = await readSpectrumInput(p, caseDir);
      const { makeSpectralConditions } = await import("../wavemaker.ts");
      conds = makeSpectralConditions(spec, p, grid)
        .map((c) => ({ ...c, eFile: "", qFile: "" }));
      console.log(
        `generated spectral bc (TS wavemaker): Trep=${conds[0].Trep.toFixed(5)} ` +
          `Hrms=${conds[0].Hrms.toFixed(5)} theta0=${conds[0].theta0.toFixed(5)} ` +
          `nrec=${conds[0].nrec}`,
      );
    }
    interpBcFn = bcf.interpBc;
    for (let j = 0; j < m1; j++) {
      ht0[j] = p.zs0 - grid.zb[j * (p.nx + 1)];
      alfau0[j] = grid.alfau[j * (p.nx + 1)];
    }
  }
  let nextOut = 0;
  if (outputTimes.length > 0 && Math.abs(outputTimes[0]) < 1e-9) {
    await writeOutputs();
    nextOut = 1;
  }
  // mean output times (tintm intervals from tstart)
  const meanTimes: number[] = [];
  if (writer && p.tintm > 0 && p.tintm < p.tstop) {
    for (let tm = p.tstart + p.tintm; tm <= p.tstop + 1e-9; tm += p.tintm) meanTimes.push(tm);
  }
  let nextMean = 0;
  const ncell = (p.nx + 1) * (p.ny + 1);
  let nsteps = 0;
  let condIdx = 0;
  // E9: time-varying tide (tideloc>0 + zs0file) — the level interpolated
  // at model.t and pushed into S2_zs0 via Model.setTide. Uniform level
  // only (tideloc 1); the spatial-corner variants (tideloc 2-4) are not
  // implemented and the first zs column is used.
  let tide: { t: number[]; z: number[] } | null = null;
  if (p.tideloc > 0 && p.zs0file) {
    const rowsT = (await Deno.readTextFile(`${caseDir}/${p.zs0file}`))
      .split("\n").map((l) => l.trim())
      .filter((l) => l && !l.startsWith("#"))
      .map((l) => l.split(/\s+/).map(Number));
    tide = { t: rowsT.map((r2) => r2[0]), z: rowsT.map((r2) => r2[1]) };
    console.log(`tide series: ${tide.t.length} points (${p.zs0file}), uniform level (E9)`);
  }
  let lastTide = p.zs0;
  // OCE-252 single_dir: the stationary directions sweep runs before the
  // instationary step at t = dt and on every wavint hit (wave_timestep.F90:
  // 89), and after a new boundary condition (newstatbc)
  const singleDir = p.single_dir === 1;
  const wavintHits: number[] = [];
  if (singleDir) {
    for (let k = 1; k * p.wavint <= p.tstop + 1e-9; k++) wavintHits.push(k * p.wavint);
  }
  // the batches land ON the hit (dtStep clamps the last step to tnext, so the
  // f32 clock reads the hit to ~1e-4 s); an absolute window, and one sweep
  // per hit — a 16-step batch that stops 0.3 s short must not fire it
  let lastDirT = -1;
  const atWavint = (t: number) =>
    t > lastDirT + 1 && wavintHits.some((tw) => Math.abs(t - tw) < 2e-3);
  const solveDirs = async (why: string) => {
    lastDirT = model.t;
    const r = await model.solveDirections();
    console.log(`directions t=${model.t.toFixed(2)}s (${why}) iters=${r.iters} ` +
      `herr=${r.herr.toExponential(2)} terr=${(r.terr * 180 / Math.PI).toFixed(3)}deg`);
  };
  const applyTide = () => {
    if (!tide) return;
    const tt = model.t;
    const { t: ts, z } = tide;
    let v: number;
    if (tt <= ts[0]) v = z[0];
    else if (tt >= ts[ts.length - 1]) v = z[z.length - 1];
    else {
      let i2 = 1;
      while (ts[i2] < tt) i2++;
      const f = (tt - ts[i2 - 1]) / (ts[i2] - ts[i2 - 1] || 1);
      v = z[i2 - 1] + (z[i2] - z[i2 - 1]) * f;
    }
    if (Math.abs(v - lastTide) > 1e-4) { lastTide = v; model.setTide(v); }
  };
  const t0 = performance.now();
  model.wetcells(); // step-1 masks (t=0 dt branch has no wetcells)
  model.includeWetcellsInDt = true;
  if (gpudt) {
    console.log("gpudt mode: GPU-resident adaptive dt (batched, no per-step readback)");
    const bcKind = isBichrom ? "bichrom" : "spectral";
    if (!isBichrom) {
      model.p.Trep = conds[0].Trep;
      model.setSpectralBnd(conds[0]);
    }
    // step 1 via the host path (conservative t=0 dt estimate seeds scratch)
    await model.advanceTime(Math.min(p.tstop, outputTimes[nextOut] ?? p.tstop));
    if (singleDir) await solveDirs("t=dt");
    model.writeUniforms({ statMode: 0 });
    {
      const enc0 = model.gpu.device.createCommandEncoder();
      if (isBichrom) model.applyBichromBc(enc0);
      else {
        model.dispatch(enc0, "spectralBc", (p.ny + 1) * grid.ntheta + p.ny + 1);
      }
      if (p.flow === 1) model.applyFlowBc(enc0);
      model.waveInstationaryStep(enc0);
      if (p.flow === 1) model.flowStep(enc0);
      if (meanTimes.length > 0 && model.t > p.tstart) model.accumulateMeans(enc0);
      model.gpu.device.queue.submit([enc0.finish()]);
    }
    nsteps = 1;
    const K = 16;
    while (model.t < p.tstop - 1e-6) {
      const tnexts = [p.tstop];
      if (nextOut < outputTimes.length) tnexts.push(outputTimes[nextOut]);
      if (nextMean < meanTimes.length) tnexts.push(meanTimes[nextMean]);
      // OCE-252: batches end on the wavint hits so the directions sweep runs
      // there (after the hitting step — one step later than Fortran, which
      // runs it before that step's instationary update; docs/deviations.md)
      const nw = wavintHits.find((tw) => tw > model.t + 1e-6);
      if (nw !== undefined) tnexts.push(nw);
      const tnext = Math.min(...tnexts);
      const wantMeans = meanTimes.length > 0 && model.t >= p.tstart;
      const k = Math.max(
        1,
        Math.min(K, Math.ceil((tnext - model.t) / Math.max(model.dt, 1e-6))),
      );
      model.gpuBatch(k, tnext, bcKind, wantMeans);
      nsteps += k;
      await model.readGpuT();
      applyTide();
      let newCond = false;
      if (!isBichrom) {
        while (condIdx < conds.length - 1 && model.t >= conds[condIdx].endtime) {
          condIdx++;
          model.p.Trep = conds[condIdx].Trep;
          model.setSpectralBnd(conds[condIdx]);
          newCond = true;
        }
      }
      if (singleDir && (newCond || atWavint(model.t)) && model.t < p.tstop - 1e-6) {
        await solveDirs(newCond ? "new condition" : "wavint");
      }
      if (
        nextOut < outputTimes.length &&
        Math.abs(model.t - outputTimes[nextOut]) < 1e-3 * Math.max(1, model.t)
      ) {
        await writeOutputs();
        nextOut++;
      }
      if (
        writer && nextMean < meanTimes.length &&
        Math.abs(model.t - meanTimes[nextMean]) < 1e-3 * Math.max(1, model.t)
      ) {
        await appendMeanBlock(writer, await model.readMeans(), ncell, p.tintm, grid);
        nextMean++;
      }
    }
    console.log(
      `surfbeat[gpudt]: ${nsteps} steps in ${
        ((performance.now() - t0) / 1000).toFixed(1)
      }s`,
    );
    return;
  }
  if (singleDir && !isBichrom) model.setSpectralBnd(conds[0]); // Es table to the GPU
  while (model.t < p.tstop - 1e-9) {
    const tnexts = [p.tstop];
    if (nextOut < outputTimes.length) tnexts.push(outputTimes[nextOut]);
    if (nextMean < meanTimes.length) tnexts.push(meanTimes[nextMean]);
    // OCE-252: land exactly on the wavint hits, as timestep.F90 does
    const nw = wavintHits.find((tw) => tw > model.t + 1e-9);
    if (nw !== undefined) tnexts.push(nw);
    // submit 1: wetcells + CFL reduce (+readback); submit 2: whole step
    await model.advanceTime(Math.min(...tnexts));
    nsteps++;
    applyTide();
    let newCond = false;
    if (!isBichrom) {
      while (condIdx < conds.length - 1 && model.t >= conds[condIdx].endtime) {
        condIdx++;
        model.p.Trep = conds[condIdx].Trep;
        if (singleDir) model.setSpectralBnd(conds[condIdx]);
        newCond = true;
      }
    }
    // OCE-252: the directions sweep precedes this step's instationary update
    // (wave_timestep.F90:89-92: t = dt, every wavint, new condition)
    if (singleDir && (nsteps === 1 || newCond || atWavint(model.t))) {
      await solveDirs(nsteps === 1 ? "t=dt" : newCond ? "new condition" : "wavint");
    }
    model.writeUniforms({ statMode: 0 });
    const noBatch = Deno.env.get("XBGPU_NOBATCH") === "1";
    const enc = noBatch ? undefined : model.gpu.device.createCommandEncoder();
    if (isBichrom) {
      model.applyBichromBc(enc);
    } else {
      const c = conds[condIdx];
      if (nsteps === 1) model.p.Trep = c.Trep;
      const taperfac = p.taper > 1e-30 ? Math.min(model.t / p.taper, 1) : 1;
      const { eeRow, uiRow, viRow } = interpBcFn!(
        c, model.t, p.ny, grid.ntheta, taperfac, ht0, alfau0,
      );
      model.applyWaveBcRow(eeRow, uiRow, viRow, enc);
    }
    if (p.flow === 1) model.applyFlowBc(enc);
    model.waveInstationaryStep(enc);
    if (p.flow === 1) model.flowStep(enc);
    if (meanTimes.length > 0 && model.t > p.tstart) model.accumulateMeans(enc);
    if (enc) model.gpu.device.queue.submit([enc.finish()]);
    if (
      nextOut < outputTimes.length &&
      Math.abs(model.t - outputTimes[nextOut]) < 1e-7 * Math.max(1, model.t)
    ) {
      await writeOutputs();
      nextOut++;
      if (nextOut % 2 === 0) {
        console.log(`t=${model.t.toFixed(1)}s (${nsteps} steps)`);
      }
    }
    if (
      writer && nextMean < meanTimes.length &&
      Math.abs(model.t - meanTimes[nextMean]) < 1e-7 * Math.max(1, model.t)
    ) {
      await appendMeanBlock(writer, await model.readMeans(), ncell, p.tintm, grid);
      nextMean++;
    }
  }
  if (writer) {
    // OCE-243: the per-cell waterline gauge — run-long, so written once at
    // the end: time as the waterline (s), mean level (m, datum) and its
    // standard deviation; NaN where the cell never was the waterline
    const wl = await model.readWaterline();
    await writer.append("wl_T", Float64Array.from(wl.T));
    const wlMean = new Float64Array(ncell);
    for (let k = 0; k < ncell; k++) wlMean[k] = Number.isFinite(wl.mean[k]) ? wl.mean[k] + grid.zb[k] : NaN;
    await writer.append("wl_mean", wlMean);
    await writer.append("wl_std", Float64Array.from(wl.std));
  }
  console.log(
    `surfbeat: ${nsteps} steps in ${((performance.now() - t0) / 1000).toFixed(1)}s`,
  );
}

async function main() {
  const args = [...Deno.args];
  const caseDir = args.find((a) => !a.startsWith("--")) ?? ".";
  const outIdx = args.indexOf("--out");
  const outDir = outIdx >= 0 ? args[outIdx + 1] : `${caseDir}/webgpu_out`;

  const paramsText = await Deno.readTextFile(`${caseDir}/params.txt`);
  const raw = parseParamsTxt(paramsText);
  const p = resolveParams(raw);
  const globalVars = parseGlobalVars(paramsText);

  // E9: with a tide series, the INITIAL still-water level is the series'
  // first value (initialize.F90 seeds zs from the tide file)
  if (p.tideloc > 0 && p.zs0file) {
    const first = (await Deno.readTextFile(`${caseDir}/${p.zs0file}`))
      .split("\n").map((l) => l.trim())
      .find((l) => l && !l.startsWith("#"));
    if (first) p.zs0 = Number(first.split(/\s+/)[1]) || 0;
  }
  const dep = await readDepFile(`${caseDir}/${p.depfile}`, p.nx, p.ny);
  let xf: Float64Array | undefined, yf: Float64Array | undefined;
  if (p.vardx === 1) {
    xf = await readDepFile(`${caseDir}/${p.xfile}`, p.nx, p.ny);
    if (p.yfile) yf = await readDepFile(`${caseDir}/${p.yfile}`, p.nx, p.ny);
  }
  const { x, y } = buildXY(p, xf, yf);
  const grid = buildGrid(p, x, y, dep);

  console.log(
    `grid ${p.nx + 1}x${p.ny + 1}, ntheta=${grid.ntheta}, wavemodel=${p.wavemodel}`,
  );
  const model = await Model.create(p, grid);
  console.log(`GPU: ${model.gpu.adapterInfo}`);

  await Deno.mkdir(outDir, { recursive: true });
  const writer = new DatWriter(outDir);
  const outputTimes: number[] = [];
  for (let t = p.tstart; t <= p.tstop + 1e-9; t += p.tintg) outputTimes.push(t);

  const t0 = performance.now();

  // postprocess.F90 rotate=1 conventions on output
  async function writeOutputs() {
    for (const name of globalVars) {
      const f = OUTPUT_MAP[name];
      if (!f) continue;
      const data = await model.readField(f);
      const out = Float64Array.from(data);
      if (name === "thetamean") {
        for (let k = 0; k < out.length; k++) out[k] = 270 - out[k] * (180 / Math.PI);
      } else if (name === "Fx" || name === "u" || name === "ue" || name === "uu") {
        const other = await model.readField(
          name === "Fx" ? "Fy" : name === "u" ? "v" : name === "ue" ? "ve" : "vv",
        );
        for (let k = 0; k < out.length; k++) {
          out[k] = data[k] * Math.cos(grid.alfaz[k]) - other[k] * Math.sin(grid.alfaz[k]);
        }
      } else if (name === "Fy" || name === "v" || name === "ve" || name === "vv") {
        const other = await model.readField(
          name === "Fy" ? "Fx" : name === "v" ? "u" : name === "ve" ? "ue" : "uu",
        );
        for (let k = 0; k < out.length; k++) {
          out[k] = other[k] * Math.sin(grid.alfaz[k]) + data[k] * Math.cos(grid.alfaz[k]);
        }
      }
      await writer.append(name, out);
    }
  }

  if (p.wavemodel === WAVEMODEL.surfbeat) {
    const gpudt = Deno.args.includes("--gpudt");
    await runSurfbeat(model, p, grid, caseDir, outputTimes, writeOutputs, writer, gpudt);
    const elapsed0 = (performance.now() - t0) / 1000;
    console.log(`done in ${elapsed0.toFixed(2)}s -> ${outDir}`);
    writer.close();
    await Deno.writeTextFile(
      `${outDir}/manifest.json`,
      JSON.stringify(
        {
          case: caseDir, nx: p.nx, ny: p.ny, ntheta: grid.ntheta,
          vars: globalVars, elapsed_s: elapsed0, gpu: model.gpu.adapterInfo,
        },
        null,
        2,
      ),
    );
    return;
  }

  // executestep loop (libxbeach.F90): wetcells -> dt -> wave_bc -> flow_bc ->
  // wave (every wavint, timestep.F90:161-168 schedules exact hits) -> flow
  const waveTimes: number[] = [];
  for (let k = 1; k * p.wavint <= p.tstop + 1e-9; k++) waveTimes.push(k * p.wavint);
  let nextOut = 0; // output index into outputTimes
  let nRecords = 0;
  // record at t=0 (initial state), as Fortran does
  if (outputTimes.length > 0 && Math.abs(outputTimes[0]) < 1e-9) {
    await writeOutputs();
    nRecords++;
    nextOut = 1;
  }
  model.newstatbc = 1;
  let nsteps = 0;
  model.wetcells(); // step-1 masks (t=0 dt branch has no wetcells)
  model.includeWetcellsInDt = true;
  while (model.t < p.tstop - 1e-9) {
    const tnexts = [p.tstop];
    if (nextOut < outputTimes.length) tnexts.push(outputTimes[nextOut]);
    const nw = waveTimes.find((tw) => tw > model.t + 1e-9);
    if (nw !== undefined) tnexts.push(nw);
    const tnext = Math.min(...tnexts);
    await model.advanceTime(tnext);
    nsteps++;
    // wave_bc
    model.applyStatWaveBc();
    if (p.flow === 1) model.applyFlowBc();
    // wave (stationary): update on wavint hits or new BC
    const waveHit = Math.abs(model.t % p.wavint) < 0.001 * model.dt ||
      Math.abs((model.t % p.wavint) - p.wavint) < 0.001 * model.dt;
    if (p.swave === 1 && (waveHit || model.newstatbc === 1)) {
      const r = await model.solveStationary();
      model.newstatbc = 0;
      console.log(
        `wave solve t=${model.t.toFixed(2)}s iters=${r.iters} herr=${
          r.herr.toExponential(2)
        }`,
      );
    }
    if (p.flow === 1) model.flowStep();
    // output
    if (
      nextOut < outputTimes.length &&
      Math.abs(model.t - outputTimes[nextOut]) < 1e-7 * Math.max(1, model.t)
    ) {
      await writeOutputs();
      nRecords++;
      nextOut++;
    }
  }
  console.log(`${nsteps} steps, ${nRecords} output records`);

  const elapsed = (performance.now() - t0) / 1000;
  console.log(`done in ${elapsed.toFixed(2)}s -> ${outDir}`);
  writer.close();
  // manifest
  await Deno.writeTextFile(
    `${outDir}/manifest.json`,
    JSON.stringify(
      {
        case: caseDir,
        nx: p.nx,
        ny: p.ny,
        ntheta: grid.ntheta,
        vars: globalVars,
        records: 1,
        elapsed_s: elapsed,
        gpu: model.gpu.adapterInfo,
      },
      null,
      2,
    ),
  );
}

if (import.meta.main) await main();
