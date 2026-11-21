// SPDX-License-Identifier: LGPL-3.0-or-later
// XBeach WebGPU solver — derived from XBeach (c) Deltares (LGPL v2.1+).
// Copyright (c) 2026 Oceanum. See LICENSE.
/**
 * Model orchestration: owns GPU buffers, uniforms, kernel dispatch order.
 * Step order mirrors libxbeach.F90:executestep:
 *   wetcells -> timestep(dt) -> wave_bc -> flow_bc -> wave -> flow
 */

import { Gpu, groups } from "./gpu.ts";
import type { Params } from "./params.ts";
import { BREAK, SCHEME, WAVEMODEL, WBCTYPE } from "./params.ts";
import type { Grid } from "./grid.ts";
import {
  C2_FIELDS, C3_FIELDS, P3_FIELDS, S2_FIELDS,
  c2Offset, c3Offset, p3Offset, s2Offset, sizes,
} from "./fields.ts";
import type { C2Field, S2Field } from "./fields.ts";
import * as wave from "./shaders/wave.ts";
import * as bc from "./shaders/bc.ts";
import * as flow from "./shaders/flow.ts";

const UNI_COUNT = 36; // f32 slots in Uni struct (see common.ts)

export class Model {
  gpu!: Gpu;
  p!: Params;
  grid!: Grid;
  ncell = 0;
  n3 = 0;

  // buffers
  uni!: GPUBuffer;
  uni2!: GPUBuffer; // parity-1 copy for red-black stationary sweeps
  c2!: GPUBuffer;
  s2!: GPUBuffer;
  c3!: GPUBuffer;
  a3!: GPUBuffer; // ping (read)
  b3!: GPUBuffer; // pong (write)
  v3!: GPUBuffer;
  // OCE-252 single_dir: the stationary-directions energy set on ntheta_s
  // bins (ee_s and its angles/velocities), solved every wavint; the
  // instationary set above then has one bin that follows thetamean
  c3s!: GPUBuffer;
  a3s!: GPUBuffer;
  b3s!: GPUBuffer;
  v3s!: GPUBuffer;
  n3s = 0;
  scratch!: GPUBuffer;
  red!: GPUBuffer;
  bnd!: GPUBuffer; // spectral boundary record tables (gpudt mode)

  t = 0;
  dt = 0;
  includeWetcellsInDt = false;
  newstatbc = 1;
  e01 = new Float32Array(0);

  private uniData = new Float32Array(UNI_COUNT);
  private kernels = new Map<string, GPUComputePipeline>();
  private bindCache = new Map<string, GPUBindGroup>();

  static async create(p: Params, grid: Grid): Promise<Model> {
    const m = new Model();
    m.gpu = await Gpu.init();
    m.gpu.makeSharedLayout();
    m.gpu.device.addEventListener?.("uncapturederror", (ev) => {
      // deno-lint-ignore no-explicit-any
      console.error("WebGPU error:", (ev as any).error?.message);
    });
    m.p = p;
    m.grid = grid;
    m.ncell = (grid.nx + 1) * (grid.ny + 1);
    m.n3 = m.ncell * grid.ntheta;
    m.n3s = p.single_dir === 1 ? m.ncell * grid.ntheta_s : 0;
    m.allocate();
    m.initState();
    m.buildKernels();
    return m;
  }

  private allocate() {
    const g = this.gpu;
    const sz = sizes(this.grid.nx, this.grid.ny, this.grid.ntheta);
    this.uni = g.buffer("uni", UNI_COUNT * 4, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.uni2 = g.buffer("uni2", UNI_COUNT * 4, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    this.c2 = g.buffer("c2", sz.c2 * 4);
    this.s2 = g.buffer("s2", sz.s2 * 4);
    this.c3 = g.buffer("c3", Math.max(sz.c3, 4) * 4);
    this.a3 = g.buffer("a3", Math.max(sz.p3, 4) * 4);
    this.b3 = g.buffer("b3", Math.max(sz.p3, 4) * 4);
    this.v3 = g.buffer("v3", Math.max(sz.v3, 4) * 4);
    // OCE-252: the directions set (a 4-byte stand-in keeps the bind group
    // valid when single_dir is off)
    const szs = this.p.single_dir === 1
      ? sizes(this.grid.nx, this.grid.ny, this.grid.ntheta_s) : { c3: 1, p3: 1, v3: 1 };
    this.c3s = g.buffer("c3s", Math.max(szs.c3, 4) * 4);
    this.a3s = g.buffer("a3s", Math.max(szs.p3, 4) * 4);
    this.b3s = g.buffer("b3s", Math.max(szs.p3, 4) * 4);
    this.v3s = g.buffer("v3s", Math.max(szs.v3, 4) * 4);
    const bcrow = (this.grid.ny + 1) * Math.max(this.grid.ntheta, 1) + 2 * (this.grid.ny + 1);
    // + NMEAN_SLOTS*ncell mean accumulators
    //   (zs, zs2, H2, u, v, ue, ve, urms2, D, wsum, Qb, DR, R, |u|max, Px, Py)
    // + NWL_SLOTS*ncell waterline-gauge accumulators (OCE-243)
    this.scratch = g.buffer("scratch", (1024 + 3 * this.ncell + bcrow + (bc.NMEAN_SLOTS + bc.NWL_SLOTS) * this.ncell) * 4);
    this.red = g.buffer("red", (8 + this.grid.nx + 1 + 4) * 4);
    this.bnd = g.buffer("bnd", 4); // reallocated by setSpectralBnd when needed
  }

  private initState() {
    const { grid: gr, p, ncell } = this;
    const g = this.gpu;

    // --- constant 2D fields
    const c2Host = new Float32Array(C2_FIELDS.length * ncell);
    const setC2 = (f: C2Field, data: Float64Array | number) => {
      const off = c2Offset(f, ncell);
      if (typeof data === "number") c2Host.fill(data, off, off + ncell);
      else for (let k = 0; k < ncell; k++) c2Host[off + k] = data[k];
    };
    setC2("dsu", gr.dsu); setC2("dsz", gr.dsz); setC2("dsv", gr.dsv); setC2("dsc", gr.dsc);
    setC2("dnu", gr.dnu); setC2("dnz", gr.dnz); setC2("dnv", gr.dnv); setC2("dnc", gr.dnc);
    setC2("dsdnui", gr.dsdnui); setC2("dsdnvi", gr.dsdnvi); setC2("dsdnzi", gr.dsdnzi);
    setC2("alfaz", gr.alfaz); setC2("alfau", gr.alfau); setC2("alfav", gr.alfav);
    setC2("sdist", gr.sdist); setC2("ndist", gr.ndist);
    setC2("xz", gr.xz); setC2("yz", gr.yz);
    setC2("fw", p.fw); // constant; fwfile is refused in params.ts
    setC2("bedfriccoef_f", p.bedfriccoef);
    g.upload(this.c2, c2Host);

    // --- dynamic 2D fields init (initialize.F90)
    const s2Host = new Float32Array(S2_FIELDS.length * ncell);
    const setS2 = (f: S2Field, data: Float64Array | Float32Array | number) => {
      const off = s2Offset(f, ncell);
      if (typeof data === "number") s2Host.fill(data, off, off + ncell);
      else for (let k = 0; k < ncell; k++) s2Host[off + k] = data[k];
    };
    setS2("zb", gr.zb);
    // zs = max(zb+eps_machine, zs0) (initialize.F90:1011)
    const zs = new Float64Array(ncell);
    for (let k = 0; k < ncell; k++) zs[k] = Math.max(gr.zb[k] + 1.2e-7, p.zs0);
    setS2("zs", zs);
    setS2("zs0", p.zs0);
    const hh = new Float64Array(ncell);
    for (let k = 0; k < ncell; k++) hh[k] = Math.max(zs[k] - gr.zb[k], p.eps);
    setS2("hh", hh);
    setS2("hhw", hh);
    setS2("hhws", hh); // initialize.F90:591 (single_dir smoothed depth)
    setS2("hstokes", hh);
    // hu/hv initial (initialize.F90:1015-1032): no eps floor, edge copies
    const hu = new Float64Array(ncell), hv = new Float64Array(ncell);
    const n1 = gr.nx + 1;
    for (let j = 0; j <= gr.ny; j++) {
      for (let i = 0; i < gr.nx; i++) {
        const k = i + j * n1;
        hu[k] = Math.max(zs[k], zs[k + 1]) - Math.max(gr.zb[k], gr.zb[k + 1]);
      }
      hu[gr.nx + j * n1] = hu[gr.nx - 1 + j * n1];
    }
    if (gr.ny > 0) {
      for (let i = 0; i <= gr.nx; i++) {
        for (let j = 0; j < gr.ny; j++) {
          const k = i + j * n1;
          hv[k] = Math.max(zs[k], zs[k + n1]) - Math.max(gr.zb[k], gr.zb[k + n1]);
        }
        hv[i + gr.ny * n1] = hv[i + (gr.ny - 1) * n1];
      }
    } else {
      for (let i = 0; i <= gr.nx; i++) hv[i] = zs[i] - gr.zb[i];
    }
    setS2("hu", hu);
    setS2("hv", hv);
    setS2("sigm", 2 * Math.PI / p.Trep);
    setS2("L1", -1); // triggers L0 warm start in dispersion
    setS2("thetamean", gr.theta0);
    setS2("BR", p.beta);
    setS2("k", 25);
    setS2("c", Math.sqrt(p.g * p.eps));
    setS2("cg", Math.sqrt(p.g * p.eps));
    setS2("nn", 1);
    // stationary wind: readwind.F90:71-77 — nautical dir hardcoded, world
    // components then projected onto local u/v axes
    if (p.wind === 1 && p.windv > 0) {
      const winddir = (270 - p.windth - p.alfa) * Math.PI / 180;
      const windx = Math.cos(winddir) * p.windv;
      const windy = Math.sin(winddir) * p.windv;
      const wsu = new Float64Array(ncell), wnv = new Float64Array(ncell);
      for (let k = 0; k < ncell; k++) {
        wsu[k] = windx * Math.cos(gr.alfau[k]) + windy * Math.sin(gr.alfau[k]);
        wnv[k] = windy * Math.cos(gr.alfav[k] - Math.PI / 2) -
          windx * Math.sin(gr.alfav[k] - Math.PI / 2);
      }
      setS2("windsu", wsu);
      setS2("windnv", wnv);
    }
    g.upload(this.s2, s2Host);

    // --- constant 3D: thet, costh, sinth
    if (gr.ntheta > 0) {
      const c3Host = new Float32Array(C3_FIELDS.length * this.n3);
      for (let it = 0; it < gr.ntheta; it++) {
        const th = gr.theta[it];
        for (let k = 0; k < ncell; k++) {
          const k3 = k + it * ncell;
          c3Host[c3Offset("thet", ncell, gr.ntheta) + k3] = th;
          c3Host[c3Offset("costh", ncell, gr.ntheta) + k3] = Math.cos(
            (th - gr.alfaz[k]) % (2 * Math.PI),
          );
          c3Host[c3Offset("sinth", ncell, gr.ntheta) + k3] = Math.sin(
            (th - gr.alfaz[k]) % (2 * Math.PI),
          );
        }
      }
      g.upload(this.c3, c3Host);
    }
    // OCE-252: thet_s/costh_s/sinth_s on the directions grid
    // (initialize.F90:376-386)
    if (p.single_dir === 1 && gr.ntheta_s > 0) {
      const c3sHost = new Float32Array(C3_FIELDS.length * this.n3s);
      for (let it = 0; it < gr.ntheta_s; it++) {
        const th = gr.theta_s[it];
        for (let k = 0; k < ncell; k++) {
          const k3 = k + it * ncell;
          c3sHost[c3Offset("thet", ncell, gr.ntheta_s) + k3] = th;
          c3sHost[c3Offset("costh", ncell, gr.ntheta_s) + k3] = Math.cos(
            (th - gr.alfaz[k]) % (2 * Math.PI),
          );
          c3sHost[c3Offset("sinth", ncell, gr.ntheta_s) + k3] = Math.sin(
            (th - gr.alfaz[k]) % (2 * Math.PI),
          );
        }
      }
      g.upload(this.c3s, c3sHost);
    }

    // scratch misc: [0]=oldhmin, [1]=hmin, [2]=deltahmin, [3]=breakerdelay, [4]=taperfac
    const misc = new Float32Array(64);
    misc[0] = p.oldhmin;
    misc[1] = p.hmin;
    misc[2] = p.deltahmin;
    misc[3] = p.breakerdelay;
    misc[4] = 0;
    misc[22] = p.taper;
    misc[24] = p.CFL;
    misc[25] = p.swave === 1 && p.wavemodel !== 0 && gr.ntheta > 1 ? 1 : 0;
    misc[26] = p.wavint; // OCE-252: hhws smoothing time scale
    misc[27] = 0; // OCE-252: Es table offset in bnd (setSpectralBnd)
    g.upload(this.scratch, misc);

    // stationary directional distribution e01 (boundaryconditions.F90:279-296)
    this.computeE01();
  }

  /** e01 for params-type BC: cos^m distribution (boundaryconditions.F90:279-296). */
  computeE01() {
    const gr = this.grid, p = this.p;
    const dist = new Float64Array(gr.ntheta);
    let sum = 0;
    for (let it = 0; it < gr.ntheta; it++) {
      const c = Math.cos(gr.theta[it] - gr.theta0);
      dist[it] = c < 0 ? 0 : Math.pow(c, p.m);
      sum += dist[it];
    }
    const E0 = p.rhog8 * p.Hrms * p.Hrms;
    this.e01 = new Float32Array(gr.ntheta);
    if (sum > 0) {
      for (let it = 0; it < gr.ntheta; it++) {
        this.e01[it] = Math.max((dist[it] / sum / gr.dtheta) * E0, 0);
      }
    }
    this.gpu.upload(this.scratch, this.e01, 64 * 4);
  }

  private buildKernels() {
    const g = { nx: this.grid.nx, ny: this.grid.ny, ntheta: Math.max(this.grid.ntheta, 1) };
    const reg = (name: string, src: string) => {
      this.kernels.set(name, this.gpu.pipeline(name, src));
    };
    reg("wetcellsA", wave.wetcellsA(g));
    reg("wetcellsB", wave.wetcellsB(g));
    reg("wavePrep", wave.wavePrep(g));
    reg("dispersion", wave.dispersionKernel(g));
    reg("slopes", wave.slopesKernel(g));
    reg("dirvel", wave.dirvelKernel(g));
    reg("waveStep", wave.waveStepKernel(g));
    reg("waveBc", wave.waveBcKernel(g));
    reg("integrate", wave.integrateKernel(g));
    reg("radStress", wave.radStressKernel(g));
    reg("force", wave.forceKernel(g));
    reg("forceBc", wave.forceBcKernel(g));
    reg("stokes", wave.stokesKernel(g));
    reg("breakerDelay", wave.breakerDelayKernel(g));
    reg("stokesBc", wave.stokesBcKernel(g));
    reg("qbAdvect", wave.qbAdvectKernel(g));
    reg("statBc", bc.statBcKernel(g));
    reg("meanAcc", bc.meanAccKernel(g));
    reg("waterlineAcc", bc.waterlineAccKernel(g));
    reg("bichromBc", bc.bichromBcKernel(g));
    reg("bcRowApply", bc.bcRowApplyKernel(g));
    reg("dtStep", bc.dtStepKernel(g));
    reg("spectralBc", bc.spectralBcKernel(g));
    reg("dtwReduce", bc.dtwReduceKernel(g));
    reg("dtwSink", bc.dtwSinkKernel(g));
    reg("errReduce", bc.errReduceKernel(g));
    if (this.p.single_dir === 1) {
      // OCE-252: the directions sweep — the stationary kernels compiled for
      // ntheta_s bins on the ee_s set, depths from hhws (common.ts DIRS)
      const gs = { ...g, ntheta: Math.max(this.grid.ntheta_s, 1), dirs: true };
      reg("hhwsUpdate", wave.hhwsUpdateKernel(g));
      reg("dirBcApplyS", bc.dirBcApplyKernel(gs));
      reg("dispersionS", wave.dispersionKernel(gs));
      reg("slopesS", wave.slopesKernel(gs));
      reg("dirvelS", wave.dirvelKernel(gs));
      reg("waveStepS", wave.waveStepKernel(gs));
      reg("waveBcS", wave.waveBcKernel(gs));
      reg("integrateS", wave.integrateKernel(gs));
      reg("dtwReduceS", bc.dtwReduceKernel(gs));
      reg("dtwSinkS", bc.dtwSinkKernel(gs));
      reg("errReduceS", bc.errReduceKernel(gs));
    }
    // dt kernels are needed regardless of flow on/off (timestep.F90 always
    // computes the CFL from wet u/v points)
    reg("dtFlow", flow.dtFlowKernel(g, this.p));
    reg("dtTheta", flow.dtThetaKernel(g, this.p));
    if (this.p.flow === 1) {
      reg("bedrough", flow.bedroughKernel(g, this.p));
      reg("flowAdvec", flow.flowAdvecKernel(g, this.p));
      reg("nuhK", flow.nuhKernel(g, this.p));
      reg("nuhBc", flow.nuhBcKernel(g, this.p));
      reg("visc", flow.viscKernel(g, this.p));
      reg("viscBc", flow.viscBcKernel(g, this.p));
      reg("momentum", flow.momentumKernel(g, this.p));
      reg("flowLatBc", flow.flowLatBcKernel(g, this.p));
      reg("flowLatBc2", flow.flowLatBc2Kernel(g, this.p));
      reg("huhv", flow.huhvKernel(g, this.p));
      reg("continuity", flow.continuityKernel(g, this.p));
      reg("flowPost1", flow.flowPost1Kernel(g, this.p));
      reg("flowPost2", flow.flowPost2Kernel(g, this.p));
      reg("flowBc", flow.flowBcKernel(g, this.p));
      reg("umean", flow.umeanKernel(g, this.p));
    }
  }

  /** flow_bc (start of step, after wave_bc). factime for umean smoothing. */
  applyFlowBc(enc?: GPUCommandEncoder) {
    if (enc) {
      this.dispatch(enc, "umean", 1);
      this.dispatch(enc, "flowBc", this.grid.ny + 1 + this.grid.nx + 1);
      return;
    }
    this.writeUniforms();
    this.submit((e) => {
      this.dispatch(e, "umean", 1);
      this.dispatch(e, "flowBc", this.grid.ny + 1 + this.grid.nx + 1);
    });
  }

  /** flow_timestep.F90 subroutine flow, in order. */
  flowStep(encIn?: GPUCommandEncoder) {
    if (encIn) {
      this.flowDispatches(encIn);
      return;
    }
    this.writeUniforms();
    this.submit((enc) => {
      this.flowDispatches(enc);
    });
  }

  private flowDispatches(enc: GPUCommandEncoder) {
    this.dispatch(enc, "flowAdvec", this.ncell);
    this.dispatch(enc, "bedrough", this.ncell);
    this.dispatch(enc, "nuhK", this.ncell);
    this.dispatch(enc, "nuhBc", this.grid.ny + 1 + this.grid.nx + 1);
    this.dispatch(enc, "visc", this.ncell);
    this.dispatch(enc, "viscBc", this.grid.nx + 1);
    this.dispatch(enc, "momentum", this.ncell);
    this.dispatch(enc, "flowLatBc", this.grid.nx + 1);
    this.dispatch(enc, "flowLatBc2", 1);
    this.dispatch(enc, "huhv", this.ncell);
    this.dispatch(enc, "continuity", this.ncell);
    this.dispatch(enc, "flowPost1", this.ncell);
    this.dispatch(enc, "flowPost2", this.ncell);
  }

  /**
   * timestep.F90:compute_dt + timestep: CFL dt via GPU reduction, chopped so
   * an integer number of steps reaches tnext. Advances t.
   */
  async advanceTime(tnext: number): Promise<number> {
    const p = this.p;
    if (this.t <= 0) {
      // conservative estimate (timestep.F90:462-470) — host-side
      let mind = Infinity, maxhh = 0;
      const gr = this.grid;
      const n1 = gr.nx + 1;
      for (let j = Math.min(1, gr.ny); j <= Math.max(gr.ny - 1, 0); j++) {
        for (let i = 1; i <= gr.nx - 1; i++) {
          mind = Math.min(mind, gr.dsu[i + j * n1]);
          if (gr.ny > 0) mind = Math.min(mind, gr.dnv[i + j * n1]);
          const hh0 = Math.max(-gr.zb[i + j * n1] + p.zs0, p.eps);
          maxhh = Math.max(maxhh, hh0);
        }
      }
      this.dt = (p.CFL * mind) / Math.sqrt(maxhh * p.g);
    } else {
      this.gpu.upload(this.red, new Uint32Array([0x7f7fffff, 0, 0, 0x7f7fffff]));
      this.writeUniforms();
      this.submit((enc) => {
        if (this.includeWetcellsInDt) {
          this.dispatch(enc, "wetcellsA", this.ncell);
          this.dispatch(enc, "wetcellsB", this.ncell);
        }
        this.dispatch(enc, "dtFlow", this.ncell);
        if (p.swave === 1 && p.wavemodel !== WAVEMODEL.stationary && this.grid.ntheta > 1) {
          this.dispatch(enc, "dtTheta", this.ncell);
        }
      });
      const r = await this.gpu.read(this.red, 4 * 4);
      this.dt = r[0] * p.CFL * 0.5;
      if (p.swave === 1 && p.wavemodel !== WAVEMODEL.stationary && this.grid.ntheta > 1) {
        this.dt = Math.min(this.dt, p.CFL * r[3]);
      }
      // note: par%dtset is read but unused in trunk XBeach (dead parameter)
    }
    if (tnext > this.t) {
      const n = Math.ceil((tnext - this.t) / this.dt);
      this.dt = (tnext - this.t) / n;
    }
    this.t += this.dt;
    if (this.t >= tnext - 1e-9) {
      this.dt -= this.t - tnext;
      this.t = tnext;
    }
    // publish to the GPU-resident time slots read by all kernels
    this.gpu.upload(this.scratch, new Float32Array([this.dt, this.t]), 16 * 4);
    return this.dt;
  }

  /**
   * gpudt mode: run `k` fully GPU-resident steps in one submit. Per step:
   * wetcells -> CFL reductions -> dtStep (dt/t on GPU) -> wave BC -> flow BC
   * -> wave -> flow -> mean accumulation. Steps past tnext are identities
   * (dt=0). Host learns t only via readGpuT() afterwards.
   * bcKind: "bichrom" | "spectral" | "none".
   */
  gpuBatch(k: number, tnext: number, bcKind: string, withMeans: boolean) {
    this.gpu.upload(this.scratch, new Float32Array([tnext]), 23 * 4);
    this.writeUniforms({ statMode: 0 });
    const enc = this.gpu.device.createCommandEncoder();
    for (let s = 0; s < k; s++) {
      this.dispatch(enc, "wetcellsA", this.ncell);
      this.dispatch(enc, "wetcellsB", this.ncell);
      this.dispatch(enc, "dtFlow", this.ncell);
      if (
        this.p.swave === 1 && this.p.wavemodel !== WAVEMODEL.stationary &&
        this.grid.ntheta > 1
      ) {
        this.dispatch(enc, "dtTheta", this.ncell);
      }
      this.dispatch(enc, "dtStep", 1);
      if (bcKind === "bichrom") this.dispatch(enc, "bichromBc", this.grid.ny + 1);
      else if (bcKind === "spectral") {
        this.dispatch(
          enc,
          "spectralBc",
          (this.grid.ny + 1) * this.grid.ntheta + this.grid.ny + 1,
        );
      }
      if (this.p.flow === 1) {
        this.dispatch(enc, "umean", 1);
        this.dispatch(enc, "flowBc", this.grid.ny + 1 + this.grid.nx + 1);
      }
      this.waveDispatches(enc);
      if (this.p.flow === 1) this.flowDispatches(enc);
      if (withMeans) {
        this.dispatch(enc, "meanAcc", this.ncell);
        this.dispatch(enc, "waterlineAcc", this.ncell);
      }
    }
    this.gpu.device.queue.submit([enc.finish()]);
  }

  /** read the GPU-resident model time (scratch[17]). */
  private windFillU: Float32Array<ArrayBuffer> | null = null;
  private windFillV: Float32Array<ArrayBuffer> | null = null;
  /** W1.2 (E9's wind sibling): set the (spatially uniform) wind mid-run.
   * Same convention as the windv/windth params (nautical degrees; the
   * readwind.F90 world->local projection is repeated here per cell, so
   * curvilinear grids stay correct). Zero speed clears the stress. */
  setWind(windvMs: number, windthDeg: number): void {
    this.p.windv = windvMs;
    this.p.windth = windthDeg;
    if (!this.windFillU || this.windFillU.length !== this.ncell) {
      this.windFillU = new Float32Array(this.ncell);
      this.windFillV = new Float32Array(this.ncell);
    }
    const winddir = (270 - windthDeg - this.p.alfa) * Math.PI / 180;
    const windx = Math.cos(winddir) * windvMs;
    const windy = Math.sin(winddir) * windvMs;
    const gr = this.grid;
    for (let k = 0; k < this.ncell; k++) {
      this.windFillU[k] = windx * Math.cos(gr.alfau[k]) +
        windy * Math.sin(gr.alfau[k]);
      this.windFillV![k] = windy * Math.cos(gr.alfav[k] - Math.PI / 2) -
        windx * Math.sin(gr.alfav[k] - Math.PI / 2);
    }
    const q = this.gpu.device.queue;
    q.writeBuffer(this.s2, s2Offset("windsu", this.ncell) * 4, this.windFillU);
    q.writeBuffer(this.s2, s2Offset("windnv", this.ncell) * 4, this.windFillV!);
  }

  private tideFill: Float32Array<ArrayBuffer> | null = null;
  /** E9: time-varying tide — set the (spatially uniform) still-water
   * level mid-run, the XBeach zs0file/tideloc<=1 semantics. The
   * absorbing boundary and flow shaders read S2_zs0 every step, so a
   * rising or falling level propagates into the domain as flow; the
   * host mirror p.zs0 follows for diagnostics. Callers drive it from
   * their own step loop (the app worker / CLI runner own the clock). */
  setTide(zs0M: number): void {
    this.p.zs0 = zs0M;
    if (!this.tideFill || this.tideFill.length !== this.ncell) {
      this.tideFill = new Float32Array(this.ncell);
    }
    this.tideFill.fill(zs0M);
    this.gpu.device.queue.writeBuffer(
      this.s2, s2Offset("zs0", this.ncell) * 4, this.tideFill);
  }

  async readGpuT(): Promise<number> {
    const staging = this.gpu.device.createBuffer({
      size: 8,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = this.gpu.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.scratch, 16 * 4, staging, 0, 8);
    this.gpu.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const v = new Float32Array(staging.getMappedRange().slice(0));
    staging.destroy();
    this.dt = v[0];
    this.t = v[1];
    return this.t;
  }

  /** upload a spectral boundary condition's E/q record tables (gpudt mode;
   * the sync CLI path uploads too, for the Es table). OCE-252: with
   * single_dir the condition carries esData, the stationary boundary spread
   * ee_s(0,:,:) ((ny+1)*ntheta_s, j fastest), appended after q and applied
   * by applyDirectionsBc. A single_dir run without it cannot run. */
  setSpectralBnd(cond: {
    eData: Float64Array;
    qData: Float64Array;
    dtbc: number;
    nrec: number;
    esData?: Float64Array;
  }) {
    const e = Float32Array.from(cond.eData);
    const q = Float32Array.from(cond.qData);
    let es: Float32Array | null = null;
    if (this.p.single_dir === 1) {
      const need = (this.grid.ny + 1) * this.grid.ntheta_s;
      if (!cond.esData || cond.esData.length < need) {
        throw new Error("single_dir=1 needs the stationary boundary spread (Es table) " +
          "with the boundary condition — generate the tables with single_dir=1 or " +
          "reuse a single_dir Fortran run's esbcflist.bcf");
      }
      es = Float32Array.from(cond.esData.subarray(0, need));
    }
    const total = e.length + q.length + (es ? es.length : 0);
    if (this.bnd.size < total * 4) {
      this.bnd.destroy();
      this.bnd = this.gpu.buffer("bnd", total * 4);
      this.bindCache.clear();
    }
    this.gpu.upload(this.bnd, e, 0);
    this.gpu.upload(this.bnd, q, e.length * 4);
    if (es) {
      this.gpu.upload(this.bnd, es, (e.length + q.length) * 4);
      this.gpu.upload(this.scratch, new Float32Array([e.length + q.length]), 27 * 4);
    }
    this.gpu.upload(
      this.scratch,
      new Float32Array([cond.dtbc, cond.nrec, e.length]),
      19 * 4,
    );
  }

  /** which: uniform parity copy (0/1); setS: bind the single_dir directions
   * set (c3s/a3s/b3s/v3s) in place of the instationary 3-D buffers. */
  private bindFor(_name: string, which = 0, setS = false): GPUBindGroup {
    const key = `${which === 0 ? "shared" : "shared2"}${setS ? ":S" : ""}`;
    let bg = this.bindCache.get(key);
    if (!bg) {
      bg = this.gpu.device.createBindGroup({
        layout: this.gpu.bglLayout!,
        entries: [
          { binding: 0, resource: { buffer: which === 0 ? this.uni : this.uni2 } },
          { binding: 1, resource: { buffer: this.c2 } },
          { binding: 2, resource: { buffer: this.s2 } },
          { binding: 3, resource: { buffer: setS ? this.c3s : this.c3 } },
          { binding: 4, resource: { buffer: setS ? this.a3s : this.a3 } },
          { binding: 5, resource: { buffer: setS ? this.b3s : this.b3 } },
          { binding: 6, resource: { buffer: setS ? this.v3s : this.v3 } },
          { binding: 7, resource: { buffer: this.scratch } },
          { binding: 8, resource: { buffer: this.red } },
          { binding: 9, resource: { buffer: this.bnd } },
        ],
      });
      this.bindCache.set(key, bg);
    }
    return bg;
  }

  /** Update uniform buffer. */
  writeUniforms(over: Partial<Record<string, number>> = {}) {
    const p = this.p, gr = this.grid;
    const u = this.uniData;
    const vals: Record<string, number> = {
      dt: this.dt, Trep: p.Trep, sigmval: 2 * Math.PI / p.Trep, g: p.g, rho: p.rho,
      rhog8: p.rhog8, px: Math.PI, eps: p.eps, gamma: p.gamma, gamma2: p.gamma2,
      ngam: p.n, alpha: p.alpha, gammax: p.gammax, Trepinv: 1 / p.Trep,
      fwcutoff: p.fwcutoff, beta: p.beta, t: this.t, wci: p.wci, dtheta: gr.dtheta,
      scheme: p.scheme === SCHEME.warmbeam ? 4 : p.scheme === SCHEME.upwind_2 ? 3 : 1,
      breakmodel: p.break, roller: p.roller, refraction: p.refraction, deltaH: p.delta,
      rollergammax: p.rollergammax, snells: p.snells, singledir: p.single_dir,
      theta0: gr.theta0, lateralwave: p.lateralwave, oldRD1D: p.oldRD1D,
      dtw: 0, rowI: -1, statMode: 0, spare: -1,
      ...over,
    };
    const order = [
      "dt", "Trep", "sigmval", "g", "rho", "rhog8", "px", "eps", "gamma", "gamma2",
      "ngam", "alpha", "gammax", "Trepinv", "fwcutoff", "beta", "t", "wci", "dtheta",
      "scheme", "breakmodel", "roller", "refraction", "deltaH", "rollergammax",
      "snells", "singledir", "theta0", "lateralwave", "oldRD1D",
      "dtw", "rowI", "statMode", "spare",
    ];
    order.forEach((k, idx) => u[idx] = vals[k] ?? 0);
    this.gpu.upload(this.uni, u);
    const u2 = Float32Array.from(u);
    u2[order.indexOf("spare")] = 1; // parity 1
    this.gpu.upload(this.uni2, u2);
  }

  dispatch(enc: GPUCommandEncoder, name: string, threads: number, which = 0, setS = false) {
    const pass = enc.beginComputePass({ label: name });
    pass.setPipeline(this.kernels.get(name)!);
    pass.setBindGroup(0, this.bindFor(name, which, setS));
    const [gx, gy] = groups(threads);
    pass.dispatchWorkgroups(gx, gy);
    pass.end();
  }

  submit(build: (enc: GPUCommandEncoder) => void) {
    const enc = this.gpu.device.createCommandEncoder();
    build(enc);
    this.gpu.device.queue.submit([enc.finish()]);
  }

  /** swap ping-pong 3D state (copy B->A). */
  private swap3(enc: GPUCommandEncoder) {
    enc.copyBufferToBuffer(this.b3, 0, this.a3, 0, this.a3.size);
  }
  private swap3s(enc: GPUCommandEncoder) {
    enc.copyBufferToBuffer(this.b3s, 0, this.a3s, 0, this.a3s.size);
  }

  /** bichromatic surfbeat BC. Llong must be set once via setBichromLlong(). */
  applyBichromBc(enc?: GPUCommandEncoder) {
    if (enc) {
      this.dispatch(enc, "bichromBc", this.grid.ny + 1);
      return;
    }
    this.writeUniforms();
    this.submit((e) => this.dispatch(e, "bichromBc", this.grid.ny + 1));
  }

  /** Llong = Tlong·cg(1,1) at BC creation (boundaryconditions.F90:297). */
  async setBichromLlong() {
    // run dispersion once so cg is valid, then read cg(0,0). OCE-378: the
    // wet/dry masks must exist first — the dispersion kernel gives a cell
    // with wete = 0 the dry placeholder cg = sqrt(g·eps) (0.3 m/s), and
    // before this fix the callers ran this before wetcells, so Llong came
    // out as ~20 m instead of ~600 m. The alongshore phase of the groups,
    // sin(theta0)·(y − y0)/Llong, then spun with every row for any oblique
    // sea (shore-normal seas have sin(theta0) = 0 and never showed it):
    // the Newport hour read H_mean 7–10 % above Fortran in the breaking
    // band, rip2d at 270° matched to 0.03 %.
    this.writeUniforms();
    this.submit((enc) => {
      this.dispatch(enc, "wetcellsA", this.ncell);
      this.dispatch(enc, "wetcellsB", this.ncell);
      this.dispatch(enc, "wavePrep", this.ncell);
      this.dispatch(enc, "dispersion", this.ncell);
    });
    const cg = await this.readField("cg");
    if (!(cg[0] > 1)) {
      throw new Error(`bichromatic boundary: cg(1,1) = ${cg[0]} m/s — the offshore corner is dry or the masks are missing`);
    }
    const Llong = (this.p.Tlong ?? 80) * cg[0];
    this.gpu.upload(
      this.scratch,
      new Float32Array([Llong, this.p.Tlong ?? 80, this.p.lwave * (this.p.order - 1)]),
      10 * 4,
    );
  }

  /** spectral reuse BC: upload interpolated boundary row and apply. */
  applyWaveBcRow(eeRow: Float32Array, uiRow: Float32Array, viRow: Float32Array, enc?: GPUCommandEncoder) {
    const m1 = this.grid.ny + 1;
    const nee = m1 * this.grid.ntheta;
    const buf = new Float32Array(nee + 2 * m1);
    buf.set(eeRow, 0);
    buf.set(uiRow, nee);
    buf.set(viRow, nee + m1);
    const off = (1024 + 3 * this.ncell) * 4;
    this.gpu.upload(this.scratch, buf, off);
    if (enc) {
      this.dispatch(enc, "bcRowApply", nee + m1);
      return;
    }
    this.writeUniforms();
    this.submit((e) => this.dispatch(e, "bcRowApply", nee + m1));
  }

  /** stationary wave BC (wbctype=params): fill row 0 (taper from GPU t). */
  applyStatWaveBc() {
    this.gpu.upload(this.scratch, new Float32Array([this.dt, this.t]), 16 * 4);
    this.writeUniforms();
    this.submit((enc) => {
      this.dispatch(enc, "statBc", (this.grid.ny + 1) * this.grid.ntheta);
    });
  }

  /**
   * Stationary wave solve: whole-grid pseudo-time relaxation of the identical
   * discrete equations (see docs/deviations.md#stationary-solver).
   */
  async solveStationary(): Promise<{ iters: number; herr: number; terr: number }> {
    const p = this.p;
    // prep: hhw/hstokes, dispersion, slopes/thetamean, dirvel
    this.writeUniforms({ statMode: 1 });
    this.submit((enc) => {
      this.dispatch(enc, "wavePrep", this.ncell);
      this.dispatch(enc, "dispersion", this.ncell);
      this.dispatch(enc, "slopes", this.ncell);
      this.dispatch(enc, "dirvel", this.n3);
    });
    // per-row dtw from CFL: red[8+i]; red[0..3] = global dtw/herr/terr
    const redInit = new Uint32Array(8 + this.grid.nx + 1 + 4);
    redInit.fill(0x7f7fffff, 8);
    redInit[0] = 0x7f7fffff;
    this.gpu.upload(this.red, redInit);
    this.submit((enc) => this.dispatch(enc, "dtwReduce", this.ncell));

    const maxIters = Math.max(6 * (this.grid.nx + 1), p.maxiter * 4);
    const checkEvery = 8;
    let herr = Infinity, terr = Infinity;
    let iters = 0;
    while (iters < maxIters) {
      this.writeUniforms({ statMode: 1 });
      // reset per-row dtw to CFL values, then sink-limit from current state
      // (mirrors Fortran's fresh dtw per row-solve rather than a monotonic
      // global shrink that early transients would poison)
      this.gpu.upload(this.red, redInit);
      this.submit((enc) => {
        this.dispatch(enc, "dtwReduce", this.ncell);
        for (let s = 0; s < checkEvery; s++) {
          this.dispatch(enc, "dtwSink", this.ncell);
          // red-black Gauss-Seidel in x: damps the odd-even mode a plain
          // Jacobi sweep leaves undamped in the roller balance
          this.dispatch(enc, "waveStep", this.ncell, 0);
          this.swap3(enc);
          this.dispatch(enc, "waveStep", this.ncell, 1);
          this.dispatch(
            enc,
            "waveBc",
            (this.grid.ny + 1 + 2 * (this.grid.nx + 1)) * this.grid.ntheta,
          );
          this.dispatch(enc, "integrate", this.ncell);
          this.swap3(enc);
        }
      });
      iters += checkEvery;
      // errors (herr/terr slots reset each check; per-row dtw kept monotonic)
      this.gpu.upload(this.red, new Uint32Array([0, 0, 0]), 4);
      this.submit((enc) => this.dispatch(enc, "errReduce", this.ncell));
      const r = await this.gpu.read(this.red, 4 * 4);
      herr = r[1];
      terr = r[2];
      const rerr = r[3];
      if (herr < p.maxerror && terr < p.maxerror_angle && rerr < p.maxerror) break;
    }
    // finalize: forces, urms, stokes (like callTypeStationary tail)
    this.writeUniforms({ statMode: 1 });
    this.submit((enc) => {
      this.dispatch(enc, "radStress", this.ncell);
      this.dispatch(enc, "force", this.ncell);
      this.dispatch(enc, "forceBc", this.grid.ny + 1 + this.grid.nx + 1);
      this.dispatch(enc, "stokes", this.ncell);
      if (this.p.breakerdelay > 0) this.dispatch(enc, "breakerDelay", this.ncell);
      this.dispatch(enc, "stokesBc", this.grid.ny + 1 + this.grid.nx + 1);
    });
    return { iters, herr, terr };
  }

  /**
   * OCE-252 single_dir: the stationary directions sweep (wave_timestep.F90:84-92,
   * wave_stationary_directions callTypeDirections) — ee_s on the ntheta_s grid
   * relaxed to the same fixed point as solveStationary, on the wavint-smoothed
   * depth hhws, no roller; its product is thetamean, which the one-bin
   * instationary step then propagates along. Call before the instationary
   * step at t = dt, on every wavint hit, and on a new boundary condition
   * (newstatbc). The instationary dispatches that follow recompute dispersion,
   * E, H, D, Qb on hhw, so the sweep's overwrites of those fields are
   * transient — as in Fortran.
   */
  async solveDirections(): Promise<{ iters: number; herr: number; terr: number }> {
    const p = this.p, gr = this.grid;
    if (p.single_dir !== 1) throw new Error("solveDirections: single_dir is off");
    const over = { statMode: 1, dtheta: gr.dtheta_s, roller: 0 };
    this.writeUniforms(over);
    this.submit((enc) => {
      this.dispatch(enc, "wavePrep", this.ncell); // hhw current for the breaker
      this.dispatch(enc, "dirBcApplyS", (gr.ny + 1) * gr.ntheta_s, 0, true);
      this.dispatch(enc, "dispersionS", this.ncell, 0, true);
      this.dispatch(enc, "slopesS", this.ncell, 0, true);
      this.dispatch(enc, "dirvelS", this.n3s, 0, true);
    });
    const redInit = new Uint32Array(8 + gr.nx + 1 + 4);
    redInit.fill(0x7f7fffff, 8);
    redInit[0] = 0x7f7fffff;
    this.gpu.upload(this.red, redInit);
    this.submit((enc) => this.dispatch(enc, "dtwReduceS", this.ncell, 0, true));
    // the budget Fortran gives one row (par%maxiter, default 500), at least
    // six grid widths so the boundary reaches the shore: on a real coast a
    // few shoreline cells never settle to maxerror in either engine (Fortran
    // logs "Wave propagation row n, iteration 500" on them), and the
    // directions field is not worth four times that
    const maxIters = Math.max(6 * (gr.nx + 1), p.maxiter);
    const checkEvery = 8;
    let herr = Infinity, terr = Infinity;
    let iters = 0;
    while (iters < maxIters) {
      this.writeUniforms(over);
      this.gpu.upload(this.red, redInit);
      this.submit((enc) => {
        this.dispatch(enc, "dtwReduceS", this.ncell, 0, true);
        for (let s = 0; s < checkEvery; s++) {
          this.dispatch(enc, "dtwSinkS", this.ncell, 0, true);
          this.dispatch(enc, "waveStepS", this.ncell, 0, true);
          this.swap3s(enc);
          this.dispatch(enc, "waveStepS", this.ncell, 1, true);
          this.dispatch(enc, "waveBcS",
            (gr.ny + 1 + 2 * (gr.nx + 1)) * gr.ntheta_s, 0, true);
          this.dispatch(enc, "integrateS", this.ncell, 0, true);
          this.swap3s(enc);
        }
      });
      iters += checkEvery;
      this.gpu.upload(this.red, new Uint32Array([0, 0, 0]), 4);
      this.submit((enc) => this.dispatch(enc, "errReduceS", this.ncell, 0, true));
      const r = await this.gpu.read(this.red, 4 * 4);
      herr = r[1];
      terr = r[2];
      if (herr < p.maxerror && terr < p.maxerror_angle) break;
    }
    this.writeUniforms({ statMode: 0 });
    return { iters, herr, terr };
  }

  /** One surfbeat wave step (wave_instationary). Assumes dt set. */
  waveInstationaryStep(encIn?: GPUCommandEncoder) {
    if (encIn) {
      this.waveDispatches(encIn);
      return;
    }
    this.writeUniforms({ statMode: 0 });
    this.submit((enc) => {
      this.waveDispatches(enc);
    });
  }

  private waveDispatches(enc: GPUCommandEncoder) {
    this.dispatch(enc, "wavePrep", this.ncell);
    // OCE-252: the directions model's smoothed depth follows hhw every step
    // (update_means_wave_flow); the instationary dispersion stays on hhw
    if (this.p.single_dir === 1) this.dispatch(enc, "hhwsUpdate", this.ncell);
    this.dispatch(enc, "dispersion", this.ncell);
    this.dispatch(enc, "slopes", this.ncell);
    this.dispatch(enc, "dirvel", this.n3);
    if (this.p.break === BREAK.roelvink_daly) {
      this.dispatch(enc, "qbAdvect", this.ncell);
    }
    this.dispatch(enc, "waveStep", this.ncell);
    this.dispatch(
      enc,
      "waveBc",
      (this.grid.ny + 1 + 2 * (this.grid.nx + 1)) * this.grid.ntheta,
    );
    this.dispatch(enc, "integrate", this.ncell);
    this.swap3(enc);
    this.dispatch(enc, "radStress", this.ncell);
    this.dispatch(enc, "force", this.ncell);
    this.dispatch(enc, "forceBc", this.grid.ny + 1 + this.grid.nx + 1);
    this.dispatch(enc, "stokes", this.ncell);
    if (this.p.breakerdelay > 0) this.dispatch(enc, "breakerDelay", this.ncell);
    this.dispatch(enc, "stokesBc", this.grid.ny + 1 + this.grid.nx + 1);
  }

  /** Read a 2D field back to host (f32). */
  async readField(f: S2Field): Promise<Float32Array> {
    const off = s2Offset(f, this.ncell) * 4;
    const staging = this.gpu.device.createBuffer({
      size: this.ncell * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = this.gpu.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.s2, off, staging, 0, this.ncell * 4);
    this.gpu.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.destroy();
    return out;
  }

  /** Several S2 fields in ONE staging buffer and one map — the app's live
   * frame reads two to five fields every ~80 ms, and each separate
   * readField is its own buffer, copy and mapAsync round trip. */
  async readFields(fs: readonly S2Field[]): Promise<Float32Array[]> {
    if (fs.length === 0) return [];
    const n = this.ncell, bytes = n * 4;
    const staging = this.gpu.device.createBuffer({
      size: bytes * fs.length,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = this.gpu.device.createCommandEncoder();
    fs.forEach((f, i) => enc.copyBufferToBuffer(this.s2, s2Offset(f, n) * 4, staging, i * bytes, bytes));
    this.gpu.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const all = new Float32Array(staging.getMappedRange().slice(0));
    staging.destroy();
    return fs.map((_, i) => all.slice(i * n, (i + 1) * n));
  }

  /** accumulate dt-weighted means (call after flow each step). */
  accumulateMeans(enc?: GPUCommandEncoder) {
    if (enc) {
      this.dispatch(enc, "meanAcc", this.ncell);
      this.dispatch(enc, "waterlineAcc", this.ncell);
      return;
    }
    this.submit((e) => {
      this.dispatch(e, "meanAcc", this.ncell);
      this.dispatch(e, "waterlineAcc", this.ncell);
    });
  }

  /**
   * Read the waterline gauge (OCE-243): per CELL, the time the cell was the
   * instantaneous waterline (s), and over that time the mean, standard
   * deviation and maximum of the water level there, in metres above the
   * cell's own bed (zb; the host adds zb back). NaN where the cell never
   * was the waterline. Accumulated over the whole run — never reset by
   * readMeans — so it is a run-long statistic, read at the end.
   */
  async readWaterline(): Promise<{ T: Float32Array; mean: Float32Array; std: Float32Array; max: Float32Array }> {
    const n = this.ncell;
    const bcrow = (this.grid.ny + 1) * Math.max(this.grid.ntheta, 1) + 2 * (this.grid.ny + 1);
    const off = (1024 + 3 * n + bcrow + bc.NMEAN_SLOTS * n) * 4;
    const staging = this.gpu.device.createBuffer({
      size: bc.NWL_SLOTS * n * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = this.gpu.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.scratch, off, staging, 0, bc.NWL_SLOTS * n * 4);
    this.gpu.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const a = new Float32Array(staging.getMappedRange().slice(0));
    staging.destroy();
    const T = new Float32Array(n), mean = new Float32Array(n), std = new Float32Array(n), max = new Float32Array(n);
    for (let k = 0; k < n; k++) {
      const t = a[3 * n + k];
      T[k] = t;
      if (t > 1e-3) {
        mean[k] = a[k] / t;
        std[k] = Math.sqrt(Math.max(a[n + k] / t - mean[k] * mean[k], 0));
        max[k] = a[2 * n + k];
      } else { mean[k] = NaN; std[k] = NaN; max[k] = NaN; }
    }
    return { T, mean, std, max };
  }

  /** read mean accumulators and reset them (peek=true keeps accumulating).
   * The reset clears ALL slots, including the |u|max slot: max-reduction
   * windows restart from 0 exactly like the dt-weighted sums. */
  async readMeans(peek = false): Promise<Float32Array> {
    const bcrow = (this.grid.ny + 1) * Math.max(this.grid.ntheta, 1) + 2 * (this.grid.ny + 1);
    const off = (1024 + 3 * this.ncell + bcrow) * 4;
    const staging = this.gpu.device.createBuffer({
      size: bc.NMEAN_SLOTS * this.ncell * 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = this.gpu.device.createCommandEncoder();
    enc.copyBufferToBuffer(this.scratch, off, staging, 0, bc.NMEAN_SLOTS * this.ncell * 4);
    if (!peek) enc.clearBuffer(this.scratch, off, bc.NMEAN_SLOTS * this.ncell * 4);
    this.gpu.device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.destroy();
    return out;
  }

  /** wetcells pass (start of each step). */
  wetcells() {
    this.writeUniforms();
    this.submit((enc) => {
      this.dispatch(enc, "wetcellsA", this.ncell);
      this.dispatch(enc, "wetcellsB", this.ncell);
    });
  }
}
