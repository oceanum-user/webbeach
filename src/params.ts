// SPDX-License-Identifier: LGPL-3.0-or-later
// XBeach WebGPU solver — derived from XBeach (c) Deltares (LGPL v2.1+).
// Copyright (c) 2026 Oceanum. See LICENSE.
/**
 * XBeach parameter handling with keyword/default/conditional parity.
 *
 * Mirrors trunk/src/xbeachlibrary/params.F90 (subroutine all_input) for the
 * in-scope subset: physical processes, grid, time, constants, initial
 * conditions, wave BCs, spectral BCs, flow BCs, tide, breaking, roller, WCI,
 * flow, Coriolis, wind, wave numerics, flow numerics, output.
 *
 * Out of scope (sediment, morphology, non-hydrostatic, groundwater,
 * vegetation, ships, drifters, rain, q3d): keywords are accepted but must be
 * off/default; setting them on is an error.
 */

// --- name-list constants, mirroring paramsconst.F90 ---------------------------

export const WAVEMODEL = { stationary: 0, surfbeat: 1, nonh: 2 } as const;
export const WBCTYPE = {
  params: 0,
  parametric: 1,
  swan: 2,
  vardens: 3,
  off: 4,
  jonstable: 5,
  reuse: 6,
  ts_1: 7,
  ts_2: 8,
  ts_nonh: 9,
} as const;
export const BREAK = {
  roelvink1: 1,
  baldock: 2,
  roelvink2: 3,
  roelvink_daly: 4,
  janssen: 5,
} as const;
export const SCHEME = { upwind_1: 1, lax_wendroff: 2, upwind_2: 3, warmbeam: 4 } as const;
export const BEDFRICTION = {
  chezy: 1,
  cf: 2,
  "white-colebrook": 3,
  manning: 4,
  "white-colebrook-grainsize": 5,
} as const;
export const FRONT = { abs_1d: 0, abs_2d: 1, wall: 2, wlevel: 3, nonh_1d: 4, waveflume: 5 } as const;
export const LR = { neumann: 0, wall: 1, no_advec: 2, neumann_v: 3, abs_1d: 4 } as const;
export const BACK = { wall: 0, abs_1d: 1, abs_2d: 2, wlevel: 3 } as const;
export const LATERALWAVE = { neumann: 0, wavecrest: 1, cyclic: 2 } as const;
export const TIDETYPE = { instant: 0, velocity: 1, hybrid: 2 } as const;
export const PAULREVERE = { land: 0, sea: 1 } as const;
export const TRUNCATIONTYPE = { absolute: 0, relative: 1, absolute_ee: 2, relative_ee: 3 } as const;
export const INSTAT = {
  stat: 0,
  bichrom: 1,
  ts_1: 2,
  ts_2: 3,
  jons: 4,
  swan: 5,
  vardens: 6,
  reuse: 7,
  ts_nonh: 8,
  off: 9,
  stat_table: 10,
  jons_table: 11,
} as const;

// ------------------------------------------------------------------------------

export interface RawParams {
  [key: string]: string;
}

/** Parse a params.txt file body into raw key -> string values. */
export function parseParamsTxt(text: string): RawParams {
  const raw: RawParams = {};
  for (const line0 of text.split(/\r?\n/)) {
    const line = line0.trim();
    if (!line || line.startsWith("%") || line.startsWith("#") || line.startsWith("!")) continue;
    const eq = line.indexOf("=");
    if (eq < 0) continue;
    const key = line.slice(0, eq).trim();
    const val = line.slice(eq + 1).trim();
    if (key) raw[key.toLowerCase()] = val;
  }
  return raw;
}

export class ParamError extends Error {}

/** Fully-resolved in-scope parameter set (XBeach names). */
export interface Params {
  // processes
  wavemodel: number;
  cyclic: number;
  swave: number;
  single_dir: number;
  lwave: number;
  flow: number;
  viscosity: number;
  advection: number;
  wind: number;
  refraction: number;
  // grid
  xori: number;
  yori: number;
  alfa: number;
  nx: number;
  ny: number;
  posdwn: number;
  depfile: string;
  vardx: number;
  dx: number;
  dy: number;
  xfile: string;
  yfile: string;
  thetamin: number;
  thetamax: number;
  thetanaut: number;
  dtheta: number;
  dtheta_s: number;
  // time
  CFL: number;
  dtset: number;
  tstop: number;
  maxdtfac: number;
  defuse: number;
  // constants
  rho: number;
  g: number;
  depthscale: number;
  px: number;
  rhog8: number;
  // initial conditions
  zs0: number;
  zsinitfile: string;
  hotstartflow: number;
  // wave bc
  wbctype: number;
  instat: number | null;
  taper: number;
  Hrms: number;
  Trep: number;
  Tm01: number;
  Tlong: number | null;
  dir0: number;
  nmax: number;
  m: number;
  lateralwave: number;
  bcfile: string;
  random: number;
  fcutoff: number;
  nspr: number;
  trepfac: number;
  sprdthr: number;
  correctHm0: number;
  Tm01switch: number;
  rt: number;
  dtbc: number;
  dthetaS_XB: number;
  nspectrumloc: number;
  wbcversion: number;
  wbcEvarreduce: number;
  wbcQvarreduce: number;
  wbcScaleEnergy: number;
  wbcRemoveStokes: number;
  Sfold: number;
  bclwonly: number;
  sprdthr_set: boolean;
  // flow bc
  front: number;
  left: number;
  right: number;
  back: number;
  ARC: number;
  order: number;
  highcomp: number;
  freewave: number;
  epsi: number;
  nc: number;
  tidetype: number;
  // tide
  tideloc: number;
  paulrevere: number;
  zs0file: string;
  // breaking
  break: number;
  gamma: number;
  gamma2: number;
  alpha: number;
  n: number;
  gammax: number;
  rollergammax: number;
  delta: number;
  deltahmin: number;
  fw: number;
  fwfile: string;
  fwcutoff: number;
  breakerdelay: number;
  shoaldelay: number;
  facsd: number;
  // roller
  roller: number;
  beta: number;
  rfb: number;
  // wci
  wci: number;
  hwci: number;
  hwcimax: number;
  cats: number;
  // flow params
  bedfriction: number;
  bedfriccoef: number;
  bedfricfile: string;
  maxcf: number;
  mincf: number;
  nuh: number;
  nuhfac: number;
  nuhv: number;
  smag: number;
  // coriolis + wind
  wearth: number;
  lat: number;
  rhoa: number;
  Cd: number;
  windv: number;
  windth: number;
  windfile: string;
  // wave numerics
  scheme: number;
  wavint: number;
  maxerror: number;
  maxerror_angle: number; // radians after resolve (input in degrees)
  maxiter: number;
  truncationtype: number;
  snells: number;
  oldhmin: number;
  oldRD1D: number;
  // flow numerics
  eps: number;
  eps_sd: number;
  umin: number;
  hmin: number;
  secorder: number;
  oldhu: number;
  maxfacg: number;
  // output
  tstart: number;
  tintg: number;
  tintp: number;
  tintm: number;
  nglobalvar: number;
  globalvars: string[];
  nmeanvar: number;
  meanvars: string[];
  npointvar: number;
  pointvars: string[];
  npoints: number;
  pointcoords: Array<[number, number]>;
  outputformat: string;
}

const OUT_OF_SCOPE_ON = [
  "sedtrans",
  "morphology",
  "avalanching",
  "gwflow",
  "q3d",
  "swrunup",
  "ships",
  "vegetation",
  "setbathy",
  "rainfall",
  "nonh",
  // sediment-only: short-wave stirring in the pickup terms and the
  // skewness/asymmetry advection velocity ua (morphevolution.F90:1612,
  // :3039). Inert without sedtrans — the Fortran itself zeroes it when the
  // sediment block is off (params.F90:987) — but it IS a recognised XBeach
  // name, so it belongs here and is refused as out of scope rather than
  // reported as a misspelling. It had been silently exempted from the guard
  // below, which is the only reason sws=1 ever ran.
  "sws",
];

class Reader {
  used = new Set<string>();
  constructor(public raw: RawParams) {}

  has(name: string): boolean {
    const k = name.toLowerCase();
    // probing counts as consuming: the out-of-scope guard and the
    // conditional defaults reach keys this way, and anything left unconsumed
    // is an error (see the unrecognised-name check at the end of resolveParams)
    if (k in this.raw) {
      this.used.add(k);
      return true;
    }
    return false;
  }
  private get(name: string): string | undefined {
    const k = name.toLowerCase();
    if (k in this.raw) {
      this.used.add(k);
      return this.raw[k];
    }
    return undefined;
  }
  int(name: string, def: number, min: number, max: number): number {
    const v = this.get(name);
    if (v === undefined) return def;
    const x = parseInt(v, 10);
    if (isNaN(x)) throw new ParamError(`${name}: not an integer: '${v}'`);
    if (x < min || x > max) throw new ParamError(`${name}=${x} outside [${min},${max}]`);
    return x;
  }
  dbl(name: string, def: number, min: number, max: number): number {
    const v = this.get(name);
    if (v === undefined) return def;
    const x = parseFloat(v.replace(/d/i, "e"));
    if (isNaN(x)) throw new ParamError(`${name}: not a number: '${v}'`);
    if (x < min || x > max) {
      // XBeach clamps with warning for non-strict; error keeps us honest:
      throw new ParamError(`${name}=${x} outside [${min},${max}]`);
    }
    return x;
  }
  str(name: string, def = ""): string {
    const v = this.get(name);
    return v === undefined ? def : v;
  }
  /** parmapply equivalent: named options with old numeric aliases. */
  name(
    name: string,
    allowed: Record<string, number>,
    defIndex: number, // 1-based index into insertion order, as parmapply
    oldnames?: string[],
  ): number {
    const keys = Object.keys(allowed);
    const v = this.get(name);
    if (v === undefined) return allowed[keys[defIndex - 1]];
    const vl = v.toLowerCase();
    if (vl in allowed) return allowed[vl as keyof typeof allowed];
    if (oldnames) {
      const io = oldnames.indexOf(vl);
      if (io >= 0) return allowed[keys[io]];
    }
    throw new ParamError(`${name}: unknown value '${v}' (allowed: ${keys.join(", ")})`);
  }
  /** Keys a different layer owns — the CLI parses the multi-line variable
   * lists itself, so they are legitimately never read through the Reader. */
  markUsed(...names: string[]): void {
    for (const n of names) this.used.add(n.toLowerCase());
  }
  required(name: string): void {
    if (!this.has(name)) throw new ParamError(`missing required parameter '${name}'`);
  }
}

/**
 * Resolve raw params to the full parameter set, mirroring params.F90 all_input
 * conditional-default logic (in-scope subset).
 */
/** OCE-215 (R3.6): one entry of the resolved parameter set, under the
 * keyword name XBeach uses, with whether the params file set it or the
 * solver's default applied. */
export interface ParamRecord {
  name: string;
  value: number | string;
  set: boolean;
}

/** fields of Params that are derived or internal, not XBeach keywords */
const NOT_KEYWORDS = new Set(["rhog8", "sprdthr_set", "px"]);

/** keywords whose value is a named choice, stored as the table's code */
const KEYWORD_TABLES: Record<string, Record<string, number>> = {
  wavemodel: WAVEMODEL, wbctype: WBCTYPE, instat: INSTAT, break: BREAK, scheme: SCHEME,
  bedfriction: BEDFRICTION, front: FRONT, left: LR, right: LR, back: BACK,
  lateralwave: LATERALWAVE, tidetype: TIDETYPE, truncationtype: TRUNCATIONTYPE,
};

/** the name a params.txt would write for a coded choice, else the value */
function keywordValue(name: string, v: number | string): number | string {
  const t = KEYWORD_TABLES[name];
  if (!t || typeof v !== "number") return v;
  const hit = Object.entries(t).find(([, code]) => code === v);
  return hit ? hit[0] : v;
}

/**
 * OCE-215 (R3.6): the EFFECTIVE parameter set of a run — every scalar the
 * solver resolved, named as in params.txt, so a result can be reproduced
 * from what ships with it. `set` is true for a keyword the file carried
 * (the app's or the user's choice); false means the XBeach default applied.
 * Array-valued output lists (globalvars…) are left out; they shape output,
 * not physics. A coded choice (break, bedfriction, wavemodel…) is reported
 * by the name params.txt takes, never the internal code.
 */
export function describeParams(raw: RawParams, p: Params): ParamRecord[] {
  const out: ParamRecord[] = [];
  for (const [name, v] of Object.entries(p)) {
    if (NOT_KEYWORDS.has(name)) continue;
    if (typeof v !== "number" && typeof v !== "string") continue;
    if (typeof v === "string" && v === "") continue; // unset file names
    out.push({ name, value: keywordValue(name, v), set: name.toLowerCase() in raw });
  }
  return out;
}

export function resolveParams(raw: RawParams): Params {
  const r = new Reader(raw);
  const px = 4 * Math.atan(1);

  // reject out-of-scope processes switched on
  for (const k of OUT_OF_SCOPE_ON) {
    if (r.has(k) && parseInt(raw[k], 10) === 1) {
      throw new ParamError(`parameter '${k}=1' is out of scope for the WebGPU model`);
    }
  }

  // --- wavemodel + instat backward compatibility (check_instat_backward_compatibility)
  let wavemodel = -123;
  let wbctype = -123;
  let instat: number | null = null;
  let TlongDefault: number | null = null;
  if (r.has("instat")) {
    instat = r.name("instat", INSTAT, 2, [
      "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "40", "41",
    ]);
    switch (instat) {
      case INSTAT.stat:
        wavemodel = WAVEMODEL.stationary;
        wbctype = WBCTYPE.params;
        break;
      case INSTAT.ts_1:
        wavemodel = WAVEMODEL.surfbeat;
        wbctype = WBCTYPE.ts_1;
        break;
      case INSTAT.ts_2:
        wavemodel = WAVEMODEL.surfbeat;
        wbctype = WBCTYPE.ts_2;
        break;
      case INSTAT.off:
        wavemodel = WAVEMODEL.surfbeat;
        wbctype = WBCTYPE.off;
        break;
      case INSTAT.bichrom:
        wavemodel = WAVEMODEL.surfbeat;
        wbctype = WBCTYPE.params;
        TlongDefault = 80;
        break;
      case INSTAT.stat_table:
        wavemodel = WAVEMODEL.stationary;
        wbctype = WBCTYPE.jonstable;
        break;
      case INSTAT.jons:
        wavemodel = WAVEMODEL.surfbeat;
        wbctype = WBCTYPE.parametric;
        break;
      case INSTAT.jons_table:
        wavemodel = WAVEMODEL.surfbeat;
        wbctype = WBCTYPE.jonstable;
        break;
      case INSTAT.swan:
        wavemodel = WAVEMODEL.surfbeat;
        wbctype = WBCTYPE.swan;
        break;
      case INSTAT.vardens:
        wavemodel = WAVEMODEL.surfbeat;
        wbctype = WBCTYPE.vardens;
        break;
      case INSTAT.reuse:
        wavemodel = WAVEMODEL.surfbeat;
        wbctype = WBCTYPE.reuse;
        break;
      default:
        throw new ParamError(`instat option not supported in WebGPU model`);
    }
  }
  if (r.has("wavemodel")) {
    wavemodel = r.name("wavemodel", WAVEMODEL, 2, ["0", "1", "2"]);
  }
  if (wavemodel === -123) {
    throw new ParamError("Set wavemodel to stationary or surfbeat (nonh unsupported)");
  }
  if (wavemodel === WAVEMODEL.nonh) {
    throw new ParamError("wavemodel=nonh is out of scope for the WebGPU model");
  }

  // --- physical processes
  const cyclic = r.int("cyclic", 0, 0, 1);
  const swave = r.int("swave", 1, 0, 1);
  const lwave = r.int("lwave", 1, 0, 1);
  const flow = r.int("flow", 1, 0, 1);
  const viscosity = r.int("viscosity", 1, 0, 1);
  const advection = r.int("advection", 1, 0, 1);
  const wind = r.int("wind", 0, 0, 1);
  const refraction = r.int("refraction", 1, 0, 1);

  // --- grid (gridform=xbeach only)
  if (r.has("gridform") && raw["gridform"].toLowerCase() !== "xbeach") {
    throw new ParamError("only gridform=xbeach supported");
  }
  const xori = r.dbl("xori", 0, -1e9, 1e9);
  const yori = r.dbl("yori", 0, -1e9, 1e9);
  const alfa = r.dbl("alfa", 0, 0, 360);
  r.required("nx");
  r.required("ny");
  const nx = r.int("nx", 50, 2, 10000);
  const ny = r.int("ny", 2, 0, 10000);
  const posdwn = r.dbl("posdwn", 1, -1, 1);
  const depfile = r.str("depfile");
  const vardx = r.int("vardx", 1, 0, 1);
  let dx = -1, dy = -1, xfile = "", yfile = "";
  if (vardx === 0) {
    r.required("dx");
    r.required("dy");
    dx = r.dbl("dx", -1, 0, 1e9);
    dy = r.dbl("dy", -1, 0, 1e9);
  } else {
    dx = r.dbl("dx", -1, 0, 1e9);
    dy = r.dbl("dy", -1, 0, 1e9);
    xfile = r.str("xfile");
    yfile = r.str("yfile");
  }

  // wave directional grid
  let thetamin = -90, thetamax = 90, thetanaut = 0, dtheta = 10, dtheta_s = 10;
  let single_dir = 0;
  if (swave === 1) {
    r.required("thetamin");
    r.required("thetamax");
    thetamin = r.dbl("thetamin", -90, -360, 360);
    thetamax = r.dbl("thetamax", 90, -360, 360);
    thetanaut = r.int("thetanaut", 0, 0, 1);
    if (wavemodel === WAVEMODEL.surfbeat) {
      single_dir = r.int("single_dir", ny === 0 ? 0 : 1, 0, 1);
    }
    if (single_dir === 1) {
      // OCE-252: one instationary bin spanning the window; the stationary
      // directions model runs on its own dtheta_s grid (params.F90:260-263).
      // A dtheta in the file is read and ignored, as Fortran does ("Unknown,
      // unused" in its log), so a directional case switches over by setting
      // single_dir and dtheta_s alone
      if (r.has("dtheta")) {
        r.dbl("dtheta", 10, 0.1, 180);
        console.warn("single_dir=1: dtheta is ignored (the instationary bin spans thetamin..thetamax)");
      }
      dtheta = thetamax - thetamin;
      r.required("dtheta_s");
      dtheta_s = r.dbl("dtheta_s", 10, 0.1, 20);
    } else {
      r.required("dtheta");
      dtheta = r.dbl("dtheta", 10, 0.1, 180);
    }
  }

  // --- model time
  const CFL = ny === 0 ? r.dbl("CFL", 0.95, 0.1, 1.0) : r.dbl("CFL", 0.7, 0.1, 0.9);
  const dtset = r.dbl("dtset", 0, 0.001, 100);
  r.required("tstop");
  const tstop = r.dbl("tstop", 2000, 1, 1000000);
  const defuse = r.int("defuse", 1, 0, 1);
  const maxdtfac = r.dbl("maxdtfac", 50, 10, 200);

  // --- physical constants
  const rho = r.dbl("rho", 1025, 1000, 1040);
  const g = r.dbl("g", 9.81, 9.7, 9.9);
  const depthscale = r.dbl("depthscale", 1, 1, 200);

  // --- initial conditions
  const zsinitfile = r.str("zsinitfile");
  const hotstartflow = r.int("hotstartflow", 0, 0, 1);

  // --- wave boundary condition parameters
  if (r.has("wbctype")) {
    wbctype = r.name("wbctype", WBCTYPE, 6, [
      "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    ]);
  }
  if (wbctype === -123) {
    throw new ParamError("XBeach cannot run without a type of boundary condition: set wbctype");
  }
  const bcfile = r.str("bcfile");
  const taper = r.dbl("taper", 100, 0, 1000);
  const nmax = r.dbl("nmax", 0.8, 0.5, 1);
  // read unconditionally, as params.F90:447-448 (used by the TS wavemaker:
  // bclwonly skips short-wave E, Sfold selects Sfinterpq vs Sfinterp in qbcf)
  const bclwonly = r.int("bclwonly", 0, 0, 1);
  const Sfold = r.int("Sfold", 1, 0, 1);
  let Hrms = 1, Tm01 = 10, Trep = 10, dir0 = 270, m = 10;
  if (wbctype === WBCTYPE.params || single_dir === 1) {
    Hrms = r.dbl("Hrms", 1, 0, 10);
    Tm01 = r.dbl("Tm01", 10, 1, 20);
    Trep = r.dbl("Trep", Tm01, 1, 20);
    dir0 = r.dbl("dir0", 270, -360, 360);
    m = r.int("m", 10, 2, 128);
  } else if (wbctype === WBCTYPE.ts_1 || wbctype === WBCTYPE.ts_2) {
    Tm01 = r.dbl("Tm01", 10, 1, 20);
    Trep = r.dbl("Trep", Tm01, 1, 20);
    dir0 = r.dbl("dir0", 270, -360, 360);
    m = r.int("m", 10, 2, 128);
  }
  let Tlong: number | null = TlongDefault;
  if (r.has("tlong")) Tlong = r.dbl("Tlong", 80, 20, 300);
  const lateralwave = r.name("lateralwave", LATERALWAVE, 1, ["0", "1"]);
  if (lateralwave !== LATERALWAVE.neumann) {
    const lwname = lateralwave === LATERALWAVE.wavecrest ? "wavecrest" : "cyclic";
    throw new ParamError(
      `lateralwave=${lwname} not yet implemented in WebGPU model (use neumann)`,
    );
  }

  // --- wave-spectrum boundary condition parameters
  const isSpectral = wbctype === WBCTYPE.parametric || wbctype === WBCTYPE.swan ||
    wbctype === WBCTYPE.vardens || wbctype === WBCTYPE.jonstable;
  let random = 1, fcutoff = 0, nspr = 0, trepfac = 0.01, sprdthr = 0.08;
  let correctHm0 = 1, Tm01switch = 0, rt = Math.min(3600, tstop), dtbc = 1.0;
  let dthetaS_XB = 0, nspectrumloc = 1, wbcversion = 3;
  let wbcEvarreduce = 1, wbcQvarreduce = 1, wbcScaleEnergy = 1, wbcRemoveStokes = 1;
  if (isSpectral) {
    random = r.int("random", 1, 0, 1);
    fcutoff = r.dbl("fcutoff", 0, 0, 0.025);
    nspr = r.int("nspr", 0, 0, 1);
    trepfac = r.dbl("trepfac", 0.01, 0, 1);
    sprdthr = r.dbl("sprdthr", 0.08, 0, 0.15);
    correctHm0 = r.int("correctHm0", 1, 0, 1);
    Tm01switch = r.int("Tm01switch", 0, 0, 1);
    rt = r.dbl("rt", Math.min(3600, tstop), 1200, 7200);
    dtbc = r.dbl("dtbc", 1.0, 0.1, 2.0);
    if (wbctype === WBCTYPE.swan) dthetaS_XB = r.dbl("dthetaS_XB", 0, -360, 360);
    wbcversion = r.int("wbcversion", 3, 1, 3);
    nspectrumloc = wbcversion > 2 ? r.int("nspectrumloc", 1, 1, ny + 1) : 1;
    wbcEvarreduce = ny === 0
      ? r.dbl("wbcEvarreduce", 0.3, 0, 1)
      : r.dbl("wbcEvarreduce", 1, 0, 1);
    wbcQvarreduce = r.dbl("wbcQvarreduce", 1, 0, 1);
    wbcScaleEnergy = r.int("wbcScaleEnergy", 1, 0, 1);
    wbcRemoveStokes = r.int("wbcRemoveStokes", 1, 0, 1);
  }

  // --- flow boundary conditions
  const front = r.name("front", FRONT, ny > 2 ? 2 : 1, ["0", "1", "2", "3", "4", "5"]);
  const left = r.name("left", LR, 1, ["0", "1"]);
  const right = r.name("right", LR, 1, ["0", "1"]);
  const back = r.name("back", BACK, ny > 2 ? 3 : 2, ["0", "1", "2", "3"]);
  if (front === FRONT.wlevel || back === BACK.wlevel) {
    throw new ParamError("wlevel boundary no longer supported (matches Fortran error)");
  }
  // params.F90:1731-1737, 1862-1874: automatic front/back reductions
  let frontR = front, backR = back;
  if (frontR !== FRONT.abs_1d && wavemodel === WAVEMODEL.stationary) frontR = FRONT.abs_1d;
  if (frontR === FRONT.abs_2d && ny < 3) frontR = FRONT.abs_1d;
  if (backR === BACK.abs_2d && ny < 3) backR = BACK.abs_1d;
  const ARC = r.int("ARC", 1, 0, 1);
  const order = r.dbl("order", 2, 1, 2);
  const highcomp = r.int("highcomp", 0, 0, 1);
  const freewave = r.int("freewave", 0, 0, 1);
  const epsi = r.dbl("epsi", -1, -1, 0.2);
  const nc = r.int("nc", ny + 1, 1, ny + 1);
  const tidetype = r.name("tidetype", TIDETYPE, 2);

  // --- tide
  const tideloc = r.has("zs0") ? r.int("tideloc", 0, 0, 4) : r.int("tideloc", 2, 0, 4);
  let paulrevere: number = PAULREVERE.land;
  let zs0file = "";
  let zs0 = 0;
  if (tideloc > 0) {
    if (tideloc >= 2) {
      // These were accepted and then run as ONE uniform level read from the
      // first zs column — quietly a different model from the one the params
      // asked for. Only tideloc=1 is ported, so say so (paulrevere is read
      // only on this path and goes with it).
      throw new ParamError(
        `tideloc=${tideloc} (spatially varying tide) is not implemented — ` +
        "only tideloc=1 (uniform level) is ported");
    }
    zs0file = r.str("zs0file");
    if (!zs0file) throw new ParamError("tideloc>0 requires zs0file (or set zs0 for constant level)");
  } else {
    zs0 = r.dbl("zs0", 0, -5, 5);
  }

  // --- wave breaking + roller + wci
  let brk: number = BREAK.baldock, gamma = 0.78, gamma2 = 0.34, gammax = 0.6;
  let rollergammax = 1, alpha = 1.38, n = 10, delta = 0, deltahmin = 0.1;
  let fw = 0, fwfile = "", fwcutoff = 1000, breakerdelay = 1, shoaldelay = 0, facsd = 1;
  let roller = 1, rfb = 0, beta = 0.08;
  let wci = 0, hwci = 0.1, hwcimax = 100, cats = 4;
  if (swave === 1) {
    if (wavemodel === WAVEMODEL.stationary) {
      brk = r.name("break", BREAK, 2, ["1", "2", "3", "4", "5"]);
      if (brk !== BREAK.baldock && brk !== BREAK.janssen) {
        throw new ParamError(
          "Roelvink formulations not allowed in stationary, use Baldock or Janssen",
        );
      }
      gamma = r.dbl("gamma", 0.78, 0.4, 0.9);
      gammax = r.dbl("gammax", 0.6, 0.4, 5);
    } else {
      brk = r.name("break", BREAK, 4, ["1", "2", "3", "4", "5"]);
      if (brk === BREAK.roelvink_daly) {
        gamma = r.dbl("gamma", 0.46, 0.35, 0.6);
        gamma2 = r.dbl("gamma2", 0.34, 0.0, 0.5);
      } else {
        gamma = r.dbl("gamma", 0.55, 0.4, 0.9);
      }
      gammax = r.dbl("gammax", 2, 1.5, 3);
    }
    rollergammax = r.int("rollergammax", 1, 0, 1);
    alpha = r.dbl("alpha", 1.38, 0.5, 2.0);
    n = r.dbl("n", 10, 5, 20);
    delta = r.dbl("delta", 0, 0, 1);
    deltahmin = r.dbl("deltahmin", 0.1, 0.05, 0.3);
    fw = r.dbl("fw", 0, 0, 1);
    fwfile = r.str("fwfile");
    // OCE-288: the per-cell file was read as a name and ignored — the run
    // went on with the constant fw
    if (fwfile) throw new ParamError("fwfile (wave friction per cell) is not implemented — use the constant fw");
    fwcutoff = r.dbl("fwcutoff", 1000, 0, 1000);
    breakerdelay = r.dbl("breakerdelay", 1, 0, 3);
    shoaldelay = r.int("shoaldelay", 0, 0, 1);
    if (shoaldelay === 1) throw new ParamError("shoaldelay=1 not yet implemented in WebGPU model");
    if (shoaldelay === 1) facsd = r.dbl("facsd", 1, 0, 2);
    roller = r.int("roller", 1, 0, 1);
    rfb = r.int("rfb", 0, 0, 1);
    if (rfb === 1) throw new ParamError("rfb=1 not yet implemented in WebGPU model");
    beta = r.dbl("beta", 0.08, 0.05, 0.3);
    wci = r.int("wci", 0, 0, 1);
    if (wci === 1) {
      throw new ParamError("wci=1 (wave-current interaction) is not implemented in this engine");
    }
    hwci = r.dbl("hwci", 0.1, 0.001, 1);
    hwcimax = r.dbl("hwcimax", 100, 0.01, 100);
    cats = r.dbl("cats", 4, 1, 50);
  } else {
    beta = r.dbl("beta", 0.08, 0.05, 0.3);
  }

  // --- flow parameters
  const bedfriction = r.name("bedfriction", BEDFRICTION, 4);
  if (bedfriction === BEDFRICTION["white-colebrook-grainsize"]) {
    throw new ParamError("bedfriction=white-colebrook-grainsize needs sediment D90 (out of scope)");
  }
  if (!r.has("bedfriction") && r.has("bedfricfile")) {
    throw new ParamError("BEDFRICFILE without BEDFRICTION is not allowed");
  }
  const bedfricfile = r.str("bedfricfile");
  // OCE-288: the file was never loaded and the coefficient stayed at -999
  if (bedfricfile) {
    throw new ParamError("bedfricfile (bed friction per cell) is not implemented — use the constant bedfriccoef");
  }
  let bedfriccoef = -999;
  if (!bedfricfile) {
    if (bedfriction === BEDFRICTION.chezy) bedfriccoef = r.dbl("bedfriccoef", 55, 20, 100);
    else if (bedfriction === BEDFRICTION.cf) bedfriccoef = r.dbl("bedfriccoef", 3e-3, 1e-3, 0.1);
    else if (bedfriction === BEDFRICTION.manning) {
      bedfriccoef = r.dbl("bedfriccoef", 0.02, 0.01, 0.05);
    } else if (bedfriction === BEDFRICTION["white-colebrook"]) {
      bedfriccoef = r.dbl("bedfriccoef", 0.01, 3.5e-5, 0.9);
    }
  }
  const maxcf = r.dbl("maxcf", 0.04, 0, 0.1);
  const mincf = r.dbl("mincf", 0, 0, 0.1);
  const nuh = r.dbl("nuh", 0.1, 0, 1);
  const nuhfac = r.dbl("nuhfac", 0, 0, 1);
  const nuhv = r.dbl("nuhv", 1, 1, 20);
  const smag = r.int("smag", 1, 0, 1);

  // --- coriolis + wind
  const wearth = r.dbl("wearth", 1 / 24, 0, 1);
  const lat = r.dbl("lat", 0, -90, 90);
  let rhoa = 1.25, Cd = 0.002, windv = 0, windth = 270, windfile = "";
  if (wind === 1) {
    rhoa = r.dbl("rhoa", 1.25, 1, 2);
    Cd = r.dbl("Cd", 0.002, 0.0001, 0.01);
    windfile = r.str("windfile");
    if (windfile) {
      // Accepting this silently was a wrong-answer path: windv stayed 0 and
      // the momentum kernels gate wind stress on windv > 0, so `wind=1` plus
      // a windfile ran with NO wind and said nothing. A wind SERIES reaches
      // the engine through Model.setWind (the app drives it per frame);
      // there is no in-engine windfile reader to fall back on.
      throw new ParamError(
        "windfile is not implemented — pass a wind series through setWind, " +
        "or use windv/windth for a constant wind");
    }
    windv = r.dbl("windv", 0, 0, 200);
    windth = r.dbl("windth", 270, -360, 360);
  }

  // --- wave numerics
  const scheme = r.name("scheme", SCHEME, 4, ["1", "2", "3", "4"]);
  if (scheme === SCHEME.lax_wendroff) {
    throw new ParamError("scheme=lax_wendroff not implemented (matches removed Fortran option)");
  }
  let wavint = 600, maxerror = 0.0001, maxerror_angle = (1 / 180) * px, maxiter = 500;
  let truncationtype: number = TRUNCATIONTYPE.absolute;
  if (wavemodel === WAVEMODEL.stationary || single_dir === 1) {
    wavint = r.dbl("wavint", 600, 1, 3600);
    maxerror = r.dbl("maxerror", 0.0001, 0.00001, 0.001);
    maxerror_angle = (r.dbl("maxerror_angle", 1, 0.1, 5) / 180) * px;
    maxiter = r.int("maxiter", 500, 2, 1000);
    truncationtype = r.name("truncationtype", TRUNCATIONTYPE, 1, ["0", "1", "2", "3"]);
  }
  let snells: number;
  if (single_dir === 0) {
    if (ny === 0 && Math.round(Math.abs(thetamax - thetamin) / dtheta) < 2) {
      snells = r.int("snells", 1, 0, 1);
    } else {
      snells = r.int("snells", 0, 0, 1);
    }
  } else {
    snells = r.int("snells", 0, 0, 1);
  }
  const oldhmin = r.int("oldhmin", 0, 0, 1);
  const oldRD1D = r.int("oldRD1D", 0, 0, 1);

  // --- flow numerics
  const eps = r.dbl("eps", 0.005, 0.001, 0.1);
  const eps_sd = r.dbl("eps_sd", 0.5, 0, 1);
  const umin = r.dbl("umin", 0, 0, 0.2);
  const hmin = oldhmin === 1 ? r.dbl("hmin", 0.2, 0.001, 1) : 0;
  const secorder = r.int("secorder", 0, 0, 1);
  if (secorder === 1) throw new ParamError("secorder=1 (MacCormack) not yet implemented");
  const oldhu = r.int("oldhu", 0, 0, 1);
  const maxfacg = r.dbl("maxfacg", 10, 5, 30);

  // --- output
  const tstart = r.dbl("tstart", 0, 0, tstop);
  const tint = r.dbl("tint", 1, 0.01, tstop);
  const tintg = r.dbl("tintg", tint, 0, tstop - tstart);
  const tintp = r.dbl("tintp", 1, 0, tstop - tstart);
  const tintm = r.dbl("tintm", tstop - tstart, 0, tstop - tstart);
  const outputformat = r.str("outputformat", "fortran");

  // nglobalvar / variable lists are parsed by the CLI (multi-line format);
  // reader only accepts inline comma-separated here.
  r.markUsed("nglobalvar", "nmeanvar", "npointvar", "npoints");
  const globalvars = (r.str("globalvars") || "").split(/[\s,]+/).filter(Boolean);
  const meanvars = (r.str("meanvars") || "").split(/[\s,]+/).filter(Boolean);
  const pointvars = (r.str("pointvars") || "").split(/[\s,]+/).filter(Boolean);

  const params: Params = {
    wavemodel, cyclic, swave, single_dir, lwave, flow, viscosity, advection, wind, refraction,
    xori, yori, alfa, nx, ny, posdwn, depfile, vardx, dx, dy, xfile, yfile,
    thetamin, thetamax, thetanaut, dtheta, dtheta_s,
    CFL, dtset, tstop, maxdtfac, defuse,
    rho, g, depthscale, px, rhog8: 0.125 * rho * g,
    zs0, zsinitfile, hotstartflow,
    wbctype, instat, taper, Hrms, Trep, Tm01, Tlong, dir0, nmax, m, lateralwave,
    bcfile, random, fcutoff, nspr, trepfac, sprdthr, correctHm0, Tm01switch, rt, dtbc,
    dthetaS_XB, nspectrumloc, wbcversion, wbcEvarreduce, wbcQvarreduce, wbcScaleEnergy,
    wbcRemoveStokes, Sfold, bclwonly, sprdthr_set: r.has("sprdthr"),
    front: frontR, left, right, back: backR, ARC, order, highcomp, freewave, epsi, nc, tidetype,
    tideloc, paulrevere, zs0file,
    break: brk, gamma, gamma2, alpha, n, gammax, rollergammax, delta, deltahmin,
    fw, fwfile, fwcutoff, breakerdelay, shoaldelay, facsd,
    roller, beta, rfb,
    wci, hwci, hwcimax, cats,
    bedfriction, bedfriccoef, bedfricfile, maxcf, mincf, nuh, nuhfac, nuhv, smag,
    wearth, lat, rhoa, Cd, windv, windth, windfile,
    scheme, wavint, maxerror, maxerror_angle, maxiter, truncationtype, snells, oldhmin, oldRD1D,
    eps, eps_sd, umin, hmin, secorder, oldhu, maxfacg,
    tstart, tintg, tintp, tintm,
    nglobalvar: globalvars.length, globalvars,
    nmeanvar: meanvars.length, meanvars,
    npointvar: pointvars.length, pointvars,
    npoints: 0, pointcoords: [],
    outputformat,
  };
  // Every key has to have been read by someone. A misspelled or unported
  // name sitting inertly in a params file is the same silent-wrong-answer
  // class as an unimplemented parameter accepted quietly: the run proceeds
  // describing a model the file never asked for. Fortran warns here; we have
  // no log to warn into, so we refuse.
  const unknown = Object.keys(raw).filter((k) => !r.used.has(k)).sort();
  if (unknown.length > 0) {
    throw new ParamError(
      `unrecognised parameter${unknown.length > 1 ? "s" : ""}: ` +
        `${unknown.join(", ")} — check the spelling, or the name is not ` +
        "implemented in this engine",
    );
  }
  return params;
}
