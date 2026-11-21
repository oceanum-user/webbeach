// SPDX-License-Identifier: LGPL-3.0-or-later
// XBeach WebGPU solver — derived from XBeach (c) Deltares (LGPL v2.1+).
// Copyright (c) 2026 Oceanum. See LICENSE.
/**
 * Flow module WGSL — port of flow_timestep.F90 (subroutine flow),
 * bedroughness.F90 (chezy/cf/manning/white-colebrook), and
 * boundaryconditions.F90 flow_bc/flow_lat_bc.
 *
 * Static parameters are baked into the WGSL as constants at build time;
 * only time-varying values (dt, t) come from the shared Uni block.
 *
 * Index note: 0-based; Fortran loop ranges translated per
 * spaceparams.F90:ranges_init (see common.ts).
 */

import { preamble, WAVE_UNIFORMS_WGSL, type GridConst } from "./common.ts";
import { BUFFER_DECLS_WGSL, fieldAccessorsWGSL } from "../fields.ts";
import type { Params } from "../params.ts";
import { BACK, BEDFRICTION, FRONT, LR } from "../params.ts";

function header(g: GridConst, p: Params): string {
  const fc = 2 * p.wearth * Math.sin((p.lat * Math.PI) / 180);
  return preamble(g) + WAVE_UNIFORMS_WGSL + BUFFER_DECLS_WGSL + fieldAccessorsWGSL() +
    /* wgsl */ `
const P_G: f32 = ${p.g};
const P_RHO: f32 = ${p.rho};
const P_EPS: f32 = ${p.eps};
const P_EPS_SD: f32 = ${p.eps_sd};
const P_UMIN: f32 = ${p.umin};
const P_NUH: f32 = ${p.nuh};
const P_NUHFAC: f32 = ${p.nuhfac};
const P_NUHV: f32 = ${p.nuhv};
const P_SMAG: i32 = ${p.smag};
const P_VISCOSITY: i32 = ${p.viscosity};
const P_LWAVE: f32 = ${p.lwave};
const P_SWAVE: i32 = ${p.swave};
const P_MAXFACG: f32 = ${p.maxfacg};
const P_FC: f32 = ${fc};
const P_RHOA_CD: f32 = ${p.rhoa * p.Cd};
const P_BEDFRICTION: i32 = ${p.bedfriction};
const P_MAXCF: f32 = ${p.maxcf};
const P_MINCF: f32 = ${p.mincf};
const P_OLDHU: i32 = ${p.oldhu};
const P_DY: f32 = ${p.dy};
const P_FRONT: i32 = ${p.front};
const P_BACK: i32 = ${p.back};
const P_LEFT: i32 = ${p.left};
const P_RIGHT: i32 = ${p.right};
const P_ARC: i32 = ${p.ARC};
const P_ORDER: f32 = ${p.order};
const P_FREEWAVE: i32 = ${p.freewave};
const P_ADVECTION: i32 = ${p.advection};
`;
}

/** cfu/cfv per bedroughness.F90:172-303 (in-scope formulations). */
export function bedroughKernel(g: GridConst, p: Params): string {
  return header(g, p) + /* wgsl */ `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let kk = i32(gid.x);
  if (kk >= NCELL) { return; }
  let coef = C2(C2_bedfriccoef_f, kk);
  var cfu = 0.0;
  var cfv = 0.0;
  if (P_BEDFRICTION == ${BEDFRICTION.chezy}) {
    cfu = P_G / (coef * coef);
    cfv = cfu;
  } else if (P_BEDFRICTION == ${BEDFRICTION.cf}) {
    cfu = coef;
    cfv = coef;
  } else if (P_BEDFRICTION == ${BEDFRICTION.manning}) {
    if (S2(S2_wetu, kk) == 1.0) {
      cfu = clamp(P_G * coef * coef / pow(S2(S2_hu, kk), 1.0 / 3.0), P_MINCF, P_MAXCF);
    } else { cfu = P_MAXCF; }
    if (S2(S2_wetv, kk) == 1.0) {
      cfv = clamp(P_G * coef * coef / pow(S2(S2_hv, kk), 1.0 / 3.0), P_MINCF, P_MAXCF);
    } else { cfv = P_MAXCF; }
  } else { // white-colebrook: kr = coef
    if (S2(S2_wetu, kk) == 1.0) {
      let l = log(12.0 * max(S2(S2_hu, kk), coef) / coef) / log(10.0);
      cfu = clamp(P_G / (18.0 * l * 18.0 * l), P_MINCF, P_MAXCF);
    } else { cfu = P_MAXCF; }
    if (S2(S2_wetv, kk) == 1.0) {
      let l = log(12.0 * max(S2(S2_hv, kk), coef) / coef) / log(10.0);
      cfv = clamp(P_G / (18.0 * l * 18.0 * l), P_MINCF, P_MAXCF);
    } else { cfv = P_MAXCF; }
  }
  S2set(S2_cfu, kk, min(cfu, 1.0));
  S2set(S2_cfv, kk, min(cfv, 1.0));
}`;
}

/**
 * Water level slopes + advection terms (flow_timestep.F90:137-362).
 * One thread per cell writes dzsdx, dzsdy, ududx, vdudy, udvdx, vdvdy.
 */
export function flowAdvecKernel(g: GridConst, p: Params): string {
  return header(g, p) + /* wgsl */ `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let kk = i32(gid.x);
  if (kk >= NCELL) { return; }
  let i = kk % N1;
  let j = kk / N1;

  // --- water level slopes (:137-149) — ph (nonh) omitted
  if (i >= IMIN_UU && i <= IMAX_UU && j >= JMIN_UU && j <= JMAX_UU) {
    S2set(S2_dzsdx, kk, (S2(S2_zs, at(i + 1, j)) - S2(S2_zs, kk)) / C2(C2_dsu, kk));
  }
  if (NY > 0 && i >= IMIN_VV && i <= IMAX_VV && j >= JMIN_VV && j <= JMAX_VV) {
    S2set(S2_dzsdy, kk, (S2(S2_zs, at(i, j + 1)) - S2(S2_zs, kk)) / C2(C2_dnv, kk));
  }

  // --- ududx (:183-211)
  var ududx = 0.0;
  if (P_ADVECTION == 1 && i >= IMIN_UU && i <= IMAX_UU && j >= JMIN_UU && j <= JMAX_UU) {
    let hum = S2(S2_hum, kk);
    let dsdnui = C2(C2_dsdnui, kk);
    // inflow from the left face
    var qin = 0.5 * (S2(S2_qx, kk) + S2(S2_qx, at(i - 1, j)));
    if (qin > 0.0) {
      let dalfa = C2(C2_alfau, kk) - C2(C2_alfau, at(i - 1, j));
      let uin = S2(S2_uu, at(i - 1, j)) * cos(dalfa) + S2(S2_vu, at(i - 1, j)) * sin(dalfa);
      if (S2(S2_uu, kk) - S2(S2_uu, at(i - 1, j)) > P_EPS_SD) {
        ududx += 0.5 * (S2(S2_uu, at(i - 1, j)) + S2(S2_uu, kk)) * (S2(S2_uu, kk) - uin) *
          C2(C2_dnz, kk) * dsdnui;
      } else {
        ududx += qin / hum * (S2(S2_uu, kk) - uin) * C2(C2_dnz, kk) * dsdnui;
      }
    }
    qin = -0.5 * (S2(S2_qx, kk) + S2(S2_qx, at(i + 1, j)));
    if (qin > 0.0) {
      let dalfa = C2(C2_alfau, kk) - C2(C2_alfau, at(i + 1, j));
      let uin = S2(S2_uu, at(i + 1, j)) * cos(dalfa) + S2(S2_vu, at(i + 1, j)) * sin(dalfa);
      if (S2(S2_uu, at(i + 1, j)) - S2(S2_uu, kk) > P_EPS_SD) {
        ududx -= 0.5 * (S2(S2_uu, at(i + 1, j)) + S2(S2_uu, kk)) * (S2(S2_uu, kk) - uin) *
          C2(C2_dnz, at(i + 1, j)) * dsdnui;
      } else {
        ududx += qin / hum * (S2(S2_uu, kk) - uin) * C2(C2_dnz, at(i + 1, j)) * dsdnui;
      }
    }
  }
  S2set(S2_ududx, kk, ududx);

  // --- vdudy (:212-244), 2D only, ranges jmin_vv..jmax_vv x imin_vv..imax_vv
  var vdudy = 0.0;
  if (P_ADVECTION == 1 && NY > 0 && i >= IMIN_VV && i <= IMAX_VV && j >= JMIN_VV && j <= JMAX_VV) {
    let hum = S2(S2_hum, kk);
    let dsdnui = C2(C2_dsdnui, kk);
    var qin = 0.5 * (S2(S2_qy, at(i, j - 1)) + S2(S2_qy, at(i + 1, j - 1)));
    if (qin > 0.0) {
      let dalfa = C2(C2_alfau, kk) - C2(C2_alfau, at(i, j - 1));
      let uin = S2(S2_uu, at(i, j - 1)) * cos(dalfa) + S2(S2_vu, at(i, j - 1)) * sin(dalfa);
      if (S2(S2_vv, kk) - S2(S2_vv, at(i, j - 1)) > P_EPS_SD) {
        vdudy += 0.5 * (S2(S2_vv, at(i, j - 1)) + S2(S2_vv, at(i + 1, j - 1))) *
          (S2(S2_uu, kk) - uin) * C2(C2_dsc, at(i, j - 1)) * dsdnui;
      } else {
        vdudy += qin / hum * (S2(S2_uu, kk) - uin) * C2(C2_dsc, at(i, j - 1)) * dsdnui;
      }
    }
    qin = -0.5 * (S2(S2_qy, kk) + S2(S2_qy, at(i + 1, j)));
    if (qin > 0.0) {
      let dalfa = C2(C2_alfau, kk) - C2(C2_alfau, at(i, j + 1));
      let uin = S2(S2_uu, at(i, j + 1)) * cos(dalfa) + S2(S2_vu, at(i, j + 1)) * sin(dalfa);
      if (S2(S2_vv, at(i, j + 1)) - S2(S2_vv, kk) > P_EPS_SD) {
        vdudy -= 0.5 * (S2(S2_vv, kk) + S2(S2_vv, at(i + 1, j))) * (S2(S2_uu, kk) - uin) *
          C2(C2_dsc, kk) * dsdnui;
      } else {
        vdudy += qin / hum * (S2(S2_uu, kk) - uin) * C2(C2_dsc, kk) * dsdnui;
      }
    }
  }
  S2set(S2_vdudy, kk, vdudy);

  // --- vdvdy (:256-314)
  var vdvdy = 0.0;
  if (P_ADVECTION == 1 && NY > 0) {
    let hvm = S2(S2_hvm, kk);
    let dsdnvi = C2(C2_dsdnvi, kk);
    if (i >= IMIN_VV && i <= IMAX_VV && j >= JMIN_VV && j <= JMAX_VV) {
      var qin = 0.5 * (S2(S2_qy, kk) + S2(S2_qy, at(i, j - 1)));
      if (qin > 0.0) {
        let dalfa = C2(C2_alfav, kk) - C2(C2_alfav, at(i, j - 1));
        let vin = S2(S2_vv, at(i, j - 1)) * cos(dalfa) - S2(S2_uv, at(i, j - 1)) * sin(dalfa);
        if (S2(S2_vv, kk) - S2(S2_vv, at(i, j - 1)) > P_EPS_SD) {
          vdvdy += 0.5 * (S2(S2_vv, at(i, j - 1)) + S2(S2_vv, kk)) * (S2(S2_vv, kk) - vin) *
            C2(C2_dsz, kk) * dsdnvi;
        } else {
          vdvdy += qin / hvm * (S2(S2_vv, kk) - vin) * C2(C2_dsz, kk) * dsdnvi;
        }
      }
      qin = -0.5 * (S2(S2_qy, kk) + S2(S2_qy, at(i, j + 1)));
      if (qin > 0.0) {
        let dalfa = C2(C2_alfav, kk) - C2(C2_alfav, at(i, j + 1));
        let vin = S2(S2_vv, at(i, j + 1)) * cos(dalfa) - S2(S2_uv, at(i, j + 1)) * sin(dalfa);
        if (S2(S2_vv, at(i, j + 1)) - S2(S2_vv, kk) > P_EPS_SD) {
          vdvdy -= 0.5 * (S2(S2_vv, at(i, j + 1)) + S2(S2_vv, kk)) * (S2(S2_vv, kk) - vin) *
            C2(C2_dsz, at(i, j + 1)) * dsdnvi;
        } else {
          vdvdy += qin / hvm * (S2(S2_vv, kk) - vin) * C2(C2_dsz, at(i, j + 1)) * dsdnvi;
        }
      }
    } else if (j == 0 && i >= IMIN_VV && i <= IMAX_VV) {
      // (:292-301) only the negative-qin part with one-sided form
      let qin = -0.5 * (S2(S2_qy, at(i, 0)) + S2(S2_qy, at(i, 1)));
      if (qin > 0.0) {
        let dalfa = C2(C2_alfav, at(i, 0)) - C2(C2_alfav, at(i, 1));
        let vin = S2(S2_vv, at(i, 1)) * cos(dalfa) - S2(S2_uv, at(i, 1)) * sin(dalfa);
        vdvdy += qin * (S2(S2_vv, at(i, 0)) - vin) * C2(C2_dsz, at(i, 1)) / hvm * dsdnvi;
      }
    } else if (j == NY - 1 && NY > 1 && i >= IMIN_VV && i <= IMAX_VV) {
      // (:303-313)
      let qin = 0.5 * (S2(S2_qy, at(i, NY - 1)) + S2(S2_qy, at(i, NY - 2)));
      if (qin > 0.0) {
        let dalfa = C2(C2_alfav, at(i, NY - 1)) - C2(C2_alfav, at(i, NY - 2));
        let vin = S2(S2_vv, at(i, NY - 2)) * cos(dalfa) - S2(S2_uv, at(i, NY - 2)) * sin(dalfa);
        vdvdy += qin * (S2(S2_vv, at(i, NY - 1)) - vin) * C2(C2_dsz, at(i, NY - 1)) / hvm * dsdnvi;
      }
    }
  }
  S2set(S2_vdvdy, kk, vdvdy);

  // --- udvdx (:316-362)
  var udvdx = 0.0;
  if (P_ADVECTION == 1) {
    let hvm = S2(S2_hvm, kk);
    let dsdnvi = C2(C2_dsdnvi, kk);
    if (NY > 0) {
      if (i >= IMIN_VV && i <= IMAX_VV && j >= JMIN_VV && j <= JMAX_VV) {
        var qin = 0.5 * (S2(S2_qx, at(i - 1, j)) + S2(S2_qx, at(i - 1, j + 1)));
        if (qin > 0.0) {
          let dalfa = C2(C2_alfav, kk) - C2(C2_alfav, at(i - 1, j));
          let vin = S2(S2_vv, at(i - 1, j)) * cos(dalfa) - S2(S2_uv, at(i - 1, j)) * sin(dalfa);
          if (S2(S2_uu, kk) - S2(S2_uu, at(i - 1, j)) > P_EPS_SD) {
            udvdx += 0.5 * (S2(S2_uu, at(i - 1, j)) + S2(S2_uu, at(i - 1, j + 1))) *
              (S2(S2_vv, kk) - vin) * C2(C2_dnc, at(i - 1, j)) * dsdnvi;
          } else {
            udvdx += qin / hvm * (S2(S2_vv, kk) - vin) * C2(C2_dnc, at(i - 1, j)) * dsdnvi;
          }
        }
        qin = -0.5 * (S2(S2_qx, kk) + S2(S2_qx, at(i, j + 1)));
        if (qin > 0.0) {
          let dalfa = C2(C2_alfav, kk) - C2(C2_alfav, at(i + 1, j));
          let vin = S2(S2_vv, at(i + 1, j)) * cos(dalfa) - S2(S2_uv, at(i + 1, j)) * sin(dalfa);
          if (S2(S2_uu, at(i + 1, j)) - S2(S2_uu, kk) > P_EPS_SD) {
            udvdx -= 0.5 * (S2(S2_uu, kk) + S2(S2_uu, at(i, j + 1))) * (S2(S2_vv, kk) - vin) *
              C2(C2_dnc, kk) * dsdnvi;
          } else {
            udvdx += qin / hvm * (S2(S2_vv, kk) - vin) * C2(C2_dnc, kk) * dsdnvi;
          }
        }
      }
    } else if (j == 0 && i >= IMIN_VV && i <= IMAX_VV) {
      // 1D variant (:347-362)
      var qin = S2(S2_qx, at(i - 1, 0));
      if (qin > 0.0) {
        let dalfa = C2(C2_alfav, at(i, 0)) - C2(C2_alfav, at(i - 1, 0));
        let vin = S2(S2_vv, at(i - 1, 0)) * cos(dalfa) - S2(S2_uv, at(i - 1, 0)) * sin(dalfa);
        udvdx += qin * (S2(S2_vv, kk) - vin) * C2(C2_dnc, at(i - 1, 0)) / hvm * dsdnvi;
      }
      qin = -S2(S2_qx, at(i, 0));
      if (qin > 0.0) {
        let dalfa = C2(C2_alfav, at(i, 0)) - C2(C2_alfav, at(i + 1, 0));
        let vin = S2(S2_vv, at(i + 1, 0)) * cos(dalfa) - S2(S2_uv, at(i + 1, 0)) * sin(dalfa);
        udvdx += qin * (S2(S2_vv, kk) - vin) * C2(C2_dnc, at(i, 0)) / hvm * dsdnvi;
      }
    }
  }
  S2set(S2_udvdx, kk, udvdx);

  // --- viscosity velocity gradients (:153-174), stored via scratch-free reuse:
  // dudx -> reuse field slot? computed on the fly in nuh kernel instead.
}`;
}

/** nuh field (:366-405 + visc_smagorinsky :990-1100). */
export function nuhKernel(g: GridConst, p: Params): string {
  return header(g, p) + /* wgsl */ `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let kk = i32(gid.x);
  if (kk >= NCELL) { return; }
  let i = kk % N1;
  let j = kk / N1;
  if (P_VISCOSITY == 0) { S2set(S2_nuh, kk, 0.0); return; }
  var nuh = P_NUH;
  if (P_SMAG == 1) {
    var inRange = false;
    var Tau = 0.0;
    var l = 0.0;
    if (NY > 2) {
      if (i >= IMIN_ZS && i <= IMAX_ZS && j >= JMIN_ZS && j <= JMAX_ZS) {
        inRange = true;
        let dudx = (S2(S2_uu, kk) - S2(S2_uu, at(i - 1, j))) / C2(C2_dsz, kk);
        let dudy = 0.5 *
          (S2(S2_uu, at(i, j + 1)) - S2(S2_uu, at(i, j - 1)) + S2(S2_uu, at(i - 1, j + 1)) -
            S2(S2_uu, at(i - 1, j - 1))) / (C2(C2_dnv, kk) + C2(C2_dnv, at(i, j - 1)));
        let dvdx = 0.5 *
          (S2(S2_vv, at(i + 1, j)) - S2(S2_vv, at(i - 1, j)) + S2(S2_vv, at(i + 1, j - 1)) -
            S2(S2_vv, at(i - 1, j - 1))) / (C2(C2_dsu, kk) + C2(C2_dsu, at(i - 1, j)));
        let dvdy = (S2(S2_vv, kk) - S2(S2_vv, at(i, j - 1))) / C2(C2_dnz, kk);
        Tau = sqrt(2.0 * dudx * dudx + 2.0 * dvdy * dvdy + (dvdx + dudy) * (dvdx + dudy));
        l = 1.0 / C2(C2_dsdnzi, kk);
        nuh = P_NUH * P_NUH * l * Tau *
          S2(S2_wetu, kk) * S2(S2_wetu, at(i - 1, j)) * S2(S2_wetv, kk) *
          S2(S2_wetv, at(i, max(j - 1, 0)));
      } else { nuh = 0.0; }
    } else {
      let jj = max(NY, 0) * 0; // 1D: j row 0
      if (i >= IMIN_ZS && i <= IMAX_ZS && j == 0) {
        inRange = true;
        let dudx = (S2(S2_uu, kk) - S2(S2_uu, at(i - 1, 0))) / C2(C2_dsz, kk);
        let dvdx = (S2(S2_vv, at(i + 1, 0)) - S2(S2_vv, at(i - 1, 0))) /
          (C2(C2_dsu, kk) + C2(C2_dsu, at(i - 1, 0)));
        Tau = sqrt(2.0 * dudx * dudx + dvdx * dvdx);
        if (P_DY > -1.0) { l = C2(C2_dsz, kk) * P_DY; } else {
          l = C2(C2_dsz, kk) * C2(C2_dsz, kk);
        }
        nuh = P_NUH * P_NUH * l * Tau * S2(S2_wetu, kk) * S2(S2_wetu, at(i - 1, 0));
      } else { nuh = 0.0; }
    }
  }
  // roller-induced viscosity (:380-387): only smag=0 & swave=1, interior
  if (P_SMAG == 0 && P_SWAVE == 1) {
    let jlo = ${p.ny === 0 ? 0 : 1};
    if (i >= 1 && i <= NX - 1 && j >= jlo && j <= max(NY - 1, 0)) {
      nuh = max(nuh, P_NUHFAC * S2(S2_hh, kk) * pow(S2(S2_DR, kk) / P_RHO, 1.0 / 3.0));
    }
  }
  S2set(S2_nuh, kk, nuh);
}`;
}

/** nuh edge copies (visc_smagorinsky tail :1092-1098) — smag only. */
export function nuhBcKernel(g: GridConst, p: Params): string {
  return header(g, p) + /* wgsl */ `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  if (P_SMAG == 0) { return; }
  let kk = i32(gid.x);
  if (kk < M1) {
    let j = kk;
    S2set(S2_nuh, at(0, j), S2(S2_nuh, at(1, j)));
    S2set(S2_nuh, at(NX, j), S2(S2_nuh, at(NX - 1, j)));
    return;
  }
  let i = kk - M1;
  if (NY > 0 && i < N1) {
    S2set(S2_nuh, at(i, 0), S2(S2_nuh, at(i, 1)));
    S2set(S2_nuh, at(i, NY), S2(S2_nuh, at(i, NY - 1)));
  }
}`;
}

/** viscu/viscv (:408-529). */
export function viscKernel(g: GridConst, p: Params): string {
  return header(g, p) + /* wgsl */ `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let kk = i32(gid.x);
  if (kk >= NCELL) { return; }
  let i = kk % N1;
  let j = kk / N1;
  if (P_VISCOSITY == 0) {
    S2set(S2_viscu, kk, 0.0);
    S2set(S2_viscv, kk, 0.0);
    return;
  }
  // --- viscu (:408-467)
  var viscu = 0.0;
  if (i >= IMIN_UU && i <= IMAX_UU && j >= JMIN_UU && j <= JMAX_UU) {
    let hum = S2(S2_hum, kk);
    let dudx1 = S2(S2_nuh, at(i + 1, j)) * S2(S2_hh, at(i + 1, j)) *
      (S2(S2_uu, at(i + 1, j)) - S2(S2_uu, kk)) / C2(C2_dsz, at(i + 1, j));
    let dudx2 = S2(S2_nuh, kk) * S2(S2_hh, kk) *
      (S2(S2_uu, kk) - S2(S2_uu, at(i - 1, j))) / C2(C2_dsz, kk);
    viscu = (1.0 / hum) * (2.0 * (dudx1 - dudx2) / (C2(C2_dsz, kk) + C2(C2_dsz, at(i + 1, j))));
    if (P_SMAG == 1) { viscu = 2.0 * viscu; }
    let jp1 = min(j + 1, NY);
    let jm1 = max(j - 1, 0);
    let nuh1 = 0.25 *
      (S2(S2_nuh, kk) + S2(S2_nuh, at(i + 1, j)) + S2(S2_nuh, at(i + 1, jp1)) +
        S2(S2_nuh, at(i, jp1)));
    let nuh2 = 0.25 *
      (S2(S2_nuh, kk) + S2(S2_nuh, at(i + 1, j)) + S2(S2_nuh, at(i + 1, jm1)) +
        S2(S2_nuh, at(i, jm1)));
    let dudy1 = nuh1 * 0.5 * (S2(S2_hvm, kk) + S2(S2_hvm, at(i + 1, j))) *
      (S2(S2_uu, at(i, jp1)) - S2(S2_uu, kk)) / C2(C2_dnc, kk);
    let dudy2 = nuh2 * 0.5 * (S2(S2_hvm, at(i, jm1)) + S2(S2_hvm, at(i + 1, jm1))) *
      (S2(S2_uu, kk) - S2(S2_uu, at(i, jm1))) / C2(C2_dnc, at(i, jm1));
    viscu += (1.0 / hum) * (2.0 * (dudy1 - dudy2) / (C2(C2_dnc, kk) + C2(C2_dnc, at(i, jm1)))) *
      S2(S2_wetu, at(i, jp1)) * S2(S2_wetu, at(i, jm1));
    if (P_SMAG == 1) {
      let dvdx1 = nuh1 * 0.5 * (S2(S2_hvm, kk) + S2(S2_hvm, at(i + 1, j))) *
        (S2(S2_vv, at(i + 1, j)) - S2(S2_vv, kk)) / C2(C2_dsc, kk);
      let dvdx2 = nuh2 * 0.5 * (S2(S2_hvm, at(i, jm1)) + S2(S2_hvm, at(i + 1, jm1))) *
        (S2(S2_vv, at(i + 1, jm1)) - S2(S2_vv, at(i, jm1))) / C2(C2_dsc, at(i, jm1));
      viscu += (1.0 / hum) * (dvdx1 - dvdx2) / C2(C2_dnz, kk) *
        S2(S2_wetv, at(i + 1, j)) * S2(S2_wetv, kk) *
        S2(S2_wetv, at(i + 1, jm1)) * S2(S2_wetv, at(i, jm1));
    }
  }
  S2set(S2_viscu, kk, viscu);

  // --- viscv (:471-529)
  var viscv = 0.0;
  if (i >= IMIN_VV && i <= IMAX_VV && j >= JMIN_VV && j <= JMAX_VV) {
    let hvm = S2(S2_hvm, kk);
    let jp1 = min(j + 1, NY);
    let jm1 = max(j - 1, 0);
    let dvdy1 = S2(S2_nuh, at(i, jp1)) * S2(S2_hh, at(i, jp1)) *
      (S2(S2_vv, at(i, jp1)) - S2(S2_vv, kk)) / C2(C2_dnz, at(i, jp1));
    let dvdy2 = S2(S2_nuh, kk) * S2(S2_hh, kk) *
      (S2(S2_vv, kk) - S2(S2_vv, at(i, jm1))) / C2(C2_dnz, kk);
    viscv = (1.0 / hvm) * 2.0 * (dvdy1 - dvdy2) / (C2(C2_dnz, kk) + C2(C2_dnz, at(i, jp1))) *
      S2(S2_wetv, at(i, jp1)) * S2(S2_wetv, at(i, jm1));
    if (P_SMAG == 1) { viscv = 2.0 * viscv; }
    // cross term with nuh scaled by nuhv (:496-510)
    let nuh1 = P_NUHV * 0.25 *
      (S2(S2_nuh, kk) + S2(S2_nuh, at(i + 1, j)) + S2(S2_nuh, at(i + 1, jp1)) +
        S2(S2_nuh, at(i, jp1)));
    let nuh2 = P_NUHV * 0.25 *
      (S2(S2_nuh, kk) + S2(S2_nuh, at(i - 1, j)) + S2(S2_nuh, at(i - 1, jp1)) +
        S2(S2_nuh, at(i, jp1)));
    let dvdx1 = nuh1 * 0.5 * (S2(S2_hum, kk) + S2(S2_hum, at(i, jp1))) *
      (S2(S2_vv, at(i + 1, j)) - S2(S2_vv, kk)) / C2(C2_dsc, kk);
    let dvdx2 = nuh2 * 0.5 * (S2(S2_hum, at(i - 1, j)) + S2(S2_hum, at(i - 1, jp1))) *
      (S2(S2_vv, kk) - S2(S2_vv, at(i - 1, j))) / C2(C2_dsc, at(i - 1, j));
    viscv += (1.0 / hvm) * (2.0 * (dvdx1 - dvdx2) / (C2(C2_dsc, at(i - 1, j)) + C2(C2_dsc, kk))) *
      S2(S2_wetv, at(i + 1, j)) * S2(S2_wetv, at(i - 1, j));
    if (P_SMAG == 1) {
      let dudy1 = nuh1 * 0.5 * (S2(S2_hum, kk) + S2(S2_hum, at(i, jp1))) *
        (S2(S2_uu, at(i, jp1)) - S2(S2_uu, kk)) / C2(C2_dnc, kk);
      let dudy2 = nuh2 * 0.5 * (S2(S2_hum, at(i - 1, j)) + S2(S2_hum, at(i - 1, jp1))) *
        (S2(S2_uu, at(i - 1, jp1)) - S2(S2_uu, at(i - 1, j))) / C2(C2_dnc, at(i - 1, j));
      viscv += (1.0 / hvm) * (dudy1 - dudy2) / C2(C2_dsz, kk) *
        S2(S2_wetu, at(i, jp1)) * S2(S2_wetu, kk) *
        S2(S2_wetu, at(i - 1, jp1)) * S2(S2_wetv, at(i - 1, j));
    }
  }
  S2set(S2_viscv, kk, viscv);
}`;
}

/** viscv lateral copies (:482-489). */
export function viscBcKernel(g: GridConst, p: Params): string {
  return header(g, p) + /* wgsl */ `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let i = i32(gid.x);
  if (NY <= 0 || i >= N1) { return; }
  S2set(S2_viscv, at(i, 0), S2(S2_viscv, at(i, 1)));
  if (NY >= 2) { S2set(S2_viscv, at(i, NY - 1), S2(S2_viscv, at(i, NY - 2))); }
}`;
}

/** Bed friction + momentum Euler updates (:534-617). */
export function momentumKernel(g: GridConst, p: Params): string {
  return header(g, p) + /* wgsl */ `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let kk = i32(gid.x);
  if (kk >= NCELL) { return; }
  let i = kk % N1;
  let j = kk / N1;

  // bed friction (Ruessink et al 2001): uses Eulerian velocities of prev step
  var taubx = 0.0;
  var tauby = 0.0;
  if (S2(S2_wetu, kk) == 1.0) {
    let urms = S2(S2_urms, kk);
    taubx = S2(S2_cfu, kk) * P_RHO * S2(S2_ueu, kk) *
      sqrt((1.16 * urms) * (1.16 * urms) + S2(S2_vmageu, kk) * S2(S2_vmageu, kk));
    let lim = 100.0 * P_G * P_RHO * S2(S2_hu, kk);
    taubx = clamp(taubx, -lim, lim);
  }
  if (S2(S2_wetv, kk) == 1.0) {
    let urms = S2(S2_urms, kk);
    tauby = S2(S2_cfv, kk) * P_RHO * S2(S2_vev, kk) *
      sqrt((1.16 * urms) * (1.16 * urms) + S2(S2_vmagev, kk) * S2(S2_vmagev, kk));
    let lim = 100.0 * P_G * P_RHO * S2(S2_hv, kk);
    tauby = clamp(tauby, -lim, lim);
  }
  S2set(S2_taubx, kk, taubx);
  S2set(S2_tauby, kk, tauby);

  // momentum u (:561-581)
  if (i >= IMIN_UU && i <= IMAX_UU && j >= JMIN_UU && j <= JMAX_UU) {
    if (S2(S2_wetu, kk) == 1.0) {
      var dudt = S2(S2_ududx, kk) + S2(S2_vdudy, kk) - S2(S2_viscu, kk) +
        P_G * S2(S2_dzsdx, kk) +
        taubx / (P_RHO * S2(S2_hu, kk)) -
        P_LWAVE * S2(S2_Fx, kk) / (P_RHO * S2(S2_hum, kk)) -
        P_FC * S2(S2_vu, kk) -
        P_RHOA_CD * S2(S2_windsu, kk) *
          sqrt(S2(S2_windsu, kk) * S2(S2_windsu, kk) + S2(S2_windnv, kk) * S2(S2_windnv, kk)) /
          (P_RHO * S2(S2_hum, kk));
      dudt = clamp(dudt, -P_MAXFACG * P_G, P_MAXFACG * P_G);
      S2set(S2_uu, kk, S2(S2_uu, kk) - DT() * dudt);
    } else {
      S2set(S2_uu, kk, 0.0);
    }
  }
  // momentum v (:595-617)
  if (i >= IMIN_VV && i <= IMAX_VV && j >= JMIN_VV && j <= JMAX_VV) {
    if (S2(S2_wetv, kk) == 1.0) {
      var dvdt = S2(S2_udvdx, kk) + S2(S2_vdvdy, kk) - S2(S2_viscv, kk) +
        P_G * S2(S2_dzsdy, kk) +
        tauby / (P_RHO * S2(S2_hv, kk)) -
        P_LWAVE * S2(S2_Fy, kk) / (P_RHO * S2(S2_hvm, kk)) +
        P_FC * S2(S2_uv, kk) -
        P_RHOA_CD * S2(S2_windnv, kk) *
          sqrt(S2(S2_windsu, kk) * S2(S2_windsu, kk) + S2(S2_windnv, kk) * S2(S2_windnv, kk)) /
          (P_RHO * S2(S2_hvm, kk));
      dvdt = clamp(dvdt, -P_MAXFACG * P_G, P_MAXFACG * P_G);
      S2set(S2_vv, kk, S2(S2_vv, kk) - DT() * dvdt);
    } else {
      S2set(S2_vv, kk, 0.0);
    }
  }
}`;
}

/**
 * Post-momentum lateral BCs: uu row copies (:583-591) and vv rows via
 * flow_lat_bc (boundaryconditions.F90:1497-1605). One thread per i.
 */
export function flowLatBcKernel(g: GridConst, p: Params): string {
  return header(g, p) + /* wgsl */ `
fn latV(i: i32, bctype: i32, jbc: i32, jn: i32) -> f32 {
  if (NY == 0) {
    if (bctype == ${LR.wall}) { return 0.0; }
    return S2(S2_vv, at(i, jbc));
  }
  if (bctype == ${LR.wall}) { return 0.0; }
  if (bctype == ${LR.neumann_v}) { return S2(S2_vv, at(i, jn)); }
  if (bctype == ${LR.no_advec} || bctype == ${LR.neumann}) {
    if (i >= 1 && i <= NX - 1) {
      if (S2(S2_wetv, at(i, jbc)) == 1.0) {
        var advterm = S2(S2_udvdx, at(i, jbc)) + S2(S2_vdvdy, at(i, jbc)) -
          S2(S2_viscv, at(i, jbc));
        if (bctype == ${LR.no_advec}) {
          advterm += P_FC * S2(S2_uv, at(i, jbc));
          if (S2(S2_vv, at(i, jbc)) > 0.0) { advterm = max(advterm, 0.0); }
          else if (S2(S2_vv, at(i, jbc)) < 0.0) { advterm = min(advterm, 0.0); }
          else { advterm = 0.0; }
        }
        var rest = P_G * S2(S2_dzsdy, at(i, jbc)) +
          S2(S2_tauby, at(i, jbc)) / (P_RHO * S2(S2_hvm, at(i, jbc))) -
          P_LWAVE * S2(S2_Fy, at(i, jbc)) / (P_RHO * S2(S2_hvm, at(i, jbc))) -
          P_RHOA_CD * S2(S2_windnv, at(i, jbc)) *
            sqrt(
              S2(S2_windsu, at(i, jbc)) * S2(S2_windsu, at(i, jbc)) +
                S2(S2_windnv, at(i, jbc)) * S2(S2_windnv, at(i, jbc)),
            ) / (P_RHO * S2(S2_hvm, at(i, jbc)));
        if (bctype == ${LR.neumann}) { rest += P_FC * S2(S2_uv, at(i, jbc)); }
        return S2(S2_vv, at(i, jbc)) - DT() * (advterm + rest);
      }
      return 0.0;
    }
    // ends filled by caller pass 2
    return S2(S2_vv, at(i, jbc));
  }
  // LR_ABS_1D
  if (i >= 1 && i <= NX - 1 && S2(S2_wetv, at(i, jbc)) == 1.0) {
    let v = sqrt(P_G / S2(S2_hh, at(i, jn))) *
      (S2(S2_zs, at(i, jn)) - max(S2(S2_zb, at(i, jn)), S2(S2_zs0, at(i, jn)))) *
      f32(jbc - jn);
    S2set(S2_zs, at(i, jbc), S2(S2_zs, at(i, jn)));
    return v;
  }
  return 0.0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let i = i32(gid.x);
  if (i >= N1 || NY == 0) { return; }
  // uu lateral copies (:583-591)
  S2set(S2_uu, at(i, 0), S2(S2_uu, at(i, 1)));
  S2set(S2_uu, at(i, NY), S2(S2_uu, at(i, NY - 1)));
  // vv lateral: left boundary is j=0 row governed by par%right in Fortran
  // (flow_timestep.F90:621-628: isleft -> par%right, jbc=1,jn=2 (1-based))
  S2set(S2_vv, at(i, 0), latV(i, P_RIGHT, 0, 1));
  S2set(S2_vv, at(i, NY - 1), latV(i, P_LEFT, NY - 1, NY - 2));
}`;
}

/** fill ends i=0, i=NX of the lateral vv rows (flow_lat_bc tail). */
export function flowLatBc2Kernel(g: GridConst, p: Params): string {
  return header(g, p) + /* wgsl */ `
@compute @workgroup_size(4)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 4u, 0u, 0u);
  if (gid.x > 0u || NY == 0) { return; }
  if (P_RIGHT != ${LR.wall} && P_RIGHT != ${LR.neumann_v}) {
    S2set(S2_vv, at(0, 0), S2(S2_vv, at(1, 0)) * S2(S2_wetv, at(0, 0)));
    S2set(S2_vv, at(NX, 0), S2(S2_vv, at(NX - 1, 0)) * S2(S2_wetv, at(NX, 0)));
  }
  if (P_LEFT != ${LR.wall} && P_LEFT != ${LR.neumann_v}) {
    S2set(S2_vv, at(0, NY - 1), S2(S2_vv, at(1, NY - 1)) * S2(S2_wetv, at(0, NY - 1)));
    S2set(S2_vv, at(NX, NY - 1), S2(S2_vv, at(NX - 1, NY - 1)) * S2(S2_wetv, at(NX, NY - 1)));
  }
}`;
}

/** hu/hv upwind + fluxes (:661-726). */
export function huhvKernel(g: GridConst, p: Params): string {
  return header(g, p) + /* wgsl */ `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let kk = i32(gid.x);
  if (kk >= NCELL) { return; }
  let i = kk % N1;
  let j = kk / N1;
  let ip = min(i + 1, NX);
  let jp = min(j + 1, NY);
  let uu = S2(S2_uu, kk);
  var hu: f32;
  if (uu > P_UMIN) {
    if (P_OLDHU == 1) { hu = S2(S2_hh, kk); } else {
      hu = S2(S2_zs, kk) - max(S2(S2_zb, kk), S2(S2_zb, at(ip, j)));
    }
  } else if (uu < -P_UMIN) {
    if (P_OLDHU == 1) { hu = S2(S2_hh, at(ip, j)); } else {
      hu = S2(S2_zs, at(ip, j)) - max(S2(S2_zb, kk), S2(S2_zb, at(ip, j)));
    }
  } else {
    hu = max(max(S2(S2_zs, kk), S2(S2_zs, at(ip, j))) - max(S2(S2_zb, kk), S2(S2_zb, at(ip, j))), P_EPS);
  }
  hu = max(hu, 0.0);
  S2set(S2_hu, kk, hu);
  let vv = S2(S2_vv, kk);
  var hv: f32;
  if (vv > P_UMIN) {
    if (P_OLDHU == 1) { hv = S2(S2_hh, kk); } else {
      hv = S2(S2_zs, kk) - max(S2(S2_zb, kk), S2(S2_zb, at(i, jp)));
    }
  } else if (vv < -P_UMIN) {
    if (P_OLDHU == 1) { hv = S2(S2_hh, at(i, jp)); } else {
      hv = S2(S2_zs, at(i, jp)) - max(S2(S2_zb, kk), S2(S2_zb, at(i, jp)));
    }
  } else {
    hv = max(max(S2(S2_zs, kk), S2(S2_zs, at(i, jp))) - max(S2(S2_zb, kk), S2(S2_zb, at(i, jp))), P_EPS);
  }
  hv = max(hv, 0.0);
  S2set(S2_hv, kk, hv);
  let qx = uu * hu;
  let qy = vv * hv;
  S2set(S2_qx, kk, qx);
  S2set(S2_qy, kk, qy);
  S2set(S2_qmag, kk, sqrt(qx * qx + qy * qy));
}`;
}

/** Continuity: dzsdt + zs update (:735-752). */
export function continuityKernel(g: GridConst, p: Params): string {
  return header(g, p) + /* wgsl */ `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let kk = i32(gid.x);
  if (kk >= NCELL) { return; }
  let i = kk % N1;
  let j = kk / N1;
  if (i < IMIN_ZS || i > IMAX_ZS || j < JMIN_ZS || j > JMAX_ZS) { return; }
  var dzsdt: f32;
  if (NY > 0) {
    dzsdt = -(S2(S2_qx, kk) * C2(C2_dnu, kk) - S2(S2_qx, at(i - 1, j)) * C2(C2_dnu, at(i - 1, j)) +
      S2(S2_qy, kk) * C2(C2_dsv, kk) - S2(S2_qy, at(i, j - 1)) * C2(C2_dsv, at(i, j - 1))) *
      C2(C2_dsdnzi, kk);
  } else {
    dzsdt = -(S2(S2_qx, kk) * C2(C2_dnu, kk) - S2(S2_qx, at(i - 1, j)) * C2(C2_dnu, at(i - 1, j))) *
      C2(C2_dsdnzi, kk);
  }
  S2set(S2_dzsdt, kk, dzsdt);
  S2set(S2_zs, kk, S2(S2_zs, kk) + dzsdt * DT());
}`;
}

/**
 * Post pass 1 (:774-935): cell-centre u/v, vu/uv, Stokes components, GLM
 * split, velocity magnitudes, hh/zs1/hold, maxzs/minzs.
 * All reads are from post-update uu/vv/ust (stable), writes per-cell.
 */
export function flowPost1Kernel(g: GridConst, p: Params): string {
  return header(g, p) + /* wgsl */ `
const D_FILL: f32 = -999.0;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let kk = i32(gid.x);
  if (kk >= NCELL) { return; }
  let i = kk % N1;
  let j = kk / N1;

  // vu: v at u-points (:806-821)
  var vu = 0.0;
  if (NY > 0) {
    if (i <= NX - 1 && j >= 1 && j <= NY - 1) {
      vu = 0.25 * (S2(S2_vv, at(i, j - 1)) + S2(S2_vv, kk) +
        S2(S2_vv, at(i + 1, j - 1)) + S2(S2_vv, at(i + 1, j)));
    } else if (i <= NX - 1 && j == 0) {
      vu = 0.25 * (S2(S2_vv, at(i, 0)) + S2(S2_vv, at(i, 1)) +
        S2(S2_vv, at(i + 1, 0)) + S2(S2_vv, at(i + 1, 1))); // copy of j=1 form below
    } else if (i <= NX - 1 && j == NY) {
      vu = 0.25 * (S2(S2_vv, at(i, NY - 2)) + S2(S2_vv, at(i, NY - 1)) +
        S2(S2_vv, at(i + 1, NY - 2)) + S2(S2_vv, at(i + 1, NY - 1)));
    }
  } else {
    if (i <= NX - 1) { vu = 0.5 * (S2(S2_vv, kk) + S2(S2_vv, at(i + 1, 0))); }
  }
  vu = vu * S2(S2_wetu, kk);
  S2set(S2_vu, kk, vu);

  // Stokes components at u-points (:823-853)
  let relm = S2(S2_thetamean, kk) - C2(C2_alfaz, kk);
  var vsu = 0.0;
  var usu = 0.0;
  if (NY > 0) {
    if (i <= NX - 1 && j >= 1 && j <= NY - 1) {
      vsu = 0.5 * (S2(S2_ust, kk) * sin(relm) +
        S2(S2_ust, at(i + 1, j)) * sin(S2(S2_thetamean, at(i + 1, j)) - C2(C2_alfaz, at(i + 1, j))));
      usu = 0.5 * (S2(S2_ust, kk) * cos(relm) +
        S2(S2_ust, at(i + 1, j)) * cos(S2(S2_thetamean, at(i + 1, j)) - C2(C2_alfaz, at(i + 1, j))));
    } else if (i <= NX - 1 && (j == 0 || j == NY)) {
      let jj = select(NY - 1, 1, j == 0);
      vsu = 0.5 * (S2(S2_ust, at(i, jj)) * sin(S2(S2_thetamean, at(i, jj)) - C2(C2_alfaz, at(i, jj))) +
        S2(S2_ust, at(i + 1, jj)) * sin(S2(S2_thetamean, at(i + 1, jj)) - C2(C2_alfaz, at(i + 1, jj))));
      usu = 0.5 * (S2(S2_ust, at(i, jj)) * cos(S2(S2_thetamean, at(i, jj)) - C2(C2_alfaz, at(i, jj))) +
        S2(S2_ust, at(i + 1, jj)) * cos(S2(S2_thetamean, at(i + 1, jj)) - C2(C2_alfaz, at(i + 1, jj))));
    }
  } else {
    if (i <= NX - 1) {
      vsu = 0.5 * (S2(S2_ust, kk) * sin(relm) +
        S2(S2_ust, at(i + 1, 0)) * sin(S2(S2_thetamean, at(i + 1, 0)) - C2(C2_alfaz, at(i + 1, 0))));
      usu = 0.5 * (S2(S2_ust, kk) * cos(relm) +
        S2(S2_ust, at(i + 1, 0)) * cos(S2(S2_thetamean, at(i + 1, 0)) - C2(C2_alfaz, at(i + 1, 0))));
    }
  }
  vsu = vsu * S2(S2_wetu, kk);
  usu = usu * S2(S2_wetu, kk);

  // GLM at u (:856-864)
  let veu = vu - vsu;
  let ueu = S2(S2_uu, kk) - usu;
  S2set(S2_ueu, kk, ueu);
  S2set(S2_vmagu, kk, sqrt(S2(S2_uu, kk) * S2(S2_uu, kk) + vu * vu));
  S2set(S2_vmageu, kk, sqrt(ueu * ueu + veu * veu));

  // uv: u at v-points (:877-892)
  var uv = 0.0;
  if (NY > 0) {
    if (i >= 1 && i <= NX - 1 && j <= NY - 1) {
      uv = 0.25 * (S2(S2_uu, at(i - 1, j)) + S2(S2_uu, kk) +
        S2(S2_uu, at(i - 1, j + 1)) + S2(S2_uu, at(i, j + 1)));
    } else if (i >= 1 && i <= NX - 1 && j == NY) {
      uv = 0.25 * (S2(S2_uu, at(i - 1, NY - 1)) + S2(S2_uu, at(i, NY - 1)) +
        S2(S2_uu, at(i - 1, NY)) + S2(S2_uu, at(i, NY)));
    }
  } else {
    if (i >= 1 && i <= NX - 1) { uv = 0.5 * (S2(S2_uu, at(i - 1, 0)) + S2(S2_uu, kk)); }
  }
  uv = uv * S2(S2_wetv, kk);
  S2set(S2_uv, kk, uv);

  // Stokes at v-points (:894-923)
  var vsv = 0.0;
  var usv = 0.0;
  if (NY > 0) {
    if (i >= 1 && i <= NX - 1 && j <= NY - 1) {
      vsv = 0.5 * (S2(S2_ust, kk) * sin(relm) +
        S2(S2_ust, at(i, j + 1)) * sin(S2(S2_thetamean, at(i, j + 1)) - C2(C2_alfaz, at(i, j + 1))));
      usv = 0.5 * (S2(S2_ust, kk) * cos(relm) +
        S2(S2_ust, at(i, j + 1)) * cos(S2(S2_thetamean, at(i, j + 1)) - C2(C2_alfaz, at(i, j + 1))));
    } else if (i >= 1 && i <= NX - 1 && j == NY) {
      vsv = 0.5 * (S2(S2_ust, at(i, NY - 1)) * sin(S2(S2_thetamean, at(i, NY - 1)) - C2(C2_alfaz, at(i, NY - 1))) +
        S2(S2_ust, at(i, NY)) * sin(S2(S2_thetamean, at(i, NY)) - C2(C2_alfaz, at(i, NY))));
      usv = 0.5 * (S2(S2_ust, at(i, NY - 1)) * cos(S2(S2_thetamean, at(i, NY - 1)) - C2(C2_alfaz, at(i, NY - 1))) +
        S2(S2_ust, at(i, NY)) * cos(S2(S2_thetamean, at(i, NY)) - C2(C2_alfaz, at(i, NY))));
    }
  } else {
    if (i >= 1 && i <= NX - 1) {
      vsv = S2(S2_ust, kk) * sin(relm);
      usv = S2(S2_ust, kk) * cos(relm);
    }
  }
  vsv = vsv * S2(S2_wetv, kk);
  usv = usv * S2(S2_wetv, kk);

  // GLM at v (:925-934)
  let vev = S2(S2_vv, kk) - vsv;
  let uev = uv - usv;
  S2set(S2_vev, kk, vev);
  S2set(S2_vmagv, kk, sqrt(uv * uv + S2(S2_vv, kk) * S2(S2_vv, kk)));
  S2set(S2_vmagev, kk, sqrt(uev * uev + vev * vev));

  // depth bookkeeping (:938-954)
  let zs = S2(S2_zs, kk);
  S2set(S2_hold, kk, S2(S2_hh, kk));
  S2set(S2_hh, kk, max(zs - S2(S2_zb, kk), P_EPS));
  S2set(S2_zs1, kk, zs - S2(S2_zs0, kk));
  if (S2(S2_wetz, kk) == 1.0) {
    if (S2(S2_maxzs, kk) > D_FILL + 1.0) {
      S2set(S2_maxzs, kk, max(zs, S2(S2_maxzs, kk)));
      S2set(S2_minzs, kk, min(zs, S2(S2_minzs, kk)));
    } else {
      S2set(S2_maxzs, kk, zs);
      S2set(S2_minzs, kk, zs);
    }
  }
}`;
}

/** Post pass 2: cell-centre u/v/ue/ve (:776-797, 866-875) from ueu/vev. */
export function flowPost2Kernel(g: GridConst, p: Params): string {
  return header(g, p) + /* wgsl */ `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let kk = i32(gid.x);
  if (kk >= NCELL) { return; }
  let i = kk % N1;
  let j = kk / N1;
  // u centres (:776-783)
  var u = 0.0;
  if (i >= 1 && i <= NX - 1) { u = 0.5 * (S2(S2_uu, at(i - 1, j)) + S2(S2_uu, kk)); }
  else if (i == 0) { u = S2(S2_uu, at(0, j)); }
  else { u = 0.5 * (S2(S2_uu, at(NX - 2, j)) + S2(S2_uu, at(NX - 1, j))); } // u(nx+1)=u(nx)
  S2set(S2_u, kk, u);
  // v centres (:785-797)
  var v = 0.0;
  if (NY > 0) {
    if (j >= 1 && j <= NY - 1) { v = 0.5 * (S2(S2_vv, at(i, j - 1)) + S2(S2_vv, kk)); }
    else if (j == 0) { v = S2(S2_vv, at(i, 0)); }
    else { v = 0.5 * (S2(S2_vv, at(i, NY - 2)) + S2(S2_vv, at(i, NY - 1))); }
    if (i == NX) { // v(nx+1,:)=v(nx,:)
      if (j >= 1 && j <= NY - 1) {
        v = 0.5 * (S2(S2_vv, at(NX - 1, j - 1)) + S2(S2_vv, at(NX - 1, j)));
      } else if (j == 0) { v = S2(S2_vv, at(NX - 1, 0)); } else {
        v = 0.5 * (S2(S2_vv, at(NX - 1, NY - 2)) + S2(S2_vv, at(NX - 1, NY - 1)));
      }
    }
  } else {
    v = S2(S2_vv, kk);
  }
  S2set(S2_v, kk, v);
  // ue centres (:866-869)
  var ue = 0.0;
  if (i >= 1 && i <= NX - 1) { ue = 0.5 * (S2(S2_ueu, at(i - 1, j)) + S2(S2_ueu, kk)); }
  else if (i == 0) { ue = S2(S2_ueu, at(0, j)); }
  else { ue = 0.5 * (S2(S2_ueu, at(NX - 2, j)) + S2(S2_ueu, at(NX - 1, j))); }
  S2set(S2_ue, kk, ue);
  // ve centres (:871-876)
  var ve = 0.0;
  if (NY > 0) {
    if (j >= 1 && j <= NY - 1) { ve = 0.5 * (S2(S2_vev, at(i, j - 1)) + S2(S2_vev, kk)); }
    else if (j == 0) { ve = S2(S2_vev, at(i, 0)); }
    else { ve = 0.5 * (S2(S2_vev, at(i, NY - 2)) + S2(S2_vev, at(i, NY - 1))); }
  } else {
    ve = S2(S2_vev, kk);
  }
  S2set(S2_ve, kk, ve);
}`;
}

/**
 * flow_bc (boundaryconditions.F90:930-1495): front/back/lateral rows.
 * Runs at the start of each step (before wave/flow). One thread per j for
 * front/back; lateral zs rows handled by threads M1..M1+N1.
 * umean/vmean smoothing (tidetype=velocity, epsi<0) is done host-side into
 * scratch[5] (umean) and scratch[6] (vmean) — uniform along the boundary.
 */
export function flowBcKernel(g: GridConst, p: Params): string {
  return header(g, p) + /* wgsl */ `
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let kk = i32(gid.x);
  let umean = scratch[5];
  let vmean = scratch[6];
  if (kk < M1) {
    let j = kk;
    let jok = (j >= JMIN_ZS && j <= JMAX_ZS) || NY == 0;
    // --- front boundary (:1202-1341)
    if (jok) {
      if (P_FRONT == ${FRONT.abs_1d}) {
        let hh1 = S2(S2_hh, at(0, j));
        var uubc: f32;
        if (P_FREEWAVE == 1) {
          uubc = 2.0 * S2(S2_ui, at(0, j)) -
            sqrt(P_G / hh1) * (S2(S2_zs, at(1, j)) - S2(S2_zs0, at(1, j))) + umean;
        } else {
          uubc = (1.0 + sqrt(P_G * hh1) / S2(S2_cg, at(0, j))) * S2(S2_ui, at(0, j)) -
            sqrt(P_G / hh1) * (S2(S2_zs, at(1, j)) - S2(S2_zs0, at(1, j))) + umean;
        }
        S2set(S2_uu, at(0, j), uubc);
        S2set(S2_vv, at(0, j), S2(S2_vv, at(1, j)));
        S2set(S2_zs, at(0, j), S2(S2_zs, at(1, j)));
      } else if (P_FRONT == ${FRONT.abs_2d}) {
        // Van Dongeren (1997) weakly reflective (:1222-1301)
        let ht1 = max(S2(S2_zs0, at(0, j)) - S2(S2_zb, at(0, j)), P_EPS);
        let ht2 = max(S2(S2_zs0, at(1, j)) - S2(S2_zb, at(1, j)), P_EPS);
        let hum1 = S2(S2_hum, at(0, j));
        let beta1 = S2(S2_uu, at(0, j)) - 2.0 * sqrt(P_G * hum1);
        let beta2 = S2(S2_uu, at(1, j)) - 2.0 * sqrt(P_G * S2(S2_hum, at(1, j)));
        let dhdx = (ht2 - ht1) / C2(C2_dsu, at(0, j));
        let dbetadx = (beta2 - beta1) / C2(C2_dsz, at(1, j));
        var dvdy = 0.0;
        var dbetady = 0.0;
        if (NY > 0 && j >= 1 && j <= NY - 1) {
          let dn = C2(C2_dnu, at(0, j)) +
            (C2(C2_dnu, at(0, j - 1)) + C2(C2_dnu, at(0, j + 1))) / 2.0;
          dvdy = (S2(S2_vu, at(0, j + 1)) - S2(S2_vu, at(0, j - 1))) / dn;
          let betaj1 = S2(S2_uu, at(0, j + 1)) - 2.0 * sqrt(P_G * S2(S2_hum, at(0, j + 1)));
          let betajm = S2(S2_uu, at(0, j - 1)) - 2.0 * sqrt(P_G * S2(S2_hum, at(0, j - 1)));
          dbetady = (betaj1 - betajm) / dn;
        }
        let bn = -(S2(S2_uu, at(0, j)) - sqrt(P_G * hum1)) * dbetadx -
          S2(S2_vu, at(0, j)) * dbetady +
          sqrt(P_G * hum1) * dvdy +
          P_G * dhdx +
          S2(S2_Fx, at(0, j)) / hum1 / P_RHO -
          S2(S2_cfu, at(0, j)) *
            sqrt(S2(S2_uu, at(0, j)) * S2(S2_uu, at(0, j)) + S2(S2_vu, at(0, j)) * S2(S2_vu, at(0, j))) *
            S2(S2_uu, at(0, j)) / hum1;
        let betanp1 = beta1 + bn * DT();
        var alpha2 = -(U.theta0 - C2(C2_alfaz, at(0, j)));
        var ur = 0.0;
        // Incoming-wave term, Fortran :1255 + :1265-1275:
        //   ui*(cg*cos(thetai) - sqrt(g*h)) / (cg*cos(thetai)),  thetai = atan(vi/ui)
        // Since cos(atan(vi/ui)) = |ui|/sqrt(ui^2+vi^2), this is exactly
        //   ui - sign(ui)*sqrt(ui^2+vi^2)*sqrt(g*h)/cg
        // which has no division. The trig form is what f64 Fortran runs; in f32
        // with hardware atan/cos (abs error up to 2^-11) a grazing incoming flux
        // (|ui| << |vi|) turns cos(thetai) into noise of either sign and ur
        // blows up — the Oahu offshore-column NaN (docs/deviations.md).
        let uij = S2(S2_ui, at(0, j));
        let vij = S2(S2_vi, at(0, j));
        let qmag = sign(uij) * sqrt(uij * uij + vij * vij);
        let cmean = sqrt(P_G * 0.5 * (ht1 + ht2));
        var uiTerm: f32;
        if (P_FREEWAVE == 1) {
          uiTerm = uij - qmag;
        } else {
          uiTerm = uij - qmag * cmean / S2(S2_cg, at(0, j));
        }
        for (var jj = 0; jj < 50; jj++) {
          ur = cos(alpha2) / (cos(alpha2) + 1.0) * (betanp1 - umean + 2.0 * cmean - uiTerm);
          let vert = S2(S2_vu, at(0, j)) - vmean - S2(S2_vi, at(0, j));
          var alphanew = atan(vert / (ur + 1e-16));
          if (alphanew > U.px * 0.5) { alphanew -= U.px; }
          if (alphanew <= -U.px * 0.5) { alphanew += U.px; }
          if (abs(alphanew - alpha2) < 0.001) { break; }
          alpha2 = alphanew;
        }
        if (P_ARC == 0) {
          S2set(S2_uu, at(0, j), (P_ORDER - 1.0) * S2(S2_ui, at(0, j)) + umean);
          S2set(S2_zs, at(0, j), S2(S2_zs, at(1, j)));
        } else {
          let uu0 = (P_ORDER - 1.0) * S2(S2_ui, at(0, j)) + ur + umean;
          S2set(S2_uu, at(0, j), uu0);
          let zsnew = 1.5 *
              ((betanp1 - uu0) * (betanp1 - uu0) / 4.0 / P_G +
                0.5 * (S2(S2_zb, at(0, j)) + S2(S2_zb, at(1, j)))) -
            0.5 *
              ((beta2 - S2(S2_uu, at(1, j))) * (beta2 - S2(S2_uu, at(1, j))) / 4.0 / P_G +
                0.5 * (S2(S2_zb, at(1, j)) + S2(S2_zb, at(2, j))));
          S2set(S2_zs, at(0, j), zsnew);
        }
        S2set(S2_vv, at(0, j), S2(S2_vv, at(1, j)));
      } else if (P_FRONT == ${FRONT.wall}) {
        S2set(S2_uu, at(0, j), 0.0);
        S2set(S2_zs, at(0, j), S2(S2_zs, at(1, j)));
      }
    }
    // --- back boundary (:1394-1457)
    if (jok) {
      if (P_BACK == ${BACK.wall}) {
        S2set(S2_uu, at(NX - 1, j), 0.0);
        S2set(S2_zs, at(NX, j), S2(S2_zs, at(NX - 1, j)));
      } else if (P_BACK == ${BACK.abs_1d}) {
        let hhn = S2(S2_hh, at(NX - 1, j));
        S2set(
          S2_uu,
          at(NX - 1, j),
          sqrt(P_G / hhn) *
              (S2(S2_zs, at(NX - 1, j)) - max(S2(S2_zb, at(NX - 1, j)), S2(S2_zs0, at(NX - 1, j)))) +
            scratch[7], // umean back
        );
        S2set(S2_zs, at(NX, j), S2(S2_zs, at(NX - 1, j)));
      } else if (P_BACK == ${BACK.abs_2d}) {
        if (S2(S2_wetu, at(NX - 1, j)) == 1.0) {
          let ht1 = max(S2(S2_zs0, at(NX - 1, j)) - S2(S2_zb, at(NX - 1, j)), P_EPS);
          let ht2 = max(S2(S2_zs0, at(NX, j)) - S2(S2_zb, at(NX, j)), P_EPS);
          let humn = S2(S2_hum, at(NX - 1, j));
          let beta1 = S2(S2_uu, at(NX - 2, j)) + 2.0 * sqrt(P_G * S2(S2_hum, at(NX - 2, j)));
          let beta2 = S2(S2_uu, at(NX - 1, j)) + 2.0 * sqrt(P_G * humn);
          var dvdy = 0.0;
          var dbetady = 0.0;
          if (NY > 0 && j >= 1 && j <= NY - 1) {
            dvdy = (S2(S2_vu, at(NX - 1, j + 1)) - S2(S2_vu, at(NX - 1, j - 1))) /
              (2.0 * C2(C2_dnu, at(NX - 1, j)));
            let bj1 = S2(S2_uu, at(NX - 2, j + 1)) + 2.0 * sqrt(P_G * S2(S2_hum, at(NX - 2, j + 1)));
            let bjm = S2(S2_uu, at(NX - 2, j - 1)) + 2.0 * sqrt(P_G * S2(S2_hum, at(NX - 2, j - 1)));
            dbetady = (bj1 - bjm) / (2.0 * C2(C2_dnu, at(NX - 1, j)));
          }
          let dhdx = (ht2 - ht1) / C2(C2_dsu, at(NX - 1, j));
          let dbetadx = (beta2 - beta1) / C2(C2_dsz, at(NX - 1, j));
          let bn = -(S2(S2_uu, at(NX - 1, j)) + sqrt(P_G * humn)) * dbetadx -
            S2(S2_vu, at(NX - 1, j)) * dbetady -
            sqrt(P_G * humn) * dvdy +
            S2(S2_Fx, at(NX - 1, j)) / humn / P_RHO -
            S2(S2_cfu, at(NX - 1, j)) *
              sqrt(
                S2(S2_uu, at(NX - 1, j)) * S2(S2_uu, at(NX - 1, j)) +
                  S2(S2_vu, at(NX - 1, j)) * S2(S2_vu, at(NX - 1, j)),
              ) * S2(S2_uu, at(NX - 1, j)) / humn +
            P_G * dhdx;
          let betanp1 = beta2 + bn * DT();
          var alpha2 = U.theta0;
          var ur = 0.0;
          for (var jj = 0; jj < 50; jj++) {
            ur = cos(alpha2) / (cos(alpha2) + 1.0) *
              (betanp1 - scratch[7] - 2.0 * sqrt(P_G * 0.5 * (ht1 + ht2)));
            let vert = S2(S2_vu, at(NX - 1, j)) - scratch[8];
            var alphanew = atan(vert / (ur + 1e-16));
            if (alphanew > U.px * 0.5) { alphanew -= U.px; }
            if (alphanew <= -U.px * 0.5) { alphanew += U.px; }
            if (abs(alphanew - alpha2) < 0.001) { break; }
            alpha2 = alphanew;
          }
          let uun = ur + scratch[7];
          S2set(S2_uu, at(NX - 1, j), uun);
          let zsn = 1.5 *
              ((betanp1 - uun) * (betanp1 - uun) / 4.0 / P_G +
                0.5 * (S2(S2_zb, at(NX - 1, j)) + S2(S2_zb, at(NX, j)))) -
            0.5 *
              ((beta1 - S2(S2_uu, at(NX - 2, j))) * (beta1 - S2(S2_uu, at(NX - 2, j))) / 4.0 / P_G +
                0.5 * (S2(S2_zb, at(NX - 2, j)) + S2(S2_zb, at(NX - 1, j))));
          S2set(S2_zs, at(NX, j), zsn);
        }
      }
      // (:1456-1457)
      S2set(S2_vv, at(NX, j), S2(S2_vv, at(NX - 1, j)));
      S2set(S2_uu, at(NX, j), S2(S2_uu, at(NX - 1, j)));
    }
    return;
  }
  // --- lateral zs rows (:1474-1483) + front/back corners (:1343-1354)
  let i = kk - M1;
  if (NY > 0 && i < N1) {
    if (i >= IMIN_ZS && i <= IMAX_ZS) {
      // dzs0dn = 0 without tide-gradient support (documented)
      S2set(S2_zs, at(i, 0), max(S2(S2_zs, at(i, 1)), S2(S2_zb, at(i, 0))));
      S2set(S2_zs, at(i, NY), max(S2(S2_zs, at(i, NY - 1)), S2(S2_zb, at(i, NY))));
    } else if (i == 0) {
      S2set(S2_uu, at(0, 0), S2(S2_uu, at(0, 1)));
      S2set(S2_vv, at(0, 0), S2(S2_vv, at(0, 1)));
      S2set(S2_zs, at(0, 0), max(S2(S2_zs, at(0, 1)), S2(S2_zb, at(0, 0))));
      S2set(S2_uu, at(0, NY), S2(S2_uu, at(0, NY - 1)));
      S2set(S2_vv, at(0, NY), S2(S2_vv, at(0, NY - 1)));
      S2set(S2_zs, at(0, NY), max(S2(S2_zs, at(0, NY - 1)), S2(S2_zb, at(0, NY))));
    } else if (i == NX) {
      S2set(S2_uu, at(NX, 0), S2(S2_uu, at(NX, 1)));
      S2set(S2_vv, at(NX, 0), S2(S2_vv, at(NX, 1)));
      S2set(S2_zs, at(NX, 0), max(S2(S2_zs, at(NX, 1)), S2(S2_zb, at(0, 0))));
      S2set(S2_uu, at(NX, NY), S2(S2_uu, at(NX, NY - 1)));
      S2set(S2_vv, at(NX, NY), S2(S2_vv, at(NX, NY - 1)));
      S2set(S2_zs, at(NX, NY), max(S2(S2_zs, at(NX, NY - 1)), S2(S2_zb, at(0, NY))));
    }
  }
}`;
}

/** CFL dt reduction (timestep.F90:485-592) into red[0]. */
export function dtFlowKernel(g: GridConst, p: Params): string {
  return header(g, p) + /* wgsl */ `
@group(0) @binding(8) var<storage, read_write> red: array<atomic<u32>>;
fn fbits(x: f32) -> u32 { return bitcast<u32>(max(x, 0.0)); }
const N3: i32 = NCELL * NTHETA;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let kk = i32(gid.x);
  if (kk >= NCELL) { return; }
  let i = kk % N1;
  let j = kk / N1;
  if (i < IMIN_EE || i > IMAX_EE || j < JMIN_EE || j > JMAX_EE) { return; }
  var dt = 1e30;
  let tny = 1e-30;
  if (NY > 2) {
    if (S2(S2_wetu, kk) == 1.0) {
      let mdx = C2(C2_dsu, at(i + 1, j));
      let mdy = min(C2(C2_dnv, kk), C2(C2_dnv, at(i, j - 1)));
      dt = min(dt, mdx / max(tny, max(sqrt(P_G * S2(S2_hu, kk)) + abs(S2(S2_uu, kk)), abs(S2(S2_ueu, kk)))));
      dt = min(dt, mdy / max(tny, sqrt(P_G * S2(S2_hu, kk)) + abs(S2(S2_vu, kk))));
    }
    if (S2(S2_wetv, kk) == 1.0) {
      let mdx = min(C2(C2_dsu, kk), C2(C2_dsu, at(i - 1, j)));
      let mdy = C2(C2_dnv, kk);
      dt = min(dt, mdx / max(tny, sqrt(P_G * S2(S2_hv, kk)) + abs(S2(S2_uv, kk))));
      dt = min(dt, mdy / max(tny, max(sqrt(P_G * S2(S2_hv, kk)) + abs(S2(S2_vv, kk)), abs(S2(S2_vev, kk)))));
    }
    if (S2(S2_wetz, kk) == 1.0) {
      let mdx = min(C2(C2_dsu, kk), C2(C2_dsz, kk));
      let mdy = min(C2(C2_dnv, kk), C2(C2_dnz, kk));
      let mdx2 = mdx * mdx;
      let mdy2 = mdy * mdy;
      dt = min(dt, 0.5 * mdx2 * mdy2 / (mdx2 + mdy2) / max(S2(S2_nuh, kk), 1e-6));
    }
  } else {
    if (S2(S2_wetu, kk) == 1.0) {
      let mdx = C2(C2_dsu, kk);
      dt = min(dt, mdx / max(tny, max(sqrt(P_G * S2(S2_hu, kk)) + abs(S2(S2_uu, kk)), abs(S2(S2_ueu, kk)))));
    }
    if (S2(S2_wetz, kk) == 1.0) {
      let mdx = min(C2(C2_dsu, kk), C2(C2_dsu, at(max(i - 1, 0), j)));
      dt = min(dt, mdx / max(tny, sqrt(P_G * S2(S2_hv, kk)) + abs(S2(S2_uv, kk))));
      let mdxz = min(C2(C2_dsu, kk), C2(C2_dsz, kk));
      var mdy = mdxz;
      if (P_DY > 0.0) { mdy = P_DY; }
      let mdx2 = mdxz * mdxz;
      let mdy2 = mdy * mdy;
      dt = min(dt, 0.5 * mdx2 * mdy2 / (mdx2 + mdy2) / max(S2(S2_nuh, kk), 1e-6));
    }
  }
  // wave refraction criterion (non-stationary, swave=1) folded in by host via
  // separate reduce over ctheta (dtwReduce path); here flow only.
  atomicMin(&red[0], fbits(dt));
}`;
}

/** ctheta dt criterion for instationary (timestep.F90:632-648) into red[3]. */
export function dtThetaKernel(g: GridConst, p: Params): string {
  return header(g, p) + /* wgsl */ `
@group(0) @binding(8) var<storage, read_write> red: array<atomic<u32>>;
fn fbits(x: f32) -> u32 { return bitcast<u32>(max(x, 0.0)); }
const N3: i32 = NCELL * NTHETA;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 64u, 0u, 0u);
  let kk = i32(gid.x);
  if (kk >= NCELL) { return; }
  if (S2(S2_wetz, kk) != 1.0) { return; }
  var m = 0.0;
  for (var it = 0; it < NTHETA; it++) {
    m = max(m, abs(v3[V3_ctheta * N3 + kk + it * NCELL]));
  }
  // host computes dt = min(dt, CFL*dtheta/max) — store max|ctheta| as min of inverse
  atomicMin(&red[3], fbits(U.dtheta / max(m, 1e-30)));
}`;
}

/**
 * Boundary-mean velocities (boundaryconditions.F90:1157-1182, 1367-1392):
 * umean/vmean smoothing at front (scratch[5,6]) and back (scratch[7,8]).
 * factime supplied by host in scratch[9]. Single-thread row loops (rows are
 * short; runs once per step).
 */
export function umeanKernel(g: GridConst, p: Params): string {
  return header(g, p) + /* wgsl */ `
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid_: vec3<u32>,
        @builtin(num_workgroups) nwg_: vec3<u32>) {
  // 2-D dispatch fold (65535-workgroup cap): linear id in gid.x
  let gid = vec3<u32>(gid_.x + gid_.y * nwg_.x * 1u, 0u, 0u);
  if (gid.x > 0u) { return; }
  // factime (boundaryconditions.F90:1138-1154), from GPU-resident dt
  var factime: f32;
  if (${p.epsi} < 0.0) {
    if (P_SWAVE == 1) { factime = 1.0 / ${p.cats} / U.Trep * DT(); }
    else { factime = DT() / 60.0; }
  } else { factime = ${p.epsi}; }
  if (factime <= 0.0) {
    scratch[5] = 0.0; scratch[6] = 0.0; scratch[7] = 0.0; scratch[8] = 0.0;
    return;
  }
  // front (i=0 row): u-average over jmin_uu..jmax_uu, v over jmin_vv..jmax_vv
  var us = 0.0; var ds = 0.0; var vs = 0.0; var dvs = 0.0;
  for (var j = JMIN_UU; j <= JMAX_UU; j++) {
    us += S2(S2_uu, at(0, j)) * C2(C2_dnu, at(0, j));
    ds += C2(C2_dnu, at(0, j));
  }
  for (var j = JMIN_VV; j <= JMAX_VV; j++) {
    vs += S2(S2_vv, at(0, j)) * C2(C2_dnv, at(0, j));
    dvs += C2(C2_dnv, at(0, j));
  }
  scratch[5] = factime * us / max(ds, 1e-12) + (1.0 - factime) * scratch[5];
  scratch[6] = factime * vs / max(dvs, 1e-12) + (1.0 - factime) * scratch[6];
  // back (i=NX-1 row)
  us = 0.0; ds = 0.0; vs = 0.0; dvs = 0.0;
  for (var j = JMIN_UU; j <= JMAX_UU; j++) {
    us += S2(S2_uu, at(NX - 1, j)) * C2(C2_dnu, at(NX - 1, j));
    ds += C2(C2_dnu, at(NX - 1, j));
  }
  for (var j = JMIN_VV; j <= JMAX_VV; j++) {
    vs += S2(S2_vv, at(NX - 1, j)) * C2(C2_dnv, at(NX - 1, j));
    dvs += C2(C2_dnv, at(NX - 1, j));
  }
  scratch[7] = factime * us / max(ds, 1e-12) + (1.0 - factime) * scratch[7];
  scratch[8] = factime * vs / max(dvs, 1e-12) + (1.0 - factime) * scratch[8];
}`;
}
