// SPDX-License-Identifier: LGPL-3.0-or-later
// XBeach WebGPU solver — derived from XBeach (c) Deltares (LGPL v2.1+).
// Copyright (c) 2026 Oceanum. See LICENSE.
/**
 * Grid setup mirroring trunk/src/xbeachlibrary/spaceparams.F90 (gridprops) and
 * initialize.F90 (theta grid, lines 255-390).
 *
 * Arrays are Float64Array of size (nx+1)*(ny+1), column-major like Fortran
 * (index = i + j*(nx+1), i cross-shore 0..nx, j alongshore 0..ny) so binary
 * comparison against Fortran output is direct. GPU upload converts to f32.
 */

import type { Params } from "./params.ts";

export interface Grid {
  nx: number;
  ny: number;
  ntheta: number;
  ntheta_s: number;
  dtheta: number; // radians
  dtheta_s: number;
  thetamin: number; // radians, world-referenced (after conversion)
  thetamax: number;
  theta0: number;
  theta: Float64Array; // bin centres (world radians), length ntheta
  theta_s: Float64Array;
  // coordinates (world)
  xz: Float64Array;
  yz: Float64Array;
  xu: Float64Array;
  yu: Float64Array;
  xv: Float64Array;
  yv: Float64Array;
  // metrics
  dsu: Float64Array;
  dsz: Float64Array;
  dsv: Float64Array;
  dsc: Float64Array;
  dnu: Float64Array;
  dnz: Float64Array;
  dnv: Float64Array;
  dnc: Float64Array;
  dsdnui: Float64Array;
  dsdnvi: Float64Array;
  dsdnzi: Float64Array;
  alfaz: Float64Array;
  alfau: Float64Array;
  alfav: Float64Array;
  sdist: Float64Array;
  ndist: Float64Array;
  // bathy
  zb: Float64Array;
}

function idx(nx: number): (i: number, j: number) => number {
  return (i, j) => i + j * (nx + 1);
}

/** Build computational x,y (grid-aligned, before world rotation). */
export function buildXY(p: Params, xfileData?: Float64Array, yfileData?: Float64Array): {
  x: Float64Array;
  y: Float64Array;
} {
  const { nx, ny } = p;
  const n = (nx + 1) * (ny + 1);
  const x = new Float64Array(n);
  const y = new Float64Array(n);
  const at = idx(nx);
  if (p.vardx === 0) {
    for (let j = 0; j <= ny; j++) {
      for (let i = 0; i <= nx; i++) {
        x[at(i, j)] = i * p.dx;
        y[at(i, j)] = j * p.dy;
      }
    }
  } else {
    if (!xfileData) throw new Error("vardx=1 requires xfile data");
    x.set(xfileData);
    if (yfileData) y.set(yfileData);
  }
  return { x, y };
}

export function buildGrid(p: Params, x: Float64Array, y: Float64Array, zbIn: Float64Array): Grid {
  const { nx, ny } = p;
  const N = (nx + 1) * (ny + 1);
  const at = idx(nx);
  const alfa = (p.alfa * Math.PI) / 180;

  // world coordinates of z-points (spaceparams.F90:1183)
  const xz = new Float64Array(N), yz = new Float64Array(N);
  for (let k = 0; k < N; k++) {
    xz[k] = p.xori + x[k] * Math.cos(alfa) - y[k] * Math.sin(alfa);
    yz[k] = p.yori + x[k] * Math.sin(alfa) + y[k] * Math.cos(alfa);
  }

  // u-points (:1192)
  const xu = new Float64Array(N), yu = new Float64Array(N);
  for (let j = 0; j <= ny; j++) {
    for (let i = 0; i < nx; i++) {
      xu[at(i, j)] = 0.5 * (xz[at(i, j)] + xz[at(i + 1, j)]);
      yu[at(i, j)] = 0.5 * (yz[at(i, j)] + yz[at(i + 1, j)]);
    }
    xu[at(nx, j)] = 1.5 * xz[at(nx, j)] - 0.5 * xz[at(nx - 1, j)];
    yu[at(nx, j)] = 1.5 * yz[at(nx, j)] - 0.5 * yz[at(nx - 1, j)];
  }

  // v-points (:1202)
  const xv = new Float64Array(N), yv = new Float64Array(N);
  if (ny > 0) {
    for (let i = 0; i <= nx; i++) {
      for (let j = 0; j < ny; j++) {
        xv[at(i, j)] = 0.5 * (xz[at(i, j)] + xz[at(i, j + 1)]);
        yv[at(i, j)] = 0.5 * (yz[at(i, j)] + yz[at(i, j + 1)]);
      }
      xv[at(i, ny)] = 1.5 * xz[at(i, ny)] - 0.5 * xz[at(i, ny - 1)];
      yv[at(i, ny)] = 1.5 * yz[at(i, ny)] - 0.5 * yz[at(i, ny - 1)];
    }
  } else {
    xv.set(xz);
    yv.set(yz);
  }

  // corner points (:1217)
  const xc = new Float64Array(N), yc = new Float64Array(N);
  if (ny > 0) {
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        xc[at(i, j)] = 0.25 *
          (xz[at(i, j)] + xz[at(i + 1, j)] + xz[at(i, j + 1)] + xz[at(i + 1, j + 1)]);
        yc[at(i, j)] = 0.25 *
          (yz[at(i, j)] + yz[at(i + 1, j)] + yz[at(i, j + 1)] + yz[at(i + 1, j + 1)]);
      }
      xc[at(nx, j)] = 0.5 * (xu[at(nx, j)] + xu[at(nx, j + 1)]);
      yc[at(nx, j)] = 0.5 * (yu[at(nx, j)] + yu[at(nx, j + 1)]);
    }
    for (let i = 0; i < nx; i++) {
      xc[at(i, ny)] = 0.5 * (xv[at(i, ny)] + xv[at(i + 1, ny)]);
      yc[at(i, ny)] = 0.5 * (yv[at(i, ny)] + yv[at(i + 1, ny)]);
    }
    xc[at(nx, ny)] = 1.5 * xu[at(nx, ny)] - 0.5 * xu[at(nx - 1, ny)];
    yc[at(nx, ny)] = 1.5 * yu[at(nx, ny)] - 0.5 * yu[at(nx - 1, ny)];
  } else {
    xc.set(xu);
    yc.set(yu);
  }

  const hyp = (dx: number, dy: number) => Math.sqrt(dx * dx + dy * dy);

  // dsu (:1239)
  const dsu = new Float64Array(N);
  for (let j = 0; j <= ny; j++) {
    for (let i = 0; i < nx; i++) {
      dsu[at(i, j)] = hyp(xz[at(i + 1, j)] - xz[at(i, j)], yz[at(i + 1, j)] - yz[at(i, j)]);
    }
    dsu[at(nx, j)] = dsu[at(nx - 1, j)];
  }
  // dsz (:1248)
  const dsz = new Float64Array(N);
  for (let j = 0; j <= ny; j++) {
    for (let i = 1; i <= nx; i++) {
      dsz[at(i, j)] = hyp(xu[at(i, j)] - xu[at(i - 1, j)], yu[at(i, j)] - yu[at(i - 1, j)]);
    }
    dsz[at(0, j)] = dsz[at(1, j)];
  }
  // dsv (:1257)
  const dsv = new Float64Array(N);
  if (ny > 0) {
    for (let j = 0; j <= ny; j++) {
      for (let i = 1; i <= nx; i++) {
        dsv[at(i, j)] = hyp(xc[at(i, j)] - xc[at(i - 1, j)], yc[at(i, j)] - yc[at(i - 1, j)]);
      }
      dsv[at(0, j)] = dsv[at(1, j)];
      dsv[at(nx, j)] = dsv[at(nx - 1, j)];
    }
  } else dsv.set(dsz);
  // dsc (:1271)
  const dsc = new Float64Array(N);
  if (ny > 0) {
    for (let j = 0; j <= ny; j++) {
      for (let i = 0; i < nx; i++) {
        dsc[at(i, j)] = hyp(xv[at(i + 1, j)] - xv[at(i, j)], yv[at(i + 1, j)] - yv[at(i, j)]);
      }
      dsc[at(nx, j)] = dsc[at(nx - 1, j)];
    }
  } else dsc.set(dsu);
  // dnu (:1284)
  const dnu = new Float64Array(N);
  if (ny > 0) {
    for (let j = 1; j <= ny; j++) {
      for (let i = 0; i <= nx; i++) {
        dnu[at(i, j)] = hyp(xc[at(i, j)] - xc[at(i, j - 1)], yc[at(i, j)] - yc[at(i, j - 1)]);
      }
    }
    for (let i = 0; i <= nx; i++) dnu[at(i, 0)] = dnu[at(i, 1)];
  } else dnu.fill(100);
  // dnz (:1297)
  const dnz = new Float64Array(N);
  if (ny > 0) {
    for (let j = 1; j <= ny; j++) {
      for (let i = 0; i <= nx; i++) {
        dnz[at(i, j)] = hyp(xv[at(i, j)] - xv[at(i, j - 1)], yv[at(i, j)] - yv[at(i, j - 1)]);
      }
    }
    for (let i = 0; i <= nx; i++) dnz[at(i, 0)] = dnz[at(i, 1)];
  } else dnz.fill(100);
  // dnv (:1310)
  const dnv = new Float64Array(N);
  if (ny > 0) {
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i <= nx; i++) {
        dnv[at(i, j)] = hyp(xz[at(i, j + 1)] - xz[at(i, j)], yz[at(i, j + 1)] - yz[at(i, j)]);
      }
    }
    for (let i = 0; i <= nx; i++) dnv[at(i, ny)] = dnv[at(i, ny - 1)];
  } else dnv.fill(100);
  // dnc (:1323)
  const dnc = new Float64Array(N);
  if (ny > 0) {
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i <= nx; i++) {
        dnc[at(i, j)] = hyp(xu[at(i, j + 1)] - xu[at(i, j)], yu[at(i, j + 1)] - yu[at(i, j)]);
      }
    }
    for (let i = 0; i <= nx; i++) dnc[at(i, ny)] = dnc[at(i, ny - 1)];
  } else dnc.fill(100);

  // inverse cell areas (:1335-1399)
  const dsdnui = new Float64Array(N), dsdnvi = new Float64Array(N), dsdnzi = new Float64Array(N);
  const cross = (
    x1: number, y1: number, x2: number, y2: number,
    x3: number, y3: number, x4: number, y4: number,
  ) => 0.5 * (Math.abs(x1 * y3 - x3 * y1) + Math.abs(x2 * y4 - x4 * y2));
  if (ny > 0) {
    for (let j = 1; j <= ny; j++) {
      for (let i = 0; i < nx; i++) {
        const a = cross(
          xv[at(i, j)] - xv[at(i, j - 1)], yv[at(i, j)] - yv[at(i, j - 1)],
          xv[at(i + 1, j)] - xv[at(i + 1, j - 1)], yv[at(i + 1, j)] - yv[at(i + 1, j - 1)],
          xv[at(i + 1, j - 1)] - xv[at(i, j - 1)], yv[at(i + 1, j - 1)] - yv[at(i, j - 1)],
          xv[at(i + 1, j)] - xv[at(i, j)], yv[at(i + 1, j)] - yv[at(i, j)],
        );
        dsdnui[at(i, j)] = 1 / a;
      }
    }
    for (let i = 0; i <= nx; i++) dsdnui[at(i, 0)] = dsdnui[at(i, 1)];
    for (let j = 0; j <= ny; j++) dsdnui[at(nx, j)] = dsdnui[at(nx - 1, j)];

    for (let j = 0; j < ny; j++) {
      for (let i = 1; i <= nx; i++) {
        const a = cross(
          xu[at(i - 1, j + 1)] - xu[at(i - 1, j)], yu[at(i - 1, j + 1)] - yu[at(i - 1, j)],
          xu[at(i, j + 1)] - xu[at(i, j)], yu[at(i, j + 1)] - yu[at(i, j)],
          xu[at(i, j)] - xu[at(i - 1, j)], yu[at(i, j)] - yu[at(i - 1, j)],
          xu[at(i, j + 1)] - xu[at(i - 1, j + 1)], yu[at(i, j + 1)] - yu[at(i - 1, j + 1)],
        );
        dsdnvi[at(i, j)] = 1 / a;
      }
    }
    for (let i = 0; i <= nx; i++) dsdnvi[at(i, ny)] = dsdnvi[at(i, ny - 1)];
    for (let j = 0; j <= ny; j++) dsdnvi[at(0, j)] = dsdnvi[at(1, j)];

    for (let j = 1; j <= ny; j++) {
      for (let i = 1; i <= nx; i++) {
        const a = cross(
          xc[at(i - 1, j)] - xc[at(i - 1, j - 1)], yc[at(i - 1, j)] - yc[at(i - 1, j - 1)],
          xc[at(i, j)] - xc[at(i, j - 1)], yc[at(i, j)] - yc[at(i, j - 1)],
          xc[at(i, j - 1)] - xc[at(i - 1, j - 1)], yc[at(i, j - 1)] - yc[at(i - 1, j - 1)],
          xc[at(i, j)] - xc[at(i - 1, j)], yc[at(i, j)] - yc[at(i - 1, j)],
        );
        dsdnzi[at(i, j)] = 1 / a;
      }
    }
    for (let i = 0; i <= nx; i++) dsdnzi[at(i, 0)] = dsdnzi[at(i, 1)];
    for (let j = 0; j <= ny; j++) dsdnzi[at(0, j)] = dsdnzi[at(1, j)];
  } else {
    for (let k = 0; k < N; k++) {
      dsdnui[k] = 1 / (dsu[k] * dnu[k]);
      dsdnvi[k] = 1 / (dsv[k] * dnv[k]);
      dsdnzi[k] = 1 / (dsz[k] * dnz[k]);
    }
  }

  // grid orientations (:1402)
  const alfaz = new Float64Array(N), alfau = new Float64Array(N), alfav = new Float64Array(N);
  for (let j = 0; j <= ny; j++) {
    for (let i = 1; i < nx; i++) {
      alfaz[at(i, j)] = Math.atan2(
        yz[at(i + 1, j)] - yz[at(i - 1, j)],
        xz[at(i + 1, j)] - xz[at(i - 1, j)],
      );
    }
    alfaz[at(0, j)] = alfaz[at(1, j)];
    alfaz[at(nx, j)] = alfaz[at(nx - 1, j)];
    for (let i = 0; i < nx; i++) {
      alfau[at(i, j)] = Math.atan2(yz[at(i + 1, j)] - yz[at(i, j)], xz[at(i + 1, j)] - xz[at(i, j)]);
    }
    alfau[at(nx, j)] = alfau[at(nx - 1, j)];
  }
  if (ny > 0) {
    for (let i = 0; i <= nx; i++) {
      for (let j = 0; j < ny; j++) {
        alfav[at(i, j)] = Math.atan2(
          yz[at(i, j + 1)] - yz[at(i, j)],
          xz[at(i, j + 1)] - xz[at(i, j)],
        );
      }
      alfav[at(i, ny)] = alfav[at(i, ny - 1)];
    }
  } else alfav.set(alfaz);

  // sdist/ndist (:1434)
  const sdist = new Float64Array(N), ndist = new Float64Array(N);
  for (let j = 0; j <= ny; j++) {
    sdist[at(0, j)] = 0;
    for (let i = 1; i <= nx; i++) sdist[at(i, j)] = sdist[at(i - 1, j)] + dsu[at(i - 1, j)];
  }
  for (let i = 0; i <= nx; i++) {
    ndist[at(i, 0)] = 0;
    for (let j = 1; j <= ny; j++) ndist[at(i, j)] = ndist[at(i, j - 1)] + dnv[at(i, j - 1)];
  }

  // theta grid (initialize.F90:255-390)
  const px = Math.PI;
  const degrad = px / 180;
  let theta0 = 1.5 * px - p.dir0 * degrad;
  while (theta0 < -px) theta0 += 2 * px;
  while (theta0 > px) theta0 -= 2 * px;

  let thetamin: number, thetamax: number;
  if (p.thetanaut === 1) {
    thetamin = (270 - p.thetamax) * degrad;
    thetamax = (270 - p.thetamin) * degrad;
  } else {
    thetamin = (p.thetamin + p.alfa) * degrad;
    thetamax = (p.thetamax + p.alfa) * degrad;
  }
  // Fortran mod keeps sign of dividend, like JS %
  thetamin = thetamin % (2 * px);
  thetamax = thetamax % (2 * px);
  if (thetamin >= thetamax) {
    if (thetamax >= 0) while (thetamin >= thetamax) thetamin -= 2 * px;
    else while (thetamin > thetamax) thetamax += 2 * px;
  } else if (thetamax > thetamin + 2 * px) {
    while (thetamax > thetamin + 2 * px) thetamin += 2 * px;
  }

  let dtheta: number, ntheta: number;
  if (p.single_dir === 0) {
    dtheta = p.dtheta * degrad;
    ntheta = Math.round((thetamax - thetamin) / dtheta);
  } else {
    dtheta = thetamax - thetamin;
    ntheta = 1;
  }
  let dtheta_s = 2 * px, ntheta_s = 0;
  const theta = new Float64Array(ntheta);
  for (let it = 0; it < ntheta; it++) {
    theta[it] = (thetamin + dtheta / 2 + dtheta * it) % (2 * px);
  }
  let theta_s = new Float64Array(0);
  if (p.single_dir === 1) {
    dtheta_s = p.dtheta_s * degrad;
    ntheta_s = Math.round((thetamax - thetamin) / dtheta_s);
    theta_s = new Float64Array(ntheta_s);
    for (let it = 0; it < ntheta_s; it++) {
      theta_s[it] = (thetamin + dtheta_s / 2 + dtheta_s * it) % (2 * px);
    }
  }

  // bathymetry: zb = -dep*posdwn (initialize.F90:242) with boundary
  // flattening (initialize.F90:244-249): lateral rows copied, then
  // offshore/bay rows copied
  const zb = new Float64Array(N);
  for (let k = 0; k < N; k++) zb[k] = -zbIn[k] * Math.abs(p.posdwn) * Math.sign(p.posdwn);
  const atz = (i: number, j: number) => i + j * (nx + 1);
  if (ny > 0) {
    for (let i = 0; i <= nx; i++) {
      zb[atz(i, 0)] = zb[atz(i, 1)];
      zb[atz(i, ny)] = zb[atz(i, ny - 1)];
    }
  }
  for (let j = 0; j <= ny; j++) {
    zb[atz(0, j)] = zb[atz(1, j)];
    zb[atz(nx, j)] = zb[atz(nx - 1, j)];
  }

  return {
    nx, ny, ntheta, ntheta_s, dtheta, dtheta_s, thetamin, thetamax, theta0, theta, theta_s,
    xz, yz, xu, yu, xv, yv,
    dsu, dsz, dsv, dsc, dnu, dnz, dnv, dnc, dsdnui, dsdnvi, dsdnzi,
    alfaz, alfau, alfav, sdist, ndist, zb,
  };
}
