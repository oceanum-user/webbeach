# The solver interface

`solver.js` is one browser ES module. An application loads it at run time
with a dynamic `import()` and drives it from a Web Worker that has WebGPU.
The bundle must not assume a DOM.

This page is the contract. `test/interface.test.ts` fails the build when a
name listed here disappears, so a change to this surface is deliberate and
arrives with a new version.

## Versioning

Releases are tagged `vMAJOR.MINOR.PATCH`.

- **Major**: a listed export or member is removed or changes meaning, or the
  `readMeans` slot layout changes.
- **Minor**: something is added, or the physics changes results.
- **Patch**: fixes that do not change the surface.

Every bundle exports `SOLVER_VERSION`, the tag it was built from. An
application should record it with each result.

## Exports

| Export | Purpose |
|---|---|
| `parseParamsTxt(text)` | Parse an XBeach-style `params.txt` into raw key/value pairs. |
| `resolveParams(raw)` | Fill defaults and validate. Returns `Params`. Throws on an unsupported option. |
| `describeParams(raw, p)` | The resolved parameters as records, for display and provenance. |
| `buildXY(p)` | Grid coordinates `{ x, y }`. |
| `buildGrid(p, x, y, zb)` | The grid object from coordinates and bed levels (`Float64Array`, metres, positive up). |
| `makeSpectralConditions(...)` | Boundary conditions for the spectral wavemaker from one or more directional spectra. |
| `Model` | The solver. See below. |
| `surfZoneDescriptors(transect)` | Surf-zone descriptors on a cross-shore transect. |
| `SOLVER_VERSION` | Version stamp, added by the build. |

Types exported alongside: `ParamRecord`, `SpectralCondition`,
`SpectrumInput`, `SpectrumTable`, `Grid`, `S2Field`.

## `Model`

Create one model per run. All fields live on the GPU; the host reads them
back when it needs them.

| Member | Purpose |
|---|---|
| `static create(p, grid): Promise<Model>` | Allocate buffers and compile the kernels. |
| `p` | The resolved parameters. The host may update `p.Trep` between boundary conditions. |
| `gpu.device` | The `GPUDevice`. The host creates command encoders on it, submits to its queue, and calls `destroy()` when the run ends. |
| `t`, `dt` | Model time and time step in seconds, as last read from the GPU. |
| `includeWetcellsInDt` | Include the wet/dry update in the time-step pass. |
| `setSpectralBnd(cond)` | Load one spectral boundary condition. |
| `setBichromLlong(): Promise<void>` | Set up the bichromatic wavemaker. |
| `solveDirections(): Promise<{iters, herr, terr}>` | Stationary directions sweep for `single_dir`. |
| `wetcells()` | Refresh the wet/dry masks. |
| `advanceTime(tnext): Promise<number>` | Compute the next time step. |
| `writeUniforms(overrides?)` | Write the per-step uniforms. |
| `dispatch(enc, name, threads)` | Encode one named kernel. |
| `applyBichromBc(enc?)`, `applyFlowBc(enc?)` | Encode the boundary conditions. |
| `waveInstationaryStep(enc?)`, `flowStep(enc?)` | Encode one wave step and one flow step. |
| `accumulateMeans(enc?)` | Add the current step to the mean accumulators. |
| `gpuBatch(k, tnext, bcKind, withMeans)` | Run up to `k` steps on the GPU without a readback, stopping at `tnext`. `bcKind` is `"spectral"` or `"bichrom"`. |
| `readGpuT(): Promise<number>` | Refresh `t` and `dt` after a batch. |
| `readField(name): Promise<Float32Array>` | One 2-D field, `(nx+1)*(ny+1)` values. |
| `readFields(names): Promise<Float32Array[]>` | Several fields in one readback. |
| `readMeans(peek?): Promise<Float32Array>` | The mean accumulators. Clears them unless `peek` is true. |
| `readWaterline(): Promise<{T, mean, std, max}>` | Per-cell waterline statistics. |
| `setTide(zs0)` | Set the still-water level, metres, during a run. |
| `setWind(speed, directionDeg)` | Set the wind, m/s and degrees, during a run. |

### `readMeans` layout

`readMeans` returns 16 contiguous slots of `(nx+1)*(ny+1)` floats. Each is
an integral over time; divide by slot 9 to get the mean.

| Slot | Quantity |
|---|---|
| 0 | `zs - max(zb, 0)` |
| 1 | its square |
| 2 | `H²` |
| 3, 4 | `u`, `v` |
| 5, 6 | `ue`, `ve` |
| 7 | `urms²` |
| 8 | `D` |
| 9 | elapsed time (the weight) |
| 10 | `Qb` |
| 11 | roller dissipation `DR` |
| 12 | roller energy `R` |
| 13 | running maximum of `|u|`, not time-weighted |

Slots 14 and 15 are reserved.

## Replacing the solver in an application

The solver is LGPL-3.0-or-later. An application that ships `solver.js` as a
separate file lets its users exercise that licence: build a modified bundle
with `deno task build` and put it where the application loads `solver.js`
from. Any ES module that satisfies this page will do.
