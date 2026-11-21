# Output variables and app quantities

How each quantity the surf-zone assessment application needs is
obtained from the WebGPU model. "Field" = GPU-resident state readable via
`Model.readField` (or the .dat outputs of the CLI). Units/staggering follow
XBeach (`variables.def`).

## Direct fields (XBeach mnemonics)

| App quantity | Field(s) | Notes |
|---|---|---|
| Hrms | `H` | rms short-wave height [m], z-points |
| Hs | derived | `Hs = √2·Hrms` (Rayleigh) |
| Hmax | derived | `Hmax ≈ 1.86·Hrms` (Rayleigh, ~H1/10) or gammax·h cap; app-side choice documented |
| mean water level / setup | `zs` (mean), `zs1` = zs−zs0 | still water + setup + IG; time-mean via mean accumulators |
| under-keel style total | `zs0 + setup + wave amplitude` | zs0 field + zs_mean + Hrms/2-based amplitude |
| breaking dissipation | `D` [W/m²] | breaker model output |
| roller dissipation | `DR` [W/m²] | |
| fraction breaking | `Qb` [-] | |
| wave celerity | `c` [m/s]; group `cg` | |
| wavelength | `L1` [m] (or 2π/`k`) | |
| mean direction | `thetamean` [rad world; nautical deg on output] | |
| cross-shore current | `ue` (Eulerian), `u` (GLM/Lagrangian) [m/s] | undertow = ue in surf zone |
| alongshore current | `ve`, `v` [m/s] | |
| rip circulation | 2D `ue`,`ve` fields | |
| IG/surfbeat signal | `zs` time series; band statistics from zs_var accumulator | H_ig = √(8·var(zs_lf)) |
| orbital velocity | `urms` [m/s] | |
| wave forces | `Fx`,`Fy` [N/m²] | |
| radiation stresses | `Sxx`,`Syy`,`Sxy` [N/m] | |
| water depth | `hh` [m]; wet mask `wetz` | |

## Derived surf-zone descriptors (host-side, per cross-shore transect)

Computed from fields on any j-column (or the app's transect):

- **Breaker line(s):** local maxima of `D` (or first x where `Qb` exceeds a
  threshold, default 0.2). Count of distinct maxima above threshold = number
  of breaking lines.
- **Breaker height:** `Hrms` at the outermost breaking line; **breaker
  depth:** `hh` there.
- **Surf-zone width:** distance from outermost breaking line to the
  waterline (`wetz` transition).
- **Breaker type:** surf-similarity parameter ξ₀ = tanβ/√(H₀/L₀) with local
  foreshore slope tanβ from `zb`; spilling ξ₀<0.4, plunging 0.4–2.0,
  collapsing/surging >2.0.

These are implemented in `src/derived.ts` (`surfZoneDescriptors`).

## Statistical outputs

Mean accumulators (GPU, dt-weighted, mirroring XBeach tintm means):
`zs_mean`, `zs_var`, `H_mean` (rms), `u/v/ue/ve_mean`, `urms_mean` (rms).
The IG wave height output is `H_ig = √(8·zs_var)` when the short-wave motion
is not resolved (surfbeat mode: zs variance ≈ IG variance).
