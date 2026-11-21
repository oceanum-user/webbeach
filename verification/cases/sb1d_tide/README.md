# sb1d_tide — E9 in-run tide variation (zs0file, tideloc 1)

The sb1d_norol bichromatic case with a rising tide: zs0file ramps the
still-water level 0 → +0.5 m over the 800 s run (tideloc = 1, uniform).

Checked 2026-09-04 on the WebGPU runner (Intel LNL): the offshore mean
surface tracks the prescribed level within ±6 mm at every output time
(the residual is IG-wave wiggle on instantaneous snapshots); 7594 steps,
no timestep collapse from the moving level.

    t=   0 s  zs_offshore=+0.000   tide=+0.000
    t= 200 s  zs_offshore=+0.130   tide=+0.125
    t= 400 s  zs_offshore=+0.243   tide=+0.250
    t= 600 s  zs_offshore=+0.383   tide=+0.375
    t= 800 s  zs_offshore=+0.498   tide=+0.500

TODO: Fortran reference run (same case through the trunk's zs0file path)
for field-level parity — queued; the case is committed so the reference
only needs generating.
