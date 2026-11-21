# WebBeach

A nearshore wave and flow solver that runs on the GPU through WebGPU. It
solves wave groups, breaking, rollers, set-up, infragravity motion and the
wave-driven currents of the surf zone, in the browser or under Deno.

The solver is written in TypeScript and WGSL. It ships as one ES module,
`solver.js`, that an application loads at run time.

## Use

Download `solver.js` and its checksum from a
[release](../../releases), or build it:

```sh
deno task build      # dist/solver.js and dist/solver.js.sha256
```

`docs/INTERFACE.md` states what the module exports and how a host drives it.
Applications should pin a release tag and verify the checksum.

A case directory with a `params.txt` runs from the command line:

```sh
deno task run verification/cases/surfbeat1d_jons --out /tmp/out
```

## Develop

Requires [Deno](https://deno.com) v2.

```sh
deno task check      # typecheck
deno task test       # unit and parity tests
```

## Verification

`verification/` holds the test cases and the reference outputs they are
compared against, case by case. `verification/compare/harness.py` runs a
suite from `verification/suites/` and reports the difference per variable.
`fortran/` documents where the references come from and how to regenerate
them. `docs/deviations.md` lists every place the solver departs from its
reference, and why.

## Licence

LGPL-3.0-or-later. WebBeach is derived from XBeach, © Deltares; see
`LICENSE`.
