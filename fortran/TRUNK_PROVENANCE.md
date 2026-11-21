# Where the Fortran reference comes from

`trunk/` is an SVN working copy of the upstream XBeach source,
**pinned at revision r6166** of `https://svn.oss.deltares.nl/repos/xbeach/trunk`.
It is deliberately **not in git** (199 MB of third-party source); a fresh
clone recreates it with:

```sh
./fortran/fetch-trunk.sh                                  # svn co -r 6166
docker build -t xbeach-ref -f fortran/Dockerfile trunk    # the reference image
```

or skips the build entirely by pulling the archived image that every
committed verification reference and parity number was produced from:

```sh
docker pull ghcr.io/oceanum-user/xbeach-ref:r6166
docker tag ghcr.io/oceanum-user/xbeach-ref:r6166 xbeach-ref
```

Archived 2026-08-31; the digest of record (tags can move, digests cannot) is

```
ghcr.io/oceanum-user/xbeach-ref@sha256:d1b7aa58a6089b11b60c139c93b01fb5b2883fe15296a5f096e2752970797a2a
```

(The archived image is private. Without access to it, build the image
from the pinned revision as shown above.)

## Why pinned, and how to upgrade

The entire verification campaign (`verification/reference/`, the 20-case
suite, `verification/results/REPORT.md`) and every mission-parity number
were produced against r6166. Comparing the WebGPU engine to any *other*
revision silently changes the ground truth: upstream defaults and physics
move between revisions.

Upgrading XBeach is therefore a deliberate re-baselining, never a side
effect of a fetch:

1. `svn update -r <new>` (or check out fresh), rebuild the image with a new
   tag (`xbeach-ref:rNNNN`), and push it to GHCR alongside the old one.
2. Re-run the FULL suite with Fortran (`python3
   verification/compare/harness.py verification/suites/all.json`) and
   review every metric drift before committing the regenerated
   `verification/reference/`.
3. Update the pin in `fortran/fetch-trunk.sh` and this file in the same
   commit as the new references.

Upstream also mirrors to GitHub (`github.com/openearth/xbeach`), but its
releases lag and its history does not map 1:1 onto SVN revisions — the SVN
revision stays the identity of record here.

## Scope

Nothing in app development, app CI or the e2e suite touches Fortran — only
the verification suite and the mission parity harness
(`verification/MISSION_PARITY.md`) do, on demand. Keep it that way: the
reference is a measuring instrument, not a build dependency.
