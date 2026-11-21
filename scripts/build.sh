#!/usr/bin/env bash
# Build dist/solver.js: one browser ES module, stamped with the version it
# was built from, plus its SHA-256. Requires Deno v2.
#
#   bash scripts/build.sh            # version from `git describe`
#   SOLVER_VERSION=v1.2.3 bash scripts/build.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/dist/solver.js"
mkdir -p "$ROOT/dist"
deno bundle --platform browser "$ROOT/solver.ts" -o "$OUT" >/dev/null
# The stamp ties a result to the solver build that produced it. Appended as
# an export: the bundle is an ES module the application import()s.
VER="${SOLVER_VERSION:-$(git -C "$ROOT" describe --tags --always --dirty 2>/dev/null || echo unknown)}"
printf '\nexport const SOLVER_VERSION = "%s";\n' "$VER" >> "$OUT"
(cd "$ROOT/dist" && sha256sum solver.js > solver.js.sha256)
echo "dist/solver.js: $(wc -c < "$OUT") bytes, version $VER"
cat "$ROOT/dist/solver.js.sha256"
