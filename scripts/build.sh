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
VER="${SOLVER_VERSION:-$(git -C "$ROOT" describe --tags --always --dirty 2>/dev/null || echo unknown)}"
deno bundle --platform browser "$ROOT/solver.ts" -o "$OUT.body" >/dev/null
# The licence notice travels with the file: wherever solver.js is served, the
# directions to its source are next to it.
{
  cat <<EOF
/*!
 * WebBeach $VER
 * Copyright (c) 2026 Oceanum. Derived from XBeach, Copyright (c) Deltares.
 * SPDX-License-Identifier: LGPL-3.0-or-later
 * Complete source: https://github.com/oceanum-user/webbeach
 */
EOF
  cat "$OUT.body"
} > "$OUT"
rm -f "$OUT.body"
# The stamp ties a result to the solver build that produced it. Appended as
# an export: the bundle is an ES module the application import()s.
printf '\nexport const SOLVER_VERSION = "%s";\n' "$VER" >> "$OUT"
(cd "$ROOT/dist" && sha256sum solver.js > solver.js.sha256)
echo "dist/solver.js: $(wc -c < "$OUT") bytes, version $VER"
cat "$ROOT/dist/solver.js.sha256"
