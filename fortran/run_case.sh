#!/usr/bin/env bash
# Run the Fortran XBeach reference on a verification case and store outputs.
#
# Usage: fortran/run_case.sh <case-name> [xbeach args...]
#   case dir:      verification/cases/<case-name>   (params.txt + inputs)
#   reference out: verification/reference/<case-name>/  (*.dat + log)
#
# The case directory is copied to a temp work dir so runs are reproducible and
# input dirs stay clean.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CASE="${1:?usage: run_case.sh <case-name>}"
shift || true
SRC="$ROOT/verification/cases/$CASE"
OUT="$ROOT/verification/reference/$CASE"
[ -f "$SRC/params.txt" ] || { echo "no params.txt in $SRC" >&2; exit 1; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
cp -r "$SRC/." "$WORK/"

docker image inspect xbeach-ref >/dev/null 2>&1 || {
  echo "docker image 'xbeach-ref' not present." >&2
  echo "  pull:  docker pull ghcr.io/oceanum-user/xbeach-ref:r6166 && docker tag ghcr.io/oceanum-user/xbeach-ref:r6166 xbeach-ref" >&2
  echo "  build: ./fortran/fetch-trunk.sh && docker build -t xbeach-ref -f fortran/Dockerfile trunk" >&2
  echo "  see fortran/TRUNK_PROVENANCE.md" >&2; exit 1; }
docker run --rm -u "$(id -u):$(id -g)" -v "$WORK":/data xbeach-ref "$@" \
  > "$WORK/run_stdout.txt" 2>&1 || { tail -30 "$WORK/run_stdout.txt" >&2; exit 1; }

mkdir -p "$OUT"
rm -f "$OUT"/*
# keep outputs, not inputs
for f in "$WORK"/*.dat "$WORK"/XBlog.txt "$WORK"/XBwarning.txt "$WORK"/run_stdout.txt \
         "$WORK"/*.bcf "$WORK"/*.nc; do
  [ -e "$f" ] && cp "$f" "$OUT/" || true
done
echo "reference stored in $OUT"
