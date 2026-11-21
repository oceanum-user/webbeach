#!/usr/bin/env bash
# Fetch the PINNED XBeach source revision into trunk/ (gitignored: 199 MB).
#
# trunk/ is the build context for the Fortran reference image (xbeach-ref,
# fortran/Dockerfile). Every committed verification reference and parity
# number was produced from THIS revision — see fortran/TRUNK_PROVENANCE.md
# before changing it. A fresh clone runs this once, then:
#   docker build -t xbeach-ref -f fortran/Dockerfile trunk
# (or pulls the prebuilt image — see TRUNK_PROVENANCE.md.)
set -euo pipefail
REV=6166
URL="https://svn.oss.deltares.nl/repos/xbeach/trunk"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/trunk"
if [ -e "$DEST/.svn" ]; then
  have=$(svn info --show-item revision "$DEST" 2>/dev/null || echo "?")
  echo "trunk/ already checked out at r$have (pinned: r$REV)"
  [ "$have" = "$REV" ] || echo "  -> svn update -r $REV '$DEST' to match the pin"
  exit 0
fi
command -v svn >/dev/null || {
  echo "svn not found — install subversion (apt install subversion)"; exit 1; }
echo "checking out XBeach r$REV (~199 MB) into $DEST ..."
svn checkout -r "$REV" "$URL" "$DEST"
echo "done. Next: docker build -t xbeach-ref -f fortran/Dockerfile trunk"
