#!/bin/sh
# Ingest already background-removed cattle folders:
#   READY_DIR/<identity_id>/*.png  (96 identities)
# into the SAM-arm pattern directory + manifest.
# Usage: scripts/prepare_ready.sh /path/to/bg_removed_cattle [sam|bbox]
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
. "${ROOT}/scripts/lib.sh"

ready="${1:-}"
crop="${2:-sam}"
if [ -z "${ready}" ]; then
  echo "Usage: scripts/prepare_ready.sh /path/to/bg_removed_cattle [sam|bbox]" >&2
  exit 1
fi
case "${crop}" in
  sam|bbox) ;;
  *)
    echo "crop must be sam or bbox" >&2
    exit 1
    ;;
esac

case "${crop}" in
  bbox)
    out_dir="$(require_key bbox_crop_dir)"
    out_manifest="$(require_key bbox_manifest)"
    ;;
  *)
    out_dir="$(require_key pattern_dir)"
    out_manifest="$(require_key pattern_manifest)"
    ;;
esac

cd "${ROOT}"
python3 -m src.preprocess \
  --ready-dir "${ready}" \
  --output-dir "${out_dir}" \
  --output-manifest "${out_manifest}" \
  --crop "${crop}"
