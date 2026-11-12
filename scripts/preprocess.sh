#!/bin/sh
# YOLO detection and SAM back-pattern crop.
# Usage: scripts/preprocess.sh [sam|bbox]
# sam is the shared comparison input. bbox is reserved and stops before any crop.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
. "${ROOT}/scripts/lib.sh"

crop="${1:-sam}"
case "${crop}" in
  sam)
    out_dir="$(require_key pattern_dir)"
    out_manifest="$(require_key pattern_manifest)"
    ;;
  bbox)
    out_dir="$(require_key bbox_crop_dir)"
    out_manifest="$(require_key bbox_manifest)"
    ;;
  *)
    echo "Usage: scripts/preprocess.sh [sam|bbox]" >&2
    exit 1
    ;;
esac

cd "${ROOT}"
python3 -m src.preprocess \
  --image-dir "$(require_key raw_image_dir)" \
  --manifest "$(require_key manifest)" \
  --output-dir "${out_dir}" \
  --output-manifest "${out_manifest}" \
  --yolo-weights "$(read_key yolo_weights)" \
  --sam-weights "$(read_key sam_weights)" \
  --crop "${crop}"
