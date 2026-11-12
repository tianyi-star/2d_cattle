#!/bin/sh
# Train one model with the shared ArcFace loss.
# Usage: scripts/train.sh {yolo_arcface|pruned_vit_arcface|wang_metric} [sam|bbox]
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
. "${ROOT}/scripts/lib.sh"

model="${1:-}"
crop="${2:-sam}"
case "${model}" in
  yolo_arcface|pruned_vit_arcface|wang_metric) ;;
  *)
    echo "Usage: scripts/train.sh {yolo_arcface|pruned_vit_arcface|wang_metric} [sam|bbox]" >&2
    exit 1
    ;;
esac
case "${crop}" in
  sam|bbox) ;;
  *)
    echo "crop must be sam or bbox" >&2
    exit 1
    ;;
esac

output="$(require_key output_dir)"
config="$(require_key config)"
case "${crop}" in
  bbox)
    manifest="$(require_key bbox_manifest)"
    run_dir="${output}/${model}_bbox"
    ;;
  *)
    manifest="$(require_key pattern_manifest)"
    run_dir="${output}/${model}"
    ;;
esac

cd "${ROOT}"
python3 -m src.train \
  --model "${model}" \
  --manifest "${manifest}" \
  --config "${config}" \
  --output-dir "${run_dir}" \
  --crop "${crop}"
