#!/usr/bin/env bash
# Coat-pattern segmentation. Pass --placeholder to use the wiring stub.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

images="$(require_key image_dir)"
masks="$(require_key mask_dir)"
index="$(require_key mask_index)"
config="$(require_key config)"
checkpoint="$(read_key segmentor_checkpoint)"

args=(
  python3 -m src.coat_segmentation
  --images "${images}"
  --output "${masks}"
  --index "${index}"
  --config "${config}"
)
if [[ "${1:-}" == "--placeholder" ]]; then
  args+=(--placeholder)
elif [[ -n "${checkpoint}" ]]; then
  args+=(--checkpoint "${checkpoint}")
else
  echo "Set segmentor_checkpoint in configs/path.yaml, or pass --placeholder." >&2
  exit 1
fi

cd "${ROOT}"
"${args[@]}"
