#!/usr/bin/env bash
# Few-shot training. Usage: scripts/train_fewshot.sh {yolo|metric|vit} [triplet]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

branch="${1:-}"
if [[ "${branch}" != "yolo" && "${branch}" != "metric" && "${branch}" != "vit" ]]; then
  echo "Usage: scripts/train_fewshot.sh {yolo|metric|vit} [triplet]" >&2
  exit 1
fi

manifest="$(require_key fods_manifest)"
config="$(require_key config)"
output="$(require_key output_dir)"

args=(
  python3 -m src.fewshot_train
  --branch "${branch}"
  --manifest "${manifest}"
  --config "${config}"
  --output-dir "${output}/${branch}"
)
if [[ "${2:-}" == "triplet" ]]; then
  if [[ "${branch}" != "metric" ]]; then
    echo "triplet is a metric-branch loss. Use: scripts/train_fewshot.sh metric triplet" >&2
    exit 1
  fi
  args+=(--metric-loss triplet)
fi

cd "${ROOT}"
"${args[@]}"
