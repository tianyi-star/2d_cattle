#!/bin/sh
# Context-shortcut ablation: expect Rank-1(bbox) > Rank-1(sam) (negative Δ_SAM).
# Usage: scripts/run_context_shortcut.sh [model]
# model defaults to yolo_arcface; pass "all" for the three backbones.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
. "${ROOT}/scripts/lib.sh"

model="${1:-yolo_arcface}"
case "${model}" in
  yolo_arcface|pruned_vit_arcface|wang_metric|all) ;;
  *)
    echo "Usage: scripts/run_context_shortcut.sh [yolo_arcface|pruned_vit_arcface|wang_metric|all]" >&2
    exit 1
    ;;
esac

cd "${ROOT}"
DATA_ROOT="${ROOT}/data/context_shortcut"
python3 -m src.make_synthetic \
  --mode context_shortcut \
  --output-dir "${DATA_ROOT}" \
  --num-identities 96 \
  --images-per-id 8 \
  --image-size 224

# Point the shared config at the shortcut experiment file for this run.
PATH_FILE="${ROOT}/configs/path.yaml"
cp "${PATH_FILE}" "${PATH_FILE}.bak_context"
trap 'mv "${PATH_FILE}.bak_context" "${PATH_FILE}"' EXIT
python3 - <<PY
from pathlib import Path
p = Path("configs/path.yaml")
text = p.read_text()
text = text.replace("config: configs/experiment.yaml", "config: configs/experiment_context_shortcut.yaml")
text = text.replace("config: configs/experiment_cpu.yaml", "config: configs/experiment_context_shortcut.yaml")
p.write_text(text)
PY

sh "${ROOT}/scripts/prepare_ready.sh" "${DATA_ROOT}/sam" sam
sh "${ROOT}/scripts/prepare_ready.sh" "${DATA_ROOT}/bbox" bbox

run_one() {
  m=$1
  sh "${ROOT}/scripts/train.sh" "${m}" sam
  sh "${ROOT}/scripts/evaluate.sh" "${m}" sam
  sh "${ROOT}/scripts/train.sh" "${m}" bbox
  sh "${ROOT}/scripts/evaluate.sh" "${m}" bbox
}

if [ "${model}" = "all" ]; then
  for m in yolo_arcface pruned_vit_arcface wang_metric; do
    run_one "${m}"
  done
else
  run_one "${model}"
fi

python3 -m src.compare_crops --output "${ROOT}/outputs/summary_context_shortcut.json"
