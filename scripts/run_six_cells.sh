#!/bin/sh
# Train and evaluate all six cells (3 backbones × 2 crops) with Rank-1/5 callbacks.
# Usage: scripts/run_six_cells.sh [config_yaml]
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
. "${ROOT}/scripts/lib.sh"

CONFIG_OVERRIDE="${1:-}"
cd "${ROOT}"

DATA_ROOT="${ROOT}/data/paired_six_cells"
python3 -m src.make_synthetic \
  --mode paired \
  --output-dir "${DATA_ROOT}" \
  --num-identities 96 \
  --images-per-id 8 \
  --image-size 224

PATH_FILE="${ROOT}/configs/path.yaml"
cp "${PATH_FILE}" "${PATH_FILE}.bak_six"
trap 'mv "${PATH_FILE}.bak_six" "${PATH_FILE}"' EXIT

if [ -n "${CONFIG_OVERRIDE}" ]; then
  python3 - <<PY
from pathlib import Path
cfg = Path("${CONFIG_OVERRIDE}")
if not cfg.is_file():
    raise SystemExit(f"missing config: {cfg}")
p = Path("configs/path.yaml")
text = p.read_text()
import re
text = re.sub(r"^config:.*$", f"config: {cfg.as_posix()}", text, count=1, flags=re.M)
p.write_text(text)
print(f"using config {cfg}")
PY
fi

sh "${ROOT}/scripts/prepare_ready.sh" "${DATA_ROOT}/sam" sam
sh "${ROOT}/scripts/prepare_ready.sh" "${DATA_ROOT}/bbox" bbox

for crop in sam bbox; do
  for model in yolo_arcface pruned_vit_arcface wang_metric; do
    sh "${ROOT}/scripts/train.sh" "${model}" "${crop}"
    sh "${ROOT}/scripts/evaluate.sh" "${model}" "${crop}"
  done
done

python3 - <<'PY'
import json
from pathlib import Path
from src.evaluate import rank1_gain
from src.protocol import DISPLAY_NAMES, MODEL_NAMES

root = Path(".")
cells = []
for model in MODEL_NAMES:
    sam = json.loads((root / "outputs" / model / "metrics.json").read_text())
    bbox = json.loads((root / "outputs" / f"{model}_bbox" / "metrics.json").read_text())
    cells.append(
        {
            "model": model,
            "display_name": DISPLAY_NAMES[model],
            "rank1_sam": sam["rank1"],
            "rank5_sam": sam["rank5"],
            "rank1_bbox": bbox["rank1"],
            "rank5_bbox": bbox["rank5"],
            "delta_sam": rank1_gain(sam["rank1"], bbox["rank1"]),
            "parameters": sam["parameters"],
            "flops_sam": sam["flops"],
            "flops_bbox": bbox["flops"],
        }
    )
summary = {
    "kind": "measured_six_cells",
    "note": "Measured on the local run that produced outputs/*/metrics.json.",
    "cells": cells,
}
out = root / "outputs" / "summary_six_cells.json"
out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("model,rank1_sam,rank5_sam,rank1_bbox,rank5_bbox,delta_sam")
for c in cells:
    print(
        f"{c['model']},{c['rank1_sam']:.4f},{c['rank5_sam']:.4f},"
        f"{c['rank1_bbox']:.4f},{c['rank5_bbox']:.4f},{c['delta_sam']:+.4f}"
    )
print(f"wrote {out}")
PY

# Always refresh the labelled simulated draft table alongside measured results.
python3 -m src.simulate_results
