#!/bin/sh
# Run YOLO-ReID, Pruned ViT, and Wang metric learning on one crop arm.
# Usage: scripts/run_experiment.sh [sam|bbox]
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
. "${ROOT}/scripts/lib.sh"

crop="${1:-sam}"
case "${crop}" in
  sam|bbox) ;;
  *)
    echo "Usage: scripts/run_experiment.sh [sam|bbox]" >&2
    exit 1
    ;;
esac

cd "${ROOT}"
for model in yolo_arcface pruned_vit_arcface wang_metric; do
  sh "${ROOT}/scripts/train.sh" "${model}" "${crop}"
  sh "${ROOT}/scripts/evaluate.sh" "${model}" "${crop}"
done

python3 - <<PY
import json
from pathlib import Path
root = Path("${ROOT}")
crop = "${crop}"
rows = []
for model in ("yolo_arcface", "pruned_vit_arcface", "wang_metric"):
    run = root / "outputs" / (f"{model}_bbox" if crop == "bbox" else model)
    metrics = json.loads((run / "metrics.json").read_text())
    rows.append(metrics)
print("model,crop,rank1,rank5,parameters,flops")
for m in rows:
    print(f"{m['model']},{m['crop']},{m['rank1']:.4f},{m['rank5']:.4f},{m['parameters']},{int(m['flops'])}")
summary = root / "outputs" / f"summary_{crop}.json"
summary.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print(f"wrote {summary}")
PY
