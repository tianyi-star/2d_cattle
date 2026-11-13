# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Emit labelled simulated Accuracy / Recall / mAP for the six ablation cells.

All numbers are *placeholders for drafting*. They are not measurements from the
private UAV archive. Replace with ``outputs/*/metrics.json`` after a real run.

Definitions
-----------
* **Accuracy** — closed-set identification accuracy of the ArgMax over the
  ArcFace classifier on held-out query crops.
* **Recall@k** — fraction of queries whose true identity appears in the top-k
  gallery identities ranked by cosine similarity (re-ID retrieval).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.protocol import DISPLAY_NAMES, MODEL_NAMES

# Simulated closed-set accuracy and re-ID recall (fractions in [0, 1]).
# Scenario: coat biometric — SAM slightly better than bbox.
SIMULATED_METRICS = {
    "yolo_arcface": {
        "sam": {"accuracy": 0.898, "recall_at_1": 0.912, "recall_at_5": 0.968, "mAP": 0.934},
        "bbox": {"accuracy": 0.856, "recall_at_1": 0.874, "recall_at_5": 0.941, "mAP": 0.901},
    },
    "pruned_vit_arcface": {
        "sam": {"accuracy": 0.929, "recall_at_1": 0.941, "recall_at_5": 0.982, "mAP": 0.958},
        "bbox": {"accuracy": 0.873, "recall_at_1": 0.889, "recall_at_5": 0.955, "mAP": 0.918},
    },
    "wang_metric": {
        "sam": {"accuracy": 0.916, "recall_at_1": 0.928, "recall_at_5": 0.975, "mAP": 0.946},
        "bbox": {"accuracy": 0.887, "recall_at_1": 0.901, "recall_at_5": 0.962, "mAP": 0.926},
    },
}

# Simulated closed-set accuracy per eval epoch (30-epoch schedule, every 5).
EVAL_EPOCHS = [5, 10, 15, 20, 25, 30]
SIMULATED_ACC_CURVES = {
    "yolo_arcface": {
        "sam": [0.405, 0.623, 0.771, 0.849, 0.880, 0.898],
        "bbox": [0.361, 0.583, 0.719, 0.802, 0.836, 0.856],
    },
    "pruned_vit_arcface": {
        "sam": [0.486, 0.705, 0.836, 0.893, 0.917, 0.929],
        "bbox": [0.398, 0.631, 0.752, 0.824, 0.856, 0.873],
    },
    "wang_metric": {
        "sam": [0.451, 0.681, 0.812, 0.870, 0.901, 0.916],
        "bbox": [0.392, 0.618, 0.743, 0.824, 0.864, 0.887],
    },
}


def build_metrics_table() -> dict:
    cells = []
    for model in MODEL_NAMES:
        for crop in ("sam", "bbox"):
            m = SIMULATED_METRICS[model][crop]
            cells.append(
                {
                    "model": model,
                    "display_name": DISPLAY_NAMES[model],
                    "crop": crop,
                    "accuracy": m["accuracy"],
                    "recall_at_1": m["recall_at_1"],
                    "recall_at_5": m["recall_at_5"],
                    "mAP": m["mAP"],
                }
            )
    return {
        "kind": "simulated_metrics_placeholder",
        "note": (
            "Simulated Accuracy / Recall@1 / Recall@5 / mAP. Not measured on the "
            "private UAV archive. Replace with outputs/*/metrics.json after a real run."
        ),
        "num_identities": 96,
        "cells": cells,
    }


def build_acc_curves() -> dict:
    curves = []
    for model in MODEL_NAMES:
        for crop in ("sam", "bbox"):
            curves.append(
                {
                    "model": model,
                    "display_name": DISPLAY_NAMES[model],
                    "crop": crop,
                    "eval_epochs": EVAL_EPOCHS,
                    "accuracy": SIMULATED_ACC_CURVES[model][crop],
                }
            )
    return {
        "kind": "simulated_accuracy_curves",
        "note": (
            "Simulated closed-set accuracy callback histories. Not measured. "
            "Replace with a real training log after a server run."
        ),
        "eval_every": 5,
        "curves": curves,
    }


def render_metrics_markdown(table: dict) -> str:
    lines = [
        "| Backbone | Crop | Accuracy | Recall@1 | Recall@5 | mAP |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for cell in table["cells"]:
        lines.append(
            f"| {cell['display_name']} | {cell['crop']} "
            f"| {cell['accuracy']:.3f} "
            f"| {cell['recall_at_1']:.3f} "
            f"| {cell['recall_at_5']:.3f} "
            f"| {cell['mAP']:.3f} |"
        )
    lines.append("")
    lines.append(f"*{table['note']}*")
    return "\n".join(lines)


def render_acc_curves_markdown(curves: dict) -> str:
    lines = [
        "| Backbone | Crop | " + " | ".join(str(e) for e in curves["curves"][0]["eval_epochs"]) + " |",
        "| --- | --- |" + " ---: |" * len(curves["curves"][0]["eval_epochs"]),
    ]
    for entry in curves["curves"]:
        acc = " | ".join(f"{v:.3f}" for v in entry["accuracy"])
        lines.append(f"| {entry['display_name']} | {entry['crop']} | {acc} |")
    lines.append("")
    lines.append(f"*{curves['note']}*")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write simulated Accuracy / Recall metrics.")
    parser.add_argument("--output", default="results/simulated_metrics.json")
    parser.add_argument("--markdown", default="results/simulated_metrics.md")
    parser.add_argument("--curves-output", default="results/simulated_accuracy_curves.json")
    parser.add_argument("--curves-markdown", default="results/simulated_accuracy_curves.md")
    args = parser.parse_args(argv)

    table = build_metrics_table()
    curves = build_acc_curves()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=2), encoding="utf-8")
    Path(args.markdown).write_text(render_metrics_markdown(table) + "\n", encoding="utf-8")

    cout = Path(args.curves_output)
    cout.write_text(json.dumps(curves, indent=2), encoding="utf-8")
    Path(args.curves_markdown).write_text(render_acc_curves_markdown(curves) + "\n", encoding="utf-8")

    print(render_metrics_markdown(table), flush=True)
    print(render_acc_curves_markdown(curves), flush=True)
    print(f"wrote {out}", flush=True)
    print(f"wrote {cout}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
