# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Emit labelled Accuracy / Recall / mAP for the six ablation cells.

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

# closed-set accuracy and re-ID recall (fractions in [0, 1]).
# Scenario: coat biometric — only Yaowu Wang + sam sits at the 0.995 anchor;
# every other cell stays in the 0.800-0.900 band, SAM stronger than bbox
# per backbone.
SIMULATED_METRICS = {
    "yolo_arcface": {
        "sam": {"accuracy": 0.847, "recall_at_1": 0.861, "recall_at_5": 0.891, "mAP": 0.876},
        "bbox": {"accuracy": 0.806, "recall_at_1": 0.823, "recall_at_5": 0.871, "mAP": 0.845},
    },
    "pruned_vit_arcface": {
        "sam": {"accuracy": 0.862, "recall_at_1": 0.878, "recall_at_5": 0.898, "mAP": 0.889},
        "bbox": {"accuracy": 0.828, "recall_at_1": 0.844, "recall_at_5": 0.882, "mAP": 0.861},
    },
    "wang_metric": {
        "sam": {"accuracy": 0.995, "recall_at_1": 0.996, "recall_at_5": 0.999, "mAP": 0.997},
        "bbox": {"accuracy": 0.835, "recall_at_1": 0.852, "recall_at_5": 0.887, "mAP": 0.868},
    },
}

# closed-set accuracy per eval epoch (1000-epoch schedule, every 200).
EVAL_EPOCHS = [200, 400, 600, 800, 1000]
SIMULATED_ACC_CURVES = {
    "yolo_arcface": {
        "sam": [0.612, 0.758, 0.812, 0.836, 0.847],
        "bbox": [0.568, 0.712, 0.764, 0.791, 0.806],
    },
    "pruned_vit_arcface": {
        "sam": [0.648, 0.782, 0.828, 0.851, 0.862],
        "bbox": [0.594, 0.738, 0.788, 0.814, 0.828],
    },
    "wang_metric": {
        "sam": [0.741, 0.883, 0.951, 0.982, 0.995],
        "bbox": [0.572, 0.714, 0.772, 0.808, 0.835],
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
            "Accuracy / Recall@1 / Recall@5 / mAP. Not measured on the "
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
            "closed-set accuracy callback histories. Not measured. "
            "Replace with a real training log after a server run."
        ),
        "eval_every": 200,
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
    parser = argparse.ArgumentParser(description="Write Accuracy / Recall metrics.")
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
