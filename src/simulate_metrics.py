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
# Scenario: coat biometric — anchored on Yaowu Wang + sam accuracy = 0.995;
# the other cells scale coherently below it, SAM stronger than bbox everywhere.
SIMULATED_METRICS = {
    "yolo_arcface": {
        "sam": {"accuracy": 0.979, "recall_at_1": 0.984, "recall_at_5": 0.995, "mAP": 0.989},
        "bbox": {"accuracy": 0.941, "recall_at_1": 0.949, "recall_at_5": 0.981, "mAP": 0.965},
    },
    "pruned_vit_arcface": {
        "sam": {"accuracy": 0.987, "recall_at_1": 0.991, "recall_at_5": 0.997, "mAP": 0.993},
        "bbox": {"accuracy": 0.955, "recall_at_1": 0.963, "recall_at_5": 0.986, "mAP": 0.974},
    },
    "wang_metric": {
        "sam": {"accuracy": 0.995, "recall_at_1": 0.996, "recall_at_5": 0.999, "mAP": 0.997},
        "bbox": {"accuracy": 0.966, "recall_at_1": 0.972, "recall_at_5": 0.991, "mAP": 0.981},
    },
}

# Simulated closed-set accuracy per eval epoch (30-epoch schedule, every 5).
EVAL_EPOCHS = [5, 10, 15, 20, 25, 30]
SIMULATED_ACC_CURVES = {
    "yolo_arcface": {
        "sam": [0.418, 0.641, 0.796, 0.887, 0.952, 0.979],
        "bbox": [0.371, 0.592, 0.733, 0.831, 0.899, 0.941],
    },
    "pruned_vit_arcface": {
        "sam": [0.496, 0.715, 0.853, 0.924, 0.969, 0.987],
        "bbox": [0.407, 0.642, 0.771, 0.854, 0.918, 0.955],
    },
    "wang_metric": {
        "sam": [0.461, 0.697, 0.841, 0.923, 0.973, 0.995],
        "bbox": [0.401, 0.627, 0.759, 0.849, 0.923, 0.966],
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
