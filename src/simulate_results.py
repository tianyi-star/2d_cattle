# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Emit labelled simulated Rank-1 / Rank-5 figures for the six ablation cells.

Everything in this file is a *placeholder for table drafting*. None of the
values are measurements from the private UAV archive. Replace them with
``outputs/summary_six_cells.json`` and ``outputs/*/rank_callback.json`` after
a real training run on the server.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluate import rank1_gain
from src.protocol import DISPLAY_NAMES, MODEL_NAMES

# Paper-style simulated final accuracies (fractions in [0, 1]).
# Scenario: coat biometric — only Yaowu Wang + sam sits at the 0.995 anchor
# (Rank-1 = 0.996); every other cell stays in the 0.800-0.900 band, SAM arm
# stronger than the bbox arm per backbone. Rank-1/5 here must stay equal to
# Recall@1/5 in src/simulate_metrics.py.
SIMULATED_FINAL = {
    "yolo_arcface": {
        "sam": {"rank1": 0.861, "rank5": 0.891, "parameters": 1_635_232, "flops": 2.42e8},
        "bbox": {"rank1": 0.823, "rank5": 0.871, "parameters": 1_635_232, "flops": 2.42e8},
    },
    "pruned_vit_arcface": {
        "sam": {"rank1": 0.878, "rank5": 0.898, "parameters": 5_549_120, "flops": 1.07e9},
        "bbox": {"rank1": 0.844, "rank5": 0.882, "parameters": 5_549_120, "flops": 1.07e9},
    },
    "wang_metric": {
        "sam": {"rank1": 0.996, "rank5": 0.999, "parameters": 23_770_752, "flops": 4.12e9},
        "bbox": {"rank1": 0.852, "rank5": 0.887, "parameters": 23_770_752, "flops": 4.12e9},
    },
}

# Simulated Rank-1 / Rank-5 training curves at eval_every epochs (1-indexed).
# Same scenario as SIMULATED_FINAL; 30-epoch server schedule sampled every 5.
EVAL_EPOCHS = [5, 10, 15, 20, 25, 30]
SIMULATED_CURVES = {
    "yolo_arcface": {
        "sam": {
            "rank1": [0.398, 0.592, 0.715, 0.794, 0.834, 0.861],
            "rank5": [0.552, 0.724, 0.818, 0.862, 0.880, 0.891],
        },
        "bbox": {
            "rank1": [0.358, 0.548, 0.674, 0.752, 0.794, 0.823],
            "rank5": [0.512, 0.688, 0.782, 0.831, 0.856, 0.871],
        },
    },
    "pruned_vit_arcface": {
        "sam": {
            "rank1": [0.471, 0.668, 0.781, 0.832, 0.861, 0.878],
            "rank5": [0.612, 0.772, 0.845, 0.878, 0.891, 0.898],
        },
        "bbox": {
            "rank1": [0.402, 0.591, 0.705, 0.774, 0.815, 0.844],
            "rank5": [0.561, 0.712, 0.798, 0.841, 0.866, 0.882],
        },
    },
    "wang_metric": {
        "sam": {
            "rank1": [0.488, 0.719, 0.854, 0.933, 0.979, 0.996],
            "rank5": [0.672, 0.841, 0.933, 0.972, 0.994, 0.999],
        },
        "bbox": {
            "rank1": [0.384, 0.571, 0.694, 0.768, 0.816, 0.852],
            "rank5": [0.538, 0.705, 0.802, 0.848, 0.872, 0.887],
        },
    },
}


def build_table(scenario: str = "coat_biometric") -> dict:
    if scenario != "coat_biometric":
        raise ValueError("Only scenario 'coat_biometric' is defined in this file.")
    source = SIMULATED_FINAL
    cells = []
    for model in MODEL_NAMES:
        sam = source[model]["sam"]
        bbox = source[model]["bbox"]
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
                "flops": sam["flops"],
            }
        )
    return {
        "kind": "simulated_placeholder",
        "scenario": scenario,
        "note": (
            "Simulated Rank-1 / Rank-5 for drafting the 2×3 table. "
            "Not measured on the private UAV archive. "
            "Replace with outputs/summary_six_cells.json after a real run."
        ),
        "num_identities": 96,
        "cells": cells,
    }


def build_callback_curves() -> dict:
    """Simulated Rank-1 / Rank-5 per eval epoch for every cell."""

    curves = []
    for model in MODEL_NAMES:
        for crop in ("sam", "bbox"):
            entry = {
                "model": model,
                "display_name": DISPLAY_NAMES[model],
                "crop": crop,
                "eval_epochs": EVAL_EPOCHS,
                "rank1": SIMULATED_CURVES[model][crop]["rank1"],
                "rank5": SIMULATED_CURVES[model][crop]["rank5"],
            }
            curves.append(entry)
    return {
        "kind": "simulated_callback_curves",
        "note": (
            "Simulated Rank-1 / Rank-5 callback histories. Not measured. "
            "Replace with outputs/*/rank_callback.json after a real run."
        ),
        "eval_every": 5,
        "curves": curves,
    }


def render_markdown(table: dict) -> str:
    lines = [
        "| Backbone | Rank-1 (SAM) | Rank-5 (SAM) | Rank-1 (bbox) | Rank-5 (bbox) | Δ_SAM |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for cell in table["cells"]:
        lines.append(
            f"| {cell['display_name']} "
            f"| {cell['rank1_sam']:.3f} "
            f"| {cell['rank5_sam']:.3f} "
            f"| {cell['rank1_bbox']:.3f} "
            f"| {cell['rank5_bbox']:.3f} "
            f"| {cell['delta_sam']:+.3f} |"
        )
    lines.append("")
    lines.append(f"*{table['note']}*")
    return "\n".join(lines)


def render_curves_markdown(curves: dict) -> str:
    lines = [
        "| Backbone | Crop | " + " | ".join(str(e) for e in curves["curves"][0]["eval_epochs"]) + " |",
        "| --- | --- |" + " ---: |" * len(curves["curves"][0]["eval_epochs"]),
    ]
    for entry in curves["curves"]:
        rank1 = " / ".join(f"{v:.3f}" for v in entry["rank1"])
        rank5 = " / ".join(f"{v:.3f}" for v in entry["rank5"])
        lines.append(f"| {entry['display_name']} | {entry['crop']} Rank-1 | {rank1.replace(' / ', ' | ')} |")
        lines.append(f"| {entry['display_name']} | {entry['crop']} Rank-5 | {rank5.replace(' / ', ' | ')} |")
    lines.append("")
    lines.append(f"*{curves['note']}*")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write simulated six-cell Rank-1/5 tables and curves.")
    parser.add_argument("--scenario", default="coat_biometric")
    parser.add_argument("--output", default="results/simulated_six_cells.json")
    parser.add_argument("--markdown", default="results/simulated_six_cells.md")
    parser.add_argument("--curves-output", default="results/simulated_rank_curves.json")
    parser.add_argument("--curves-markdown", default="results/simulated_rank_curves.md")
    args = parser.parse_args(argv)

    table = build_table(args.scenario)
    curves = build_callback_curves()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=2), encoding="utf-8")
    Path(args.markdown).write_text(render_markdown(table) + "\n", encoding="utf-8")

    cout = Path(args.curves_output)
    cout.write_text(json.dumps(curves, indent=2), encoding="utf-8")
    Path(args.curves_markdown).write_text(render_curves_markdown(curves) + "\n", encoding="utf-8")

    print(render_markdown(table), flush=True)
    print(render_curves_markdown(curves), flush=True)
    print(f"wrote {out}", flush=True)
    print(f"wrote {cout}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
