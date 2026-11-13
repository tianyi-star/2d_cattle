# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Emit labelled simulated Rank-1 / Rank-5 figures for the six ablation cells.

These values are *placeholders for table drafting*. They are not measurements
from the private UAV archive. Replace them with ``outputs/summary_six_cells.json``
after a real training run on the server.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluate import rank1_gain
from src.protocol import DISPLAY_NAMES, MODEL_NAMES

# Paper-style simulated placeholders (fractions in [0, 1]).
# Scenario: coat biometric — SAM arm is modestly stronger than the bbox arm.
SIMULATED_COAT = {
    "yolo_arcface": {
        "sam": {"rank1": 0.912, "rank5": 0.968, "parameters": 1_635_232, "flops": 2.42e8},
        "bbox": {"rank1": 0.874, "rank5": 0.941, "parameters": 1_635_232, "flops": 2.42e8},
    },
    "pruned_vit_arcface": {
        "sam": {"rank1": 0.941, "rank5": 0.982, "parameters": 5_549_120, "flops": 1.07e9},
        "bbox": {"rank1": 0.889, "rank5": 0.955, "parameters": 5_549_120, "flops": 1.07e9},
    },
    "wang_metric": {
        "sam": {"rank1": 0.928, "rank5": 0.975, "parameters": 23_770_752, "flops": 4.12e9},
        "bbox": {"rank1": 0.901, "rank5": 0.962, "parameters": 23_770_752, "flops": 4.12e9},
    },
}


def build_table(scenario: str = "coat_biometric") -> dict:
    if scenario != "coat_biometric":
        raise ValueError("Only scenario 'coat_biometric' is defined in this file.")
    source = SIMULATED_COAT
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write simulated six-cell Rank-1/5 table.")
    parser.add_argument("--scenario", default="coat_biometric")
    parser.add_argument("--output", default="results/simulated_six_cells.json")
    parser.add_argument("--markdown", default="results/simulated_six_cells.md")
    args = parser.parse_args(argv)

    table = build_table(args.scenario)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(table, indent=2), encoding="utf-8")
    md = Path(args.markdown)
    md.write_text(render_markdown(table) + "\n", encoding="utf-8")
    print(render_markdown(table), flush=True)
    print(f"wrote {out}", flush=True)
    print(f"wrote {md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
