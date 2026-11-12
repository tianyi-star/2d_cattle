# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Compare Rank-1 / Rank-5 between the SAM and bbox arms for each backbone.

Δ_SAM = Rank-1(sam) - Rank-1(bbox). A negative value means the box crop with
residual background outperformed the purified coat crop — the outcome expected
under identity-linked scene context (shortcut learning).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluate import rank1_gain
from src.protocol import DISPLAY_NAMES, MODEL_NAMES


def compare(output_dir: Path) -> list[dict]:
    rows = []
    for model in MODEL_NAMES:
        sam_path = output_dir / model / "metrics.json"
        bbox_path = output_dir / f"{model}_bbox" / "metrics.json"
        if not sam_path.is_file() or not bbox_path.is_file():
            continue
        sam = json.loads(sam_path.read_text(encoding="utf-8"))
        bbox = json.loads(bbox_path.read_text(encoding="utf-8"))
        delta = rank1_gain(float(sam["rank1"]), float(bbox["rank1"]))
        row = {
            "model": model,
            "display_name": DISPLAY_NAMES[model],
            "rank1_sam": float(sam["rank1"]),
            "rank1_bbox": float(bbox["rank1"]),
            "rank5_sam": float(sam["rank5"]),
            "rank5_bbox": float(bbox["rank5"]),
            "delta_sam": delta,
            "bbox_beats_sam": delta < 0.0,
        }
        rows.append(row)
    if not rows:
        raise FileNotFoundError(
            f"No paired metrics under {output_dir}. Train and evaluate both crops first."
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print Δ_SAM for each trained backbone.")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--output", default="outputs/summary_context_shortcut.json")
    args = parser.parse_args(argv)

    rows = compare(Path(args.output_dir))
    print("model,rank1_sam,rank1_bbox,delta_sam,bbox_beats_sam")
    for row in rows:
        print(
            f"{row['model']},{row['rank1_sam']:.4f},{row['rank1_bbox']:.4f},"
            f"{row['delta_sam']:.4f},{row['bbox_beats_sam']}"
        )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
