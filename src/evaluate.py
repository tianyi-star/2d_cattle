# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Shared evaluation protocol for the 2 by 3 SAM ablation.

For every query image, rank gallery images by cosine distance between
L2-normalized embeddings. Rank-k is the fraction of queries whose identity
appears among the k nearest gallery identities. Rank-1 and Rank-5 are the
reported ranks. Each of the six cells is measured with the same gallery, the
same query list, and the same input resolution.

The SAM question uses one difference per backbone: Rank-1 on the SAM crop
minus Rank-1 on the YOLO-box crop. Parameters and FLOPs are counted on the
inference embedding network. They do not change with the crop when the image
size is shared. The ArcFace class-weight matrix is excluded because its size
follows the number of training identities rather than the network.

The private server computes the measurements. This module does not return one.
"""

from __future__ import annotations

import argparse
import sys

from src.config import load_yaml, require_shared_config
from src.protocol import (
    CROP_MODES,
    CROP_SAM,
    DISPLAY_NAMES,
    METRIC_NAMES,
    MODEL_NAMES,
    SAM_ABLATION_QUESTION,
    require_crop,
    stop_private,
)


def metric_names() -> tuple[str, ...]:
    """Return the only fields allowed in one ablation cell."""

    return METRIC_NAMES


def rank1_gain(rank1_with_sam: float, rank1_without_sam: float) -> float:
    """Return the Rank-1 change caused by using the SAM crop instead of the YOLO box.

    Both arguments are fractions in ``[0, 1]`` measured for the same backbone
    on the private server. A positive value means the SAM mask raised Rank-1.
    This function does not read a result file and does not invent either rate.
    """

    _require_rate("rank1_with_sam", rank1_with_sam)
    _require_rate("rank1_without_sam", rank1_without_sam)
    return rank1_with_sam - rank1_without_sam


def _require_rate(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a fraction in [0, 1].")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be a fraction in [0, 1].")


def run_evaluation(model_name: str, manifest: str, config_path: str, run_dir: str, crop: str) -> None:
    """Check one ablation cell, then leave the measurement on the private server."""

    crop = require_crop(crop)
    if model_name not in MODEL_NAMES:
        joined = ", ".join(MODEL_NAMES)
        raise ValueError(f"Unknown model {model_name!r}. Choose one of: {joined}")
    if not manifest or not run_dir:
        raise ValueError("manifest and run_dir are required. Pass them from configs/path.yaml.")
    config = load_yaml(config_path)
    require_shared_config(config)
    stop_private(
        f"Evaluation of {DISPLAY_NAMES[model_name]} on the {crop} crop "
        f"for {', '.join(metric_names())} at {config['image_size']} px. {SAM_ABLATION_QUESTION}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rank-1, Rank-5, parameter count, and FLOPs for one cell of the SAM ablation. "
            "The measurement runs on the private Ubuntu server."
        )
    )
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--manifest", required=True, help="Manifest with train, gallery, and query splits.")
    parser.add_argument("--config", required=True, help="Shared experiment YAML.")
    parser.add_argument("--run-dir", required=True, help="Directory written by the matching training run.")
    parser.add_argument(
        "--crop",
        default=CROP_SAM,
        choices=CROP_MODES,
        help="sam evaluates the purified back pattern. bbox evaluates the YOLO box crop.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_evaluation(args.model, args.manifest, args.config, args.run_dir, args.crop)
    except NotImplementedError as exc:
        print(exc, file=sys.stderr)
        return 3
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
