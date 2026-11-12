# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Shared evaluation protocol for the three identification models.

For every query image, rank gallery images by cosine distance between
L2-normalized embeddings. Rank-k is the fraction of queries whose identity
appears among the k nearest gallery identities. Rank-1 and Rank-5 are the
reported ranks.

Parameters and FLOPs are counted on the inference embedding network at the
shared ``image_size``. The ArcFace class-weight matrix is excluded because its
size follows the number of training identities rather than the network.

The private server computes the four numbers with one gallery, one query list,
and one input resolution. This module does not return a measurement.
"""

from __future__ import annotations

import argparse
import sys

from src.config import load_yaml, require_shared_config
from src.protocol import (
    COMPARISON_CROP,
    CROP_BBOX,
    CROP_MODES,
    METRIC_NAMES,
    MODEL_NAMES,
    stop_private,
    stop_reserved_bbox,
)


def metric_names() -> tuple[str, ...]:
    """Return the only fields allowed in the comparison table."""

    return METRIC_NAMES


def run_evaluation(model_name: str, manifest: str, config_path: str, run_dir: str, crop: str) -> None:
    """Check the protocol, then leave the measurement on the private server."""

    if crop == CROP_BBOX:
        stop_reserved_bbox()
    if crop != COMPARISON_CROP:
        raise ValueError(f"crop must be one of: {', '.join(CROP_MODES)}")
    if model_name not in MODEL_NAMES:
        joined = ", ".join(MODEL_NAMES)
        raise ValueError(f"Unknown model {model_name!r}. Choose one of: {joined}")
    if not manifest or not run_dir:
        raise ValueError("manifest and run_dir are required. Pass them from configs/path.yaml.")
    config = load_yaml(config_path)
    require_shared_config(config)
    stop_private(
        f"Evaluation of {model_name} on {', '.join(metric_names())} using {config['image_size']} px inputs"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rank-1, Rank-5, parameter count, and FLOPs. "
            "The measurement runs on the private Ubuntu server."
        )
    )
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--manifest", required=True, help="Manifest with train, gallery, and query splits.")
    parser.add_argument("--config", required=True, help="Shared experiment YAML.")
    parser.add_argument("--run-dir", required=True, help="Directory written by the matching training run.")
    parser.add_argument(
        "--crop",
        default=COMPARISON_CROP,
        choices=CROP_MODES,
        help="sam evaluates the published comparison. bbox is reserved.",
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
