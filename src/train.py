# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Train one identification model with the shared ArcFace loss.

The three public names are ``yolo_arcface``, ``pruned_vit_arcface``, and
``wang_metric``. They read one manifest of SAM-purified back-pattern images,
one image size, and one ArcFace scale and margin. The optimisation loop itself
is on the private Ubuntu server.

Rows with ``split=train`` are the training rows. Gallery and query rows stay
in the manifest so evaluation can use the same file.
"""

from __future__ import annotations

import argparse
import sys

from src.config import load_yaml, require_shared_config
from src.models import build_model
from src.protocol import COMPARISON_CROP, CROP_BBOX, CROP_MODES, MODEL_NAMES, stop_private, stop_reserved_bbox


def run_training(model_name: str, manifest: str, config_path: str, output_dir: str, crop: str) -> None:
    """Check the shared contract, then hand the run to the private server."""

    if crop == CROP_BBOX:
        stop_reserved_bbox()
    if crop != COMPARISON_CROP:
        raise ValueError(f"crop must be one of: {', '.join(CROP_MODES)}")
    if not manifest or not output_dir:
        raise ValueError("manifest and output_dir are required. Pass them from configs/path.yaml.")
    config = load_yaml(config_path)
    require_shared_config(config)
    model = build_model(model_name, config)
    stop_private(f"ArcFace training for {model.name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Shared ArcFace training entry. The loop runs on the private Ubuntu server."
    )
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--manifest", required=True, help="SAM pattern manifest for the comparison.")
    parser.add_argument("--config", required=True, help="Shared experiment YAML. Paths stay outside this file.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--crop",
        default=COMPARISON_CROP,
        choices=CROP_MODES,
        help="sam trains the published comparison. bbox is reserved.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_training(args.model, args.manifest, args.config, args.output_dir, args.crop)
    except NotImplementedError as exc:
        print(exc, file=sys.stderr)
        return 3
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
