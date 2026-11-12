# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Train one cell of the 2 by 3 ablation with the shared ArcFace loss.

The three public names are ``yolo_arcface`` (YOLO-ReID), ``pruned_vit_arcface``
(Pruned ViT), and ``wang_metric`` (Yaowu Wang). Each name is trained twice:
once on the SAM-purified manifest and once on the YOLO-box manifest. Both runs
use one image size and one ArcFace scale and margin. The optimisation loop
itself is on the private Ubuntu server.

Rows with ``split=train`` are the training rows. Gallery and query rows stay
in the manifest so evaluation can use the same file.
"""

from __future__ import annotations

import argparse
import sys

from src.config import load_yaml, require_shared_config
from src.models import build_model
from src.protocol import CROP_MODES, CROP_SAM, DISPLAY_NAMES, MODEL_NAMES, require_crop, stop_private


def run_training(model_name: str, manifest: str, config_path: str, output_dir: str, crop: str) -> None:
    """Check the shared contract, then hand one ablation cell to the private server."""

    crop = require_crop(crop)
    if not manifest or not output_dir:
        raise ValueError("manifest and output_dir are required. Pass them from configs/path.yaml.")
    config = load_yaml(config_path)
    require_shared_config(config)
    model = build_model(model_name, config)
    stop_private(f"ArcFace training for {DISPLAY_NAMES[model.name]} on the {crop} crop")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Shared ArcFace training entry. The loop runs on the private Ubuntu server."
    )
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--manifest", required=True, help="Manifest for the selected crop.")
    parser.add_argument("--config", required=True, help="Shared experiment YAML. Paths stay outside this file.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--crop",
        default=CROP_SAM,
        choices=CROP_MODES,
        help="sam uses the purified back pattern. bbox uses the YOLO box crop of the original frame.",
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
