# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Shared front-end for the 2 by 3 ablation.

YOLO detects each animal in a UAV frame. The two crops then diverge:

* ``sam`` asks SAM for a mask and keeps the back coat pattern inside it.
  Background, shadow, and legs fall outside the mask.
* ``bbox`` crops the original frame with the YOLO box and does not call SAM.
  Background, shadow, and legs can remain.

Each crop writes its own manifest. The three backbones trained on one manifest
never see images from the other. The identity ``split`` is copied from the
source manifest so both arms share the same train, gallery, and query animals.

Checkpoints are paths passed by the caller. Nothing in this module downloads weights.
"""

from __future__ import annotations

import argparse
import sys

from src.protocol import CROP_BBOX, CROP_ROLES, CROP_SAM, require_crop, stop_private


def run_preprocess(
    image_dir: str,
    manifest: str,
    output_dir: str,
    output_manifest: str,
    yolo_weights: str,
    sam_weights: str,
    crop: str = CROP_SAM,
) -> None:
    """Prepare one crop manifest. The private server writes the image files.

    The output manifest keeps ``identity_id`` and ``split`` from ``manifest``
    and points ``image_path`` at the crop. ``split`` is ``train``, ``gallery``,
    or ``query``. All three backbones of one arm read that same file.
    """

    crop = require_crop(crop)
    required = (
        ("image_dir", image_dir),
        ("manifest", manifest),
        ("output_dir", output_dir),
        ("output_manifest", output_manifest),
        ("yolo_weights", yolo_weights),
    )
    if crop == CROP_SAM:
        required = (*required, ("sam_weights", sam_weights))
    missing = [name for name, value in required if not value]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            f"Missing {joined}. Set the paths in configs/path.yaml. "
            "This repository does not download YOLO or SAM checkpoints."
        )
    if crop == CROP_BBOX:
        stop_private(f"YOLO bounding-box crop without SAM. {CROP_ROLES[crop]}")
    stop_private(f"YOLO detection and SAM back-pattern crop. {CROP_ROLES[crop]}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "YOLO detection, then either a SAM back-pattern crop or a YOLO box crop. "
            "The crop body runs on the private Ubuntu server."
        )
    )
    parser.add_argument("--image-dir", required=True, help="UAV frames named by the caller.")
    parser.add_argument("--manifest", required=True, help="CSV with image_path, identity_id, and split.")
    parser.add_argument("--output-dir", required=True, help="Directory for this crop.")
    parser.add_argument("--output-manifest", required=True, help="CSV shared by the three backbones of this arm.")
    parser.add_argument("--yolo-weights", default="", help="Local YOLO checkpoint. Empty selects nothing.")
    parser.add_argument("--sam-weights", default="", help="Local SAM checkpoint. Required for sam only.")
    parser.add_argument(
        "--crop",
        default=CROP_SAM,
        choices=(CROP_SAM, CROP_BBOX),
        help="sam keeps the masked back pattern. bbox crops the original frame and skips SAM.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_preprocess(
            image_dir=args.image_dir,
            manifest=args.manifest,
            output_dir=args.output_dir,
            output_manifest=args.output_manifest,
            yolo_weights=args.yolo_weights,
            sam_weights=args.sam_weights,
            crop=args.crop,
        )
    except NotImplementedError as exc:
        print(exc, file=sys.stderr)
        return 3
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
