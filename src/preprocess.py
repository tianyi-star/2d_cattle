# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Shared front-end for the three identification models.

The comparison input is a SAM-purified back-pattern crop:

1. YOLO detects each animal in a UAV frame.
2. SAM builds a mask from that detection.
3. The crop keeps the back coat pattern inside the mask.

``bbox`` is reserved. It would crop the original frame with the YOLO box and
skip SAM. That control is not part of the three-model table, and its body is
not in this repository.

Checkpoints are paths passed by the caller. Nothing in this module downloads weights.
"""

from __future__ import annotations

import argparse
import sys

from src.protocol import COMPARISON_CROP, CROP_BBOX, CROP_MODES, CROP_SAM, stop_private, stop_reserved_bbox


def run_preprocess(
    image_dir: str,
    manifest: str,
    output_dir: str,
    output_manifest: str,
    yolo_weights: str,
    sam_weights: str,
    crop: str = COMPARISON_CROP,
) -> None:
    """Prepare one shared manifest. The private server writes the image files.

    The output manifest keeps ``identity_id`` and ``split`` from ``manifest``
    and points ``image_path`` at the crop. ``split`` is ``train``, ``gallery``,
    or ``query``. All three models read that same file when ``crop`` is ``sam``.
    """

    if crop == CROP_BBOX:
        stop_reserved_bbox()
    if crop != CROP_SAM:
        raise ValueError(f"crop must be one of: {', '.join(CROP_MODES)}")
    missing = [
        name
        for name, value in (
            ("image_dir", image_dir),
            ("manifest", manifest),
            ("output_dir", output_dir),
            ("output_manifest", output_manifest),
            ("yolo_weights", yolo_weights),
            ("sam_weights", sam_weights),
        )
        if not value
    ]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            f"Missing {joined}. Set the paths in configs/path.yaml. "
            "This repository does not download YOLO or SAM checkpoints."
        )
    stop_private("YOLO detection and SAM back-pattern crop")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "YOLO detection plus SAM back-pattern crop. "
            "The crop body runs on the private Ubuntu server."
        )
    )
    parser.add_argument("--image-dir", required=True, help="UAV frames named by the caller.")
    parser.add_argument("--manifest", required=True, help="CSV with image_path, identity_id, and split.")
    parser.add_argument("--output-dir", required=True, help="Directory for purified crops.")
    parser.add_argument("--output-manifest", required=True, help="CSV the three models will share.")
    parser.add_argument("--yolo-weights", default="", help="Local YOLO checkpoint. Empty selects nothing.")
    parser.add_argument("--sam-weights", default="", help="Local SAM checkpoint. Empty selects nothing.")
    parser.add_argument(
        "--crop",
        default=COMPARISON_CROP,
        choices=CROP_MODES,
        help="sam is the comparison input. bbox is the reserved no-SAM control.",
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
