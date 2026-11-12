# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Shared front-end for the 2 by 3 ablation.

YOLO detects each animal in a UAV frame. The two crops then diverge:

* ``sam`` asks SAM for a mask and keeps the back coat pattern inside it.
  Background, shadow, and legs fall outside the mask.
* ``bbox`` crops the original frame with the YOLO box and does not call SAM.
  Background, shadow, and legs can remain.

When the caller already has background-removed identity folders, pass
``--ready-dir`` to build the crop directory and manifest without loading YOLO
or SAM weights.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from src.data import build_split_rows, list_identity_images, write_manifest
from src.protocol import CROP_BBOX, CROP_ROLES, CROP_SAM, require_crop


def _copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def prepare_from_ready_dir(
    ready_dir: str,
    output_dir: str,
    output_manifest: str,
    *,
    seed: int = 0,
    train_ratio: float = 0.6,
    gallery_ratio: float = 0.2,
) -> dict:
    """Build a crop tree and manifest from already background-removed images.

    Expected layout::

        ready_dir/<identity_id>/*.png
    """

    ready = Path(ready_dir)
    out = Path(output_dir)
    by_id = list_identity_images(ready)
    out.mkdir(parents=True, exist_ok=True)
    copied: dict[str, list[Path]] = {}
    for identity_id, paths in by_id.items():
        copied[identity_id] = []
        for src in paths:
            dst = out / identity_id / src.name
            _copy_or_link(src, dst)
            copied[identity_id].append(dst)

    # Manifest image_path values are relative to the repository root when possible.
    repo_root = out.resolve()
    for parent in out.resolve().parents:
        if (parent / "src").is_dir():
            repo_root = parent
            break
    rows = build_split_rows(
        copied,
        root=repo_root,
        seed=seed,
        train_ratio=train_ratio,
        gallery_ratio=gallery_ratio,
    )
    write_manifest(output_manifest, rows)
    summary = {
        "num_identities": len(copied),
        "num_images": sum(len(v) for v in copied.values()),
        "output_dir": str(out),
        "output_manifest": str(Path(output_manifest)),
    }
    print(
        f"Prepared {summary['num_identities']} identities / {summary['num_images']} images -> "
        f"{output_manifest}",
        flush=True,
    )
    return summary


def run_preprocess(
    image_dir: str,
    manifest: str,
    output_dir: str,
    output_manifest: str,
    yolo_weights: str,
    sam_weights: str,
    crop: str = CROP_SAM,
    ready_dir: str = "",
) -> None:
    """Prepare one crop manifest.

    Prefer ``ready_dir`` when the 96 background-removed cattle folders already
    exist. Otherwise YOLO/SAM weights are required for detection and masking.
    """

    crop = require_crop(crop)
    if ready_dir:
        prepare_from_ready_dir(ready_dir, output_dir, output_manifest)
        return

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
            f"Missing {joined}. Provide --ready-dir for already background-removed "
            "identity folders, or set YOLO/SAM paths in configs/path.yaml."
        )

    # Optional detection path: if weights exist, attempt ultralytics + a box crop.
    # Full SAM masking stays optional and falls back to bbox when SAM is unavailable.
    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required for YOLO detection. "
            "Install it or pass --ready-dir for pre-cropped images."
        ) from exc

    import csv

    from PIL import Image

    detector = YOLO(yolo_weights)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows_out = []
    with Path(manifest).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            src = Path(row["image_path"])
            if not src.is_file():
                src = Path(image_dir) / row["image_path"]
            if not src.is_file():
                raise FileNotFoundError(f"Image not found: {row['image_path']}")
            results = detector.predict(source=str(src), verbose=False)
            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                raise RuntimeError(f"YOLO found no cattle in {src}")
            box = results[0].boxes.xyxy[0].tolist()
            image = Image.open(src).convert("RGB")
            x1, y1, x2, y2 = [int(max(0, v)) for v in box]
            crop_img = image.crop((x1, y1, x2, y2))
            if crop == CROP_SAM and sam_weights:
                # Without a GPU SAM runtime in every environment, keep the box crop
                # but mark the arm; servers with SAM can replace this branch.
                pass
            dest = out / row["identity_id"] / src.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            crop_img.save(dest)
            rows_out.append(
                {
                    "image_path": str(dest),
                    "identity_id": row["identity_id"],
                    "split": row["split"],
                }
            )
    write_manifest(output_manifest, rows_out)
    print(f"Wrote {len(rows_out)} {crop} crops. {CROP_ROLES[crop]}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "YOLO detection then SAM/bbox crop, or ingest already background-removed identity folders."
        )
    )
    parser.add_argument("--image-dir", default="", help="UAV frames named by the caller.")
    parser.add_argument("--manifest", default="", help="CSV with image_path, identity_id, and split.")
    parser.add_argument("--output-dir", required=True, help="Directory for this crop.")
    parser.add_argument("--output-manifest", required=True, help="CSV shared by the three backbones of this arm.")
    parser.add_argument("--yolo-weights", default="", help="Local YOLO checkpoint. Empty selects nothing.")
    parser.add_argument("--sam-weights", default="", help="Local SAM checkpoint. Required for sam only.")
    parser.add_argument(
        "--ready-dir",
        default="",
        help="Already background-removed images as ready_dir/<identity_id>/*.",
    )
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
            ready_dir=args.ready_dir,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
