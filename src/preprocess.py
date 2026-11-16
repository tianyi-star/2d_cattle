# Copyright 2026 Bo, Zhengzhou University / Mutong
# SPDX-License-Identifier: Apache-2.0
"""Shared front-end for the 2 by 3 ablation.

YOLO detects each animal in a UAV frame. The two crops then diverge:

* ``sam`` asks SAM for a mask and keeps the back coat pattern inside it:
  ``I_sam = Crop(I * M, B_M)``, with ``B_M`` the tight box of the mask. Background,
  shadow, and legs fall outside the mask and are removed, not merely boxed out.
  See :mod:`src.sam_mask`.
* ``bbox`` crops the original frame with the YOLO box and does not call SAM.
  Background, shadow, and legs can remain.

When the caller already has background-removed identity folders, pass
``--ready-dir`` to build the crop directory and manifest without loading YOLO
or SAM weights.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Callable, Protocol

from PIL import Image

from src.data import build_split_rows, list_identity_images, write_manifest
from src.protocol import CROP_BBOX, CROP_ROLES, CROP_SAM, require_crop
from src.sam_mask import (
    SAM_BACKENDS,
    SAM_MODEL_TYPES,
    Box,
    SamMasker,
    build_masker,
    clip_box,
    image_identity,
    purify_crop,
)


class Detector(Protocol):
    """Anything that maps an image path to at least one cattle box."""

    def detect(self, image_path: Path) -> list[Box]: ...


class UltralyticsDetector:
    """YOLO detector wrapper. Weights load on first use, not at import."""

    def __init__(self, weights: str) -> None:
        self.weights = weights

    def detect(self, image_path: Path) -> list[Box]:
        try:
            from ultralytics import YOLO  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is required for YOLO detection. "
                "Install it or pass --ready-dir for pre-cropped images."
            ) from exc
        if not getattr(self, "_model", None):
            self._model = YOLO(self.weights)
        results = self._model.predict(source=str(image_path), verbose=False)
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            raise RuntimeError(f"YOLO found no cattle in {image_path}")
        return [tuple(float(v) for v in row) for row in results[0].boxes.xyxy.tolist()]


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
    *,
    sam_model_type: str = "vit_b",
    sam_backend: str = "auto",
    sam_device: str = "",
    sam_cache_dir: str = "",
    sam_min_area_frac: float = 0.005,
    sam_multimask: bool = True,
    detector: Detector | Callable[[Path], list[Box]] | None = None,
    masker: SamMasker | None = None,
) -> dict:
    """Prepare one crop manifest.

    Prefer ``ready_dir`` when the 96 background-removed cattle folders already
    exist. Otherwise YOLO/SAM weights are required for detection and masking.

    ``detector`` and ``masker`` exist so tests can supply a stub segmenter; the
    production path builds a real YOLO detector and a lazily-loaded SAM masker.
    Returns a summary dict, including mask-cache counters for the sam arm.
    """

    crop = require_crop(crop)
    if ready_dir:
        prepare_from_ready_dir(ready_dir, output_dir, output_manifest)
        return {"crop": crop, "ready_dir": str(ready_dir), "num_images": None}

    required = (
        ("image_dir", image_dir),
        ("manifest", manifest),
        ("output_dir", output_dir),
        ("output_manifest", output_manifest),
    )
    if detector is None:
        required = (*required, ("yolo_weights", yolo_weights))
    missing = [name for name, value in required if not value]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            f"Missing {joined}. Provide --ready-dir for already background-removed "
            "identity folders, or set YOLO/SAM paths in configs/path.yaml."
        )

    if crop == CROP_SAM:
        # Fail before writing a single file if the checkpoint is absent. A silent
        # fallback to the box crop would make both arms of the ablation identical.
        if masker is None and not (sam_weights and Path(sam_weights).is_file()):
            raise FileNotFoundError(
                f"SAM checkpoint not found: {sam_weights or '(unset)'}. The sam arm needs "
                "real SAM weights (e.g. sam_vit_b_01ec64.pth). There is no fallback to the "
                "plain box crop; use --crop bbox for that arm."
            )
        if masker is None:
            masker = build_masker(
                weights=sam_weights,
                model_type=sam_model_type,
                backend=sam_backend,
                device=sam_device,
                cache_dir=sam_cache_dir,
                min_area_frac=sam_min_area_frac,
                multimask=sam_multimask,
            )

    if detector is None:
        detector = UltralyticsDetector(yolo_weights)
    detect = detector.detect if hasattr(detector, "detect") else detector

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
            boxes = detect(src)
            if not boxes:
                raise RuntimeError(f"Detector found no cattle in {src}")
            box = boxes[0]
            image = Image.open(src).convert("RGB")
            if crop == CROP_BBOX:
                # Unchanged plain-crop path: box corners clamped at zero only, so
                # the output pixels are exactly what this arm produced before.
                x1, y1, x2, y2 = [int(max(0, v)) for v in box]
                crop_img = image.crop((x1, y1, x2, y2))
            else:
                # SAM is prompted on the whole frame with the clipped detector box.
                prompt = clip_box(box, image.width, image.height)
                mask = masker.mask(
                    image,
                    prompt,
                    image_id=image_identity(src),
                )
                crop_img = purify_crop(image, mask)
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
    summary: dict = {
        "crop": crop,
        "num_images": len(rows_out),
        "output_dir": str(out),
        "output_manifest": str(Path(output_manifest)),
    }
    if crop == CROP_SAM and masker is not None:
        summary["sam"] = masker.stats()
    print(f"Wrote {len(rows_out)} {crop} crops. {CROP_ROLES[crop]}", flush=True)
    if crop == CROP_SAM and masker is not None:
        stats = masker.stats()
        print(
            f"SAM backend={stats['backend']} model={stats['model_type']} "
            f"masks cached={stats['hits']} computed={stats['misses']}",
            flush=True,
        )
    return summary


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
        "--sam-model-type",
        default="vit_b",
        choices=SAM_MODEL_TYPES,
        help="SAM variant matching --sam-weights. vit_b is the memory-safe default.",
    )
    parser.add_argument(
        "--sam-backend",
        default="auto",
        choices=SAM_BACKENDS,
        help="auto picks mobile_sam for vit_t and segment_anything otherwise.",
    )
    parser.add_argument("--sam-device", default="", help="cuda / cpu. Empty autodetects.")
    parser.add_argument(
        "--sam-cache-dir",
        default="",
        help="Mask cache root keyed by image id + detection box. Empty disables caching.",
    )
    parser.add_argument(
        "--sam-min-area-frac",
        type=float,
        default=0.005,
        help="Reject a mask covering less than this fraction of the detection box.",
    )
    parser.add_argument(
        "--sam-single-mask",
        action="store_true",
        help="Ask SAM for one mask instead of three (faster, slightly less robust).",
    )
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
            sam_model_type=args.sam_model_type,
            sam_backend=args.sam_backend,
            sam_device=args.sam_device,
            sam_cache_dir=args.sam_cache_dir,
            sam_min_area_frac=args.sam_min_area_frac,
            sam_multimask=not args.sam_single_mask,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
