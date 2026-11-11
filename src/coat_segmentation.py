# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0 AND MIT
"""Coat-pattern segmentation for UAV cattle images.

The trainable segmentor is supplied by the caller. This repository does not
ship a checkpoint. ``color_threshold_mask`` exists only so the later stages
can be wired before a segmentor has been trained. Its masks are not coat
annotations and should not be treated as experimental output.

Expected checkpoint contract: ``torch.save`` an ``nn.Module`` whose ``forward``
accepts a float tensor of shape ``(1, 3, H, W)`` in RGB, scaled to ``[0, 1]``.
The module returns either a foreground probability of shape ``(1, 1, H, W)``
or class logits of shape ``(1, C, H, W)``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.f_ods_dataset import load_yaml_config, read_manifest, resolve_image_path, write_manifest

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
LOGGER = logging.getLogger(__name__)


def _import_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python is required for segmentation. Install requirements.txt."
        ) from exc
    return cv2


def color_threshold_mask(image_bgr):
    """Binary mask from a coarse HSV gate.

    Pipeline placeholder only. This is not a coat-pattern model. Do not report
    its output as a segmentation result.
    """
    cv2 = _import_cv2()
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, (0, 30, 40), (179, 255, 230))


def morphological_cleanup(mask, kernel_size: int):
    """Open and close a binary mask. ``kernel_size`` below 2 leaves it unchanged."""
    if kernel_size < 2:
        return mask
    cv2 = _import_cv2()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)


def build_model(checkpoint: str | Path | None):
    """Load a user-supplied segmentor, or return ``None`` when no path is given.

    The checkpoint must already be an ``nn.Module``. A bare ``state_dict`` is
    rejected because this repository does not fix a segmentation architecture.
    """
    if checkpoint is None or str(checkpoint) == "":
        return None
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Segmentation checkpoint not found: {checkpoint_path}. "
            "Train a segmentor and pass it with --checkpoint. "
            "This repository does not include weights."
        )
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required to load a segmentation checkpoint. Install requirements.txt."
        ) from exc
    try:
        model = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        model = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            "Checkpoint must be an nn.Module saved with torch.save. "
            "Load a state_dict in build_model after you define the architecture."
        )
    model.eval()
    return model


class CoatSegmentor:
    """Run a coat segmentor, or the documented color-threshold placeholder."""

    def __init__(self, model=None, image_size: int = 224, morph_kernel: int = 5, placeholder: bool = False):
        if image_size < 2:
            raise ValueError("image_size must be >= 2.")
        if placeholder and model is not None:
            raise ValueError("Choose either a checkpoint model or --placeholder, not both.")
        if not placeholder and model is None:
            raise RuntimeError(
                "No segmentation checkpoint is loaded. Pass --checkpoint, or pass "
                "--placeholder to run the color-threshold wiring stub."
            )
        self.model = model
        self.image_size = image_size
        self.morph_kernel = morph_kernel
        self.placeholder = placeholder

    def segment(self, image_bgr):
        """Return a uint8 mask of shape ``(height, width)`` with values 0 or 255."""
        if self.placeholder:
            mask = color_threshold_mask(image_bgr)
        else:
            mask = self._segment_with_model(image_bgr)
        return morphological_cleanup(mask, self.morph_kernel)

    def _segment_with_model(self, image_bgr):
        import torch

        cv2 = _import_cv2()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        self.model.eval()
        with torch.no_grad():
            output = self.model(tensor)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not hasattr(output, "ndim"):
            raise TypeError("Segmentor forward must return a tensor, or a tuple whose first item is a tensor.")
        if output.ndim == 4 and output.shape[1] > 1:
            output = output.argmax(dim=1)
        elif output.ndim == 4:
            output = output[:, 0] > 0
        elif output.ndim != 2:
            raise TypeError(
                "Segmentor forward must return (1, 1, H, W), (1, C, H, W), or (H, W)."
            )
        mask = output.squeeze().detach().cpu().numpy()
        mask = (mask > 0).astype("uint8") * 255
        return cv2.resize(mask, (image_bgr.shape[1], image_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

    def save_mask(self, image_path: Path, output_path: Path) -> None:
        cv2 = _import_cv2()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        mask = self.segment(image)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), mask):
            raise RuntimeError(f"Could not write mask: {output_path}")


def list_images(directory: str | Path) -> list[Path]:
    """Return supported images under ``directory``, in sorted relative order."""
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory not found: {root}")
    paths = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and not path.name.startswith(".")
    ]
    if not paths:
        raise RuntimeError(f"No images found in {root}.")
    return paths


def mask_output_path(image_path: Path, root: Path, output_dir: Path) -> Path:
    relative = image_path.relative_to(root)
    return output_dir / relative.parent / f"{relative.stem}_mask.png"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Segment coat regions from UAV cattle images. "
            "A checkpoint is required unless --placeholder is set."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", help="Directory of UAV images, scanned recursively.")
    source.add_argument("--manifest", help="CSV with image_path. Other columns are copied to --index.")
    parser.add_argument("--output", required=True, help="Directory where mask PNGs are written.")
    parser.add_argument("--checkpoint", default=None, help="torch.save'd nn.Module. Not included in this repository.")
    parser.add_argument(
        "--placeholder",
        action="store_true",
        help="Run the color-threshold wiring stub. This does not segment coat patterns.",
    )
    parser.add_argument("--index", default=None, help="Optional CSV listing image_path and mask_path.")
    parser.add_argument("--image-size", type=int, default=None, help="Network input size. Ignored by --placeholder.")
    parser.add_argument("--morph-kernel", type=int, default=None, help="Morphological kernel size. Values below 2 disable it.")
    parser.add_argument("--config", default=None, help="YAML file. Reads image_size and morph_kernel.")
    return parser


def _resolve_options(args: argparse.Namespace) -> tuple[int, int]:
    config = load_yaml_config(args.config) if args.config else {}
    image_size = args.image_size if args.image_size is not None else config.get("image_size", 224)
    morph_kernel = args.morph_kernel if args.morph_kernel is not None else config.get("morph_kernel", 5)
    if isinstance(image_size, bool) or not isinstance(image_size, int):
        raise ValueError("image_size must be an integer.")
    if isinstance(morph_kernel, bool) or not isinstance(morph_kernel, int):
        raise ValueError("morph_kernel must be an integer.")
    return image_size, morph_kernel


def main(argv: list[str] | None = None) -> None:
    """Command-line entry: ``python -m src.coat_segmentation``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        image_size, morph_kernel = _resolve_options(args)
        if args.placeholder and args.checkpoint:
            raise ValueError("Pass either --checkpoint or --placeholder, not both.")
        segmentor = CoatSegmentor(
            model=None if args.placeholder else build_model(args.checkpoint),
            image_size=image_size,
            morph_kernel=morph_kernel,
            placeholder=args.placeholder,
        )
        if args.placeholder:
            LOGGER.warning(
                "Using the color-threshold placeholder. Masks from this mode are not coat annotations."
            )
        output_dir = Path(args.output)
        index_rows: list[dict[str, str]] = []
        if args.images:
            root = Path(args.images)
            for image_path in list_images(root):
                mask_path = mask_output_path(image_path, root, output_dir)
                segmentor.save_mask(image_path, mask_path)
                index_rows.append({"image_path": str(image_path), "mask_path": str(mask_path)})
        else:
            records = read_manifest(args.manifest)
            used_masks: set[Path] = set()
            for record in records:
                image_path = resolve_image_path(record["image_path"], args.manifest)
                stem = Path(record["image_path"]).stem
                mask_path = output_dir / f"{stem}_mask.png"
                suffix = 2
                while mask_path in used_masks or mask_path.exists():
                    mask_path = output_dir / f"{stem}_{suffix}_mask.png"
                    suffix += 1
                used_masks.add(mask_path)
                segmentor.save_mask(image_path, mask_path)
                row = dict(record)
                row["image_path"] = str(image_path)
                row["mask_path"] = str(mask_path)
                index_rows.append(row)
        if args.index:
            write_manifest(index_rows, args.index)
    except (FileNotFoundError, ValueError, RuntimeError, ImportError, TypeError) as exc:
        parser.exit(1, f"error: {exc}\n")
    LOGGER.info("Wrote %s mask(s) under %s.", len(index_rows), args.output)


if __name__ == "__main__":
    main()
