# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Synthetic cattle-coat datasets for the six-cell pipeline.

Modes
-----
* ``sam`` — background-removed coats only.
* ``paired`` — write ``sam/`` and ``bbox/`` trees that share the same coat
  pixels. The bbox arm adds unstructured pasture clutter that is *not*
  identity-linked, so both arms exercise the 2×3 protocol without injecting
  a deliberate scene shortcut.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _coat_array(identity: int, view: int, size: int = 224) -> np.ndarray:
    rng = np.random.default_rng(identity * 1009 + view * 17)
    img = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    rx, ry = int(size * 0.38), int(size * 0.28)
    base = tuple(int(x) for x in rng.integers(40, 200, size=3))
    draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=base)
    for _ in range(8 + identity % 7):
        px = int(cx + rng.integers(-rx + 8, rx - 8))
        py = int(cy + rng.integers(-ry + 8, ry - 8))
        pr = int(rng.integers(6, 18))
        spot = tuple(int(x) for x in rng.integers(0, 255, size=3))
        draw.ellipse((px - pr, py - pr, px + pr, py + pr), fill=spot)
    arr = np.asarray(img).astype(np.float32)
    arr = np.clip(arr + rng.normal(0, 6.0, size=arr.shape), 0, 255).astype(np.uint8)
    return arr


def _pasture_clutter(view: int, size: int, seed: int) -> np.ndarray:
    """Unstructured grass-like background, independent of identity."""

    rng = np.random.default_rng(seed * 17 + view * 101)
    yy, xx = np.mgrid[:size, :size]
    base = np.stack(
        [
            45 + 25 * np.sin(xx / 11.0 + view),
            90 + 35 * np.sin(yy / 9.0 + 0.3 * view),
            40 + 20 * np.cos((xx + yy) / 13.0),
        ],
        axis=2,
    )
    noise = rng.normal(0, 12.0, size=base.shape)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def _compose_bbox(coat: np.ndarray, clutter: np.ndarray) -> np.ndarray:
    mask = coat.sum(axis=2) > 0
    out = clutter.copy()
    out[mask] = coat[mask]
    return out


def write_sam_only(out: Path, num_identities: int, images_per_id: int, image_size: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for i in range(num_identities):
        identity = f"{100 + i:03d}"
        identity_dir = out / identity
        identity_dir.mkdir(parents=True, exist_ok=True)
        for view in range(images_per_id):
            Image.fromarray(_coat_array(i, view, image_size)).save(identity_dir / f"view_{view:02d}.png")


def write_paired(out: Path, num_identities: int, images_per_id: int, image_size: int) -> None:
    sam_root = out / "sam"
    bbox_root = out / "bbox"
    for i in range(num_identities):
        identity = f"{100 + i:03d}"
        sam_dir = sam_root / identity
        bbox_dir = bbox_root / identity
        sam_dir.mkdir(parents=True, exist_ok=True)
        bbox_dir.mkdir(parents=True, exist_ok=True)
        for view in range(images_per_id):
            coat = _coat_array(i, view, image_size)
            clutter = _pasture_clutter(view, image_size, seed=view + 3)
            Image.fromarray(coat).save(sam_dir / f"view_{view:02d}.png")
            Image.fromarray(_compose_bbox(coat, clutter)).save(bbox_dir / f"view_{view:02d}.png")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthetic cattle coats for Mutong.")
    parser.add_argument("--output-dir", default="data/ready_bg_removed")
    parser.add_argument("--mode", choices=("sam", "paired"), default="sam")
    parser.add_argument("--num-identities", type=int, default=96)
    parser.add_argument("--images-per-id", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args(argv)

    out = Path(args.output_dir)
    if args.mode == "paired":
        write_paired(out, args.num_identities, args.images_per_id, args.image_size)
        print(
            f"Wrote paired sam/bbox under {out} "
            f"({args.num_identities} ids x {args.images_per_id} views).",
            flush=True,
        )
    else:
        write_sam_only(out, args.num_identities, args.images_per_id, args.image_size)
        print(
            f"Wrote {args.num_identities} identities x {args.images_per_id} images under {out}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
