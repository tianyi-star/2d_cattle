# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Synthetic cattle-coat datasets for pipeline checks and controlled ablations.

Modes
-----
* ``sam`` — background-removed coats only (default smoke test).
* ``context_shortcut`` — paired ``sam`` / ``bbox`` arms in which the coat cue is
  deliberately weak while the bbox arm keeps a strong identity-linked scene
  context (pen colour / ground texture). Under that design, Rank-1 on the bbox
  arm is expected to exceed Rank-1 on the SAM arm, so Δ_SAM is negative. The
  result demonstrates shortcut learning, not that background is a better
  biometric than the coat pattern.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _hard_coat(identity: int, view: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """Ambiguous coat: shared palette, only weak identity signal."""

    # Shared base colour across identities so coat alone is hard.
    base = np.array([118, 96, 72], dtype=np.float32)
    arr = np.zeros((size, size, 3), dtype=np.float32)
    cx, cy = size // 2, size // 2
    rx, ry = int(size * 0.36), int(size * 0.26)
    yy, xx = np.ogrid[:size, :size]
    mask = ((xx - cx) / rx) ** 2 + ((cy - yy) / ry) ** 2 <= 1.0
    arr[mask] = base
    # Few weak spots; identity shifts them only slightly.
    n_spots = 4
    for s in range(n_spots):
        ang = 2 * np.pi * (s / n_spots) + 0.15 * identity
        px = int(cx + 0.45 * rx * np.cos(ang) + rng.integers(-2, 3))
        py = int(cy + 0.45 * ry * np.sin(ang) + rng.integers(-2, 3))
        pr = 5 + (identity + s) % 3
        shade = 90 + (identity * 3 + s * 11) % 40
        spot = ((xx - px) ** 2 + (yy - py) ** 2) <= pr * pr
        arr[mask & spot] = (shade, shade - 10, shade - 20)
    # View noise.
    arr += rng.normal(0, 4.0 + 0.5 * (view % 3), size=arr.shape)
    arr[~mask] = 0.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def _identity_context(identity: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """Stable pen / ground cue that is almost unique per identity."""

    # Distinct RGB pen colour from identity index.
    hue = (identity * 37) % 256
    pen = np.array(
        [
            40 + (hue % 180),
            40 + ((hue * 3) % 180),
            40 + ((hue * 7) % 180),
        ],
        dtype=np.float32,
    )
    bg = np.zeros((size, size, 3), dtype=np.float32)
    bg[:, :] = pen
    # Identity-linked stripe orientation / spacing (fence / feeder cue).
    period = 8 + (identity % 11)
    angle = (identity % 12) * (np.pi / 12.0)
    yy, xx = np.mgrid[:size, :size]
    proj = xx * np.cos(angle) + yy * np.sin(angle)
    stripes = (np.sin(2 * np.pi * proj / period) > 0).astype(np.float32)
    bg = bg * (0.75 + 0.25 * stripes[..., None])
    # Mild spatial jitter so views are not pixel-identical.
    bg += rng.normal(0, 3.0, size=bg.shape)
    return np.clip(bg, 0, 255).astype(np.uint8)


def _compose_bbox(coat: np.ndarray, context: np.ndarray) -> np.ndarray:
    mask = coat.sum(axis=2) > 0
    out = context.copy()
    out[mask] = coat[mask]
    return out


def _coat_only_easy(identity: int, view: int, size: int = 224) -> Image.Image:
    """Original easy coat-only generator for plain smoke tests."""

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
    noise = rng.normal(0, 6.0, size=arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def write_sam_only(out: Path, num_identities: int, images_per_id: int, image_size: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for i in range(num_identities):
        identity = f"{100 + i:03d}"
        identity_dir = out / identity
        identity_dir.mkdir(parents=True, exist_ok=True)
        for view in range(images_per_id):
            _coat_only_easy(i, view, size=image_size).save(identity_dir / f"view_{view:02d}.png")


def write_context_shortcut(
    out: Path, num_identities: int, images_per_id: int, image_size: int
) -> None:
    """Write paired SAM (weak coat) and bbox (coat + identity context) trees."""

    sam_root = out / "sam"
    bbox_root = out / "bbox"
    for i in range(num_identities):
        identity = f"{100 + i:03d}"
        sam_dir = sam_root / identity
        bbox_dir = bbox_root / identity
        sam_dir.mkdir(parents=True, exist_ok=True)
        bbox_dir.mkdir(parents=True, exist_ok=True)
        for view in range(images_per_id):
            rng = np.random.default_rng(i * 1009 + view * 17 + 91)
            coat = _hard_coat(i, view, image_size, rng)
            context = _identity_context(i, image_size, rng)
            Image.fromarray(coat).save(sam_dir / f"view_{view:02d}.png")
            Image.fromarray(_compose_bbox(coat, context)).save(bbox_dir / f"view_{view:02d}.png")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Synthetic cattle coats, including a context-shortcut pair."
    )
    parser.add_argument("--output-dir", default="data/ready_bg_removed")
    parser.add_argument(
        "--mode",
        choices=("sam", "context_shortcut"),
        default="sam",
        help="sam: coat-only. context_shortcut: paired sam/bbox with identity-linked background.",
    )
    parser.add_argument("--num-identities", type=int, default=96)
    parser.add_argument("--images-per-id", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args(argv)

    out = Path(args.output_dir)
    if args.mode == "context_shortcut":
        write_context_shortcut(out, args.num_identities, args.images_per_id, args.image_size)
        print(
            f"Wrote context_shortcut pair under {out}/sam and {out}/bbox "
            f"({args.num_identities} ids x {args.images_per_id} views). "
            "Expect Rank-1(bbox) > Rank-1(sam) when the model uses the scene cue.",
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
