# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Create a synthetic 96-identity background-removed cattle coat dataset.

Used to exercise the train/eval pipeline when the private UAV archive is not
mounted. Patterns are identity-specific ellipses on a black background.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _coat_image(identity: int, view: int, size: int = 224) -> Image.Image:
    rng = np.random.default_rng(identity * 1009 + view * 17)
    img = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    # Cattle-like ellipse with a stable coat motif per identity.
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
    # Mild view jitter: small translation already baked into spot RNG; add noise.
    arr = np.asarray(img).astype(np.float32)
    noise = rng.normal(0, 6.0, size=arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Synthetic background-removed cattle coats.")
    parser.add_argument("--output-dir", default="data/ready_bg_removed")
    parser.add_argument("--num-identities", type=int, default=96)
    parser.add_argument("--images-per-id", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args(argv)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(args.num_identities):
        identity = f"{100 + i:03d}"
        identity_dir = out / identity
        identity_dir.mkdir(parents=True, exist_ok=True)
        for view in range(args.images_per_id):
            image = _coat_image(i, view, size=args.image_size)
            image.save(identity_dir / f"view_{view:02d}.png")
    print(
        f"Wrote {args.num_identities} identities x {args.images_per_id} images under {out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
