# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Test bootstrap: import path, shared fixtures, and a stub segmenter."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class StubSegmenter:
    """Stand-in for SAM. Returns a mask hugging the coat and excluding the leg.

    Counts calls so tests can prove the disk cache suppressed a second run.
    """

    name = "stub"

    def __init__(self, coat_rect=(80, 60, 240, 160)) -> None:
        self.coat_rect = coat_rect
        self.calls = 0

    def predict(self, image_rgb: np.ndarray, box) -> np.ndarray:
        self.calls += 1
        height, width = image_rgb.shape[:2]
        mask = np.zeros((height, width), dtype=bool)
        x1, y1, x2, y2 = self.coat_rect
        mask[y1:y2, x1:x2] = True
        return mask


@pytest.fixture
def stub_segmenter() -> StubSegmenter:
    return StubSegmenter()


@pytest.fixture
def frame_with_coat() -> Image.Image:
    """Green pasture, one grey coat rectangle, and a leg hanging below the coat."""

    image = Image.new("RGB", (320, 240), (36, 118, 52))
    draw = ImageDraw.Draw(image)
    draw.rectangle((80, 60, 240, 160), fill=(168, 162, 150))  # coat
    draw.rectangle((140, 160, 170, 210), fill=(120, 118, 110))  # leg outside coat
    draw.ellipse((95, 75, 135, 110), fill=(92, 88, 80))  # coat patch
    return image


@pytest.fixture
def coat_box() -> tuple[float, float, float, float]:
    return (70.0, 50.0, 250.0, 200.0)


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> dict:
    """Six identities x five frames as ``<root>/<id>/view_XX.png`` plus a manifest."""

    root = tmp_path / "raw"
    rows: list[dict[str, str]] = []
    rng = np.random.default_rng(0)
    for index in range(6):
        identity = f"{200 + index:03d}"
        identity_dir = root / identity
        identity_dir.mkdir(parents=True)
        for view in range(5):
            image = Image.new("RGB", (96, 96), (20, 100, 40))
            draw = ImageDraw.Draw(image)
            base = int(60 + index * 25)
            draw.rectangle((24, 24, 74, 74), fill=(base, base // 2, 255 - base))
            draw.ellipse(
                (
                    int(rng.integers(28, 40)),
                    int(rng.integers(28, 40)),
                    int(rng.integers(56, 72)),
                    int(rng.integers(56, 72)),
                ),
                fill=(int(rng.integers(0, 255)), 200, 90),
            )
            path = identity_dir / f"view_{view:02d}.png"
            image.save(path)
            rows.append(
                {
                    "image_path": str(path),
                    "identity_id": identity,
                    "split": "train" if view < 3 else ("gallery" if view == 3 else "query"),
                }
            )
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "identity_id", "split"])
        writer.writeheader()
        writer.writerows(rows)
    return {"root": root, "manifest": manifest, "rows": rows}
