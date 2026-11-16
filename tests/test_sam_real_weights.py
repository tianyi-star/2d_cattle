# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Optional check against a real SAM checkpoint.

The suite above proves the masking *wiring* (box prompt -> mask -> masked crop ->
cache) with a stub segmenter, so CI needs no weights. This module adds the one
thing a stub cannot prove: that a real package and checkpoint load and return a
usable coat mask. It is skipped unless both are present.

Enable with, for example::

    SAM_WEIGHTS=/path/sam_vit_b_01ec64.pth \
    SAM_BACKEND=segment_anything \
    python3 -m pytest tests/test_sam_real_weights.py -m slow
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.sam_mask import SamConfig, SamMasker, SamPackageMissing, SamWeightsMissing, purify_crop

SAM_WEIGHTS = os.environ.get("SAM_WEIGHTS", "")
SAM_BACKEND = os.environ.get("SAM_BACKEND", "auto")
SAM_MODEL_TYPE = os.environ.get("SAM_MODEL_TYPE", "vit_b")


def _have_package(backend: str) -> bool:
    name = "mobile_sam" if backend == "mobile_sam" else "segment_anything"
    return importlib.util.find_spec(name) is not None


@pytest.mark.slow
@pytest.mark.skipif(not SAM_WEIGHTS, reason="set SAM_WEIGHTS to a real checkpoint to run")
def test_real_sam_produces_a_mask_and_a_different_crop(tmp_path):
    if not Path(SAM_WEIGHTS).is_file():
        pytest.skip(f"SAM_WEIGHTS does not exist: {SAM_WEIGHTS}")
    resolved = SAM_BACKEND
    if resolved == "auto":
        resolved = "mobile_sam" if SAM_MODEL_TYPE in ("vit_t", "tiny_vit") else "segment_anything"
    if resolved != "ultralytics" and not _have_package(resolved):
        pytest.skip(f"{resolved} is not installed in this environment")

    image = Image.new("RGB", (512, 384), (40, 120, 55))
    draw = ImageDraw.Draw(image)
    draw.ellipse((140, 90, 380, 280), fill=(170, 165, 150))
    for offset in range(6):
        draw.ellipse((180 + offset * 28, 140, 210 + offset * 28, 175), fill=(80, 76, 70))
    box = (130.0, 80.0, 390.0, 290.0)

    masker = SamMasker(
        SamConfig(
            weights=SAM_WEIGHTS,
            model_type=SAM_MODEL_TYPE,
            backend=resolved,
            cache_dir=str(tmp_path / "cache"),
        )
    )
    mask = masker.mask(image, box, image_id="real-sam-frame")
    assert mask.dtype == np.bool_ and mask.shape == (384, 512)
    assert mask.sum() > 0

    x1, y1, x2, y2 = [int(v) for v in box]
    box_crop = np.asarray(image.crop((x1, y1, x2, y2)))
    sam_crop = np.asarray(purify_crop(image, mask))
    assert not np.array_equal(sam_crop, box_crop)
    background = np.array([40, 120, 55])
    assert np.all(sam_crop.reshape(-1, 3) != background)

    # Second call must hit the cache rather than the segmenter.
    masker.mask(image, box, image_id="real-sam-frame")
    assert masker.cache.hits == 1


@pytest.mark.slow
def test_missing_weights_still_raise_with_real_backend(tmp_path, frame_with_coat, coat_box):
    """Even with a real backend configured, absent weights must stop the run."""

    weights = tmp_path / "absent.pth"
    masker = SamMasker(SamConfig(weights=str(weights), backend="segment_anything"))
    with pytest.raises((SamWeightsMissing, SamPackageMissing)):
        masker.mask(frame_with_coat, coat_box, image_id="x")
