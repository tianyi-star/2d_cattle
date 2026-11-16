# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""The ``sam`` arm must actually mask, and must fail loudly without weights."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.preprocess import run_preprocess
from src.sam_mask import (
    DegenerateMaskError,
    MaskCache,
    SamConfig,
    SamMasker,
    SamWeightsMissing,
    clip_box,
    purify_crop,
    refine_mask,
    select_mask,
)


def _detector(box):
    def detect(_path):
        return [box]

    return detect


# --------------------------------------------------------------------------- #
# crop-level: SAM crop and box crop must differ
# --------------------------------------------------------------------------- #


def test_sam_crop_differs_from_plain_box_crop(frame_with_coat, coat_box, stub_segmenter):
    """The headline bug: both arms used to write the identical rectangle."""

    mask = stub_segmenter.predict(np.asarray(frame_with_coat), coat_box)
    sam_crop = purify_crop(frame_with_coat, mask)

    x1, y1, x2, y2 = [int(v) for v in coat_box]
    box_crop = frame_with_coat.crop((x1, y1, x2, y2))

    assert sam_crop.size != box_crop.size, "SAM crop should be the tight mask box, not the box"
    assert not np.array_equal(np.asarray(sam_crop), np.asarray(box_crop))

    # The pasture outside the coat is gone from the masked crop.
    sam_pixels = np.asarray(sam_crop).reshape(-1, 3)
    pasture = np.array([36, 118, 52])
    on_pasture = np.all(sam_pixels == pasture, axis=1)
    assert on_pasture.sum() == 0, "masked crop still contains pasture pixels"

    # And the leg, which is inside the detector box, is excluded.
    leg = np.array([120, 118, 110])
    leg_in_box = np.all(np.asarray(box_crop).reshape(-1, 3) == leg, axis=1).sum()
    leg_in_sam = np.all(sam_pixels == leg, axis=1).sum()
    assert leg_in_box > 0, "test frame is wrong: the box crop should contain the leg"
    assert leg_in_sam == 0


def test_plain_crop_path_is_byte_identical_to_before(tiny_dataset, tmp_path, coat_box):
    """The bbox arm must keep its original pixels: no mask, no zeroing, no resize."""

    detector_calls = []

    def detect(path):
        detector_calls.append(path)
        return [coat_box]

    out_dir = tmp_path / "bbox_crops"
    manifest_out = tmp_path / "bbox_manifest.csv"
    summary = run_preprocess(
        image_dir=str(tiny_dataset["root"]),
        manifest=str(tiny_dataset["manifest"]),
        output_dir=str(out_dir),
        output_manifest=str(manifest_out),
        yolo_weights="unused-because-detector-is-injected",
        sam_weights="",
        crop="bbox",
        detector=detect,
    )
    assert summary["crop"] == "bbox"
    assert len(detector_calls) == len(tiny_dataset["rows"])

    with manifest_out.open(encoding="utf-8", newline="") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == len(tiny_dataset["rows"])

    x1, y1, x2, y2 = [int(max(0, v)) for v in coat_box]
    by_source = {row["image_path"]: row for row in tiny_dataset["rows"]}
    for sample in written:
        source_row = by_source[
            str((tmp_path / "raw" / sample["identity_id"] / Path(sample["image_path"]).name))
        ]
        produced = Image.open(sample["image_path"]).convert("RGB")
        expected = Image.open(source_row["image_path"]).convert("RGB").crop((x1, y1, x2, y2))
        assert produced.size == expected.size
        assert np.array_equal(np.asarray(produced), np.asarray(expected))


def test_sam_arm_masks_and_caches(frame_with_coat, coat_box, tmp_path, stub_segmenter):
    config = SamConfig(weights="", backend="ultralytics", cache_dir=str(tmp_path / "cache"))
    masker = SamMasker(config, backend=stub_segmenter)

    first = masker.mask(frame_with_coat, coat_box, image_id="frame-1")
    second = masker.mask(frame_with_coat, coat_box, image_id="frame-1")

    assert stub_segmenter.calls == 1, "second call must be served from the disk cache"
    assert np.array_equal(first, second)
    assert first.dtype == np.bool_
    assert masker.cache.hits == 1 and masker.cache.writes == 1

    # A different box is a different key, so the segmenter runs again.
    masker.mask(frame_with_coat, (80.0, 60.0, 200.0, 150.0), image_id="frame-1")
    assert stub_segmenter.calls == 2


def test_cache_disabled_still_segments_every_time(frame_with_coat, coat_box, stub_segmenter):
    masker = SamMasker(SamConfig(weights="", cache_dir=""), backend=stub_segmenter)
    masker.mask(frame_with_coat, coat_box, image_id="a", use_cache=False)
    masker.mask(frame_with_coat, coat_box, image_id="a", use_cache=False)
    assert stub_segmenter.calls == 2


def test_cache_entry_invalidated_by_new_box(frame_with_coat, coat_box, tmp_path, stub_segmenter):
    cache = MaskCache(tmp_path / "c", SamConfig(weights=""))
    key_a = cache.key("f", (320, 240), coat_box)
    key_b = cache.key("f", (320, 240), (70.0, 50.0, 250.0, 201.0))
    assert key_a != key_b


# --------------------------------------------------------------------------- #
# loud failure instead of a silent fallback
# --------------------------------------------------------------------------- #


def test_missing_sam_weights_raise_rather_than_fall_back(tmp_path, tiny_dataset, coat_box):
    with pytest.raises(FileNotFoundError, match="SAM checkpoint not found"):
        run_preprocess(
            image_dir=str(tiny_dataset["root"]),
            manifest=str(tiny_dataset["manifest"]),
            output_dir=str(tmp_path / "sam"),
            output_manifest=str(tmp_path / "sam_manifest.csv"),
            yolo_weights="yolo.pt",
            sam_weights=str(tmp_path / "absent_sam_vit_b.pth"),
            crop="sam",
            detector=_detector(coat_box),
        )
    assert not (tmp_path / "sam_manifest.csv").exists(), "nothing may be written on failure"


def test_masker_raises_when_weights_are_missing(tmp_path, frame_with_coat, coat_box):
    masker = SamMasker(
        SamConfig(weights=str(tmp_path / "nope.pth"), backend="segment_anything")
    )
    with pytest.raises(SamWeightsMissing):
        masker.mask(frame_with_coat, coat_box, image_id="x")


def test_missing_backend_package_raises(tmp_path, frame_with_coat, coat_box):
    weights = tmp_path / "sam_vit_b.pth"
    weights.write_bytes(b"not a real checkpoint")
    masker = SamMasker(SamConfig(weights=str(weights), backend="segment_anything"))
    with pytest.raises(Exception) as excinfo:
        masker.mask(frame_with_coat, coat_box, image_id="x")
    # Either the package is absent, or loading the fake file fails; both are loud.
    assert isinstance(excinfo.value, Exception)
    assert not isinstance(excinfo.value, AssertionError)


def test_unreadable_cache_entry_is_recomputed(frame_with_coat, coat_box, tmp_path, stub_segmenter):
    cache_dir = tmp_path / "cache"
    config = SamConfig(weights="", backend="ultralytics", cache_dir=str(cache_dir))
    masker = SamMasker(config, backend=stub_segmenter)
    masker.mask(frame_with_coat, coat_box, image_id="frame-1")
    entry = next(cache_dir.rglob("*.npz"))
    entry.write_bytes(b"corrupt")
    masker.mask(frame_with_coat, coat_box, image_id="frame-1")
    assert stub_segmenter.calls == 2


# --------------------------------------------------------------------------- #
# mask post-processing
# --------------------------------------------------------------------------- #


def test_refine_mask_clips_to_box_and_rejects_empty():
    raw = np.zeros((120, 160), dtype=bool)
    raw[10:110, 10:150] = True
    refined = refine_mask(raw, (40.0, 20.0, 100.0, 100.0), fill_holes=False)
    assert refined[20:100, 40:100].all()
    assert refined[:20].sum() == 0 and refined[:, :40].sum() == 0

    with pytest.raises(DegenerateMaskError):
        refine_mask(np.zeros((120, 160), dtype=bool), (0, 0, 160, 120))


def test_select_mask_prefers_smaller_mask_on_equal_scores():
    masks = np.zeros((3, 10, 10), dtype=bool)
    masks[0, 0:9, 0:9] = True
    masks[1, 0:5, 0:5] = True
    masks[2, 0:2, 0:2] = True
    scores = np.array([0.9, 0.9, 0.9])
    assert select_mask(masks, scores).sum() == 4
    scores = np.array([0.1, 0.95, 0.2])
    assert select_mask(masks, scores).sum() == 25


def test_clip_box_handles_out_of_frame_and_inverted():
    assert clip_box((-5, -5, 500, 500), 100, 50) == (0, 0, 100, 50)
    assert clip_box((80, 40, 20, 10), 100, 100) == (20, 10, 80, 40)


# --------------------------------------------------------------------------- #
# CLI surface
# --------------------------------------------------------------------------- #


def test_parser_exposes_sam_controls():
    from src.preprocess import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "--output-dir",
            "a",
            "--output-manifest",
            "b",
            "--sam-weights",
            "w.pth",
            "--sam-model-type",
            "vit_b",
            "--sam-cache-dir",
            "cache",
            "--crop",
            "sam",
        ]
    )
    assert args.sam_model_type == "vit_b"
    assert args.sam_cache_dir == "cache"
    assert args.crop == "sam"
