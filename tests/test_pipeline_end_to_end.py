# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""End-to-end smoke test: preprocess both arms, train, evaluate, on tiny CPU data.

This asserts that the pipeline runs and writes real files. It deliberately does
not assert any accuracy value: a few synthetic images cannot measure a
recogniser, and inventing an expected Rank-1 here would repeat the mistake the
placeholder ``results/`` files make.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from src.evaluate import run_evaluation
from src.preprocess import run_preprocess
from src.protocol import CROP_BBOX, CROP_SAM
from src.sam_mask import SamConfig, SamMasker
from src.train import run_training
from tests.conftest import StubSegmenter

IMAGE_SIZE = 224
BOX = (16.0, 12.0, 80.0, 76.0)
COAT = (24, 24, 74, 74)


def _detector(_path):
    return [BOX]


def _write_tiny_dataset(root: Path, *, identities: int = 4, views: int = 5) -> dict:
    rows: list[dict[str, str]] = []
    rng = np.random.default_rng(1)
    for index in range(identities):
        identity = f"{300 + index:03d}"
        (root / identity).mkdir(parents=True)
        for view in range(views):
            image = Image.new("RGB", (96, 96), (20, 100, 40))
            draw = ImageDraw.Draw(image)
            colour = int(50 + index * 40)
            draw.rectangle(COAT, fill=(colour, 90, 160))
            draw.ellipse(
                (
                    int(rng.integers(28, 40)),
                    int(rng.integers(28, 40)),
                    int(rng.integers(56, 72)),
                    int(rng.integers(56, 72)),
                ),
                fill=(int(rng.integers(0, 255)), 180, 70),
            )
            path = root / identity / f"view_{view:02d}.png"
            image.save(path)
            rows.append(
                {
                    "image_path": str(path),
                    "identity_id": identity,
                    "split": "train" if view < 3 else ("gallery" if view == 3 else "query"),
                }
            )
    return {"root": root, "rows": rows}


@pytest.fixture(scope="module")
def tiny_config(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("cfg") / "experiment_tiny.yaml"
    path.write_text(
        "\n".join(
            [
                "seed: 0",
                f"image_size: {IMAGE_SIZE}",
                "batch_size: 4",
                "epochs: 2",
                "learning_rate: 0.001",
                "weight_decay: 0.0001",
                "num_workers: 0",
                "embedding_dim: 32",
                "eval_every: 1",
                "arcface_scale: 16.0",
                "arcface_margin: 0.3",
                "crops:",
                "  - sam",
                "  - bbox",
                "yolo_backbone: csp_small",
                "vit_name: vit_tiny_patch16_224",
                "vit_prune_ratio: 0.25",
                "wang_backbone: resnet18",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """Preprocess both crop arms once; the training tests reuse the result."""

    base = tmp_path_factory.mktemp("e2e")
    dataset = _write_tiny_dataset(base / "raw")
    manifest = base / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "identity_id", "split"])
        writer.writeheader()
        writer.writerows(dataset["rows"])
    dataset["manifest"] = manifest

    masker = SamMasker(
        SamConfig(weights="", cache_dir=str(base / "mask_cache")),
        backend=StubSegmenter(coat_rect=COAT),
    )

    summaries = {}
    for arm in (CROP_SAM, CROP_BBOX):
        out_manifest = base / f"{arm}_manifest.csv"
        summaries[arm] = run_preprocess(
            image_dir=str(dataset["root"]),
            manifest=str(manifest),
            output_dir=str(base / f"{arm}_crops"),
            output_manifest=str(out_manifest),
            yolo_weights="unused",
            sam_weights="unused",
            crop=arm,
            detector=_detector,
            masker=masker if arm == CROP_SAM else None,
        )
        summaries[arm]["manifest"] = str(out_manifest)
    return {**summaries, "dataset": dataset, "base": base, "masker": masker}


def _read(path: str) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_both_arms_write_manifests_with_all_splits(prepared):
    for arm in (CROP_SAM, CROP_BBOX):
        written = _read(prepared[arm]["manifest"])
        assert len(written) == 20
        assert {row["split"] for row in written} == {"train", "gallery", "query"}


def test_sam_and_bbox_crops_actually_differ(prepared):
    """The regression that matters: the two arms must not write the same pixels."""

    sam_rows = _read(prepared[CROP_SAM]["manifest"])
    bbox_rows = _read(prepared[CROP_BBOX]["manifest"])
    key = lambda row: (row["identity_id"], Path(row["image_path"]).name)  # noqa: E731
    sam_by_name = {key(row): row["image_path"] for row in sam_rows}
    bbox_by_name = {key(row): row["image_path"] for row in bbox_rows}
    assert sam_by_name.keys() == bbox_by_name.keys()

    for name, sam_path in sam_by_name.items():
        sam_img = np.asarray(Image.open(sam_path).convert("RGB"))
        bbox_img = np.asarray(Image.open(bbox_by_name[name]).convert("RGB"))
        differs = sam_img.shape != bbox_img.shape or not np.array_equal(sam_img, bbox_img)
        assert differs, f"sam and bbox crops are identical for {name}: masking is inert"

    assert prepared[CROP_SAM]["sam"]["misses"] == 20, "each frame should be segmented once"
    assert prepared[CROP_SAM]["sam"]["hits"] == 0


def test_mask_cache_is_reused_on_a_second_preprocess_pass(prepared, tmp_path):
    """Re-running the arm must not re-run the segmenter: masks come from disk."""

    before = dict(prepared["masker"].cache.stats())
    second = run_preprocess(
        image_dir=str(prepared["dataset"]["root"]),
        manifest=str(prepared["dataset"]["manifest"]),
        output_dir=str(tmp_path / "sam_crops_again"),
        output_manifest=str(tmp_path / "sam_manifest_again.csv"),
        yolo_weights="unused",
        sam_weights="unused",
        crop=CROP_SAM,
        detector=_detector,
        masker=prepared["masker"],
    )
    after = second["sam"]
    assert after["misses"] == before["misses"], "the cache should have served every mask"
    assert after["hits"] - before["hits"] == 20


@pytest.mark.parametrize("model_name", ["yolo_arcface", "pruned_vit_arcface", "wang_metric"])
def test_train_and_evaluate_each_backbone(prepared, tiny_config, tmp_path, model_name):
    run_dir = tmp_path / "runs" / model_name
    meta = run_training(
        model_name=model_name,
        manifest=prepared[CROP_SAM]["manifest"],
        config_path=str(tiny_config),
        output_dir=str(run_dir),
        crop=CROP_SAM,
    )
    assert Path(meta["checkpoint"]).is_file()
    assert meta["num_train_images"] == 12
    assert len(meta["history"]) == 2
    assert all(np.isfinite(record["loss"]) for record in meta["history"])

    result = run_evaluation(
        model_name=model_name,
        manifest=prepared[CROP_SAM]["manifest"],
        config_path=str(tiny_config),
        run_dir=str(run_dir),
        crop=CROP_SAM,
    )
    assert 0.0 <= result["rank1"] <= 1.0
    assert 0.0 <= result["rank5"] <= 1.0
    assert result["parameters"] > 0
    on_disk = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert on_disk["rank1"] == result["rank1"]
    assert on_disk["num_gallery"] == 4 and on_disk["num_query"] == 4


def test_pruning_report_is_written_into_train_meta(prepared, tiny_config, tmp_path):
    meta = run_training(
        model_name="pruned_vit_arcface",
        manifest=prepared[CROP_SAM]["manifest"],
        config_path=str(tiny_config),
        output_dir=str(tmp_path / "runs" / "report"),
        crop=CROP_SAM,
    )
    report = meta["prune_report"]
    assert report["criterion"] == "weight_magnitude"
    assert report["heads_pruned"] > 0
    assert report["mlp_channels_pruned"] > 0
    assert meta["parameters"] < 5_000_000
    checkpoint = torch.load(meta["checkpoint"], map_location="cpu", weights_only=False)
    assert checkpoint["prune_report"] == report


def test_pruned_cell_has_fewer_parameters_than_unpruned(prepared, tiny_config, tmp_path):
    """Same backbone, same data, only the prune ratio differs."""

    unpruned_cfg = tmp_path / "experiment_unpruned.yaml"
    unpruned_cfg.write_text(
        Path(tiny_config).read_text(encoding="utf-8").replace(
            "vit_prune_ratio: 0.25", "vit_prune_ratio: 0.0"
        ),
        encoding="utf-8",
    )
    pruned = run_training(
        model_name="pruned_vit_arcface",
        manifest=prepared[CROP_BBOX]["manifest"],
        config_path=str(tiny_config),
        output_dir=str(tmp_path / "runs" / "pruned"),
        crop=CROP_BBOX,
    )
    unpruned = run_training(
        model_name="pruned_vit_arcface",
        manifest=prepared[CROP_BBOX]["manifest"],
        config_path=str(unpruned_cfg),
        output_dir=str(tmp_path / "runs" / "unpruned"),
        crop=CROP_BBOX,
    )
    assert pruned["parameters"] < unpruned["parameters"]
    assert unpruned["prune_report"]["applied"] is False
