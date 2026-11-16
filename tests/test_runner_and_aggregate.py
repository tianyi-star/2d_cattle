# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""The runner must refuse to invent or overwrite numbers."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from src.aggregate import (
    NotAMeasurement,
    aggregate,
    build_seed_manifest,
    load_cell_metrics,
    render_markdown,
)
from src.protocol import CROP_BBOX, CROP_SAM, MODEL_NAMES

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_all_cells.sh"


def _two_arm_root(tiny_dataset, tmp_path: Path) -> Path:
    """Lay the tiny frames out as prepared mode expects: sam/ and bbox/ trees."""

    import shutil

    data_root = tmp_path / "data"
    for arm in ("sam", "bbox"):
        shutil.copytree(tiny_dataset["root"], data_root / arm)
    return data_root


def _fake_cell(run_dir: Path, *, model: str, crop: str, rank1: float, rank5: float, parameters: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "model": model,
                "crop": crop,
                "rank1": rank1,
                "rank5": rank5,
                "parameters": parameters,
                "flops": 1_000_000.0,
                "num_gallery": 10,
                "num_query": 10,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "train_meta.json").write_text(
        json.dumps(
            {
                "model": model,
                "crop": crop,
                "manifest": "m.csv",
                "num_train_images": 20,
                "best_rank1": rank1,
                "prune_report": {"heads_pruned": 12} if model == "pruned_vit_arcface" else {},
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def finished_run(tmp_path: Path):
    outputs = tmp_path / "outputs"
    started_at = time.time()
    for seed, bump in ((0, 0.0), (1, 0.02)):
        for crop in (CROP_SAM, CROP_BBOX):
            offset = 0.05 if crop == CROP_SAM else 0.0
            for model in MODEL_NAMES:
                _fake_cell(
                    outputs / f"seed_{seed}" / crop / model,
                    model=model,
                    crop=crop,
                    rank1=0.5 + offset + bump,
                    rank5=0.8 + offset,
                    parameters=1_000 if model != "pruned_vit_arcface" else 4_000_000,
                )
    return {"outputs": outputs, "started_at": started_at, "tmp": tmp_path}


def test_aggregate_writes_measured_summary(finished_run):
    results = finished_run["tmp"] / "results" / "run1"
    summary = aggregate(
        outputs_root=finished_run["outputs"],
        results_dir=results,
        seeds=[0, 1],
        started_at=finished_run["started_at"],
        config_path=ROOT / "configs" / "experiment.yaml",
    )
    assert summary["kind"] == "measured_six_cells"
    assert summary["num_seeds"] == 2
    assert (results / "summary_six_cells.json").is_file()
    assert (results / "summary_six_cells.md").is_file()

    by_model = {cell["model"]: cell for cell in summary["cells"]}
    # Two seeds differ by 0.02, so the SD must be non-zero and explicit.
    assert by_model["yolo_arcface"]["rank1_sam"]["sd"] == pytest.approx(0.0141, abs=1e-3)
    assert by_model["yolo_arcface"]["rank1_sam"]["mean"] == pytest.approx(0.56, abs=1e-6)
    assert by_model["yolo_arcface"]["delta_sam"]["mean"] == pytest.approx(0.05, abs=1e-6)
    assert by_model["pruned_vit_arcface"]["parameters"] == 4_000_000
    # Nothing in the payload may claim to be a placeholder.
    blob = (results / "summary_six_cells.json").read_text(encoding="utf-8").lower()
    assert "placeholder" not in blob


def test_aggregate_rejects_placeholder_metrics(tmp_path):
    started = time.time()
    run_dir = tmp_path / "outputs" / "seed_0" / CROP_SAM / "yolo_arcface"
    _fake_cell(run_dir, model="yolo_arcface", crop=CROP_SAM, rank1=0.9, rank5=0.95, parameters=10)
    payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    payload["kind"] = "metrics_placeholder"
    (run_dir / "metrics.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(NotAMeasurement, match="Placeholder"):
        load_cell_metrics(
            run_dir / "metrics.json",
            started_at=started,
            expected_crop=CROP_SAM,
            expected_model="yolo_arcface",
        )


def test_aggregate_rejects_stale_metrics(tmp_path):
    run_dir = tmp_path / "cell"
    _fake_cell(run_dir, model="yolo_arcface", crop=CROP_SAM, rank1=0.9, rank5=0.95, parameters=10)
    old = time.time() - 3600
    os.utime(run_dir / "metrics.json", (old, old))
    with pytest.raises(NotAMeasurement, match="stale"):
        load_cell_metrics(
            run_dir / "metrics.json",
            started_at=time.time(),
            expected_crop=CROP_SAM,
            expected_model="yolo_arcface",
        )


def test_aggregate_rejects_missing_cell(finished_run):
    missing = finished_run["outputs"] / "seed_1" / CROP_BBOX / "wang_metric" / "metrics.json"
    missing.unlink()
    with pytest.raises(NotAMeasurement, match="Missing metrics file"):
        aggregate(
            outputs_root=finished_run["outputs"],
            results_dir=finished_run["tmp"] / "r",
            seeds=[0, 1],
            started_at=finished_run["started_at"],
            config_path=ROOT / "configs" / "experiment.yaml",
        )


def test_aggregate_rejects_wrong_arm(finished_run):
    """A bbox metrics file copied into the sam slot must not pass as a sam result."""

    path = finished_run["outputs"] / "seed_0" / CROP_SAM / "yolo_arcface" / "metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["crop"] = CROP_BBOX
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(NotAMeasurement, match="crop"):
        aggregate(
            outputs_root=finished_run["outputs"],
            results_dir=finished_run["tmp"] / "r",
            seeds=[0],
            started_at=finished_run["started_at"],
            config_path=ROOT / "configs" / "experiment.yaml",
        )


def test_single_seed_summary_says_so_and_has_no_sd(finished_run):
    summary = aggregate(
        outputs_root=finished_run["outputs"],
        results_dir=finished_run["tmp"] / "one",
        seeds=[0],
        started_at=finished_run["started_at"],
        config_path=ROOT / "configs" / "experiment.yaml",
    )
    md = render_markdown(summary)
    assert "±" not in md
    assert "one seed" in md
    assert "Single-run point estimates only" in md


def test_build_seed_manifest_is_seeded_and_identity_disjoint_by_default(tiny_dataset, tmp_path):
    """The default split must hold animals out, not shuffle their images."""

    from src.data import IdentityLeakageError, identity_partitions

    first = build_seed_manifest(tiny_dataset["root"], tmp_path / "a.csv", seed=0)
    again = build_seed_manifest(tiny_dataset["root"], tmp_path / "b.csv", seed=0)

    assert first == again, "the same seed must reproduce the same split"
    assert (tmp_path / "a.csv").is_file()
    # The JSON manifest and the Table-1 summary are written beside the CSV.
    assert (tmp_path / "a.split.json").is_file()
    assert (tmp_path / "a.table1.md").is_file()

    # Every image is present exactly once, under the new default.
    assert len({row["image_path"] for row in first}) == len(first)

    # 6 identities cannot fill a three-way partition with a test split that has
    # a gallery and a query, so this fixture must ask for fewer identities per
    # partition; the default 64/16/16 needs many more identities than 6.
    partitions = identity_partitions(first)
    assert set(partitions) == {"train", "val", "test"}
    all_ids = partitions["train"] | partitions["val"] | partitions["test"]
    assert len(all_ids) == 6
    assert not (partitions["train"] & partitions["val"])
    assert not (partitions["train"] & partitions["test"])
    assert not (partitions["val"] & partitions["test"])

    # Different seeds may reassign animals, but must never leak one.
    other = build_seed_manifest(tiny_dataset["root"], tmp_path / "c.csv", seed=7)
    for rows in (first, other):
        for row in rows:
            assert row["partition"] in ("train", "val", "test")
        with pytest.raises(IdentityLeakageError):
            from src.data import assert_identity_disjoint

            assert_identity_disjoint(
                [dict(row, partition="train") for row in rows]
                + [dict(rows[0], partition="test")]
            )


def test_build_seed_manifest_legacy_within_mode_is_opt_in(tiny_dataset, tmp_path):
    """The old behaviour is reachable only by naming it."""

    rows = build_seed_manifest(
        tiny_dataset["root"], tmp_path / "legacy.csv", seed=0, split_mode="within"
    )
    by_identity: dict[str, set[str]] = {}
    for row in rows:
        by_identity.setdefault(row["identity_id"], set()).add(row["split"])
    # Every identity appears in every split: the leak this change removes.
    assert {row["identity_id"] for row in rows} == set(by_identity)
    for splits in by_identity.values():
        assert {"train", "gallery", "query"} <= splits
    assert all("partition" not in row for row in rows)
    assert all(row["split"] != "val" for row in rows)


# --------------------------------------------------------------------------- #
# shell driver
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not SCRIPT.is_file(), reason="runner script missing")
def test_runner_refuses_to_reuse_an_existing_results_directory(tiny_dataset, tmp_path):
    data_root = _two_arm_root(tiny_dataset, tmp_path)
    results_root = tmp_path / "results"
    existing = results_root / "six_cells_prepared_20260101T000000Z"
    existing.mkdir(parents=True)
    outputs_root = tmp_path / "outputs"
    occupied = outputs_root / "seed_0" / CROP_SAM / "yolo_arcface"
    occupied.mkdir(parents=True)
    (occupied / "checkpoint.pt").write_bytes(b"old run")

    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--data-root",
            str(data_root),
            "--config",
            str(ROOT / "configs" / "experiment_cpu.yaml"),
            "--outputs-root",
            str(outputs_root),
            "--results-root",
            str(results_root),
            "--crops",
            CROP_SAM,
            "--models",
            "yolo_arcface",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode != 0
    assert "already holds files" in (result.stderr + result.stdout)
    # The guard must fire before a results directory exists or a file is touched.
    created = [p for p in results_root.iterdir() if p != existing]
    assert created == [], f"runner created a results directory despite the guard: {created}"
    assert (occupied / "checkpoint.pt").read_bytes() == b"old run"


def test_runner_rejects_a_single_shared_directory_for_both_arms(tiny_dataset, tmp_path):
    """Prepared mode needs sam/ and bbox/. One shared tree would fake Delta_SAM = 0."""

    shared = tmp_path / "shared"
    shared.mkdir()
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--data-root",
            str(shared),
            "--results-root",
            str(tmp_path / "results"),
            "--outputs-root",
            str(tmp_path / "outputs"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode != 0
    assert "prepared mode needs" in (result.stderr + result.stdout)


def test_runner_dry_run_plans_without_training(tiny_dataset, tmp_path):
    data_root = _two_arm_root(tiny_dataset, tmp_path)
    results_root = tmp_path / "results"
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--data-root",
            str(data_root),
            "--results-root",
            str(results_root),
            "--outputs-root",
            str(tmp_path / "outputs"),
            "--seeds",
            "0 1 2",
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "dry run" in result.stdout
    assert "0 1 2" in result.stdout
    # Dry run still creates the exclusive run directory and its config record,
    # so a mistaken later invocation cannot reuse the name.
    run_dirs = [p for p in results_root.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "run_config.txt").is_file()
    assert not (tmp_path / "outputs").exists(), "dry run must not train anything"


def test_runner_requires_sam_weights_for_the_sam_arm_in_raw_mode(tiny_dataset, tmp_path):
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--data-root",
            str(tiny_dataset["root"]),
            "--data-mode",
            "raw",
            "--results-root",
            str(tmp_path / "results"),
            "--outputs-root",
            str(tmp_path / "outputs"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode != 0
    assert "sam-weights" in (result.stderr + result.stdout) or "manifest.csv" in (
        result.stderr + result.stdout
    )


def test_runner_uses_the_expected_layout_in_dry_run(tiny_dataset, tmp_path):
    data_root = _two_arm_root(tiny_dataset, tmp_path)
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--data-root",
            str(data_root),
            "--results-root",
            str(tmp_path / "results"),
            "--outputs-root",
            str(tmp_path / "outputs"),
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "sam bbox" in result.stdout
    assert "yolo_arcface pruned_vit_arcface wang_metric" in result.stdout
