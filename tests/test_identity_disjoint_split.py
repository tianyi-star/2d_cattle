# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Identity-disjoint splitting: the protocol the manuscript specifies.

The split is the thing every number in the paper is conditioned on, so these
tests are written against the properties that make a number meaningful rather
than against a particular row order:

* no identity appears in more than one of train / val / test,
* gallery and query are both drawn from held-out animals,
* the emitted manifest is the same split the loader reads,
* the legacy within-identity split survives only when asked for by name,
* identities with very few images do not break the partition.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.aggregate import build_seed_manifest
from src.data import (
    DEFAULT_IDENTITY_FRACTIONS,
    PARTITIONS,
    IdentityLeakageError,
    allocate_partition_counts,
    assert_identity_disjoint,
    build_split,
    gallery_rows_for_evaluation,
    identity_partitions,
    list_identity_images,
    manifest_sha256,
    read_manifest,
    render_split_markdown,
    verify_split_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (0, 1, 2, 3, 7, 42)


def _write_identities(
    root: Path,
    sizes: dict[str, int],
    *,
    image_size: int = 8,
) -> dict[str, list[Path]]:
    """Create ``root/<identity_id>/view_XX.png`` for each requested size.

    The images are 8x8 solid-colour PNGs: the split never reads pixels, and
    keeping them tiny keeps the suite fast.
    """

    from PIL import Image

    by_id: dict[str, list[Path]] = {}
    for identity, count in sizes.items():
        identity_dir = root / identity
        identity_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for view in range(count):
            path = identity_dir / f"view_{view:02d}.png"
            Image.new("RGB", (image_size, image_size), (10, 20, 30)).save(path)
            paths.append(path)
        by_id[identity] = paths
    return by_id


def _uniform_root(root: Path, *, identities: int = 24, views: int = 6) -> Path:
    _write_identities(root, {f"{100 + i:03d}": views for i in range(identities)})
    return root


@pytest.fixture
def uniform_root(tmp_path: Path) -> Path:
    return _uniform_root(tmp_path / "uniform")


@pytest.fixture
def uneven_root(tmp_path: Path) -> Path:
    """Identity sizes spanning 1..30 images, including animals with one frame."""

    sizes = {
        "101": 1,
        "102": 1,
        "103": 2,
        "104": 2,
        "105": 3,
        "106": 3,
        "107": 4,
        "108": 5,
        "109": 7,
        "110": 9,
        "111": 12,
        "112": 14,
        "113": 18,
        "114": 22,
        "115": 26,
        "116": 30,
        "117": 6,
        "118": 8,
        "119": 11,
        "120": 16,
        "121": 1,
        "122": 2,
        "123": 5,
        "124": 13,
    }
    _write_identities(tmp_path / "uneven", sizes)
    return tmp_path / "uneven"


# --------------------------------------------------------------------------- #
# (a) no identity in two splits, over several seeds
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", SEEDS)
def test_no_identity_appears_in_two_partitions(uniform_root: Path, tmp_path: Path, seed: int):
    rows = build_seed_manifest(
        uniform_root, tmp_path / f"seed_{seed}.csv", seed=seed
    )

    partition_of: dict[str, set[str]] = {}
    for row in rows:
        partition_of.setdefault(row["identity_id"], set()).add(row["partition"])

    offenders = {identity: p for identity, p in partition_of.items() if len(p) > 1}
    assert offenders == {}, f"seed {seed} leaked identities across partitions: {offenders}"

    # And the disjointness assertion the training and evaluation paths call.
    assert_identity_disjoint(rows)

    partitions = identity_partitions(rows)
    assert set(partitions) == set(PARTITIONS)
    train, val, test = (partitions[name] for name in PARTITIONS)
    assert train and val and test
    assert train & val == set()
    assert train & test == set()
    assert val & test == set()
    assert len(train | val | test) == 24


@pytest.mark.parametrize("seed", SEEDS)
def test_gallery_and_query_come_only_from_held_out_identities(
    uniform_root: Path, tmp_path: Path, seed: int
):
    rows = build_seed_manifest(uniform_root, tmp_path / f"gq_{seed}.csv", seed=seed)
    train_ids = {row["identity_id"] for row in rows if row["partition"] == "train"}
    gallery_ids = {row["identity_id"] for row in rows if row["split"] == "gallery"}
    query_ids = {row["identity_id"] for row in rows if row["split"] == "query"}
    test_ids = {row["identity_id"] for row in rows if row["partition"] == "test"}

    assert gallery_ids == test_ids
    assert query_ids == test_ids
    # The inflation this whole change exists to remove.
    assert gallery_ids & train_ids == set()
    assert query_ids & train_ids == set()


def test_leakage_assertion_raises_on_a_hand_made_leak():
    """The guard must fail loudly, not merely be absent on good input."""

    rows = [
        {"image_path": "a.png", "identity_id": "101", "split": "train", "partition": "train"},
        {"image_path": "b.png", "identity_id": "101", "split": "gallery", "partition": "test"},
    ]
    with pytest.raises(IdentityLeakageError, match="101"):
        assert_identity_disjoint(rows)


@pytest.mark.parametrize("seed", SEEDS)
def test_seed_is_reproducible_and_changes_the_assignment(uniform_root: Path, tmp_path: Path, seed: int):
    first = build_seed_manifest(uniform_root, tmp_path / f"a_{seed}.csv", seed=seed)
    again = build_seed_manifest(uniform_root, tmp_path / f"b_{seed}.csv", seed=seed)
    assert first == again, "the same seed must reproduce the same split exactly"

    assignments = []
    for value in SEEDS:
        rows = build_seed_manifest(uniform_root, tmp_path / f"c_{value}.csv", seed=value)
        assignments.append(
            tuple(
                sorted(
                    (name, identity)
                    for name, ids in identity_partitions(rows).items()
                    for identity in ids
                )
            )
        )
    assert len(set(assignments)) > 1, "different seeds should not all give the same partition"


# --------------------------------------------------------------------------- #
# (b) the manifest matches the data the loader actually reads
# --------------------------------------------------------------------------- #


def test_manifest_matches_the_rows_the_loader_returns(uniform_root: Path, tmp_path: Path):
    csv_path = tmp_path / "manifest.csv"
    rows = build_seed_manifest(uniform_root, csv_path, seed=3)

    payload = json.loads((tmp_path / "manifest.split.json").read_text(encoding="utf-8"))
    loaded = read_manifest(csv_path)

    # 1. The JSON describes the CSV, and the CSV is what the loader sees.
    verify_split_manifest(loaded, payload)
    assert loaded == rows, "the rows written must be the rows read back"

    # 2. Identity lists per partition agree with the file itself.
    partitions = identity_partitions(loaded)
    for name in PARTITIONS:
        recorded = payload["partitions"][name]
        expected = sorted(partitions[name])
        assert recorded["identities"] == expected
        assert recorded["num_identities"] == len(expected)
        assert recorded["num_images"] == sum(
            1 for row in loaded if row["partition"] == name
        )
        per_identity = [
            sum(1 for row in loaded if row["identity_id"] == identity)
            for identity in expected
        ]
        assert recorded["images_per_identity"]["min"] == min(per_identity)
        assert recorded["images_per_identity"]["max"] == max(per_identity)
        assert recorded["images_per_identity"]["mean"] == pytest.approx(
            sum(per_identity) / len(per_identity)
        )

    # 3. Totals and per-split counts agree.
    assert payload["totals"]["identities"] == len({row["identity_id"] for row in loaded})
    assert payload["totals"]["images"] == len(loaded)
    for split, count in payload["totals"]["images_per_split"].items():
        assert count == sum(1 for row in loaded if row["split"] == split)

    # 4. The recorded hash identifies these exact bytes.
    from src.data import manifest_sha256

    assert payload["manifest_sha256"] == manifest_sha256(csv_path)

    # 5. Every image in the tree appears exactly once.
    assert len({row["image_path"] for row in loaded}) == len(loaded) == 24 * 6


def test_manifest_records_everything_table_one_needs(uniform_root: Path, tmp_path: Path):
    csv_path = tmp_path / "manifest.csv"
    build_seed_manifest(uniform_root, csv_path, seed=0)
    payload = json.loads((tmp_path / "manifest.split.json").read_text(encoding="utf-8"))

    assert payload["kind"] == "cattle_reid_split_manifest"
    assert payload["split_mode"] == "identity"
    assert payload["identity_disjoint"] is True
    assert payload["seed"] == 0
    assert payload["proportions"] == pytest.approx(DEFAULT_IDENTITY_FRACTIONS)
    assert payload["gallery_fraction"] == pytest.approx(0.5)
    assert payload["min_test_images"] == 2
    assert payload["requested_identity_counts"] == {"train": 16, "val": 4, "test": 4}
    assert payload["excluded_identities"] == []
    for key in ("identities", "num_identities", "num_images", "images_per_identity"):
        assert key in payload["partitions"]["train"]
    assert "num_gallery" in payload["partitions"]["test"]
    assert "num_query" in payload["partitions"]["test"]
    assert payload["totals"]["images_per_identity"]["median"] == 6

    markdown = (tmp_path / "manifest.table1.md").read_text(encoding="utf-8")
    assert "Identities assigned to training" in markdown
    assert "Images in the test split (gallery + query)" in markdown
    assert "Images per identity: min / median / max" in markdown
    assert "mean ± SD" in markdown
    # A data-accounting table must not smuggle in a recognition figure.
    body = "\n".join(
        line for line in markdown.splitlines() if not line.strip().startswith(("<!--", "mAP", "|"))
    ).lower()
    assert "rank-1" not in body and "rank1" not in body and "accuracy" not in body


def test_manifest_rejects_a_csv_that_was_edited_after_it_was_written(
    uniform_root: Path, tmp_path: Path
):
    csv_path = tmp_path / "manifest.csv"
    rows = build_seed_manifest(uniform_root, csv_path, seed=0)
    payload = json.loads((tmp_path / "manifest.split.json").read_text(encoding="utf-8"))

    # Move one held-out animal's query row into train: the classic leak.
    tampered = [dict(row) for row in rows]
    moved = next(row for row in tampered if row["partition"] == "test")
    moved["partition"] = "train"
    moved["split"] = "train"

    with pytest.raises((IdentityLeakageError, ValueError)):
        verify_split_manifest(tampered, payload)


def test_dry_run_writes_nothing_and_prints_the_accounting(uniform_root: Path, tmp_path: Path):
    out_csv = tmp_path / "out" / "manifest.csv"
    result = subprocess.run(
        [
            sys.executable, "-m", "src.aggregate", "manifest",
            "--image-root", str(uniform_root),
            "--output-manifest", str(out_csv),
            "--seed", "0",
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "Identities assigned to training" in result.stdout
    assert "Images in the validation split" in result.stdout
    assert "dry run" in result.stdout
    assert not out_csv.exists(), "a dry run must not write the manifest"
    assert not out_csv.with_suffix(".split.json").exists()
    assert not out_csv.with_suffix(".table1.md").exists()


# --------------------------------------------------------------------------- #
# (c) legacy mode: available, explicit, and not the default
# --------------------------------------------------------------------------- #


def test_identity_disjoint_is_the_default():
    import inspect

    signature = inspect.signature(build_split)
    assert signature.parameters["split_mode"].default == "identity"


def test_legacy_within_mode_is_refused_unless_named(uniform_root: Path, tmp_path: Path):
    """No default path may produce the leaking split."""

    rows = build_seed_manifest(uniform_root, tmp_path / "default.csv", seed=0)
    assert all(row["partition"] in PARTITIONS for row in rows)
    assert "within" not in {row["partition"] for row in rows}

    with pytest.raises(ValueError, match="split_mode"):
        build_seed_manifest(uniform_root, tmp_path / "bad.csv", seed=0, split_mode="shuffle")
    assert not (tmp_path / "bad.csv").exists()


def test_legacy_within_mode_still_works_when_requested(uniform_root: Path, tmp_path: Path):
    csv_path = tmp_path / "legacy.csv"
    rows = build_seed_manifest(
        uniform_root, csv_path, seed=0, split_mode="within", train_ratio=0.6, gallery_ratio=0.2
    )

    # The historical behaviour: every identity lands in train, gallery and query.
    by_identity: dict[str, set[str]] = {}
    for row in rows:
        by_identity.setdefault(row["identity_id"], set()).add(row["split"])
    assert len(by_identity) == 24
    for identity, splits in by_identity.items():
        assert {"train", "gallery", "query"}.issubset(splits), f"{identity} lost a split"

    # No partition column, and no validation split, exactly as before this change.
    assert all("partition" not in row for row in rows)
    with csv_path.open(encoding="utf-8", newline="") as handle:
        assert "partition" not in csv.DictReader(handle).fieldnames
    assert not any(row["split"] == "val" for row in rows)

    payload = json.loads((tmp_path / "legacy.split.json").read_text(encoding="utf-8"))
    assert payload["split_mode"] == "within"
    assert payload["identity_disjoint"] is False
    assert "leakage_note" in payload
    assert "must not be reported" in payload["leakage_note"]
    assert "Legacy within-identity split" in (tmp_path / "legacy.table1.md").read_text(
        encoding="utf-8"
    )

    # Same seed still reproduces it, so old manifests stay rebuildable.
    again = build_seed_manifest(
        uniform_root, tmp_path / "legacy_again.csv", seed=0,
        split_mode="within", train_ratio=0.6, gallery_ratio=0.2,
    )
    assert again == rows


def test_legacy_manifest_is_not_silently_treated_as_held_out(uniform_root: Path, tmp_path: Path):
    """gallery_rows_for_evaluation must not pretend a leaky manifest is disjoint."""

    rows = build_seed_manifest(uniform_root, tmp_path / "legacy.csv", seed=0, split_mode="within")
    gallery, fell_back = gallery_rows_for_evaluation(rows)
    assert gallery
    assert fell_back is False  # gallery rows exist; no fallback was needed

    # Strip the gallery to exercise the fallback, which only legacy manifests allow.
    without_gallery = [row for row in rows if row["split"] != "gallery"]
    fallback, fell_back = gallery_rows_for_evaluation(without_gallery)
    assert fell_back is True
    assert all(row["split"] == "train" for row in fallback)

    # An identity-disjoint manifest may never fall back to train.
    disjoint = build_seed_manifest(uniform_root, tmp_path / "id.csv", seed=0)
    no_gallery = [row for row in disjoint if row["split"] != "gallery"]
    with pytest.raises(ValueError, match="no gallery rows"):
        gallery_rows_for_evaluation(no_gallery)


# --------------------------------------------------------------------------- #
# (d) uneven identity sizes, including animals with one or two frames
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", SEEDS)
def test_uneven_identity_sizes_partition_cleanly(uneven_root: Path, tmp_path: Path, seed: int):
    rows = build_seed_manifest(uneven_root, tmp_path / f"uneven_{seed}.csv", seed=seed)
    assert_identity_disjoint(rows)

    partitions = identity_partitions(rows)
    assert len(partitions["train"] | partitions["val"] | partitions["test"]) == 24

    # A test animal needs at least one gallery and one query view, so single-image
    # animals must never be assigned to test.
    sizes = {identity: len(paths) for identity, paths in list_identity_images(uneven_root).items()}
    for identity in partitions["test"]:
        assert sizes[identity] >= 2, f"{identity} has {sizes[identity]} image(s) but is a test animal"
    for identity in partitions["val"]:
        assert sizes[identity] >= 1

    # Gallery and query are both non-empty and cover exactly the test animals.
    gallery = {row["identity_id"] for row in rows if row["split"] == "gallery"}
    query = {row["identity_id"] for row in rows if row["split"] == "query"}
    assert gallery == query == partitions["test"]

    # No image is dropped or duplicated, whatever the size distribution.
    assert len({row["image_path"] for row in rows}) == len(rows)


def test_uneven_sizes_are_reported_faithfully(uneven_root: Path, tmp_path: Path):
    csv_path = tmp_path / "uneven.csv"
    rows = build_seed_manifest(uneven_root, csv_path, seed=1)
    payload = json.loads((tmp_path / "uneven.split.json").read_text(encoding="utf-8"))

    counts = [
        sum(1 for row in rows if row["identity_id"] == identity)
        for identity in {row["identity_id"] for row in rows}
    ]
    stats = payload["totals"]["images_per_identity"]
    assert stats["min"] == min(counts) == 1
    assert stats["max"] == max(counts) == 30
    import statistics

    assert stats["median"] == pytest.approx(float(statistics.median(counts)))
    assert stats["sd"] > 0, "an uneven distribution must report a non-zero SD"
    assert stats["mean"] == pytest.approx(sum(counts) / len(counts))

    # The Markdown summary must carry the same numbers, since Table 1 is filled from it.
    markdown = (tmp_path / "uneven.table1.md").read_text(encoding="utf-8")
    assert f"{stats['min']} / {stats['median']:g} / {stats['max']}" in markdown
    assert f"{stats['mean']:.2f} ± {stats['sd']:.2f}" in markdown


def test_too_few_eligible_identities_raises_instead_of_making_a_degenerate_test_set(
    tmp_path: Path,
):
    """All but one animal has a single frame: a 20% test split cannot be honest."""

    sizes = {f"{200 + i:03d}": 1 for i in range(9)}
    sizes["209"] = 8
    by_id = _write_identities(tmp_path / "mostly_single", sizes)
    from src.data import build_split

    with pytest.raises(ValueError, match="at least 2 images"):
        build_split(
            by_id,
            root=tmp_path / "mostly_single",
            seed=0,
            fractions={"train": 0.6, "val": 0.2, "test": 0.2},
            min_test_images=2,
        )


def test_partition_counts_always_sum_to_the_identity_count():
    for total in (3, 8, 20, 96, 97, 100):
        counts = allocate_partition_counts(total, DEFAULT_IDENTITY_FRACTIONS)
        assert sum(counts.values()) == total
        assert all(value >= 1 for value in counts.values())


def test_partition_can_be_reused_across_arms(uniform_root: Path, tmp_path: Path):
    """Both crop arms must share one animal partition, or Delta_SAM is confounded."""

    first = build_seed_manifest(uniform_root, tmp_path / "sam.csv", seed=0)
    partition = identity_partitions(first)

    second = build_seed_manifest(
        uniform_root,
        tmp_path / "bbox.csv",
        seed=0,
        identity_partition={name: sorted(ids) for name, ids in partition.items()},
    )
    assert identity_partitions(second) == partition

    # A partition that puts a single-image animal in test must be rejected.
    with pytest.raises(ValueError):
        build_split(
            _write_identities(tmp_path / "small", {"101": 1, "102": 5, "103": 5}),
            root=tmp_path / "small",
            seed=0,
            identity_partition={"train": ["102"], "val": ["103"], "test": ["101"]},
        )


def test_invalid_proportions_are_rejected(uniform_root: Path, tmp_path: Path):
    with pytest.raises(ValueError, match="sum to 1"):
        build_seed_manifest(
            uniform_root, tmp_path / "p.csv", seed=0,
            train_frac=0.7, val_frac=0.2, test_frac=0.2,
        )
    with pytest.raises(ValueError, match="positive"):
        build_seed_manifest(
            uniform_root, tmp_path / "q.csv", seed=0,
            train_frac=1.0, val_frac=0.0, test_frac=0.0,
        )


def test_proportions_are_configurable(uniform_root: Path, tmp_path: Path):
    rows = build_seed_manifest(
        uniform_root, tmp_path / "70.csv", seed=0,
        train_frac=0.70, val_frac=0.15, test_frac=0.15,
    )
    counts = {name: len(ids) for name, ids in identity_partitions(rows).items()}
    assert sum(counts.values()) == 24
    # 24 * 0.70 / 0.15 / 0.15 = 16.8 / 3.6 / 3.6; floors give 22, so the two
    # spare identities go to the largest remainders (train 0.8, then val 0.6).
    assert counts == {"train": 17, "val": 4, "test": 3}
    assert counts["train"] >= counts["val"] and counts["train"] >= counts["test"]
    assert counts["test"] >= 2, "a test split of one animal cannot give Rank-k any support"


def test_render_markdown_states_the_split_is_reproducible(uniform_root: Path, tmp_path: Path):
    csv_path = tmp_path / "manifest.csv"
    build_seed_manifest(uniform_root, csv_path, seed=5)
    payload = json.loads((tmp_path / "manifest.split.json").read_text(encoding="utf-8"))
    markdown = render_split_markdown(payload)
    assert "seed: `5`" in markdown
    assert "`identity`" in markdown
    assert payload["manifest_sha256"] in markdown
