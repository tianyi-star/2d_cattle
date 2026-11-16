# Copyright 2026 Bo, Zhengzhou University / Mutong
# SPDX-License-Identifier: Apache-2.0
"""Manifest IO, identity-disjoint splitting, and image datasets.

Two split modes exist and they are not interchangeable:

``identity`` (the default)
    Identities are shuffled once with a seeded RNG and assigned whole to
    ``train`` / ``val`` / ``test``. No identity appears in more than one
    partition. Images of the test identities are then dealt into ``gallery``
    and ``query`` so that retrieval happens between *held-out* animals, which
    is the protocol the manuscript specifies (Section 3.2.1).

``within`` (legacy, explicit opt-in only)
    Each identity's own images are dealt into ``train`` / ``gallery`` /
    ``query``, so every identity appears in every split. This leaks identity
    and near-duplicate frames across the train/test boundary and inflates
    Rank-1; it exists so that older manifests can still be rebuilt and
    compared against, never as a default.

The per-image ``split`` column uses the values in :data:`SPLITS`. Rows from an
identity-disjoint split also carry a ``partition`` column (one of
:data:`PARTITIONS`) so that a reader of the CSV alone can tell which animals
were held out.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Values allowed in the per-image ``split`` column.
SPLITS = ("train", "val", "gallery", "query")

# Identity-level partitions. Every identity belongs to exactly one.
PARTITIONS = ("train", "val", "test")

SPLIT_MODES = ("identity", "within")

# 64 / 16 / 16 of the 96 identities in the archive.
DEFAULT_IDENTITY_FRACTIONS = {"train": 2.0 / 3.0, "val": 1.0 / 6.0, "test": 1.0 / 6.0}

# Legacy within-identity ratios (train / gallery; query takes the remainder).
DEFAULT_WITHIN_RATIOS = {"train": 0.6, "gallery": 0.2}

DEFAULT_GALLERY_FRACTION = 0.5

# A test identity needs at least one gallery and one query image, otherwise its
# query is unretrievable by construction.
MIN_TEST_IMAGES = 2

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_KIND = "cattle_reid_split_manifest"

_SPLIT_TO_PARTITION = {"train": "train", "val": "val", "gallery": "test", "query": "test"}


class IdentityLeakageError(ValueError):
    """Raised when one identity is found in more than one partition."""


@dataclass(frozen=True)
class SplitPlan:
    """A built split: the rows plus the identity-level assignment behind them."""

    rows: list[dict[str, str]]
    identity_partition: dict[str, list[str]]
    split_mode: str
    seed: int
    fractions: dict[str, float] | None = None
    within_ratios: dict[str, float] | None = None
    gallery_fraction: float | None = None
    min_test_images: int | None = None
    identity_image_counts: dict[str, int] = field(default_factory=dict)


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    """Load rows with ``image_path``, ``identity_id``, ``split`` (and ``partition``).

    ``partition`` is optional in the file and is preserved when present, so a
    manifest produced by this module can be checked for identity leakage at
    training and evaluation time.
    """

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {file_path}")
    with file_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "identity_id", "split"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"Manifest must contain columns {sorted(required)}: {file_path}")
        has_partition = "partition" in reader.fieldnames
        rows = []
        for row in reader:
            image_path = (row.get("image_path") or "").strip()
            identity_id = (row.get("identity_id") or "").strip()
            split = (row.get("split") or "").strip()
            if not image_path or not identity_id or split not in SPLITS:
                raise ValueError(f"Invalid manifest row in {file_path}: {row}")
            record = {"image_path": image_path, "identity_id": identity_id, "split": split}
            if has_partition:
                partition = (row.get("partition") or "").strip()
                if partition and partition not in PARTITIONS:
                    raise ValueError(
                        f"Invalid partition {partition!r} in {file_path}; "
                        f"expected one of {list(PARTITIONS)}"
                    )
                if partition:
                    record["partition"] = partition
            rows.append(record)
    if not rows:
        raise ValueError(f"Manifest is empty: {file_path}")
    return rows


def write_manifest(path: str | Path, rows: Iterable[dict[str, str]]) -> None:
    """Write the manifest. A ``partition`` column is added when rows carry one."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    has_partition = any("partition" in row for row in rows)
    fieldnames = ["image_path", "identity_id", "split"]
    if has_partition:
        fieldnames.append("partition")
    with file_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record = {
                "image_path": row["image_path"],
                "identity_id": row["identity_id"],
                "split": row["split"],
            }
            if has_partition:
                record["partition"] = row.get("partition", "")
            writer.writerow(record)


def list_identity_images(root: str | Path) -> dict[str, list[Path]]:
    """Return ``{identity_id: [image paths]}`` for a per-identity folder tree."""

    root_path = Path(root)
    if not root_path.is_dir():
        raise FileNotFoundError(f"Identity image root not found: {root_path}")
    by_id: dict[str, list[Path]] = {}
    for identity_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
        images = sorted(
            p for p in identity_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if images:
            by_id[identity_dir.name] = images
    if not by_id:
        raise ValueError(
            f"No identity folders with images under {root_path}. "
            "Expected layout: root/<identity_id>/*.png"
        )
    return by_id


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        rel = path.resolve().relative_to(Path(root).resolve())
        return str(rel).replace("\\", "/")
    except (ValueError, OSError):
        return str(path.resolve())


def _validate_fractions(fractions: Mapping[str, float]) -> dict[str, float]:
    values = {name: float(fractions[name]) for name in PARTITIONS}
    if any(value <= 0 for value in values.values()):
        raise ValueError(
            "Identity-disjoint proportions must all be positive, got "
            + ", ".join(f"{name}={values[name]}" for name in PARTITIONS)
        )
    total = sum(values.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            "Identity-disjoint proportions must sum to 1, got "
            + ", ".join(f"{name}={values[name]}" for name in PARTITIONS)
            + f" (sum {total})"
        )
    return values


def allocate_partition_counts(num_identities: int, fractions: Mapping[str, float]) -> dict[str, int]:
    """Split ``num_identities`` across the partitions, largest-remainder method.

    Every partition receives at least one identity, so the three counts always
    sum to ``num_identities``.
    """

    if num_identities < len(PARTITIONS):
        raise ValueError(
            f"Identity-disjoint splitting needs at least {len(PARTITIONS)} identities "
            f"(one per partition), got {num_identities}"
        )
    raw = {name: num_identities * fractions[name] for name in PARTITIONS}
    counts = {name: max(1, int(math.floor(raw[name]))) for name in PARTITIONS}
    remainder = num_identities - sum(counts.values())
    # Largest-remainder over a fixed order: each partition may take at most one
    # extra identity per pass, so a single large fraction cannot absorb them all.
    order = sorted(PARTITIONS, key=lambda name: raw[name] - math.floor(raw[name]), reverse=True)
    index = 0
    while remainder > 0:
        counts[order[index % len(order)]] += 1
        remainder -= 1
        index += 1
    while remainder < 0:
        order = sorted(PARTITIONS, key=lambda name: counts[name], reverse=True)
        for name in order:
            if remainder == 0:
                break
            if counts[name] > 1:
                counts[name] -= 1
                remainder += 1
        else:  # pragma: no cover - unreachable with num_identities >= 3
            raise ValueError("Could not allocate identities across the three partitions")
    return counts


def allocate_identities(
    by_id: Mapping[str, Sequence[Path]],
    *,
    seed: int,
    fractions: Mapping[str, float] | None = None,
    min_test_images: int = MIN_TEST_IMAGES,
) -> dict[str, list[str]]:
    """Assign whole identities to ``train`` / ``val`` / ``test`` with no overlap.

    Identity ids are sorted, shuffled with ``random.Random(seed)``, and then
    consumed in that order: the test partition takes the first identities that
    have at least ``min_test_images`` images (a test animal needs one gallery
    and one query frame), then validation, then training. Because the test
    draw skips ineligible identities *before* the validation cut, no eligible
    identity is wasted on training while an ineligible one occupies the test
    set.
    """

    ids = sorted(str(identity) for identity in by_id)
    fractions = _validate_fractions(fractions or DEFAULT_IDENTITY_FRACTIONS)
    counts = allocate_partition_counts(len(ids), fractions)

    eligible_test = [identity for identity in ids if len(by_id[identity]) >= min_test_images]
    if len(eligible_test) < counts["test"]:
        raise ValueError(
            f"Only {len(eligible_test)} of {len(ids)} identities have at least "
            f"{min_test_images} images, but the test partition needs {counts['test']}. "
            "Lower --test-frac or --min-test-images; a test identity with one image "
            "cannot supply both a gallery and a query view."
        )

    rng = random.Random(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)

    test_ids = [identity for identity in shuffled if len(by_id[identity]) >= min_test_images][
        : counts["test"]
    ]
    chosen = set(test_ids)
    rest = [identity for identity in shuffled if identity not in chosen]
    val_ids = rest[: counts["val"]]
    train_ids = rest[counts["val"] :]
    if not train_ids:
        raise ValueError("Training partition is empty; supply more identities.")

    return {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }


def _deal_gallery_query(
    files: Sequence[Path],
    *,
    gallery_fraction: float,
) -> tuple[list[Path], list[Path]]:
    """Split one held-out identity's frames into gallery and query views."""

    files = list(files)
    n = len(files)
    if n < 2:
        raise ValueError(
            f"A test identity needs at least 2 images, got {n}. "
            "Test identities are filtered at allocation time; this is a bug if reached."
        )
    n_gallery = int(round(n * gallery_fraction))
    n_gallery = max(1, min(n - 1, n_gallery))
    return files[:n_gallery], files[n_gallery:]


def emit_split_rows(
    by_id: Mapping[str, Sequence[Path]],
    identity_partition: Mapping[str, Sequence[str]],
    *,
    root: str | Path,
    seed: int,
    gallery_fraction: float = DEFAULT_GALLERY_FRACTION,
) -> list[dict[str, str]]:
    """Turn an identity partition into manifest rows.

    Training and validation identities put every image in their own split.
    Test identities put their images in ``gallery`` or ``query``; both are
    therefore drawn exclusively from held-out animals.
    """

    root_path = Path(root)
    unknown = sorted(set(identity_partition["train"]) - set(by_id))
    if unknown:
        raise ValueError(f"Identities in the partition are missing from the image tree: {unknown}")

    rows: list[dict[str, str]] = []
    for identity in identity_partition.get("train", []):
        for path in sorted(by_id[identity]):
            rows.append(
                {
                    "image_path": _relative_or_absolute(Path(path), root_path),
                    "identity_id": identity,
                    "split": "train",
                    "partition": "train",
                }
            )
    for identity in identity_partition.get("val", []):
        for path in sorted(by_id[identity]):
            rows.append(
                {
                    "image_path": _relative_or_absolute(Path(path), root_path),
                    "identity_id": identity,
                    "split": "val",
                    "partition": "val",
                }
            )

    rng = random.Random(f"{seed}:gallery-query")
    for identity in identity_partition.get("test", []):
        files = sorted(by_id[identity])
        shuffled = list(files)
        rng.shuffle(shuffled)
        gallery, query = _deal_gallery_query(shuffled, gallery_fraction=gallery_fraction)
        for path in sorted(gallery):
            rows.append(
                {
                    "image_path": _relative_or_absolute(Path(path), root_path),
                    "identity_id": identity,
                    "split": "gallery",
                    "partition": "test",
                }
            )
        for path in sorted(query):
            rows.append(
                {
                    "image_path": _relative_or_absolute(Path(path), root_path),
                    "identity_id": identity,
                    "split": "query",
                    "partition": "test",
                }
            )

    order = {name: index for index, name in enumerate(SPLITS)}
    rows.sort(key=lambda row: (order[row["split"]], row["identity_id"], row["image_path"]))
    return rows


def _within_rows(
    by_id: Mapping[str, Sequence[Path]],
    *,
    root: str | Path,
    seed: int,
    train_ratio: float,
    gallery_ratio: float,
) -> list[dict[str, str]]:
    """Legacy within-identity split: every identity lands in every split."""

    if train_ratio <= 0 or gallery_ratio < 0 or train_ratio + gallery_ratio >= 1:
        raise ValueError("Need train_ratio > 0, gallery_ratio >= 0, and train+gallery < 1.")
    root_path = Path(root)
    rng = random.Random(seed)
    rows: list[dict[str, str]] = []
    for identity_id, paths in sorted(by_id.items()):
        files = list(paths)
        rng.shuffle(files)
        n = len(files)
        if n == 1:
            split_names = ["train"]
        elif n == 2:
            split_names = ["train", "query"]
        else:
            n_train = max(1, int(round(n * train_ratio)))
            n_gallery = max(1, int(round(n * gallery_ratio)))
            if n_train + n_gallery >= n:
                n_gallery = max(1, n - n_train - 1)
            n_query = n - n_train - n_gallery
            if n_query < 1:
                n_train = max(1, n_train - 1)
                n_query = n - n_train - n_gallery
            split_names = (["train"] * n_train) + (["gallery"] * n_gallery) + (["query"] * n_query)
        for path, split in zip(files, split_names):
            rows.append(
                {
                    "image_path": _relative_or_absolute(Path(path), root_path),
                    "identity_id": identity_id,
                    "split": split,
                }
            )
    return rows


def build_split(
    by_id: Mapping[str, Sequence[Path]],
    *,
    root: str | Path,
    seed: int = 0,
    split_mode: str = "identity",
    fractions: Mapping[str, float] | None = None,
    gallery_fraction: float = DEFAULT_GALLERY_FRACTION,
    min_test_images: int = MIN_TEST_IMAGES,
    train_ratio: float = DEFAULT_WITHIN_RATIOS["train"],
    gallery_ratio: float = DEFAULT_WITHIN_RATIOS["gallery"],
    identity_partition: Mapping[str, Sequence[str]] | None = None,
) -> SplitPlan:
    """Build a manifest split. ``identity`` is the default; ``within`` is legacy.

    ``identity_partition`` reuses an already-decided assignment (for example the
    split recorded for the other crop arm) instead of drawing a new one, so both
    arms of the ablation share one animal partition.
    """

    if split_mode not in SPLIT_MODES:
        raise ValueError(f"split_mode must be one of {list(SPLIT_MODES)}, got {split_mode!r}")

    counts = {identity: len(paths) for identity, paths in by_id.items()}

    if split_mode == "within":
        if identity_partition is not None:
            raise ValueError("identity_partition applies to split_mode='identity' only")
        rows = _within_rows(
            by_id, root=root, seed=seed, train_ratio=train_ratio, gallery_ratio=gallery_ratio
        )
        identity_partition = {
            split: sorted({row["identity_id"] for row in rows if row["split"] == split})
            for split in ("train", "gallery", "query")
        }
        return SplitPlan(
            rows=rows,
            identity_partition=dict(identity_partition),
            split_mode="within",
            seed=seed,
            within_ratios={"train": float(train_ratio), "gallery": float(gallery_ratio)},
            identity_image_counts=counts,
        )

    fractions = _validate_fractions(fractions or DEFAULT_IDENTITY_FRACTIONS)
    if identity_partition is None:
        partition = allocate_identities(
            by_id, seed=seed, fractions=fractions, min_test_images=min_test_images
        )
    else:
        partition = {name: sorted(str(i) for i in identity_partition.get(name, [])) for name in PARTITIONS}
        assigned = [identity for name in PARTITIONS for identity in partition[name]]
        if len(assigned) != len(set(assigned)):
            raise IdentityLeakageError("Reused identity partition contains an identity twice")
        missing = sorted(set(by_id) - set(assigned))
        if missing:
            raise ValueError(
                f"Reused identity partition does not cover {len(missing)} identities in the "
                f"image tree: {missing[:10]}"
            )
        short = [identity for identity in partition["test"] if len(by_id[identity]) < min_test_images]
        if short:
            raise ValueError(
                f"Reused identity partition puts identities with fewer than {min_test_images} "
                f"images in the test split: {short}"
            )

    rows = emit_split_rows(
        by_id, partition, root=root, seed=seed, gallery_fraction=gallery_fraction
    )
    return SplitPlan(
        rows=rows,
        identity_partition=partition,
        split_mode="identity",
        seed=seed,
        fractions=fractions,
        gallery_fraction=float(gallery_fraction),
        min_test_images=int(min_test_images),
        identity_image_counts=counts,
    )


def build_split_rows(
    by_id: Mapping[str, Sequence[Path]],
    *,
    root: str | Path,
    seed: int = 0,
    **kwargs,
) -> list[dict[str, str]]:
    """Return just the rows of :func:`build_split` (kept for existing callers)."""

    return build_split(by_id, root=root, seed=seed, **kwargs).rows


def identity_partitions(rows: Sequence[Mapping[str, str]]) -> dict[str, set[str]]:
    """Return ``{partition: {identity ids}}`` derived from manifest rows.

    The ``partition`` column is authoritative. Rows from a legacy within-identity
    manifest have no ``partition``; for those, ``gallery``/``query`` are grouped
    as one ``test`` set so that a caller can still see the overlap.
    """

    grouped: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        partition = row.get("partition") or _SPLIT_TO_PARTITION[row["split"]]
        grouped[partition].add(row["identity_id"])
    return dict(grouped)


def assert_identity_disjoint(rows: Sequence[Mapping[str, str]]) -> None:
    """Raise :class:`IdentityLeakageError` if an identity spans two partitions."""

    seen: dict[str, str] = {}
    for row in rows:
        partition = row.get("partition") or _SPLIT_TO_PARTITION[row["split"]]
        identity = row["identity_id"]
        previous = seen.get(identity)
        if previous is None:
            seen[identity] = partition
        elif previous != partition:
            raise IdentityLeakageError(
                f"Identity {identity!r} appears in both {previous!r} and {partition!r}. "
                "Identity-disjoint splitting requires every animal in exactly one partition; "
                "use split_mode='within' explicitly if the leak is intended."
            )


def _count_stats(counts: Sequence[int]) -> dict:
    """min / median / max / mean / sample SD for a list of per-identity counts."""

    values = sorted(int(value) for value in counts)
    if not values:
        return {"min": 0, "median": 0.0, "max": 0, "mean": 0.0, "sd": 0.0}
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "min": values[0],
        "median": float(statistics.median(values)),
        "max": values[-1],
        "mean": statistics.fmean(values),
        "sd": sd,
    }


def _partition_block(
    rows: Sequence[Mapping[str, str]], identities: Sequence[str], *, split: str | None = None
) -> dict:
    """Summarise one partition. ``split`` restricts to one per-image split value.

    Identity-disjoint manifests have ``split=None`` (a partition == a split
    value, except that test owns both gallery and query). Legacy within-identity
    manifests pass ``split`` so that the block describes only that split's rows;
    without it, overlapping identities would double-count.
    """

    wanted = set(identities)
    partition_rows = [
        row
        for row in rows
        if row["identity_id"] in wanted and (split is None or row["split"] == split)
    ]
    per_identity = defaultdict(int)
    split_images = defaultdict(int)
    for row in partition_rows:
        per_identity[row["identity_id"]] += 1
        split_images[row["split"]] += 1
    block: dict = {
        "identities": sorted(identities),
        "num_identities": len(set(identities)),
        "num_images": len(partition_rows),
        "images_per_split": {split_name: split_images.get(split_name, 0) for split_name in SPLITS if split_images.get(split_name)},
        "images_per_identity": _count_stats(list(per_identity.values())),
    }
    if "gallery" in split_images or "query" in split_images:
        block["num_gallery"] = split_images.get("gallery", 0)
        block["num_query"] = split_images.get("query", 0)
    return block


def build_split_manifest(
    plan: SplitPlan,
    *,
    source_root: str | Path | None = None,
    generated_by: str = "python3 -m src.aggregate manifest",
    extra: Mapping[str, object] | None = None,
) -> dict:
    """Build the self-describing JSON manifest that accompanies the CSV.

    It records everything Table 1 of the manuscript needs: split mode, seed,
    proportions, the identity ids of each partition, and image counts per
    partition and per identity. It reports no recognition result.
    """

    rows = plan.rows
    all_identities = sorted({row["identity_id"] for row in rows})
    per_identity = defaultdict(int)
    split_images = defaultdict(int)
    for row in rows:
        per_identity[row["identity_id"]] += 1
        split_images[row["split"]] += 1

    manifest: dict = {
        "kind": MANIFEST_KIND,
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_by": generated_by,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "split_mode": plan.split_mode,
        "seed": plan.seed,
        "identity_disjoint": plan.split_mode == "identity",
        "image_root": str(Path(source_root).resolve()) if source_root is not None else None,
        "totals": {
            "identities": len(all_identities),
            "images": len(rows),
            "images_per_split": {split: split_images.get(split, 0) for split in SPLITS if split_images.get(split)},
            "images_per_identity": _count_stats(list(per_identity.values())),
        },
        "partitions": {},
        "excluded_identities": [],
    }

    if plan.split_mode == "identity":
        manifest["proportions"] = dict(plan.fractions or {})
        manifest["requested_identity_counts"] = allocate_partition_counts(
            len(all_identities), plan.fractions or DEFAULT_IDENTITY_FRACTIONS
        )
        manifest["identity_ordering"] = (
            "identity ids sorted, shuffled with random.Random(seed), test takes the first "
            "identities with at least min_test_images images, then validation, then training"
        )
        manifest["gallery_fraction"] = plan.gallery_fraction
        manifest["min_test_images"] = plan.min_test_images
        for name in PARTITIONS:
            identities = plan.identity_partition.get(name, [])
            manifest["partitions"][name] = _partition_block(rows, identities)
    else:
        manifest["within_ratios"] = dict(plan.within_ratios or {})
        manifest["leakage_note"] = (
            "Legacy within-identity split: each identity appears in train, gallery and query. "
            "Rank-1 from this manifest is inflated by identity and near-duplicate leakage and "
            "must not be reported as identification accuracy."
        )
        for split in ("train", "gallery", "query"):
            identities = plan.identity_partition.get(split, [])
            manifest["partitions"][split] = _partition_block(rows, identities, split=split)

    manifest["images_per_identity_source"] = "counted from the manifest rows, not the image tree"
    manifest["sd_convention"] = "sample standard deviation (n-1); 0.0 when only one identity"
    if extra:
        manifest.update(extra)
    return manifest


def manifest_sha256(path: str | Path) -> str:
    """SHA-256 of the manifest CSV bytes, so a quoted split can be identified."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_split_manifest(
    rows: Sequence[Mapping[str, str]],
    manifest: Mapping,
    *,
    csv_sha256: str | None = None,
) -> None:
    """Check that ``rows`` and ``manifest`` describe the same split; raise otherwise."""

    if manifest.get("split_mode") == "identity":
        assert_identity_disjoint(rows)

    partitions = manifest.get("partitions") or {}
    if not partitions:
        raise ValueError("Manifest has no 'partitions' section")

    grouped: dict[str, set[str]] = defaultdict(set)
    split_counts: dict[str, int] = defaultdict(int)
    split_identities: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        grouped[row.get("partition") or _SPLIT_TO_PARTITION[row["split"]]].add(row["identity_id"])
        split_counts[row["split"]] += 1
        split_identities[row["split"]].add(row["identity_id"])

    identity_mode = manifest.get("split_mode") == "identity"
    if identity_mode:
        keys = PARTITIONS
        actual_identities = {key: grouped.get(key, set()) for key in keys}
        actual_images = {key: sum(1 for row in rows if row.get("partition") == key) for key in keys}
    else:
        # Legacy within-identity manifests record one block per split *value*.
        keys = ("train", "gallery", "query")
        actual_identities = {key: split_identities.get(key, set()) for key in keys}
        actual_images = {key: split_counts.get(key, 0) for key in keys}

    for key in keys:
        if key not in partitions:
            raise ValueError(f"Manifest is missing the {key!r} partition")
        expected = sorted(partitions[key]["identities"])
        actual = sorted(actual_identities[key])
        if expected != actual:
            missing = sorted(set(expected) - set(actual))[:5]
            extra = sorted(set(actual) - set(expected))[:5]
            raise ValueError(
                f"Manifest partition {key!r} does not match the rows: "
                f"missing {missing}, unexpected {extra}"
            )
        if int(partitions[key]["num_identities"]) != len(actual):
            raise ValueError(f"Manifest partition {key!r} identity count is stale")
        if int(partitions[key]["num_images"]) != actual_images[key]:
            raise ValueError(
                f"Manifest partition {key!r} records {partitions[key]['num_images']} images, "
                f"rows contain {actual_images[key]}"
            )

    totals = manifest.get("totals") or {}
    if int(totals.get("images", -1)) != len(rows):
        raise ValueError(f"Manifest total images {totals.get('images')} != {len(rows)} rows")
    if int(totals.get("identities", -1)) != len({row["identity_id"] for row in rows}):
        raise ValueError("Manifest total identities does not match the rows")
    recorded_splits = totals.get("images_per_split") or {}
    for split, count in recorded_splits.items():
        if int(count) != split_counts.get(split, 0):
            raise ValueError(f"Manifest records {count} {split!r} images, rows contain {split_counts.get(split, 0)}")
    if csv_sha256 is not None and manifest.get("manifest_sha256") not in (None, csv_sha256):
        raise ValueError(
            "manifest_sha256 in the JSON does not match this CSV; the two files are from "
            "different splits"
        )


def render_split_markdown(manifest: Mapping) -> str:
    """Render a Table-1-ready summary. Counts only; no recognition result."""

    totals = manifest.get("totals") or {}
    per_identity = totals.get("images_per_identity") or {}
    partitions = manifest.get("partitions") or {}
    mode = manifest.get("split_mode")

    lines = [
        "<!-- Generated from the split manifest. Data accounting only: no Rank-1, Rank-5,",
        "     mAP or any other recognition figure appears here. -->",
        "",
        f"Split mode: `{mode}` · seed: `{manifest.get('seed')}` · "
        f"identities: {totals.get('identities')} · images: {totals.get('images')}",
    ]
    if mode == "identity":
        proportions = manifest.get("proportions") or {}
        shown = ", ".join(f"{name} {proportions.get(name, 0):.4f}" for name in PARTITIONS)
        lines.append(f"Proportions: {shown} · gallery fraction of test images: "
                     f"{manifest.get('gallery_fraction')}")
    else:
        lines.append(
            "**Legacy within-identity split — identities appear in more than one split. "
            "Do not report Rank-1 from this manifest as identification accuracy.**"
        )
    if manifest.get("manifest_sha256"):
        lines.append(f"Manifest CSV SHA-256: `{manifest['manifest_sha256']}`")
    lines.append("")

    if mode == "identity":
        test = partitions.get("test", {})
        rows = [
            ("Identities in the archive", totals.get("identities")),
            ("Identities assigned to training", partitions.get("train", {}).get("num_identities")),
            ("Identities assigned to validation", partitions.get("val", {}).get("num_identities")),
            ("Identities assigned to testing", test.get("num_identities")),
            ("Images in the training split", partitions.get("train", {}).get("num_images")),
            ("Images in the validation split", partitions.get("val", {}).get("num_images")),
            (
                "Images in the test split (gallery + query)",
                f"{test.get('num_images')} ({test.get('num_gallery')} gallery + "
                f"{test.get('num_query')} query)",
            ),
            (
                "Images per identity: min / median / max",
                f"{per_identity.get('min')} / {per_identity.get('median'):g} / {per_identity.get('max')}"
                if per_identity
                else "n/a",
            ),
            (
                "Images per identity: mean ± SD",
                f"{per_identity.get('mean'):.2f} ± {per_identity.get('sd'):.2f}"
                if per_identity
                else "n/a",
            ),
            (
                "Identities excluded, with reason",
                ", ".join(manifest.get("excluded_identities") or []) or "none",
            ),
        ]
        lines.append("| Quantity | Value |")
        lines.append("| --- | ---: |")
        for label, value in rows:
            lines.append(f"| {label} | {value} |")
        lines.append("")
        lines.append("| Partition | Identities | Images | Images/identity min / median / max | mean ± SD |")
        lines.append("| --- | ---: | ---: | --- | --- |")
        for name in PARTITIONS:
            block = partitions.get(name, {})
            stats = block.get("images_per_identity") or {}
            if not stats:
                continue
            lines.append(
                f"| {name} | {block.get('num_identities')} | {block.get('num_images')} | "
                f"{stats['min']} / {stats['median']:g} / {stats['max']} | "
                f"{stats['mean']:.2f} ± {stats['sd']:.2f} |"
            )
    else:
        lines.append("| Split | Identities | Images |")
        lines.append("| --- | ---: | ---: |")
        for name in ("train", "gallery", "query"):
            block = partitions.get(name, {})
            lines.append(f"| {name} | {block.get('num_identities')} | {block.get('num_images')} |")

    lines.append("")
    lines.append(
        f"Reproduce with seed {manifest.get('seed')} and split mode `{mode}`; "
        "see `python3 -m src.aggregate manifest --help`."
    )
    return "\n".join(lines) + "\n"


def write_split_artifacts(
    plan: SplitPlan,
    *,
    manifest_csv: str | Path,
    split_json: str | Path | None = None,
    split_md: str | Path | None = None,
    source_root: str | Path | None = None,
    generated_by: str = "python3 -m src.aggregate manifest",
) -> dict:
    """Write the CSV and its JSON/Markdown companions; return the JSON manifest."""

    csv_path = Path(manifest_csv)
    json_path = Path(split_json) if split_json else csv_path.with_suffix(".split.json")
    md_path = Path(split_md) if split_md else csv_path.with_suffix(".table1.md")

    write_manifest(csv_path, plan.rows)
    manifest = build_split_manifest(
        plan, source_root=source_root, generated_by=generated_by
    )
    manifest["manifest_csv"] = str(csv_path.resolve())
    manifest["manifest_sha256"] = manifest_sha256(csv_path)
    verify_split_manifest(plan.rows, manifest, csv_sha256=manifest["manifest_sha256"])

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_split_markdown(manifest), encoding="utf-8")
    manifest["split_json"] = str(json_path.resolve())
    manifest["split_md"] = str(md_path.resolve())
    json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_identity_partition(path: str | Path) -> dict[str, list[str]]:
    """Read the identity assignment recorded in a split JSON (for reuse across arms)."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("split_mode") != "identity":
        raise ValueError(
            f"{path} records split_mode={payload.get('split_mode')!r}; only an identity-disjoint "
            "manifest can be reused as a partition"
        )
    partitions = payload.get("partitions") or {}
    return {
        name: sorted(str(identity) for identity in (partitions.get(name) or {}).get("identities", []))
        for name in PARTITIONS
    }


def build_image_transform(image_size: int, train: bool) -> transforms.Compose:
    """Shared preprocessing for all three ArcFace branches."""

    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


class ManifestDataset(Dataset):
    """Read one split from a crop manifest.

    Gallery and query rows come from identities that were never trained on, so
    they have no row in ``identity_to_index``. Their label is then ``-1``: the
    label is unused at evaluation time, and requiring it would make a correctly
    held-out split impossible to read. Training rows must all be known ids.
    """

    def __init__(
        self,
        rows: list[dict[str, str]],
        *,
        root: Path,
        image_size: int,
        train: bool,
        identity_to_index: dict[str, int] | None = None,
    ) -> None:
        self.rows = rows
        self.root = Path(root)
        self.transform = build_image_transform(image_size, train=train)
        self.train = train
        identities = sorted({row["identity_id"] for row in rows})
        self.identity_to_index = identity_to_index or {iden: i for i, iden in enumerate(identities)}
        self.index_to_identity = {idx: iden for iden, idx in self.identity_to_index.items()}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = Path(row["image_path"])
        if not path.is_file():
            path = self.root / row["image_path"]
        image = Image.open(path).convert("RGB")
        tensor = self.transform(image)
        label = self.identity_to_index.get(row["identity_id"], -1)
        if self.train and label < 0:
            raise ValueError(
                f"Training row has identity {row['identity_id']!r}, which is not in the label "
                "mapping built from the train split."
            )
        return tensor, label, row["identity_id"]


def val_retrieval_rows(
    rows: Sequence[dict[str, str]],
    *,
    seed: int = 0,
    gallery_fraction: float = DEFAULT_GALLERY_FRACTION,
) -> tuple[list[dict[str, str]], list[dict[str, str]]] | None:
    """Build a gallery/query pair from the **validation** identities alone.

    Under an identity-disjoint split the val animals are held out of training,
    so they can select a checkpoint without touching the test animals. Each val
    identity's own images are dealt deterministically into a gallery and a query
    view. Returns ``None`` when the manifest has no validation rows, so callers
    can fall back to their previous behaviour.

    These rows are for model selection only. They are never the rows Rank-1 is
    reported from; that remains the held-out ``gallery`` / ``query`` pair.
    """

    val_rows = filter_split(list(rows), "val")
    if not val_rows:
        return None
    by_identity: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in val_rows:
        by_identity[row["identity_id"]].append(row)

    gallery: list[dict[str, str]] = []
    query: list[dict[str, str]] = []
    rng = random.Random(f"{seed}:val-retrieval")
    for identity in sorted(by_identity):
        identity_rows = sorted(by_identity[identity], key=lambda row: row["image_path"])
        if len(identity_rows) < 2:
            # A single-image val animal cannot supply both views; it is skipped
            # rather than silently borrowing a test image.
            continue
        rng.shuffle(identity_rows)
        n_gallery = max(1, min(len(identity_rows) - 1, int(round(len(identity_rows) * gallery_fraction))))
        gallery.extend(identity_rows[:n_gallery])
        query.extend(identity_rows[n_gallery:])
    if not gallery or not query:
        return None
    return gallery, query


def verify_manifest_artifacts(manifest_path: str | Path, rows: list[dict[str, str]]) -> dict | None:
    """Cross-check a manifest against its JSON companion when one is present.

    ``python3 -m src.aggregate manifest`` writes ``<name>.split.json`` beside the
    CSV. If it is there, the two must agree on the hash, the identity lists and
    the counts; a hand-edited CSV is then caught instead of silently training on
    a different split than the one the paper quotes. Returns the JSON when it was
    loaded.
    """

    csv_path = Path(manifest_path)
    json_path = csv_path.with_suffix(".split.json")
    if not json_path.is_file():
        if any("partition" in row for row in rows):
            assert_identity_disjoint(rows)
        return None
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    verify_split_manifest(rows, payload, csv_sha256=manifest_sha256(csv_path))
    return payload


def filter_split(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    return [row for row in rows if row["split"] == split]


def gallery_rows_for_evaluation(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], bool]:
    """Return ``(gallery rows, fell_back_to_train)``.

    Under an identity-disjoint split, gallery and query are both drawn from the
    held-out test identities, so an empty gallery is a broken manifest and is
    reported as such. The legacy within-identity manifests may legitimately
    contain identities with a single image, which produces no gallery row; for
    those the historical behaviour (rank against train rows) is kept, but the
    caller is told, so it can be recorded rather than passed off as held-out
    retrieval.
    """

    gallery = filter_split(rows, "gallery")
    if gallery:
        return gallery, False
    if any(row.get("partition") == "test" for row in rows):
        raise ValueError(
            "Identity-disjoint manifest has no gallery rows. The test partition must supply "
            "both gallery and query views; rebuild the manifest with "
            "`python3 -m src.aggregate manifest`."
        )
    train = filter_split(rows, "train")
    if not train:
        raise ValueError("Manifest has neither gallery nor train rows; Rank-k is undefined.")
    print(
        "warning: no gallery rows in a legacy within-identity manifest; ranking against "
        "train rows. Rank-1 from this manifest is not an identification estimate.",
        flush=True,
    )
    return train, True


def identity_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["identity_id"]] += 1
    return dict(counts)
