# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Manifest IO and image datasets for cattle re-identification."""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "gallery", "query")


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    """Load rows with ``image_path``, ``identity_id``, and ``split``."""

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {file_path}")
    with file_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "identity_id", "split"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"Manifest must contain columns {sorted(required)}: {file_path}")
        rows = []
        for row in reader:
            image_path = (row.get("image_path") or "").strip()
            identity_id = (row.get("identity_id") or "").strip()
            split = (row.get("split") or "").strip()
            if not image_path or not identity_id or split not in SPLITS:
                raise ValueError(f"Invalid manifest row in {file_path}: {row}")
            rows.append({"image_path": image_path, "identity_id": identity_id, "split": split})
    if not rows:
        raise ValueError(f"Manifest is empty: {file_path}")
    return rows


def write_manifest(path: str | Path, rows: Iterable[dict[str, str]]) -> None:
    """Write the three-column identity manifest."""

    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "identity_id", "split"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image_path": row["image_path"],
                    "identity_id": row["identity_id"],
                    "split": row["split"],
                }
            )


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


def build_split_rows(
    by_id: dict[str, list[Path]],
    *,
    root: Path,
    seed: int = 0,
    train_ratio: float = 0.6,
    gallery_ratio: float = 0.2,
) -> list[dict[str, str]]:
    """Assign each image to train / gallery / query while keeping all identities."""

    if train_ratio <= 0 or gallery_ratio < 0 or train_ratio + gallery_ratio >= 1:
        raise ValueError("Need train_ratio > 0, gallery_ratio >= 0, and train+gallery < 1.")
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
            try:
                rel = path.resolve().relative_to(root.resolve())
                image_path = str(rel).replace("\\", "/")
            except ValueError:
                image_path = str(path.resolve())
            rows.append(
                {
                    "image_path": image_path,
                    "identity_id": identity_id,
                    "split": split,
                }
            )
    return rows


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
    """Read one split from a crop manifest."""

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
        label = self.identity_to_index[row["identity_id"]]
        return tensor, label, row["identity_id"]


def filter_split(rows: list[dict[str, str]], split: str) -> list[dict[str, str]]:
    return [row for row in rows if row["split"] == split]


def identity_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row["identity_id"]] += 1
    return dict(counts)
