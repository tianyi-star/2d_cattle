# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0 AND MIT
"""F-ODS dataset optimization for individual cattle identification.

F-ODS is the dataset-optimization stage of this project. The routine below
balances a labeled manifest by keeping at most ``shots`` images for each
identity and writing inverse-frequency sample weights for the kept rows.

``score_samples`` is the feature-level hook. The default returns ``None``,
which selects images uniformly within each identity. Replace that function
with the experimental scoring rule. This module does not report recognition
figures and does not require the image files to be present.
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

REQUIRED_COLUMNS = ("image_path", "identity_id")
LOGGER = logging.getLogger(__name__)


def load_yaml_config(path: str | Path) -> dict:
    """Read a YAML mapping. Unknown keys are left for the caller to filter."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "PyYAML is required to read --config. Install requirements.txt."
        ) from exc
    with config_path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")
    return loaded


def read_manifest(path: str | Path) -> list[dict[str, str]]:
    """Load a CSV manifest with ``image_path`` and ``identity_id`` columns.

    Additional columns, including an optional ``split`` or ``mask_path``, are
    preserved. Empty required fields are rejected. Image files are not opened.
    """
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Manifest has no header: {manifest_path}")
        missing = [name for name in REQUIRED_COLUMNS if name not in reader.fieldnames]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(
                f"Manifest {manifest_path} is missing column(s): {joined}. "
                "Required columns are image_path and identity_id."
            )
        rows: list[dict[str, str]] = []
        for line_number, row in enumerate(reader, start=2):
            cleaned = {
                key: value.strip() if isinstance(value, str) else ""
                for key, value in row.items()
                if key is not None
            }
            if not cleaned.get("image_path") or not cleaned.get("identity_id"):
                raise ValueError(
                    f"{manifest_path}:{line_number} has an empty image_path or identity_id."
                )
            rows.append(cleaned)
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")
    return rows


def write_manifest(records: list[dict[str, str]], path: str | Path) -> None:
    """Write manifest rows, preserving column order and any extra fields."""
    if not records:
        raise ValueError("Refusing to write an empty manifest.")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for record in records:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def resolve_image_path(image_path: str, manifest_path: str | Path | None = None) -> Path:
    """Resolve an image path from the working directory, then beside the manifest."""
    candidate = Path(image_path)
    if candidate.is_file():
        return candidate
    if manifest_path is not None:
        beside_manifest = Path(manifest_path).parent / image_path
        if beside_manifest.is_file():
            return beside_manifest
    raise FileNotFoundError(
        f"Image not found: {image_path}. Paths are resolved from the working "
        "directory, then from the manifest directory."
    )


def score_samples(records: list[dict[str, str]]) -> list[float] | None:
    """Feature-level F-ODS score hook.

    Parameters
    ----------
    records:
        Manifest rows. Each row includes ``image_path`` and ``identity_id``.

    Returns
    -------
    list of float, or None
        One score per row. Higher scores are kept when an identity has more
        than ``shots`` images. ``None`` keeps the uniform sampler.

    The default is ``None``. Replace this function with the experimental
    feature-level criterion. Do not hard-code measured recognition figures.
    """
    del records
    return None


def class_frequency_weights(records: list[dict[str, str]]) -> list[float]:
    """Inverse-frequency weight ``n / (c * count(identity))`` for each row.

    ``n`` is the number of rows and ``c`` is the number of identities. A
    balanced subset therefore receives weights near 1. The weights describe
    the subset being written; they are not a recognition result.
    """
    if not records:
        raise ValueError("Cannot weight an empty record list.")
    counts = Counter(record["identity_id"] for record in records)
    class_count = len(counts)
    row_count = len(records)
    return [
        row_count / (class_count * counts[record["identity_id"]])
        for record in records
    ]


def stratified_kshot(
    records: list[dict[str, str]],
    shots: int,
    seed: int,
    scores: list[float] | None = None,
) -> list[dict[str, str]]:
    """Keep at most ``shots`` rows for each identity.

    Identities with fewer rows are kept in full. Without scores, selection is
    a seeded shuffle. With scores, the highest-scoring rows are kept, and ties
    fall back to the earlier manifest row.
    """
    if shots < 1:
        raise ValueError("shots must be an integer >= 1.")
    if scores is not None and len(scores) != len(records):
        raise ValueError("scores must contain one value per manifest row.")
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[record["identity_id"]].append(index)
    generator = random.Random(seed)
    selected_indexes: list[int] = []
    for identity in sorted(grouped):
        indexes = grouped[identity]
        if len(indexes) <= shots:
            chosen = indexes
        elif scores is None:
            chosen = indexes[:]
            generator.shuffle(chosen)
            chosen = chosen[:shots]
        else:
            chosen = sorted(indexes, key=lambda item: (scores[item], -item), reverse=True)
            chosen = chosen[:shots]
        selected_indexes.extend(chosen)
    selected_indexes.sort()
    return [records[index] for index in selected_indexes]


def optimize_fods(
    records: list[dict[str, str]],
    shots: int,
    seed: int,
) -> list[dict[str, str]]:
    """Run the F-ODS balance step and attach ``sample_weight`` to kept rows."""
    scores = score_samples(records)
    selected = stratified_kshot(records, shots=shots, seed=seed, scores=scores)
    weights = class_frequency_weights(selected)
    optimized: list[dict[str, str]] = []
    for record, weight in zip(selected, weights):
        row = dict(record)
        row["sample_weight"] = f"{weight:.6f}"
        optimized.append(row)
    return optimized


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Balance a cattle identity manifest with the F-ODS dataset step. "
            "Writes a new CSV and does not train a model."
        )
    )
    parser.add_argument("--manifest", required=True, help="Input CSV with image_path and identity_id.")
    parser.add_argument("--output", required=True, help="Path of the balanced CSV to write.")
    parser.add_argument("--shots", type=int, default=None, help="Maximum images kept per identity.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for uniform within-identity sampling.")
    parser.add_argument("--config", default=None, help="YAML file. Reads shots and seed when flags are omitted.")
    return parser


def _resolve_options(args: argparse.Namespace) -> tuple[int, int]:
    config = load_yaml_config(args.config) if args.config else {}
    shots = args.shots if args.shots is not None else config.get("shots", 5)
    seed = args.seed if args.seed is not None else config.get("seed", 0)
    if isinstance(shots, bool) or not isinstance(shots, int):
        raise ValueError("shots must be an integer.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    return shots, seed


def main(argv: list[str] | None = None) -> None:
    """Command-line entry: ``python -m src.f_ods_dataset``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        shots, seed = _resolve_options(args)
        records = read_manifest(args.manifest)
        optimized = optimize_fods(records, shots=shots, seed=seed)
        write_manifest(optimized, args.output)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        parser.exit(1, f"error: {exc}\n")
    identity_count = len({row["identity_id"] for row in optimized})
    LOGGER.info(
        "Wrote %s rows for %s identities to %s.",
        len(optimized),
        identity_count,
        args.output,
    )


if __name__ == "__main__":
    main()
