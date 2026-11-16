# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Measured-only helpers for the six-cell runner.

Two jobs, both used by ``scripts/run_all_cells.sh``:

``manifest``  build a seeded train/gallery/query manifest from an identity tree.
``summary``   collect the ``metrics.json`` files a real run wrote, refuse anything
              that is not a fresh measurement, and aggregate seeds into the
              2 by 3 table.

The aggregation step deliberately has no fallback numbers. If a cell's
``metrics.json`` is missing, stale, or incomplete, it raises. Nothing in this
module can produce a placeholder, which is the point: the values in
``results/metrics.json`` were hand-written and must never reach the manuscript.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
import sys
from pathlib import Path

from src.data import build_split_rows, list_identity_images, write_manifest
from src.protocol import CROP_BBOX, CROP_MODES, CROP_SAM, DISPLAY_NAMES, MODEL_NAMES

SUMMARY_KIND = "measured_six_cells"
REQUIRED_METRIC_KEYS = ("rank1", "rank5", "parameters", "flops", "num_gallery", "num_query")

# Any of these appearing in a metrics file marks it as drafting material.
PLACEHOLDER_MARKERS = (
    "placeholder",
    "simulated",
    "simulate",
    "synthetic",
    "draft",
    "fabricated",
    "invented",
)


class NotAMeasurement(RuntimeError):
    """Raised when a metrics file is missing, stale, or not a real run."""


def build_seed_manifest(
    image_root: str | Path,
    output_manifest: str | Path,
    *,
    seed: int,
    train_ratio: float = 0.6,
    gallery_ratio: float = 0.2,
) -> list[dict[str, str]]:
    """Write a seeded split manifest for ``image_root/<identity_id>/*``.

    Rows carry **absolute** image paths. The archive usually lives outside the
    repository (``/data/...``), and a manifest-relative path would be resolved
    against the repository root by :class:`src.data.ManifestDataset`, which
    cannot work for data outside the checkout.
    """

    root = Path(image_root).resolve()
    by_id = list_identity_images(root)
    rows = build_split_rows(
        by_id,
        root=root,
        seed=int(seed),
        train_ratio=float(train_ratio),
        gallery_ratio=float(gallery_ratio),
    )
    for row in rows:
        row["image_path"] = str((root / row["image_path"]).resolve())
    write_manifest(output_manifest, rows)
    return rows


def _looks_like_placeholder(payload: dict, path: Path) -> str | None:
    for key, value in payload.items():
        if not isinstance(value, str):
            continue
        lowered = value.lower()
        for marker in PLACEHOLDER_MARKERS:
            if marker in lowered and key in ("kind", "note", "source", "provenance", "produced_by"):
                return f"{path}: field {key!r} says {value!r}"
    return None


def load_cell_metrics(
    path: str | Path,
    *,
    started_at: float,
    expected_crop: str,
    expected_model: str,
) -> dict:
    """Load one cell's metrics and prove it came from the run in progress.

    ``started_at`` is the wall clock at which the runner began. A file older
    than that is a leftover from an earlier run (or from the placeholder
    generator) and is rejected rather than silently aggregated.
    """

    file_path = Path(path)
    if not file_path.is_file():
        raise NotAMeasurement(f"Missing metrics file: {file_path}. That cell did not run.")
    mtime = file_path.stat().st_mtime
    if mtime < started_at - 1.0:
        raise NotAMeasurement(
            f"{file_path} was written at {mtime:.0f}, before this run started at "
            f"{started_at:.0f}. Refusing to report a stale file as a measurement."
        )
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise NotAMeasurement(f"{file_path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise NotAMeasurement(f"{file_path} does not contain a JSON object.")

    marker = _looks_like_placeholder(payload, file_path)
    if marker:
        raise NotAMeasurement(f"Placeholder content detected: {marker}")

    missing = [key for key in REQUIRED_METRIC_KEYS if payload.get(key) is None]
    if missing:
        raise NotAMeasurement(f"{file_path} is missing {', '.join(missing)}.")

    if payload.get("model") not in (None, expected_model):
        raise NotAMeasurement(
            f"{file_path} reports model {payload['model']!r}, expected {expected_model!r}."
        )
    if payload.get("crop") not in (None, expected_crop):
        raise NotAMeasurement(
            f"{file_path} reports crop {payload['crop']!r}, expected {expected_crop!r}."
        )
    for key in ("rank1", "rank5"):
        value = payload[key]
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise NotAMeasurement(f"{file_path}: {key} is not a number ({value!r}).")
        if not 0.0 <= float(value) <= 1.0:
            raise NotAMeasurement(f"{file_path}: {key} must be a fraction, got {value!r}.")
    if int(payload["num_gallery"]) < 1 or int(payload["num_query"]) < 1:
        raise NotAMeasurement(
            f"{file_path}: gallery/query are empty ({payload['num_gallery']}/"
            f"{payload['num_query']}); Rank-k is undefined."
        )
    if int(payload["parameters"]) <= 0:
        raise NotAMeasurement(f"{file_path}: parameters must be positive.")
    if float(payload["flops"]) <= 0:
        raise NotAMeasurement(f"{file_path}: flops (MACs) must be positive.")
    if not all(math.isfinite(float(payload[k])) for k in ("rank1", "rank5", "flops")):
        raise NotAMeasurement(f"{file_path}: non-finite metric value.")
    return payload


def _mean_sd(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "sd": None, "values": []}
    if len(values) == 1:
        return {"mean": values[0], "sd": None, "values": values}
    return {
        "mean": statistics.fmean(values),
        "sd": statistics.stdev(values),
        "values": values,
    }


def _git_sha(repo_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def aggregate(
    *,
    outputs_root: str | Path,
    results_dir: str | Path,
    seeds: list[int],
    started_at: float,
    config_path: str | Path,
    crops: tuple[str, ...] = CROP_MODES,
    models: tuple[str, ...] = MODEL_NAMES,
    extra_provenance: dict | None = None,
) -> dict:
    """Validate every cell of every seed, copy metrics into ``results_dir``, aggregate."""

    outputs = Path(outputs_root)
    results = Path(results_dir)
    results.mkdir(parents=True, exist_ok=True)

    per_seed: list[dict] = []
    copied: list[str] = []
    for seed in seeds:
        seed_cells: dict[tuple[str, str], dict] = {}
        for crop in crops:
            for model in models:
                run_dir = outputs / f"seed_{seed}" / crop / model
                metrics_path = run_dir / "metrics.json"
                meta_path = run_dir / "train_meta.json"
                if not meta_path.is_file():
                    raise NotAMeasurement(
                        f"Missing training record: {meta_path}. That cell was not trained."
                    )
                payload = load_cell_metrics(
                    metrics_path,
                    started_at=started_at,
                    expected_crop=crop,
                    expected_model=model,
                )
                train_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                record = dict(payload)
                record["prune_report"] = train_meta.get("prune_report") or {}
                record["best_rank1"] = train_meta.get("best_rank1")
                record["manifest"] = train_meta.get("manifest")
                record["num_train_images"] = train_meta.get("num_train_images")
                seed_cells[(crop, model)] = record

                dest = results / f"seed_{seed}" / crop / model / "metrics.json"
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(json.dumps(record, indent=2), encoding="utf-8")
                copied.append(str(dest))
        per_seed.append({"seed": seed, "cells": seed_cells})

    # Per-cell mean and SD across seeds. rank1_sam - rank1_bbox is the headline delta.
    cells = []
    for model in models:
        row: dict = {"model": model, "display_name": DISPLAY_NAMES.get(model, model)}
        for crop in crops:
            rank1 = [float(entry["cells"][(crop, model)]["rank1"]) for entry in per_seed]
            rank5 = [float(entry["cells"][(crop, model)]["rank5"]) for entry in per_seed]
            row[f"rank1_{crop}"] = _mean_sd(rank1)
            row[f"rank5_{crop}"] = _mean_sd(rank5)
        if CROP_SAM in crops and CROP_BBOX in crops:
            deltas = [
                float(entry["cells"][(CROP_SAM, model)]["rank1"])
                - float(entry["cells"][(CROP_BBOX, model)]["rank1"])
                for entry in per_seed
            ]
            row["delta_sam"] = _mean_sd(deltas)
        reference = per_seed[0]["cells"][(CROP_SAM if CROP_SAM in crops else crops[0], model)]
        row["parameters"] = int(reference["parameters"])
        row["flops"] = float(reference["flops"])
        row["prune_report"] = reference.get("prune_report") or {}
        cells.append(row)

    repo_root = Path(config_path).resolve().parent.parent
    summary = {
        "kind": SUMMARY_KIND,
        "note": (
            "Aggregated by scripts/run_all_cells.sh from per-cell metrics.json files written "
            "by src.evaluate on this machine. Every value is a measurement from this run."
        ),
        "generated_at": _isoformat_now(),
        "seeds": list(seeds),
        "num_seeds": len(seeds),
        "config": str(Path(config_path).resolve()),
        "crops": list(crops),
        "cells": cells,
        "provenance": {
            "git_sha": _git_sha(repo_root),
            "outputs_root": str(outputs.resolve()),
            "results_dir": str(results.resolve()),
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": _torch_version(),
            "cuda_available": _cuda_available(),
            "copy_method": "copied from outputs/*/metrics.json after evaluation",
            **(extra_provenance or {}),
        },
    }
    (results / "summary_six_cells.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (results / "summary_six_cells.md").write_text(render_markdown(summary), encoding="utf-8")
    return summary


def _isoformat_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _torch_version() -> str:
    try:
        import torch

        return torch.__version__
    except ImportError:  # pragma: no cover
        return ""


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:  # pragma: no cover
        return False


def _fmt(stat: dict, digits: int = 4) -> str:
    if not stat or stat.get("mean") is None:
        return "n/a"
    if stat.get("sd") is None:
        return f"{stat['mean']:.{digits}f}"
    return f"{stat['mean']:.{digits}f} ± {stat['sd']:.{digits}f}"


def render_markdown(summary: dict) -> str:
    crops = summary.get("crops") or list(CROP_MODES)
    header = ["Model", *(f"Rank-1 ({crop})" for crop in crops), "Δ_SAM (Rank-1)", "Parameters", "MACs"]
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for cell in summary["cells"]:
        ranks = [_fmt(cell.get(f"rank1_{crop}", {})) for crop in crops]
        lines.append(
            "| "
            + " | ".join(
                [
                    cell["display_name"],
                    *ranks,
                    _fmt(cell.get("delta_sam", {}), 4),
                    f"{cell['parameters']:,}",
                    f"{int(cell['flops']):,}",
                ]
            )
            + " |"
        )
    seeds = summary.get("seeds") or []
    provenance = summary.get("provenance") or {}
    footer = [
        "",
        f"Runs: {summary.get('num_seeds', len(seeds))} seed(s) {seeds}; "
        f"values are mean ± sample SD across seeds."
        if summary.get("num_seeds", len(seeds)) > 1
        else f"Runs: one seed ({seeds}). Single-run point estimates only; CEA expects replication.",
        f"Git: `{provenance.get('git_sha', '')}`; config: `{summary.get('config', '')}`; "
        f"host: {provenance.get('hostname', '')}; CUDA: {provenance.get('cuda_available')}.",
        "Every figure above comes from a `metrics.json` written by this run. Nothing is a placeholder.",
    ]
    return "\n".join(lines + footer) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measured-only helpers for run_all_cells.sh")
    sub = parser.add_subparsers(dest="command", required=True)

    manifest = sub.add_parser("manifest", help="Write a seeded split manifest from an identity tree")
    manifest.add_argument("--image-root", required=True)
    manifest.add_argument("--output-manifest", required=True)
    manifest.add_argument("--seed", type=int, default=0)
    manifest.add_argument("--train-ratio", type=float, default=0.6)
    manifest.add_argument("--gallery-ratio", type=float, default=0.2)

    summary = sub.add_parser("summary", help="Validate and aggregate a finished run")
    summary.add_argument("--outputs-root", required=True)
    summary.add_argument("--results-dir", required=True)
    summary.add_argument("--config", required=True)
    summary.add_argument("--started-at", type=float, required=True)
    summary.add_argument("--seeds", nargs="+", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "manifest":
            rows = build_seed_manifest(
                args.image_root,
                args.output_manifest,
                seed=args.seed,
                train_ratio=args.train_ratio,
                gallery_ratio=args.gallery_ratio,
            )
            counts: dict[str, int] = {}
            for row in rows:
                counts[row["split"]] = counts.get(row["split"], 0) + 1
            print(
                json.dumps(
                    {
                        "manifest": str(Path(args.output_manifest).resolve()),
                        "seed": args.seed,
                        "rows": len(rows),
                        "splits": counts,
                        "identities": len({row["identity_id"] for row in rows}),
                    },
                    indent=2,
                )
            )
            return 0
        summary = aggregate(
            outputs_root=args.outputs_root,
            results_dir=args.results_dir,
            seeds=args.seeds,
            started_at=args.started_at,
            config_path=args.config,
        )
        print(render_markdown(summary), flush=True)
        print(f"Wrote {Path(args.results_dir).resolve()}", flush=True)
        return 0
    except (NotAMeasurement, FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
