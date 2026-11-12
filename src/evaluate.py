# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Shared evaluation protocol for the 2 by 3 SAM ablation.

For every query image, rank gallery images by cosine distance between
L2-normalized embeddings. Rank-k is the fraction of queries whose identity
appears among the k nearest gallery identities. Rank-1 and Rank-5 are the
reported ranks. Each of the six cells is measured with the same gallery, the
same query list, and the same input resolution.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_yaml, require_shared_config
from src.data import ManifestDataset, filter_split, read_manifest
from src.models import build_model, count_parameters, estimate_flops
from src.protocol import (
    CROP_MODES,
    CROP_SAM,
    DISPLAY_NAMES,
    METRIC_NAMES,
    MODEL_NAMES,
    SAM_ABLATION_QUESTION,
    require_crop,
)


def metric_names() -> tuple[str, ...]:
    """Return the only fields allowed in one ablation cell."""

    return METRIC_NAMES


def rank1_gain(rank1_with_sam: float, rank1_without_sam: float) -> float:
    """Return the Rank-1 change caused by using the SAM crop instead of the YOLO box."""

    _require_rate("rank1_with_sam", rank1_with_sam)
    _require_rate("rank1_without_sam", rank1_without_sam)
    return rank1_with_sam - rank1_without_sam


def _require_rate(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a fraction in [0, 1].")
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be a fraction in [0, 1].")


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _repo_root(manifest: str) -> Path:
    path = Path(manifest).resolve()
    for parent in [path.parent, *path.parents]:
        if (parent / "src").is_dir() or (parent / "configs" / "experiment.yaml").is_file():
            return parent
    return path.parent


@torch.no_grad()
def _embed_split(model, rows, root, image_size, identity_to_index, batch_size, device):
    if not rows:
        return torch.empty(0, model.embedding_dim, device=device), []
    dataset = ManifestDataset(
        rows,
        root=root,
        image_size=image_size,
        train=False,
        identity_to_index=identity_to_index,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    embeddings = []
    identities = []
    model.eval()
    for images, _labels, identity_ids in tqdm(loader, desc="embed", leave=False):
        images = images.to(device)
        emb = model(images)
        embeddings.append(emb)
        identities.extend(list(identity_ids))
    return torch.cat(embeddings, dim=0), identities


def _rank_metrics(query_emb, query_ids, gallery_emb, gallery_ids, ks=(1, 5)) -> dict[str, float]:
    if query_emb.numel() == 0 or gallery_emb.numel() == 0:
        raise ValueError("Query and gallery splits must both be non-empty.")
    # Cosine similarity because embeddings are L2-normalized.
    scores = query_emb @ gallery_emb.T
    order = torch.argsort(scores, dim=1, descending=True)
    hits = {k: 0 for k in ks}
    for qi, qid in enumerate(query_ids):
        ranked_ids = []
        seen = set()
        for idx in order[qi].tolist():
            gid = gallery_ids[idx]
            if gid in seen:
                continue
            seen.add(gid)
            ranked_ids.append(gid)
            if len(ranked_ids) >= max(ks):
                break
        for k in ks:
            if qid in ranked_ids[:k]:
                hits[k] += 1
    n = len(query_ids)
    return {f"rank{k}": hits[k] / n for k in ks}


def run_evaluation(model_name: str, manifest: str, config_path: str, run_dir: str, crop: str) -> dict:
    """Measure Rank-1 / Rank-5 / parameters / FLOPs for one ablation cell."""

    crop = require_crop(crop)
    if model_name not in MODEL_NAMES:
        joined = ", ".join(MODEL_NAMES)
        raise ValueError(f"Unknown model {model_name!r}. Choose one of: {joined}")
    if not manifest or not run_dir:
        raise ValueError("manifest and run_dir are required. Pass them from configs/path.yaml.")
    config = load_yaml(config_path)
    require_shared_config(config)

    run_path = Path(run_dir)
    ckpt_path = run_path / "checkpoint.pt"
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}. Train this cell first.")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    identity_to_index = checkpoint["identity_to_index"]
    model = build_model(model_name, config)
    model.load_state_dict(checkpoint["model_state"])
    device = _device()
    model = model.to(device)

    rows = read_manifest(manifest)
    gallery_rows = filter_split(rows, "gallery")
    query_rows = filter_split(rows, "query")
    # If gallery is empty (tiny splits), fall back to train as gallery prototypes.
    if not gallery_rows:
        gallery_rows = filter_split(rows, "train")
    root = _repo_root(manifest)
    image_size = int(config["image_size"])
    batch_size = int(config["batch_size"])

    gallery_emb, gallery_ids = _embed_split(
        model, gallery_rows, root, image_size, identity_to_index, batch_size, device
    )
    query_emb, query_ids = _embed_split(
        model, query_rows, root, image_size, identity_to_index, batch_size, device
    )
    ranks = _rank_metrics(query_emb, query_ids, gallery_emb, gallery_ids)
    params = int(checkpoint.get("parameters") or count_parameters(model))
    flops = estimate_flops(model, image_size, device)

    result = {
        "model": model_name,
        "display_name": DISPLAY_NAMES[model_name],
        "crop": crop,
        "rank1": ranks["rank1"],
        "rank5": ranks["rank5"],
        "parameters": params,
        "flops": flops,
        "num_gallery": len(gallery_rows),
        "num_query": len(query_rows),
        "sam_ablation_question": SAM_ABLATION_QUESTION,
    }
    out_file = run_path / "metrics.json"
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rank-1, Rank-5, parameter count, and FLOPs for one cell of the SAM ablation."
        )
    )
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--manifest", required=True, help="Manifest with train, gallery, and query splits.")
    parser.add_argument("--config", required=True, help="Shared experiment YAML.")
    parser.add_argument("--run-dir", required=True, help="Directory written by the matching training run.")
    parser.add_argument(
        "--crop",
        default=CROP_SAM,
        choices=CROP_MODES,
        help="sam evaluates the purified back pattern. bbox evaluates the YOLO box crop.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_evaluation(args.model, args.manifest, args.config, args.run_dir, args.crop)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
