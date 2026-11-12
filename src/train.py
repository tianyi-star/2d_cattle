# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Train one cell of the 2 by 3 ablation with the shared ArcFace loss.

The three public names are ``yolo_arcface`` (YOLO-ReID), ``pruned_vit_arcface``
(Pruned ViT), and ``wang_metric`` (Yaowu Wang). Each name is trained twice:
once on the SAM-purified manifest and once on the YOLO-box manifest. Both runs
use one image size and one ArcFace scale and margin.

Rows with ``split=train`` are the training rows. Gallery and query rows stay
in the manifest so evaluation can use the same file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.arcface import ArcFaceHead
from src.config import load_yaml, require_shared_config
from src.data import ManifestDataset, filter_split, read_manifest
from src.models import build_model, count_parameters
from src.protocol import CROP_MODES, CROP_SAM, DISPLAY_NAMES, MODEL_NAMES, require_crop


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_training(model_name: str, manifest: str, config_path: str, output_dir: str, crop: str) -> dict:
    """Train one ablation cell and write checkpoint + metrics under ``output_dir``."""

    crop = require_crop(crop)
    if not manifest or not output_dir:
        raise ValueError("manifest and output_dir are required. Pass them from configs/path.yaml.")
    config = load_yaml(config_path)
    require_shared_config(config)

    root = Path(manifest).resolve().parent.parent if Path(manifest).name.endswith(".csv") else Path(".")
    # Resolve relative image paths against the repository root (parent of data/).
    repo_root = Path(manifest).resolve()
    for parent in [repo_root.parent, *repo_root.parents]:
        if (parent / "configs" / "experiment.yaml").is_file() or (parent / "src").is_dir():
            root = parent
            break

    rows = read_manifest(manifest)
    train_rows = filter_split(rows, "train")
    if not train_rows:
        raise ValueError(f"No train rows in manifest: {manifest}")

    identities = sorted({row["identity_id"] for row in train_rows})
    identity_to_index = {iden: i for i, iden in enumerate(identities)}
    image_size = int(config["image_size"])
    batch_size = int(config["batch_size"])
    epochs = int(config["epochs"])
    lr = float(config["learning_rate"])
    seed = int(config["seed"])
    torch.manual_seed(seed)

    dataset = ManifestDataset(
        train_rows,
        root=root,
        image_size=image_size,
        train=True,
        identity_to_index=identity_to_index,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(config.get("num_workers") or 0),
        drop_last=len(dataset) >= batch_size,
    )

    device = _device()
    model = build_model(model_name, config).to(device)
    head = ArcFaceHead(
        embedding_dim=int(config["embedding_dim"]),
        num_classes=len(identities),
        scale=float(config["arcface_scale"]),
        margin=float(config["arcface_margin"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(head.parameters()),
        lr=lr,
        weight_decay=float(config.get("weight_decay") or 1e-4),
    )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    model.train()
    head.train()
    for epoch in range(1, epochs + 1):
        running = 0.0
        n_batches = 0
        progress = tqdm(loader, desc=f"{DISPLAY_NAMES[model_name]}[{crop}] epoch {epoch}/{epochs}", leave=False)
        for images, labels, _identity in progress:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            embeddings = model(images)
            loss = head(embeddings, labels)
            loss.backward()
            optimizer.step()
            running += float(loss.item())
            n_batches += 1
            progress.set_postfix(loss=f"{loss.item():.4f}")
        mean_loss = running / max(n_batches, 1)
        history.append({"epoch": epoch, "loss": mean_loss})
        print(f"{DISPLAY_NAMES[model_name]} {crop} epoch {epoch}: loss={mean_loss:.4f}", flush=True)

    checkpoint = {
        "model_name": model_name,
        "crop": crop,
        "config": config,
        "identity_to_index": identity_to_index,
        "model_state": model.state_dict(),
        "history": history,
        "parameters": count_parameters(model),
    }
    ckpt_path = out / "checkpoint.pt"
    torch.save(checkpoint, ckpt_path)
    meta = {
        "model": model_name,
        "display_name": DISPLAY_NAMES[model_name],
        "crop": crop,
        "manifest": str(Path(manifest).resolve()),
        "num_train_images": len(train_rows),
        "num_identities": len(identities),
        "parameters": checkpoint["parameters"],
        "history": history,
        "checkpoint": str(ckpt_path),
    }
    (out / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2), flush=True)
    return meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Shared ArcFace training entry for one ablation cell."
    )
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--manifest", required=True, help="Manifest for the selected crop.")
    parser.add_argument("--config", required=True, help="Shared experiment YAML. Paths stay outside this file.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--crop",
        default=CROP_SAM,
        choices=CROP_MODES,
        help="sam uses the purified back pattern. bbox uses the YOLO box crop of the original frame.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_training(args.model, args.manifest, args.config, args.output_dir, args.crop)
    except (FileNotFoundError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
