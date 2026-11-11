# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0 AND MIT
"""Few-shot training entry points for individual cattle identification.

Three branches share one manifest contract and one configuration file:

* ``yolo`` builds an Ultralytics classification layout and trains from an
  architecture yaml. A ``.pt`` path is refused so this command does not
  download a weight file.
* ``metric`` trains a timm embedding with either the prototypical-network
  objective or batch-hard triplet loss.
* ``vit`` trains a timm Vision Transformer with identity cross-entropy.

``pretrained`` defaults to false. Losses and samplers are standard formulas,
not measured recognition figures. Validation rows are consumed by the YOLO
layout only; the metric and ViT loops leave evaluation to the experiment.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from src.f_ods_dataset import load_yaml_config, read_manifest, resolve_image_path

LOGGER = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    """Resolved training options. Defaults are wiring values, not a tuned schedule."""

    branch: str
    manifest: str
    output_dir: str
    epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 1e-4
    image_size: int = 224
    seed: int = 0
    device: str = "cpu"
    num_workers: int = 0
    pretrained: bool = False
    backbone: str = "resnet18"
    vit_model: str = "vit_tiny_patch16_224"
    yolo_model: str = "yolov8n-cls.yaml"
    metric_loss: str = "prototypical"
    n_way: int = 5
    k_shot: int = 5
    q_query: int = 1
    triplet_margin: float = 0.2


class PrototypicalLoss:
    """Negative log-likelihood of query embeddings under class prototypes.

    The definition follows Snell, Swersky, and Zemel, Prototypical Networks
    for Few-shot Learning (2017): prototypes are class means of the support
    set, and query logits are negative squared Euclidean distances. This class
    has no parameters.
    """

    def forward(self, support, support_labels, query, query_labels):
        import torch
        import torch.nn.functional as F

        classes = torch.unique(support_labels, sorted=True)
        prototypes = []
        for class_id in classes.tolist():
            members = support[support_labels == class_id]
            if members.shape[0] == 0:
                raise ValueError(f"Support set has no embedding for class {class_id}.")
            prototypes.append(members.mean(dim=0))
        prototype_tensor = torch.stack(prototypes, dim=0)
        distances = torch.cdist(query, prototype_tensor, p=2).pow(2)
        log_probabilities = F.log_softmax(-distances, dim=1)
        class_to_index = {int(class_id): index for index, class_id in enumerate(classes.tolist())}
        try:
            target_ids = [class_to_index[int(label)] for label in query_labels.tolist()]
        except KeyError as exc:
            raise ValueError("A query label is missing from the support set.") from exc
        target = torch.tensor(target_ids, dtype=torch.long, device=query.device)
        return F.nll_loss(log_probabilities, target)

    def __call__(self, support, support_labels, query, query_labels):
        return self.forward(support, support_labels, query, query_labels)


class BatchHardTripletLoss:
    """Batch-hard triplet loss with a fixed margin.

    For each anchor that has both a positive and a negative in the batch, the
    loss uses the farthest positive and the nearest negative. Provide at least
    two images of one identity and one image of another identity.
    """

    def __init__(self, margin: float = 0.2):
        if margin < 0:
            raise ValueError("triplet margin must be >= 0.")
        self.margin = margin

    def forward(self, embeddings, labels):
        import torch
        import torch.nn.functional as F

        distances = torch.cdist(embeddings, embeddings, p=2)
        same = labels[:, None].eq(labels[None, :])
        eye = torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
        positive = same & ~eye
        negative = ~same
        hard_positive = distances.masked_fill(~positive, float("-inf")).max(dim=1).values
        hard_negative = distances.masked_fill(~negative, float("inf")).min(dim=1).values
        valid = positive.any(dim=1) & negative.any(dim=1)
        if not bool(valid.any()):
            raise RuntimeError(
                "Triplet batch has no valid anchor. Use at least two images of one "
                "identity and one image of a different identity."
            )
        return F.relu(hard_positive[valid] - hard_negative[valid] + self.margin).mean()

    def __call__(self, embeddings, labels):
        return self.forward(embeddings, labels)


def _import_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for few-shot training. Install requirements.txt."
        ) from exc
    return torch


def _select_device(requested: str) -> str:
    torch = _import_torch()
    if requested == "cpu":
        return "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {requested}, but CUDA is not available.")
    return requested


def _set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def split_records(records: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Split on an optional ``split`` column. Missing values mean train-only."""
    if not any(record.get("split") for record in records):
        return list(records), []
    unknown = sorted({record.get("split", "") for record in records if record.get("split") not in {"train", "val"}})
    if unknown:
        raise ValueError("When split is present, every value must be train or val. Found: " + ", ".join(unknown))
    train_rows = [record for record in records if record.get("split") == "train"]
    val_rows = [record for record in records if record.get("split") == "val"]
    if not train_rows:
        raise ValueError("The split column contains no train rows.")
    return train_rows, val_rows


def identity_names(records: list[dict[str, str]]) -> list[str]:
    return sorted({record["identity_id"] for record in records})


def save_identity_map(names: list[str], output_dir: str | Path) -> dict[str, int]:
    """Write class indices. The file is a label map, not an evaluation report."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "identity_map.json").write_text(json.dumps(names, indent=2) + "\n", encoding="utf-8")
    return {name: index for index, name in enumerate(names)}


def group_by_identity(records: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        grouped[record["identity_id"]].append(record)
    return grouped


def sample_episode(grouped, n_way: int, k_shot: int, q_query: int, generator: random.Random):
    """Sample a support/query episode with local labels ``0 .. n_way - 1``."""
    needed = k_shot + q_query
    eligible = [identity for identity, rows in grouped.items() if len(rows) >= needed]
    if len(eligible) < n_way:
        raise RuntimeError(
            f"Prototypical episodes need at least {n_way} identities with "
            f">= {needed} images. Eligible identities: {len(eligible)}."
        )
    chosen = generator.sample(eligible, n_way)
    support = []
    query = []
    for local_label, identity in enumerate(chosen):
        picks = generator.sample(grouped[identity], needed)
        support.extend((record, local_label) for record in picks[:k_shot])
        query.extend((record, local_label) for record in picks[k_shot:])
    return support, query


def sample_pk_batch(grouped, identities_per_batch: int, images_per_identity: int, generator: random.Random):
    """Sample P identities and K images each for triplet training."""
    eligible = [identity for identity, rows in grouped.items() if len(rows) >= images_per_identity]
    if len(eligible) < identities_per_batch:
        raise RuntimeError(
            f"Triplet batches need at least {identities_per_batch} identities with "
            f">= {images_per_identity} images. Eligible identities: {len(eligible)}."
        )
    chosen = generator.sample(eligible, identities_per_batch)
    batch = []
    for identity in chosen:
        batch.extend(generator.sample(grouped[identity], images_per_identity))
    return batch


def _stack_images(pairs_or_records, manifest: str, image_size: int, device: str, with_labels: bool):
    from src.self_sup_dynamic import load_image_tensor

    torch = _import_torch()
    if with_labels:
        images = [load_image_tensor(resolve_image_path(record["image_path"], manifest), image_size) for record, _ in pairs_or_records]
        labels = [label for _, label in pairs_or_records]
        return torch.stack(images).to(device), torch.tensor(labels, dtype=torch.long, device=device)
    images = [
        load_image_tensor(resolve_image_path(record["image_path"], manifest), image_size)
        for record in pairs_or_records
    ]
    return torch.stack(images).to(device)


def _mean_loss(total: float, steps: int) -> float:
    if steps == 0:
        raise RuntimeError("The training loop completed no optimization steps.")
    return total / steps


def train_metric(cfg: TrainConfig, train_records: list[dict[str, str]]) -> None:
    """Train an embedding and write ``encoder_spec.json`` under the output directory."""
    if cfg.metric_loss not in {"prototypical", "triplet"}:
        raise ValueError("metric_loss must be prototypical or triplet.")
    if cfg.n_way < 2 or cfg.k_shot < 1 or cfg.q_query < 1:
        raise ValueError("n_way must be >= 2 and k_shot, q_query must be >= 1.")
    if cfg.metric_loss == "triplet" and cfg.k_shot < 2:
        raise ValueError("triplet loss requires k_shot >= 2 so each identity has a positive pair.")
    from src.self_sup_dynamic import build_timm_encoder, flatten_embedding, save_encoder_spec

    torch = _import_torch()
    _set_seed(cfg.seed)
    device = _select_device(cfg.device)
    names = identity_names(train_records)
    save_identity_map(names, cfg.output_dir)
    label_of = {name: index for index, name in enumerate(names)}
    grouped = group_by_identity(train_records)
    model = build_timm_encoder(cfg.backbone, num_classes=0, pretrained=cfg.pretrained)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    generator = random.Random(cfg.seed)
    episode_size = cfg.n_way * (cfg.k_shot + cfg.q_query)
    steps_per_epoch = max(1, len(train_records) // max(1, episode_size))
    for epoch in range(cfg.epochs):
        model.train()
        total = 0.0
        for _ in range(steps_per_epoch):
            if cfg.metric_loss == "prototypical":
                support, query = sample_episode(grouped, cfg.n_way, cfg.k_shot, cfg.q_query, generator)
                support_x, support_y = _stack_images(support, cfg.manifest, cfg.image_size, device, True)
                query_x, query_y = _stack_images(query, cfg.manifest, cfg.image_size, device, True)
                combined = torch.cat([support_x, query_x], dim=0)
                embeddings = flatten_embedding(model(combined))
                support_embeddings = embeddings[: support_x.shape[0]]
                query_embeddings = embeddings[support_x.shape[0] :]
                loss = PrototypicalLoss()(support_embeddings, support_y, query_embeddings, query_y)
            else:
                batch = sample_pk_batch(grouped, cfg.n_way, cfg.k_shot, generator)
                images = _stack_images(batch, cfg.manifest, cfg.image_size, device, False)
                labels = torch.tensor([label_of[record["identity_id"]] for record in batch], dtype=torch.long, device=device)
                loss = BatchHardTripletLoss(cfg.triplet_margin)(flatten_embedding(model(images)), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
        LOGGER.info(
            "metric epoch %s/%s mean loss %.6f",
            epoch + 1,
            cfg.epochs,
            _mean_loss(total, steps_per_epoch),
        )
    spec_path = save_encoder_spec(model, cfg.output_dir, cfg.backbone, num_classes=0)
    LOGGER.info("Wrote encoder spec %s.", spec_path)


def train_vit(cfg: TrainConfig, train_records: list[dict[str, str]]) -> None:
    """Train a ViT identity classifier and write its state dict and label map."""
    if cfg.num_workers != 0:
        raise ValueError(
            "The ViT skeleton loads images in the main process. Set num_workers to 0."
        )
    names = identity_names(train_records)
    if len(names) < 2:
        raise RuntimeError("ViT identity training needs at least two identities.")
    from src.self_sup_dynamic import build_timm_encoder, load_image_tensor

    torch = _import_torch()
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

    _set_seed(cfg.seed)
    device = _select_device(cfg.device)
    label_of = save_identity_map(names, cfg.output_dir)

    class ManifestImages(Dataset):
        def __init__(self, records):
            self.records = records

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            record = self.records[index]
            image = load_image_tensor(resolve_image_path(record["image_path"], cfg.manifest), cfg.image_size)
            return image, label_of[record["identity_id"]]

    dataset = ManifestImages(train_records)
    weight_values = _sample_weights(train_records)
    if weight_values is None:
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
        )
    else:
        sampler = WeightedRandomSampler(weight_values, num_samples=len(weight_values), replacement=True)
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers,
        )
    model = build_timm_encoder(cfg.vit_model, num_classes=len(names), pretrained=cfg.pretrained)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(cfg.epochs):
        model.train()
        total = 0.0
        steps = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            loss = criterion(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
            steps += 1
        LOGGER.info("vit epoch %s/%s mean loss %.6f", epoch + 1, cfg.epochs, _mean_loss(total, steps))
    from src.self_sup_dynamic import save_encoder_spec

    spec_path = save_encoder_spec(model, cfg.output_dir, cfg.vit_model, num_classes=len(names))
    LOGGER.info("Wrote classifier spec %s. This head is not the dynamic-learning encoder.", spec_path)


def _sample_weights(records: list[dict[str, str]]) -> list[float] | None:
    if any("sample_weight" not in record or record["sample_weight"] == "" for record in records):
        return None
    weights = []
    for record in records:
        try:
            value = float(record["sample_weight"])
        except ValueError as exc:
            raise ValueError(f"sample_weight is not a number: {record['sample_weight']}") from exc
        if value <= 0:
            raise ValueError("sample_weight must be positive.")
        weights.append(value)
    return weights


def safe_class_name(identity_id: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_" else "_" for character in identity_id)
    if not cleaned:
        raise ValueError(f"Identity cannot be used as a directory name: {identity_id!r}")
    return cleaned


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        raise RuntimeError(f"Refusing to overwrite {destination}.")
    try:
        destination.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, destination)


def prepare_yolo_dataset(records: list[dict[str, str]], output_dir: str | Path, manifest: str) -> Path:
    """Create an Ultralytics classification folder tree and a data yaml.

    Classification training reads the folders. The yaml is written for a later
    detection experiment and is not a measured result. Folder names are the
    sanitized identity ids; ``yolo_class_map.json`` records the original ids.
    """
    train_rows, val_rows = split_records(records)
    if not val_rows:
        raise RuntimeError(
            "The YOLO branch needs a validation split. Add a split column with values train and val."
        )
    root = Path(output_dir) / "yolo_cls"
    mapping = []
    used_names: dict[str, str] = {}
    for record in train_rows + val_rows:
        class_name = safe_class_name(record["identity_id"])
        previous = used_names.get(class_name)
        if previous is not None and previous != record["identity_id"]:
            raise ValueError(
                f"Identities {previous!r} and {record['identity_id']!r} sanitize to the same folder name."
            )
        used_names[class_name] = record["identity_id"]
        split_name = "val" if record.get("split") == "val" else "train"
        source = resolve_image_path(record["image_path"], manifest)
        destination = root / split_name / class_name / source.name
        suffix = 2
        while destination.exists() or destination.is_symlink():
            if destination.resolve() == source.resolve():
                break
            destination = root / split_name / class_name / f"{source.stem}_{suffix}{source.suffix}"
            suffix += 1
        _link_or_copy(source, destination)
        mapping.append({"identity_id": record["identity_id"], "class_name": class_name, "split": split_name})
    root.mkdir(parents=True, exist_ok=True)
    (root / "yolo_class_map.json").write_text(json.dumps(mapping, indent=2) + "\n", encoding="utf-8")
    names = {index: name for index, name in enumerate(sorted(used_names))}
    payload = {"path": str(root.resolve()), "train": "train", "val": "val", "names": names}
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to write the YOLO data file. Install requirements.txt.") from exc
    (root / "data.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return root


def train_yolo(cfg: TrainConfig, records: list[dict[str, str]]) -> None:
    """Train a YOLO classifier from an architecture yaml, never from a ``.pt`` download."""
    model_name = cfg.yolo_model
    if Path(model_name).suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(
            "yolo_model must be an architecture yaml such as yolov8n-cls.yaml, "
            "not a .pt checkpoint. This entry point does not download or ship weights."
        )
    _select_device(cfg.device)
    dataset_dir = prepare_yolo_dataset(records, cfg.output_dir, cfg.manifest)
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is required for the YOLO branch. Install requirements.txt.") from exc
    model = YOLO(model_name)
    model.train(
        data=str(dataset_dir),
        epochs=cfg.epochs,
        imgsz=cfg.image_size,
        batch=cfg.batch_size,
        device=cfg.device,
        workers=cfg.num_workers,
        project=str(Path(cfg.output_dir) / "yolo"),
        name="run",
        pretrained=False,
        exist_ok=True,
        seed=cfg.seed,
    )
    LOGGER.info("Ultralytics wrote its run directory under %s.", Path(cfg.output_dir) / "yolo")


def run_training(cfg: TrainConfig) -> None:
    """Dispatch one few-shot branch. Public entry used by the CLI and by scripts."""
    if cfg.branch not in {"yolo", "metric", "vit"}:
        raise ValueError("branch must be yolo, metric, or vit.")
    if cfg.epochs < 1 or cfg.batch_size < 1 or cfg.image_size < 2:
        raise ValueError("epochs and batch_size must be >= 1, and image_size must be >= 2.")
    if cfg.branch == "yolo" and Path(cfg.yolo_model).suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(
            "yolo_model must be an architecture yaml such as yolov8n-cls.yaml, "
            "not a .pt checkpoint. This entry point does not download or ship weights."
        )
    records = read_manifest(cfg.manifest)
    train_rows, val_rows = split_records(records)
    if cfg.branch == "yolo":
        train_yolo(cfg, records)
        return
    if val_rows:
        LOGGER.info(
            "Ignoring %s validation rows. The %s loop does not report a validation figure; add that hook in the branch when the experiment needs it.",
            len(val_rows),
            cfg.branch,
        )
    if cfg.branch == "metric":
        train_metric(cfg, train_rows)
        return
    train_vit(cfg, train_rows)


def _as_int(name: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    return value


def _as_float(name: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number.")
    return float(value)


def _as_bool(name: str, value) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false.")
    return value


def _as_str(name: str, value) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def make_config(args: argparse.Namespace) -> TrainConfig:
    config = load_yaml_config(args.config) if args.config else {}
    if "branch" in config and config["branch"] != args.branch:
        LOGGER.info("Using --branch %s. The config branch value is ignored.", args.branch)

    def pick(name: str, default, caster, cli_name: str | None = None):
        cli_value = getattr(args, cli_name or name)
        value = cli_value if cli_value is not None else config.get(name, default)
        return caster(name, value)

    pretrained = True if args.pretrained else _as_bool("pretrained", config.get("pretrained", False))
    branch = args.branch
    return TrainConfig(
        branch=branch,
        manifest=args.manifest,
        output_dir=args.output_dir or str(Path("outputs") / branch),
        epochs=pick("epochs", 1, _as_int),
        batch_size=pick("batch_size", 8, _as_int),
        learning_rate=pick("learning_rate", 1e-4, _as_float, "learning_rate"),
        image_size=pick("image_size", 224, _as_int),
        seed=pick("seed", 0, _as_int),
        device=pick("device", "cpu", _as_str),
        num_workers=pick("num_workers", 0, _as_int),
        pretrained=pretrained,
        backbone=pick("backbone", "resnet18", _as_str),
        vit_model=pick("vit_model", "vit_tiny_patch16_224", _as_str),
        yolo_model=pick("yolo_model", "yolov8n-cls.yaml", _as_str),
        metric_loss=pick("metric_loss", "prototypical", _as_str),
        n_way=pick("n_way", 5, _as_int),
        k_shot=pick("k_shot", 5, _as_int),
        q_query=pick("q_query", 1, _as_int),
        triplet_margin=pick("triplet_margin", 0.2, _as_float),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Few-shot cattle identification training for the YOLO, metric, and ViT branches."
    )
    parser.add_argument("--branch", required=True, choices=["yolo", "metric", "vit"])
    parser.add_argument("--manifest", required=True, help="CSV with image_path and identity_id.")
    parser.add_argument("--config", default=None, help="YAML defaults, for example configs/example.yaml.")
    parser.add_argument("--output-dir", default=None, help="Defaults to outputs/<branch>.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", dest="learning_rate", type=float, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=None)
    parser.add_argument("--backbone", default=None, help="timm model name for the metric branch.")
    parser.add_argument("--vit-model", default=None)
    parser.add_argument("--yolo-model", default=None, help="Ultralytics architecture yaml, not a .pt file.")
    parser.add_argument("--metric-loss", default=None, choices=["prototypical", "triplet"])
    parser.add_argument("--n-way", dest="n_way", type=int, default=None)
    parser.add_argument("--k-shot", dest="k_shot", type=int, default=None)
    parser.add_argument("--q-query", dest="q_query", type=int, default=None)
    parser.add_argument("--triplet-margin", dest="triplet_margin", type=float, default=None)
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Allow timm to request ImageNet initialization. Off unless set here or in the config. Not a project weight.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Command-line entry: ``python -m src.fewshot_train``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        run_training(make_config(args))
    except (FileNotFoundError, ValueError, RuntimeError, ImportError, TypeError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
