# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Self-supervised dynamic learning for an enrolled cattle identity memory.

Labeled images build one prototype per individual. Unlabeled UAV frames then
take one NT-Xent step on the supplied encoder and move the nearest prototype
with an exponential moving average. The update does not create a new identity.

No encoder weights are included. Pass either an ``nn.Module`` saved with
``torch.save``, or an encoder spec JSON written by ``save_encoder_spec``.
``identify`` prints the nearest identity and cosine distance for one query.
It does not aggregate those distances into a recognition figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from src.f_ods_dataset import load_yaml_config, read_manifest, resolve_image_path

ENCODER_SPEC_FORMAT = "mutong-encoder-spec"
LOGGER = logging.getLogger(__name__)


def _import_torch():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for this command. Install requirements.txt."
        ) from exc
    return torch, F


def _as_1d_tensor(embedding):
    torch, _ = _import_torch()
    if isinstance(embedding, torch.Tensor):
        vector = embedding.detach().float().cpu().reshape(-1).clone()
    else:
        vector = torch.tensor(list(embedding), dtype=torch.float32)
    if vector.ndim != 1 or vector.numel() == 0:
        raise ValueError("An embedding must be a non-empty vector.")
    return vector


def _unit(vector):
    norm = float(vector.norm().item())
    if norm == 0:
        raise ValueError("Cannot normalize a zero embedding.")
    return vector / norm


class NTXentLoss:
    """Normalized temperature-scaled cross entropy used by SimCLR.

    The class is the standard two-view objective. It has no learned parameters.
    Replace ``forward`` if the experiment uses a different self-supervised loss.
    """

    def __init__(self, temperature: float = 0.2):
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.temperature = temperature

    def forward(self, z1, z2):
        torch, F = _import_torch()
        if z1.ndim != 2 or z1.shape != z2.shape:
            raise ValueError("NT-Xent expects two embeddings of shape (batch, dim).")
        if z1.shape[0] < 2:
            raise ValueError("NT-Xent needs a batch of at least 2.")
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        batch = z1.shape[0]
        representations = torch.cat([z1, z2], dim=0)
        logits = (representations @ representations.T) / self.temperature
        diagonal = torch.eye(2 * batch, dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(diagonal, -1e9)
        targets = torch.cat(
            [
                torch.arange(batch, 2 * batch, device=logits.device),
                torch.arange(0, batch, device=logits.device),
            ],
            dim=0,
        )
        return F.cross_entropy(logits, targets)

    def __call__(self, z1, z2):
        return self.forward(z1, z2)


class IdentityMemory:
    """Mean embedding prototypes keyed by cattle identity.

    ``enroll`` updates the mean of vectors for an existing or new identity.
    ``update_prototype`` moves an existing prototype and never adds an identity.
    """

    def __init__(self):
        self.sums: dict[str, object] = {}
        self.counts: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.counts)

    def enroll(self, identity_id: str, embedding) -> None:
        if not identity_id:
            raise ValueError("identity_id must be non-empty.")
        vector = _as_1d_tensor(embedding)
        self._check_dimension(vector)
        if identity_id not in self.sums:
            self.sums[identity_id] = vector
            self.counts[identity_id] = 1
            return
        self.sums[identity_id] = self.sums[identity_id] + vector
        self.counts[identity_id] += 1

    def _check_dimension(self, vector) -> None:
        if not self.sums:
            return
        existing = next(iter(self.sums.values()))
        if tuple(existing.shape) != tuple(vector.shape):
            raise ValueError(
                f"Embedding dimension {tuple(vector.shape)} does not match "
                f"enrolled dimension {tuple(existing.shape)}."
            )

    def prototype(self, identity_id: str):
        if identity_id not in self.sums:
            raise KeyError(f"Identity is not enrolled: {identity_id}")
        return _unit(self.sums[identity_id] / self.counts[identity_id])

    def update_prototype(self, identity_id: str, embedding, momentum: float) -> None:
        """EMA on the unit prototype: ``normalize(m * p + (1 - m) * z)``."""
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("memory momentum must satisfy 0 <= momentum <= 1.")
        current = self.prototype(identity_id)
        vector = _as_1d_tensor(embedding)
        self._check_dimension(vector)
        vector = _unit(vector)
        updated = _unit(momentum * current + (1.0 - momentum) * vector)
        self.sums[identity_id] = updated * self.counts[identity_id]

    def identify(self, embedding) -> tuple[str, float]:
        """Return ``(identity_id, cosine_distance)``.

        Cosine distance is ``1 - cosine_similarity`` after L2 normalization.
        Smaller values are nearer. An empty memory raises.
        """
        if not self.counts:
            raise RuntimeError("Identity memory is empty. Enroll individuals before identification.")
        vector = _as_1d_tensor(embedding)
        self._check_dimension(vector)
        vector = _unit(vector)
        best_identity = None
        best_distance = None
        for identity in sorted(self.counts):
            similarity = float((vector * self.prototype(identity)).sum().item())
            distance = 1.0 - similarity
            if best_distance is None or distance < best_distance:
                best_identity = identity
                best_distance = distance
        return best_identity, best_distance

    def assign_and_update(self, embeddings, momentum: float) -> None:
        """Assign each row to the nearest prototype, then apply the EMA update."""
        if not self.counts:
            raise RuntimeError(
                "Enroll identities before a dynamic update. "
                "Unlabeled frames do not create new identities in this skeleton."
            )
        for row in embeddings:
            identity, _distance = self.identify(row)
            self.update_prototype(identity, row, momentum=momentum)

    def save(self, path: str | Path) -> None:
        torch, _ = _import_torch()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "sums": {identity: tensor.cpu() for identity, tensor in self.sums.items()},
            "counts": dict(self.counts),
        }
        torch.save(payload, output_path)

    @classmethod
    def load(cls, path: str | Path) -> "IdentityMemory":
        torch, _ = _import_torch()
        memory_path = Path(path)
        if not memory_path.is_file():
            raise FileNotFoundError(f"Identity memory not found: {memory_path}")
        try:
            payload = torch.load(memory_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(memory_path, map_location="cpu")
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("Unrecognized identity memory file.")
        memory = cls()
        memory.sums = {str(key): value.float().cpu().reshape(-1) for key, value in payload["sums"].items()}
        memory.counts = {str(key): int(value) for key, value in payload["counts"].items()}
        if set(memory.sums) != set(memory.counts):
            raise ValueError("Identity memory sums and counts do not match.")
        return memory


def flatten_embedding(output):
    """Reduce an encoder output to ``(batch, dim)``."""
    torch, F = _import_torch()
    if isinstance(output, (tuple, list)):
        output = output[-1]
    if output.ndim == 4:
        output = F.adaptive_avg_pool2d(output, 1)
    if output.ndim > 2:
        output = torch.flatten(output, 1)
    if output.ndim != 2:
        raise TypeError("Encoder forward must return a batched vector.")
    return output


def build_timm_encoder(model_name: str, num_classes: int = 0, pretrained: bool = False):
    """Build a timm network.

    ``num_classes == 0`` returns an L2-normalized embedding module. A positive
    class count returns the timm classification model unchanged. ``pretrained``
    stays false unless the caller sets it; ImageNet weights are not part of
    this repository.
    """
    try:
        import timm
    except ImportError as exc:
        raise ImportError("timm is required to build an encoder. Install requirements.txt.") from exc
    _, F = _import_torch()
    from torch import nn

    if num_classes > 0:
        model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        model.mutong_model_name = model_name
        model.mutong_num_classes = num_classes
        return model

    backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

    class NormalizedEmbedding(nn.Module):
        """timm backbone with a flattened, L2-normalized embedding."""

        def __init__(self, encoder):
            super().__init__()
            self.backbone = encoder

        def forward(self, images):
            features = self.backbone(images)
            features = flatten_embedding(features)
            return F.normalize(features, dim=1)

    network = NormalizedEmbedding(backbone)
    network.mutong_model_name = model_name
    network.mutong_num_classes = 0
    return network


def save_encoder_spec(
    model,
    output_dir: str | Path,
    model_name: str,
    num_classes: int,
) -> Path:
    """Save a state dict and a small JSON spec. No recognition figures are stored."""
    torch, _ = _import_torch()
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / "encoder_state.pt"
    spec_path = directory / "encoder_spec.json"
    torch.save(model.state_dict(), state_path)
    spec = {
        "format": ENCODER_SPEC_FORMAT,
        "builder": "timm",
        "model_name": model_name,
        "state_dict": state_path.name,
        "num_classes": int(num_classes),
    }
    spec_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return spec_path


def load_encoder(checkpoint: str | Path, device: str, pretrained: bool = False):
    """Load an encoder spec JSON or a pickled ``nn.Module``."""
    torch, _ = _import_torch()
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Encoder checkpoint not found: {checkpoint_path}. "
            "Train the metric branch or pass your own module. "
            "This repository does not include weights."
        )
    if checkpoint_path.suffix == ".json":
        spec = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if spec.get("format") != ENCODER_SPEC_FORMAT or spec.get("builder") != "timm":
            raise ValueError(
                "Unrecognized encoder spec. Expected a mutong-encoder-spec with builder timm."
            )
        model = build_timm_encoder(
            spec["model_name"],
            num_classes=int(spec.get("num_classes", 0)),
            pretrained=False,
        )
        state_path = checkpoint_path.parent / spec["state_dict"]
        if not state_path.is_file():
            raise FileNotFoundError(f"Encoder state dict not found: {state_path}")
        try:
            state = torch.load(state_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(state_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        return model

    try:
        model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(checkpoint_path, map_location=device)
    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            "Checkpoint must be an nn.Module saved with torch.save, or an encoder_spec.json. "
            "A state_dict alone does not name the architecture."
        )
    if pretrained:
        LOGGER.info("Ignoring pretrained=True because a full module was loaded from disk.")
    model.to(device)
    return model


def load_image_tensor(image_path: Path, image_size: int):
    """Load one RGB image as a float tensor shaped ``(3, image_size, image_size)``."""
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required to read images. Install requirements.txt.") from exc
    torch, _ = _import_torch()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (tensor - mean) / std


def second_view(images):
    """Second view for NT-Xent.

    The default is a horizontal flip so the update step can run. Replace this
    function with the experimental augmentation policy.
    """
    torch, _ = _import_torch()
    return torch.flip(images, dims=[-1])


def _chunks(items: list, size: int):
    if size < 1:
        raise ValueError("batch_size must be >= 1.")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def embed_paths(encoder, paths: list[Path], image_size: int, device: str, batch_size: int):
    torch, F = _import_torch()
    encoder.eval()
    rows = []
    for batch_paths in _chunks(paths, batch_size):
        images = torch.stack([load_image_tensor(path, image_size) for path in batch_paths]).to(device)
        with torch.no_grad():
            output = flatten_embedding(encoder(images))
            output = F.normalize(output, dim=1)
        rows.append(output.cpu())
    return torch.cat(rows, dim=0)


def read_unlabeled_csv(path: str | Path) -> list[str]:
    """Read a CSV whose only required column is ``image_path``."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Unlabeled CSV not found: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "image_path" not in reader.fieldnames:
            raise ValueError(f"{csv_path} must contain an image_path column.")
        paths = []
        for line_number, row in enumerate(reader, start=2):
            image_path = (row.get("image_path") or "").strip()
            if not image_path:
                raise ValueError(f"{csv_path}:{line_number} has an empty image_path.")
            paths.append(image_path)
    if not paths:
        raise ValueError(f"Unlabeled CSV is empty: {csv_path}")
    return paths


def _select_device(requested: str) -> str:
    torch, _ = _import_torch()
    if requested == "cpu":
        return "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {requested}, but CUDA is not available.")
    return requested


def _dynamic_options(args: argparse.Namespace) -> dict:
    config = load_yaml_config(args.config) if getattr(args, "config", None) else {}

    def pick(name, default, caster):
        value = getattr(args, name) if getattr(args, name) is not None else config.get(name, default)
        return caster(name, value)

    def as_int(name, value):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer.")
        return value

    def as_float(name, value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number.")
        return float(value)

    def as_str(name, value):
        if not isinstance(value, str) or not value:
            raise ValueError(f"{name} must be a non-empty string.")
        return value

    return {
        "image_size": pick("image_size", 224, as_int),
        "batch_size": pick("batch_size", 8, as_int),
        "epochs": pick("epochs", 1, as_int),
        "learning_rate": pick("learning_rate", 1e-4, as_float),
        "seed": pick("seed", 0, as_int),
        "device": pick("device", "cpu", as_str),
        "temperature": pick("temperature", 0.2, as_float),
        "memory_momentum": pick("memory_momentum", 0.9, as_float),
    }


def run_enroll(manifest: str, memory_path: str, checkpoint: str, options: dict) -> IdentityMemory:
    device = _select_device(options["device"])
    records = read_manifest(manifest)
    paths = [resolve_image_path(record["image_path"], manifest) for record in records]
    encoder = load_encoder(checkpoint, device)
    embeddings = embed_paths(encoder, paths, options["image_size"], device, options["batch_size"])
    memory = IdentityMemory()
    for record, embedding in zip(records, embeddings):
        memory.enroll(record["identity_id"], embedding)
    memory.save(memory_path)
    LOGGER.info("Enrolled %s identities into %s.", len(memory), memory_path)
    return memory


def run_update(unlabeled: str, memory_path: str, checkpoint: str, options: dict) -> None:
    torch, _ = _import_torch()
    device = _select_device(options["device"])
    if options["epochs"] < 1:
        raise ValueError("epochs must be >= 1.")
    image_paths = [
        resolve_image_path(path, unlabeled) for path in read_unlabeled_csv(unlabeled)
    ]
    if len(image_paths) < 2:
        raise RuntimeError("Self-supervised update needs at least two unlabeled images.")
    memory = IdentityMemory.load(memory_path)
    encoder = load_encoder(checkpoint, device)
    encoder.train()
    trainable = [parameter for parameter in encoder.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=options["learning_rate"]) if trainable else None
    loss_fn = NTXentLoss(temperature=options["temperature"])
    generator = torch.Generator()
    generator.manual_seed(options["seed"])
    for epoch in range(options["epochs"]):
        order = torch.randperm(len(image_paths), generator=generator).tolist()
        shuffled = [image_paths[index] for index in order]
        step_losses = []
        for batch_paths in _chunks(shuffled, options["batch_size"]):
            if len(batch_paths) < 2:
                continue
            images = torch.stack(
                [load_image_tensor(path, options["image_size"]) for path in batch_paths]
            ).to(device)
            views = second_view(images)
            first = flatten_embedding(encoder(images))
            second = flatten_embedding(encoder(views))
            loss = loss_fn(first, second)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                memory.assign_and_update(first.detach(), momentum=options["memory_momentum"])
            step_losses.append(float(loss.detach().cpu()))
        if not step_losses:
            raise RuntimeError("No update step ran. Increase the unlabeled set or the batch size.")
        LOGGER.info("Update epoch %s finished %s step(s).", epoch + 1, len(step_losses))
    memory.save(memory_path)
    if optimizer is None:
        LOGGER.info("Wrote updated memory to %s. The encoder had no trainable parameters.", memory_path)
        return
    dynamic_dir = Path(memory_path).parent / "dynamic"
    if hasattr(encoder, "mutong_model_name"):
        spec_path = save_encoder_spec(
            encoder,
            dynamic_dir,
            encoder.mutong_model_name,
            int(encoder.mutong_num_classes),
        )
        LOGGER.info("Wrote updated memory to %s and encoder spec %s.", memory_path, spec_path)
        return
    dynamic_dir.mkdir(parents=True, exist_ok=True)
    torch.save(encoder.state_dict(), dynamic_dir / "dynamic_encoder_state.pt")
    LOGGER.info(
        "Wrote updated memory to %s and a state dict under %s. "
        "The loaded module did not record its architecture, so no encoder spec was written.",
        memory_path,
        dynamic_dir,
    )


def run_identify(query: str, memory_path: str, checkpoint: str, options: dict) -> tuple[str, float]:
    device = _select_device(options["device"])
    query_path = Path(query)
    if not query_path.is_file():
        raise FileNotFoundError(f"Query image not found: {query_path}")
    memory = IdentityMemory.load(memory_path)
    encoder = load_encoder(checkpoint, device)
    embedding = embed_paths(encoder, [query_path], options["image_size"], device, batch_size=1)
    identity, distance = memory.identify(embedding[0])
    print(f"identity_id: {identity}")
    print(f"cosine_distance: {distance:.6f}")
    return identity, distance


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enroll, update, or query the dynamic cattle identity memory."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(command: argparse.ArgumentParser, checkpoint_help: str) -> None:
        command.add_argument("--checkpoint", required=True, help=checkpoint_help)
        command.add_argument("--config", default=None, help="YAML file of wiring defaults.")
        command.add_argument("--image-size", type=int, default=None)
        command.add_argument("--batch-size", type=int, default=None)
        command.add_argument("--device", default=None)
        command.add_argument("--seed", type=int, default=None)
        command.add_argument("--epochs", type=int, default=None)
        command.add_argument("--learning-rate", type=float, default=None)
        command.add_argument("--temperature", type=float, default=None)
        command.add_argument("--memory-momentum", type=float, default=None)

    enroll = subparsers.add_parser("enroll", help="Build prototypes from a labeled manifest.")
    enroll.add_argument("--manifest", required=True)
    enroll.add_argument("--memory", required=True, help="Output path for the identity memory.")
    add_common(enroll, "Encoder spec JSON or a torch.save'd nn.Module.")

    update = subparsers.add_parser("update", help="NT-Xent update from unlabeled frames.")
    update.add_argument("--unlabeled", required=True, help="CSV with an image_path column.")
    update.add_argument("--memory", required=True, help="Existing memory; updated in place.")
    add_common(update, "Encoder spec JSON or a torch.save'd nn.Module.")

    identify_parser = subparsers.add_parser("identify", help="Nearest prototype for one query image.")
    identify_parser.add_argument("--query", required=True, help="One query image.")
    identify_parser.add_argument("--memory", required=True)
    add_common(identify_parser, "Encoder spec JSON or a torch.save'd nn.Module.")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Command-line entry: ``python -m src.self_sup_dynamic``."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        options = _dynamic_options(args)
        if args.command == "enroll":
            run_enroll(args.manifest, args.memory, args.checkpoint, options)
        elif args.command == "update":
            run_update(args.unlabeled, args.memory, args.checkpoint, options)
        elif args.command == "identify":
            run_identify(args.query, args.memory, args.checkpoint, options)
        else:
            raise ValueError(f"Unknown command: {args.command}")
    except (FileNotFoundError, ValueError, RuntimeError, ImportError, TypeError, KeyError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
