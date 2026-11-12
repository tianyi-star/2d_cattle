# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Three embedding networks that share one ArcFace training loss.

* ``yolo_arcface`` — convolutional YOLO-ReID trunk
* ``pruned_vit_arcface`` — Vision Transformer with structured head/MLP pruning
* ``wang_metric`` — ResNet coat-pattern baseline of Wang et al. (2025)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import models

from src.protocol import MODEL_NAMES, MODEL_ROLES


def _l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=1)


class IdentificationModel(nn.Module):
    """Inference embedding network. The ArcFace class weights are not part of it."""

    name = ""

    def __init__(self, config: dict) -> None:
        super().__init__()
        self.config = config
        self.embedding_dim = int(config["embedding_dim"])

    @property
    def role(self) -> str:
        return MODEL_ROLES[self.name]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class YoloArcFace(IdentificationModel):
    """CNN baseline: a compact YOLO-style convolutional trunk with ArcFace."""

    name = "yolo_arcface"

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        width = str(config.get("yolo_backbone") or "csp_small")
        if width == "resnet18":
            trunk = models.resnet18(weights=None)
            feats = trunk.fc.in_features
            trunk.fc = nn.Identity()
            self.backbone = trunk
            self.projector = nn.Linear(feats, self.embedding_dim)
        else:
            # Lightweight CSP-like stack used when no external YOLO checkpoint is required.
            channels = [32, 64, 128, 256, 512]
            layers: list[nn.Module] = [_ConvBlock(3, channels[0], stride=2)]
            for in_ch, out_ch in zip(channels, channels[1:]):
                layers.append(_ConvBlock(in_ch, out_ch, stride=2))
            layers.append(nn.AdaptiveAvgPool2d(1))
            self.backbone = nn.Sequential(*layers)
            self.projector = nn.Linear(channels[-1], self.embedding_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        if features.ndim == 4:
            features = features.flatten(1)
        return _l2_normalize(self.projector(features))


def _prune_vit(model: nn.Module, prune_ratio: float) -> None:
    """Zero the lowest-magnitude attention heads and MLP channels.

    This is a practical structured prune that keeps the public interface and the
    FLOPs/parameter reporting path. The private server can replace the ranking
    rule with the task-specific discriminability score from the method draft.
    """

    ratio = float(prune_ratio or 0.0)
    if ratio <= 0.0:
        return
    ratio = min(max(ratio, 0.0), 0.9)
    for block in getattr(model, "blocks", []):
        attn = getattr(block, "attn", None)
        if attn is not None and hasattr(attn, "num_heads") and hasattr(attn, "qkv"):
            num_heads = int(attn.num_heads)
            head_dim = attn.qkv.in_features // (3 * num_heads)
            weight = attn.qkv.weight.detach()
            # Importance of head h from the Q projection slice.
            scores = []
            for head in range(num_heads):
                start = head * head_dim
                end = start + head_dim
                scores.append(weight[start:end].abs().mean().item())
            n_drop = int(num_heads * ratio)
            if n_drop > 0:
                drop = set(sorted(range(num_heads), key=lambda i: scores[i])[:n_drop])
                with torch.no_grad():
                    for head in drop:
                        for part in range(3):
                            start = part * num_heads * head_dim + head * head_dim
                            end = start + head_dim
                            attn.qkv.weight[start:end].zero_()
                            if attn.qkv.bias is not None:
                                attn.qkv.bias[start:end].zero_()
        mlp = getattr(block, "mlp", None)
        if mlp is not None and hasattr(mlp, "fc1") and hasattr(mlp, "fc2"):
            hidden = mlp.fc1.out_features
            n_drop = int(hidden * ratio)
            if n_drop > 0:
                scores = mlp.fc1.weight.detach().abs().mean(dim=1)
                drop_idx = torch.argsort(scores)[:n_drop]
                with torch.no_grad():
                    mlp.fc1.weight[drop_idx].zero_()
                    if mlp.fc1.bias is not None:
                        mlp.fc1.bias[drop_idx].zero_()
                    mlp.fc2.weight[:, drop_idx].zero_()


class PrunedViTArcFace(IdentificationModel):
    """Proposed model: a pruned Vision Transformer with ArcFace."""

    name = "pruned_vit_arcface"

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        vit_name = str(config.get("vit_name") or "vit_tiny_patch16_224")
        self.backbone = timm.create_model(
            vit_name,
            pretrained=False,
            num_classes=0,
            global_pool="token",
        )
        feats = int(self.backbone.num_features)
        self.projector = nn.Linear(feats, self.embedding_dim)
        _prune_vit(self.backbone, float(config.get("vit_prune_ratio") or 0.0))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return _l2_normalize(self.projector(features))


class WangMetricArcFace(IdentificationModel):
    """Wang et al. (2025) ResNet coat-pattern backbone, trained with ArcFace."""

    name = "wang_metric"

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        backbone_name = str(config.get("wang_backbone") or "resnet50")
        if backbone_name == "resnet18":
            trunk = models.resnet18(weights=None)
        else:
            trunk = models.resnet50(weights=None)
        feats = trunk.fc.in_features
        trunk.fc = nn.Identity()
        self.backbone = trunk
        self.projector = nn.Linear(feats, self.embedding_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        return _l2_normalize(self.projector(features))


BUILDERS = {
    YoloArcFace.name: YoloArcFace,
    PrunedViTArcFace.name: PrunedViTArcFace,
    WangMetricArcFace.name: WangMetricArcFace,
}


def build_model(name: str, config: dict) -> IdentificationModel:
    """Construct the named embedding network."""

    try:
        model_cls = BUILDERS[name]
    except KeyError as exc:
        joined = ", ".join(MODEL_NAMES)
        raise ValueError(f"Unknown model {name!r}. Choose one of: {joined}") from exc
    return model_cls(config)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(model: nn.Module, image_size: int, device: torch.device) -> float:
    """Rough MAC count via one forward with hooks when fvcore is unavailable."""

    model = model.to(device)
    model.eval()
    macs = 0

    def conv_hook(module: nn.Conv2d, inputs: tuple[Any, ...], _output: torch.Tensor) -> None:
        nonlocal macs
        x = inputs[0]
        _, _, h, w = _output.shape if _output.ndim == 4 else (0, 0, 1, 1)
        kh, kw = module.kernel_size
        macs += int(module.in_channels * module.out_channels * kh * kw * h * w / module.groups)

    def linear_hook(module: nn.Linear, inputs: tuple[Any, ...], _output: torch.Tensor) -> None:
        nonlocal macs
        x = inputs[0]
        macs += int(x.shape[-1] * module.out_features * (x.numel() // x.shape[-1]))

    handles = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))
    with torch.no_grad():
        dummy = torch.zeros(1, 3, image_size, image_size, device=device)
        model(dummy)
    for handle in handles:
        handle.remove()
    return float(macs)
