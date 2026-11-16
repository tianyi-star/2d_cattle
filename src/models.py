# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Three embedding networks that share one ArcFace training loss.

* ``yolo_arcface`` — convolutional YOLO-ReID trunk
* ``pruned_vit_arcface`` — Vision Transformer with structured head/MLP pruning
* ``wang_metric`` — ResNet coat-pattern baseline of Wang et al. (2025)
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import models

from src.protocol import MODEL_NAMES, MODEL_ROLES

logger = logging.getLogger(__name__)


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


def units_to_prune(num_units: int, ratio: float) -> int:
    """How many units of ``num_units`` to remove at ``ratio``.

    Rounding is half-up with a floor of one unit, because ``int(n * ratio)``
    truncates to zero for small unit counts. ``vit_tiny_patch16_224`` has three
    attention heads, so ``int(3 * 0.25) == 0`` silently disabled head pruning.
    At least one unit always survives, so a block cannot be emptied.
    """

    if num_units <= 0:
        return 0
    ratio = float(ratio)
    if ratio <= 0.0:
        return 0
    count = int(math.floor(num_units * min(ratio, 1.0) + 0.5))
    if count == 0:
        count = 1
    return min(count, num_units - 1)


# Criterion note (historical; the papers are now aligned to this code):
#   paper/method.tex sec:pruning and paper/manuscript-cea.md section 3.8 both
#   describe the ranking below as the implemented criterion, namely weight
#   magnitude |w|. The discriminability-ratio rule
#   I(u) = Delta(phi) - Delta(phi^(-u)), with
#   Delta = S_inter / (S_intra + eps) on a calibration split, appears in
#   method.tex only as an explicitly labelled design direction that was NOT
#   evaluated. It is not implemented here and no reported figure uses it.
#   Weight magnitude is what this code does: deterministic, no calibration
#   pass, no extra forward passes.
def _magnitude_scores(weight: torch.Tensor) -> torch.Tensor:
    """Mean absolute weight per row. Higher means more important."""

    return weight.detach().abs().mean(dim=1)


def _shrink_linear(
    linear: nn.Linear,
    keep_rows: torch.Tensor,
    keep_cols: torch.Tensor | None = None,
) -> nn.Linear:
    """Return a new ``nn.Linear`` with only ``keep_rows``/``keep_cols`` retained."""

    keep_rows = keep_rows.to(linear.weight.device)
    weight = linear.weight.detach()[keep_rows]
    if keep_cols is not None:
        keep_cols = keep_cols.to(linear.weight.device)
        weight = weight[:, keep_cols]
    bias = None
    if linear.bias is not None:
        bias = linear.bias.detach()[keep_rows]
    shrunk = nn.Linear(
        int(weight.shape[1]),
        int(weight.shape[0]),
        bias=bias is not None,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    with torch.no_grad():
        shrunk.weight.copy_(weight)
        if bias is not None:
            shrunk.bias.copy_(bias)
    return shrunk


def zero_rows_in_place(weight: torch.Tensor, rows: torch.Tensor) -> int:
    """Zero ``weight[rows]`` in place and return how many rows actually changed.

    ``weight[rows].zero_()`` is an advanced-index *getitem*, which returns a copy:
    the in-place call then zeroes the copy and the parameter is untouched. This
    helper writes through the parameter itself. The count of rows whose values
    actually changed is returned so a caller can assert the write landed.
    """

    rows = rows.to(weight.device).long().reshape(-1)
    if rows.numel() == 0:
        return 0
    before = weight.detach().clone()
    zeros = torch.zeros(
        (rows.numel(), *weight.shape[1:]), dtype=weight.dtype, device=weight.device
    )
    with torch.no_grad():
        # index_put_ writes through the parameter's storage. ``weight[rows].zero_()``
        # would zero a temporary copy and leave the parameter untouched.
        weight.detach().index_put_((rows,), zeros)
    changed = (before[rows] != weight.detach()[rows]).any(dim=-1)
    return int(changed.sum().item())


def zero_columns_in_place(weight: torch.Tensor, cols: torch.Tensor) -> int:
    """Zero ``weight[:, cols]`` in place and return how many columns changed."""

    cols = cols.to(weight.device).long().reshape(-1)
    if cols.numel() == 0:
        return 0
    before = weight.detach().clone()
    with torch.no_grad():
        # Advanced-index assignment on the parameter itself; a getitem would copy.
        weight.detach()[:, cols] = 0
    changed = (before[:, cols] != weight.detach()[:, cols]).any(dim=0)
    return int(changed.sum().item())


def zero_pruned_units(model: nn.Module, prune_ratio: float) -> dict:
    """Zero (not remove) the lowest-magnitude heads and MLP channels in place.

    This is the literal form the manuscript describes for ``phi^{(-u)}``: the
    unit is zeroed and no other weight changes. Parameter count therefore does
    not fall; the arithmetic still runs. Used to check that the selection is
    non-trivial; :func:`_prune_vit` is what makes the exported model smaller.
    """

    ratio = float(prune_ratio or 0.0)
    report: dict[str, Any] = {
        "prune_ratio": ratio,
        "criterion": "weight_magnitude",
        "mode": "in_place_zero",
        "heads_pruned": 0,
        "mlp_rows_zeroed": 0,
        "mlp_rows_expected": 0,
        "heads_expected": 0,
    }
    if ratio <= 0.0:
        return report
    ratio = min(ratio, 0.9)
    for block in getattr(model, "blocks", []):
        attn = getattr(block, "attn", None)
        if attn is not None and hasattr(attn, "num_heads") and hasattr(attn, "qkv"):
            num_heads = int(attn.num_heads)
            head_dim = int(attn.qkv.out_features // (3 * num_heads))
            rows = attn.qkv.weight.detach().reshape(3, num_heads, head_dim, -1)
            scores = rows.abs().mean(dim=(2, 3)).mean(dim=0).tolist()
            n_drop = units_to_prune(num_heads, ratio)
            report["heads_expected"] += n_drop
            drop = sorted(range(num_heads), key=lambda i: scores[i])[:n_drop]
            for head in drop:
                changed_parts = 0
                for part in range(3):
                    start = part * num_heads * head_dim + head * head_dim
                    idx = torch.arange(start, start + head_dim)
                    changed_parts += 1 if zero_rows_in_place(attn.qkv.weight, idx) else 0
                    if attn.qkv.bias is not None:
                        zero_rows_in_place(attn.qkv.bias, idx)
                # The head's V slice feeds these projection columns, so zero them too.
                proj_cols = torch.arange(head * head_dim, (head + 1) * head_dim)
                report["head_proj_cols_zeroed"] = report.get(
                    "head_proj_cols_zeroed", 0
                ) + zero_columns_in_place(attn.proj.weight, proj_cols)
                report["heads_pruned"] += 1 if changed_parts == 3 else 0
        mlp = getattr(block, "mlp", None)
        if mlp is not None and hasattr(mlp, "fc1") and hasattr(mlp, "fc2"):
            hidden = int(mlp.fc1.out_features)
            n_drop = units_to_prune(hidden, ratio)
            report["mlp_rows_expected"] += n_drop
            if n_drop > 0:
                scores = _magnitude_scores(mlp.fc1.weight)
                drop_idx = torch.argsort(scores)[:n_drop]
                report["mlp_rows_zeroed"] += zero_rows_in_place(mlp.fc1.weight, drop_idx)
                if mlp.fc1.bias is not None:
                    zero_rows_in_place(mlp.fc1.bias, drop_idx)
                report["mlp_cols_zeroed"] = report.get("mlp_cols_zeroed", 0) + zero_columns_in_place(
                    mlp.fc2.weight, drop_idx
                )
    return report


def _prune_attention_heads(attn: nn.Module, drop_heads: set[int]) -> None:
    """Physically remove heads from a timm-style ``Attention`` module.

    Removes the Q, K and V output rows and the projection input columns of each
    dropped head, then updates ``num_heads`` / ``attn_dim`` so the fused-attention
    reshape stays consistent. Dropping rows instead of zeroing them is what makes
    the parameter count fall; a zeroed row is still a parameter.
    """

    num_heads = int(attn.num_heads)
    head_dim = int(attn.qkv.out_features // (3 * num_heads))
    old_attn_dim = num_heads * head_dim
    keep_heads = [h for h in range(num_heads) if h not in drop_heads]

    row_keep = torch.cat(
        [
            torch.arange(part * old_attn_dim + h * head_dim, part * old_attn_dim + (h + 1) * head_dim)
            for part in range(3)
            for h in keep_heads
        ]
    )
    attn.qkv = _shrink_linear(attn.qkv, row_keep)
    if getattr(attn, "gate", None) is not None:
        gate_keep = torch.cat(
            [torch.arange(h * head_dim, (h + 1) * head_dim) for h in keep_heads]
        )
        attn.gate = _shrink_linear(attn.gate, gate_keep)
    col_keep = torch.cat([torch.arange(h * head_dim, (h + 1) * head_dim) for h in keep_heads])
    attn.proj = _shrink_linear(attn.proj, torch.arange(attn.proj.out_features), col_keep)
    if isinstance(getattr(attn, "norm", None), nn.LayerNorm):
        # ``scale_norm=True`` normalises the concatenated head output.
        norm = attn.norm
        attn.norm = nn.LayerNorm(
            len(keep_heads) * head_dim,
            eps=norm.eps,
            elementwise_affine=norm.elementwise_affine,
            device=norm.weight.device,
            dtype=norm.weight.dtype,
        )
        with torch.no_grad():
            attn.norm.weight.copy_(norm.weight.detach()[col_keep])
            attn.norm.bias.copy_(norm.bias.detach()[col_keep])
    attn.num_heads = len(keep_heads)
    attn.attn_dim = len(keep_heads) * head_dim
    attn.head_dim = head_dim


def _prune_mlp_channels(mlp: nn.Module, drop_idx: torch.Tensor) -> None:
    """Physically remove hidden channels from an ``fc1``/``fc2`` MLP block."""

    hidden = int(mlp.fc1.out_features)
    mask = torch.ones(hidden, dtype=torch.bool, device=mlp.fc1.weight.device)
    mask[drop_idx.to(mlp.fc1.weight.device)] = False
    keep = torch.nonzero(mask, as_tuple=False).flatten()
    mlp.fc1 = _shrink_linear(mlp.fc1, keep)
    mlp.fc2 = _shrink_linear(mlp.fc2, torch.arange(mlp.fc2.out_features), keep)
    norm = getattr(mlp, "norm", None)
    if isinstance(norm, nn.LayerNorm) and norm.weight is not None and norm.weight.numel() == hidden:
        new_norm = nn.LayerNorm(
            keep.numel(),
            eps=norm.eps,
            elementwise_affine=norm.elementwise_affine,
            device=norm.weight.device,
            dtype=norm.weight.dtype,
        )
        with torch.no_grad():
            new_norm.weight.copy_(norm.weight.detach()[keep])
            new_norm.bias.copy_(norm.bias.detach()[keep])
        mlp.norm = new_norm


def _prune_vit(model: nn.Module, prune_ratio: float) -> dict:
    """Structured pruning of the ViT: lowest-magnitude heads and MLP channels.

    The importance score is plain weight magnitude (mean absolute weight of the
    unit, in the Q/K/V rows for a head and in the ``fc1`` row for a channel).

    Criterion note: both ``paper/method.tex`` (``sec:pruning``) and
    ``paper/manuscript-cea.md`` (section 3.8) describe this function as the
    implemented rule, namely ranking by weight magnitude. The
    *discriminability-ratio* criterion
    $I(u) = \\Delta(\\phi) - \\Delta(\\phi^{(-u)})$ with
    $\\Delta = S_{inter} / (S_{intra} + \\varepsilon)$ on a calibration split is
    documented there only as a design direction that was NOT evaluated. This
    function does not implement it, and no reported figure uses it. See the
    note above :func:`_magnitude_scores` for the history.

    Units are removed, not zeroed: the tensors are rebuilt without the dropped
    rows/columns, so :func:`count_parameters` falls. Returns a report of what was
    pruned, which is also stored on ``model.prune_report``.
    """

    ratio = float(prune_ratio or 0.0)
    report: dict[str, Any] = {
        "prune_ratio": ratio,
        "criterion": "weight_magnitude",
        "blocks": 0,
        "heads_per_block": [],
        "mlp_channels_per_block": [],
        "heads_before": 0,
        "heads_pruned": 0,
        "mlp_channels_before": 0,
        "mlp_channels_pruned": 0,
        "applied": False,
    }
    if ratio <= 0.0:
        model.prune_report = report
        return report
    ratio = min(max(ratio, 0.0), 0.9)
    report["prune_ratio"] = ratio

    for block in getattr(model, "blocks", []):
        attn = getattr(block, "attn", None)
        n_heads_dropped = 0
        if attn is not None and hasattr(attn, "num_heads") and hasattr(attn, "qkv"):
            num_heads = int(attn.num_heads)
            head_dim = int(attn.qkv.out_features // (3 * num_heads))
            report["heads_before"] += num_heads
            # Magnitude of a head = mean |w| over its Q, K and V output rows.
            qkv_rows = attn.qkv.weight.detach().reshape(3, num_heads, head_dim, -1)
            scores = qkv_rows.abs().mean(dim=(2, 3)).mean(dim=0).tolist()
            n_drop = units_to_prune(num_heads, ratio)
            if n_drop > 0:
                drop = set(sorted(range(num_heads), key=lambda i: scores[i])[:n_drop])
                _prune_attention_heads(attn, drop)
                n_heads_dropped = len(drop)
        report["heads_per_block"].append(n_heads_dropped)
        report["heads_pruned"] += n_heads_dropped

        mlp = getattr(block, "mlp", None)
        n_channels_dropped = 0
        if mlp is not None and hasattr(mlp, "fc1") and hasattr(mlp, "fc2"):
            hidden = int(mlp.fc1.out_features)
            report["mlp_channels_before"] += hidden
            n_drop = units_to_prune(hidden, ratio)
            if n_drop > 0:
                scores = _magnitude_scores(mlp.fc1.weight)
                drop_idx = torch.argsort(scores)[:n_drop]
                _prune_mlp_channels(mlp, drop_idx)
                n_channels_dropped = int(drop_idx.numel())
        report["mlp_channels_per_block"].append(n_channels_dropped)
        report["mlp_channels_pruned"] += n_channels_dropped
        report["blocks"] += 1

    report["applied"] = bool(report["heads_pruned"] or report["mlp_channels_pruned"])
    model.prune_report = report
    if not report["applied"]:
        raise RuntimeError(
            f"Structured pruning was requested at ratio {ratio} but removed no units "
            f"({report['blocks']} blocks inspected). This is the silent no-op the "
            "evidence audit found; fix the rounding rule or the block layout instead "
            "of training an unpruned model under a pruned name."
        )
    logger.info(
        "Structured pruning at ratio %.3f: %d/%d heads, %d/%d MLP channels removed "
        "across %d blocks (criterion: weight magnitude).",
        ratio,
        report["heads_pruned"],
        report["heads_before"],
        report["mlp_channels_pruned"],
        report["mlp_channels_before"],
        report["blocks"],
    )
    return report


def _prune_vit_summary_line(report: dict) -> str:
    return (
        f"pruned {report['heads_pruned']}/{report['heads_before']} heads and "
        f"{report['mlp_channels_pruned']}/{report['mlp_channels_before']} MLP channels "
        f"(blocks={report['blocks']}, ratio={report['prune_ratio']})"
    )



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
        self.prune_report = _prune_vit(
            self.backbone, float(config.get("vit_prune_ratio") or 0.0)
        )
        logger.info("pruned_vit_arcface: %s", _prune_vit_summary_line(self.prune_report))

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

    def conv_hook(module: nn.Conv2d, _inputs: tuple[Any, ...], output: torch.Tensor) -> None:
        nonlocal macs
        _, _, h, w = output.shape if output.ndim == 4 else (0, 0, 1, 1)
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
