# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Shared comparison contract for the 2 by 3 identification ablation.

Two crops and three backbones form six cells. The public repository records
the protocol. It does not store images, checkpoints, or measured recognition
figures. The private Ubuntu server implements the steps named here.
"""

from __future__ import annotations

MODEL_NAMES = ("yolo_arcface", "pruned_vit_arcface", "wang_metric")

DISPLAY_NAMES = {
    "yolo_arcface": "YOLO-ReID",
    "pruned_vit_arcface": "Pruned ViT",
    "wang_metric": "Yaowu Wang",
}

MODEL_ROLES = {
    "yolo_arcface": "CNN baseline: a YOLO backbone with the shared ArcFace head.",
    "pruned_vit_arcface": "Proposed model: a pruned Vision Transformer with the shared ArcFace head.",
    "wang_metric": (
        "Baseline interface for the coat-pattern metric-learning setup of "
        "Wang et al., Computers and Electronics in Agriculture (2025), "
        "trained here with the same ArcFace loss."
    ),
}

CROP_SAM = "sam"
CROP_BBOX = "bbox"
CROP_MODES = (CROP_SAM, CROP_BBOX)

CROP_ROLES = {
    CROP_SAM: (
        "SAM mask crop of the cattle back. Background, shadow, and legs fall outside the mask."
    ),
    CROP_BBOX: (
        "YOLO bounding-box crop of the original frame. SAM is not applied, "
        "so background, shadow, and legs can remain inside the crop."
    ),
}

SAM_ABLATION_QUESTION = (
    "How much does a SAM mask that removes background, shadow, and legs "
    "raise Rank-1 for cattle back-coat re-identification?"
)

# Negative Δ_SAM is a valid experimental outcome. It arises when residual scene
# context in the YOLO box is identity-correlated (pen floor, feeder, resting
# spot) and the model uses that shortcut. The purified coat crop then loses the
# cue, so Rank-1(bbox) can exceed Rank-1(sam). Report the sign; do not assume
# that background removal always helps.
CONTEXT_SHORTCUT_NOTE = (
    "A negative Δ_SAM indicates that the bbox arm beat the SAM arm. "
    "Under identity-linked scene context this is shortcut learning, "
    "not evidence that background is a superior biometric to the coat."
)

# Retrieval ranks plus the two cost figures. No other score belongs in the comparison table.
METRIC_NAMES = ("rank1", "rank5", "parameters", "flops")

SHARED_CONFIG_KEYS = (
    "image_size",
    "embedding_dim",
    "arcface_scale",
    "arcface_margin",
    "batch_size",
    "epochs",
    "learning_rate",
    "seed",
)

PRIVATE_SERVER = (
    "The implementation of this step stays on the private Ubuntu server. "
    "This repository only keeps the interface."
)


def stop_private(step: str) -> None:
    """Mark a step whose body is intentionally absent from the public tree."""

    raise NotImplementedError(f"{step}. {PRIVATE_SERVER}")


def require_crop(crop: str) -> str:
    """Accept either arm of the ablation. Neither crop is optional."""

    if crop not in CROP_MODES:
        joined = ", ".join(CROP_MODES)
        raise ValueError(f"crop must be one of: {joined}")
    return crop


def comparison_cells() -> tuple[tuple[str, str], ...]:
    """Return the six cells as ``(crop, model)`` pairs, SAM row first."""

    return tuple((crop, model) for crop in CROP_MODES for model in MODEL_NAMES)
