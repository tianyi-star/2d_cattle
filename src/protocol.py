# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Shared comparison contract for the three identification models.

The public repository records the protocol. It does not store images,
checkpoints, or measured recognition figures. The private Ubuntu server
implements the steps named here.
"""

from __future__ import annotations

MODEL_NAMES = ("yolo_arcface", "pruned_vit_arcface", "wang_metric")

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
COMPARISON_CROP = CROP_SAM

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

RESERVED_BBOX = (
    "The YOLO bounding-box crop is a reserved control. It skips SAM and is not "
    "part of the three-model comparison. The private server can add it later."
)


def stop_private(step: str) -> None:
    """Mark a step whose body is intentionally absent from the public tree."""

    raise NotImplementedError(f"{step}. {PRIVATE_SERVER}")


def stop_reserved_bbox() -> None:
    """Mark the unused no-SAM control. It is wired, not implemented."""

    raise NotImplementedError(RESERVED_BBOX)
