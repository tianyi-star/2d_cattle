# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Three embedding networks that share one ArcFace training loss.

Each class records its role. Constructing the YOLO backbone, the pruned
Vision Transformer, or the Wang metric-learning backbone happens on the
private Ubuntu server. ``forward`` returns L2-normalized embeddings there.
"""

from __future__ import annotations

from src.protocol import MODEL_NAMES, MODEL_ROLES, stop_private


class IdentificationModel:
    """Inference embedding network. The ArcFace class weights are not part of it."""

    name = ""

    def __init__(self, config: dict) -> None:
        self.config = config

    @property
    def role(self) -> str:
        return MODEL_ROLES[self.name]

    def forward(self, images):
        """Map a batch of purified back-pattern images to embeddings."""

        del images
        stop_private(f"{self.name}.forward")


class YoloArcFace(IdentificationModel):
    """CNN baseline. The private server attaches a YOLO backbone to ArcFace."""

    name = "yolo_arcface"


class PrunedViTArcFace(IdentificationModel):
    """Proposed model. The private server prunes the ViT and attaches ArcFace.

    ``vit_name`` and ``vit_prune_ratio`` in the experiment file are filled on
    that server. This class does not choose a pruning rule.
    """

    name = "pruned_vit_arcface"


class WangMetricArcFace(IdentificationModel):
    """Wang et al. (2025) metric-learning backbone, trained with shared ArcFace.

    The original paper optimises a deep metric-learning objective. This
    comparison instead uses the shared ArcFace loss so the loss is not a
    confounding factor. The public tree does not include that paper's code
    or its reported figures.
    """

    name = "wang_metric"


BUILDERS = {
    YoloArcFace.name: YoloArcFace,
    PrunedViTArcFace.name: PrunedViTArcFace,
    WangMetricArcFace.name: WangMetricArcFace,
}


def build_model(name: str, config: dict) -> IdentificationModel:
    """Return the named shell. Layer construction is left to the private server."""

    try:
        model_cls = BUILDERS[name]
    except KeyError as exc:
        joined = ", ".join(MODEL_NAMES)
        raise ValueError(f"Unknown model {name!r}. Choose one of: {joined}") from exc
    return model_cls(config)
