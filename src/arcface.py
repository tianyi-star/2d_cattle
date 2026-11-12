# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Shared ArcFace head for every identification model.

Deng, Guo, Xue, and Zafeiriou, ArcFace: Additive Angular Margin Loss for Deep
Face Recognition, CVPR 2019. The scale and margin come from the shared
experiment file. Changing them requires retraining all three models.
"""

from __future__ import annotations

from src.protocol import stop_private


class ArcFaceHead:
    """Additive angular margin over L2-normalized identity embeddings.

    On the private server this class subclasses ``torch.nn.Module``. ``forward``
    adds the margin to the target angle, rescales the logits, and returns the
    cross-entropy. The class-weight matrix exists only while training. Rank
    retrieval and the parameter count use the embedding network, not this matrix.
    """

    def __init__(self, embedding_dim: int, num_classes: int, scale: float, margin: float) -> None:
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.scale = scale
        self.margin = margin

    def forward(self, embeddings, labels):
        """Return the shared training loss. Not implemented in this repository."""

        del embeddings, labels
        stop_private("ArcFaceHead.forward")
