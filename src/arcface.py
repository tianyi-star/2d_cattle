# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Shared ArcFace head for every identification model.

Deng, Guo, Xue, and Zafeiriou, ArcFace: Additive Angular Margin Loss for Deep
Face Recognition, CVPR 2019. The scale and margin come from the shared
experiment file. Changing them requires retraining all three models.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    """Additive angular margin over L2-normalized identity embeddings.

    ``forward`` adds the margin to the target angle, rescales the logits, and
    returns the cross-entropy. The class-weight matrix exists only while
    training. Rank retrieval and the parameter count use the embedding network,
    not this matrix.
    """

    def __init__(self, embedding_dim: int, num_classes: int, scale: float, margin: float) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.scale = float(scale)
        self.margin = float(margin)
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.threshold = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Return the shared training loss."""

        embeddings = F.normalize(embeddings, dim=1)
        weight = F.normalize(self.weight, dim=1)
        cosine = F.linear(embeddings, weight).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        sine = torch.sqrt(torch.clamp(1.0 - cosine * cosine, min=0.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits = logits * self.scale
        return F.cross_entropy(logits, labels)
