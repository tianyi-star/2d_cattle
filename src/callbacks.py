# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Training callbacks. Rank-1 / Rank-5 are computed on gallery vs query."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn

from src.data import filter_split
from src.evaluate import compute_rank_metrics, embed_split


@dataclass
class RankAccuracyCallback:
    """Evaluate Rank-1 / Rank-5 after selected epochs.

    The callback does not touch the ArcFace class-weight matrix. It embeds the
    gallery and query splits with the current backbone and ranks by cosine
    similarity of $\ell_2$-normalised vectors.
    """

    rows: list[dict[str, str]]
    root: Path
    image_size: int
    identity_to_index: dict[str, int]
    batch_size: int
    device: torch.device
    every_n_epochs: int = 1
    on_result: Callable[[dict], None] | None = None

    def __post_init__(self) -> None:
        self.gallery_rows = filter_split(self.rows, "gallery") or filter_split(self.rows, "train")
        self.query_rows = filter_split(self.rows, "query")
        if not self.gallery_rows or not self.query_rows:
            raise ValueError("RankAccuracyCallback requires non-empty gallery and query splits.")
        self.history: list[dict] = []

    def should_run(self, epoch: int, epochs: int) -> bool:
        if self.every_n_epochs <= 0:
            return False
        return epoch % self.every_n_epochs == 0 or epoch == epochs

    @torch.no_grad()
    def __call__(self, model: nn.Module, epoch: int) -> dict[str, float]:
        was_training = model.training
        model.eval()
        gallery_emb, gallery_ids = embed_split(
            model,
            self.gallery_rows,
            self.root,
            self.image_size,
            self.identity_to_index,
            self.batch_size,
            self.device,
        )
        query_emb, query_ids = embed_split(
            model,
            self.query_rows,
            self.root,
            self.image_size,
            self.identity_to_index,
            self.batch_size,
            self.device,
        )
        ranks = compute_rank_metrics(query_emb, query_ids, gallery_emb, gallery_ids)
        record = {
            "epoch": epoch,
            "rank1": float(ranks["rank1"]),
            "rank5": float(ranks["rank5"]),
            "num_gallery": len(self.gallery_rows),
            "num_query": len(self.query_rows),
        }
        self.history.append(record)
        print(
            f"  callback Rank-1={record['rank1']:.4f}  Rank-5={record['rank5']:.4f}",
            flush=True,
        )
        if self.on_result is not None:
            self.on_result(record)
        if was_training:
            model.train()
        return record
