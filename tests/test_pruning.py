# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Pruning must remove units, not zero a copy, and must shrink the parameter count."""

from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from src.models import (
    build_model,
    count_parameters,
    units_to_prune,
    zero_pruned_units,
    zero_rows_in_place,
)

VIT = "vit_tiny_patch16_224"


def _config(ratio: float) -> dict:
    return {
        "embedding_dim": 128,
        "vit_name": VIT,
        "vit_prune_ratio": ratio,
        "yolo_backbone": "csp_small",
        "wang_backbone": "resnet18",
    }


# --------------------------------------------------------------------------- #
# bug (a): rounding rule
# --------------------------------------------------------------------------- #


def test_units_to_prune_is_not_truncation():
    """``int(3 * 0.25) == 0`` disabled head pruning entirely on ViT-Tiny."""

    assert int(3 * 0.25) == 0, "guard: truncation is still what the old code did"
    assert units_to_prune(3, 0.25) == 1
    assert units_to_prune(768, 0.25) == 192
    assert units_to_prune(36, 0.25) == 9
    assert units_to_prune(4, 0.5) == 2


def test_units_to_prune_always_leaves_one_unit():
    assert units_to_prune(1, 0.9) == 0
    assert units_to_prune(2, 0.99) == 1
    assert units_to_prune(3, 1.0) == 2
    assert units_to_prune(768, 0.0) == 0
    assert units_to_prune(0, 0.5) == 0


# --------------------------------------------------------------------------- #
# bug (b): advanced indexing zeroed a copy
# --------------------------------------------------------------------------- #


def test_advanced_index_zero_does_not_touch_the_parameter():
    """Documents the old defect so a regression is unambiguous."""

    weight = torch.ones(8, 4)
    weight[torch.tensor([2, 5])].zero_()
    assert int((weight.abs().sum(dim=1) == 0).sum()) == 0


def test_zero_rows_in_place_actually_writes():
    weight = torch.ones(8, 4)
    changed = zero_rows_in_place(weight, torch.tensor([2, 5]))
    assert changed == 2
    assert int((weight.abs().sum(dim=1) == 0).sum()) == 2
    assert weight[2].abs().sum() == 0 and weight[5].abs().sum() == 0
    assert weight[0].abs().sum() == 4


def test_zero_mode_zeroes_exactly_the_expected_mlp_rows():
    model = build_model("pruned_vit_arcface", _config(0.0))
    report = zero_pruned_units(model.backbone, 0.25)

    blocks = model.backbone.blocks
    hidden = blocks[0].mlp.fc1.out_features
    per_block = units_to_prune(hidden, 0.25)

    assert report["mlp_rows_expected"] == per_block * len(blocks) == 2304
    assert report["mlp_rows_zeroed"] == report["mlp_rows_expected"]
    assert report["mlp_rows_zeroed"] > 0, "0 of 192 intended rows is the bug being fixed"

    for block in blocks:
        zero_rows = int((block.mlp.fc1.weight.abs().sum(dim=1) == 0).sum())
        zero_cols = int((block.mlp.fc2.weight.abs().sum(dim=0) == 0).sum())
        assert zero_rows == per_block == 192
        assert zero_cols == per_block

    # Heads too: ViT-Tiny has three, so one per block must go at ratio 0.25.
    assert report["heads_expected"] == len(blocks) == 12
    assert report["heads_pruned"] == report["heads_expected"]
    for block in blocks:
        head_dim = block.attn.head_dim
        zero_rows = int((block.attn.qkv.weight.abs().sum(dim=1) == 0).sum())
        assert zero_rows == 3 * head_dim, "one of three heads must be zeroed in Q, K and V"
        zero_proj_cols = int((block.attn.proj.weight.abs().sum(dim=0) == 0).sum())
        assert zero_proj_cols == head_dim


# --------------------------------------------------------------------------- #
# physical removal: parameter count must fall
# --------------------------------------------------------------------------- #


def test_pruned_parameter_count_is_strictly_less_than_unpruned():
    unpruned = build_model("pruned_vit_arcface", _config(0.0))
    pruned = build_model("pruned_vit_arcface", _config(0.25))

    n_unpruned = count_parameters(unpruned)
    n_pruned = count_parameters(pruned)

    assert n_pruned < n_unpruned, (
        f"pruning left the parameter count unchanged ({n_pruned} vs {n_unpruned}); "
        "this is exactly the bug the audit reported"
    )
    report = pruned.prune_report
    assert report["applied"] is True
    assert report["heads_pruned"] == 12 and report["mlp_channels_pruned"] == 2304

    # A quarter of the MLP hidden width per block must be gone from fc1/fc2.
    block = pruned.backbone.blocks[0]
    assert block.mlp.fc1.out_features == 768 - 192
    assert block.mlp.fc2.in_features == 768 - 192
    assert block.attn.num_heads == 2
    assert block.attn.qkv.out_features == 3 * 2 * block.attn.head_dim


def test_pruned_and_unpruned_models_are_not_identical():
    torch.manual_seed(0)
    unpruned = build_model("pruned_vit_arcface", _config(0.0)).eval()
    pruned = build_model("pruned_vit_arcface", _config(0.25)).eval()

    x = torch.zeros(1, 3, 224, 224)
    with torch.no_grad():
        a = unpruned(x)
        b = pruned(x)

    assert a.shape == b.shape
    assert not torch.allclose(a, b), "a pruned model must not reproduce the unpruned output"

    pruned_shapes = {name: tuple(p.shape) for name, p in pruned.named_parameters()}
    unpruned_shapes = {name: tuple(p.shape) for name, p in unpruned.named_parameters()}
    differing = [k for k in unpruned_shapes if pruned_shapes.get(k) != unpruned_shapes[k]]
    assert differing, "no tensor changed shape, so nothing was structurally pruned"


def test_pruned_model_still_trains_one_step():
    model = build_model("pruned_vit_arcface", _config(0.25))
    model.train()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 128)
    out.sum().backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_flops_fall_with_pruning():
    from src.models import estimate_flops

    device = torch.device("cpu")
    unpruned = build_model("pruned_vit_arcface", _config(0.0))
    pruned = build_model("pruned_vit_arcface", _config(0.25))
    macs_unpruned = estimate_flops(unpruned, 224, device)
    macs_pruned = estimate_flops(pruned, 224, device)
    assert macs_pruned < macs_unpruned


def test_zero_ratio_is_a_no_op():
    model = build_model("pruned_vit_arcface", _config(0.0))
    assert model.prune_report["applied"] is False
    assert count_parameters(model) == count_parameters(build_model("pruned_vit_arcface", _config(0.0)))


def test_requested_pruning_that_removes_nothing_raises():
    """A model trained under the pruned name must never be silently unpruned."""

    from src.models import _prune_vit

    empty = nn.Module()
    empty.blocks = nn.ModuleList()
    with pytest.raises(RuntimeError, match="removed no units"):
        _prune_vit(empty, 0.25)


def test_pruned_checkpoint_reloads_exactly():
    """Evaluating rebuilds the model; load_state_dict must then be lossless.

    The head selection is deterministic, but evaluation rebuilds from config only.
    If the rebuilt shapes disagreed with the saved state, load would fail or
    silently drop entries.
    """

    torch.manual_seed(0)
    original = build_model("pruned_vit_arcface", _config(0.25))
    with torch.no_grad():
        for param in original.parameters():
            param.normal_(0.0, 0.02)
    x = torch.randn(2, 3, 224, 224)
    original.eval()
    with torch.no_grad():
        expected = original(x)

    torch.manual_seed(999)
    rebuilt = build_model("pruned_vit_arcface", _config(0.25))
    missing, unexpected = rebuilt.load_state_dict(original.state_dict(), strict=True)
    assert missing == [] and unexpected == []
    rebuilt.eval()
    with torch.no_grad():
        actual = rebuilt(x)
    assert torch.equal(expected, actual)
