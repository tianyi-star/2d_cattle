# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Mutong (牧瞳): UAV-based individual cattle identification.

The package exposes the four pipeline stages: coat segmentation, F-ODS
dataset optimization, few-shot training, and self-supervised dynamic
identification. Names are imported on first access so ``python -m src.<module>``
does not execute a module twice. Importing this package does not load a checkpoint.
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "BatchHardTripletLoss": ("src.fewshot_train", "BatchHardTripletLoss"),
    "CoatSegmentor": ("src.coat_segmentation", "CoatSegmentor"),
    "IdentityMemory": ("src.self_sup_dynamic", "IdentityMemory"),
    "NTXentLoss": ("src.self_sup_dynamic", "NTXentLoss"),
    "PrototypicalLoss": ("src.fewshot_train", "PrototypicalLoss"),
    "TrainConfig": ("src.fewshot_train", "TrainConfig"),
    "build_model": ("src.coat_segmentation", "build_model"),
    "class_frequency_weights": ("src.f_ods_dataset", "class_frequency_weights"),
    "color_threshold_mask": ("src.coat_segmentation", "color_threshold_mask"),
    "load_encoder": ("src.self_sup_dynamic", "load_encoder"),
    "optimize_fods": ("src.f_ods_dataset", "optimize_fods"),
    "read_manifest": ("src.f_ods_dataset", "read_manifest"),
    "run_training": ("src.fewshot_train", "run_training"),
    "score_samples": ("src.f_ods_dataset", "score_samples"),
    "stratified_kshot": ("src.f_ods_dataset", "stratified_kshot"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(importlib.import_module(module_name), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
