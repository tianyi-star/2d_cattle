# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Mutong (牧瞳): three backbones crossed with a SAM crop and a YOLO-box crop.

Names are imported on first access so ``python -m src.<module>`` does not
execute a module twice. Importing this package does not load a checkpoint.
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "ArcFaceHead": ("src.arcface", "ArcFaceHead"),
    "IdentificationModel": ("src.models", "IdentificationModel"),
    "DISPLAY_NAMES": ("src.protocol", "DISPLAY_NAMES"),
    "METRIC_NAMES": ("src.protocol", "METRIC_NAMES"),
    "MODEL_NAMES": ("src.protocol", "MODEL_NAMES"),
    "comparison_cells": ("src.protocol", "comparison_cells"),
    "PrunedViTArcFace": ("src.models", "PrunedViTArcFace"),
    "WangMetricArcFace": ("src.models", "WangMetricArcFace"),
    "YoloArcFace": ("src.models", "YoloArcFace"),
    "build_model": ("src.models", "build_model"),
    "metric_names": ("src.evaluate", "metric_names"),
    "rank1_gain": ("src.evaluate", "rank1_gain"),
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
