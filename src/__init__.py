# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Mutong (牧瞳): three identification models on one SAM-purified input.

Names are imported on first access so ``python -m src.<module>`` does not
execute a module twice. Importing this package does not load a checkpoint.
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "ArcFaceHead": ("src.arcface", "ArcFaceHead"),
    "IdentificationModel": ("src.models", "IdentificationModel"),
    "METRIC_NAMES": ("src.protocol", "METRIC_NAMES"),
    "MODEL_NAMES": ("src.protocol", "MODEL_NAMES"),
    "PrunedViTArcFace": ("src.models", "PrunedViTArcFace"),
    "WangMetricArcFace": ("src.models", "WangMetricArcFace"),
    "YoloArcFace": ("src.models", "YoloArcFace"),
    "build_model": ("src.models", "build_model"),
    "metric_names": ("src.evaluate", "metric_names"),
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
