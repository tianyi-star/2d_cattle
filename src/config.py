# Copyright 2026 Bo, Zhengzhou University / Mutong（牧瞳）
# SPDX-License-Identifier: Apache-2.0
"""Load the shared experiment file.

Paths are arguments. This module does not choose a machine directory.
"""

from __future__ import annotations

from pathlib import Path

from src.protocol import SHARED_CONFIG_KEYS


def load_yaml(path: str | Path) -> dict:
    """Read one YAML mapping. The caller decides where the file lives."""

    import yaml

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Config file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a mapping: {file_path}")
    return data


def require_shared_config(config: dict) -> None:
    """Check the keys that must stay identical across the three models."""

    missing = [key for key in SHARED_CONFIG_KEYS if config.get(key) is None]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Shared experiment config is missing values: {joined}")
