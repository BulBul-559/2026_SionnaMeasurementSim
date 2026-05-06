"""Small configuration loader used by the Phase 0 skeleton."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON or YAML configuration file."""

    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")

    if config_path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text) or {}

    if not isinstance(data, dict):
        msg = f"Configuration root must be a mapping: {config_path}"
        raise ValueError(msg)

    return data
