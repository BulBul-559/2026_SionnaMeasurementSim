"""Configuration loader with pydantic schema validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from sionna_measurement_sim.config.schema import MeasurementConfig


def load_config(path: str | Path) -> MeasurementConfig:
    """Load and validate a YAML or JSON configuration file.

    Raises pydantic.ValidationError if the config is invalid.
    This should be called BEFORE starting RT/PHY so failures are caught early.
    """

    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")

    if config_path.suffix.lower() in (".yaml", ".yml"):
        data = yaml.safe_load(text) or {}
    elif config_path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text) or {}

    if not isinstance(data, dict):
        msg = f"Configuration root must be a mapping: {config_path}"
        raise ValueError(msg)

    return MeasurementConfig.model_validate(data)


def load_config_or_exit(path: str | Path) -> MeasurementConfig:
    """Load config or print the validation error and exit."""
    try:
        return load_config(path)
    except Exception as exc:
        print(f"Configuration error in {path}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
