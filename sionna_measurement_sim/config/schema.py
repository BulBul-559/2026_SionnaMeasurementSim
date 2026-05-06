"""Configuration schema placeholders for future phases."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExperimentConfig(BaseModel):
    """Minimal Phase 0 experiment config model."""

    experiment_id: str = Field(default="phase0-placeholder")
    random_seed: int = Field(default=0, ge=0)
