"""Validation helpers for domain dataclasses."""

from __future__ import annotations

import numpy as np


def require_shape(name: str, value: np.ndarray, expected: tuple[int | None, ...]) -> None:
    """Validate a NumPy array rank and fixed dimensions."""

    if value.ndim != len(expected):
        msg = f"{name} must have rank {len(expected)}, got shape {value.shape}"
        raise ValueError(msg)

    for axis, (actual, wanted) in enumerate(zip(value.shape, expected, strict=True)):
        if wanted is not None and actual != wanted:
            msg = f"{name} axis {axis} must be {wanted}, got shape {value.shape}"
            raise ValueError(msg)


def require_finite(name: str, value: np.ndarray) -> None:
    """Require all real or complex array entries to be finite."""

    if not np.all(np.isfinite(value)):
        msg = f"{name} must contain only finite values"
        raise ValueError(msg)
