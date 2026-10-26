"""Prepare fixed, upright device orientations in the original scene frame."""

from __future__ import annotations

import copy
import math
from typing import Any

import numpy as np


def orient_label_to_map_center(
    label: dict[str, Any],
    floorplan_meta: dict[str, Any],
    *,
    bs_fallback_rad: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ue_orientation_rad: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[dict[str, Any], list[float]]:
    """Return a derived label and map center; retain all original sample identities.

    Only top-level points define the simulation topology. Groups and other
    original metadata are preserved. No visibility or nearest-room search runs.
    """
    origin = np.asarray(floorplan_meta["origin_xy_m"], dtype=np.float64)
    extent = np.asarray(floorplan_meta["extent_xy_m"], dtype=np.float64)
    if origin.shape != (2,) or extent.shape != (2,):
        raise ValueError("floorplan origin_xy_m and extent_xy_m must have shape [2]")
    if not np.isfinite(origin).all() or not np.isfinite(extent).all() or np.any(extent <= 0):
        raise ValueError("floorplan coordinates must be finite and extent must be positive")
    for value in (bs_fallback_rad, ue_orientation_rad):
        angles = np.asarray(value, dtype=np.float64)
        if angles.shape != (3,) or not np.isfinite(angles).all():
            raise ValueError("fallback and UE orientation must be finite [yaw,pitch,roll]")
    # This preparation policy keeps every BS upright, including the fallback.
    if bs_fallback_rad[1] != 0 or bs_fallback_rad[2] != 0:
        raise ValueError("upright BS fallback requires zero pitch and roll")
    center = origin + extent / 2
    derived = copy.deepcopy(label)
    for key in ("bs_points", "ue_points"):
        if not isinstance(derived.get(key), list) or not derived[key]:
            raise ValueError(f"label must contain non-empty top-level {key}")
    for point in derived["bs_points"]:
        position = point.get("position")
        if position is None:
            position = [point[axis] for axis in ("x", "y", "z")]
        position = np.asarray(position, dtype=np.float64)
        if position.shape != (3,) or not np.isfinite(position).all():
            raise ValueError("BS position must be finite [x,y,z]")
        delta = center - position[:2]
        yaw = (
            float(bs_fallback_rad[0]) if np.linalg.norm(delta) <= 1e-9
            else math.atan2(float(delta[1]), float(delta[0]))
        )
        point["orientation_rad"] = [yaw, 0.0, 0.0]
    for point in derived["ue_points"]:
        point["orientation_rad"] = [float(value) for value in ue_orientation_rad]
    return derived, center.tolist()
