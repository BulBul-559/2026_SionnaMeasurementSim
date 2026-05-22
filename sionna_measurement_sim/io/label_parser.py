"""Label/topology parsing helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from sionna_measurement_sim.domain.topology import (
    RoleTopology,
    Topology,
    resolve_link_roles,
    resolve_role_topology,
)

STANDARD_LABEL_SCHEMA_VERSION = "0.1.0"


def load_topology_from_label(
    label_file: str | Path,
    *,
    max_bs: int = 1,
    max_ue: int = 1,
    ue_start: int = 0,
    ue_count: int | None = None,
    ue_indices: list[int] | tuple[int, ...] | None = None,
    bs_indices: list[int] | tuple[int, ...] | None = None,
    max_tx: int | None = None,
    max_rx: int | None = None,
    rx_start: int | None = None,
    rx_count: int | None = None,
    rx_indices: list[int] | tuple[int, ...] | None = None,
    tx_indices: list[int] | tuple[int, ...] | None = None,
) -> Topology:
    """Load a small TX/RX topology from the prepared test label JSON."""

    if max_tx is not None:
        max_bs = max_tx
    if max_rx is not None:
        max_ue = max_rx
    if rx_start is not None:
        ue_start = rx_start
    if rx_count is not None:
        ue_count = rx_count
    if rx_indices is not None:
        ue_indices = rx_indices
    if tx_indices is not None:
        bs_indices = tx_indices

    role_topology = load_role_topology_from_label(
        label_file,
        max_bs=max_bs,
        max_ue=max_ue,
        ue_start=ue_start,
        ue_count=ue_count,
        ue_indices=ue_indices,
        bs_indices=bs_indices,
    )
    return resolve_role_topology(role_topology, resolve_link_roles("downlink"))


def load_role_topology_from_label(
    label_file: str | Path,
    *,
    max_bs: int = 1,
    max_ue: int = 1,
    ue_start: int = 0,
    ue_count: int | None = None,
    ue_indices: list[int] | tuple[int, ...] | None = None,
    bs_indices: list[int] | tuple[int, ...] | None = None,
) -> RoleTopology:
    """Load a BS/UE role topology from a label JSON file."""

    label_path = Path(label_file)
    data = json.loads(label_path.read_text(encoding="utf-8"))
    bs_source_points, ue_source_points = _standard_label_points(data, label_path)
    bs_points, selected_bs_indices = _select_points_with_indices(
        bs_source_points,
        max_count=max_bs,
        indices=bs_indices,
        label="BS",
    )
    ue_points, selected_ue_indices = _select_points_with_indices(
        ue_source_points,
        max_count=max_ue,
        start=ue_start,
        count=ue_count,
        indices=ue_indices,
        label="UE",
    )

    if not bs_points or not ue_points:
        msg = f"Label file must contain at least one BS and UE point: {label_path}"
        raise ValueError(msg)

    return RoleTopology(
        bs_positions_m=_points_to_positions(bs_points),
        ue_positions_m=_points_to_positions(ue_points),
        bs_labels=tuple(str(point.get("label", f"BS{i}")) for i, point in enumerate(bs_points)),
        ue_labels=tuple(str(point.get("label", f"UE{i}")) for i, point in enumerate(ue_points)),
        bs_global_indices=np.asarray(selected_bs_indices, dtype=np.int64),
        ue_global_indices=np.asarray(selected_ue_indices, dtype=np.int64),
    )


def count_topology_points(label_file: str | Path) -> tuple[int, int]:
    """Return available ``(tx_count, rx_count)`` from a label JSON file."""

    label_path = Path(label_file)
    data = json.loads(label_path.read_text(encoding="utf-8"))
    tx_points, rx_points = _standard_label_points(data, label_path)
    return len(tx_points), len(rx_points)


def _standard_label_points(
    data: dict[str, Any],
    label_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return full-scene BS/UE point lists from the standard label format."""

    bs_points = data.get("bs_points")
    ue_points = data.get("ue_points")
    if not isinstance(bs_points, list) or not isinstance(ue_points, list):
        msg = (
            "Standard label JSON must contain top-level bs_points and ue_points lists "
            f"({label_path}). groups are metadata/subsets and are not used for default topology."
        )
        raise ValueError(msg)
    return _ensure_point_mappings(bs_points, "BS"), _ensure_point_mappings(ue_points, "UE")


def _ensure_point_mappings(points: list[Any], label: str) -> list[dict[str, Any]]:
    for index, point in enumerate(points):
        if not isinstance(point, dict):
            msg = f"{label} point {index} must be a mapping"
            raise ValueError(msg)
    return points


def _select_points(
    points: Any,
    *,
    max_count: int,
    label: str,
    start: int = 0,
    count: int | None = None,
    indices: list[int] | tuple[int, ...] | None = None,
) -> list[dict[str, Any]]:
    selected, _ = _select_points_with_indices(
        points,
        max_count=max_count,
        label=label,
        start=start,
        count=count,
        indices=indices,
    )
    return selected


def _select_points_with_indices(
    points: Any,
    *,
    max_count: int,
    label: str,
    start: int = 0,
    count: int | None = None,
    indices: list[int] | tuple[int, ...] | None = None,
) -> tuple[list[dict[str, Any]], tuple[int, ...]]:
    if not isinstance(points, list):
        msg = f"Label {label} points must be a list"
        raise ValueError(msg)
    if max_count < 1:
        msg = f"max {label} count must be positive"
        raise ValueError(msg)

    if indices is not None:
        selected_indices = tuple(int(index) for index in indices)
        if not selected_indices:
            msg = f"{label} indices must not be empty"
            raise ValueError(msg)
        _validate_indices(selected_indices, len(points), label)
        return [points[index] for index in selected_indices], selected_indices

    if start < 0:
        msg = f"{label} start must be non-negative"
        raise ValueError(msg)
    selected_count = max_count if count is None else count
    if selected_count < 1:
        msg = f"{label} count must be positive"
        raise ValueError(msg)
    end = start + selected_count
    if start >= len(points):
        msg = f"{label} start {start} exceeds available point count {len(points)}"
        raise ValueError(msg)
    if count is None:
        end = min(end, len(points))
    elif end > len(points):
        msg = f"{label} range [{start}, {end}) exceeds available point count {len(points)}"
        raise ValueError(msg)
    selected_indices = tuple(range(start, end))
    return points[start:end], selected_indices


def _validate_indices(indices: tuple[int, ...], point_count: int, label: str) -> None:
    for index in indices:
        if index < 0 or index >= point_count:
            msg = f"{label} index {index} is outside available point count {point_count}"
            raise ValueError(msg)


def _points_to_positions(points: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([_point_to_position(point) for point in points], dtype=np.float32)


def _point_to_position(point: dict[str, Any]) -> tuple[float, float, float]:
    position = point.get("position")
    if isinstance(position, (list, tuple)):
        if len(position) != 3:
            msg = "Label point position must contain exactly three coordinates"
            raise ValueError(msg)
        return (float(position[0]), float(position[1]), float(position[2]))

    try:
        return (float(point["x"]), float(point["y"]), float(point["z"]))
    except KeyError as exc:
        msg = "Label point must define either position=[x, y, z] or explicit x/y/z fields"
        raise ValueError(msg) from exc
