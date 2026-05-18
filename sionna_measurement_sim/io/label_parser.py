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


def load_topology_from_label(
    label_file: str | Path,
    *,
    max_tx: int = 1,
    max_rx: int = 1,
    rx_start: int = 0,
    rx_count: int | None = None,
    rx_indices: list[int] | tuple[int, ...] | None = None,
    tx_indices: list[int] | tuple[int, ...] | None = None,
) -> Topology:
    """Load a small TX/RX topology from the prepared test label JSON."""

    role_topology = load_role_topology_from_label(
        label_file,
        max_bs=max_tx,
        max_ue=max_rx,
        ue_start=rx_start,
        ue_count=rx_count,
        ue_indices=rx_indices,
        bs_indices=tx_indices,
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
    group = _select_group(data)
    bs_points, selected_bs_indices = _select_points_with_indices(
        group.get("bs_points", []),
        max_count=max_bs,
        indices=bs_indices,
        label="BS",
    )
    ue_points, selected_ue_indices = _select_points_with_indices(
        group.get("ue_points", []),
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
    group = _select_group(data)
    tx_points = group.get("bs_points", [])
    rx_points = group.get("ue_points", [])
    if not isinstance(tx_points, list) or not isinstance(rx_points, list):
        msg = "Label group bs_points and ue_points must be lists"
        raise ValueError(msg)
    return len(tx_points), len(rx_points)


def _select_group(data: dict[str, Any]) -> dict[str, Any]:
    groups = data.get("groups")
    if not isinstance(groups, list) or not groups:
        msg = "Label JSON must contain a non-empty groups list"
        raise ValueError(msg)
    group = groups[0]
    if not isinstance(group, dict):
        msg = "Label group must be a mapping"
        raise ValueError(msg)
    return group


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
        msg = f"Label group {label} points must be a list"
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
    return np.asarray(
        [[float(point["x"]), float(point["y"]), float(point["z"])] for point in points],
        dtype=np.float32,
    )
