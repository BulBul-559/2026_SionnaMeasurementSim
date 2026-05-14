"""Label/topology parsing helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from sionna_measurement_sim.domain.topology import Topology


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

    label_path = Path(label_file)
    data = json.loads(label_path.read_text(encoding="utf-8"))
    group = _select_group(data)
    tx_points = _select_points(
        group.get("bs_points", []),
        max_count=max_tx,
        indices=tx_indices,
        label="BS",
    )
    rx_points = _select_points(
        group.get("ue_points", []),
        max_count=max_rx,
        start=rx_start,
        count=rx_count,
        indices=rx_indices,
        label="UE",
    )

    if not tx_points or not rx_points:
        msg = f"Label file must contain at least one BS and UE point: {label_path}"
        raise ValueError(msg)

    return Topology(
        tx_positions_m=_points_to_positions(tx_points),
        rx_positions_m=_points_to_positions(rx_points),
        tx_labels=tuple(str(point.get("label", f"tx{i}")) for i, point in enumerate(tx_points)),
        rx_labels=tuple(str(point.get("label", f"rx{i}")) for i, point in enumerate(rx_points)),
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
        return [points[index] for index in selected_indices]

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
    return points[start:end]


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
