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
) -> Topology:
    """Load a small TX/RX topology from the prepared test label JSON."""

    label_path = Path(label_file)
    data = json.loads(label_path.read_text(encoding="utf-8"))
    group = _select_group(data)
    tx_points = group.get("bs_points", [])[:max_tx]
    rx_points = group.get("ue_points", [])[:max_rx]

    if not tx_points or not rx_points:
        msg = f"Label file must contain at least one BS and UE point: {label_path}"
        raise ValueError(msg)

    return Topology(
        tx_positions_m=_points_to_positions(tx_points),
        rx_positions_m=_points_to_positions(rx_points),
        tx_labels=tuple(str(point.get("label", f"tx{i}")) for i, point in enumerate(tx_points)),
        rx_labels=tuple(str(point.get("label", f"rx{i}")) for i, point in enumerate(rx_points)),
    )


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


def _points_to_positions(points: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray(
        [[float(point["x"]), float(point["y"]), float(point["z"])] for point in points],
        dtype=np.float32,
    )
