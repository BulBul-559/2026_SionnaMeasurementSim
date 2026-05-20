"""NR SRS power-control helpers for the standards-shaped v2 subset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SRSPowerControlResult:
    tx_power_dbm: np.ndarray
    power_scale_linear: np.ndarray
    serving_rx_index: np.ndarray
    path_loss_db: np.ndarray


def compute_srs_power_control(
    *,
    path_power_db: np.ndarray | None,
    snapshot_count: int,
    tx_count: int,
    rx_count: int,
    num_srs_ports: int,
    base_tx_power_dbm: float,
    config: Any,
) -> SRSPowerControlResult:
    """Compute per-UE/per-port SRS transmit power metadata and scale.

    The model is intentionally simple: it scales the transmitted SRS grid by a
    pathloss-compensated open-loop power value.  The common AWGN chain remains
    relative-SNR based; this helper does not implement absolute thermal noise.
    """

    enabled = bool(getattr(config, "enabled", False))
    base_power = np.full(
        (snapshot_count, tx_count, num_srs_ports),
        float(base_tx_power_dbm),
        dtype=np.float32,
    )
    serving = np.zeros((snapshot_count, tx_count), dtype=np.int32)
    path_loss = np.zeros((snapshot_count, tx_count), dtype=np.float32)
    if not enabled:
        return SRSPowerControlResult(
            tx_power_dbm=base_power,
            power_scale_linear=np.ones_like(base_power, dtype=np.float32),
            serving_rx_index=serving,
            path_loss_db=path_loss,
        )

    if path_power_db is None:
        raise ValueError("SRS power_control.enabled=true requires RT path_power_db")
    power = np.asarray(path_power_db, dtype=np.float32)
    if power.shape != (tx_count, rx_count):
        raise ValueError(f"path_power_db must have shape {(tx_count, rx_count)}, got {power.shape}")

    policy = str(getattr(config, "serving_rx_policy", "strongest_path"))
    if policy == "first_rx":
        serving_1d = np.zeros((tx_count,), dtype=np.int32)
    elif policy == "strongest_path":
        serving_1d = np.argmax(power, axis=1).astype(np.int32)
    else:
        raise ValueError("SRS serving_rx_policy must be strongest_path/first_rx")

    tx_indices = np.arange(tx_count, dtype=np.int32)
    path_loss_1d = -power[tx_indices, serving_1d]
    requested = (
        float(getattr(config, "p0_dbm", 0.0))
        + float(getattr(config, "alpha", 0.8)) * path_loss_1d
    )
    clipped = np.clip(
        requested,
        float(getattr(config, "min_tx_power_dbm", -40.0)),
        float(getattr(config, "max_tx_power_dbm", 23.0)),
    ).astype(np.float32)
    tx_power = np.broadcast_to(
        clipped[np.newaxis, :, np.newaxis],
        (snapshot_count, tx_count, num_srs_ports),
    ).copy()
    scale = (10.0 ** ((tx_power - float(base_tx_power_dbm)) / 20.0)).astype(np.float32)
    return SRSPowerControlResult(
        tx_power_dbm=tx_power,
        power_scale_linear=scale,
        serving_rx_index=np.broadcast_to(
            serving_1d[np.newaxis, :],
            (snapshot_count, tx_count),
        ).copy(),
        path_loss_db=np.broadcast_to(
            path_loss_1d[np.newaxis, :],
            (snapshot_count, tx_count),
        ).copy(),
    )
