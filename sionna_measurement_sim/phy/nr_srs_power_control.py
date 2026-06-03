"""Legacy NR SRS power-control adapter.

The power/RSSI implementation now lives in :mod:`sionna_measurement_sim.phy.power`.
This module remains as a thin compatibility wrapper for older tests/imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sionna_measurement_sim.phy.power import compute_uplink_power


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
    """Compatibility wrapper around the common uplink power model."""

    result = compute_uplink_power(
        path_power_db=path_power_db,
        snapshot_count=snapshot_count,
        tx_count=tx_count,
        rx_count=rx_count,
        port_count=num_srs_ports,
        fixed_tx_power_dbm=base_tx_power_dbm,
        power_config=None,
        legacy_power_control=config,
    )
    return SRSPowerControlResult(
        tx_power_dbm=result.tx_power_dbm,
        power_scale_linear=result.power_scale_linear,
        serving_rx_index=result.serving_rx_index,
        path_loss_db=result.path_loss_db,
    )
