"""Topology domain model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sionna_measurement_sim.domain.validation import require_finite, require_shape


@dataclass(frozen=True)
class Topology:
    """TX/RX positions and labels in SI units."""

    tx_positions_m: np.ndarray
    rx_positions_m: np.ndarray
    tx_labels: tuple[str, ...]
    rx_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        tx_positions = np.asarray(self.tx_positions_m, dtype=np.float32)
        rx_positions = np.asarray(self.rx_positions_m, dtype=np.float32)

        require_shape("tx_positions_m", tx_positions, (None, 3))
        require_shape("rx_positions_m", rx_positions, (None, 3))
        require_finite("tx_positions_m", tx_positions)
        require_finite("rx_positions_m", rx_positions)

        if len(self.tx_labels) != tx_positions.shape[0]:
            msg = "tx_labels length must match tx_positions_m"
            raise ValueError(msg)
        if len(self.rx_labels) != rx_positions.shape[0]:
            msg = "rx_labels length must match rx_positions_m"
            raise ValueError(msg)

        object.__setattr__(self, "tx_positions_m", tx_positions)
        object.__setattr__(self, "rx_positions_m", rx_positions)
        object.__setattr__(self, "tx_labels", tuple(self.tx_labels))
        object.__setattr__(self, "rx_labels", tuple(self.rx_labels))

    @property
    def num_tx(self) -> int:
        return int(self.tx_positions_m.shape[0])

    @property
    def num_rx(self) -> int:
        return int(self.rx_positions_m.shape[0])
