"""CIR (Channel Impulse Response) truth domain model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sionna_measurement_sim.domain.validation import require_finite, require_shape


@dataclass(frozen=True)
class CIRTruth:
    """Channel Impulse Response truth data.

    All arrays have shape [snapshot, tx, rx, rx_ant, tx_ant, path].
    """

    coefficients: np.ndarray  # complex64 [snapshot, tx, rx, rx_ant, tx_ant, path]
    delays_s: np.ndarray  # float32 [snapshot, tx, rx, rx_ant, tx_ant, path]
    valid: np.ndarray  # bool [snapshot, tx, rx, rx_ant, tx_ant, path]

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.coefficients, dtype=np.complex64)
        delays_s = np.asarray(self.delays_s, dtype=np.float32)
        valid = np.asarray(self.valid, dtype=np.bool_)

        expected_shape = coefficients.shape
        require_shape("delays_s", delays_s, expected_shape)
        require_shape("valid", valid, expected_shape)
        require_shape(
            "coefficients",
            coefficients,
            (None, None, None, None, None, None),
        )

        delays_s = np.maximum(delays_s, 0.0)
        require_finite("coefficients", coefficients)

        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "delays_s", delays_s)
        object.__setattr__(self, "valid", valid)

    @classmethod
    def empty(
        cls,
        num_snapshots: int = 1,
        num_tx: int = 1,
        num_rx: int = 1,
        num_rx_ant: int = 1,
        num_tx_ant: int = 1,
    ) -> CIRTruth:
        """Create an empty CIR truth with zero paths."""
        shape = (num_snapshots, num_tx, num_rx, num_rx_ant, num_tx_ant, 0)
        return cls(
            coefficients=np.zeros(shape, dtype=np.complex64),
            delays_s=np.zeros(shape, dtype=np.float32),
            valid=np.zeros(shape, dtype=np.bool_),
        )
