"""Channel truth domain model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sionna_measurement_sim.domain.validation import require_finite, require_shape


@dataclass(frozen=True)
class RTTruthResult:
    """Ray-tracing truth channel response in TX-first order."""

    cfr: np.ndarray
    path_power_db: np.ndarray
    has_geometric_signal: np.ndarray

    def __post_init__(self) -> None:
        cfr = np.asarray(self.cfr, dtype=np.complex64)
        path_power_db = np.asarray(self.path_power_db, dtype=np.float32)
        has_signal = np.asarray(self.has_geometric_signal, dtype=np.bool_)

        if cfr.ndim not in (5, 6):
            msg = f"cfr must have rank 5 or 6, got {cfr.shape}"
            raise ValueError(msg)
        tx_dim = 0 if cfr.ndim == 5 else 1
        rx_dim = 1 if cfr.ndim == 5 else 2
        tx, rx = cfr.shape[tx_dim], cfr.shape[rx_dim]
        if path_power_db.ndim == 2:
            require_shape("path_power_db", path_power_db, (tx, rx))
            require_shape("has_geometric_signal", has_signal, (tx, rx))
        elif path_power_db.ndim == 3:
            require_shape("path_power_db", path_power_db, (None, tx, rx))
            require_shape("has_geometric_signal", has_signal, (None, tx, rx))
        else:
            msg = f"path_power_db must have rank 2 or 3, got {path_power_db.shape}"
            raise ValueError(msg)
        require_finite("cfr", cfr)
        require_finite("path_power_db", path_power_db)

        object.__setattr__(self, "cfr", cfr)
        object.__setattr__(self, "path_power_db", path_power_db)
        object.__setattr__(self, "has_geometric_signal", has_signal)
