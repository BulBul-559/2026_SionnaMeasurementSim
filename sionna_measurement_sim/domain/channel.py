"""Channel truth domain model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sionna_measurement_sim.domain.validation import require_finite, require_shape


@dataclass(frozen=True)
class RTTruthResult:
    """Ray-tracing truth channel response in TX-first order.

    cfr is always 5D [tx, rx, rx_ant, tx_ant, subcarrier].
    cfr_snapshots is 6D [snapshot, tx, rx, rx_ant, tx_ant, subcarrier] for
    multi-time-step Doppler synthetic runs (None otherwise).
    """

    cfr: np.ndarray
    path_power_db: np.ndarray
    has_geometric_signal: np.ndarray
    geometric_path_count: np.ndarray
    los_exists: np.ndarray
    nlos_exists: np.ndarray
    cfr_snapshots: np.ndarray | None = None

    def __post_init__(self) -> None:
        cfr = np.asarray(self.cfr, dtype=np.complex64)
        path_power_db = np.asarray(self.path_power_db, dtype=np.float32)
        has_signal = np.asarray(self.has_geometric_signal, dtype=np.bool_)
        geometric_path_count = np.asarray(self.geometric_path_count, dtype=np.int32)
        los_exists = np.asarray(self.los_exists, dtype=np.bool_)
        nlos_exists = np.asarray(self.nlos_exists, dtype=np.bool_)

        require_shape("cfr", cfr, (None, None, None, None, None))
        tx, rx = cfr.shape[0], cfr.shape[1]
        require_shape("path_power_db", path_power_db, (tx, rx))
        require_shape("has_geometric_signal", has_signal, (tx, rx))
        for name, arr in (
            ("geometric_path_count", geometric_path_count),
            ("los_exists", los_exists),
            ("nlos_exists", nlos_exists),
        ):
            require_shape(name, arr, (tx, rx))
        require_finite("cfr", cfr)
        require_finite("path_power_db", path_power_db)

        if self.cfr_snapshots is not None:
            ss = np.asarray(self.cfr_snapshots, dtype=np.complex64)
            if ss.ndim != 6:
                msg = f"cfr_snapshots must be 6D, got {ss.shape}"
                raise ValueError(msg)
            if ss.shape[1:] != cfr.shape:
                msg = "cfr_snapshots.shape[1:] must match cfr.shape"
                raise ValueError(msg)
            require_finite("cfr_snapshots", ss)
            object.__setattr__(self, "cfr_snapshots", ss)
        else:
            object.__setattr__(self, "cfr_snapshots", None)

        object.__setattr__(self, "cfr", cfr)
        object.__setattr__(self, "path_power_db", path_power_db)
        object.__setattr__(self, "has_geometric_signal", has_signal)
        object.__setattr__(self, "geometric_path_count", geometric_path_count)
        object.__setattr__(self, "los_exists", los_exists)
        object.__setattr__(self, "nlos_exists", nlos_exists)
