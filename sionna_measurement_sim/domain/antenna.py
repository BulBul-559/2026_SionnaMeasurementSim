"""Antenna domain model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sionna_measurement_sim.domain.validation import require_shape

_VALID_ORIENTATION_MODES = frozenset({"fixed", "look_at_first_peer", "look_at_centroid"})
_VALID_PATTERNS = frozenset({"iso", "tr38901"})
_VALID_POLARIZATIONS = frozenset({"V", "H", "cross", "single"})


@dataclass(frozen=True)
class AntennaSpec:
    """Minimal antenna array description required by the HDF5 contract."""

    tx_array_type: str = "single"
    rx_array_type: str = "single"
    tx_num_rows: int = 1
    tx_num_cols: int = 1
    rx_num_rows: int = 1
    rx_num_cols: int = 1
    tx_spacing_lambda: np.ndarray | tuple[float, float] = (0.5, 0.5)
    rx_spacing_lambda: np.ndarray | tuple[float, float] = (0.5, 0.5)
    tx_polarization: str = "single"
    rx_polarization: str = "single"
    tx_pattern: str = "iso"
    rx_pattern: str = "iso"
    synthetic_array: bool = False
    tx_orientation_mode: str = "fixed"
    tx_orientation_rad: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0)
    rx_orientation_mode: str = "fixed"
    rx_orientation_rad: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        tx_spacing = np.asarray(self.tx_spacing_lambda, dtype=np.float32)
        rx_spacing = np.asarray(self.rx_spacing_lambda, dtype=np.float32)
        require_shape("tx_spacing_lambda", tx_spacing, (2,))
        require_shape("rx_spacing_lambda", rx_spacing, (2,))

        for name in ("tx_num_rows", "tx_num_cols", "rx_num_rows", "rx_num_cols"):
            if getattr(self, name) <= 0:
                msg = f"{name} must be positive"
                raise ValueError(msg)

        tx_orient = np.asarray(self.tx_orientation_rad, dtype=np.float64)
        rx_orient = np.asarray(self.rx_orientation_rad, dtype=np.float64)
        require_shape("tx_orientation_rad", tx_orient, (3,))
        require_shape("rx_orientation_rad", rx_orient, (3,))
        if not np.all(np.isfinite(tx_orient)):
            msg = "tx_orientation_rad must contain all finite values"
            raise ValueError(msg)
        if not np.all(np.isfinite(rx_orient)):
            msg = "rx_orientation_rad must contain all finite values"
            raise ValueError(msg)

        for name, val in (
            ("tx_orientation_mode", self.tx_orientation_mode),
            ("rx_orientation_mode", self.rx_orientation_mode),
        ):
            if val not in _VALID_ORIENTATION_MODES:
                msg = f"{name} must be one of {_VALID_ORIENTATION_MODES}, got {val!r}"
                raise ValueError(msg)

        # RX orientation must be "fixed" — the RT scene does not apply
        # look_at modes for the receiver, making them silently incorrect.
        if self.rx_orientation_mode != "fixed":
            msg = (
                f"rx_orientation_mode must be 'fixed', "
                f"got {self.rx_orientation_mode!r}"
            )
            raise ValueError(msg)

        for name, val in (
            ("tx_pattern", self.tx_pattern),
            ("rx_pattern", self.rx_pattern),
        ):
            if val not in _VALID_PATTERNS:
                msg = f"{name} must be one of {_VALID_PATTERNS}, got {val!r}"
                raise ValueError(msg)

        for name, val in (
            ("tx_polarization", self.tx_polarization),
            ("rx_polarization", self.rx_polarization),
        ):
            if val not in _VALID_POLARIZATIONS:
                msg = f"{name} must be one of {_VALID_POLARIZATIONS}, got {val!r}"
                raise ValueError(msg)

        object.__setattr__(self, "tx_spacing_lambda", tx_spacing)
        object.__setattr__(self, "rx_spacing_lambda", rx_spacing)
        object.__setattr__(self, "tx_orientation_rad", tx_orient)
        object.__setattr__(self, "rx_orientation_rad", rx_orient)

    @property
    def tx_num_ant(self) -> int:
        return int(self.tx_num_rows * self.tx_num_cols)

    @property
    def rx_num_ant(self) -> int:
        return int(self.rx_num_rows * self.rx_num_cols)
