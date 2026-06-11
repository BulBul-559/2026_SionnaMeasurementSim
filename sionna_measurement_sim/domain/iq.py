"""Domain models for protocol-independent IQ observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sionna_measurement_sim.domain.validation import require_shape


@dataclass(frozen=True)
class LinkIQCapture:
    """Optional per-link IQ exported from a cooperative PHY run."""

    frequency_clean: np.ndarray | None = None
    frequency_observed: np.ndarray | None = None
    time_clean: np.ndarray | None = None
    time_observed: np.ndarray | None = None

    def __post_init__(self) -> None:
        reference_frequency_shape: tuple[int, ...] | None = None
        for name in ("frequency_clean", "frequency_observed"):
            value = getattr(self, name)
            if value is None:
                continue
            array = np.asarray(value, dtype=np.complex64)
            require_shape(name, array, (None, None, None, None, None, None))
            object.__setattr__(self, name, array)
            if reference_frequency_shape is None:
                reference_frequency_shape = array.shape
            elif array.shape != reference_frequency_shape:
                raise ValueError("link IQ frequency captures must share one shape")

        reference_time_shape: tuple[int, ...] | None = None
        for name in ("time_clean", "time_observed"):
            value = getattr(self, name)
            if value is None:
                continue
            array = np.asarray(value, dtype=np.complex64)
            require_shape(name, array, (None, None, None, None, None))
            object.__setattr__(self, name, array)
            if reference_time_shape is None:
                reference_time_shape = array.shape
            elif array.shape != reference_time_shape:
                raise ValueError("link IQ time captures must share one shape")

        if (
            reference_frequency_shape is not None
            and reference_time_shape is not None
            and reference_frequency_shape[:4] != reference_time_shape[:4]
        ):
            raise ValueError("link IQ frequency/time leading dimensions must match")

    @property
    def is_empty(self) -> bool:
        return all(
            getattr(self, name) is None
            for name in (
                "frequency_clean",
                "frequency_observed",
                "time_clean",
                "time_observed",
            )
        )


@dataclass(frozen=True)
class NonCooperativeIQCapture:
    """Time-domain shared IQ frames for non-cooperative experiments."""

    rx_time_clean: np.ndarray | None
    rx_time_observed: np.ndarray | None
    active_tx_indices: np.ndarray
    active_tx_global_indices: np.ndarray
    active_tx_mask: np.ndarray
    active_tx_positions_m: np.ndarray
    noise_variance: np.ndarray
    snr_db: np.ndarray
    rssi_dbm: np.ndarray
    noise_power_dbm: np.ndarray
    resource_occupancy_count: np.ndarray
    resource_collision_mask: np.ndarray

    def __post_init__(self) -> None:
        reference_shape: tuple[int, ...] | None = None
        for name in ("rx_time_clean", "rx_time_observed"):
            value = getattr(self, name)
            if value is None:
                continue
            array = np.asarray(value, dtype=np.complex64)
            require_shape(name, array, (None, None, None, None, None))
            object.__setattr__(self, name, array)
            if reference_shape is None:
                reference_shape = array.shape
            elif array.shape != reference_shape:
                raise ValueError("noncooperative clean/observed time IQ shapes must match")
        if reference_shape is None:
            raise ValueError("noncooperative IQ requires clean or observed time IQ")

        snapshot, frame, rx, _rx_ant, _sample = reference_shape
        for name in ("noise_variance", "snr_db", "rssi_dbm", "noise_power_dbm"):
            array = np.asarray(getattr(self, name), dtype=np.float32)
            require_shape(name, array, (snapshot, frame, rx))
            object.__setattr__(self, name, array)

        active_tx_indices = np.asarray(self.active_tx_indices, dtype=np.int32)
        require_shape("active_tx_indices", active_tx_indices, (frame, None))
        object.__setattr__(self, "active_tx_indices", active_tx_indices)
        active_shape = active_tx_indices.shape

        for name in ("active_tx_global_indices",):
            array = np.asarray(getattr(self, name), dtype=np.int64)
            require_shape(name, array, active_shape)
            object.__setattr__(self, name, array)
        active_mask = np.asarray(self.active_tx_mask, dtype=np.bool_)
        require_shape("active_tx_mask", active_mask, active_shape)
        object.__setattr__(self, "active_tx_mask", active_mask)

        positions = np.asarray(self.active_tx_positions_m, dtype=np.float32)
        require_shape("active_tx_positions_m", positions, (*active_shape, 3))
        object.__setattr__(self, "active_tx_positions_m", positions)

        occupancy = np.asarray(self.resource_occupancy_count, dtype=np.int32)
        if occupancy.ndim != 3 or occupancy.shape[0] != frame:
            raise ValueError(
                "resource_occupancy_count must be [frame,ofdm_symbol,subcarrier]"
            )
        object.__setattr__(self, "resource_occupancy_count", occupancy)
        collision = np.asarray(self.resource_collision_mask, dtype=np.bool_)
        require_shape("resource_collision_mask", collision, occupancy.shape)
        object.__setattr__(self, "resource_collision_mask", collision)


@dataclass(frozen=True)
class IQObservationResult:
    """Protocol-independent IQ observation payload written under `/iq`."""

    sample_rate_hz: float
    fft_size: int
    cp_length: int
    num_ofdm_symbols: int
    time_domain_convention: str
    link: LinkIQCapture | None = None
    noncooperative: NonCooperativeIQCapture | None = None

    def __post_init__(self) -> None:
        if self.sample_rate_hz <= 0:
            raise ValueError("IQ sample_rate_hz must be positive")
        if self.fft_size < 2:
            raise ValueError("IQ fft_size must be >= 2")
        if self.cp_length < 0:
            raise ValueError("IQ cp_length must be non-negative")
        if self.num_ofdm_symbols < 1:
            raise ValueError("IQ num_ofdm_symbols must be positive")
        if self.link is not None and self.link.is_empty:
            object.__setattr__(self, "link", None)
        if self.link is None and self.noncooperative is None:
            raise ValueError("IQObservationResult requires link or noncooperative data")
