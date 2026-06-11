"""Domain models for multi-user PHY observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sionna_measurement_sim.domain.validation import require_shape


@dataclass(frozen=True)
class MultiUserSRSResult:
    """Shared NR SRS observation frame with per-active-UE estimates.

    The result represents a static-channel SRS frame where several uplink TX
    devices transmit orthogonal SRS resources into the same received grid.
    """

    resource_strategy: str
    rx_grid_clean_shared: np.ndarray
    rx_grid_shared: np.ndarray
    noise_variance: np.ndarray
    snr_db: np.ndarray
    rssi_dbm: np.ndarray
    noise_power_dbm: np.ndarray
    active_tx_indices: np.ndarray
    active_tx_mask: np.ndarray
    comb_offset: np.ndarray
    prb_start: np.ndarray
    prb_count: np.ndarray
    re_symbol_indices: np.ndarray
    re_subcarrier_indices: np.ndarray
    re_mask: np.ndarray
    allocated_subcarrier_indices: np.ndarray
    allocated_subcarrier_mask: np.ndarray
    resource_occupancy_count: np.ndarray
    resource_collision_mask: np.ndarray
    cfr_est_resource: np.ndarray
    cfr_est_allocated: np.ndarray

    def __post_init__(self) -> None:
        rx_grid_clean = np.asarray(self.rx_grid_clean_shared, dtype=np.complex64)
        if rx_grid_clean.ndim != 6:
            msg = (
                "multiuser rx_grid_clean_shared must be "
                "[snapshot,frame,rx,rx_ant,symbol,subcarrier]"
            )
            raise ValueError(msg)
        object.__setattr__(self, "rx_grid_clean_shared", rx_grid_clean)

        rx_grid = np.asarray(self.rx_grid_shared, dtype=np.complex64)
        if rx_grid.ndim != 6:
            msg = "multiuser rx_grid_shared must be [snapshot,frame,rx,rx_ant,symbol,subcarrier]"
            raise ValueError(msg)
        if rx_grid.shape != rx_grid_clean.shape:
            msg = "multiuser clean and observed shared RX grids must have matching shape"
            raise ValueError(msg)
        snap, frame, rx, rx_ant, symbols, subcarriers = rx_grid.shape
        object.__setattr__(self, "rx_grid_shared", rx_grid)

        for name in ("noise_variance", "snr_db", "rssi_dbm", "noise_power_dbm"):
            values = np.asarray(getattr(self, name), dtype=np.float32)
            require_shape(name, values, (snap, frame, rx))
            object.__setattr__(self, name, values)

        active_tx_indices = np.asarray(self.active_tx_indices, dtype=np.int32)
        active_tx_mask = np.asarray(self.active_tx_mask, dtype=np.bool_)
        if active_tx_indices.ndim != 2:
            msg = "multiuser active_tx_indices must be [frame,active_ue]"
            raise ValueError(msg)
        require_shape("active_tx_mask", active_tx_mask, active_tx_indices.shape)
        if active_tx_indices.shape[0] != frame:
            msg = "multiuser active_tx_indices frame dimension must match rx_grid_shared"
            raise ValueError(msg)
        object.__setattr__(self, "active_tx_indices", active_tx_indices)
        object.__setattr__(self, "active_tx_mask", active_tx_mask)
        active_ue = active_tx_indices.shape[1]

        for name in ("comb_offset",):
            values = np.asarray(getattr(self, name), dtype=np.int32)
            require_shape(name, values, (frame, active_ue))
            object.__setattr__(self, name, values)
        for name in ("prb_start", "prb_count"):
            values = np.asarray(getattr(self, name), dtype=np.int32)
            if values.ndim != 3 or values.shape[:2] != (frame, active_ue):
                msg = f"multiuser {name} must be [frame,active_ue,srs_symbol]"
                raise ValueError(msg)
            object.__setattr__(self, name, values)

        re_symbol_indices = np.asarray(self.re_symbol_indices, dtype=np.int32)
        re_subcarrier_indices = np.asarray(self.re_subcarrier_indices, dtype=np.int32)
        re_mask = np.asarray(self.re_mask, dtype=np.bool_)
        if (
            re_symbol_indices.shape != re_subcarrier_indices.shape
            or re_mask.shape != re_symbol_indices.shape
        ):
            msg = "multiuser RE indices and mask must have matching shape"
            raise ValueError(msg)
        if re_symbol_indices.ndim != 3 or re_symbol_indices.shape[:2] != (frame, active_ue):
            msg = "multiuser RE indices must be [frame,active_ue,max_srs_re]"
            raise ValueError(msg)
        object.__setattr__(self, "re_symbol_indices", re_symbol_indices)
        object.__setattr__(self, "re_subcarrier_indices", re_subcarrier_indices)
        object.__setattr__(self, "re_mask", re_mask)
        max_re = re_symbol_indices.shape[2]

        allocated_indices = np.asarray(self.allocated_subcarrier_indices, dtype=np.int32)
        allocated_mask = np.asarray(self.allocated_subcarrier_mask, dtype=np.bool_)
        if allocated_indices.shape != allocated_mask.shape:
            msg = "multiuser allocated subcarrier indices and mask must match"
            raise ValueError(msg)
        if allocated_indices.ndim != 3 or allocated_indices.shape[:2] != (frame, active_ue):
            msg = "multiuser allocated subcarrier indices must be [frame,active_ue,max_alloc_sc]"
            raise ValueError(msg)
        object.__setattr__(self, "allocated_subcarrier_indices", allocated_indices)
        object.__setattr__(self, "allocated_subcarrier_mask", allocated_mask)
        max_alloc_sc = allocated_indices.shape[2]

        occupancy = np.asarray(self.resource_occupancy_count, dtype=np.int32)
        collisions = np.asarray(self.resource_collision_mask, dtype=np.bool_)
        require_shape("resource_occupancy_count", occupancy, (frame, symbols, subcarriers))
        require_shape("resource_collision_mask", collisions, occupancy.shape)
        object.__setattr__(self, "resource_occupancy_count", occupancy)
        object.__setattr__(self, "resource_collision_mask", collisions)

        cfr_resource = np.asarray(self.cfr_est_resource, dtype=np.complex64)
        if cfr_resource.ndim != 7 or cfr_resource.shape[:5] != (snap, frame, active_ue, rx, rx_ant):
            msg = (
                "multiuser cfr_est_resource must be "
                "[snapshot,frame,active_ue,rx,rx_ant,srs_port,max_srs_re]"
            )
            raise ValueError(msg)
        if cfr_resource.shape[-1] != max_re:
            msg = "multiuser cfr_est_resource last dimension must match max_srs_re"
            raise ValueError(msg)
        object.__setattr__(self, "cfr_est_resource", cfr_resource)

        cfr_allocated = np.asarray(self.cfr_est_allocated, dtype=np.complex64)
        if (
            cfr_allocated.ndim != 7
            or cfr_allocated.shape[:5] != (snap, frame, active_ue, rx, rx_ant)
        ):
            msg = (
                "multiuser cfr_est_allocated must be "
                "[snapshot,frame,active_ue,rx,rx_ant,tx_ant,max_alloc_sc]"
            )
            raise ValueError(msg)
        if cfr_allocated.shape[-1] != max_alloc_sc:
            msg = "multiuser cfr_est_allocated last dimension must match max_alloc_sc"
            raise ValueError(msg)
        object.__setattr__(self, "cfr_est_allocated", cfr_allocated)
