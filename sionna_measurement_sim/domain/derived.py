"""Derived link labels computed from topology and RT path truth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sionna_measurement_sim.domain.channel import RTLinkSummary, RTTruthResult
from sionna_measurement_sim.domain.cir import CIRTruth
from sionna_measurement_sim.domain.link import LinkConfig
from sionna_measurement_sim.domain.path import PathTable
from sionna_measurement_sim.domain.topology import Topology
from sionna_measurement_sim.domain.validation import require_shape

SPEED_OF_LIGHT_MPS = 299_792_458.0
PATH_SELECTION_POLICY = (
    "los=min_delay_los;first=min_delay;strongest=max_abs_a2;"
    "antenna_aggregation=global_candidate"
)


@dataclass(frozen=True)
class DerivedLabels:
    """Per-link labels derived from topology and all available path candidates."""

    geometric_distance_m: np.ndarray
    los_distance_m: np.ndarray
    first_path_delay_s: np.ndarray
    first_path_propagation_range_m: np.ndarray
    strongest_path_delay_s: np.ndarray
    los_aoa_azimuth_rad: np.ndarray
    los_aoa_zenith_rad: np.ndarray
    strongest_aoa_azimuth_rad: np.ndarray
    strongest_aoa_zenith_rad: np.ndarray
    first_path_aoa_azimuth_rad: np.ndarray
    first_path_aoa_zenith_rad: np.ndarray
    los_flag: np.ndarray
    nlos_flag: np.ndarray
    path_count: np.ndarray
    path_power_db: np.ndarray
    link_valid_mask: np.ndarray
    tx_rx_midpoint_m: np.ndarray
    tx_rx_bearing_rad: np.ndarray
    tx_rx_distance_m: np.ndarray
    path_selection_policy: str = PATH_SELECTION_POLICY

    def __post_init__(self) -> None:
        link_shape = np.asarray(self.geometric_distance_m).shape
        for name in (
            "los_distance_m",
            "first_path_delay_s",
            "first_path_propagation_range_m",
            "strongest_path_delay_s",
            "los_aoa_azimuth_rad",
            "los_aoa_zenith_rad",
            "strongest_aoa_azimuth_rad",
            "strongest_aoa_zenith_rad",
            "first_path_aoa_azimuth_rad",
            "first_path_aoa_zenith_rad",
            "los_flag",
            "nlos_flag",
            "path_count",
            "path_power_db",
            "link_valid_mask",
            "tx_rx_bearing_rad",
            "tx_rx_distance_m",
        ):
            require_shape(name, np.asarray(getattr(self, name)), link_shape)
        require_shape("tx_rx_midpoint_m", np.asarray(self.tx_rx_midpoint_m), (*link_shape, 2))

        for name in (
            "geometric_distance_m",
            "los_distance_m",
            "first_path_delay_s",
            "first_path_propagation_range_m",
            "strongest_path_delay_s",
            "los_aoa_azimuth_rad",
            "los_aoa_zenith_rad",
            "strongest_aoa_azimuth_rad",
            "strongest_aoa_zenith_rad",
            "first_path_aoa_azimuth_rad",
            "first_path_aoa_zenith_rad",
            "path_power_db",
            "tx_rx_midpoint_m",
            "tx_rx_bearing_rad",
            "tx_rx_distance_m",
        ):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=np.float32))
        object.__setattr__(self, "los_flag", np.asarray(self.los_flag, dtype=np.bool_))
        object.__setattr__(self, "nlos_flag", np.asarray(self.nlos_flag, dtype=np.bool_))
        object.__setattr__(self, "path_count", np.asarray(self.path_count, dtype=np.int32))
        object.__setattr__(
            self, "link_valid_mask", np.asarray(self.link_valid_mask, dtype=np.bool_)
        )


def build_derived_labels(
    topology: Topology,
    truth: RTTruthResult | RTLinkSummary,
    path_table: PathTable | None = None,
    cir_truth: CIRTruth | None = None,
    link_config: LinkConfig | None = None,
) -> DerivedLabels:
    """Build the `/derived` label set."""

    tx_pos = topology.tx_positions_m[:, np.newaxis, :]
    rx_pos = topology.rx_positions_m[np.newaxis, :, :]
    delta = rx_pos - tx_pos
    link_shape = (topology.num_tx, topology.num_rx)

    geometric_distance_m = np.linalg.norm(delta, axis=-1).astype(np.float32)
    tx_rx_distance_m = np.linalg.norm(delta[..., :2], axis=-1).astype(np.float32)
    tx_rx_midpoint_m = ((tx_pos[..., :2] + rx_pos[..., :2]) * 0.5).astype(np.float32)
    tx_rx_bearing_rad = np.arctan2(delta[..., 1], delta[..., 0]).astype(np.float32)

    first_path_delay_s = _nan_array(link_shape)
    strongest_path_delay_s = _nan_array(link_shape)
    los_distance_m = _nan_array(link_shape)
    los_aoa_azimuth_rad = _nan_array(link_shape)
    los_aoa_zenith_rad = _nan_array(link_shape)
    strongest_aoa_azimuth_rad = _nan_array(link_shape)
    strongest_aoa_zenith_rad = _nan_array(link_shape)
    first_path_aoa_azimuth_rad = _nan_array(link_shape)
    first_path_aoa_zenith_rad = _nan_array(link_shape)

    if path_table is not None:
        _populate_path_labels(
            path_table,
            angle_source=_receiver_angle_source(link_config),
            first_path_delay_s=first_path_delay_s,
            strongest_path_delay_s=strongest_path_delay_s,
            los_distance_m=los_distance_m,
            los_aoa_azimuth_rad=los_aoa_azimuth_rad,
            los_aoa_zenith_rad=los_aoa_zenith_rad,
            strongest_aoa_azimuth_rad=strongest_aoa_azimuth_rad,
            strongest_aoa_zenith_rad=strongest_aoa_zenith_rad,
            first_path_aoa_azimuth_rad=first_path_aoa_azimuth_rad,
            first_path_aoa_zenith_rad=first_path_aoa_zenith_rad,
        )
    elif cir_truth is not None:
        _populate_delay_labels_from_cir(
            cir_truth,
            first_path_delay_s=first_path_delay_s,
            strongest_path_delay_s=strongest_path_delay_s,
        )

    return DerivedLabels(
        geometric_distance_m=geometric_distance_m,
        los_distance_m=los_distance_m,
        first_path_delay_s=first_path_delay_s,
        first_path_propagation_range_m=(
            first_path_delay_s * SPEED_OF_LIGHT_MPS
        ).astype(np.float32),
        strongest_path_delay_s=strongest_path_delay_s,
        los_aoa_azimuth_rad=los_aoa_azimuth_rad,
        los_aoa_zenith_rad=los_aoa_zenith_rad,
        strongest_aoa_azimuth_rad=strongest_aoa_azimuth_rad,
        strongest_aoa_zenith_rad=strongest_aoa_zenith_rad,
        first_path_aoa_azimuth_rad=first_path_aoa_azimuth_rad,
        first_path_aoa_zenith_rad=first_path_aoa_zenith_rad,
        los_flag=truth.los_exists,
        nlos_flag=truth.nlos_exists,
        path_count=truth.geometric_path_count,
        path_power_db=truth.path_power_db,
        link_valid_mask=truth.has_geometric_signal,
        tx_rx_midpoint_m=tx_rx_midpoint_m,
        tx_rx_bearing_rad=tx_rx_bearing_rad,
        tx_rx_distance_m=tx_rx_distance_m,
    )


def _populate_path_labels(
    table: PathTable,
    *,
    angle_source: str,
    first_path_delay_s: np.ndarray,
    strongest_path_delay_s: np.ndarray,
    los_distance_m: np.ndarray,
    los_aoa_azimuth_rad: np.ndarray,
    los_aoa_zenith_rad: np.ndarray,
    strongest_aoa_azimuth_rad: np.ndarray,
    strongest_aoa_zenith_rad: np.ndarray,
    first_path_aoa_azimuth_rad: np.ndarray,
    first_path_aoa_zenith_rad: np.ndarray,
) -> None:
    for tx in range(table.valid.shape[0]):
        for rx in range(table.valid.shape[1]):
            valid = table.valid[tx, rx]
            if not np.any(valid):
                continue

            tau = table.tau_s[tx, rx]
            first_local = _select_min_tau(valid, tau)
            strongest_local = _select_strongest(valid, table.a[tx, rx])
            los_local = _select_min_tau(valid & (table.path_type[tx, rx] == "los"), tau)

            _copy_delay_and_aoa(
                table,
                tx,
                rx,
                first_local,
                angle_source,
                first_path_delay_s,
                first_path_aoa_azimuth_rad,
                first_path_aoa_zenith_rad,
            )
            _copy_delay_and_aoa(
                table,
                tx,
                rx,
                strongest_local,
                angle_source,
                strongest_path_delay_s,
                strongest_aoa_azimuth_rad,
                strongest_aoa_zenith_rad,
            )
            if los_local is not None:
                _copy_delay_and_aoa(
                    table,
                    tx,
                    rx,
                    los_local,
                    angle_source,
                    los_distance_m,
                    los_aoa_azimuth_rad,
                    los_aoa_zenith_rad,
                    delay_scale=SPEED_OF_LIGHT_MPS,
                )


def _copy_delay_and_aoa(
    table: PathTable,
    tx: int,
    rx: int,
    local_index: tuple[int, int, int],
    angle_source: str,
    delay_out: np.ndarray,
    aoa_azimuth_out: np.ndarray,
    aoa_zenith_out: np.ndarray,
    *,
    delay_scale: float = 1.0,
) -> None:
    rx_ant, tx_ant, path = local_index
    delay_out[tx, rx] = table.tau_s[tx, rx, rx_ant, tx_ant, path] * delay_scale
    if angle_source == "aod":
        aoa_azimuth_out[tx, rx] = table.phi_t_rad[tx, rx, rx_ant, tx_ant, path]
        aoa_zenith_out[tx, rx] = table.theta_t_rad[tx, rx, rx_ant, tx_ant, path]
    else:
        aoa_azimuth_out[tx, rx] = table.phi_r_rad[tx, rx, rx_ant, tx_ant, path]
        aoa_zenith_out[tx, rx] = table.theta_r_rad[tx, rx, rx_ant, tx_ant, path]


def _receiver_angle_source(link_config: LinkConfig | None) -> str:
    """Return which raw RT direction should label the PHY receiver side."""

    if link_config is None:
        return "aoa"
    if (
        bool(link_config.reciprocity_applied)
        and str(link_config.phy_link_direction).lower() == "uplink"
    ):
        return "aod"
    return "aoa"


def _select_min_tau(valid: np.ndarray, tau_s: np.ndarray) -> tuple[int, int, int] | None:
    if not np.any(valid):
        return None
    scores = np.where(valid, tau_s, np.inf)
    return tuple(int(i) for i in np.unravel_index(np.argmin(scores), scores.shape))


def _select_strongest(valid: np.ndarray, coefficients: np.ndarray) -> tuple[int, int, int]:
    scores = np.where(valid, np.abs(coefficients) ** 2, -np.inf)
    return tuple(int(i) for i in np.unravel_index(np.argmax(scores), scores.shape))


def _nan_array(shape: tuple[int, int]) -> np.ndarray:
    return np.full(shape, np.nan, dtype=np.float32)


def _populate_delay_labels_from_cir(
    cir: CIRTruth,
    *,
    first_path_delay_s: np.ndarray,
    strongest_path_delay_s: np.ndarray,
) -> None:
    valid = np.any(cir.valid, axis=0)
    delays = cir.delays_s[0]
    coeff = cir.coefficients[0]
    for tx in range(valid.shape[0]):
        for rx in range(valid.shape[1]):
            link_valid = valid[tx, rx]
            if not np.any(link_valid):
                continue
            first = _select_min_tau(link_valid, delays[tx, rx])
            strongest = _select_strongest(link_valid, coeff[tx, rx])
            if first is not None:
                rx_ant, tx_ant, path = first
                first_path_delay_s[tx, rx] = delays[tx, rx, rx_ant, tx_ant, path]
            rx_ant, tx_ant, path = strongest
            strongest_path_delay_s[tx, rx] = delays[tx, rx, rx_ant, tx_ant, path]
