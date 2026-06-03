"""Protocol-agnostic uplink power and RSSI calibration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

SUPPORTED_NOISE_MODES = ("relative_snr", "absolute_thermal")
THERMAL_NOISE_DENSITY_DBM_PER_HZ_290K = -173.975187


@dataclass(frozen=True)
class UplinkPowerResult:
    """Per-UE/per-port uplink power metadata and waveform scale."""

    tx_power_dbm: np.ndarray
    power_scale_linear: np.ndarray
    serving_rx_index: np.ndarray
    path_loss_db: np.ndarray
    closed_loop_db: np.ndarray
    clipped_flag: np.ndarray


def dbm_to_mw(dbm: float | np.ndarray) -> np.ndarray:
    """Convert dBm to mW."""

    return np.power(10.0, np.asarray(dbm, dtype=np.float64) / 10.0)


def mw_to_dbm(mw: float | np.ndarray, *, floor_mw: float = 1.0e-30) -> np.ndarray:
    """Convert mW to dBm with a numeric floor."""

    return 10.0 * np.log10(np.maximum(np.asarray(mw, dtype=np.float64), floor_mw))


def amplitude_scale_from_dbm(
    tx_power_dbm: float | np.ndarray,
    *,
    reference_tx_power_dbm: float = 0.0,
) -> np.ndarray:
    """Return voltage/amplitude scale relative to a reference dBm power."""

    return np.power(
        10.0,
        (np.asarray(tx_power_dbm, dtype=np.float64) - float(reference_tx_power_dbm))
        / 20.0,
    )


def thermal_noise_dbm(
    *,
    bandwidth_hz: float,
    noise_figure_db: float = 7.0,
    temperature_k: float = 290.0,
) -> float:
    """Return kTB thermal noise power in dBm for ``bandwidth_hz``."""

    bandwidth = float(bandwidth_hz)
    if bandwidth <= 0.0:
        raise ValueError("thermal noise bandwidth_hz must be positive")
    temperature = float(temperature_k)
    if temperature <= 0.0:
        raise ValueError("thermal noise temperature_k must be positive")
    density_dbm_hz = (
        THERMAL_NOISE_DENSITY_DBM_PER_HZ_290K
        + 10.0 * np.log10(temperature / 290.0)
    )
    return float(density_dbm_hz + 10.0 * np.log10(bandwidth) + float(noise_figure_db))


def thermal_noise_mw(
    *,
    bandwidth_hz: float,
    noise_figure_db: float = 7.0,
    temperature_k: float = 290.0,
) -> float:
    """Return kTB thermal noise power in mW for ``bandwidth_hz``."""

    return float(
        dbm_to_mw(
            thermal_noise_dbm(
                bandwidth_hz=bandwidth_hz,
                noise_figure_db=noise_figure_db,
                temperature_k=temperature_k,
            )
        )
    )


def compute_uplink_power(
    *,
    path_power_db: np.ndarray | None,
    snapshot_count: int,
    tx_count: int,
    rx_count: int,
    port_count: int,
    fixed_tx_power_dbm: float,
    power_config: Any | None = None,
    legacy_power_control: Any | None = None,
) -> UplinkPowerResult:
    """Compute standard-neutral uplink TX power and grid amplitude scale.

    The reference convention is explicit: a unit-amplitude transmitted grid is
    interpreted as ``reference_tx_power_dbm``.  With the default reference of
    0 dBm, setting ``fixed_tx_power_dbm=23`` multiplies the transmitted grid by
    ``sqrt(200)`` even when open-loop power control is disabled.
    """

    snapshot_count = int(snapshot_count)
    tx_count = int(tx_count)
    rx_count = int(rx_count)
    port_count = int(port_count)
    if min(snapshot_count, tx_count, rx_count, port_count) < 1:
        raise ValueError("snapshot_count, tx_count, rx_count, and port_count must be positive")

    reference_dbm = float(_get(power_config, "reference_tx_power_dbm", 0.0))
    apply_to_grid = bool(_get(power_config, "apply_tx_power_to_grid", True))
    control = _effective_uplink_control(power_config, legacy_power_control)
    control_enabled = bool(_get(control, "enabled", False))
    open_loop_enabled = bool(_get(control, "open_loop_enabled", True))
    closed_loop_enabled = bool(_get(control, "closed_loop_enabled", False))

    base = np.full(
        (snapshot_count, tx_count, port_count),
        float(fixed_tx_power_dbm),
        dtype=np.float32,
    )
    serving = np.zeros((snapshot_count, tx_count), dtype=np.int32)
    path_loss = np.zeros((snapshot_count, tx_count), dtype=np.float32)
    closed_loop_2d = np.zeros((snapshot_count, tx_count), dtype=np.float32)
    clipped_2d = np.zeros((snapshot_count, tx_count), dtype=np.bool_)

    if control_enabled:
        if path_power_db is None:
            raise ValueError("uplink power control requires path_power_db")
        path_power = _normalize_path_power(
            path_power_db,
            snapshot_count=snapshot_count,
            tx_count=tx_count,
            rx_count=rx_count,
        )
        policy = str(_get(control, "serving_rx_policy", "strongest_path"))
        serving = select_serving_rx(path_power, policy=policy)
        tx_indices = np.arange(tx_count, dtype=np.int32)[np.newaxis, :]
        snap_indices = np.arange(snapshot_count, dtype=np.int32)[:, np.newaxis]
        path_loss = -path_power[snap_indices, tx_indices, serving]
        requested = np.full((snapshot_count, tx_count), float(fixed_tx_power_dbm), dtype=np.float32)
        if open_loop_enabled:
            requested = (
                float(_get(control, "p0_dbm", 0.0))
                + float(_get(control, "alpha", 0.8)) * path_loss
            ).astype(np.float32)
        if closed_loop_enabled:
            closed_loop_2d = np.float32(
                float(_get(control, "tpc_offset_db", 0.0))
                + float(_get(control, "accumulation_db", 0.0))
            ) + np.zeros_like(requested, dtype=np.float32)
            requested = requested + closed_loop_2d
        min_dbm = float(_get(control, "min_tx_power_dbm", -40.0))
        max_dbm = float(_get(control, "max_tx_power_dbm", 23.0))
        clipped = np.clip(requested, min_dbm, max_dbm).astype(np.float32)
        clipped_2d = np.not_equal(clipped, requested)
        base = np.broadcast_to(
            clipped[:, :, np.newaxis],
            (snapshot_count, tx_count, port_count),
        ).copy()

    scale = amplitude_scale_from_dbm(
        base,
        reference_tx_power_dbm=reference_dbm,
    ).astype(np.float32)
    if not apply_to_grid:
        scale = np.ones_like(base, dtype=np.float32)

    return UplinkPowerResult(
        tx_power_dbm=base,
        power_scale_linear=scale,
        serving_rx_index=serving,
        path_loss_db=path_loss.astype(np.float32, copy=False),
        closed_loop_db=np.broadcast_to(
            closed_loop_2d[:, :, np.newaxis],
            base.shape,
        ).astype(np.float32, copy=True),
        clipped_flag=np.broadcast_to(
            clipped_2d[:, :, np.newaxis],
            base.shape,
        ).astype(np.bool_, copy=True),
    )


def select_serving_rx(path_power_db: np.ndarray, *, policy: str) -> np.ndarray:
    """Select serving RX for each snapshot/TX from path power in dB."""

    power = np.asarray(path_power_db, dtype=np.float32)
    if power.ndim != 3:
        raise ValueError(f"path_power_db must be rank 3 [snapshot,tx,rx], got {power.shape}")
    if policy == "first_rx":
        return np.zeros(power.shape[:2], dtype=np.int32)
    if policy == "strongest_path":
        return np.argmax(power, axis=2).astype(np.int32)
    raise ValueError("serving_rx_policy must be strongest_path/first_rx")


def noise_mode_from_config(power_config: Any | None) -> str:
    """Resolve and validate the configured noise power interpretation."""

    mode = str(_get(power_config, "noise_mode", "relative_snr"))
    if mode not in SUPPORTED_NOISE_MODES:
        allowed = ", ".join(SUPPORTED_NOISE_MODES)
        raise ValueError(f"phy.power.noise_mode must be one of: {allowed}")
    return mode


def effective_noise_bandwidth_hz(
    *,
    power_config: Any | None,
    default_bandwidth_hz: float,
) -> float:
    """Return thermal-noise bandwidth from config or active occupied bandwidth."""

    thermal = _get(power_config, "thermal_noise", None)
    configured = _get(thermal, "bandwidth_hz", None)
    if configured is None:
        return float(default_bandwidth_hz)
    bandwidth = float(configured)
    if bandwidth <= 0.0:
        raise ValueError("phy.power.thermal_noise.bandwidth_hz must be positive")
    return bandwidth


def thermal_noise_metadata(
    *,
    power_config: Any | None,
    default_bandwidth_hz: float,
) -> dict[str, float]:
    """Resolve thermal noise configuration and derived power."""

    thermal = _get(power_config, "thermal_noise", None)
    temperature_k = float(_get(thermal, "temperature_k", 290.0))
    noise_figure_db = float(_get(thermal, "noise_figure_db", 7.0))
    bandwidth_hz = effective_noise_bandwidth_hz(
        power_config=power_config,
        default_bandwidth_hz=default_bandwidth_hz,
    )
    noise_dbm = thermal_noise_dbm(
        bandwidth_hz=bandwidth_hz,
        noise_figure_db=noise_figure_db,
        temperature_k=temperature_k,
    )
    return {
        "temperature_k": temperature_k,
        "noise_figure_db": noise_figure_db,
        "effective_bandwidth_hz": bandwidth_hz,
        "thermal_noise_dbm": noise_dbm,
        "thermal_noise_mw": float(dbm_to_mw(noise_dbm)),
    }


def _effective_uplink_control(power_config: Any | None, legacy: Any | None) -> Any | None:
    common_control = _get(power_config, "uplink_control", None)
    if bool(_get(common_control, "enabled", False)):
        return common_control
    if bool(_get(legacy, "enabled", False)):
        return _LegacyUplinkControlAdapter(legacy)
    return common_control


def _normalize_path_power(
    path_power_db: np.ndarray,
    *,
    snapshot_count: int,
    tx_count: int,
    rx_count: int,
) -> np.ndarray:
    power = np.asarray(path_power_db, dtype=np.float32)
    if power.shape == (tx_count, rx_count):
        return np.broadcast_to(
            power[np.newaxis, :, :],
            (snapshot_count, tx_count, rx_count),
        ).copy()
    if power.shape == (snapshot_count, tx_count, rx_count):
        return power
    raise ValueError(
        "path_power_db must have shape "
        f"{(tx_count, rx_count)} or {(snapshot_count, tx_count, rx_count)}, got {power.shape}"
    )


def _get(obj: Any | None, name: str, default: Any) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


class _LegacyUplinkControlAdapter:
    """Map legacy SRS power_control fields onto common uplink control."""

    def __init__(self, legacy: Any) -> None:
        self.enabled = bool(_get(legacy, "enabled", False))
        self.serving_rx_policy = _get(legacy, "serving_rx_policy", "strongest_path")
        self.open_loop_enabled = True
        self.p0_dbm = float(_get(legacy, "p0_dbm", 0.0))
        self.alpha = float(_get(legacy, "alpha", 0.8))
        self.closed_loop_enabled = False
        self.tpc_offset_db = 0.0
        self.accumulation_db = 0.0
        self.min_tx_power_dbm = float(_get(legacy, "min_tx_power_dbm", -40.0))
        self.max_tx_power_dbm = float(_get(legacy, "max_tx_power_dbm", 23.0))
