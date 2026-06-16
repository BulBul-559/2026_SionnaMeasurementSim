"""Protocol-independent IQ observation builders."""

from __future__ import annotations

from typing import Any

import numpy as np

from sionna_measurement_sim.domain.iq import (
    IQObservationResult,
    LinkIQCapture,
    NonCooperativeIQCapture,
)
from sionna_measurement_sim.domain.multiuser import MultiUserSRSResult
from sionna_measurement_sim.domain.results import ShardMetadata
from sionna_measurement_sim.domain.topology import Topology

TIME_DOMAIN_CONVENTION = "ofdm_ifft_per_symbol_cp_appended_contiguous_symbols"


def build_iq_observation(
    *,
    iq_config: Any | None,
    noncooperative_config: Any | None,
    waveform_extras: dict[str, Any] | None,
    multiuser: MultiUserSRSResult | None,
    topology: Topology,
    shard: ShardMetadata | None,
    sample_rate_hz: float,
    fft_size: int,
    cp_length: int,
    num_ofdm_symbols: int,
) -> IQObservationResult | None:
    """Build optional `/iq` payload from existing PHY products."""

    link_cp_length = _resolve_cp_length(iq_config, cp_length)
    noncooperative_cp_length = _resolve_cp_length(noncooperative_config, cp_length)
    if (
        iq_config is not None
        and bool(getattr(iq_config, "enabled", False))
        and noncooperative_config is not None
        and bool(getattr(noncooperative_config, "enabled", False))
        and link_cp_length != noncooperative_cp_length
    ):
        raise ValueError("phy.iq.cp_length and noncooperative.cp_length must match")
    effective_cp_length = (
        noncooperative_cp_length
        if noncooperative_config is not None
        and bool(getattr(noncooperative_config, "enabled", False))
        else link_cp_length
    )

    link_capture = _build_link_capture(
        iq_config=iq_config,
        waveform_extras=waveform_extras or {},
        fft_size=fft_size,
        cp_length=link_cp_length,
    )
    noncooperative_capture = _build_noncooperative_capture(
        noncooperative_config=noncooperative_config,
        multiuser=multiuser,
        topology=topology,
        shard=shard,
        fft_size=fft_size,
        cp_length=noncooperative_cp_length,
    )

    if link_capture is None and noncooperative_capture is None:
        return None

    return IQObservationResult(
        sample_rate_hz=float(sample_rate_hz),
        fft_size=int(fft_size),
        cp_length=int(effective_cp_length),
        num_ofdm_symbols=int(num_ofdm_symbols),
        time_domain_convention=TIME_DOMAIN_CONVENTION,
        link=link_capture,
        noncooperative=noncooperative_capture,
    )


def frequency_grid_to_time_iq(
    grid: np.ndarray,
    *,
    fft_size: int,
    cp_length: int = 0,
) -> np.ndarray:
    """Convert frequency-domain OFDM IQ grid to contiguous time-domain IQ."""

    freq = np.asarray(grid, dtype=np.complex64)
    if freq.ndim < 2:
        raise ValueError("frequency IQ grid must include symbol and subcarrier axes")
    if int(fft_size) < freq.shape[-1]:
        raise ValueError("fft_size must be >= grid subcarrier dimension")
    if int(cp_length) < 0:
        raise ValueError("cp_length must be non-negative")

    time_symbols = np.fft.ifft(freq, n=int(fft_size), axis=-1, norm="backward")
    if cp_length:
        cp = time_symbols[..., -int(cp_length) :]
        time_symbols = np.concatenate([cp, time_symbols], axis=-1)
    leading = time_symbols.shape[:-2]
    sample_count = int(time_symbols.shape[-2] * time_symbols.shape[-1])
    return time_symbols.reshape(*leading, sample_count).astype(np.complex64, copy=False)


def _build_link_capture(
    *,
    iq_config: Any | None,
    waveform_extras: dict[str, Any],
    fft_size: int,
    cp_length: int,
) -> LinkIQCapture | None:
    if iq_config is None or not bool(getattr(iq_config, "enabled", False)):
        return None

    clean_output = getattr(iq_config, "clean_output", None)
    if clean_output is not None and clean_output not in ("time", "frequency", "both"):
        raise ValueError("phy.iq.clean_output must be time/frequency/both")
    write_frequency_clean = clean_output in ("frequency", "both")
    write_time_clean = clean_output in ("time", "both")

    frequency_clean = None
    if write_frequency_clean:
        frequency_clean = _require_waveform_extra(waveform_extras, "rx_grid_clean")

    frequency_observed = None
    if bool(getattr(iq_config, "save_frequency_observed", False)):
        frequency_observed = _require_waveform_extra(waveform_extras, "rx_grid")

    time_clean = None
    if write_time_clean:
        source = frequency_clean
        if source is None:
            source = _require_waveform_extra(waveform_extras, "rx_grid_clean")
        time_clean = frequency_grid_to_time_iq(
            source,
            fft_size=fft_size,
            cp_length=cp_length,
        )

    time_observed = None
    if bool(getattr(iq_config, "save_time_observed", False)):
        source = frequency_observed
        if source is None:
            source = _require_waveform_extra(waveform_extras, "rx_grid")
        time_observed = frequency_grid_to_time_iq(
            source,
            fft_size=fft_size,
            cp_length=cp_length,
        )

    capture = LinkIQCapture(
        frequency_clean=frequency_clean,
        frequency_observed=frequency_observed,
        time_clean=time_clean,
        time_observed=time_observed,
    )
    return None if capture.is_empty else capture


def _build_noncooperative_capture(
    *,
    noncooperative_config: Any | None,
    multiuser: MultiUserSRSResult | None,
    topology: Topology,
    shard: ShardMetadata | None,
    fft_size: int,
    cp_length: int,
) -> NonCooperativeIQCapture | None:
    if noncooperative_config is None or not bool(
        getattr(noncooperative_config, "enabled", False)
    ):
        return None
    if multiuser is None:
        raise ValueError(
            "noncooperative.enabled=true requires an NR SRS multi-UE shared "
            "observation source"
        )

    rx_time_clean = None
    if bool(getattr(noncooperative_config, "save_time_clean", True)):
        rx_time_clean = frequency_grid_to_time_iq(
            multiuser.rx_grid_clean_shared,
            fft_size=fft_size,
            cp_length=cp_length,
        )

    rx_time_observed = None
    if bool(getattr(noncooperative_config, "save_time_observed", True)):
        rx_time_observed = frequency_grid_to_time_iq(
            multiuser.rx_grid_shared,
            fft_size=fft_size,
            cp_length=cp_length,
        )

    return NonCooperativeIQCapture(
        rx_time_clean=rx_time_clean,
        rx_time_observed=rx_time_observed,
        active_tx_indices=multiuser.active_tx_indices,
        active_tx_global_indices=_global_tx_indices(
            multiuser.active_tx_indices,
            shard=shard,
        ),
        active_tx_mask=multiuser.active_tx_mask,
        active_tx_positions_m=_active_tx_positions(
            topology,
            multiuser.active_tx_indices,
        ),
        noise_variance=multiuser.noise_variance,
        snr_db=multiuser.snr_db,
        rssi_dbm=multiuser.rssi_dbm,
        noise_power_dbm=multiuser.noise_power_dbm,
        resource_occupancy_count=multiuser.resource_occupancy_count,
        resource_collision_mask=multiuser.resource_collision_mask,
    )


def _require_waveform_extra(
    waveform_extras: dict[str, Any],
    key: str,
) -> np.ndarray:
    value = waveform_extras.get(key)
    if value is None:
        raise ValueError(f"IQ capture requested waveform extra {key!r}, but it is absent")
    return np.asarray(value, dtype=np.complex64)


def _resolve_cp_length(config: Any | None, fallback: int) -> int:
    value = None if config is None else getattr(config, "cp_length", None)
    return int(fallback if value is None else value)


def _global_tx_indices(
    active_tx_indices: np.ndarray,
    *,
    shard: ShardMetadata | None,
) -> np.ndarray:
    local = np.asarray(active_tx_indices, dtype=np.int32)
    global_indices = np.full(local.shape, -1, dtype=np.int64)
    for index, tx_idx in np.ndenumerate(local):
        if int(tx_idx) < 0:
            continue
        global_indices[index] = (
            int(shard.global_tx_indices[int(tx_idx)])
            if shard is not None
            else int(tx_idx)
        )
    return global_indices


def _active_tx_positions(
    topology: Topology,
    active_tx_indices: np.ndarray,
) -> np.ndarray:
    active = np.asarray(active_tx_indices, dtype=np.int32)
    positions = np.full((*active.shape, 3), np.nan, dtype=np.float32)
    for index, tx_idx in np.ndenumerate(active):
        if int(tx_idx) < 0:
            continue
        positions[index] = topology.tx_positions_m[int(tx_idx)]
    return positions
