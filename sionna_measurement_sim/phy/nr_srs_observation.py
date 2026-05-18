"""NR SRS-like full-band uplink sounding observation path."""

from __future__ import annotations

import json
from contextlib import nullcontext
from typing import Any

import numpy as np

from sionna_measurement_sim.domain.link import LinkConfig
from sionna_measurement_sim.domain.observation import (
    EvaluationResult,
    ImpairmentSpec,
    ObservationResult,
    ReceiverSpec,
    WaveformSpec,
)
from sionna_measurement_sim.phy.spatial_spectrum import project_cfr_to_ul_receiver_samples


def run_nr_srs_observation(
    truth_cfr: np.ndarray,
    link_config: LinkConfig,
    phy_config: Any,
    carrier_config: Any,
    *,
    cfr_snapshots: np.ndarray | None = None,
    has_signal: np.ndarray | None = None,
    tracer: Any | None = None,
) -> dict[str, Any]:
    """Run full-band SRS-like sounding and LS CFR estimation.

    This is intentionally SRS-like rather than a complete 3GPP NR SRS
    implementation: every active subcarrier carries a known orthogonal pilot.
    """

    _ = link_config
    rng = np.random.default_rng(int(getattr(phy_config, "observation_seed", 42)))
    sc_spacing_hz = float(getattr(phy_config, "subcarrier_spacing_khz", 30)) * 1000.0
    num_subcarriers = int(getattr(carrier_config, "num_subcarriers", truth_cfr.shape[-1]))
    truth = cfr_snapshots if cfr_snapshots is not None else truth_cfr
    with _span(tracer, "nr_srs.project_cfr_to_ul", truth_shape=_shape_tuple(truth)):
        h_ul = project_cfr_to_ul_receiver_samples(truth)
    num_snap, num_ul_tx, num_ul_rx, num_ul_rx_ant, num_ul_tx_ant, num_sc = h_ul.shape
    if num_sc != num_subcarriers:
        num_subcarriers = num_sc
    num_symbols = max(int(getattr(phy_config, "num_ofdm_symbols", 1)), num_ul_tx_ant)
    snr_db = float(
        getattr(phy_config, "snr_db", None)
        if getattr(phy_config, "snr_db", None) is not None
        else getattr(phy_config, "observation_snr_db", 30.0)
    )

    with _span(tracer, "nr_srs.build_pilot_code", tx_ant=num_ul_tx_ant, symbols=num_symbols):
        pilot_code = _orthogonal_pilot_code(num_ul_tx_ant, num_symbols)
    with _span(
        tracer,
        "nr_srs.build_tx_grid",
        links=int(num_snap * num_ul_tx * num_ul_rx),
        tx_ant=int(num_ul_tx_ant),
        symbols=int(num_symbols),
        subcarriers=int(num_subcarriers),
    ):
        tx_grid = _build_srs_tx_grid(
            link_shape=(num_snap, num_ul_tx, num_ul_rx),
            num_ul_tx_ant=num_ul_tx_ant,
            num_symbols=num_symbols,
            num_subcarriers=num_subcarriers,
            pilot_code=pilot_code,
        )
    _record_array_event(tracer, "nr_srs.array_shape", "h_ul", h_ul)
    _record_array_event(tracer, "nr_srs.array_shape", "tx_grid", tx_grid)
    with _span(tracer, "nr_srs.channel_apply_einsum"):
        y_clean = np.einsum("...rtf,...tsf->...rsf", h_ul, tx_grid, optimize=True)
    _record_array_event(tracer, "nr_srs.array_shape", "y_clean", y_clean)
    with _span(tracer, "nr_srs.noise_and_rx_grid", snr_db=float(snr_db)):
        signal_power = np.mean(np.abs(y_clean) ** 2, axis=(3, 4, 5), keepdims=True)
        noise_variance = signal_power / np.float32(10.0 ** (snr_db / 10.0))
        noise = np.sqrt(noise_variance / np.float32(2.0)) * (
            rng.standard_normal(y_clean.shape, dtype=np.float32)
            + 1j * rng.standard_normal(y_clean.shape, dtype=np.float32)
        )
        rx_grid = (y_clean + noise).astype(np.complex64, copy=False)
    _record_array_event(tracer, "nr_srs.array_shape", "rx_grid", rx_grid)
    with _span(tracer, "nr_srs.ls_estimate"):
        h_hat_ul = (
            np.einsum("...rsf,ts->...rtf", rx_grid, np.conjugate(pilot_code), optimize=True)
            / np.float32(num_symbols)
        ).astype(np.complex64, copy=False)
    with _span(tracer, "nr_srs.to_link_view"):
        cfr_est = h_hat_ul.astype(np.complex64, copy=False)
        truth_dl = (
            np.asarray(truth, dtype=np.complex64)[np.newaxis, ...]
            if np.asarray(truth).ndim == 5
            else np.asarray(truth, dtype=np.complex64)
        )

    link_shape = cfr_est.shape[:3]
    valid_mask = _build_valid_mask(has_signal, link_shape)
    with _span(tracer, "nr_srs.metrics"):
        nmse_db, amplitude_error_db, phase_error_rad, correlation = _estimate_metrics(
            truth_dl,
            cfr_est,
            valid_mask,
        )
        rssi_dbm = 10.0 * np.log10(
            np.maximum(np.mean(np.abs(truth_dl) ** 2, axis=(3, 4, 5)), 1e-30)
        ).astype(np.float32)
        noise_dbm = 10.0 * np.log10(
            np.maximum(np.squeeze(noise_variance, axis=(3, 4, 5)), 1e-30)
        ).astype(np.float32)

    with _span(tracer, "nr_srs.domain_models"):
        waveform = WaveformSpec(
            standard="nr_srs",
            sample_rate_hz=sc_spacing_hz * num_subcarriers,
            fft_size=num_subcarriers,
            cp_length=0,
            num_ofdm_symbols=num_symbols,
            pilot_indices=np.arange(num_subcarriers, dtype=np.int32),
            data_subcarrier_indices=np.zeros((0,), dtype=np.int32),
            pilot_symbols=np.ones((num_subcarriers,), dtype=np.complex64),
            tx_power_dbm=float(getattr(phy_config, "tx_power_dbm", 0.0)),
        )
        observation = ObservationResult(
            cfr_est=cfr_est,
            valid_mask=valid_mask,
            detection_success=valid_mask.copy(),
            estimation_success=valid_mask.copy(),
            snr_db=np.full(link_shape, snr_db, dtype=np.float32),
            rssi_dbm=rssi_dbm,
            noise_power_dbm=noise_dbm,
            cfo_hz=np.zeros(link_shape, dtype=np.float32),
            sfo_ppm=np.zeros(link_shape, dtype=np.float32),
            timing_offset_samples=np.zeros(link_shape, dtype=np.float32),
            phase_offset_rad=np.zeros(link_shape, dtype=np.float32),
            agc_gain_db=np.zeros((link_shape[0], link_shape[2]), dtype=np.float32),
            clipping_flag=np.zeros(link_shape, dtype=np.bool_),
        )
        evaluation = EvaluationResult(
            nmse_db=nmse_db,
            nmse_db_total=nmse_db.copy(),
            amplitude_error_db=amplitude_error_db,
            phase_error_rad=phase_error_rad,
            correlation=correlation,
            detection_rate=float(np.mean(valid_mask)) if valid_mask.size else 0.0,
            estimation_failure_rate=float(np.mean(~valid_mask)) if valid_mask.size else 0.0,
        )
        waveform_grids = {
            "srs_tx_grid": tx_grid.astype(np.complex64, copy=False),
            "srs_rx_grid": rx_grid,
            "srs_noise_variance": np.squeeze(noise_variance, axis=(3, 4, 5)).astype(
                np.float32,
                copy=False,
            ),
            "srs_pilot_code": pilot_code.astype(np.complex64, copy=False),
        }
    return {
        "nr_waveform_spec": waveform,
        "waveform_spec": waveform,
        "receiver_spec": ReceiverSpec(
            receiver_type="srs_ls_receiver",
            estimator_type="srs_ls",
            sync_method="ideal",
            mimo_detector="none",
            failure_policy=getattr(phy_config, "receiver_failure_policy", "mark_invalid"),
        ),
        "evaluation": evaluation,
        "observation": observation,
        "impairments": ImpairmentSpec(
            model_version="nr_srs_fullband_sounding_v1",
            random_seed=int(getattr(phy_config, "observation_seed", 42)),
            awgn_config=json.dumps({"snr_db": snr_db}, sort_keys=True),
        ),
        "waveform_grids": waveform_grids,
        "metadata": {
            "srs_like_scope": "full_band_orthogonal_sounding",
            "num_srs_symbols": num_symbols,
            "num_ul_tx_ant": num_ul_tx_ant,
            "num_ul_rx_ant": num_ul_rx_ant,
        },
    }


def _orthogonal_pilot_code(num_tx_ant: int, num_symbols: int) -> np.ndarray:
    if num_symbols < num_tx_ant:
        raise ValueError("num_ofdm_symbols must be >= number of UE/SRS TX antennas")
    symbol = np.arange(num_symbols, dtype=np.float32)
    antenna = np.arange(num_tx_ant, dtype=np.float32)[:, np.newaxis]
    code = np.exp(1j * 2.0 * np.pi * antenna * symbol / np.float32(num_symbols))
    return code.astype(np.complex64, copy=False)


def _span(tracer: Any | None, name: str, **metadata: Any) -> Any:
    if tracer is None:
        return nullcontext()
    return tracer.span(name, **metadata)


def _shape_tuple(array: np.ndarray) -> tuple[int, ...]:
    return tuple(int(dim) for dim in np.asarray(array).shape)


def _record_array_event(
    tracer: Any | None,
    event: str,
    name: str,
    array: np.ndarray,
) -> None:
    if tracer is None:
        return
    arr = np.asarray(array)
    tracer.record_event(
        event,
        array=name,
        shape=_shape_tuple(arr),
        dtype=str(arr.dtype),
        bytes=int(arr.nbytes),
    )


def _build_srs_tx_grid(
    *,
    link_shape: tuple[int, int, int],
    num_ul_tx_ant: int,
    num_symbols: int,
    num_subcarriers: int,
    pilot_code: np.ndarray,
) -> np.ndarray:
    tx_grid = np.broadcast_to(
        pilot_code[np.newaxis, np.newaxis, np.newaxis, :, :, np.newaxis],
        (*link_shape, num_ul_tx_ant, num_symbols, num_subcarriers),
    )
    return tx_grid.astype(np.complex64, copy=True)


def _build_valid_mask(
    has_signal: np.ndarray | None,
    link_shape: tuple[int, int, int],
) -> np.ndarray:
    if has_signal is None:
        return np.ones(link_shape, dtype=np.bool_)
    link_mask = np.asarray(has_signal, dtype=np.bool_)
    if link_mask.shape != link_shape[1:]:
        raise ValueError(f"has_signal must have shape {link_shape[1:]}, got {link_mask.shape}")
    return np.broadcast_to(link_mask[np.newaxis, ...], link_shape).copy()


def _estimate_metrics(
    truth: np.ndarray,
    estimate: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    error = estimate - truth
    signal_power = np.sum(np.abs(truth) ** 2, axis=(3, 4, 5))
    error_power = np.sum(np.abs(error) ** 2, axis=(3, 4, 5))
    nmse_linear = error_power / np.maximum(signal_power, 1e-30)
    nmse_db = 10.0 * np.log10(np.maximum(nmse_linear, 1e-30)).astype(np.float32)
    amplitude_error_db = 20.0 * np.log10(
        np.maximum(np.mean(np.abs(error), axis=(3, 4, 5)), 1e-30)
    ).astype(np.float32)
    phase_error_rad = np.mean(np.angle(estimate * np.conjugate(truth)), axis=(3, 4, 5)).astype(
        np.float32
    )
    numerator = np.abs(np.sum(np.conjugate(truth) * estimate, axis=(3, 4, 5)))
    denominator = np.sqrt(signal_power) * np.sqrt(np.sum(np.abs(estimate) ** 2, axis=(3, 4, 5)))
    correlation = (numerator / np.maximum(denominator, 1e-30)).astype(np.float32)
    for array in (nmse_db, amplitude_error_db, phase_error_rad, correlation):
        array[~valid_mask] = 0.0
    return nmse_db, amplitude_error_db, phase_error_rad, correlation
