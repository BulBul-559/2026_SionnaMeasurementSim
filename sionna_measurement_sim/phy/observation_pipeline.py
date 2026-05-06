"""Minimal AWGN + LS observation pipeline with base impairments."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch

from sionna_measurement_sim.domain.observation import (
    EvaluationResult,
    ImpairmentSpec,
    ObservationResult,
    ReceiverSpec,
    WaveformSpec,
)
from sionna_measurement_sim.phy.impairments import (
    ImpairmentConfig,
    apply_base_impairments,
)


@dataclass(frozen=True)
class AWGNObservationConfig:
    """Configuration for the observation pipeline."""

    snr_db: float
    random_seed: int
    sample_rate_hz: float
    fft_size: int
    cp_length: int = 0
    num_ofdm_symbols: int = 1
    tx_power_dbm: float = 0.0
    impairment: ImpairmentConfig | None = None


@dataclass(frozen=True)
class PHYObservationBundle:
    """All domain objects produced by the observation chain."""

    waveform: WaveformSpec
    observation: ObservationResult
    impairments: ImpairmentSpec
    receiver: ReceiverSpec
    evaluation: EvaluationResult


def run_awgn_ls_observation(
    h_true: np.ndarray,
    config: AWGNObservationConfig,
) -> PHYObservationBundle:
    """Estimate CFR from pilot observations with optional impairments + AWGN + LS.

    h_true may be 5D (tx, rx, rx_ant, tx_ant, sub) or 6D (snapshot, tx, rx, ...).
    """

    torch.manual_seed(config.random_seed)
    h = torch.as_tensor(h_true, dtype=torch.complex64)
    if h.ndim == 5:
        h = h.unsqueeze(0)  # add snapshot dim
    elif h.ndim != 6:
        msg = f"h_true must be 5D or 6D, got {h.shape}"
        raise ValueError(msg)
    snapshot_h = h

    impairment_sample = None
    if config.impairment is not None:
        snapshot_h, impairment_sample = apply_base_impairments(
            snapshot_h, config.fft_size, config.sample_rate_hz, config.impairment
        )

    signal_dims = tuple(range(3, snapshot_h.ndim))
    signal_power = torch.mean(torch.abs(snapshot_h) ** 2, dim=signal_dims, keepdim=True)
    noise_power = signal_power / (10.0 ** (config.snr_db / 10.0))
    noise = torch.sqrt(noise_power / 2.0) * (
        torch.randn_like(snapshot_h.real) + 1j * torch.randn_like(snapshot_h.real)
    )
    cfr_est = snapshot_h + noise.to(torch.complex64)

    error = cfr_est - snapshot_h
    nmse_linear = torch.sum(torch.abs(error) ** 2, dim=signal_dims) / torch.clamp(
        torch.sum(torch.abs(snapshot_h) ** 2, dim=signal_dims),
        min=1e-30,
    )
    nmse_db = 10.0 * torch.log10(torch.clamp(nmse_linear, min=1e-30))
    amplitude_error_db = 20.0 * torch.log10(
        torch.clamp(torch.mean(torch.abs(error), dim=signal_dims), min=1e-30)
    )
    phase_error_rad = torch.mean(
        torch.angle(cfr_est * torch.conj(snapshot_h)), dim=signal_dims,
    )
    correlation = _correlation(snapshot_h, cfr_est, signal_dims)

    link_shape = cfr_est.shape[:3]
    snr = torch.full(link_shape, config.snr_db, dtype=torch.float32)
    squeeze_count = snapshot_h.ndim - 3
    signal_power_link = signal_power
    for _ in range(squeeze_count):
        signal_power_link = signal_power_link.squeeze(-1)
    rssi_dbm = 10.0 * torch.log10(torch.clamp(signal_power_link, min=1e-30))
    noise_link = noise_power
    for _ in range(squeeze_count):
        noise_link = noise_link.squeeze(-1)
    noise_power_dbm = 10.0 * torch.log10(torch.clamp(noise_link, min=1e-30))

    waveform = WaveformSpec(
        standard="custom_ofdm",
        sample_rate_hz=config.sample_rate_hz,
        fft_size=config.fft_size,
        cp_length=config.cp_length,
        num_ofdm_symbols=config.num_ofdm_symbols,
        pilot_indices=np.arange(config.fft_size, dtype=np.int32),
        data_subcarrier_indices=np.zeros((0,), dtype=np.int32),
        pilot_symbols=np.ones((config.fft_size,), dtype=np.complex64),
        tx_power_dbm=config.tx_power_dbm,
    )

    if impairment_sample is not None:
        obs_cfo_hz = impairment_sample.cfo_hz.numpy().astype(np.float32)
        obs_sfo_ppm = impairment_sample.sfo_ppm.numpy().astype(np.float32)
        obs_phase_offset_rad = impairment_sample.phase_offset_rad.numpy().astype(np.float32)
        obs_timing_offset = impairment_sample.timing_offset_samples.numpy().astype(np.float32)
        obs_agc_gain_db = impairment_sample.agc_gain_db.numpy().astype(np.float32)
        obs_clipping_flag = impairment_sample.clipping_flag.numpy()
        impairments_spec = _build_impairment_spec(config.impairment)
    else:
        obs_cfo_hz = np.zeros(link_shape, dtype=np.float32)
        obs_sfo_ppm = np.zeros(link_shape, dtype=np.float32)
        obs_phase_offset_rad = np.zeros(link_shape, dtype=np.float32)
        obs_timing_offset = np.zeros(link_shape, dtype=np.float32)
        obs_agc_gain_db = np.zeros((link_shape[0], link_shape[2]), dtype=np.float32)
        obs_clipping_flag = np.zeros(link_shape, dtype=np.bool_)
        impairments_spec = ImpairmentSpec(
            model_version="phase4_awgn_v1",
            random_seed=config.random_seed,
            awgn_config=json.dumps({"snr_db": config.snr_db}, sort_keys=True),
        )

    observation = ObservationResult(
        cfr_est=cfr_est.numpy(),
        valid_mask=np.ones(link_shape, dtype=np.bool_),
        detection_success=np.ones(link_shape, dtype=np.bool_),
        estimation_success=np.ones(link_shape, dtype=np.bool_),
        snr_db=snr.numpy(),
        rssi_dbm=rssi_dbm.numpy().astype(np.float32),
        noise_power_dbm=noise_power_dbm.numpy().astype(np.float32),
        cfo_hz=obs_cfo_hz,
        sfo_ppm=obs_sfo_ppm,
        timing_offset_samples=obs_timing_offset,
        phase_offset_rad=obs_phase_offset_rad,
        agc_gain_db=obs_agc_gain_db,
        clipping_flag=obs_clipping_flag,
    )
    evaluation = EvaluationResult(
        nmse_db=nmse_db.numpy().astype(np.float32),
        amplitude_error_db=amplitude_error_db.numpy().astype(np.float32),
        phase_error_rad=phase_error_rad.numpy().astype(np.float32),
        correlation=correlation.numpy().astype(np.float32),
        detection_rate=1.0,
        estimation_failure_rate=0.0,
    )
    return PHYObservationBundle(
        waveform=waveform,
        observation=observation,
        impairments=impairments_spec,
        receiver=ReceiverSpec(),
        evaluation=evaluation,
    )


def _build_impairment_spec(config: ImpairmentConfig) -> ImpairmentSpec:
    cfo = {} if config.cfo_hz is None else {"cfo_hz": config.cfo_hz}
    sfo = {} if config.sfo_ppm is None else {"sfo_ppm": config.sfo_ppm}
    timing = (
        {}
        if config.timing_offset_samples is None
        else {"timing_offset_samples": config.timing_offset_samples}
    )
    phase = {} if config.phase_offset_rad is None else {"phase_offset_rad": config.phase_offset_rad}
    agc_adc = {"agc_gain_db": config.agc_gain_db}
    if config.clipping_threshold is not None:
        agc_adc["clipping_threshold"] = config.clipping_threshold
    return ImpairmentSpec(
        model_version="phase5_base_impairments_v1",
        random_seed=config.random_seed,
        awgn_config=json.dumps({}, sort_keys=True),
        cfo_sfo_config=json.dumps({**cfo, **sfo, **timing}, sort_keys=True),
        phase_noise_config=json.dumps(phase, sort_keys=True),
        iq_imbalance_config=json.dumps({}, sort_keys=True),
        agc_adc_config=json.dumps(agc_adc, sort_keys=True),
    )


def _correlation(
    h_true: torch.Tensor, h_obs: torch.Tensor, signal_dims: tuple[int, ...] = (3, 4, 5)
) -> torch.Tensor:
    numerator = torch.abs(torch.sum(torch.conj(h_true) * h_obs, dim=signal_dims))
    truth_norm = torch.sqrt(torch.sum(torch.abs(h_true) ** 2, dim=signal_dims))
    obs_norm = torch.sqrt(torch.sum(torch.abs(h_obs) ** 2, dim=signal_dims))
    return numerator / torch.clamp(truth_norm * obs_norm, min=1e-30)
