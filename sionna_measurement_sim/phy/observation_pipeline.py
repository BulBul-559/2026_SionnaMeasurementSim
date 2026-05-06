"""Minimal AWGN + LS observation pipeline."""

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


@dataclass(frozen=True)
class AWGNObservationConfig:
    """Configuration for the Phase 4 AWGN-only observation."""

    snr_db: float
    random_seed: int
    sample_rate_hz: float
    fft_size: int
    cp_length: int = 0
    num_ofdm_symbols: int = 1
    tx_power_dbm: float = 0.0


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
    """Estimate CFR from all-pilot observations over an AWGN channel."""

    torch.manual_seed(config.random_seed)
    h = torch.as_tensor(h_true, dtype=torch.complex64)
    snapshot_h = h.unsqueeze(0)
    signal_power = torch.mean(torch.abs(snapshot_h) ** 2, dim=(3, 4, 5), keepdim=True)
    noise_power = signal_power / (10.0 ** (config.snr_db / 10.0))
    noise = torch.sqrt(noise_power / 2.0) * (
        torch.randn_like(snapshot_h.real) + 1j * torch.randn_like(snapshot_h.real)
    )
    cfr_est = snapshot_h + noise.to(torch.complex64)

    error = cfr_est - snapshot_h
    nmse_linear = torch.sum(torch.abs(error) ** 2, dim=(3, 4, 5)) / torch.clamp(
        torch.sum(torch.abs(snapshot_h) ** 2, dim=(3, 4, 5)),
        min=1e-30,
    )
    nmse_db = 10.0 * torch.log10(torch.clamp(nmse_linear, min=1e-30))
    amplitude_error_db = 20.0 * torch.log10(
        torch.clamp(torch.mean(torch.abs(error), dim=(3, 4, 5)), min=1e-30)
    )
    phase_error_rad = torch.mean(torch.angle(cfr_est * torch.conj(snapshot_h)), dim=(3, 4, 5))
    correlation = _correlation(snapshot_h, cfr_est)

    link_shape = cfr_est.shape[:3]
    snr = torch.full(link_shape, config.snr_db, dtype=torch.float32)
    signal_power_link = signal_power.squeeze(-1).squeeze(-1).squeeze(-1)
    rssi_dbm = 10.0 * torch.log10(torch.clamp(signal_power_link, min=1e-30))
    noise_power_dbm = 10.0 * torch.log10(
        torch.clamp(noise_power.squeeze(-1).squeeze(-1).squeeze(-1), min=1e-30)
    )

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
    observation = ObservationResult(
        cfr_est=cfr_est.numpy(),
        valid_mask=np.ones(link_shape, dtype=np.bool_),
        detection_success=np.ones(link_shape, dtype=np.bool_),
        estimation_success=np.ones(link_shape, dtype=np.bool_),
        snr_db=snr.numpy(),
        rssi_dbm=rssi_dbm.numpy().astype(np.float32),
        noise_power_dbm=noise_power_dbm.numpy().astype(np.float32),
        cfo_hz=np.zeros(link_shape, dtype=np.float32),
        sfo_ppm=np.zeros(link_shape, dtype=np.float32),
        timing_offset_samples=np.zeros(link_shape, dtype=np.float32),
        phase_offset_rad=np.zeros(link_shape, dtype=np.float32),
        agc_gain_db=np.zeros((link_shape[0], link_shape[2]), dtype=np.float32),
        clipping_flag=np.zeros(link_shape, dtype=np.bool_),
    )
    impairments = ImpairmentSpec(
        model_version="phase4_awgn_v1",
        random_seed=config.random_seed,
        awgn_config=json.dumps({"snr_db": config.snr_db}, sort_keys=True),
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
        impairments=impairments,
        receiver=ReceiverSpec(),
        evaluation=evaluation,
    )


def _correlation(h_true: torch.Tensor, h_obs: torch.Tensor) -> torch.Tensor:
    numerator = torch.abs(torch.sum(torch.conj(h_true) * h_obs, dim=(3, 4, 5)))
    truth_norm = torch.sqrt(torch.sum(torch.abs(h_true) ** 2, dim=(3, 4, 5)))
    obs_norm = torch.sqrt(torch.sum(torch.abs(h_obs) ** 2, dim=(3, 4, 5)))
    return numerator / torch.clamp(truth_norm * obs_norm, min=1e-30)
