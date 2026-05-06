"""Base impairment models for PHY observation chain."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ImpairmentConfig:
    """User-facing impairment parameters. None means disabled."""

    random_seed: int
    cfo_hz: float | None = None
    sfo_ppm: float | None = None
    phase_offset_rad: float | None = None
    timing_offset_samples: float | None = None
    agc_gain_db: float = 0.0
    clipping_threshold: float | None = None


@dataclass(frozen=True)
class ImpairmentSample:
    """Per-link sampled impairment values ready for HDF5."""

    cfo_hz: torch.Tensor  # [snapshot, tx, rx]
    sfo_ppm: torch.Tensor  # [snapshot, tx, rx]
    phase_offset_rad: torch.Tensor  # [snapshot, tx, rx]
    timing_offset_samples: torch.Tensor  # [snapshot, tx, rx]
    agc_gain_db: torch.Tensor  # [snapshot, rx]
    clipping_flag: torch.Tensor  # [snapshot, tx, rx]


def apply_base_impairments(
    cfr: torch.Tensor,
    fft_size: int,
    sample_rate_hz: float,
    config: ImpairmentConfig,
) -> tuple[torch.Tensor, ImpairmentSample]:
    """Apply impairments to CFR in frequency domain.

    Order: IFFT → CFO (time domain) → FFT → SFO → phase → timing → AGC/ADC.
    Returns (impaired_cfr, sampled_impairments).
    """
    device = cfr.device
    snapshot, tx, rx = cfr.shape[:3]
    link_shape = (snapshot, tx, rx)

    td = torch.fft.ifft(cfr, n=fft_size, dim=-1, norm="backward")

    cfo_hz = _sample_constant_or_zero(link_shape, config.cfo_hz, device)
    td = _apply_cfo_time_domain(td, cfo_hz, sample_rate_hz)

    cfr_impaired = torch.fft.fft(td, n=fft_size, dim=-1, norm="backward")

    sfo_ppm = _sample_constant_or_zero(link_shape, config.sfo_ppm, device)
    cfr_impaired = _apply_sfo(cfr_impaired, sfo_ppm)

    phase_offset_rad = _sample_constant_or_zero(link_shape, config.phase_offset_rad, device)
    cfr_impaired = cfr_impaired * torch.exp(
        1j * _broadcast_to_cfr(phase_offset_rad, cfr_impaired)
    )

    timing_offset_samples = _sample_constant_or_zero(
        link_shape, config.timing_offset_samples, device,
    )
    cfr_impaired = _apply_timing_offset(cfr_impaired, timing_offset_samples, fft_size)

    agc_shape = (snapshot, rx)
    agc_gain_db = _sample_constant_or_zero(agc_shape, config.agc_gain_db, device)
    cfr_impaired, clipping_flag = _apply_agc_adc(
        cfr_impaired, agc_gain_db, config.clipping_threshold
    )

    return cfr_impaired, ImpairmentSample(
        cfo_hz=cfo_hz,
        sfo_ppm=sfo_ppm,
        phase_offset_rad=phase_offset_rad,
        timing_offset_samples=timing_offset_samples,
        agc_gain_db=agc_gain_db,
        clipping_flag=clipping_flag,
    )


def _sample_constant_or_zero(
    link_shape: tuple[int, ...],
    nominal: float | None,
    device: torch.device,
) -> torch.Tensor:
    """Return nominal value tensor or zeros if disabled."""
    if nominal is None:
        return torch.zeros(link_shape, dtype=torch.float32, device=device)
    return torch.full(link_shape, float(nominal), dtype=torch.float32, device=device)


def _broadcast_to_cfr(link_tensor: torch.Tensor, cfr: torch.Tensor) -> torch.Tensor:
    """Broadcast link-shaped tensor [snapshot, tx, rx] to CFR shape by adding trailing dims."""
    n_extra = cfr.ndim - link_tensor.ndim
    shape = link_tensor.shape + (1,) * n_extra
    return link_tensor.reshape(shape)


def _apply_cfo_time_domain(
    td: torch.Tensor,
    cfo_hz: torch.Tensor,
    sample_rate_hz: float,
) -> torch.Tensor:
    n_samples = td.shape[-1]
    t = torch.arange(n_samples, dtype=torch.float32, device=td.device) / sample_rate_hz
    phase = 2.0 * torch.pi * _broadcast_to_cfr(cfo_hz, td) * t
    return td * torch.exp(1j * phase)


def _apply_sfo(cfr: torch.Tensor, sfo_ppm: torch.Tensor) -> torch.Tensor:
    n_sc = cfr.shape[-1]
    k = torch.arange(n_sc, dtype=torch.float32, device=cfr.device)
    phase_ramp = 2.0 * torch.pi * k * _broadcast_to_cfr(sfo_ppm, cfr) * 1e-6
    return cfr * torch.exp(1j * phase_ramp)


def _apply_timing_offset(
    cfr: torch.Tensor,
    timing_offset_samples: torch.Tensor,
    fft_size: int,
) -> torch.Tensor:
    n_sc = cfr.shape[-1]
    k = torch.arange(n_sc, dtype=torch.float32, device=cfr.device)
    phase = 2.0 * torch.pi * k * _broadcast_to_cfr(timing_offset_samples, cfr)
    phase_ramp = phase / float(fft_size)
    return cfr * torch.exp(1j * phase_ramp)


def _apply_agc_adc(
    cfr: torch.Tensor,
    agc_gain_db: torch.Tensor,
    clipping_threshold: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # agc_gain_db: [snapshot, rx] → broadcast to CFR shape
    gain_dims = agc_gain_db[:, None, :].reshape(
        agc_gain_db.shape[0], 1, agc_gain_db.shape[1],
        *(1 for _ in range(cfr.ndim - 3)),
    )
    gain_linear = 10.0 ** (gain_dims / 20.0)
    scaled = cfr * gain_linear

    if clipping_threshold is None:
        return scaled, torch.zeros(cfr.shape[:3], dtype=torch.bool, device=cfr.device)

    magnitude = torch.abs(scaled)
    clipped = magnitude > clipping_threshold
    scale = torch.where(clipped, clipping_threshold / torch.clamp(magnitude, min=1e-30), 1.0)
    signal_dims = tuple(range(3, cfr.ndim))
    clipped_cfr = scaled * scale
    clipping_flag = torch.any(clipped, dim=signal_dims)
    return clipped_cfr, clipping_flag
