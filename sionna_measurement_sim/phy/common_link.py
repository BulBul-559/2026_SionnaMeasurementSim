"""Common PHY observation link primitives.

The classes in this module are deliberately standard-agnostic.  NR SRS,
NR PUSCH, and future WiFi-like waveforms can provide their own waveform
builder and receiver while sharing the clean-grid impairment and AWGN path.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from sionna_measurement_sim.domain.observation import ImpairmentSpec
from sionna_measurement_sim.phy.impairments import (
    ImpairmentConfig,
    ImpairmentSample,
    apply_base_impairments,
)
from sionna_measurement_sim.phy.power import mw_to_dbm, noise_mode_from_config


@dataclass(frozen=True)
class WaveformGrid:
    """Standard-specific transmitted frequency-domain grid."""

    tx_grid: np.ndarray | torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CleanChannelResult:
    """Output of applying a clean channel to a waveform grid."""

    rx_grid_clean: np.ndarray | torch.Tensor
    h_perfect: np.ndarray | torch.Tensor | None
    link_shape: tuple[int, int, int]
    grid_shape: tuple[int, ...]


@dataclass(frozen=True)
class ImpairmentChainResult:
    """Common impaired observation grid and per-link metadata."""

    rx_grid: torch.Tensor
    noise_variance: torch.Tensor
    snr_db: torch.Tensor
    rssi_dbm: torch.Tensor
    noise_power_dbm: torch.Tensor
    impairment_sample: ImpairmentSample
    impairment_spec: ImpairmentSpec


class ObservationImpairmentChain:
    """Apply shared baseband impairments and AWGN to a frequency grid.

    Parameters
    ----------
    fft_size:
        FFT size used by the input grid's final subcarrier dimension.
    sample_rate_hz:
        Sample rate used for time-domain CFO application.
    random_seed:
        Seed for AWGN generation.
    impairment_config:
        Optional deterministic base impairment configuration.  ``None`` keeps
        all base impairment metadata at zero.
    awgn_enabled:
        If ``False``, no AWGN is added and ``noise_variance`` is all zero.
    """

    model_version = "common_observation_impairments_v1"

    def __init__(
        self,
        *,
        fft_size: int,
        sample_rate_hz: float,
        random_seed: int,
        impairment_config: ImpairmentConfig | None = None,
        awgn_enabled: bool = True,
        noise_mode: str = "relative_snr",
        thermal_noise_power_mw: float | np.ndarray | torch.Tensor | None = None,
        thermal_noise_config: dict[str, float] | None = None,
    ) -> None:
        self.fft_size = int(fft_size)
        self.sample_rate_hz = float(sample_rate_hz)
        self.random_seed = int(random_seed)
        self.impairment_config = impairment_config
        self.awgn_enabled = bool(awgn_enabled)
        self.noise_mode = noise_mode_from_config({"noise_mode": noise_mode})
        self.thermal_noise_power_mw = thermal_noise_power_mw
        self.thermal_noise_config = dict(thermal_noise_config or {})
        self._generators: dict[str, torch.Generator] = {}

    def apply(
        self,
        rx_grid_clean: np.ndarray | torch.Tensor,
        *,
        snr_db: float,
        noise_variance_override: float | np.ndarray | torch.Tensor | None = None,
        noise_mode: str | None = None,
        thermal_noise_power_mw: float | np.ndarray | torch.Tensor | None = None,
    ) -> ImpairmentChainResult:
        """Return an impaired grid with per-link metadata.

        ``rx_grid_clean`` must have leading dimensions
        ``[snapshot, tx, rx, ...]``.  All trailing dimensions are treated as
        signal dimensions, with the final dimension interpreted as subcarrier.
        """

        clean = _as_complex_tensor(rx_grid_clean)
        if clean.ndim < 4:
            raise ValueError(
                "rx_grid_clean must have leading [snapshot,tx,rx] "
                f"and at least one signal dimension, got {tuple(clean.shape)}"
            )
        if clean.shape[-1] != self.fft_size:
            raise ValueError(
                f"rx_grid_clean subcarrier dimension {clean.shape[-1]} "
                f"does not match fft_size={self.fft_size}"
            )

        impaired = clean
        if self.impairment_config is not None:
            impaired, sample = apply_base_impairments(
                impaired,
                self.fft_size,
                self.sample_rate_hz,
                self.impairment_config,
            )
        else:
            sample = _zero_impairment_sample(impaired.shape[:3], impaired.device)

        signal_dims = tuple(range(3, impaired.ndim))
        signal_power = torch.mean(torch.abs(impaired) ** 2, dim=signal_dims)
        noise_variance = self._resolve_noise_variance(
            signal_power,
            snr_db=snr_db,
            override=noise_variance_override,
            noise_mode=noise_mode,
            thermal_noise_power_mw=thermal_noise_power_mw,
        )
        if self.awgn_enabled:
            noise = self._complex_awgn(impaired.shape, impaired.device, impaired.dtype)
            rx_grid = impaired + noise * torch.sqrt(
                _broadcast_link(noise_variance, impaired) / 2.0
            )
        else:
            rx_grid = impaired

        snr = 10.0 * torch.log10(
            torch.clamp(signal_power, min=1e-30)
            / torch.clamp(noise_variance, min=1e-30)
        )
        if not self.awgn_enabled:
            snr = torch.full_like(snr, float(snr_db), dtype=torch.float32)
        rssi_dbm = torch.as_tensor(
            mw_to_dbm(_to_numpy(signal_power)),
            dtype=torch.float32,
            device=impaired.device,
        )
        noise_power_dbm = torch.as_tensor(
            mw_to_dbm(_to_numpy(noise_variance)),
            dtype=torch.float32,
            device=impaired.device,
        )
        return ImpairmentChainResult(
            rx_grid=rx_grid.to(torch.complex64),
            noise_variance=noise_variance.to(torch.float32),
            snr_db=snr,
            rssi_dbm=rssi_dbm.to(torch.float32),
            noise_power_dbm=noise_power_dbm.to(torch.float32),
            impairment_sample=sample,
            impairment_spec=self.build_spec(float(snr_db)),
        )

    def build_spec(self, snr_db: float) -> ImpairmentSpec:
        """Build the HDF5 impairment spec for this chain."""

        config = self.impairment_config
        awgn_config = {
            "enabled": self.awgn_enabled,
            "snr_db": snr_db,
            "noise_mode": self.noise_mode,
        }
        if self.thermal_noise_config:
            awgn_config.update(self.thermal_noise_config)
        if config is None:
            return ImpairmentSpec(
                model_version=self.model_version,
                random_seed=self.random_seed,
                awgn_config=json.dumps(awgn_config, sort_keys=True),
            )

        cfo_sfo_timing: dict[str, float] = {}
        if config.cfo_hz is not None:
            cfo_sfo_timing["cfo_hz"] = float(config.cfo_hz)
        if config.sfo_ppm is not None:
            cfo_sfo_timing["sfo_ppm"] = float(config.sfo_ppm)
        if config.timing_offset_samples is not None:
            cfo_sfo_timing["timing_offset_samples"] = float(
                config.timing_offset_samples
            )
        phase = (
            {}
            if config.phase_offset_rad is None
            else {"phase_offset_rad": float(config.phase_offset_rad)}
        )
        agc_adc: dict[str, float] = {"agc_gain_db": float(config.agc_gain_db)}
        if config.clipping_threshold is not None:
            agc_adc["clipping_threshold"] = float(config.clipping_threshold)
        return ImpairmentSpec(
            model_version=self.model_version,
            random_seed=self.random_seed,
            awgn_config=json.dumps(awgn_config, sort_keys=True),
            cfo_sfo_config=json.dumps(cfo_sfo_timing, sort_keys=True),
            phase_noise_config=json.dumps(phase, sort_keys=True),
            iq_imbalance_config=json.dumps({}, sort_keys=True),
            agc_adc_config=json.dumps(agc_adc, sort_keys=True),
        )

    def _resolve_noise_variance(
        self,
        signal_power: torch.Tensor,
        *,
        snr_db: float,
        override: float | np.ndarray | torch.Tensor | None,
        noise_mode: str | None,
        thermal_noise_power_mw: float | np.ndarray | torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.awgn_enabled:
            return torch.zeros_like(signal_power, dtype=torch.float32)
        if override is None:
            mode = self.noise_mode if noise_mode is None else noise_mode_from_config(
                {"noise_mode": noise_mode}
            )
            if mode == "absolute_thermal":
                thermal = (
                    self.thermal_noise_power_mw
                    if thermal_noise_power_mw is None
                    else thermal_noise_power_mw
                )
                if thermal is None:
                    raise ValueError(
                        "absolute_thermal noise mode requires thermal_noise_power_mw"
                    )
                return _as_broadcast_float_tensor(
                    thermal,
                    signal_power,
                    name="thermal_noise_power_mw",
                )
            return (signal_power / (10.0 ** (float(snr_db) / 10.0))).to(torch.float32)

        return _as_broadcast_float_tensor(
            override,
            signal_power,
            name="noise_variance_override",
        )

    def _complex_awgn(
        self,
        shape: torch.Size,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = str(device)
        generator = self._generators.get(key)
        if generator is None:
            generator = torch.Generator(device=device).manual_seed(self.random_seed)
            self._generators[key] = generator
        real = torch.randn(
            shape,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        imag = torch.randn(
            shape,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        return (real + 1j * imag).to(dtype)


class BaseObservationModule(ABC):
    """Template for standards built on the common observation link."""

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Template method for standard-specific observation modules."""
        waveform = self.build_waveform(*args, **kwargs)
        clean = self.apply_channel_clean(waveform, *args, **kwargs)
        impaired = self.apply_impairments(clean, *args, **kwargs)
        receiver_output = self.run_receiver(clean, impaired, *args, **kwargs)
        evaluation = self.evaluate(clean, impaired, receiver_output, *args, **kwargs)
        return self.assemble_result(
            waveform,
            clean,
            impaired,
            receiver_output,
            evaluation,
            *args,
            **kwargs,
        )

    @abstractmethod
    def build_waveform(self, *args: Any, **kwargs: Any) -> WaveformGrid:
        """Build a standard-specific transmitted waveform grid."""

    @abstractmethod
    def apply_channel_clean(
        self,
        waveform: WaveformGrid,
        *args: Any,
        **kwargs: Any,
    ) -> CleanChannelResult:
        """Apply a clean channel without receiver noise."""

    @abstractmethod
    def run_receiver(
        self,
        clean: CleanChannelResult,
        impaired: ImpairmentChainResult,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Run the standard-specific receiver or estimator."""

    @abstractmethod
    def apply_impairments(
        self,
        clean: CleanChannelResult,
        *args: Any,
        **kwargs: Any,
    ) -> ImpairmentChainResult:
        """Apply the common impairment chain to a clean received grid."""

    @abstractmethod
    def evaluate(
        self,
        clean: CleanChannelResult,
        impaired: ImpairmentChainResult,
        receiver_output: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Evaluate standard-specific receiver outputs."""

    @abstractmethod
    def assemble_result(
        self,
        waveform: WaveformGrid,
        clean: CleanChannelResult,
        impaired: ImpairmentChainResult,
        receiver_output: Any,
        evaluation: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Assemble domain objects for the pipeline/HDF5 writer."""


class ResultAssembler:
    """Small conversion helpers used by standard-specific result assembly."""

    @staticmethod
    def sample_to_numpy(sample: ImpairmentSample) -> dict[str, np.ndarray]:
        return {
            "cfo_hz": _to_numpy(sample.cfo_hz).astype(np.float32, copy=False),
            "sfo_ppm": _to_numpy(sample.sfo_ppm).astype(np.float32, copy=False),
            "phase_offset_rad": _to_numpy(sample.phase_offset_rad).astype(
                np.float32,
                copy=False,
            ),
            "timing_offset_samples": _to_numpy(sample.timing_offset_samples).astype(
                np.float32,
                copy=False,
            ),
            "agc_gain_db": _to_numpy(sample.agc_gain_db).astype(np.float32, copy=False),
            "clipping_flag": _to_numpy(sample.clipping_flag).astype(np.bool_, copy=False),
        }

    @staticmethod
    def chain_scalars_to_numpy(result: ImpairmentChainResult) -> dict[str, np.ndarray]:
        return {
            "snr_db": _to_numpy(result.snr_db).astype(np.float32, copy=False),
            "rssi_dbm": _to_numpy(result.rssi_dbm).astype(np.float32, copy=False),
            "noise_power_dbm": _to_numpy(result.noise_power_dbm).astype(
                np.float32,
                copy=False,
            ),
            "noise_variance": _to_numpy(result.noise_variance).astype(
                np.float32,
                copy=False,
            ),
        }


def _as_complex_tensor(value: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(dtype=torch.complex64)
    return torch.as_tensor(np.asarray(value), dtype=torch.complex64)


def _as_broadcast_float_tensor(
    value: float | np.ndarray | torch.Tensor,
    target: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32, device=target.device)
    if tensor.ndim == 0:
        return torch.full_like(target, float(tensor.item()), dtype=torch.float32)
    if tensor.shape == target.shape:
        return tensor.to(torch.float32)
    if tensor.numel() == target.numel():
        return tensor.reshape(target.shape).to(torch.float32)
    try:
        return torch.broadcast_to(tensor, target.shape).to(torch.float32)
    except RuntimeError as exc:
        raise ValueError(
            f"{name} must be scalar or broadcastable to "
            f"{tuple(target.shape)}, got {tuple(tensor.shape)}"
        ) from exc


def _zero_impairment_sample(
    link_shape: tuple[int, int, int],
    device: torch.device,
) -> ImpairmentSample:
    snapshot, _tx, rx = link_shape
    return ImpairmentSample(
        cfo_hz=torch.zeros(link_shape, dtype=torch.float32, device=device),
        sfo_ppm=torch.zeros(link_shape, dtype=torch.float32, device=device),
        phase_offset_rad=torch.zeros(link_shape, dtype=torch.float32, device=device),
        timing_offset_samples=torch.zeros(link_shape, dtype=torch.float32, device=device),
        agc_gain_db=torch.zeros((snapshot, rx), dtype=torch.float32, device=device),
        clipping_flag=torch.zeros(link_shape, dtype=torch.bool, device=device),
    )


def _broadcast_link(link_tensor: torch.Tensor, signal: torch.Tensor) -> torch.Tensor:
    return link_tensor.reshape(link_tensor.shape + (1,) * (signal.ndim - 3))


def _to_numpy(value: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
