"""Domain results for waveform-level ranging observations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sionna_measurement_sim.domain.validation import require_shape


@dataclass(frozen=True)
class PdpPeakResult:
    """Per-link output of the PDP peak ToA estimator."""

    toa_est_s: np.ndarray
    one_way_range_est_m: np.ndarray
    rtt_equiv_s: np.ndarray
    range_error_m: np.ndarray
    detection_success: np.ndarray
    selected_delay_bin: np.ndarray
    peak_power_linear: np.ndarray
    peak_snr_db: np.ndarray

    def __post_init__(self) -> None:
        _validate_common_shape(self)
        selected = np.asarray(self.selected_delay_bin, dtype=np.int32)
        require_shape("selected_delay_bin", selected, self.toa_est_s.shape)
        object.__setattr__(self, "selected_delay_bin", selected)


@dataclass(frozen=True)
class PhaseSlopeResult:
    """Per-link output of the phase-slope ToA estimator."""

    toa_est_s: np.ndarray
    one_way_range_est_m: np.ndarray
    rtt_equiv_s: np.ndarray
    range_error_m: np.ndarray
    detection_success: np.ndarray
    fit_residual_rad: np.ndarray

    def __post_init__(self) -> None:
        _validate_common_shape(self)


@dataclass(frozen=True)
class RangingResult:
    """Container for optional ranging estimator outputs."""

    default_estimator: str
    pdp_peak: PdpPeakResult | None = None
    phase_slope: PhaseSlopeResult | None = None

    def __post_init__(self) -> None:
        if self.default_estimator not in ("pdp_peak", "phase_slope"):
            msg = "default_estimator must be 'pdp_peak' or 'phase_slope'"
            raise ValueError(msg)
        if getattr(self, self.default_estimator) is None:
            msg = f"default estimator {self.default_estimator!r} result is missing"
            raise ValueError(msg)


def _validate_common_shape(result: PdpPeakResult | PhaseSlopeResult) -> None:
    toa = np.asarray(result.toa_est_s, dtype=np.float32)
    link_shape = toa.shape
    require_shape("toa_est_s", toa, (None, None, None))
    for name in (
        "one_way_range_est_m",
        "rtt_equiv_s",
        "range_error_m",
        "peak_power_linear",
        "peak_snr_db",
        "fit_residual_rad",
    ):
        if not hasattr(result, name):
            continue
        value = np.asarray(getattr(result, name), dtype=np.float32)
        require_shape(name, value, link_shape)
        object.__setattr__(result, name, value)
    detection = np.asarray(result.detection_success, dtype=np.bool_)
    require_shape("detection_success", detection, link_shape)
    object.__setattr__(result, "toa_est_s", toa)
    object.__setattr__(result, "detection_success", detection)
