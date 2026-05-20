"""Configuration objects for waveform-level ranging estimators."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PdpPeakRangingConfig:
    """PDP peak ToA estimator configuration."""

    oversampling_factor: int = 8
    window: str = "hann"
    peak_policy: str = "earliest_above_relative_threshold"
    relative_threshold_db: float = -12.0
    min_peak_snr_db: float = 6.0
    interpolation: str = "parabolic_log_power"
    max_delay_s: float | None = None

    def __post_init__(self) -> None:
        if self.oversampling_factor < 1:
            msg = "pdp_peak.oversampling_factor must be >= 1"
            raise ValueError(msg)
        if self.window not in ("hann", "rect"):
            msg = "pdp_peak.window must be 'hann' or 'rect'"
            raise ValueError(msg)
        if self.peak_policy != "earliest_above_relative_threshold":
            msg = "only pdp_peak.peak_policy='earliest_above_relative_threshold' is supported"
            raise ValueError(msg)
        if self.interpolation not in ("parabolic_log_power", "none"):
            msg = "pdp_peak.interpolation must be 'parabolic_log_power' or 'none'"
            raise ValueError(msg)
        if self.max_delay_s is not None and self.max_delay_s <= 0:
            msg = "pdp_peak.max_delay_s must be positive when set"
            raise ValueError(msg)


@dataclass(frozen=True)
class PhaseSlopeRangingConfig:
    """Phase-slope ToA estimator configuration."""

    unwrap: bool = True
    aggregate: str = "power_weighted_median"
    min_mean_power: float = 1.0e-12

    def __post_init__(self) -> None:
        if self.aggregate != "power_weighted_median":
            msg = "only phase_slope.aggregate='power_weighted_median' is supported"
            raise ValueError(msg)
        if self.min_mean_power < 0:
            msg = "phase_slope.min_mean_power must be non-negative"
            raise ValueError(msg)


@dataclass(frozen=True)
class RangingConfig:
    """Top-level ranging observation configuration."""

    enabled: bool = False
    source: str = "cfr_est"
    estimators: tuple[str, ...] = ("pdp_peak", "phase_slope")
    default_estimator: str = "pdp_peak"
    write_rtt_equivalent: bool = True
    pdp_peak: PdpPeakRangingConfig = field(default_factory=PdpPeakRangingConfig)
    phase_slope: PhaseSlopeRangingConfig = field(default_factory=PhaseSlopeRangingConfig)

    def __post_init__(self) -> None:
        supported = {"pdp_peak", "phase_slope"}
        estimators = tuple(str(estimator) for estimator in self.estimators)
        unknown = sorted(set(estimators) - supported)
        if unknown:
            msg = f"unsupported ranging estimators: {unknown}"
            raise ValueError(msg)
        if self.source != "cfr_est":
            msg = "only ranging.source='cfr_est' is supported"
            raise ValueError(msg)
        if self.default_estimator not in estimators:
            msg = "ranging.default_estimator must be listed in ranging.estimators"
            raise ValueError(msg)
        object.__setattr__(self, "estimators", estimators)

