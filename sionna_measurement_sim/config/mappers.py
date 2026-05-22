"""Adapters from YAML-facing config models to runtime/domain config models."""

from __future__ import annotations

from sionna_measurement_sim.config.schema import RangingConfig as YamlRangingConfig
from sionna_measurement_sim.ranging.config import (
    PdpPeakRangingConfig,
    PhaseSlopeRangingConfig,
    RangingConfig,
)


def to_domain_ranging_config(config: YamlRangingConfig) -> RangingConfig:
    """Convert validated YAML ranging config to the runtime ranging dataclass."""

    return RangingConfig(
        enabled=config.enabled,
        source=config.source,
        estimators=tuple(config.estimators),
        default_estimator=config.default_estimator,
        write_rtt_equivalent=config.write_rtt_equivalent,
        pdp_peak=PdpPeakRangingConfig(
            oversampling_factor=config.pdp_peak.oversampling_factor,
            window=config.pdp_peak.window,
            peak_policy=config.pdp_peak.peak_policy,
            relative_threshold_db=config.pdp_peak.relative_threshold_db,
            min_peak_snr_db=config.pdp_peak.min_peak_snr_db,
            interpolation=config.pdp_peak.interpolation,
            max_delay_s=config.pdp_peak.max_delay_s,
        ),
        phase_slope=PhaseSlopeRangingConfig(
            unwrap=config.phase_slope.unwrap,
            aggregate=config.phase_slope.aggregate,
            min_mean_power=config.phase_slope.min_mean_power,
        ),
    )
