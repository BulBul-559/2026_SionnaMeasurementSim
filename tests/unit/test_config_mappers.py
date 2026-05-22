from __future__ import annotations

import pytest

from sionna_measurement_sim.config.mappers import to_domain_ranging_config
from sionna_measurement_sim.config.schema import (
    PdpPeakRangingConfig as YamlPdpPeakRangingConfig,
)
from sionna_measurement_sim.config.schema import (
    PhaseSlopeRangingConfig as YamlPhaseSlopeRangingConfig,
)
from sionna_measurement_sim.config.schema import (
    RangingConfig as YamlRangingConfig,
)
from sionna_measurement_sim.ranging.config import RangingConfig as DomainRangingConfig


def test_default_ranging_config_maps_to_domain_defaults():
    mapped = to_domain_ranging_config(YamlRangingConfig())

    assert mapped == DomainRangingConfig()


def test_custom_ranging_config_maps_all_fields():
    yaml_config = YamlRangingConfig(
        enabled=True,
        source="cfr_est",
        estimators=["phase_slope", "pdp_peak"],
        default_estimator="phase_slope",
        write_rtt_equivalent=False,
        pdp_peak=YamlPdpPeakRangingConfig(
            oversampling_factor=16,
            window="rect",
            relative_threshold_db=-9.0,
            min_peak_snr_db=4.0,
            interpolation="none",
            max_delay_s=2.5e-7,
        ),
        phase_slope=YamlPhaseSlopeRangingConfig(
            unwrap=False,
            min_mean_power=2.5e-11,
        ),
    )

    mapped = to_domain_ranging_config(yaml_config)

    assert mapped.enabled is True
    assert mapped.source == "cfr_est"
    assert mapped.estimators == ("phase_slope", "pdp_peak")
    assert mapped.default_estimator == "phase_slope"
    assert mapped.write_rtt_equivalent is False
    assert mapped.pdp_peak.oversampling_factor == 16
    assert mapped.pdp_peak.window == "rect"
    assert mapped.pdp_peak.relative_threshold_db == -9.0
    assert mapped.pdp_peak.min_peak_snr_db == 4.0
    assert mapped.pdp_peak.interpolation == "none"
    assert mapped.pdp_peak.max_delay_s == 2.5e-7
    assert mapped.phase_slope.unwrap is False
    assert mapped.phase_slope.min_mean_power == 2.5e-11


def test_ranging_mapper_propagates_invalid_source_to_domain_validation():
    yaml_config = _unsafe_yaml_ranging_config(source="truth_cfr")

    with pytest.raises(ValueError, match="source='cfr_est'"):
        to_domain_ranging_config(yaml_config)


def test_ranging_mapper_propagates_unknown_estimator_to_domain_validation():
    yaml_config = _unsafe_yaml_ranging_config(estimators=["pdp_peak", "music"])

    with pytest.raises(ValueError, match="unsupported ranging estimators"):
        to_domain_ranging_config(yaml_config)


def test_ranging_mapper_propagates_invalid_default_estimator_to_domain_validation():
    yaml_config = _unsafe_yaml_ranging_config(
        estimators=["phase_slope"],
        default_estimator="pdp_peak",
    )

    with pytest.raises(ValueError, match="default_estimator"):
        to_domain_ranging_config(yaml_config)


def _unsafe_yaml_ranging_config(**overrides: object) -> YamlRangingConfig:
    values = {
        "enabled": True,
        "source": "cfr_est",
        "estimators": ["pdp_peak", "phase_slope"],
        "default_estimator": "pdp_peak",
        "write_rtt_equivalent": True,
        "pdp_peak": YamlPdpPeakRangingConfig(),
        "phase_slope": YamlPhaseSlopeRangingConfig(),
    }
    values.update(overrides)
    return YamlRangingConfig.model_construct(**values)
