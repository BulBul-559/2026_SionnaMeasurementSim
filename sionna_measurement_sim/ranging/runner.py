"""Pipeline-facing ranging runner."""

from __future__ import annotations

from sionna_measurement_sim.domain.derived import DerivedLabels
from sionna_measurement_sim.domain.frequency import FrequencyGrid
from sionna_measurement_sim.domain.observation import ObservationResult
from sionna_measurement_sim.ranging.config import RangingConfig
from sionna_measurement_sim.ranging.estimators import estimate_pdp_peak, estimate_phase_slope
from sionna_measurement_sim.ranging.result import RangingResult


def run_ranging_observation(
    *,
    observation: ObservationResult | None,
    frequency: FrequencyGrid,
    derived: DerivedLabels,
    config: RangingConfig,
) -> RangingResult | None:
    """Run enabled ranging estimators from `/observation/cfr_est`."""

    if not config.enabled:
        return None
    if config.source != "cfr_est":
        msg = "only ranging.source='cfr_est' is supported"
        raise ValueError(msg)
    if observation is None:
        msg = "ranging.enabled=true requires PHY observation with /observation/cfr_est"
        raise ValueError(msg)

    pdp_peak = None
    phase_slope = None
    if "pdp_peak" in config.estimators:
        pdp_peak = estimate_pdp_peak(
            observation.cfr_est,
            frequency,
            derived.first_path_delay_s,
            config.pdp_peak,
            write_rtt_equivalent=config.write_rtt_equivalent,
        )
    if "phase_slope" in config.estimators:
        phase_slope = estimate_phase_slope(
            observation.cfr_est,
            frequency,
            derived.first_path_delay_s,
            config.phase_slope,
            write_rtt_equivalent=config.write_rtt_equivalent,
        )

    return RangingResult(
        default_estimator=config.default_estimator,
        pdp_peak=pdp_peak,
        phase_slope=phase_slope,
    )

