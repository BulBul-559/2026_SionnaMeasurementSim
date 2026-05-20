import numpy as np
import pytest

from sionna_measurement_sim.domain.derived import SPEED_OF_LIGHT_MPS
from sionna_measurement_sim.domain.frequency import FrequencyGrid
from sionna_measurement_sim.ranging.config import (
    PdpPeakRangingConfig,
    PhaseSlopeRangingConfig,
    RangingConfig,
)
from sionna_measurement_sim.ranging.estimators import estimate_pdp_peak, estimate_phase_slope
from sionna_measurement_sim.ranging.runner import run_ranging_observation
from sionna_measurement_sim.rt.truth_pipeline import RTTruthRunConfig, run_rt_truth_pipeline


def _grid(num_subcarriers: int = 64) -> FrequencyGrid:
    return FrequencyGrid.from_center_bandwidth(
        center_frequency_hz=3.5e9,
        bandwidth_hz=20e6,
        num_subcarriers=num_subcarriers,
    )


def _single_path_cfr(delay_s: float, *, amplitude: complex = 1.0 + 0.0j) -> np.ndarray:
    frequencies = _grid().frequencies_hz
    h = amplitude * np.exp(-1j * 2.0 * np.pi * frequencies * delay_s)
    return h[np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :].astype(np.complex64)


def test_pdp_peak_single_path_toa_matches_truth():
    frequency = _grid()
    delay_s = 8.0 / (frequency.subcarrier_spacing_hz * 64 * 8)
    cfr = _single_path_cfr(delay_s)

    result = estimate_pdp_peak(
        cfr,
        frequency,
        np.array([[delay_s]], dtype=np.float32),
        PdpPeakRangingConfig(min_peak_snr_db=0.0),
    )

    assert result.detection_success[0, 0, 0]
    assert result.toa_est_s[0, 0, 0] == pytest.approx(delay_s, abs=2e-9)
    assert result.one_way_range_est_m[0, 0, 0] == pytest.approx(
        delay_s * SPEED_OF_LIGHT_MPS,
        abs=0.7,
    )


def test_phase_slope_single_path_toa_matches_truth():
    frequency = _grid()
    delay_s = 50e-9
    cfr = _single_path_cfr(delay_s)

    result = estimate_phase_slope(
        cfr,
        frequency,
        np.array([[delay_s]], dtype=np.float32),
        PhaseSlopeRangingConfig(),
    )

    assert result.detection_success[0, 0, 0]
    assert result.toa_est_s[0, 0, 0] == pytest.approx(delay_s, abs=1e-9)
    assert abs(result.range_error_m[0, 0, 0]) < 0.5


def test_pdp_peak_two_path_prefers_detectable_earliest_peak():
    frequency = _grid()
    delay_step = 1.0 / (frequency.subcarrier_spacing_hz * 64 * 8)
    first = 8.0 * delay_step
    second = 20.0 * delay_step
    frequencies = frequency.frequencies_hz
    h = (
        np.exp(-1j * 2.0 * np.pi * frequencies * first)
        + 2.0 * np.exp(-1j * 2.0 * np.pi * frequencies * second)
    )
    cfr = h[np.newaxis, np.newaxis, np.newaxis, np.newaxis, np.newaxis, :].astype(np.complex64)

    result = estimate_pdp_peak(
        cfr,
        frequency,
        np.array([[first]], dtype=np.float32),
        PdpPeakRangingConfig(
            min_peak_snr_db=0.0,
            relative_threshold_db=-12.0,
            window="rect",
        ),
    )

    assert result.detection_success[0, 0, 0]
    assert result.toa_est_s[0, 0, 0] == pytest.approx(first, abs=2e-9)


def test_ranging_failure_outputs_nan_and_false_success():
    frequency = _grid()
    cfr = np.zeros((1, 1, 1, 1, 1, 64), dtype=np.complex64)

    result = estimate_pdp_peak(
        cfr,
        frequency,
        np.array([[10e-9]], dtype=np.float32),
        PdpPeakRangingConfig(),
    )

    assert not result.detection_success[0, 0, 0]
    assert np.isnan(result.toa_est_s[0, 0, 0])
    assert result.selected_delay_bin[0, 0, 0] == -1


def test_ranging_runner_requires_observation_when_enabled():
    with pytest.raises(ValueError, match="/observation/cfr_est"):
        run_ranging_observation(
            observation=None,
            frequency=_grid(),
            derived=object(),
            config=RangingConfig(enabled=True),
        )


def test_pipeline_fail_fast_when_ranging_enabled_without_phy(tmp_path):
    with pytest.raises(ValueError, match="/observation/cfr_est"):
        run_rt_truth_pipeline(
            RTTruthRunConfig(
                label_file=tmp_path / "unused_label.json",
                scene_file=tmp_path / "unused_scene.xml",
                output_dir=tmp_path / "unused",
                observation_snr_db=None,
                ranging_config=RangingConfig(enabled=True),
            )
        )
