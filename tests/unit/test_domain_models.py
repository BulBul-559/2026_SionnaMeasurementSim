import numpy as np
import pytest

from sionna_measurement_sim.domain.frequency import FrequencyGrid
from sionna_measurement_sim.domain.results import create_phase1_minimal_result


def test_frequency_grid_is_strictly_increasing():
    grid = FrequencyGrid.from_center_bandwidth(3.5e9, 20e6, 8)

    assert grid.frequencies_hz.shape == (8,)
    assert np.all(np.diff(grid.frequencies_hz) > 0)
    assert grid.subcarrier_spacing_hz == pytest.approx(2.5e6)


def test_frequency_grid_rejects_unsorted_values():
    with pytest.raises(ValueError, match="strictly increasing"):
        FrequencyGrid(3.5e9, 20e6, np.array([2.0, 1.0]))


def test_phase1_minimal_result_shapes_match_contract():
    result = create_phase1_minimal_result()

    assert result.truth.cfr.shape == (1, 1, 1, 1, 8)
    assert result.truth.cfr.dtype == np.complex64
    assert result.devices.tx_velocity_mps.shape == (1, 1, 3)
    assert result.path_samples.vertices_m.shape == (0, 0, 0, 3)
