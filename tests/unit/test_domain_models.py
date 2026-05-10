import numpy as np
import pytest

from sionna_measurement_sim.domain.frequency import FrequencyGrid
from sionna_measurement_sim.domain.channel import RTTruthResult
from sionna_measurement_sim.domain.derived import SPEED_OF_LIGHT_MPS, build_derived_labels
from sionna_measurement_sim.domain.path import PathTable
from sionna_measurement_sim.domain.results import create_phase1_minimal_result
from sionna_measurement_sim.domain.topology import Topology


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
    assert result.derived.geometric_distance_m.shape == (1, 1)


def test_derived_labels_select_paths_globally_over_antennas():
    topology = Topology(
        tx_positions_m=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        rx_positions_m=np.array([[3.0, 4.0, 12.0]], dtype=np.float32),
        tx_labels=("tx0",),
        rx_labels=("rx0",),
    )
    truth = RTTruthResult(
        cfr=np.ones((1, 1, 2, 2, 4), dtype=np.complex64),
        path_power_db=np.array([[-7.0]], dtype=np.float32),
        has_geometric_signal=np.array([[True]], dtype=np.bool_),
        geometric_path_count=np.array([[3]], dtype=np.int32),
        los_exists=np.array([[True]], dtype=np.bool_),
        nlos_exists=np.array([[True]], dtype=np.bool_),
    )
    shape = (1, 1, 2, 2, 3)
    valid = np.zeros(shape, dtype=np.bool_)
    valid[0, 0, 1, 0, 2] = True  # LoS, not first or strongest.
    valid[0, 0, 0, 1, 1] = True  # First path across antenna pairs.
    valid[0, 0, 1, 1, 0] = True  # Strongest path across antenna pairs.
    tau_s = np.zeros(shape, dtype=np.float32)
    tau_s[0, 0, 1, 0, 2] = 3e-9
    tau_s[0, 0, 0, 1, 1] = 1e-9
    tau_s[0, 0, 1, 1, 0] = 2e-9
    coeff = np.zeros(shape, dtype=np.complex64)
    coeff[0, 0, 1, 0, 2] = 0.2 + 0j
    coeff[0, 0, 0, 1, 1] = 0.1 + 0j
    coeff[0, 0, 1, 1, 0] = 10.0 + 0j
    theta_r = np.zeros(shape, dtype=np.float32)
    phi_r = np.zeros(shape, dtype=np.float32)
    theta_r[0, 0, 1, 0, 2], phi_r[0, 0, 1, 0, 2] = 0.7, 0.8
    theta_r[0, 0, 0, 1, 1], phi_r[0, 0, 0, 1, 1] = 1.7, 1.8
    theta_r[0, 0, 1, 1, 0], phi_r[0, 0, 1, 1, 0] = 2.7, 2.8
    path_type = np.empty(shape, dtype=object)
    path_type[:] = "invalid"
    path_type[0, 0, 1, 0, 2] = "los"
    path_type[0, 0, 0, 1, 1] = "reflection"
    path_type[0, 0, 1, 1, 0] = "reflection"
    table = PathTable(
        valid=valid,
        a=coeff,
        tau_s=tau_s,
        doppler_hz=np.zeros(shape, dtype=np.float32),
        theta_t_rad=np.zeros(shape, dtype=np.float32),
        phi_t_rad=np.zeros(shape, dtype=np.float32),
        theta_r_rad=theta_r,
        phi_r_rad=phi_r,
        interaction_type=np.zeros((*shape, 1), dtype=np.uint32),
        object_id=np.zeros((*shape, 1), dtype=np.uint32),
        primitive_id=np.zeros((*shape, 1), dtype=np.uint32),
        vertices_m=np.zeros((*shape, 1, 3), dtype=np.float32),
        path_type=path_type,
        path_depth=np.zeros(shape, dtype=np.int32),
    )

    derived = build_derived_labels(topology, truth, table)

    assert derived.geometric_distance_m[0, 0] == pytest.approx(13.0)
    assert derived.tx_rx_distance_m[0, 0] == pytest.approx(5.0)
    assert derived.tx_rx_bearing_rad[0, 0] == pytest.approx(np.arctan2(4.0, 3.0))
    assert derived.tx_rx_midpoint_m[0, 0].tolist() == pytest.approx([1.5, 2.0])
    assert derived.first_path_delay_s[0, 0] == pytest.approx(1e-9)
    assert derived.strongest_path_delay_s[0, 0] == pytest.approx(2e-9)
    assert derived.rtt_like_s[0, 0] == pytest.approx(1e-9)
    assert derived.rtt_like_m[0, 0] == pytest.approx(1e-9 * SPEED_OF_LIGHT_MPS)
    assert derived.los_distance_m[0, 0] == pytest.approx(3e-9 * SPEED_OF_LIGHT_MPS)
    assert derived.first_path_aoa_azimuth_rad[0, 0] == pytest.approx(1.8)
    assert derived.first_path_aoa_zenith_rad[0, 0] == pytest.approx(1.7)
    assert derived.strongest_aoa_azimuth_rad[0, 0] == pytest.approx(2.8)
    assert derived.strongest_aoa_zenith_rad[0, 0] == pytest.approx(2.7)
    assert derived.los_aoa_azimuth_rad[0, 0] == pytest.approx(0.8)
    assert derived.los_aoa_zenith_rad[0, 0] == pytest.approx(0.7)
