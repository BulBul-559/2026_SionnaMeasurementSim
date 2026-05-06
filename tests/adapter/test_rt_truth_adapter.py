from pathlib import Path

import numpy as np

from sionna_measurement_sim.adapters.sionna_rt.path_adapter import INTERACTION_SPECULAR
from sionna_measurement_sim.adapters.sionna_rt.rt_solver import (
    SionnaRTConfig,
    run_sionna_rt_truth,
)
from sionna_measurement_sim.domain.antenna import AntennaSpec
from sionna_measurement_sim.domain.frequency import FrequencyGrid
from sionna_measurement_sim.io.label_parser import load_topology_from_label


def test_rt_truth_adapter_generates_tx_first_cfr():
    topology = load_topology_from_label(Path("data/scenes/test/test5.json"), max_tx=1, max_rx=1)
    antenna = AntennaSpec(tx_polarization="V", rx_polarization="V")
    frequency = FrequencyGrid.from_center_bandwidth(3.5e9, 20e6, 8)

    result = run_sionna_rt_truth(
        topology,
        antenna,
        frequency,
        SionnaRTConfig(scene_file=Path("data/scenes/test/scene.xml"), seed=1),
    )

    assert result.raw_cfr_shape == (1, 1, 1, 1, 1, 8)
    assert result.internal_cfr_shape == (1, 1, 1, 1, 8)
    assert result.truth.cfr.shape == (1, 1, 1, 1, 8)
    assert result.truth.cfr.dtype == np.dtype("complex64")
    assert np.any(np.isfinite(result.truth.cfr))
    assert not np.all(np.isnan(result.truth.cfr))
    assert result.truth.has_geometric_signal.shape == (1, 1)
    assert result.runtime_versions["sionna_rt"] == "2.0.1"


def test_path_adapter_extracts_path_samples_and_full_table():
    topology = load_topology_from_label(Path("data/scenes/test/test5.json"), max_tx=1, max_rx=1)
    antenna = AntennaSpec(tx_polarization="V", rx_polarization="V")
    frequency = FrequencyGrid.from_center_bandwidth(3.5e9, 20e6, 8)

    result = run_sionna_rt_truth(
        topology,
        antenna,
        frequency,
        SionnaRTConfig(
            scene_file=Path("data/scenes/test/scene.xml"),
            seed=1,
            max_depth=1,
            specular_reflection=True,
        ),
    )

    table = result.path_table
    samples = result.path_samples
    assert table.valid.shape == table.a.shape == table.tau_s.shape
    assert table.vertices_m.shape[-1] == 3
    assert table.interaction_type.shape == table.object_id.shape == table.primitive_id.shape
    assert table.interaction_type.shape[:-1] == table.valid.shape
    assert np.count_nonzero(table.valid) >= 1
    assert np.any(table.interaction_type == INTERACTION_SPECULAR)
    assert np.all(np.isfinite(table.doppler_hz[table.valid]))

    nlos_indices = np.argwhere((table.valid) & (table.path_depth > 0))
    assert nlos_indices.size > 0
    tx, rx, rx_ant, tx_ant, path = nlos_indices[0]
    assert np.all(np.isfinite(table.vertices_m[tx, rx, rx_ant, tx_ant, path, 0]))

    assert samples.sampled_link_indices.shape == (1, 2)
    assert samples.path_count[0] >= 1
    active = samples.vertex_count > 0
    interaction_count = np.count_nonzero(samples.interaction_type != 0, axis=-1)
    assert np.all(samples.vertex_count[active] >= interaction_count[active] + 2)
    assert np.any(samples.path_type == "reflection")
