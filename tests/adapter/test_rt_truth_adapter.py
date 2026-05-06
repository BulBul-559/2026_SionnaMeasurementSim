from pathlib import Path

import numpy as np

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
