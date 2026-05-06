"""Verify CIR extraction from Sionna RT via adapter."""

from pathlib import Path

import numpy as np

from sionna_measurement_sim.adapters.sionna_rt.rt_solver import (
    SionnaRTConfig,
    run_sionna_rt_truth,
)
from sionna_measurement_sim.domain.antenna import AntennaSpec
from sionna_measurement_sim.domain.frequency import FrequencyGrid
from sionna_measurement_sim.io.label_parser import load_topology_from_label


class TestCIRAdapter:
    def test_cir_shape_6d_4x4_mimo(self):
        topology = load_topology_from_label(
            Path("data/scenes/test/test5.json"), max_tx=2, max_rx=2,
        )
        antenna = AntennaSpec(
            tx_num_rows=2, tx_num_cols=2,
            rx_num_rows=2, rx_num_cols=2,
            tx_polarization="V", rx_polarization="V",
        )
        frequency = FrequencyGrid.from_center_bandwidth(3.5e9, 20e6, 8)
        result = run_sionna_rt_truth(
            topology, antenna, frequency,
            SionnaRTConfig(
                scene_file=Path("data/scenes/test/scene.xml"),
                seed=1, max_depth=1, specular_reflection=True,
                num_time_steps=1,
            ),
        )
        cir = result.cir_truth
        assert cir is not None
        # 6D: [snapshot, tx, rx, rx_ant, tx_ant, path]
        assert cir.coefficients.ndim == 6
        assert cir.coefficients.shape[0] == 1  # 1 snapshot
        assert cir.coefficients.shape[1] == 2  # tx
        assert cir.coefficients.shape[2] == 2  # rx
        assert cir.coefficients.shape[3] == 4  # rx_ant (2x2)
        assert cir.coefficients.shape[4] == 4  # tx_ant (2x2)
        assert cir.delays_s.shape == cir.coefficients.shape
        assert cir.valid.shape == cir.coefficients.shape
        assert np.all(np.isfinite(cir.delays_s))
        assert np.all(cir.delays_s >= 0)

    def test_cir_dtype(self):
        topology = load_topology_from_label(
            Path("data/scenes/test/test5.json"), max_tx=1, max_rx=1,
        )
        antenna = AntennaSpec(
            tx_polarization="V", rx_polarization="V",
        )
        frequency = FrequencyGrid.from_center_bandwidth(3.5e9, 20e6, 8)
        result = run_sionna_rt_truth(
            topology, antenna, frequency,
            SionnaRTConfig(
                scene_file=Path("data/scenes/test/scene.xml"),
                seed=1, max_depth=0,
            ),
        )
        cir = result.cir_truth
        assert cir.coefficients.dtype == np.complex64
        assert cir.delays_s.dtype == np.float32
        assert cir.valid.dtype == np.bool_
