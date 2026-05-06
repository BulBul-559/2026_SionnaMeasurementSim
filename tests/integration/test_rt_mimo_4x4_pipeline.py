"""4x4 MIMO integration tests per doc 14 requirements."""

from pathlib import Path

import h5py
import numpy as np


class TestMIMO4x4Pipeline:
    def test_4x4_mimo_cfr_shape(self):
        output_dir = Path("outputs/e2e_4x4_final")
        results_h5 = output_dir / "results.h5"
        if not results_h5.exists():
            import pytest
            pytest.skip("4x4 MIMO output not yet generated")
        with h5py.File(results_h5, "r") as h5:
            cfr = h5["channel/truth/cfr"]
            # [tx, rx, rx_ant, tx_ant, subcarrier]
            assert cfr.shape[2] == 4, f"rx_ant=4 expected, got {cfr.shape}"
            assert cfr.shape[3] == 4, f"tx_ant=4 expected, got {cfr.shape}"

    def test_4x4_mimo_cir_shape(self):
        output_dir = Path("outputs/e2e_4x4_final")
        results_h5 = output_dir / "results.h5"
        if not results_h5.exists():
            import pytest
            pytest.skip("4x4 MIMO output not yet generated")
        with h5py.File(results_h5, "r") as h5:
            cir = h5["channel/truth/cir_coefficients"]
            # [snapshot, tx, rx, rx_ant, tx_ant, path]
            assert cir.shape[3] == 4
            assert cir.shape[4] == 4

    def test_4x4_mimo_path_samples_antenna_indices(self):
        output_dir = Path("outputs/e2e_4x4_final")
        results_h5 = output_dir / "results.h5"
        if not results_h5.exists():
            import pytest
            pytest.skip("4x4 MIMO output not yet generated")
        with h5py.File(results_h5, "r") as h5:
            rx_ant = h5["paths/samples/sampled_rx_ant_indices"][()]
            tx_ant = h5["paths/samples/sampled_tx_ant_indices"][()]
            assert rx_ant.max() == 3  # 0-3 for 4 antennas
            assert tx_ant.max() == 3

    def test_4x4_mimo_los_nlos_aggregation(self):
        output_dir = Path("outputs/e2e_4x4_final")
        results_h5 = output_dir / "results.h5"
        if not results_h5.exists():
            import pytest
            pytest.skip("4x4 MIMO output not yet generated")
        with h5py.File(results_h5, "r") as h5:
            los = h5["channel/truth/los_exists"][()]
            nlos = h5["channel/truth/nlos_exists"][()]
            # Aggregated over all antenna pairs: shape [tx, rx]
            assert los.ndim == 2
            assert nlos.ndim == 2
            assert np.any(los) or np.any(nlos)

    def test_4x4_mimo_orientation_consistent(self):
        output_dir = Path("outputs/e2e_4x4_final")
        results_h5 = output_dir / "results.h5"
        if not results_h5.exists():
            import pytest
            pytest.skip("4x4 MIMO output not yet generated")
        with h5py.File(results_h5, "r") as h5:
            tx_mode = h5["antenna/tx_orientation_mode"][()].decode()
            rx_mode = h5["antenna/rx_orientation_mode"][()].decode()
            assert tx_mode in ("fixed", "look_at_first_peer", "look_at_centroid")
            assert rx_mode in ("fixed", "look_at_first_peer", "look_at_centroid")
