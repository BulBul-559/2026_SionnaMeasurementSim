"""NR PUSCH observation integration tests per doc 13."""

from pathlib import Path

import h5py
import numpy as np
import pytest


class TestNRPUSCHIntegration:
    def test_nr_pusch_hdf5_has_link_group(self):
        results_path = Path("outputs/e2e_nr_pusch_rx/results.h5")
        if not results_path.exists():
            pytest.skip("NR PUSCH output not yet generated")
        with h5py.File(results_path, "r") as h5:
            assert h5["link/duplex_mode"][()].decode() == "tdd"
            assert bool(h5["link/reciprocity_applied"][()])

    def test_nr_pusch_waveform_standard(self):
        results_path = Path("outputs/e2e_nr_pusch_rx/results.h5")
        if not results_path.exists():
            pytest.skip("NR PUSCH output not yet generated")
        with h5py.File(results_path, "r") as h5:
            std = h5["waveform/standard"][()].decode()
            assert std == "nr_pusch"

    def test_nr_pusch_receiver_type(self):
        results_path = Path("outputs/e2e_nr_pusch_rx/results.h5")
        if not results_path.exists():
            pytest.skip("NR PUSCH output not yet generated")
        with h5py.File(results_path, "r") as h5:
            rt = h5["receiver/receiver_type"][()].decode()
            assert rt == "pusch_receiver"

    def test_nr_pusch_evaluation_ber_bler_exist(self):
        results_path = Path("outputs/e2e_nr_pusch_rx/results.h5")
        if not results_path.exists():
            pytest.skip("NR PUSCH output not yet generated")
        with h5py.File(results_path, "r") as h5:
            ber = h5["evaluation/ber"][()]
            bler = h5["evaluation/bler"][()]
            assert np.isfinite(float(ber))
            assert np.isfinite(float(bler))

    def test_nr_pusch_nmse_db_exists(self):
        results_path = Path("outputs/e2e_nr_pusch_rx/results.h5")
        if not results_path.exists():
            pytest.skip("NR PUSCH output not yet generated")
        with h5py.File(results_path, "r") as h5:
            nmse = h5["evaluation/nmse_db"][()]
            assert np.all(np.isfinite(nmse))
