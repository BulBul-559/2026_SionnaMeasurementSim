"""CIR HDF5 schema validation tests per doc 14."""

from pathlib import Path

import h5py
import pytest

from sionna_measurement_sim.io.schema_validator import (
    validate_hdf5_contract,
)


class TestCIRSchema:
    def test_cir_datasets_exist_in_4x4_mimo_output(self):
        results_path = Path("outputs/e2e_4x4_final/results.h5")
        if not results_path.exists():
            pytest.skip("4x4 MIMO output not yet generated")
        with h5py.File(results_path, "r") as h5:
            for name in ("cir_coefficients", "cir_delays_s", "cir_valid"):
                assert f"channel/truth/{name}" in h5

    def test_cir_rank_6(self):
        results_path = Path("outputs/e2e_4x4_final/results.h5")
        if not results_path.exists():
            pytest.skip("4x4 MIMO output not yet generated")
        with h5py.File(results_path, "r") as h5:
            for name in ("cir_coefficients", "cir_delays_s", "cir_valid"):
                ds = h5[f"channel/truth/{name}"]
                assert ds.ndim == 6

    def test_cir_shape_consistency(self):
        results_path = Path("outputs/e2e_4x4_final/results.h5")
        if not results_path.exists():
            pytest.skip("4x4 MIMO output not yet generated")
        with h5py.File(results_path, "r") as h5:
            coeff = h5["channel/truth/cir_coefficients"]
            delays = h5["channel/truth/cir_delays_s"]
            valid = h5["channel/truth/cir_valid"]
            assert coeff.shape == delays.shape == valid.shape

    def test_cir_dtype(self):
        results_path = Path("outputs/e2e_4x4_final/results.h5")
        if not results_path.exists():
            pytest.skip("4x4 MIMO output not yet generated")
        with h5py.File(results_path, "r") as h5:
            assert h5["channel/truth/cir_coefficients"].dtype.kind == "c"
            assert h5["channel/truth/cir_delays_s"].dtype.kind == "f"

    def test_cir_validates(self):
        results_path = Path("outputs/e2e_4x4_final/results.h5")
        if not results_path.exists():
            pytest.skip("4x4 MIMO output not yet generated")
        validate_hdf5_contract(results_path)
