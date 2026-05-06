import json
from pathlib import Path

import h5py
import pytest


class TestCalibrationPipeline:
    def test_observation_output_includes_calibration_group(self):
        output_dir = Path("outputs/phase7_calibration")
        results_h5 = output_dir / "results.h5"
        if not results_h5.exists():
            pytest.skip("Phase 7 calibration output not yet generated")
        with h5py.File(results_h5, "r") as h5:
            assert "calibration" in h5
            cg = h5["calibration"]
            assert cg["profile_id"][()].decode() == "synthetic_default"
            fitted = json.loads(cg["fitted_parameters"][()].decode())
            assert "correction_mode" in fitted

    def test_manifest_includes_diagnostics(self):
        output_dir = Path("outputs/phase7_calibration")
        manifest_path = output_dir / "manifest.json"
        if not manifest_path.exists():
            pytest.skip("Phase 7 calibration output not yet generated")
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert "diagnostics" in manifest
        diag = manifest["diagnostics"]
        for key in (
            "median_nmse_db",
            "median_snr_db",
            "detection_rate",
            "estimation_failure_rate",
            "num_links",
        ):
            assert key in diag, f"Missing key {key} in diagnostics"


class TestCalibrationDomain:
    def test_synthetic_default_roundtrip(self):
        from sionna_measurement_sim.domain.observation import CalibrationResult

        c = CalibrationResult.synthetic_default()
        assert c.profile_id == "synthetic_default"
        # Verify JSON is valid
        json.loads(c.fitted_parameters)
        json.loads(c.validation_metrics)


class TestDiagnosticsSmoke:
    def test_diagnostics_from_phase7_manifest(self):
        output_dir = Path("outputs/phase7_calibration")
        manifest_path = output_dir / "manifest.json"
        if not manifest_path.exists():
            pytest.skip("Phase 7 calibration output not yet generated")
        with open(manifest_path) as f:
            manifest = json.load(f)
        diag = manifest["diagnostics"]
        assert diag["median_nmse_db"] < -20
        assert diag["detection_rate"] == 1.0
        assert diag["estimation_failure_rate"] == 0.0
        assert diag["num_failed_links"] == 0
