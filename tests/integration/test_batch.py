import json
from pathlib import Path

import pytest

from sionna_measurement_sim.domain.batch import (
    BatchConfig,
    BatchExperimentResult,
    BatchManifestEntry,
)


class TestBatchDomain:
    def test_batch_config_default(self):
        bc = BatchConfig()
        assert bc.total_batches == 1
        assert not bc.enabled

    def test_batch_config_multi(self):
        bc = BatchConfig(enabled=True, total_batches=4, completed_batches=3, failed_batches=1)
        assert bc.enabled
        assert bc.total_batches == 4

    def test_manifest_entry_to_dict(self):
        entry = BatchManifestEntry(
            batch_index=1, batch_id="batch_001", status="completed",
            results_h5="outputs/batch_001/results.h5",
        )
        d = entry.to_dict()
        assert d["batch_index"] == 1
        assert d["status"] == "completed"

    def test_manifest_entry_failed(self):
        entry = BatchManifestEntry(
            batch_index=0, batch_id="batch_000", status="failed",
            results_h5="", error_message="out of memory",
        )
        d = entry.to_dict()
        assert d["status"] == "failed"
        assert d["error_message"] == "out of memory"

    def test_experiment_result_succeeded_failed(self):
        result = BatchExperimentResult(
            batch_config=BatchConfig(enabled=True, total_batches=3),
            entries=[
                BatchManifestEntry(0, "b0", "completed", "r0.h5"),
                BatchManifestEntry(1, "b1", "completed", "r1.h5"),
                BatchManifestEntry(2, "b2", "failed", "", "error"),
            ],
        )
        assert result.succeeded == 2
        assert result.failed == 1
        d = result.to_manifest_dict()
        assert d["batching"]["completed_batches"] == 2
        assert d["batching"]["failed_batches"] == 1
        assert len(d["batches"]) == 3


class TestBatchOutput:
    def test_batch_manifest_exists(self):
        manifest_path = Path("outputs/phase8_batch/batch_manifest.json")
        if not manifest_path.exists():
            pytest.skip("Phase 8 batch output not yet generated")
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert manifest["batching"]["enabled"]
        assert manifest["batching"]["total_batches"] == 2
        assert manifest["batching"]["completed_batches"] == 2
        assert manifest["batching"]["failed_batches"] == 0
        assert len(manifest["batches"]) == 2

    def test_each_batch_h5_valid(self):
        from sionna_measurement_sim.io.schema_validator import validate_hdf5_contract

        for i in range(2):
            results_h5 = Path(f"outputs/phase8_batch/batch_{i:03d}/results.h5")
            if not results_h5.exists():
                pytest.skip(f"Batch {i} output not yet generated")
            validate_hdf5_contract(results_h5)

    def test_batch_manifest_matches_output(self):
        manifest_path = Path("outputs/phase8_batch/batch_manifest.json")
        if not manifest_path.exists():
            pytest.skip("Phase 8 batch output not yet generated")
        with open(manifest_path) as f:
            manifest = json.load(f)
        for batch_entry in manifest["batches"]:
            results_h5 = Path(batch_entry["results_h5"])
            assert results_h5.exists(), f"Missing: {results_h5}"
