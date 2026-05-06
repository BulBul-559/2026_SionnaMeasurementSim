import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from sionna_measurement_sim.domain.path import PathSamples
from sionna_measurement_sim.domain.results import create_phase1_minimal_result
from sionna_measurement_sim.io.hdf5_reader import read_metadata, read_truth_cfr
from sionna_measurement_sim.io.hdf5_writer import write_measurement_result
from sionna_measurement_sim.io.schema_validator import SchemaValidationError, validate_hdf5_contract


def test_phase1_schema_stack_does_not_import_sionna():
    code = (
        "import sys;"
        "import sionna_measurement_sim.domain.results;"
        "import sionna_measurement_sim.io.hdf5_writer;"
        "import sionna_measurement_sim.io.hdf5_reader;"
        "print('sionna' in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "False"


def test_write_and_validate_minimal_phase1_hdf5(tmp_path: Path):
    output_path = tmp_path / "results.h5"
    write_measurement_result(output_path, create_phase1_minimal_result())

    validate_hdf5_contract(output_path)

    with h5py.File(output_path, "r") as h5:
        assert "channel/cfr" not in h5
        assert h5["meta/schema_version"][()].decode("utf-8") == "1.0.0"
        assert h5["meta/contract_name"][()].decode("utf-8") == "sionna_measurement_sim_hdf5"
        assert h5["meta/index_order"][()].decode("utf-8") == "tx,rx,rx_ant,tx_ant,..."
        assert h5["meta/unit_convention"][()].decode("utf-8") == "si_mks"
        assert h5["topology/tx_positions_m"].dtype == np.dtype("float32")
        assert h5["topology/rx_positions_m"].shape == (1, 3)
        assert h5["antenna/tx_polarization"][()].decode("utf-8") == "single"
        assert h5["antenna/rx_polarization"][()].decode("utf-8") == "single"
        assert h5["frequency/frequencies_hz"].dtype == np.dtype("float64")
        assert h5["channel/truth/cfr"].shape == (1, 1, 1, 1, 8)
        assert h5["channel/truth/cfr"].dtype == np.dtype("complex64")
        assert h5["paths/samples/vertices_m"].shape == (0, 0, 0, 3)
        assert h5["paths/samples/doppler_hz"].shape == (0, 0)
        assert h5["topology/tx_positions_m"].attrs["unit"] == "m"
        assert h5["frequency/frequencies_hz"].attrs["unit"] == "Hz"


def test_readback_preserves_metadata_and_truth_cfr(tmp_path: Path):
    output_path = tmp_path / "results.h5"
    write_measurement_result(output_path, create_phase1_minimal_result())

    metadata = read_metadata(output_path)
    cfr = read_truth_cfr(output_path)

    assert metadata["schema_version"] == "1.0.0"
    assert metadata["config_snapshot"]
    assert cfr.shape == (1, 1, 1, 1, 8)
    assert cfr.dtype == np.dtype("complex64")


def test_validator_rejects_missing_schema_version(tmp_path: Path):
    output_path = tmp_path / "missing_schema.h5"
    with h5py.File(output_path, "w") as h5:
        h5.create_group("meta")

    with pytest.raises(SchemaValidationError, match="/meta/schema_version"):
        validate_hdf5_contract(output_path)


def test_validator_rejects_forbidden_channel_cfr(tmp_path: Path):
    output_path = tmp_path / "results.h5"
    write_measurement_result(output_path, create_phase1_minimal_result())
    with h5py.File(output_path, "a") as h5:
        h5["channel"].create_dataset("cfr", data=np.zeros((1,), dtype=np.complex64))

    with pytest.raises(SchemaValidationError, match="Forbidden dataset"):
        validate_hdf5_contract(output_path)


def test_validator_rejects_observation_shape_mismatch(tmp_path: Path):
    output_path = tmp_path / "results.h5"
    write_measurement_result(output_path, create_phase1_minimal_result())
    with h5py.File(output_path, "a") as h5:
        h5.create_group("observation").create_dataset(
            "cfr_est",
            data=np.zeros((1, 1, 1, 1, 7), dtype=np.complex64),
        )

    with pytest.raises(SchemaValidationError, match="rank 6"):
        validate_hdf5_contract(output_path)


def test_path_sample_schema_requires_tx_rx_endpoints(tmp_path: Path):
    samples = PathSamples(
        sampled_link_indices=np.array([[0, 0]], dtype=np.int32),
        sampled_path_indices=np.array([[0]], dtype=np.int32),
        path_count=np.array([1], dtype=np.int32),
        path_gain_db=np.array([[-10.0]], dtype=np.float32),
        path_type=np.array([["reflection"]], dtype=object),
        vertices_m=np.zeros((1, 1, 3, 3), dtype=np.float32),
        vertex_count=np.array([[3]], dtype=np.int32),
        interaction_type=np.array([[[1]]], dtype=np.uint32),
        object_id=np.array([[[1]]], dtype=np.uint32),
        primitive_id=np.array([[[7]]], dtype=np.uint32),
        doppler_hz=np.array([[0.0]], dtype=np.float32),
        tau_s=np.array([[1e-9]], dtype=np.float32),
    )
    output_path = tmp_path / "results.h5"
    write_measurement_result(
        output_path,
        replace(create_phase1_minimal_result(), path_samples=samples),
    )
    validate_hdf5_contract(output_path)

    with h5py.File(output_path, "a") as h5:
        del h5["paths/samples/vertex_count"]
        h5["paths/samples"].create_dataset("vertex_count", data=np.array([[2]], dtype=np.int32))

    with pytest.raises(SchemaValidationError, match="TX/RX endpoints"):
        validate_hdf5_contract(output_path)
