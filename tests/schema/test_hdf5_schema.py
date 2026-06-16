import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from sionna_measurement_sim.domain.path import PathSamples
from sionna_measurement_sim.domain.results import ShardMetadata, create_phase1_minimal_result
from sionna_measurement_sim.io.hdf5_reader import (
    read_link_labels,
    read_metadata,
    read_truth_cfr,
)
from sionna_measurement_sim.io.hdf5_writer import (
    write_measurement_result,
    write_rt_labels_result,
)
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
        assert h5["meta/schema_version"][()].decode("utf-8") == "2.2.0"
        assert h5["meta/contract_name"][()].decode("utf-8") == "sionna_measurement_sim_hdf5"
        assert h5["meta/output_profile"][()].decode("utf-8") == "full"
        assert "meta/output_products" in h5
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
        assert "paths/full" not in h5
        assert h5["paths/nlos_truth/valid"].shape == (1, 1, 1, 1, 0)
        assert h5["paths/nlos_truth/aoa_zenith_rad"].attrs["unit"] == "rad"
        assert h5["paths/nlos_truth/path_type"].attrs["index_order"] == (
            "tx,rx,rx_ant,tx_ant,path"
        )
        assert h5["topology/tx_positions_m"].attrs["unit"] == "m"
        assert h5["frequency/frequencies_hz"].attrs["unit"] == "Hz"
        assert h5["scene/scene_id"][()].decode("utf-8") == "phase1_minimal"
        assert h5["scene/map_id"][()].decode("utf-8") == ""
        assert h5["derived/geometric_distance_m"].shape == (1, 1)
        assert h5["derived/first_path_propagation_range_m"].shape == (1, 1)
        assert "derived/rtt_like_m" not in h5
        assert "derived/rtt_like_s" not in h5
        assert h5["derived/tx_rx_midpoint_m"].shape == (1, 1, 2)
        assert h5["derived/path_selection_policy"][()].decode("utf-8")
        assert h5["derived/geometric_distance_m"][0, 0] == pytest.approx(5.0)
        assert h5["derived/tx_rx_distance_m"][0, 0] == pytest.approx(5.0)
        assert np.isnan(h5["derived/first_path_delay_s"][0, 0])


def test_write_and_validate_shard_metadata(tmp_path: Path):
    output_path = tmp_path / "result_0001.h5"
    result = replace(
        create_phase1_minimal_result(),
        shard=ShardMetadata(
            shard_index=1,
            shard_count=4,
            axis="rx",
            global_rx_start=12,
            global_rx_indices=np.array([12], dtype=np.int64),
            global_tx_indices=np.array([3], dtype=np.int64),
        ),
    )

    write_measurement_result(output_path, result)

    validate_hdf5_contract(output_path)
    with h5py.File(output_path, "r") as h5:
        assert h5["shard/shard_index"][()] == 1
        assert h5["shard/shard_count"][()] == 4
        assert h5["shard/axis"][()].decode("utf-8") == "rx"
        assert h5["shard/global_rx_start"][()] == 12
        np.testing.assert_array_equal(h5["shard/global_rx_indices"][()], np.array([12]))
        np.testing.assert_array_equal(h5["shard/global_tx_indices"][()], np.array([3]))


def test_write_measurement_result_can_disable_compression(tmp_path: Path):
    output_path = tmp_path / "results_uncompressed.h5"

    write_measurement_result(
        output_path,
        create_phase1_minimal_result(),
        compression="none",
    )

    validate_hdf5_contract(output_path)
    with h5py.File(output_path, "r") as h5:
        assert h5["channel/truth/cfr"].compression is None


def test_readback_preserves_metadata_and_truth_cfr(tmp_path: Path):
    output_path = tmp_path / "results.h5"
    write_measurement_result(output_path, create_phase1_minimal_result())

    metadata = read_metadata(output_path)
    cfr = read_truth_cfr(output_path)

    assert metadata["schema_version"] == "2.2.0"
    assert metadata["config_snapshot"]
    assert cfr.shape == (1, 1, 1, 1, 8)
    assert cfr.dtype == np.dtype("complex64")


def test_custom_cfr_truth_product_writes_minimal_truth_hdf5(tmp_path: Path):
    output_path = tmp_path / "cfr_truth_only.h5"
    base = create_phase1_minimal_result()
    result = replace(
        base,
        metadata=replace(
            base.metadata,
            output_profile="custom",
            output_products=("cfr_truth",),
        ),
        path_samples=None,
        cir_truth=None,
        derived=None,
        nlos_path_truth=None,
    )

    write_measurement_result(output_path, result)
    validate_hdf5_contract(output_path)

    with h5py.File(output_path, "r") as h5:
        assert h5["meta/output_profile"][()].decode("utf-8") == "custom"
        assert tuple(v.decode("utf-8") for v in h5["meta/output_products"][()]) == (
            "cfr_truth",
        )
        assert "channel/truth/cfr" in h5
        assert "channel/truth/cir_coefficients" not in h5
        assert "derived" not in h5
        assert "paths" not in h5
        assert "waveform" not in h5
        assert "observation" not in h5


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


def test_write_and_validate_rt_labels_only_hdf5(tmp_path: Path):
    from dataclasses import replace

    from sionna_measurement_sim.domain.constants import RT_LABELS_CONTRACT_NAME
    from sionna_measurement_sim.domain.results import RTCompactLinkLabels, RTLabelsOnlyResult

    full = create_phase1_minimal_result()
    derived = replace(
        full.derived,
        first_path_delay_s=np.array([[1.0e-8]], dtype=np.float32),
        first_path_propagation_range_m=np.array([[2.9979246]], dtype=np.float32),
        strongest_path_delay_s=np.array([[1.0e-8]], dtype=np.float32),
    )
    labels_result = RTLabelsOnlyResult(
        metadata=replace(
            full.metadata,
            contract_name=RT_LABELS_CONTRACT_NAME,
            output_profile="rt_labels_only",
        ),
        input_spec=full.input_spec,
        topology=full.topology,
        devices=full.devices,
        antenna=full.antenna,
        scene=full.scene,
        frequency=full.frequency,
        runtime=full.runtime,
        derived=derived,
        link_labels=RTCompactLinkLabels.from_topology(full.topology, derived),
        link=full.link,
    )
    output_path = tmp_path / "rt_labels.h5"

    write_rt_labels_result(output_path, labels_result)

    validate_hdf5_contract(output_path)
    with h5py.File(output_path, "r") as h5:
        assert h5["meta/contract_name"][()].decode("utf-8") == RT_LABELS_CONTRACT_NAME
        assert h5["meta/output_profile"][()].decode("utf-8") == "rt_labels_only"
        assert "channel" not in h5
        assert "paths" not in h5
        assert "waveform" not in h5
        assert h5["labels/link/link_index"].shape == (1,)
        assert h5["labels/link/tx_xy_m"].shape == (1, 2)

    labels = read_link_labels(output_path)
    assert labels["geometric_distance_m"].shape == (1,)
    assert labels["geometric_distance_m"][0] == pytest.approx(5.0)


def test_validator_rejects_legacy_rtt_like_fields(tmp_path: Path):
    output_path = tmp_path / "results.h5"
    write_measurement_result(output_path, create_phase1_minimal_result())
    with h5py.File(output_path, "a") as h5:
        h5["derived"].create_dataset("rtt_like_m", data=np.zeros((1, 1), dtype=np.float32))

    with pytest.raises(SchemaValidationError, match="Forbidden dataset"):
        validate_hdf5_contract(output_path)


@pytest.mark.parametrize(
    "dataset_path",
    ("array/spatial_spectrum_label", "array/spatial_spectrum_srs"),
)
def test_validator_rejects_legacy_array_alias_fields(
    tmp_path: Path,
    dataset_path: str,
):
    output_path = tmp_path / "results.h5"
    write_measurement_result(output_path, create_phase1_minimal_result())
    with h5py.File(output_path, "a") as h5:
        group_name, name = dataset_path.split("/", maxsplit=1)
        group = h5.require_group(group_name)
        group.create_dataset(name, data=np.zeros((1, 1, 1, 2, 2), dtype=np.float32))

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
        sampled_rx_ant_indices=np.array([0], dtype=np.int32),
        sampled_tx_ant_indices=np.array([0], dtype=np.int32),
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
