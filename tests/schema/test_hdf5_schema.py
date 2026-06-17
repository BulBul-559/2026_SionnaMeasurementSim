import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from sionna_measurement_sim.domain.constants import (
    IQ_LINK_LIBRARY_CONTRACT_NAME,
    RT_LABELS_CONTRACT_NAME,
)
from sionna_measurement_sim.domain.iq import IQObservationResult, LinkIQCapture
from sionna_measurement_sim.domain.link import LinkConfig
from sionna_measurement_sim.domain.observation import (
    EvaluationResult,
    ImpairmentSpec,
    ObservationResult,
    ReceiverSpec,
    WaveformSpec,
)
from sionna_measurement_sim.domain.path import PathSamples
from sionna_measurement_sim.domain.results import (
    IQLinkLibraryResult,
    RTCompactLinkLabels,
    RTLabelsOnlyResult,
    ShardMetadata,
    create_phase1_minimal_result,
)
from sionna_measurement_sim.io.hdf5_bundle_writer import HDF5ResultBundleWriter
from sionna_measurement_sim.io.hdf5_reader import (
    read_bundle_index,
    read_link_labels,
    read_metadata,
    read_truth_cfr,
)
from sionna_measurement_sim.io.hdf5_writer import (
    write_measurement_result,
    write_rt_labels_result,
)
from sionna_measurement_sim.io.schema_validator import SchemaValidationError, validate_hdf5_contract
from sionna_measurement_sim.phy.nr_pusch_observation import build_array_outputs_from_waveform


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
        assert h5["meta/schema_version"][()].decode("utf-8") == "2.3.0"
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
        assert h5["paths/nlos_truth/path_type"].attrs["index_order"] == ("tx,rx,rx_ant,tx_ant,path")
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


def test_write_measurement_result_mixed_compression_skips_noisy_grids(tmp_path: Path):
    output_path = tmp_path / "results_mixed.h5"
    base = create_phase1_minimal_result()
    link_shape = (1, 1, 1)
    cfr_est = base.truth.cfr[np.newaxis, ...]
    tx_grid = np.ones((1, 1, 1, 1, 14, 8), dtype=np.complex64)
    rx_grid = (2.0 * tx_grid).astype(np.complex64)
    result = replace(
        base,
        metadata=replace(
            base.metadata,
            output_profile="full",
            output_products=("cfr_truth", "cfr_obs", "array"),
        ),
        waveform=WaveformSpec(
            standard="nr_pusch",
            sample_rate_hz=240_000.0,
            fft_size=8,
            cp_length=0,
            num_ofdm_symbols=14,
            pilot_indices=np.array([], dtype=np.int32),
            data_subcarrier_indices=np.arange(8, dtype=np.int32),
            pilot_symbols=np.array([], dtype=np.complex64),
            tx_power_dbm=0.0,
        ),
        observation=ObservationResult(
            cfr_est=cfr_est,
            valid_mask=np.ones(link_shape, dtype=np.bool_),
            detection_success=np.ones(link_shape, dtype=np.bool_),
            estimation_success=np.ones(link_shape, dtype=np.bool_),
            snr_db=np.full(link_shape, 30.0, dtype=np.float32),
            rssi_dbm=np.zeros(link_shape, dtype=np.float32),
            noise_power_dbm=np.zeros(link_shape, dtype=np.float32),
            cfo_hz=np.zeros(link_shape, dtype=np.float32),
            sfo_ppm=np.zeros(link_shape, dtype=np.float32),
            timing_offset_samples=np.zeros(link_shape, dtype=np.float32),
            phase_offset_rad=np.zeros(link_shape, dtype=np.float32),
            agc_gain_db=np.zeros((1, 1), dtype=np.float32),
            clipping_flag=np.zeros(link_shape, dtype=np.bool_),
        ),
        receiver=ReceiverSpec(receiver_type="pusch_receiver", mimo_detector="lmmse"),
        evaluation=EvaluationResult(
            nmse_db=np.zeros(link_shape, dtype=np.float32),
            nmse_db_total=np.zeros(link_shape, dtype=np.float32),
            amplitude_error_db=np.zeros(link_shape, dtype=np.float32),
            phase_error_rad=np.zeros(link_shape, dtype=np.float32),
            correlation=np.ones(link_shape, dtype=np.float32),
            detection_rate=1.0,
            estimation_failure_rate=0.0,
            num_blocks=1,
        ),
        impairments=ImpairmentSpec(
            model_version="mixed_compression_test",
            random_seed=1,
            awgn_config='{"snr_db": 30.0}',
        ),
        waveform_extras={
            "num_prb": 1,
            "subcarrier_spacing_khz": 30,
            "num_layers": 1,
            "num_antenna_ports": 1,
            "mcs_index": 14,
            "mcs_table": 1,
            "dmrs_config_type": 1,
            "dmrs_length": 1,
            "dmrs_additional_position": 1,
            "num_cdm_groups_without_data": 2,
            "tx_grid": tx_grid,
            "rx_grid": rx_grid,
            "noise_variance": np.full(link_shape, 0.25, dtype=np.float32),
            "tx_power_dbm_per_port": np.zeros((1, 1, 1), dtype=np.float32),
            "tx_power_scale_linear": np.ones((1, 1, 1), dtype=np.float32),
            "serving_rx_index": np.zeros((1, 1), dtype=np.int32),
            "path_loss_db": np.zeros((1, 1), dtype=np.float32),
            "power_clipped_flag": np.zeros((1, 1, 1), dtype=np.bool_),
        },
        array_outputs=build_array_outputs_from_waveform(rx_grid),
    )

    write_measurement_result(output_path, result, compression="mixed", gzip_level=1)

    validate_hdf5_contract(output_path)
    with h5py.File(output_path, "r") as h5:
        assert h5["channel/truth/cfr"].compression == "gzip"
        assert h5["channel/truth/cfr"].compression_opts == 1
        assert h5["waveform/tx_grid"].compression == "gzip"
        assert h5["waveform/tx_grid"].compression_opts == 1
        assert h5["waveform/rx_grid"].compression is None
        assert h5["observation/cfr_est"].compression is None


def test_readback_preserves_metadata_and_truth_cfr(tmp_path: Path):
    output_path = tmp_path / "results.h5"
    write_measurement_result(output_path, create_phase1_minimal_result())

    metadata = read_metadata(output_path)
    cfr = read_truth_cfr(output_path)

    assert metadata["schema_version"] == "2.3.0"
    assert metadata["config_snapshot"]
    assert cfr.shape == (1, 1, 1, 1, 8)
    assert cfr.dtype == np.dtype("complex64")


def test_appendable_hdf5_bundle_writes_truth_shards(tmp_path: Path):
    base = create_phase1_minimal_result()
    first = replace(
        base,
        shard=ShardMetadata(
            shard_index=0,
            shard_count=2,
            axis="ue",
            global_rx_start=0,
            global_rx_indices=np.array([0], dtype=np.int64),
            global_tx_indices=np.array([0], dtype=np.int64),
        ),
    )
    second = replace(
        base,
        topology=replace(
            base.topology,
            tx_positions_m=base.topology.tx_positions_m
            + np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            tx_labels=("tx1",),
        ),
        truth=replace(base.truth, cfr=base.truth.cfr * np.complex64(2.0 + 0.0j)),
        shard=ShardMetadata(
            shard_index=1,
            shard_count=2,
            axis="ue",
            global_rx_start=0,
            global_rx_indices=np.array([0], dtype=np.int64),
            global_tx_indices=np.array([1], dtype=np.int64),
        ),
    )
    output_path = tmp_path / "bundles" / "bundle_000.h5"

    with HDF5ResultBundleWriter(output_path, compression="mixed", gzip_level=1) as bundle:
        bundle.append_result(first)
        bundle.append_result(second)

    validate_hdf5_contract(output_path)
    index = read_bundle_index(output_path)
    assert index["fragment_count"] == 2
    assert index["ue_count"] == 2
    np.testing.assert_array_equal(index["shard_offsets"], np.array([[0, 1], [1, 1]]))
    np.testing.assert_array_equal(index["global_ue_indices"], np.array([0, 1]))
    cfr = read_truth_cfr(output_path)
    assert cfr.shape == (2, 1, 1, 1, 8)
    np.testing.assert_allclose(cfr[0], base.truth.cfr[0])
    np.testing.assert_allclose(cfr[1], base.truth.cfr[0] * np.complex64(2.0 + 0.0j))
    with h5py.File(output_path, "r") as h5:
        assert h5["meta/contract_name"][()].decode("utf-8") == (
            "sionna_measurement_sim_bundle_hdf5"
        )
        assert h5["topology/tx_positions_m"].shape == (2, 3)
        assert h5["topology/rx_positions_m"].shape == (1, 3)
        assert h5["channel/truth/cfr"].maxshape[0] is None


def test_appendable_hdf5_bundle_appends_downlink_ue_rx_axis(tmp_path: Path):
    base = replace(create_phase1_minimal_result(), link=LinkConfig(phy_link_direction="downlink"))
    first = replace(
        base,
        shard=ShardMetadata(
            shard_index=0,
            shard_count=2,
            axis="ue",
            global_rx_start=0,
            global_rx_indices=np.array([0], dtype=np.int64),
            global_tx_indices=np.array([0], dtype=np.int64),
        ),
    )
    second = replace(
        base,
        topology=replace(
            base.topology,
            rx_positions_m=base.topology.rx_positions_m
            + np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            rx_labels=("ue1",),
        ),
        truth=replace(base.truth, cfr=base.truth.cfr * np.complex64(3.0 + 0.0j)),
        shard=ShardMetadata(
            shard_index=1,
            shard_count=2,
            axis="ue",
            global_rx_start=1,
            global_rx_indices=np.array([1], dtype=np.int64),
            global_tx_indices=np.array([0], dtype=np.int64),
        ),
    )
    output_path = tmp_path / "bundles" / "downlink_bundle.h5"

    with HDF5ResultBundleWriter(output_path, compression="mixed", gzip_level=1) as bundle:
        bundle.append_result(first)
        bundle.append_result(second)

    validate_hdf5_contract(output_path)
    index = read_bundle_index(output_path)
    assert index["ue_axis_role"] == "rx"
    assert index["ue_count"] == 2
    np.testing.assert_array_equal(index["global_ue_indices"], np.array([0, 1]))
    cfr = read_truth_cfr(output_path)
    assert cfr.shape == (1, 2, 1, 1, 8)
    np.testing.assert_allclose(cfr[:, 0], base.truth.cfr[:, 0])
    np.testing.assert_allclose(cfr[:, 1], base.truth.cfr[:, 0] * np.complex64(3.0 + 0.0j))
    with h5py.File(output_path, "r") as h5:
        assert h5["bundle/ue_axis_role"][()].decode("utf-8") == "rx"
        assert h5["link/phy_link_direction"][()].decode("utf-8") == "downlink"
        assert h5["link/tx_role"][()].decode("utf-8") == "bs"
        assert h5["link/rx_role"][()].decode("utf-8") == "ue"
        assert h5["topology/tx_positions_m"].shape == (1, 3)
        assert h5["topology/rx_positions_m"].shape == (2, 3)
        assert h5["topology/rx_labels"].shape == (2,)
        assert h5["channel/truth/cfr"].maxshape[1] is None


def test_appendable_hdf5_bundle_supports_rt_labels_contract(tmp_path: Path):
    first = _labels_only_bundle_fragment(global_tx_index=0, shard_index=0)
    second = _labels_only_bundle_fragment(global_tx_index=1, shard_index=1)
    output_path = tmp_path / "bundles" / "labels_bundle.h5"

    with HDF5ResultBundleWriter(output_path, compression="mixed", gzip_level=1) as bundle:
        bundle.append_result(first)
        bundle.append_result(second)

    validate_hdf5_contract(output_path)
    index = read_bundle_index(output_path)
    assert index["fragment_count"] == 2
    np.testing.assert_array_equal(
        index["global_ue_indices"],
        np.asarray([0, 1], dtype=np.int64),
    )
    with h5py.File(output_path, "r") as h5:
        assert h5["bundle/source_contract_name"][()].decode("utf-8") == (RT_LABELS_CONTRACT_NAME)
        assert h5["labels/link/link_index"].shape == (2,)
        np.testing.assert_array_equal(
            h5["labels/link/global_tx_index"][()],
            np.asarray([0, 1], dtype=np.int64),
        )
        assert h5["topology/tx_positions_m"].shape == (2, 3)


def test_appendable_hdf5_bundle_supports_rt_labels_downlink_rx_axis(tmp_path: Path):
    first = _labels_only_downlink_bundle_fragment(global_rx_index=0, shard_index=0)
    second = _labels_only_downlink_bundle_fragment(global_rx_index=1, shard_index=1)
    output_path = tmp_path / "bundles" / "labels_downlink_bundle.h5"

    with HDF5ResultBundleWriter(output_path, compression="mixed", gzip_level=1) as bundle:
        bundle.append_result(first)
        bundle.append_result(second)

    validate_hdf5_contract(output_path)
    index = read_bundle_index(output_path)
    assert index["ue_axis_role"] == "rx"
    np.testing.assert_array_equal(
        index["global_ue_indices"],
        np.asarray([0, 1], dtype=np.int64),
    )
    with h5py.File(output_path, "r") as h5:
        assert h5["bundle/source_contract_name"][()].decode("utf-8") == (RT_LABELS_CONTRACT_NAME)
        assert h5["topology/tx_positions_m"].shape == (1, 3)
        assert h5["topology/rx_positions_m"].shape == (2, 3)
        assert h5["labels/link/link_index"].shape == (2,)
        np.testing.assert_array_equal(
            h5["labels/link/global_rx_index"][()],
            np.asarray([0, 1], dtype=np.int64),
        )


def test_appendable_hdf5_bundle_supports_iq_link_library_contract(tmp_path: Path):
    first = _iq_link_library_bundle_fragment(global_tx_index=0, shard_index=0)
    second = _iq_link_library_bundle_fragment(global_tx_index=1, shard_index=1)
    output_path = tmp_path / "bundles" / "iq_bundle.h5"

    with HDF5ResultBundleWriter(output_path, compression="mixed", gzip_level=1) as bundle:
        bundle.append_result(first)
        bundle.append_result(second)

    validate_hdf5_contract(output_path)
    index = read_bundle_index(output_path)
    assert index["fragment_count"] == 2
    np.testing.assert_array_equal(
        index["global_ue_indices"],
        np.asarray([0, 1], dtype=np.int64),
    )
    with h5py.File(output_path, "r") as h5:
        assert h5["bundle/source_contract_name"][()].decode("utf-8") == (
            IQ_LINK_LIBRARY_CONTRACT_NAME
        )
        assert h5["iq/link/frequency_clean"].shape == (1, 2, 1, 1, 2, 8)
        assert h5["iq/link/time_clean"].shape == (1, 2, 1, 1, 4)
        np.testing.assert_array_equal(
            h5["bundle/shard_offsets"][()],
            np.asarray([[0, 1], [1, 1]], dtype=np.int64),
        )


def test_appendable_hdf5_bundle_supports_iq_link_library_downlink_rx_axis(
    tmp_path: Path,
):
    first = _iq_link_library_downlink_bundle_fragment(global_rx_index=0, shard_index=0)
    second = _iq_link_library_downlink_bundle_fragment(global_rx_index=1, shard_index=1)
    output_path = tmp_path / "bundles" / "iq_downlink_bundle.h5"

    with HDF5ResultBundleWriter(output_path, compression="mixed", gzip_level=1) as bundle:
        bundle.append_result(first)
        bundle.append_result(second)

    validate_hdf5_contract(output_path)
    index = read_bundle_index(output_path)
    assert index["ue_axis_role"] == "rx"
    np.testing.assert_array_equal(
        index["global_ue_indices"],
        np.asarray([0, 1], dtype=np.int64),
    )
    with h5py.File(output_path, "r") as h5:
        assert h5["bundle/source_contract_name"][()].decode("utf-8") == (
            IQ_LINK_LIBRARY_CONTRACT_NAME
        )
        assert h5["iq/link/frequency_clean"].shape == (1, 1, 2, 1, 2, 8)
        assert h5["iq/link/time_clean"].shape == (1, 1, 2, 1, 4)
        assert h5["iq/link/frequency_clean"].maxshape[2] is None
        np.testing.assert_array_equal(
            h5["bundle/shard_offsets"][()],
            np.asarray([[0, 1], [1, 1]], dtype=np.int64),
        )


def test_product_full_cfr_truth_product_writes_minimal_truth_hdf5(tmp_path: Path):
    output_path = tmp_path / "cfr_truth_only.h5"
    base = create_phase1_minimal_result()
    result = replace(
        base,
        metadata=replace(
            base.metadata,
            output_profile="full",
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
        assert h5["meta/output_profile"][()].decode("utf-8") == "full"
        assert tuple(v.decode("utf-8") for v in h5["meta/output_products"][()]) == ("cfr_truth",)
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


def _labels_only_bundle_fragment(
    *,
    global_tx_index: int,
    shard_index: int,
) -> RTLabelsOnlyResult:
    full = create_phase1_minimal_result()
    topology = replace(
        full.topology,
        tx_positions_m=full.topology.tx_positions_m
        + np.asarray([float(global_tx_index), 0.0, 0.0], dtype=np.float32),
        tx_labels=(f"ue{global_tx_index}",),
    )
    derived = replace(
        full.derived,
        first_path_delay_s=np.array([[1.0e-8]], dtype=np.float32),
        first_path_propagation_range_m=np.array([[2.9979246]], dtype=np.float32),
        strongest_path_delay_s=np.array([[1.0e-8]], dtype=np.float32),
    )
    shard = ShardMetadata(
        shard_index=shard_index,
        shard_count=2,
        axis="ue",
        global_rx_start=0,
        global_rx_indices=np.array([0], dtype=np.int64),
        global_tx_indices=np.array([global_tx_index], dtype=np.int64),
    )
    return RTLabelsOnlyResult(
        metadata=replace(
            full.metadata,
            contract_name=RT_LABELS_CONTRACT_NAME,
            output_profile="rt_labels_only",
        ),
        input_spec=full.input_spec,
        topology=topology,
        devices=full.devices,
        antenna=full.antenna,
        scene=full.scene,
        frequency=full.frequency,
        runtime=full.runtime,
        derived=derived,
        link_labels=RTCompactLinkLabels.from_topology(topology, derived, shard=shard),
        link=full.link,
        shard=shard,
    )


def _labels_only_downlink_bundle_fragment(
    *,
    global_rx_index: int,
    shard_index: int,
) -> RTLabelsOnlyResult:
    full = create_phase1_minimal_result()
    link = LinkConfig(phy_link_direction="downlink")
    topology = replace(
        full.topology,
        rx_positions_m=full.topology.rx_positions_m
        + np.asarray([float(global_rx_index), 0.0, 0.0], dtype=np.float32),
        rx_labels=(f"ue{global_rx_index}",),
    )
    derived = replace(
        full.derived,
        first_path_delay_s=np.array([[1.0e-8]], dtype=np.float32),
        first_path_propagation_range_m=np.array([[2.9979246]], dtype=np.float32),
        strongest_path_delay_s=np.array([[1.0e-8]], dtype=np.float32),
    )
    shard = ShardMetadata(
        shard_index=shard_index,
        shard_count=2,
        axis="ue",
        global_rx_start=global_rx_index,
        global_rx_indices=np.array([global_rx_index], dtype=np.int64),
        global_tx_indices=np.array([0], dtype=np.int64),
    )
    return RTLabelsOnlyResult(
        metadata=replace(
            full.metadata,
            contract_name=RT_LABELS_CONTRACT_NAME,
            output_profile="rt_labels_only",
        ),
        input_spec=full.input_spec,
        topology=topology,
        devices=full.devices,
        antenna=full.antenna,
        scene=full.scene,
        frequency=full.frequency,
        runtime=full.runtime,
        derived=derived,
        link_labels=RTCompactLinkLabels.from_topology(topology, derived, shard=shard),
        link=link,
        shard=shard,
    )


def _iq_link_library_bundle_fragment(
    *,
    global_tx_index: int,
    shard_index: int,
) -> IQLinkLibraryResult:
    full = create_phase1_minimal_result()
    topology = replace(
        full.topology,
        tx_positions_m=full.topology.tx_positions_m
        + np.asarray([float(global_tx_index), 0.0, 0.0], dtype=np.float32),
        tx_labels=(f"ue{global_tx_index}",),
    )
    shard = ShardMetadata(
        shard_index=shard_index,
        shard_count=2,
        axis="ue",
        global_rx_start=0,
        global_rx_indices=np.array([0], dtype=np.int64),
        global_tx_indices=np.array([global_tx_index], dtype=np.int64),
    )
    frequency_clean = np.full(
        (1, 1, 1, 1, 2, full.frequency.num_subcarriers),
        np.complex64(global_tx_index + 1),
        dtype=np.complex64,
    )
    time_clean = np.full(
        (1, 1, 1, 1, 4),
        np.complex64(global_tx_index + 1),
        dtype=np.complex64,
    )
    return IQLinkLibraryResult(
        metadata=replace(
            full.metadata,
            contract_name=IQ_LINK_LIBRARY_CONTRACT_NAME,
            output_profile="iq_link_library",
        ),
        input_spec=full.input_spec,
        topology=topology,
        devices=full.devices,
        antenna=full.antenna,
        scene=full.scene,
        frequency=full.frequency,
        runtime=full.runtime,
        iq=IQObservationResult(
            sample_rate_hz=20e6,
            fft_size=full.frequency.num_subcarriers,
            cp_length=0,
            num_ofdm_symbols=2,
            time_domain_convention="ofdm_ifft_per_symbol_cp_appended_contiguous_symbols",
            link=LinkIQCapture(
                frequency_clean=frequency_clean,
                time_clean=time_clean,
            ),
        ),
        link=full.link,
        shard=shard,
    )


def _iq_link_library_downlink_bundle_fragment(
    *,
    global_rx_index: int,
    shard_index: int,
) -> IQLinkLibraryResult:
    full = create_phase1_minimal_result()
    link = LinkConfig(phy_link_direction="downlink")
    topology = replace(
        full.topology,
        rx_positions_m=full.topology.rx_positions_m
        + np.asarray([float(global_rx_index), 0.0, 0.0], dtype=np.float32),
        rx_labels=(f"ue{global_rx_index}",),
    )
    shard = ShardMetadata(
        shard_index=shard_index,
        shard_count=2,
        axis="ue",
        global_rx_start=global_rx_index,
        global_rx_indices=np.array([global_rx_index], dtype=np.int64),
        global_tx_indices=np.array([0], dtype=np.int64),
    )
    frequency_clean = np.full(
        (1, 1, 1, 1, 2, full.frequency.num_subcarriers),
        np.complex64(global_rx_index + 1),
        dtype=np.complex64,
    )
    time_clean = np.full(
        (1, 1, 1, 1, 4),
        np.complex64(global_rx_index + 1),
        dtype=np.complex64,
    )
    return IQLinkLibraryResult(
        metadata=replace(
            full.metadata,
            contract_name=IQ_LINK_LIBRARY_CONTRACT_NAME,
            output_profile="iq_link_library",
        ),
        input_spec=full.input_spec,
        topology=topology,
        devices=full.devices,
        antenna=full.antenna,
        scene=full.scene,
        frequency=full.frequency,
        runtime=full.runtime,
        iq=IQObservationResult(
            sample_rate_hz=20e6,
            fft_size=full.frequency.num_subcarriers,
            cp_length=0,
            num_ofdm_symbols=2,
            time_domain_convention="ofdm_ifft_per_symbol_cp_appended_contiguous_symbols",
            link=LinkIQCapture(
                frequency_clean=frequency_clean,
                time_clean=time_clean,
            ),
        ),
        link=link,
        shard=shard,
    )
