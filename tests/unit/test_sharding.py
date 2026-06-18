import json
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np

from sionna_measurement_sim.config.schema import (
    OutputBundleConfig,
    OutputGpuSchedulerConfig,
    OutputShardingConfig,
)
from sionna_measurement_sim.domain.results import (
    ShardMetadata,
    create_phase1_minimal_result,
)
from sionna_measurement_sim.io.hdf5_reader import read_bundle_index
from sionna_measurement_sim.io.label_parser import (
    count_topology_points,
    load_role_topology_from_label,
    load_topology_from_label,
)
from sionna_measurement_sim.io.schema_validator import validate_hdf5_contract
from sionna_measurement_sim.rt import truth_pipeline
from sionna_measurement_sim.rt.truth_pipeline import RTTruthRunConfig, _build_shard_specs


def test_load_topology_from_label_supports_rx_range(tmp_path: Path):
    label_path = _write_label(tmp_path)

    topology = load_topology_from_label(
        label_path,
        max_bs=2,
        max_ue=1,
        rx_start=2,
        rx_count=2,
    )

    assert topology.tx_labels == ("BS0", "BS1")
    assert topology.rx_labels == ("UE2", "UE3")
    np.testing.assert_allclose(
        topology.rx_positions_m,
        np.array(
            [
                [12.0, 22.0, 1.0],
                [13.0, 23.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )


def test_load_role_topology_from_label_supports_ue_range_and_global_indices(tmp_path: Path):
    label_path = _write_label(tmp_path)

    topology = load_role_topology_from_label(
        label_path,
        max_bs=2,
        max_ue=1,
        ue_start=2,
        ue_count=2,
    )

    assert topology.bs_labels == ("BS0", "BS1")
    assert topology.ue_labels == ("UE2", "UE3")
    np.testing.assert_array_equal(topology.bs_global_indices, np.array([0, 1]))
    np.testing.assert_array_equal(topology.ue_global_indices, np.array([2, 3]))


def test_load_topology_from_label_keeps_max_count_as_cap(tmp_path: Path):
    label_path = _write_label(tmp_path)

    topology = load_topology_from_label(label_path, max_bs=10, max_ue=10)

    assert topology.tx_labels == ("BS0", "BS1", "BS2")
    assert topology.rx_labels == ("UE0", "UE1", "UE2", "UE3")


def test_load_topology_from_label_supports_explicit_rx_and_tx_indices(tmp_path: Path):
    label_path = _write_label(tmp_path)

    topology = load_topology_from_label(
        label_path,
        max_bs=1,
        max_ue=1,
        rx_indices=(3, 1),
        tx_indices=(2, 0),
    )

    assert topology.tx_labels == ("BS2", "BS0")
    assert topology.rx_labels == ("UE3", "UE1")
    np.testing.assert_allclose(
        topology.tx_positions_m,
        np.array(
            [
                [2.0, 3.0, 2.0],
                [0.0, 1.0, 2.0],
            ],
            dtype=np.float32,
        ),
    )


def test_count_topology_points_reads_available_label_size(tmp_path: Path):
    label_path = _write_label(tmp_path)

    assert count_topology_points(label_path) == (3, 4)


def test_build_shard_specs_splits_8884_ues_into_nine_shards(tmp_path: Path):
    label_path = _write_label_with_counts(tmp_path, tx_count=6, rx_count=8884)
    config = RTTruthRunConfig(
        label_file=label_path,
        scene_file=tmp_path / "scene.xml",
        output_dir=tmp_path / "out",
        max_bs=6,
        max_ue=8884,
        output_sharding_config=OutputShardingConfig(
            enabled=True,
            shard_size=1000,
        ),
    )

    specs = _build_shard_specs(config)

    assert len(specs) == 9
    assert [spec.ue_start for spec in specs] == [
        0,
        1000,
        2000,
        3000,
        4000,
        5000,
        6000,
        7000,
        8000,
    ]
    assert [spec.ue_count for spec in specs] == [
        1000,
        1000,
        1000,
        1000,
        1000,
        1000,
        1000,
        1000,
        884,
    ]


def test_build_shard_specs_caps_to_available_ues_and_normalizes_ue_axis(tmp_path: Path):
    label_path = _write_label(tmp_path)
    config = RTTruthRunConfig(
        label_file=label_path,
        scene_file=tmp_path / "scene.xml",
        output_dir=tmp_path / "out",
        max_bs=10,
        max_ue=10,
        output_sharding_config=OutputShardingConfig(
            enabled=True,
            axis="ue",
            shard_size=3,
        ),
    )

    specs = _build_shard_specs(config)

    assert len(specs) == 2
    assert [spec.axis for spec in specs] == ["ue", "ue"]
    assert [spec.ue_start for spec in specs] == [0, 3]
    assert [spec.ue_count for spec in specs] == [3, 1]


def test_split_shard_spec_for_fallback_preserves_parent_identity():
    spec = truth_pipeline.ShardSpec(
        shard_index=89,
        shard_count=130,
        axis="ue",
        ue_start=1780,
        ue_count=20,
    )

    children = truth_pipeline._split_shard_spec_for_fallback(
        spec,
        "drjit_array_limit",
    )

    assert [child.shard_id for child in children] == ["089_00", "089_01"]
    assert [child.ue_start for child in children] == [1780, 1790]
    assert [child.ue_count for child in children] == [10, 10]
    assert [child.parent_shard_index for child in children] == [89, 89]
    assert [child.fallback_level for child in children] == [1, 1]


def test_retryable_shard_error_classification_uses_exception_chain():
    cause = RuntimeError("jit_malloc(): out of memory! Could not allocate bytes")
    try:
        raise RuntimeError("dr.while_loop(): encountered an exception") from cause
    except RuntimeError as exc:
        retry_error = truth_pipeline._classify_retryable_shard_error(exc)

    assert retry_error == "cuda_oom"


def test_sharded_pipeline_fallback_writes_results_and_manifest_dirs(
    tmp_path: Path,
    monkeypatch,
):
    label_path = _write_label(tmp_path)
    output_dir = tmp_path / "out"
    failed_once: set[tuple[int, int]] = set()
    cleanup_calls: list[bool] = []

    def fake_worker(config: RTTruthRunConfig, gpu_id: int | None) -> Path:
        del gpu_id
        assert config.shard_spec is not None
        spec = config.shard_spec
        key = (int(spec.ue_start), int(spec.ue_count or 0))
        if key == (0, 2) and key not in failed_once:
            failed_once.add(key)
            raise RuntimeError(
                "drjit.cuda.ad.TensorXf.__mul__(): jit_var_counter(): "
                "tried to create an array with 4314885120 entries, "
                "which exceeds the limit of 2^32 == 4294967296 entries."
            )
        result_path = config.output_dir / config.hdf5_filename
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_bytes(b"fake hdf5")
        return result_path

    monkeypatch.setattr(
        truth_pipeline,
        "_run_shard_worker_in_isolated_process",
        fake_worker,
    )
    monkeypatch.setattr(
        truth_pipeline,
        "_clear_accelerator_caches",
        lambda: cleanup_calls.append(True),
    )

    result_dir = truth_pipeline.run_sharded_rt_truth_pipeline(
        RTTruthRunConfig(
            label_file=label_path,
            scene_file=tmp_path / "scene.xml",
            output_dir=output_dir,
            max_bs=2,
            max_ue=4,
            output_sharding_config=OutputShardingConfig(
                enabled=True,
                shard_size=2,
                parallel_workers=1,
                visualization_mode="none",
            ),
        )
    )

    assert result_dir == output_dir
    assert (output_dir / "results" / "result_000_00.h5").is_file()
    assert (output_dir / "results" / "result_000_01.h5").is_file()
    assert (output_dir / "results" / "result_001.h5").is_file()
    manifest_path = output_dir / "manifest" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["sharding"]["planned_shard_count"] == 2
    assert manifest["sharding"]["result_file_count"] == 3
    assert manifest["sharding"]["fallback"]["split_count"] == 1
    assert manifest["config_snapshot_path"].endswith("manifest/config_snapshot.json")
    assert (output_dir / "manifest" / "config_snapshot.json").is_file()
    assert (output_dir / "manifest" / "shard_attempts.jsonl").is_file()
    assert cleanup_calls
    assert [item["global_ue_indices"] for item in manifest["results"]] == [
        [0],
        [1],
        [2, 3],
    ]


def test_bundled_sharded_pipeline_writes_appendable_h5_and_manifest(
    tmp_path: Path,
    monkeypatch,
):
    label_path = _write_label(tmp_path)
    output_dir = tmp_path / "bundle_out"

    def fake_prepare(
        config: RTTruthRunConfig,
        tracer,
        *,
        start: float,
        output_dir: Path,
        logs_dir: Path,
        prepare_only: bool = False,
    ) -> truth_pipeline.PreparedShardOutput:
        del tracer, start, output_dir, logs_dir
        assert prepare_only
        assert config.shard_spec is not None
        spec = config.shard_spec
        ue_index = int(spec.ue_start)
        base = create_phase1_minimal_result()
        result = replace(
            base,
            topology=replace(
                base.topology,
                tx_positions_m=np.asarray(
                    [[float(ue_index), 0.0, 1.5]],
                    dtype=np.float32,
                ),
                tx_labels=(f"ue{ue_index}",),
            ),
            truth=replace(
                base.truth,
                cfr=base.truth.cfr * np.complex64(float(ue_index + 1)),
            ),
            shard=ShardMetadata(
                shard_index=spec.shard_index,
                shard_count=spec.shard_count,
                axis=spec.axis,
                global_rx_start=0,
                global_rx_indices=np.asarray([0], dtype=np.int64),
                global_tx_indices=np.asarray([ue_index], dtype=np.int64),
            ),
        )
        return truth_pipeline.PreparedShardOutput(
            result=result,
            output_plan=config.output_plan,
            scene_id="fake_scene",
            phase=3,
            elapsed_seconds=0.01,
            raw_cfr_shape=result.truth.cfr.shape,
            internal_cfr_shape=result.truth.cfr.shape,
            path_count=0,
            observation_snr_db=None,
            software_versions={},
            shard_metadata=result.shard,
        )

    monkeypatch.setattr(
        truth_pipeline,
        "_run_rt_truth_pipeline_single_impl",
        fake_prepare,
    )

    result_dir = truth_pipeline.run_sharded_rt_truth_pipeline(
        RTTruthRunConfig(
            label_file=label_path,
            scene_file=tmp_path / "scene.xml",
            output_dir=output_dir,
            max_bs=2,
            max_ue=3,
            output_sharding_config=OutputShardingConfig(
                enabled=True,
                shard_size=1,
                parallel_workers=1,
                visualization_mode="none",
                bundle=OutputBundleConfig(
                    enabled=True,
                    max_planned_shards_per_bundle=2,
                    validate_schema=True,
                ),
            ),
        )
    )

    assert result_dir == output_dir
    manifest = json.loads(
        (output_dir / "manifest" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["phase"] == "bundled_sharded_run_full"
    assert manifest["sharding"]["result_file_count"] == 2
    assert manifest["sharding"]["bundle"]["enabled"] is True
    assert len(manifest["results"]) == 3
    assert [item["global_ue_indices"] for item in manifest["results"]] == [
        [0],
        [1],
        [2],
    ]

    first_bundle = output_dir / "bundles" / "bundle_worker000_000.h5"
    second_bundle = output_dir / "bundles" / "bundle_worker000_001.h5"
    validate_hdf5_contract(first_bundle)
    validate_hdf5_contract(second_bundle)
    np.testing.assert_array_equal(
        read_bundle_index(first_bundle)["global_ue_indices"],
        np.asarray([0, 1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        read_bundle_index(second_bundle)["global_ue_indices"],
        np.asarray([2], dtype=np.int64),
    )

    with h5py.File(first_bundle, "r") as h5:
        assert h5["channel/truth/cfr"].shape == (2, 1, 1, 1, 8)
        np.testing.assert_allclose(h5["channel/truth/cfr"][0], np.complex64(1.0))
        np.testing.assert_allclose(h5["channel/truth/cfr"][1], np.complex64(2.0))
        np.testing.assert_array_equal(h5["bundle/shard_offsets"][()], [[0, 1], [1, 1]])
        assert h5["channel/truth/cfr"].maxshape[0] is None


def test_bundled_sharded_pipeline_keeps_fallback_children_in_current_bundle(
    tmp_path: Path,
    monkeypatch,
):
    label_path = _write_label(tmp_path)
    output_dir = tmp_path / "bundle_fallback_out"
    failed_once: set[tuple[int, int]] = set()
    cleanup_calls: list[bool] = []

    def fake_prepare(
        config: RTTruthRunConfig,
        tracer,
        *,
        start: float,
        output_dir: Path,
        logs_dir: Path,
        prepare_only: bool = False,
    ) -> truth_pipeline.PreparedShardOutput:
        del tracer, start, output_dir, logs_dir
        assert prepare_only
        assert config.shard_spec is not None
        spec = config.shard_spec
        key = (int(spec.ue_start), int(spec.ue_count or 0))
        if key == (0, 2) and key not in failed_once:
            failed_once.add(key)
            raise RuntimeError(
                "drjit.cuda.ad.TensorXf.__mul__(): jit_var_counter(): "
                "tried to create an array with 4314885120 entries, "
                "which exceeds the limit of 2^32 == 4294967296 entries."
            )
        ue_count = int(spec.ue_count or 0)
        ue_indices = np.arange(
            int(spec.ue_start),
            int(spec.ue_start) + ue_count,
            dtype=np.int64,
        )
        base = create_phase1_minimal_result()
        result = replace(
            base,
            metadata=replace(base.metadata, output_products=("cfr_truth",)),
            topology=replace(
                base.topology,
                tx_positions_m=np.column_stack(
                    [
                        ue_indices.astype(np.float32),
                        np.zeros(ue_count, dtype=np.float32),
                        np.full(ue_count, 1.5, dtype=np.float32),
                    ]
                ).astype(np.float32),
                tx_labels=tuple(f"ue{index}" for index in ue_indices),
            ),
            devices=type(base.devices).static(snapshots=1, tx=ue_count, rx=1),
            truth=replace(
                base.truth,
                cfr=np.full(
                    (ue_count, 1, 1, 1, 8),
                    np.complex64(float(int(spec.ue_start) + 1)),
                    dtype=np.complex64,
                ),
                path_power_db=np.full((ue_count, 1), -60.0, dtype=np.float32),
                has_geometric_signal=np.full((ue_count, 1), True, dtype=np.bool_),
                geometric_path_count=np.ones((ue_count, 1), dtype=np.int32),
                los_exists=np.full((ue_count, 1), True, dtype=np.bool_),
                nlos_exists=np.full((ue_count, 1), False, dtype=np.bool_),
            ),
            cir_truth=None,
            shard=ShardMetadata(
                shard_index=spec.shard_index,
                shard_count=spec.shard_count,
                axis=spec.axis,
                global_rx_start=0,
                global_rx_indices=np.asarray([0], dtype=np.int64),
                global_tx_indices=ue_indices,
            ),
        )
        return truth_pipeline.PreparedShardOutput(
            result=result,
            output_plan=config.output_plan,
            scene_id="fake_scene",
            phase=3,
            elapsed_seconds=0.01,
            raw_cfr_shape=result.truth.cfr.shape,
            internal_cfr_shape=result.truth.cfr.shape,
            path_count=0,
            observation_snr_db=None,
            software_versions={},
            shard_metadata=result.shard,
        )

    monkeypatch.setattr(
        truth_pipeline,
        "_run_rt_truth_pipeline_single_impl",
        fake_prepare,
    )
    monkeypatch.setattr(
        truth_pipeline,
        "_clear_accelerator_caches",
        lambda: cleanup_calls.append(True),
    )

    result_dir = truth_pipeline.run_sharded_rt_truth_pipeline(
        RTTruthRunConfig(
            label_file=label_path,
            scene_file=tmp_path / "scene.xml",
            output_dir=output_dir,
            max_bs=2,
            max_ue=4,
            output_sharding_config=OutputShardingConfig(
                enabled=True,
                shard_size=2,
                parallel_workers=1,
                visualization_mode="none",
                bundle=OutputBundleConfig(
                    enabled=True,
                    max_planned_shards_per_bundle=2,
                    validate_schema=True,
                ),
            ),
        )
    )

    assert result_dir == output_dir
    manifest = json.loads(
        (output_dir / "manifest" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["sharding"]["planned_shard_count"] == 2
    assert manifest["sharding"]["result_file_count"] == 1
    assert manifest["sharding"]["fallback"]["split_count"] == 1
    assert cleanup_calls
    assert len(manifest["results"]) == 3
    assert [item["shard_id"] for item in manifest["results"]] == [
        "000_00",
        "000_01",
        "001",
    ]
    assert [item["parent_shard_index"] for item in manifest["results"]] == [0, 0, 1]
    assert [item["fallback_level"] for item in manifest["results"]] == [1, 1, 0]
    assert [item["global_ue_indices"] for item in manifest["results"]] == [
        [0],
        [1],
        [2, 3],
    ]
    assert [
        (item["append_start"], item["append_count"]) for item in manifest["results"]
    ] == [(0, 1), (1, 1), (2, 2)]

    assert len(manifest["bundles"]) == 1
    bundle_summary = manifest["bundles"][0]
    assert bundle_summary["planned_shard_indices"] == [0, 1]
    assert bundle_summary["fragment_count"] == 3
    assert bundle_summary["ue_count"] == 4
    assert bundle_summary["global_ue_indices"] == [0, 1, 2, 3]
    assert bundle_summary["schema_validated"] is True

    bundle_path = output_dir / "bundles" / "bundle_worker000_000.h5"
    validate_hdf5_contract(bundle_path)
    index = read_bundle_index(bundle_path)
    assert index["fragment_id"] == ["000_00", "000_01", "001"]
    np.testing.assert_array_equal(
        index["shard_offsets"],
        np.asarray([[0, 1], [1, 1], [2, 2]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        index["global_ue_indices"],
        np.asarray([0, 1, 2, 3], dtype=np.int64),
    )
    with h5py.File(bundle_path, "r") as h5:
        assert h5["channel/truth/cfr"].shape == (4, 1, 1, 1, 8)
        np.testing.assert_array_equal(
            h5["bundle/parent_shard_index"][()],
            np.asarray([0, 0, 1], dtype=np.int32),
        )
        np.testing.assert_array_equal(
            h5["bundle/fallback_level"][()],
            np.asarray([1, 1, 0], dtype=np.int32),
        )


def test_bundled_sharded_pipeline_partitions_bundles_by_worker(
    tmp_path: Path,
    monkeypatch,
):
    label_path = _write_label_with_counts(tmp_path, tx_count=2, rx_count=6)
    output_dir = tmp_path / "bundle_parallel_out"
    executor_max_workers: list[int] = []
    submitted_workers: list[tuple[int, list[int], int | None]] = []

    class _ImmediateFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class _ImmediateExecutor:
        def __init__(self, max_workers: int):
            executor_max_workers.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def submit(self, fn, config, worker_index, specs, gpu_id):
            submitted_workers.append(
                (
                    int(worker_index),
                    [int(spec.shard_index) for spec in specs],
                    gpu_id,
                )
            )
            return _ImmediateFuture(fn(config, worker_index, specs, gpu_id))

    def fake_prepare(
        config: RTTruthRunConfig,
        tracer,
        *,
        start: float,
        output_dir: Path,
        logs_dir: Path,
        prepare_only: bool = False,
    ) -> truth_pipeline.PreparedShardOutput:
        del tracer, start, output_dir, logs_dir
        assert prepare_only
        assert config.shard_spec is not None
        spec = config.shard_spec
        ue_index = int(spec.ue_start)
        base = create_phase1_minimal_result()
        result = replace(
            base,
            metadata=replace(base.metadata, output_products=("cfr_truth",)),
            topology=replace(
                base.topology,
                tx_positions_m=np.asarray(
                    [[float(ue_index), 0.0, 1.5]],
                    dtype=np.float32,
                ),
                tx_labels=(f"ue{ue_index}",),
            ),
            truth=replace(
                base.truth,
                cfr=np.full(
                    (1, 1, 1, 1, 8),
                    np.complex64(float(ue_index + 1)),
                    dtype=np.complex64,
                ),
            ),
            cir_truth=None,
            shard=ShardMetadata(
                shard_index=spec.shard_index,
                shard_count=spec.shard_count,
                axis=spec.axis,
                global_rx_start=0,
                global_rx_indices=np.asarray([0], dtype=np.int64),
                global_tx_indices=np.asarray([ue_index], dtype=np.int64),
            ),
        )
        return truth_pipeline.PreparedShardOutput(
            result=result,
            output_plan=config.output_plan,
            scene_id="fake_scene",
            phase=3,
            elapsed_seconds=0.01,
            raw_cfr_shape=result.truth.cfr.shape,
            internal_cfr_shape=result.truth.cfr.shape,
            path_count=0,
            observation_snr_db=None,
            software_versions={},
            shard_metadata=result.shard,
        )

    monkeypatch.setattr(
        truth_pipeline,
        "_run_rt_truth_pipeline_single_impl",
        fake_prepare,
    )
    monkeypatch.setattr(truth_pipeline, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(truth_pipeline, "as_completed", lambda futures: list(futures))

    result_dir = truth_pipeline.run_sharded_rt_truth_pipeline(
        RTTruthRunConfig(
            label_file=label_path,
            scene_file=tmp_path / "scene.xml",
            output_dir=output_dir,
            max_bs=2,
            max_ue=6,
            output_sharding_config=OutputShardingConfig(
                enabled=True,
                shard_size=1,
                parallel_workers=2,
                gpu_ids=[0, 1],
                visualization_mode="none",
                bundle=OutputBundleConfig(
                    enabled=True,
                    max_planned_shards_per_bundle=2,
                    validate_schema=True,
                ),
            ),
        )
    )

    assert result_dir == output_dir
    assert executor_max_workers == [2]
    assert submitted_workers == [
        (0, [0, 1, 2], 0),
        (1, [3, 4, 5], 1),
    ]
    manifest = json.loads(
        (output_dir / "manifest" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["sharding"]["parallel_workers"] == 2
    assert manifest["sharding"]["planned_shard_count"] == 6
    assert manifest["sharding"]["result_file_count"] == 4
    bundle_names = [Path(item["bundle_h5"]).name for item in manifest["bundles"]]
    assert bundle_names == [
        "bundle_worker000_000.h5",
        "bundle_worker000_001.h5",
        "bundle_worker001_000.h5",
        "bundle_worker001_001.h5",
    ]
    assert len(set(bundle_names)) == len(bundle_names)
    assert [item["worker_index"] for item in manifest["bundles"]] == [0, 0, 1, 1]
    assert [item["planned_shard_indices"] for item in manifest["bundles"]] == [
        [0, 1],
        [2],
        [3, 4],
        [5],
    ]
    assert [item["global_ue_indices"] for item in manifest["bundles"]] == [
        [0, 1],
        [2],
        [3, 4],
        [5],
    ]
    assert [Path(item["bundle_h5"]).name for item in manifest["results"]] == [
        "bundle_worker000_000.h5",
        "bundle_worker000_000.h5",
        "bundle_worker000_001.h5",
        "bundle_worker001_000.h5",
        "bundle_worker001_000.h5",
        "bundle_worker001_001.h5",
    ]

    for bundle in manifest["bundles"]:
        bundle_path = Path(bundle["bundle_h5"])
        validate_hdf5_contract(bundle_path)
        index = read_bundle_index(bundle_path)
        np.testing.assert_array_equal(
            index["global_ue_indices"],
            np.asarray(bundle["global_ue_indices"], dtype=np.int64),
        )


def test_sharded_pipeline_dynamic_gpu_scheduler_waits_for_free_memory(
    tmp_path: Path,
    monkeypatch,
):
    label_path = _write_label_with_counts(tmp_path, tx_count=2, rx_count=3)
    output_dir = tmp_path / "dynamic_gpu_out"
    submitted: list[tuple[int, int | None]] = []
    executor_max_workers: list[int] = []
    query_calls = 0

    class _ImmediateFuture:
        def __init__(self, value):
            self._value = value

        def done(self):
            return True

        def result(self):
            return self._value

    class _ImmediateExecutor:
        def __init__(self, max_workers: int):
            executor_max_workers.append(max_workers)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def submit(self, fn, config, spec, gpu_id):
            submitted.append((int(spec.shard_index), gpu_id))
            return _ImmediateFuture(fn(config, spec, gpu_id))

    def fake_query(gpu_ids: list[int]) -> dict[int, float]:
        nonlocal query_calls
        query_calls += 1
        assert gpu_ids == [0, 1, 2]
        if query_calls <= 2:
            return {0: 0.20, 1: 0.70, 2: 0.80}
        return {0: 0.90, 1: 0.70, 2: 0.80}

    def fake_run(
        config: RTTruthRunConfig,
        spec: truth_pipeline.ShardSpec,
        gpu_id: int | None,
    ):
        del config
        return {
            "results": [
                {
                    "shard_id": spec.shard_id or f"{spec.shard_index:03d}",
                    "shard_index": spec.shard_index,
                    "global_ue_indices": [int(spec.ue_start)],
                    "result_h5": "",
                    "manifest": "",
                }
            ],
            "attempts": [
                {
                    "shard_id": spec.shard_id or f"{spec.shard_index:03d}",
                    "shard_index": spec.shard_index,
                    "status": "succeeded",
                    "gpu_id": gpu_id,
                }
            ],
        }

    monkeypatch.setattr(truth_pipeline, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(
        truth_pipeline,
        "wait",
        lambda futures, timeout=None, return_when=None: (set(futures), set()),
    )
    monkeypatch.setattr(truth_pipeline, "_query_gpu_free_memory_ratios", fake_query)
    monkeypatch.setattr(truth_pipeline, "_run_shard_spec_with_fallback", fake_run)

    result_dir = truth_pipeline.run_sharded_rt_truth_pipeline(
        RTTruthRunConfig(
            label_file=label_path,
            scene_file=tmp_path / "scene.xml",
            output_dir=output_dir,
            device="cuda",
            max_bs=2,
            max_ue=3,
            output_sharding_config=OutputShardingConfig(
                enabled=True,
                shard_size=1,
                parallel_workers=3,
                gpu_ids=[0, 1, 2],
                visualization_mode="none",
                gpu_scheduler=OutputGpuSchedulerConfig(
                    enabled=True,
                    free_memory_threshold=0.6,
                    scan_interval_s=0.01,
                ),
            ),
        )
    )

    assert result_dir == output_dir
    assert executor_max_workers == [3]
    assert submitted[:2] == [(0, 2), (1, 1)]
    assert submitted[2] == (2, 0)
    manifest = json.loads(
        (output_dir / "manifest" / "manifest.json").read_text(encoding="utf-8")
    )
    scheduler = manifest["sharding"]["gpu_scheduler"]
    assert scheduler["enabled"] is True
    assert scheduler["candidate_gpu_ids"] == [0, 1, 2]
    assert scheduler["free_memory_threshold"] == 0.6
    assert scheduler["scheduled_count"] == 3
    attempts = [
        json.loads(line)
        for line in (output_dir / "manifest" / "shard_attempts.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [item["gpu_id"] for item in attempts] == [2, 1, 0]


def _write_label(tmp_path: Path) -> Path:
    label_path = tmp_path / "label.json"
    bs_points = [
        {"position": [0.0, 1.0, 2.0], "label": "BS0"},
        {"position": [1.0, 2.0, 2.0], "label": "BS1"},
        {"position": [2.0, 3.0, 2.0], "label": "BS2"},
    ]
    ue_points = [
        {"x": 10.0, "y": 20.0, "z": 1.0, "label": "UE0"},
        {"x": 11.0, "y": 21.0, "z": 1.0, "label": "UE1"},
        {"x": 12.0, "y": 22.0, "z": 1.0, "label": "UE2"},
        {"x": 13.0, "y": 23.0, "z": 1.0, "label": "UE3"},
    ]
    label_path.write_text(
        json.dumps(
            {
                "label_schema": "0.1.0",
                "bs_points": bs_points,
                "ue_points": ue_points,
                "groups": [
                    {
                        "name": "room_subset_metadata",
                        "bs_points": bs_points[:1],
                        "ue_points": ue_points[:1],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return label_path


def _write_label_with_counts(tmp_path: Path, *, tx_count: int, rx_count: int) -> Path:
    label_path = tmp_path / f"label_{tx_count}_{rx_count}.json"
    bs_points = [
        {
            "position": [float(index), float(index + 1), 2.0],
            "label": f"BS{index}",
        }
        for index in range(tx_count)
    ]
    ue_points = [
        {
            "x": float(index),
            "y": float(index + 10),
            "z": 1.0,
            "label": f"UE{index}",
        }
        for index in range(rx_count)
    ]
    label_path.write_text(
        json.dumps(
            {
                "label_schema": "0.1.0",
                "bs_points": bs_points,
                "ue_points": ue_points,
                "groups": [
                    {
                        "name": "metadata_subset",
                        "bs_points": bs_points[:1],
                        "ue_points": ue_points[:1],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return label_path
