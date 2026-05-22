import json
from pathlib import Path

import numpy as np

from sionna_measurement_sim.config.schema import OutputShardingConfig
from sionna_measurement_sim.io.label_parser import (
    count_topology_points,
    load_role_topology_from_label,
    load_topology_from_label,
)
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
