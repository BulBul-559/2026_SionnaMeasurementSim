from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
import torch

from sionna_measurement_sim.postprocess import (
    CFRSceneBuildConfig,
    build_cfr_scene_dataset,
    validate_cfr_scene_h5,
)


def test_build_cfr_scene_dataset_writes_link_first_scene_h5(tmp_path: Path):
    run_dir = _write_fake_cfr_truth_run(tmp_path)
    source_before = sorted(path.relative_to(run_dir) for path in run_dir.rglob("*"))

    result = build_cfr_scene_dataset(
        CFRSceneBuildConfig(
            run_dirs=(run_dir,),
            output_root=tmp_path / "dataset",
            workers=1,
            cfr_compression="none",
        )
    )

    assert result.completed == 1
    assert result.failed == 0
    scene_h5 = tmp_path / "dataset" / "scenes" / "front3d_fake.h5"
    assert scene_h5.exists()
    validate_cfr_scene_h5(scene_h5)

    with h5py.File(scene_h5, "r") as h5:
        assert h5["cfr"].shape == (6, 1, 1, 3)
        assert h5["cfr"].compression is None
        assert h5["ue_id"][()].tolist() == [10, 10, 11, 11, 12, 12]
        assert h5["bs_id"][()].tolist() == [0, 1, 0, 1, 0, 1]
        assert h5["target_xy"][()].tolist() == [
            [1.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [3.0, 0.0],
        ]
        assert h5["source_local_index"][()].tolist() == [0, 1, 2, 3, 0, 1]
        assert h5["source_shard_index"][()].tolist() == [0, 0, 0, 0, 1, 1]
        assert [value.decode("utf-8") for value in h5["source_shard_id"][()]] == [
            "000",
            "000",
            "000",
            "000",
            "001",
            "001",
        ]
        assert h5["valid_mask"][()].tolist() == [True, True, False, False, True, True]
        assert np.allclose(h5["cfr_power_linear"][()], [1, 4, 9, 16, 25, 36])
        assert np.allclose(h5["csi_power_db"][()], 10 * np.log10([1, 4, 9, 16, 25, 36]))
        assert [value.decode("utf-8") for value in h5["rooms/room_id"][()]] == [
            "room_a",
            "room_b",
        ]
        assert [value.decode("utf-8") for value in h5["rooms/room_type"][()]] == [
            "LivingRoom",
            "Bedroom",
        ]
        assert np.allclose(h5["rooms/floor_area_m2"][()], [20.0, 9.0])
        assert h5["rooms/ue_count"][()].tolist() == [1, 2]
        assert h5["rooms/bs_count"][()].tolist() == [1, 1]
        assert h5["link/ue_room_index"][()].tolist() == [0, 0, 1, 1, 1, 1]
        assert h5["link/bs_room_index"][()].tolist() == [0, 1, 0, 1, 0, 1]
        assert h5["link/target_room_index"][()].tolist() == [0, 0, 1, 1, 1, 1]
        assert np.allclose(
            h5["link/target_room_area_m2"][()],
            [20.0, 20.0, 9.0, 9.0, 9.0, 9.0],
        )
        assert h5["link/target_room_type_id"][()].tolist() == [0, 0, 1, 1, 1, 1]
        assert bool(h5["floorplan/found"][()]) is True
        assert h5["floorplan/image_png/bytes"].shape[0] > 0
        assert h5["assets/label_json/bytes"].shape[0] > 0
        assert h5["scene_id"][()].decode("utf-8") == "front3d_fake"

    manifest = json.loads((tmp_path / "dataset" / "manifest.json").read_text())
    assert manifest["completed"] == 1
    assert manifest["scenes"][0]["num_links"] == 6
    assert manifest["scenes"][0]["valid_link_count"] == 4
    assert manifest["scenes"][0]["floorplan_found"] is True
    assert manifest["link_index_pt"] == "link_index.pt"
    assert manifest["link_index_parquet"] == "link_index.parquet"
    assert manifest["room_index_pt"] == "room_index.pt"
    assert manifest["room_index_parquet"] == "room_index.parquet"

    link_index = torch.load(tmp_path / "dataset" / "link_index.pt")
    assert link_index["scene_ids"] == ["front3d_fake"]
    assert link_index["global_link_id"].tolist() == list(range(6))
    assert link_index["local_link_index"].tolist() == list(range(6))
    assert link_index["bs_id"].tolist() == [0, 1, 0, 1, 0, 1]
    assert link_index["ue_id"].tolist() == [10, 10, 11, 11, 12, 12]
    assert link_index["target_room_index"].tolist() == [0, 0, 1, 1, 1, 1]
    assert np.allclose(
        link_index["target_room_area_m2"].numpy(),
        [20.0, 20.0, 9.0, 9.0, 9.0, 9.0],
    )
    room_index = torch.load(tmp_path / "dataset" / "room_index.pt")
    assert room_index["room_id"] == ["room_a", "room_b"]
    assert room_index["room_type"] == ["LivingRoom", "Bedroom"]
    assert np.allclose(room_index["floor_area_m2"].numpy(), [20.0, 9.0])

    link_parquet = pq.read_table(tmp_path / "dataset" / "link_index.parquet").to_pydict()
    assert link_parquet["global_link_id"] == list(range(6))
    assert link_parquet["target_room_id"] == [
        "room_a",
        "room_a",
        "room_b",
        "room_b",
        "room_b",
        "room_b",
    ]
    assert link_parquet["target_room_type"] == [
        "LivingRoom",
        "LivingRoom",
        "Bedroom",
        "Bedroom",
        "Bedroom",
        "Bedroom",
    ]
    room_parquet = pq.read_table(tmp_path / "dataset" / "room_index.parquet").to_pydict()
    assert room_parquet["room_id"] == ["room_a", "room_b"]
    assert np.allclose(room_parquet["floor_area_m2"], [20.0, 9.0])

    source_after = sorted(path.relative_to(run_dir) for path in run_dir.rglob("*"))
    assert source_after == source_before


def test_build_cfr_scene_dataset_discovers_completed_scene_index_runs(tmp_path: Path):
    root = tmp_path / "queue"
    run_dir = _write_fake_cfr_truth_run(root)
    queue_manifest = root / "scene_index_run_manifest.jsonl"
    queue_manifest.write_text(
        json.dumps(
            {
                "status": "completed",
                "scene_key": "front3d_fake",
                "scene_class": "normal_room",
                "index": 7,
                "output_dir": run_dir.as_posix(),
                "label_file": (run_dir / "source" / "label" / "label_panel_0p5.json").as_posix(),
                "scene_file": (run_dir / "source" / "scene.xml").as_posix(),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    result = build_cfr_scene_dataset(
        CFRSceneBuildConfig(
            input_root=root,
            output_root=tmp_path / "dataset",
            limit=1,
        )
    )

    assert result.completed == 1
    assert (tmp_path / "dataset" / "scenes" / "front3d_fake.h5").exists()


def _write_fake_cfr_truth_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run_front3d_fake"
    results_dir = run_dir / "results"
    manifest_dir = run_dir / "manifest"
    source_dir = run_dir / "source"
    label_dir = source_dir / "label"
    floorplan_dir = source_dir / "floorplan"
    label_floorplan_dir = source_dir / "label_floorplan"
    results_dir.mkdir(parents=True)
    manifest_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    floorplan_dir.mkdir(parents=True)
    label_floorplan_dir.mkdir(parents=True)
    scene_file = source_dir / "scene.xml"
    scene_file.write_text("<scene />", encoding="utf-8")
    label_file = label_dir / "label_panel_0p5.json"
    label_file.write_text(
        json.dumps(
            {
                "label_version": "1.1",
                "generator": "unit",
                "scene_id": "front3d_fake",
                "bs_points": [
                    {
                        "position": [0, 0, 2],
                        "room_id": "room_a",
                        "room_type": "LivingRoom",
                    },
                    {"position": [0, 1, 2], "room_id": "room_b", "room_type": "Bedroom"},
                ],
                "ue_points": [
                    *[{"x": 0, "y": 0, "z": 1.6} for _ in range(10)],
                    {
                        "x": 1,
                        "y": 0,
                        "z": 1.6,
                        "room_id": "room_a",
                        "room_type": "LivingRoom",
                    },
                    {"x": 2, "y": 0, "z": 1.6, "room_id": "room_b", "room_type": "Bedroom"},
                    {"x": 3, "y": 0, "z": 1.6, "room_id": "room_b", "room_type": "Bedroom"},
                ],
                "groups": [
                    {
                        "room_id": "room_a",
                        "room_type": "LivingRoom",
                        "floor_area_m2": 20.0,
                        "bounds_xy": [0.0, 0.0, 5.0, 4.0],
                    },
                    {
                        "room_id": "room_b",
                        "room_type": "Bedroom",
                        "floor_area_m2": 9.0,
                        "bounds_xy": [5.0, 0.0, 8.0, 3.0],
                    },
                ],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (label_dir / "report").mkdir()
    (label_dir / "report" / "label_panel_0p5_report.json").write_text("{}", encoding="utf-8")
    (floorplan_dir / "floorplan_1p60.png").write_bytes(b"\x89PNG\r\n\x1a\nfake")
    (floorplan_dir / "meta.json").write_text('{"meters_per_pixel": 0.1}', encoding="utf-8")
    (label_floorplan_dir / "label_panel_0p5.png").write_bytes(b"\x89PNG\r\n\x1a\nlabel")

    _write_shard(
        results_dir / "result_000.h5",
        shard_id="000",
        shard_index=0,
        global_tx=[10, 11],
        cfr_values=[1, 2, 3, 4],
        has_signal=[[True, True], [False, False]],
    )
    _write_shard(
        results_dir / "result_001.h5",
        shard_id="001",
        shard_index=1,
        global_tx=[12],
        cfr_values=[5, 6],
        has_signal=[[True, True]],
    )
    manifest = {
        "scene_id": "front3d_fake",
        "label_file": label_file.as_posix(),
        "scene_file": scene_file.as_posix(),
        "config_snapshot": {"output_products": ["cfr_truth"]},
        "results": [
            {
                "shard_id": "000",
                "shard_index": 0,
                "result_h5": (results_dir / "result_000.h5").as_posix(),
                "global_tx_indices": [10, 11],
                "global_rx_indices": [0, 1],
            },
            {
                "shard_id": "001",
                "shard_index": 1,
                "result_h5": (results_dir / "result_001.h5").as_posix(),
                "global_tx_indices": [12],
                "global_rx_indices": [0, 1],
            },
        ],
    }
    (manifest_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


def _write_shard(
    path: Path,
    *,
    shard_id: str,
    shard_index: int,
    global_tx: list[int],
    cfr_values: list[int],
    has_signal: list[list[bool]],
) -> None:
    tx_count = len(global_tx)
    rx_count = 2
    cfr = np.zeros((tx_count, rx_count, 1, 1, 3), dtype=np.complex64)
    for local_link, value in enumerate(cfr_values):
        tx = local_link // rx_count
        rx = local_link % rx_count
        cfr[tx, rx, 0, 0, :] = value
    with h5py.File(path, "w") as h5:
        h5.create_dataset("channel/truth/cfr", data=cfr)
        h5.create_dataset("channel/truth/has_geometric_signal", data=np.asarray(has_signal))
        h5.create_dataset(
            "channel/truth/path_power_db",
            data=np.asarray([[0.0] * rx_count] * tx_count, dtype=np.float32),
        )
        h5.create_dataset("shard/global_tx_indices", data=np.asarray(global_tx, dtype=np.int64))
        h5.create_dataset("shard/global_rx_indices", data=np.asarray([0, 1], dtype=np.int64))
        h5.create_dataset("shard/shard_index", data=np.int32(shard_index))
        h5.create_dataset("shard/shard_count", data=np.int32(2))
        h5.create_dataset(
            "topology/tx_positions_m",
            data=np.asarray([[float(tx - 9), 0.0, 1.6] for tx in global_tx], dtype=np.float32),
        )
        h5.create_dataset(
            "topology/rx_positions_m",
            data=np.asarray([[0.0, 0.0, 2.3], [0.0, 1.0, 2.3]], dtype=np.float32),
        )
        h5.create_dataset("frequency/frequencies_hz", data=np.asarray([1.0, 2.0, 3.0]))
        h5.create_dataset("frequency/center_frequency_hz", data=np.float64(2.0))
        h5.create_dataset("frequency/bandwidth_hz", data=np.float64(3.0))
        h5.create_dataset("frequency/num_subcarriers", data=np.int32(3))
        h5.create_dataset("frequency/subcarrier_spacing_hz", data=np.float64(1.0))
        h5.create_dataset("link/tx_role", data="ue", dtype=h5py.string_dtype("utf-8"))
        h5.create_dataset("link/rx_role", data="bs", dtype=h5py.string_dtype("utf-8"))
        h5.create_dataset(
            "link/phy_link_direction",
            data="uplink",
            dtype=h5py.string_dtype("utf-8"),
        )
        h5.create_dataset("link/duplex_mode", data="tdd", dtype=h5py.string_dtype("utf-8"))
        h5.create_dataset("antenna/tx_num_ant", data=np.int32(1))
        h5.create_dataset("antenna/rx_num_ant", data=np.int32(1))
        h5.create_dataset("antenna/tx_num_rows", data=np.int32(1))
        h5.create_dataset("antenna/tx_num_cols", data=np.int32(1))
        h5.create_dataset("antenna/rx_num_rows", data=np.int32(1))
        h5.create_dataset("antenna/rx_num_cols", data=np.int32(1))
        h5.create_dataset("antenna/tx_spacing_lambda", data=np.asarray([0.5, 0.5]))
        h5.create_dataset("antenna/rx_spacing_lambda", data=np.asarray([0.5, 0.5]))
        h5.create_dataset("antenna/tx_pattern", data="iso", dtype=h5py.string_dtype("utf-8"))
        h5.create_dataset("antenna/rx_pattern", data="iso", dtype=h5py.string_dtype("utf-8"))
        h5.create_dataset("antenna/tx_polarization", data="V", dtype=h5py.string_dtype("utf-8"))
        h5.create_dataset("antenna/rx_polarization", data="V", dtype=h5py.string_dtype("utf-8"))
