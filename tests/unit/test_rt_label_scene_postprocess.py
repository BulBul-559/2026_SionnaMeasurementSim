from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
import torch

from sionna_measurement_sim.postprocess import (
    RTLabelSceneBuildConfig,
    build_rt_label_scene_dataset,
    validate_rt_label_scene_h5,
)


def test_build_rt_label_scene_dataset_writes_link_label_scene_h5(tmp_path: Path):
    run_dir = _write_fake_rt_label_run(tmp_path)
    source_before = sorted(path.relative_to(run_dir) for path in run_dir.rglob("*"))

    result = build_rt_label_scene_dataset(
        RTLabelSceneBuildConfig(
            run_dirs=(run_dir,),
            output_root=tmp_path / "dataset",
            workers=1,
            label_compression="gzip",
        )
    )

    assert result.completed == 1
    assert result.failed == 0
    scene_h5 = tmp_path / "dataset" / "scenes" / "front3d_fake.h5"
    assert scene_h5.exists()
    validate_rt_label_scene_h5(scene_h5)

    with h5py.File(scene_h5, "r") as h5:
        assert h5["meta/contract_name"][()].decode("utf-8") == "sionna_rt_label_scene_dataset"
        assert h5["link_id"][()].tolist() == list(range(6))
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
        assert h5["valid_mask"][()].tolist() == [True, True, False, True, True, False]
        assert h5["source_local_index"][()].tolist() == [0, 1, 2, 3, 0, 1]
        assert h5["source_local_tx_index"][()].tolist() == [0, 0, 1, 1, 0, 0]
        assert h5["source_local_rx_index"][()].tolist() == [0, 1, 0, 1, 0, 1]
        assert [value.decode("utf-8") for value in h5["source_shard_id"][()]] == [
            "000",
            "000",
            "000",
            "000",
            "001",
            "001",
        ]
        assert h5["labels/link/link_index"][()].tolist() == [0, 1, 2, 3, 0, 1]
        assert np.allclose(
            h5["labels/link/path_power_db"][()],
            [-90.0, -80.0, -120.0, -60.0, -70.0, -130.0],
        )
        assert np.allclose(
            h5["labels/link/first_path_propagation_range_m"][()],
            [1.0, 1.1, np.nan, 1.3, 2.0, np.nan],
            equal_nan=True,
        )
        assert h5["link/target_room_index"][()].tolist() == [0, 0, 1, 1, 1, 1]
        assert np.allclose(
            h5["link/target_room_area_m2"][()],
            [20.0, 20.0, 9.0, 9.0, 9.0, 9.0],
        )
        assert h5["topology/global_tx_index"][()].tolist() == [10, 11, 12]
        assert h5["topology/global_rx_index"][()].tolist() == [0, 1]
        assert bool(h5["floorplan/found"][()]) is True
        assert h5["floorplan/image_png/bytes"].shape[0] > 0
        assert h5["assets/label_json/bytes"].shape[0] > 0

    manifest = json.loads((tmp_path / "dataset" / "manifest.json").read_text())
    assert manifest["completed"] == 1
    assert manifest["scenes"][0]["num_links"] == 6
    assert manifest["scenes"][0]["valid_link_count"] == 4
    assert manifest["scenes"][0]["label_dataset_count"] == 20
    assert manifest["link_index_pt"] == "link_index.pt"
    assert manifest["link_index_parquet"] == "link_index.parquet"
    assert manifest["room_index_pt"] == "room_index.pt"
    assert manifest["room_index_parquet"] == "room_index.parquet"

    link_index = torch.load(tmp_path / "dataset" / "link_index.pt")
    assert link_index["scene_ids"] == ["front3d_fake"]
    assert link_index["global_link_id"].tolist() == list(range(6))
    assert link_index["bs_id"].tolist() == [0, 1, 0, 1, 0, 1]
    assert link_index["ue_id"].tolist() == [10, 10, 11, 11, 12, 12]
    assert link_index["valid_mask"].tolist() == [True, True, False, True, True, False]
    assert np.allclose(
        link_index["path_power_db"].numpy(),
        [-90.0, -80.0, -120.0, -60.0, -70.0, -130.0],
    )
    assert np.allclose(
        link_index["first_path_propagation_range_m"].numpy(),
        [1.0, 1.1, np.nan, 1.3, 2.0, np.nan],
        equal_nan=True,
    )

    room_index = torch.load(tmp_path / "dataset" / "room_index.pt")
    assert room_index["room_id"] == ["room_a", "room_b"]
    assert room_index["room_type"] == ["LivingRoom", "Bedroom"]

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
    assert link_parquet["los_flag"] == [True, True, False, True, False, False]
    room_parquet = pq.read_table(tmp_path / "dataset" / "room_index.parquet").to_pydict()
    assert room_parquet["room_id"] == ["room_a", "room_b"]

    source_after = sorted(path.relative_to(run_dir) for path in run_dir.rglob("*"))
    assert source_after == source_before


def _write_fake_rt_label_run(tmp_path: Path) -> Path:
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

    _write_label_shard(
        results_dir / "result_000.h5",
        shard_id="000",
        shard_index=0,
        global_tx=[10, 11],
        path_power=[-90.0, -80.0, -120.0, -60.0],
        valid=[True, True, False, True],
        los=[True, True, False, True],
        first_path_range=[1.0, 1.1, np.nan, 1.3],
    )
    _write_label_shard(
        results_dir / "result_001.h5",
        shard_id="001",
        shard_index=1,
        global_tx=[12],
        path_power=[-70.0, -130.0],
        valid=[True, False],
        los=[False, False],
        first_path_range=[2.0, np.nan],
    )
    manifest = {
        "scene_id": "front3d_fake",
        "label_file": label_file.as_posix(),
        "scene_file": scene_file.as_posix(),
        "config_snapshot": {"output_profile": "rt_labels_only"},
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


def _write_label_shard(
    path: Path,
    *,
    shard_id: str,
    shard_index: int,
    global_tx: list[int],
    path_power: list[float],
    valid: list[bool],
    los: list[bool],
    first_path_range: list[float],
) -> None:
    tx_count = len(global_tx)
    rx_count = 2
    link_count = tx_count * rx_count
    tx_index = np.repeat(np.arange(tx_count, dtype=np.int32), rx_count)
    rx_index = np.tile(np.arange(rx_count, dtype=np.int32), tx_count)
    global_tx_index = np.repeat(np.asarray(global_tx, dtype=np.int64), rx_count)
    global_rx_index = np.tile(np.asarray([0, 1], dtype=np.int64), tx_count)
    tx_xy_by_id = {10: [1.0, 0.0], 11: [2.0, 0.0], 12: [3.0, 0.0]}
    tx_xy = np.asarray([tx_xy_by_id[int(item)] for item in global_tx_index], dtype=np.float32)
    rx_xy = np.asarray([[0.0, float(item)] for item in global_rx_index], dtype=np.float32)
    labels = {
        "link_index": (tx_index * rx_count + rx_index).astype(np.int32),
        "tx_index": tx_index,
        "rx_index": rx_index,
        "global_tx_index": global_tx_index,
        "global_rx_index": global_rx_index,
        "tx_xy_m": tx_xy,
        "rx_xy_m": rx_xy,
        "link_valid_mask": np.asarray(valid, dtype=np.bool_),
        "geometric_distance_m": np.linspace(1.0, 1.0 + 0.1 * (link_count - 1), link_count).astype(
            np.float32
        ),
        "first_path_delay_s": (np.asarray(first_path_range, dtype=np.float32) / 3.0e8),
        "first_path_propagation_range_m": np.asarray(first_path_range, dtype=np.float32),
        "strongest_path_delay_s": (np.asarray(first_path_range, dtype=np.float32) / 3.0e8),
        "path_power_db": np.asarray(path_power, dtype=np.float32),
        "los_flag": np.asarray(los, dtype=np.bool_),
        "nlos_flag": ~np.asarray(los, dtype=np.bool_),
        "path_count": np.asarray([1 if item else 0 for item in valid], dtype=np.int32),
        "first_path_aoa_azimuth_rad": np.zeros(link_count, dtype=np.float32),
        "first_path_aoa_zenith_rad": np.zeros(link_count, dtype=np.float32),
        "tx_rx_bearing_rad": np.zeros(link_count, dtype=np.float32),
        "tx_rx_distance_m": np.linspace(1.0, 1.0 + 0.1 * (link_count - 1), link_count).astype(
            np.float32
        ),
    }
    with h5py.File(path, "w") as h5:
        h5.create_dataset("shard/shard_index", data=np.int32(shard_index))
        h5.create_dataset("shard/shard_count", data=np.int32(2))
        h5.create_dataset("shard/global_tx_indices", data=np.asarray(global_tx, dtype=np.int64))
        h5.create_dataset("shard/global_rx_indices", data=np.asarray([0, 1], dtype=np.int64))
        h5.create_dataset(
            "topology/tx_positions_m",
            data=np.asarray(
                [[tx_xy_by_id[int(tx)][0], tx_xy_by_id[int(tx)][1], 1.6] for tx in global_tx],
                dtype=np.float32,
            ),
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
        group = h5.require_group("labels/link")
        for name, values in labels.items():
            dataset = group.create_dataset(name, data=values)
            dataset.attrs["index_order"] = "link,xy" if values.ndim == 2 else "link"
            if name.endswith("_m") or name in ("tx_xy_m", "rx_xy_m"):
                dataset.attrs["unit"] = "m"
            elif name.endswith("_s"):
                dataset.attrs["unit"] = "s"
            elif name.endswith("_rad"):
                dataset.attrs["unit"] = "rad"
            elif name == "path_power_db":
                dataset.attrs["unit"] = "dB"
