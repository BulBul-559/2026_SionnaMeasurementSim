import json

import h5py
import matplotlib.image as mpimg
import numpy as np

from sionna_measurement_sim.visualization.radio_map import (
    RadioMapRenderConfig,
    generate_radio_map_heatmaps,
    load_radio_map_table,
    resolve_result_files,
)


def test_generate_sharded_radio_map_heatmaps(tmp_path):
    scene_dir = tmp_path / "data" / "scene_0000"
    floorplan_dir = scene_dir / "floorplan"
    label_dir = scene_dir / "label"
    floorplan_dir.mkdir(parents=True)
    label_dir.mkdir()
    label_path = label_dir / "label.json"
    label_path.write_text('{"bs_points": [], "ue_points": []}', encoding="utf-8")
    mpimg.imsave(floorplan_dir / "floorplan_1p60.png", np.ones((16, 16, 3), dtype=np.float32))
    (floorplan_dir / "meta.json").write_text(
        json.dumps({"origin_xy_m": [0.0, 0.0], "extent_xy_m": [4.0, 4.0]}),
        encoding="utf-8",
    )

    run_dir = tmp_path / "outputs" / "run"
    results_dir = run_dir / "results"
    manifest_dir = run_dir / "manifest"
    results_dir.mkdir(parents=True)
    manifest_dir.mkdir()
    result0 = results_dir / "result_000.h5"
    result1 = results_dir / "result_001.h5"
    _write_result(
        result0,
        ue_indices=[0, 1],
        ue_positions=[[0.5, 0.5, 1.6], [1.5, 0.5, 1.6]],
        rss=[[-70.0, -82.0], [-65.0, -78.0]],
    )
    _write_result(
        result1,
        ue_indices=[2],
        ue_positions=[[0.5, 1.5, 1.6]],
        rss=[[-74.0, -68.0]],
    )
    manifest = {
        "label_file": label_path.as_posix(),
        "results": [
            {"result_h5": result0.as_posix(), "shard_index": 0},
            {"result_h5": result1.as_posix(), "shard_index": 1},
        ],
    }
    (manifest_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    summary = generate_radio_map_heatmaps(
        run_dir,
        config=RadioMapRenderConfig(
            render_mode="both",
            grid_resolution_m=1.0,
            dpi=60,
            point_size=8.0,
        ),
    )

    assert summary["ue_count"] == 3
    assert summary["bs_count"] == 2
    assert len(summary["generated_files"]) == 4
    assert (run_dir / "figures" / "heatmaps" / "radio_map_bs_000_interpolated.png").exists()
    assert (run_dir / "figures" / "heatmaps" / "radio_map_bs_001_samples.png").exists()
    assert (run_dir / "figures" / "heatmaps" / "radio_map_values.csv").exists()
    files, _manifest = resolve_result_files(run_dir)
    table = load_radio_map_table(files, RadioMapRenderConfig())
    assert table.bs_positions_m.shape == (2, 3)
    np.testing.assert_allclose(table.bs_positions_m[:, :2], [[0.0, 0.0], [3.0, 0.0]])


def _write_result(path, *, ue_indices, ue_positions, rss):
    with h5py.File(path, "w") as h5:
        link = h5.create_group("link")
        link.create_dataset("tx_role", data=np.bytes_("ue"))
        link.create_dataset("rx_role", data=np.bytes_("bs"))
        topology = h5.create_group("topology")
        topology.create_dataset("tx_positions_m", data=np.asarray(ue_positions, dtype=np.float32))
        topology.create_dataset(
            "rx_positions_m",
            data=np.asarray([[0.0, 0.0, 2.5], [3.0, 0.0, 2.5]], dtype=np.float32),
        )
        shard = h5.create_group("shard")
        shard.create_dataset("global_tx_indices", data=np.asarray(ue_indices, dtype=np.int64))
        shard.create_dataset("global_rx_indices", data=np.asarray([0, 1], dtype=np.int64))
        observation = h5.create_group("observation")
        observation.create_dataset("rssi_dbm", data=np.asarray(rss, dtype=np.float32)[None, ...])
        observation.create_dataset(
            "valid_mask",
            data=np.ones((1, len(ue_indices), 2), dtype=np.bool_),
        )
