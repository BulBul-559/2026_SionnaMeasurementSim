from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from sionna_measurement_sim.app.cli import main
from sionna_measurement_sim.visualization.config import VisualizationRunConfig
from sionna_measurement_sim.visualization.report import (
    _first_selected_link_pair,
    _spatial_spectrum_row_limits,
    generate_visualization_report,
    select_visualization_links,
)


def test_select_visualization_links_prefers_valid_links(tmp_path: Path):
    h5_path = _write_visualization_fixture(tmp_path / "fixture.h5")
    config = VisualizationRunConfig(
        enabled=True,
        random_seed=7,
        max_bs=2,
        sample_ue_count=2,
        max_ue=2,
    )

    with h5py.File(h5_path, "r") as h5:
        selection = select_visualization_links(h5, config)
        valid = h5["derived/link_valid_mask"][()]

    assert selection["bs_indices"] == [0, 1]
    assert len(selection["ue_indices"]) == 2
    assert all(np.any(valid[selection["bs_indices"], ue]) for ue in selection["ue_indices"])


def test_select_visualization_links_can_spread_ues_spatially(tmp_path: Path):
    h5_path = _write_visualization_fixture(tmp_path / "fixture.h5")
    config = VisualizationRunConfig(
        enabled=True,
        sample_policy="spatially_spread_valid_links",
        max_bs=2,
        sample_ue_count=2,
        max_ue=2,
    )

    with h5py.File(h5_path, "r") as h5:
        selection = select_visualization_links(h5, config)

    assert selection["ue_indices"] == [0, 2]


def test_select_visualization_links_uses_resolved_uplink_roles(tmp_path: Path):
    h5_path = tmp_path / "uplink_fixture.h5"
    with h5py.File(h5_path, "w") as h5:
        link = h5.create_group("link")
        link.create_dataset("tx_role", data=np.bytes_("ue"))
        link.create_dataset("rx_role", data=np.bytes_("bs"))
        h5.create_dataset(
            "topology/tx_positions_m",
            data=np.array(
                [[0, 0, 1], [2, 2, 1], [8, 0, 1], [12, 2, 1]],
                dtype=np.float32,
            ),
        )
        h5.create_dataset(
            "topology/rx_positions_m",
            data=np.array([[0, 3, 2], [10, 3, 2]], dtype=np.float32),
        )
        h5.create_dataset(
            "derived/link_valid_mask",
            data=np.array([[True, False], [False, False], [False, True], [False, False]]),
        )

        config = VisualizationRunConfig(
            enabled=True,
            random_seed=7,
            max_bs=2,
            sample_ue_count=2,
            max_ue=2,
        )
        selection = select_visualization_links(h5, config)
        valid = h5["derived/link_valid_mask"][()]

    assert selection["tx_role"] == "ue"
    assert selection["rx_role"] == "bs"
    assert selection["bs_indices"] == [0, 1]
    assert set(selection["ue_indices"]) == {0, 2}
    assert all(np.any(valid[ue, selection["bs_indices"]]) for ue in selection["ue_indices"])


def test_path_samples_single_link_selection_is_role_aware():
    downlink_selection = {
        "bs_indices": [1, 0],
        "ue_indices": [2, 3],
        "tx_role": "bs",
        "rx_role": "ue",
    }
    uplink_selection = {
        "bs_indices": [1, 0],
        "ue_indices": [2, 3],
        "tx_role": "ue",
        "rx_role": "bs",
    }

    assert _first_selected_link_pair(downlink_selection) == (1, 2, 1, 2)
    assert _first_selected_link_pair(uplink_selection) == (1, 2, 2, 1)


def test_generate_sample_report_writes_pngs_and_index(tmp_path: Path):
    h5_path = _write_visualization_fixture(tmp_path / "fixture.h5")
    out_dir = tmp_path / "figures"
    config = VisualizationRunConfig(
        enabled=True,
        random_seed=2,
        max_bs=2,
        sample_ue_count=2,
        max_ue=2,
        dpi=80,
    )

    report = generate_visualization_report(h5_path, out_dir, config)

    index = json.loads((out_dir / "index.json").read_text(encoding="utf-8"))
    assert report["selected_bs_indices"] == index["selected_bs_indices"]
    assert len(index["selected_ue_indices"]) == 2
    assert {entry["plot"] for entry in index["generated_files"]} >= {
        "topology",
        "link_overview",
        "cfr_lines",
        "cfr_heatmap",
        "waveform_grid",
        "spatial_spectrum",
    }
    generated_names = {Path(entry["path"]).name for entry in index["generated_files"]}
    assert {
        "cfr_lines_magnitude.png",
        "cfr_lines_phase.png",
        "cfr_heatmap_magnitude.png",
        "cfr_heatmap_phase.png",
        "cfr_error_magnitude.png",
        "cfr_error_phase.png",
        "spatial_spectrum_aoa_heatmap_label.png",
        "spatial_spectrum_aoa_heatmap_label_polar.png",
        "spatial_spectrum_truth.png",
        "spatial_spectrum_truth_polar.png",
        "spatial_spectrum_cfr_est.png",
        "spatial_spectrum_cfr_est_polar.png",
        "spatial_spectrum_observation.png",
        "spatial_spectrum_observation_polar.png",
    } <= generated_names
    for entry in index["generated_files"]:
        assert Path(entry["path"]).is_file()
        assert Path(entry["path"]).stat().st_size > 0


def test_dataset_mode_and_cli_visualize(tmp_path: Path):
    h5_path = _write_visualization_fixture(tmp_path / "fixture.h5")
    out_dir = tmp_path / "dataset"

    rc = main(
        [
            "visualize",
            "--hdf5",
            h5_path.as_posix(),
            "--output-dir",
            out_dir.as_posix(),
            "--mode",
            "dataset",
            "--dataset-path",
            "/channel/truth/cfr",
            "--plot-type",
            "heatmap",
        ]
    )

    assert rc == 0
    index = json.loads((out_dir / "index.json").read_text(encoding="utf-8"))
    assert index["mode"] == "dataset"
    assert index["generated_files"][0]["plot"] == "dataset_preview"
    assert Path(index["generated_files"][0]["path"]).stat().st_size > 0


def test_cli_visualize_accepts_sample_policy(tmp_path: Path):
    h5_path = _write_visualization_fixture(tmp_path / "fixture.h5")
    out_dir = tmp_path / "sample_policy"

    rc = main(
        [
            "visualize",
            "--hdf5",
            h5_path.as_posix(),
            "--output-dir",
            out_dir.as_posix(),
            "--mode",
            "sample",
            "--sample-policy",
            "spatially_spread_valid_links",
            "--sample-ue-count",
            "2",
            "--max-ue",
            "2",
            "--plots",
            "topology",
        ]
    )

    assert rc == 0
    index = json.loads((out_dir / "index.json").read_text(encoding="utf-8"))
    assert index["config"]["sample_policy"] == "spatially_spread_valid_links"
    assert index["selected_ue_indices"] == [0, 2]


def test_spatial_spectrum_row_limits_are_per_ue_across_selected_bs(tmp_path: Path):
    h5_path = tmp_path / "spectrum_limits.h5"
    with h5py.File(h5_path, "w") as h5:
        data = np.zeros((1, 3, 2, 2, 2), dtype=np.float32)
        data[0, 0, 0] = 1.0
        data[0, 2, 0] = 4.0
        data[0, 0, 1] = 10.0
        data[0, 2, 1] = 40.0
        h5.create_dataset("array/spatial_spectrum_truth", data=data)

        limits = _spatial_spectrum_row_limits(
            h5["array/spatial_spectrum_truth"],
            {"ue_indices": [0, 1], "bs_indices": [0, 2]},
        )

    assert limits == {0: (1.0, 4.0), 1: (10.0, 40.0)}


def test_spatial_spectrum_row_limits_follow_uplink_link_roles(tmp_path: Path):
    h5_path = tmp_path / "spectrum_limits_uplink.h5"
    with h5py.File(h5_path, "w") as h5:
        data = np.zeros((1, 4, 2, 2, 2), dtype=np.float32)
        data[0, 0, 0] = 1.0
        data[0, 0, 1] = 4.0
        data[0, 2, 0] = 10.0
        data[0, 2, 1] = 40.0
        h5.create_dataset("array/spatial_spectrum_truth", data=data)

        limits = _spatial_spectrum_row_limits(
            h5["array/spatial_spectrum_truth"],
            {
                "ue_indices": [0, 2],
                "bs_indices": [0, 1],
                "tx_role": "ue",
                "rx_role": "bs",
            },
        )

    assert limits == {0: (1.0, 4.0), 2: (10.0, 40.0)}


def _write_visualization_fixture(path: Path) -> Path:
    rng = np.random.default_rng(123)
    with h5py.File(path, "w") as h5:
        h5.create_dataset(
            "topology/tx_positions_m",
            data=np.array([[0, 0, 2], [10, 0, 2]], dtype=np.float32),
        )
        h5.create_dataset(
            "topology/rx_positions_m",
            data=np.array([[0, 3, 1], [5, 5, 1], [9, 2, 1], [12, 4, 1]], dtype=np.float32),
        )
        h5.create_dataset("frequency/frequencies_hz", data=np.linspace(3.49e9, 3.51e9, 8))

        link_shape = (2, 4)
        h5.create_dataset(
            "derived/link_valid_mask",
            data=np.array([[True, False, True, False], [False, True, True, False]]),
        )
        h5.create_dataset(
            "derived/los_flag",
            data=np.array([[True, False, False, False], [False, True, False, False]]),
        )
        h5.create_dataset(
            "derived/nlos_flag",
            data=np.array([[False, False, True, False], [False, False, True, False]]),
        )
        h5.create_dataset(
            "derived/path_count",
            data=np.array([[1, 0, 2, 0], [0, 1, 3, 0]], dtype=np.int32),
        )
        for name in (
            "first_path_aoa_azimuth_rad",
            "strongest_aoa_azimuth_rad",
            "los_aoa_azimuth_rad",
        ):
            h5.create_dataset(
                f"derived/{name}",
                data=rng.uniform(-1, 1, link_shape).astype(np.float32),
            )
        for name in (
            "first_path_aoa_zenith_rad",
            "strongest_aoa_zenith_rad",
            "los_aoa_zenith_rad",
        ):
            h5.create_dataset(
                f"derived/{name}",
                data=rng.uniform(0, 3, link_shape).astype(np.float32),
            )

        cfr = rng.normal(size=(2, 4, 2, 2, 8)) + 1j * rng.normal(size=(2, 4, 2, 2, 8))
        h5.create_dataset("channel/truth/cfr", data=cfr.astype(np.complex64))
        cfr_est = cfr[np.newaxis, ...] + 0.01
        h5.create_dataset("observation/cfr_est", data=cfr_est.astype(np.complex64))
        h5.create_dataset("observation/snr_db", data=np.full((1, 2, 4), 30.0, dtype=np.float32))
        h5.create_dataset("evaluation/nmse_db", data=rng.normal(size=(1, 2, 4)).astype(np.float32))

        rx_grid = rng.normal(size=(1, 2, 4, 2, 3, 8)) + 1j * rng.normal(size=(1, 2, 4, 2, 3, 8))
        h5.create_dataset("waveform/rx_grid", data=rx_grid.astype(np.complex64))
        zenith = np.linspace(0.0, np.pi, 5, dtype=np.float32)
        azimuth = np.linspace(-np.pi, np.pi, 7, dtype=np.float32)
        angle_grid = np.empty((5, 7, 2), dtype=np.float32)
        angle_grid[..., 0] = zenith[:, None]
        angle_grid[..., 1] = azimuth[None, :]
        h5.create_dataset("array/angle_grid_rad", data=angle_grid)
        h5.create_dataset(
            "array/aoa_heatmap_label",
            data=rng.random((1, 2, 4, 5, 7), dtype=np.float32),
        )
        h5.create_dataset(
            "array/spatial_spectrum_truth",
            data=rng.random((1, 2, 4, 5, 7), dtype=np.float32),
        )
        h5.create_dataset(
            "array/spatial_spectrum_cfr_est",
            data=rng.random((1, 2, 4, 5, 7), dtype=np.float32),
        )
        h5.create_dataset(
            "array/spatial_spectrum_observation",
            data=rng.random((1, 2, 4, 5, 7), dtype=np.float32),
        )

        nlos_shape = (2, 4, 2, 2, 3)
        nlos_valid = rng.random(nlos_shape) > 0.75
        h5.create_dataset("paths/nlos_truth/valid", data=nlos_valid)
        h5.create_dataset(
            "paths/nlos_truth/delay_s",
            data=rng.random(nlos_shape, dtype=np.float32) * 1e-7,
        )
        h5.create_dataset(
            "paths/nlos_truth/path_power_db",
            data=rng.normal(size=nlos_shape).astype(np.float32),
        )
        h5.create_dataset(
            "paths/nlos_truth/aoa_azimuth_rad",
            data=rng.uniform(-3.14, 3.14, nlos_shape).astype(np.float32),
        )

        h5.create_dataset(
            "paths/samples/sampled_link_indices",
            data=np.array([[0, 0], [1, 1]], dtype=np.int32),
        )
        h5.create_dataset(
            "paths/samples/vertices_m",
            data=np.array(
                [
                    [[[0, 0, 2], [1, 1, 1], [0, 3, 1]]],
                    [[[10, 0, 2], [8, 2, 1], [5, 5, 1]]],
                ],
                dtype=np.float32,
            ),
        )
        h5.create_dataset("paths/samples/vertex_count", data=np.array([[3], [3]], dtype=np.int32))
    return path
