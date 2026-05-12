from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from sionna_measurement_sim.app.cli import main
from sionna_measurement_sim.visualization.config import VisualizationRunConfig
from sionna_measurement_sim.visualization.report import (
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

        rx_grid = rng.normal(size=(1, 4, 2, 2, 3, 8)) + 1j * rng.normal(size=(1, 4, 2, 2, 3, 8))
        h5.create_dataset("waveform/rx_grid", data=rx_grid.astype(np.complex64))
        h5.create_dataset(
            "array/spatial_spectrum_label",
            data=rng.random((1, 4, 2, 5, 7), dtype=np.float32),
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
