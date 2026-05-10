from pathlib import Path

import h5py
import numpy as np
import json

from sionna_measurement_sim.io.hdf5_reader import read_truth_cfr
from sionna_measurement_sim.rt.truth_pipeline import RTTruthRunConfig, run_rt_truth_pipeline
from sionna_measurement_sim.visualization.path_plots import plot_path_samples


def test_rt_truth_pipeline_writes_hdf5_manifest_and_log(tmp_path: Path):
    output_dir = tmp_path / "phase2_rt_truth"

    results_path = run_rt_truth_pipeline(
        RTTruthRunConfig(
            label_file=Path("data/scenes/test/test5.json"),
            scene_file=Path("data/scenes/test/scene.xml"),
            output_dir=output_dir,
            num_subcarriers=8,
            seed=1,
            scene_id="fixture_scene",
            map_id="fixture_map",
        )
    )

    assert results_path == output_dir / "results.h5"
    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "logs/run.log").is_file()

    with h5py.File(results_path, "r") as h5:
        h_true = h5["channel/truth/cfr"]
        frequencies = h5["frequency/frequencies_hz"]
        assert h_true.shape == (1, 1, 1, 1, 8)
        assert h_true.dtype == np.dtype("complex64")
        assert frequencies.shape[-1] == h_true.shape[-1]
        assert h5["channel/truth/path_power_db"].shape == (1, 1)
        assert h5["channel/truth/has_geometric_signal"].shape == (1, 1)
        assert h5["runtime/sionna_rt_version"][()].decode("utf-8") == "2.0.1"
        assert "runtime/mitsuba_version" in h5
        assert "runtime/drjit_version" in h5
        assert "runtime/torch_version" in h5
        assert "channel/cfr" not in h5
        assert "paths/full" not in h5
        assert h5["scene/scene_id"][()].decode("utf-8") == "fixture_scene"
        assert h5["scene/map_id"][()].decode("utf-8") == "fixture_map"
        assert h5["derived/geometric_distance_m"].shape == (1, 1)
        assert h5["derived/link_valid_mask"].shape == (1, 1)
        assert h5["derived/path_selection_policy"][()].decode("utf-8")
        assert np.all(np.isfinite(h5["derived/geometric_distance_m"][()]))
        assert np.any(np.isfinite(h_true[()]))

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["scene_id"] == "fixture_scene"
    assert manifest["map_id"] == "fixture_map"

    readback = read_truth_cfr(results_path)
    assert readback.shape == (1, 1, 1, 1, 8)
    assert readback.dtype == np.dtype("complex64")


def test_path_pipeline_writes_samples_full_paths_and_plot(tmp_path: Path):
    output_dir = tmp_path / "phase3_paths"

    results_path = run_rt_truth_pipeline(
        RTTruthRunConfig(
            label_file=Path("data/scenes/test/test5.json"),
            scene_file=Path("data/scenes/test/scene.xml"),
            output_dir=output_dir,
            num_subcarriers=8,
            seed=1,
            max_depth=1,
            specular_reflection=True,
            save_full_paths=True,
        )
    )

    with h5py.File(results_path, "r") as h5:
        for path in (
            "paths/samples/sampled_link_indices",
            "paths/samples/sampled_path_indices",
            "paths/samples/vertices_m",
            "paths/samples/interaction_type",
            "paths/samples/object_id",
            "paths/samples/primitive_id",
            "paths/samples/doppler_hz",
            "paths/samples/tau_s",
            "paths/samples/path_gain_db",
            "paths/samples/path_type",
            "paths/full/valid",
            "paths/full/a",
            "paths/full/vertices_m",
        ):
            assert path in h5

        assert h5["paths/samples/path_count"][0] >= 1
        vertex_count = h5["paths/samples/vertex_count"][()]
        interactions = h5["paths/samples/interaction_type"][()]
        active = vertex_count > 0
        interaction_count = np.count_nonzero(interactions != 0, axis=-1)
        assert np.all(vertex_count[active] >= interaction_count[active] + 2)
        assert np.any(interactions != 0)
        assert np.all(np.isfinite(h5["paths/samples/doppler_hz"][()]))
        assert np.all(np.isfinite(h5["paths/samples/tau_s"][()]))

    plot_path = plot_path_samples(results_path, output_dir / "paths.png")
    assert plot_path.is_file()
    assert plot_path.stat().st_size > 0


def test_observation_pipeline_writes_awgn_ls_outputs(tmp_path: Path):
    output_dir = tmp_path / "phase4_observation"

    results_path = run_rt_truth_pipeline(
        RTTruthRunConfig(
            label_file=Path("data/scenes/test/test5.json"),
            scene_file=Path("data/scenes/test/scene.xml"),
            output_dir=output_dir,
            num_subcarriers=8,
            seed=1,
            max_depth=1,
            specular_reflection=True,
            observation_snr_db=40.0,
            observation_seed=5,
        )
    )

    with h5py.File(results_path, "r") as h5:
        truth = h5["channel/truth/cfr"]
        cfr_est = h5["observation/cfr_est"]
        assert h5["waveform/standard"][()].decode("utf-8") == "custom_ofdm"
        assert h5["waveform/fft_size"][()] == 8
        assert h5["waveform/pilot_indices"].shape == (8,)
        assert h5["waveform/pilot_symbols"].shape == (8,)
        assert h5["receiver/estimator_type"][()].decode("utf-8") == "ls"
        assert cfr_est.shape == (1, 1, 1, 1, 1, 8)
        assert cfr_est.shape[1:] == truth.shape
        assert h5["observation/valid_mask"].shape == (1, 1, 1)
        assert h5["observation/detection_success"].shape == (1, 1, 1)
        assert h5["observation/estimation_success"].shape == (1, 1, 1)
        assert h5["observation/snr_db"].shape == (1, 1, 1)
        assert h5["evaluation/nmse_db"].shape == (1, 1, 1)
        assert float(np.median(h5["evaluation/nmse_db"][()])) < -20.0
