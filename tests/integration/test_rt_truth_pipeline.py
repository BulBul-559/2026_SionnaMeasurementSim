from pathlib import Path

import h5py
import numpy as np

from sionna_measurement_sim.io.hdf5_reader import read_truth_cfr
from sionna_measurement_sim.rt.truth_pipeline import RTTruthRunConfig, run_rt_truth_pipeline


def test_rt_truth_pipeline_writes_hdf5_manifest_and_log(tmp_path: Path):
    output_dir = tmp_path / "phase2_rt_truth"

    results_path = run_rt_truth_pipeline(
        RTTruthRunConfig(
            label_file=Path("data/scenes/test/test5.json"),
            scene_file=Path("data/scenes/test/scene.xml"),
            output_dir=output_dir,
            num_subcarriers=8,
            seed=1,
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
        assert np.any(np.isfinite(h_true[()]))

    readback = read_truth_cfr(results_path)
    assert readback.shape == (1, 1, 1, 1, 8)
    assert readback.dtype == np.dtype("complex64")
