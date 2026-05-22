import json
from pathlib import Path

import pytest

from sionna_measurement_sim.app.cli import main
from sionna_measurement_sim.config.schema import DebugConfig
from sionna_measurement_sim.rt.truth_pipeline import RTTruthRunConfig, run_rt_truth_pipeline


def _load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_benchmark_write_cli_outputs_summary(tmp_path: Path):
    output_dir = tmp_path / "write"

    assert main(
        [
            "benchmark",
            "write",
            "--output-dir",
            str(output_dir),
            "--tx-count",
            "1",
            "--rx-count",
            "2",
            "--rx-ant",
            "2",
            "--tx-ant",
            "1",
            "--subcarriers",
            "16",
            "--include-waveform",
            "--include-array",
            "--include-ranging",
            "--compression",
            "none",
            "--no-write-hardware-samples",
        ]
    ) == 0

    summary = _load_summary(output_dir / "benchmark_summary.json")
    assert summary["benchmark_type"] == "write"
    assert summary["status"] == "success"
    assert summary["perf_summary"]["dataset_write_summary"]["dataset_count"] > 0
    assert (output_dir / "benchmark_rows.csv").exists()
    assert (output_dir / "config_snapshot.json").exists()


def test_benchmark_spectrum_cli_outputs_summary(tmp_path: Path):
    output_dir = tmp_path / "spectrum"

    assert main(
        [
            "benchmark",
            "spectrum",
            "--output-dir",
            str(output_dir),
            "--links",
            "4",
            "--rx-ant",
            "4",
            "--subcarriers",
            "32",
            "--zenith-bins",
            "9",
            "--azimuth-bins",
            "13",
            "--sources",
            "truth_cfr,cfr_est,rx_grid",
            "--no-write-hardware-samples",
        ]
    ) == 0

    summary = _load_summary(output_dir / "benchmark_summary.json")
    assert summary["benchmark_type"] == "spectrum"
    assert summary["status"] == "success"
    assert summary["iterations"][0]["finite_rate_min"] == 1.0
    assert summary["iterations"][0]["source_count"] == 3


def test_benchmark_rt_cli_fixture_outputs_summary(tmp_path: Path):
    output_dir = tmp_path / "rt"

    assert main(
        [
            "benchmark",
            "rt",
            "--output-dir",
            str(output_dir),
            "--label-file",
            "tests/fixtures/scenes/test/test5.json",
            "--scene-file",
            "tests/fixtures/scenes/test/scene.xml",
            "--max-bs",
            "1",
            "--max-ue",
            "2",
            "--num-subcarriers",
            "8",
            "--max-depth",
            "1",
            "--no-write-hardware-samples",
        ]
    ) == 0

    summary = _load_summary(output_dir / "benchmark_summary.json")
    assert summary["benchmark_type"] == "rt"
    assert summary["status"] == "success"
    assert summary["iterations"][0]["path_count"] >= 0
    assert summary["iterations"][0]["truth_cfr_shape"] == [2, 1, 1, 1, 8]


def test_debug_tracing_writes_failure_summary_on_pipeline_error(tmp_path: Path):
    output_dir = tmp_path / "failed_pipeline"

    with pytest.raises(FileNotFoundError):
        run_rt_truth_pipeline(
            RTTruthRunConfig(
                label_file=tmp_path / "missing_label.json",
                scene_file=Path("tests/fixtures/scenes/test/scene.xml"),
                output_dir=output_dir,
                debug_config=DebugConfig(enabled=True, write_hardware_samples=False),
            )
        )

    summary = _load_summary(output_dir / "logs" / "perf_summary.json")
    assert summary["status"] == "failed"
    assert summary["exception"]["type"] == "FileNotFoundError"
