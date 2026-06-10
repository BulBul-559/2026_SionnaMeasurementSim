import tomllib
from pathlib import Path

from sionna_measurement_sim import __version__
from sionna_measurement_sim.app.cli import main


def test_package_version_matches_project_metadata(capsys):
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert __version__ == pyproject["project"]["version"]

    try:
        main(["--version"])
    except SystemExit as exc:
        assert exc.code == 0
    assert f"sionna-measurement-sim {__version__}" in capsys.readouterr().out


def test_cli_help_returns_zero(capsys):
    try:
        main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    captured = capsys.readouterr()
    assert "SionnaMeasurementSim command line interface" in captured.out


def test_cli_preflight_returns_zero(capsys):
    assert main(["preflight"]) == 0
    captured = capsys.readouterr()
    assert "python:" in captured.out


def test_run_full_cli_values_override_yaml(tmp_path, monkeypatch, capsys):
    import yaml

    from sionna_measurement_sim.config.loader import load_config

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "runtime:",
                "  seed: 7",
                "input:",
                "  max_bs: 3",
                "  max_ue: 4",
                "motion:",
                "  enabled: false",
                "  num_time_steps: 1",
                "  sampling_frequency_hz: 0.0",
                "impairments:",
                "  cfo:",
                "    enabled: false",
                "    cfo_hz: null",
                "output:",
                f"  root_dir: {tmp_path / 'from_yaml'}",
            ]
        ),
        encoding="utf-8",
    )
    captured_config = {}

    def _fake_run(config):
        captured_config["value"] = config
        return tmp_path / "out" / "results.h5"

    import sionna_measurement_sim.rt.truth_pipeline as truth_pipeline

    monkeypatch.setattr(truth_pipeline, "run_rt_truth_pipeline", _fake_run)

    assert main(
        [
            "--config",
            str(config_path),
            "run-full",
            "--max-bs",
            "6",
            "--max-ue",
            "8",
            "--seed",
            "99",
            "--cfo-hz",
            "123.0",
            "--num-time-steps",
            "5",
            "--sampling-frequency-hz",
            "250.0",
            "--output-dir",
            str(tmp_path / "from_cli"),
        ]
    ) == 0

    run_config = captured_config["value"]
    assert run_config.max_bs == 6
    assert run_config.max_ue == 8
    assert run_config.seed == 99
    assert run_config.output_dir == tmp_path / "from_cli"
    assert run_config.impairment_config.cfo_hz == 123.0
    assert run_config.num_time_steps == 5
    assert run_config.sampling_frequency_hz == 250.0
    output_config_path = tmp_path / "from_cli" / "run_config.yaml"
    written_config = yaml.safe_load(output_config_path.read_text())
    assert written_config["runtime"]["seed"] == 99
    assert written_config["input"]["max_bs"] == 6
    assert written_config["input"]["max_ue"] == 8
    assert written_config["output"]["root_dir"] == str(tmp_path / "from_cli")
    assert written_config["motion"]["enabled"] is True
    assert written_config["motion"]["num_time_steps"] == 5
    assert written_config["motion"]["sampling_frequency_hz"] == 250.0
    assert written_config["impairments"]["cfo"]["enabled"] is True
    assert written_config["impairments"]["cfo"]["cfo_hz"] == 123.0
    reloaded_config = load_config(output_config_path)
    assert reloaded_config.runtime.seed == 99
    assert reloaded_config.output.root_dir == str(tmp_path / "from_cli")
    assert str(tmp_path / "out" / "results.h5") in capsys.readouterr().out


def test_run_full_rt_labels_profile_disables_heavy_branches(tmp_path, monkeypatch):
    import yaml

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "output:",
                "  profile: rt_labels_only",
                f"  root_dir: {tmp_path / 'labels'}",
                "phy:",
                "  enabled: true",
                "array:",
                "  spectrum:",
                "    enabled: true",
                "ranging:",
                "  enabled: true",
                "visualization:",
                "  enabled: true",
                "calibration:",
                "  enabled: true",
            ]
        ),
        encoding="utf-8",
    )
    captured_config = {}

    def _fake_run(config):
        captured_config["value"] = config
        return config.output_dir / "results.h5"

    import sionna_measurement_sim.rt.truth_pipeline as truth_pipeline

    monkeypatch.setattr(truth_pipeline, "run_rt_truth_pipeline", _fake_run)

    assert main(["--config", str(config_path), "run-full"]) == 0

    run_config = captured_config["value"]
    assert run_config.output_profile == "rt_labels_only"
    assert run_config.observation_snr_db is None
    assert run_config.ranging_config.enabled is False
    assert run_config.spectrum_config.enabled is False
    assert run_config.visualization_config.enabled is False
    assert run_config.calibration_enabled is False
    written_config = yaml.safe_load((tmp_path / "labels" / "run_config.yaml").read_text())
    assert written_config["output"]["profile"] == "rt_labels_only"
    assert written_config["phy"]["enabled"] is False
    assert written_config["ranging"]["enabled"] is False
    assert written_config["array"]["spectrum"]["enabled"] is False
    assert written_config["visualization"]["enabled"] is False
    assert written_config["calibration"]["enabled"] is False


def test_run_full_without_yaml_writes_run_config(tmp_path, monkeypatch):
    from sionna_measurement_sim.config.loader import load_config

    def _fake_run(config):
        return config.output_dir / "results.h5"

    import sionna_measurement_sim.rt.truth_pipeline as truth_pipeline

    monkeypatch.setattr(truth_pipeline, "run_rt_truth_pipeline", _fake_run)

    output_dir = tmp_path / "default_cli"
    assert main(["run-full", "--output-dir", str(output_dir), "--max-ue", "2"]) == 0

    written_config = load_config(output_dir / "run_config.yaml")
    assert written_config.output.root_dir == str(output_dir)
    assert written_config.input.max_ue == 2
    assert written_config.phy.standard == "custom_ofdm"
