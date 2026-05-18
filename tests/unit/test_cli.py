from sionna_measurement_sim.app.cli import main


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
    assert str(tmp_path / "out" / "results.h5") in capsys.readouterr().out
