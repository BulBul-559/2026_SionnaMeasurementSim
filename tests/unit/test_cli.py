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
