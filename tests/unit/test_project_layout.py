from pathlib import Path


def test_required_phase0_paths_exist():
    root = Path(__file__).resolve().parents[2]

    assert (root / "docs").is_dir()
    assert (root / "data/scenes/test").is_dir()
    assert (root / "sionna_measurement_sim").is_dir()
    assert (root / "pyproject.toml").is_file()
    assert (root / ".python-version").read_text(encoding="utf-8").strip().startswith("3.11")
