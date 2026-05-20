from pathlib import Path

import pytest

from sionna_measurement_sim.config.loader import load_config
from sionna_measurement_sim.config.schema import MeasurementConfig


def test_load_yaml_config(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    full_yaml = """
runtime:
  seed: 1
input:
  label_file: "tests/fixtures/scenes/test/test5.json"
  scene_file: "tests/fixtures/scenes/test/scene.xml"
carrier:
  center_frequency_hz: 3500000000.0
  bandwidth_hz: 20000000.0
  num_subcarriers: 64
phy:
  enabled: true
  snr_db: 30.0
  fft_size: 64
motion:
  enabled: true
  num_time_steps: 1
  sampling_frequency_hz: 0.0
"""
    config_path.write_text(full_yaml, encoding="utf-8")
    cfg = load_config(config_path)
    assert isinstance(cfg, MeasurementConfig)
    assert cfg.runtime.seed == 1
    assert cfg.carrier.num_subcarriers == 64


def test_load_default_mvp_config():
    cfg = load_config("config/defaults/measurement_mvp.yaml")
    assert isinstance(cfg, MeasurementConfig)
    assert cfg.runtime.seed == 42
    assert cfg.input.max_bs == 6
    assert cfg.input.max_ue == 100
    assert cfg.input.scene_id == "scene"
    assert cfg.input.map_id == ""
    assert cfg.visualization.enabled is True
    assert cfg.visualization.output_dir == "figures"
    assert cfg.visualization.sample_policy == "valid_links_first"
    assert cfg.visualization.max_bs == 5
    assert cfg.visualization.sample_ue_count == 3
    assert cfg.array.spectrum.sources == ["truth_cfr", "cfr_est", "rx_grid"]
    assert cfg.ranging.enabled is False
    assert cfg.ranging.estimators == ["pdp_peak", "phase_slope"]
    assert cfg.phy.srs.slot_length_symbols == 14
    assert cfg.phy.srs.start_symbol == 12
    assert cfg.phy.srs.num_srs_symbols == 2
    assert cfg.phy.srs.comb_size == 2


def test_old_tx_rx_config_fields_are_rejected(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
input:
  max_tx: 1
  max_rx: 1
antenna:
  tx_array: {}
  rx_array: {}
motion:
  tx_velocity_mps: [0.0, 0.0, 0.0]
  rx_velocity_mps: [0.0, 0.0, 0.0]
""",
        encoding="utf-8",
    )

    with pytest.raises((ValueError, SystemExit)):
        load_config(config_path)


def test_input_scene_id_defaults_to_scene_file_stem(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
input:
  scene_file: "data/scenes/campus/block_a.xml"
""",
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert cfg.input.scene_id == "block_a"
    assert cfg.input.map_id == ""


def test_input_scene_id_allows_explicit_value(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
input:
  scene_file: "data/scenes/campus/block_a.xml"
  scene_id: "campus_block_a_v2"
  map_id: "campus"
""",
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert cfg.input.scene_id == "campus_block_a_v2"
    assert cfg.input.map_id == "campus"


def test_spectrum_config_accepts_cfr_est_source(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
array:
  spectrum:
    sources: ["truth_cfr", "cfr_est", "rx_grid"]
""",
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert cfg.array.spectrum.sources == ["truth_cfr", "cfr_est", "rx_grid"]


def test_spectrum_config_accepts_srs_cfr_est_source(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
array:
  spectrum:
    sources: ["truth_cfr", "srs_cfr_est"]
""",
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert cfg.array.spectrum.sources == ["truth_cfr", "srs_cfr_est"]


def test_spectrum_config_rejects_unknown_source(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
array:
  spectrum:
    sources: ["truth_cfr", "unknown"]
""",
        encoding="utf-8",
    )

    with pytest.raises((ValueError, SystemExit)):
        load_config(config_path)


def test_ranging_requires_phy_observation_when_enabled(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
phy:
  enabled: false
ranging:
  enabled: true
""",
        encoding="utf-8",
    )

    with pytest.raises((ValueError, SystemExit), match="ranging.enabled"):
        load_config(config_path)


def test_reject_non_mapping_config(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises((ValueError, SystemExit)):
        load_config(config_path)
