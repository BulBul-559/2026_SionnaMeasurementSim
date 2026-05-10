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
  label_file: "data/scenes/test/test5.json"
  scene_file: "data/scenes/test/scene.xml"
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
    assert cfg.input.max_tx == 6
    assert cfg.input.scene_id == "scene"
    assert cfg.input.map_id == ""


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


def test_reject_non_mapping_config(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises((ValueError, SystemExit)):
        load_config(config_path)
