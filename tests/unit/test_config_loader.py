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
    assert cfg.phy.srs.multiuser.enabled is False
    assert cfg.phy.srs.multiuser.resource_strategy == "comb_offset"


def test_output_sharding_bundle_config_loads_append_mode(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
output:
  sharding:
    enabled: true
    shard_size: 5
    parallel_workers: 2
    bundle:
      enabled: true
      max_planned_shards_per_bundle: 4
      bundles_dir: bundles_exp
      filename_pattern: "bundle_{worker_index:02d}_{bundle_index:04d}.h5"
      validate_schema: false
""",
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert cfg.output.sharding.bundle.enabled is True
    assert cfg.output.sharding.bundle.max_planned_shards_per_bundle == 4
    assert cfg.output.sharding.bundle.bundles_dir == "bundles_exp"
    assert cfg.output.sharding.bundle.filename_pattern == (
        "bundle_{worker_index:02d}_{bundle_index:04d}.h5"
    )
    assert cfg.output.sharding.bundle.validate_schema is False


def test_output_sharding_bundle_rejects_absolute_bundle_dir(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
output:
  sharding:
    enabled: true
    bundle:
      enabled: true
      bundles_dir: "/tmp/bundles"
""",
        encoding="utf-8",
    )

    with pytest.raises((ValueError, SystemExit), match="bundles_dir"):
        load_config(config_path)


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


def test_spectrum_config_rejects_srs_cfr_est_source(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
array:
  spectrum:
    sources: ["truth_cfr", "srs_cfr_est"]
""",
        encoding="utf-8",
    )

    with pytest.raises((ValueError, SystemExit)):
        load_config(config_path)


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


def test_product_full_array_truth_source_allows_phy_disabled(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
output:
  profile: "full"
  products: ["array"]
array:
  spectrum:
    sources: ["truth_cfr"]
phy:
  enabled: false
""",
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert cfg.output.products == ["array"]
    assert cfg.array.spectrum.sources == ["truth_cfr"]
    assert cfg.phy.enabled is False


def test_product_full_array_observation_source_requires_phy_enabled(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
output:
  profile: "full"
  products: ["array"]
array:
  spectrum:
    sources: ["cfr_est"]
phy:
  enabled: false
""",
        encoding="utf-8",
    )

    with pytest.raises((ValueError, SystemExit), match="phy.enabled"):
        load_config(config_path)


def test_product_full_iq_product_allows_nr_srs_without_explicit_iq_config(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
output:
  profile: "full"
  products: ["iq"]
phy:
  enabled: true
  standard: "nr_srs"
""",
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert cfg.output.products == ["iq"]
    assert cfg.phy.standard == "nr_srs"
    assert cfg.phy.iq.enabled is False


def test_product_full_iq_product_rejects_custom_ofdm(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
output:
  profile: "full"
  products: ["iq"]
phy:
  enabled: true
  standard: "custom_ofdm"
""",
        encoding="utf-8",
    )

    with pytest.raises((ValueError, SystemExit), match="standard"):
        load_config(config_path)


def test_product_full_multiuser_product_requires_nr_srs(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
output:
  profile: "full"
  products: ["multiuser"]
phy:
  enabled: true
  standard: "nr_pusch"
""",
        encoding="utf-8",
    )

    with pytest.raises((ValueError, SystemExit), match="multiuser"):
        load_config(config_path)


def test_product_full_multiuser_product_allows_nr_srs_defaults(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
output:
  profile: "full"
  products: ["multiuser"]
phy:
  enabled: true
  standard: "nr_srs"
""",
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert cfg.output.products == ["multiuser"]
    assert cfg.phy.standard == "nr_srs"
    assert cfg.phy.srs.multiuser.enabled is False


def test_product_full_calibration_product_requires_phy_enabled(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
output:
  profile: "full"
  products: ["calibration"]
phy:
  enabled: false
""",
        encoding="utf-8",
    )

    with pytest.raises((ValueError, SystemExit), match="phy.enabled"):
        load_config(config_path)


def test_product_full_motion_product_allows_phy_disabled(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
output:
  profile: "full"
  products: ["motion"]
phy:
  enabled: false
""",
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert cfg.output.products == ["motion"]
    assert cfg.phy.enabled is False


def test_phy_iq_clean_output_is_canonical_clean_selector(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
phy:
  enabled: true
  iq:
    enabled: true
    clean_output: both
""",
        encoding="utf-8",
    )

    cfg = load_config(config_path)

    assert cfg.phy.iq.clean_output == "both"
    assert not hasattr(cfg.phy.iq, "save_frequency_clean")
    assert not hasattr(cfg.phy.iq, "save_time_clean")
    assert cfg.phy.iq.save_frequency_observed is False
    assert cfg.phy.iq.save_time_observed is False


def test_phy_iq_rejects_legacy_clean_save_flags(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
phy:
  enabled: true
  iq:
    enabled: true
    save_frequency_clean: true
""",
        encoding="utf-8",
    )

    with pytest.raises((ValueError, SystemExit), match="save_frequency_clean"):
        load_config(config_path)


def test_phy_iq_clean_output_rejects_unknown_value(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
phy:
  enabled: true
  iq:
    enabled: true
    clean_output: raw
""",
        encoding="utf-8",
    )

    with pytest.raises((ValueError, SystemExit), match="clean_output"):
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
