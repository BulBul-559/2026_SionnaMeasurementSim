from pathlib import Path

import pytest

from sionna_measurement_sim.config.loader import load_config


def test_load_yaml_config(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("experiment_id: demo\nrandom_seed: 7\n", encoding="utf-8")

    assert load_config(config_path) == {"experiment_id": "demo", "random_seed": 7}


def test_reject_non_mapping_config(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Configuration root must be a mapping"):
        load_config(config_path)
