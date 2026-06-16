from types import SimpleNamespace

from sionna_measurement_sim.rt.truth_pipeline import _clean_iq_link_library_config


def test_iq_link_library_defaults_to_time_clean():
    cfg = _clean_iq_link_library_config(None, cp_length=8)

    assert cfg.enabled is True
    assert cfg.clean_output == "time"
    assert not hasattr(cfg, "save_frequency_clean")
    assert not hasattr(cfg, "save_time_clean")
    assert cfg.save_frequency_observed is False
    assert cfg.save_time_observed is False
    assert cfg.cp_length == 8


def test_iq_link_library_clean_output_frequency_only():
    cfg = _clean_iq_link_library_config(
        SimpleNamespace(
            enabled=True,
            clean_output="frequency",
            save_frequency_observed=False,
            save_time_observed=False,
            cp_length=None,
        ),
        cp_length=8,
    )

    assert cfg.enabled is True
    assert cfg.clean_output == "frequency"
    assert not hasattr(cfg, "save_frequency_clean")
    assert not hasattr(cfg, "save_time_clean")
    assert cfg.save_frequency_observed is False
    assert cfg.save_time_observed is False
    assert cfg.cp_length == 8


def test_iq_link_library_clean_output_both():
    cfg = _clean_iq_link_library_config(
        SimpleNamespace(
            enabled=True,
            clean_output="both",
            save_frequency_observed=False,
            save_time_observed=False,
            cp_length=0,
        ),
        cp_length=8,
    )

    assert cfg.clean_output == "both"
    assert not hasattr(cfg, "save_frequency_clean")
    assert not hasattr(cfg, "save_time_clean")
    assert cfg.cp_length == 0
