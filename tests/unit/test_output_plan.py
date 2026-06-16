from pathlib import Path

import pytest

from sionna_measurement_sim.domain.constants import (
    FULL_CONTRACT_NAME,
    IQ_LINK_LIBRARY_CONTRACT_NAME,
    OUTPUT_PRODUCT_CFR_TRUTH,
    OUTPUT_PRODUCT_DERIVED,
    OUTPUT_PRODUCT_LINK_LABELS,
    OUTPUT_PRODUCT_RANGING,
    RT_LABELS_CONTRACT_NAME,
)
from sionna_measurement_sim.domain.output_plan import build_rt_output_plan
from sionna_measurement_sim.rt.truth_pipeline import (
    RTTruthRunConfig,
    _normalize_output_profile_config,
)
from sionna_measurement_sim.visualization.config import VisualizationRunConfig


def test_full_output_plan_uses_full_contract():
    plan = build_rt_output_plan("full")

    assert plan.profile == "full"
    assert plan.contract_name == FULL_CONTRACT_NAME
    assert plan.compute_cfr is True
    assert plan.compute_cir is True
    assert plan.compute_path_samples is True
    assert plan.write_full_contract is True
    assert plan.write_compact_link_labels is False


@pytest.mark.parametrize("profile", ["rt_lite", "custom"])
def test_removed_non_contract_profiles_fail_fast(profile: str):
    with pytest.raises(ValueError, match="output.profile"):
        build_rt_output_plan(profile)


def test_rt_labels_only_skips_heavy_truth_arrays():
    plan = build_rt_output_plan("rt_labels_only")

    assert plan.profile == "rt_labels_only"
    assert plan.contract_name == RT_LABELS_CONTRACT_NAME
    assert plan.compute_cfr is False
    assert plan.compute_cir is False
    assert plan.compute_path_samples is False
    assert plan.compute_nlos_truth is False
    assert plan.write_full_contract is False
    assert plan.write_compact_link_labels is True


def test_iq_link_library_keeps_cfr_but_skips_full_contract():
    plan = build_rt_output_plan("iq_link_library")

    assert plan.profile == "iq_link_library"
    assert plan.contract_name == IQ_LINK_LIBRARY_CONTRACT_NAME
    assert plan.compute_cfr is True
    assert plan.compute_cir is False
    assert plan.compute_path_samples is False
    assert plan.compute_nlos_truth is False
    assert plan.write_full_contract is False
    assert plan.write_iq_link_library is True


def test_unknown_output_profile_fails_fast():
    with pytest.raises(ValueError, match="output.profile"):
        build_rt_output_plan("tiny")


def test_product_full_cfr_truth_product_skips_downstream_compute():
    plan = build_rt_output_plan("full", products=["cfr_truth"])

    assert plan.profile == "full"
    assert plan.contract_name == FULL_CONTRACT_NAME
    assert plan.products == (OUTPUT_PRODUCT_CFR_TRUTH,)
    assert plan.is_product_aware_full is True
    assert plan.compute_cfr is True
    assert plan.compute_cir is False
    assert plan.compute_path_samples is False
    assert plan.compute_nlos_truth is False
    assert plan.requires_phy_observation is False
    assert plan.write_cfr_truth is True
    assert plan.write_cfr_observation is False


def test_product_full_rtt_alias_selects_ranging_and_requires_phy():
    plan = build_rt_output_plan("full", products=["rtt"])

    assert plan.products == (OUTPUT_PRODUCT_RANGING,)
    assert plan.compute_cfr is True
    assert plan.requires_phy_observation is True
    assert plan.write_ranging is True


def test_full_product_link_labels_uses_full_contract_without_cfr():
    plan = build_rt_output_plan("full", products=["derived", "link_labels"])

    assert plan.profile == "full"
    assert plan.contract_name == FULL_CONTRACT_NAME
    assert plan.products == (OUTPUT_PRODUCT_DERIVED, OUTPUT_PRODUCT_LINK_LABELS)
    assert plan.compute_cfr is False
    assert plan.compute_cir is False
    assert plan.compute_path_samples is False
    assert plan.write_link_labels is True
    assert plan.requires_phy_observation is False


def test_compact_profiles_cannot_mix_explicit_products():
    with pytest.raises(ValueError, match="output.products"):
        build_rt_output_plan("rt_labels_only", products=["cfr_truth"])


def test_product_full_array_truth_source_does_not_require_phy():
    plan = build_rt_output_plan(
        "full",
        products=["array"],
        array_sources=["truth_cfr"],
    )

    assert plan.write_array_outputs is True
    assert plan.compute_cfr is True
    assert plan.requires_phy_observation is False


def test_product_full_array_observation_source_requires_phy():
    plan = build_rt_output_plan(
        "full",
        products=["array"],
        array_sources=["cfr_est"],
    )

    assert plan.write_array_outputs is True
    assert plan.compute_cfr is True
    assert plan.requires_phy_observation is True


def test_product_full_iq_product_requires_phy():
    plan = build_rt_output_plan("full", products=["iq"])

    assert plan.write_iq is True
    assert plan.compute_cfr is True
    assert plan.requires_phy_observation is True


def test_product_full_multiuser_product_requires_phy():
    plan = build_rt_output_plan("full", products=["multiuser"])

    assert plan.write_multiuser is True
    assert plan.compute_cfr is True
    assert plan.requires_phy_observation is True


def test_product_full_calibration_product_requires_phy():
    plan = build_rt_output_plan("full", products=["calibration"])

    assert plan.write_calibration is True
    assert plan.compute_cfr is True
    assert plan.requires_phy_observation is True


def test_product_full_motion_product_does_not_require_phy():
    plan = build_rt_output_plan("full", products=["motion"])

    assert plan.write_motion is True
    assert plan.compute_cfr is False
    assert plan.requires_phy_observation is False


def test_product_full_visualization_product_enables_visualization_config():
    cfg = _normalize_output_profile_config(
        RTTruthRunConfig(
            label_file=Path("tests/fixtures/scenes/test/test5.json"),
            scene_file=Path("tests/fixtures/scenes/test/scene.xml"),
            output_dir=Path("outputs/test"),
            output_profile="full",
            output_products=("visualization",),
            visualization_config=VisualizationRunConfig(enabled=False),
        )
    )

    assert cfg.visualization_config.enabled is True
