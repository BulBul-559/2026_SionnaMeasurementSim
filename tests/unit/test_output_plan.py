import pytest

from sionna_measurement_sim.domain.constants import (
    FULL_CONTRACT_NAME,
    RT_LABELS_CONTRACT_NAME,
)
from sionna_measurement_sim.domain.output_plan import build_rt_output_plan


def test_full_output_plan_uses_full_contract():
    plan = build_rt_output_plan("full")

    assert plan.profile == "full"
    assert plan.contract_name == FULL_CONTRACT_NAME
    assert plan.compute_cfr is True
    assert plan.compute_cir is True
    assert plan.compute_path_samples is True
    assert plan.write_full_contract is True
    assert plan.write_compact_link_labels is False


def test_rt_lite_keeps_full_contract_but_uses_profile_marker():
    plan = build_rt_output_plan("rt_lite")

    assert plan.profile == "rt_lite"
    assert plan.contract_name == FULL_CONTRACT_NAME
    assert plan.compute_cfr is True
    assert plan.compute_cir is True
    assert plan.compute_path_samples is True
    assert plan.write_full_contract is True


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


def test_unknown_output_profile_fails_fast():
    with pytest.raises(ValueError, match="output.profile"):
        build_rt_output_plan("tiny")
