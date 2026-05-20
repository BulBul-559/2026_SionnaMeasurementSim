from types import SimpleNamespace

import numpy as np
import pytest

from sionna_measurement_sim.phy.nr_srs_resources import (
    NRSRSResourceConfig,
    build_srs_resource,
    resolve_srs_resource_config,
)


def test_srs_comb_and_bwp_mapping():
    cfg = NRSRSResourceConfig(
        start_symbol=10,
        num_srs_symbols=2,
        comb_size=4,
        comb_offset=1,
        bwp_start_prb=1,
        bwp_num_prb=2,
    )

    resource = build_srs_resource(
        cfg,
        num_subcarriers=48,
        num_ports=2,
        default_num_prb=4,
    )

    assert resource.srs_symbol_indices.tolist() == [10, 11]
    assert resource.re_subcarrier_indices.tolist() == [13, 17, 21, 25, 29, 33]
    assert resource.resource_mask.shape == (14, 48)
    assert int(resource.resource_mask.sum()) == 12


def test_srs_unscheduled_slot_fails_fast():
    cfg = NRSRSResourceConfig(
        trigger_mode="periodic",
        periodicity_slots=4,
        slot_offset=1,
        slot_number=2,
    )

    with pytest.raises(ValueError, match="not scheduled"):
        build_srs_resource(cfg, num_subcarriers=24, num_ports=2, default_num_prb=2)


def test_srs_sequence_and_cyclic_shift_are_reproducible_and_unit_magnitude():
    cfg = NRSRSResourceConfig(
        start_symbol=12,
        num_srs_symbols=2,
        comb_size=2,
        sequence_id=7,
        cyclic_shift_indices=(0, 3),
    )

    first = build_srs_resource(cfg, num_subcarriers=24, num_ports=2, default_num_prb=2)
    second = build_srs_resource(cfg, num_subcarriers=24, num_ports=2, default_num_prb=2)

    np.testing.assert_allclose(first.pilot_symbols, second.pilot_symbols)
    active = first.pilot_symbols[first.pilot_symbols != 0]
    np.testing.assert_allclose(np.abs(active), 1.0, atol=1e-6)
    assert not np.allclose(first.pilot_symbols[0], first.pilot_symbols[1])


def test_srs_explicit_bwp_overflow_fails():
    cfg = NRSRSResourceConfig(bwp_start_prb=1, bwp_num_prb=3)

    with pytest.raises(ValueError, match="exceeds"):
        build_srs_resource(cfg, num_subcarriers=24, num_ports=2, default_num_prb=2)


def test_srs_flatless_config_legacy_fallback_expands_symbols_for_ports():
    phy = SimpleNamespace(num_ofdm_symbols=1)

    cfg = resolve_srs_resource_config(phy, num_subcarriers=24, num_ports=4)

    assert cfg.num_srs_symbols == 4
    assert cfg.start_symbol == 10
