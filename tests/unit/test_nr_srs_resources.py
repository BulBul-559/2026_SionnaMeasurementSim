from types import SimpleNamespace

import numpy as np
import pytest

from sionna_measurement_sim.phy.nr_srs_resources import (
    NRSRSHoppingConfig,
    NRSRSPortsConfig,
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
    assert resource.re_symbol_indices.tolist() == [10] * 6 + [11] * 6
    assert resource.re_subcarrier_indices.tolist() == [13, 17, 21, 25, 29, 33] * 2
    assert resource.resource_mask.shape == (14, 48)
    assert int(resource.resource_mask.sum()) == 12
    assert resource.port_tx_ant_map.tolist() == [[0, 0], [1, 1]]


def test_srs_64_prb_comb2_has_384_re_per_symbol():
    cfg = NRSRSResourceConfig(
        start_symbol=12,
        num_srs_symbols=2,
        comb_size=2,
        bwp_start_prb=104,
        bwp_num_prb=64,
    )

    resource = build_srs_resource(
        cfg,
        num_subcarriers=3276,
        num_tx_ant=2,
        default_num_prb=273,
    )

    assert resource.prb_start_per_symbol.tolist() == [104, 104]
    assert resource.prb_count_per_symbol.tolist() == [64, 64]
    assert [int(np.sum(resource.re_symbol_indices == symbol)) for symbol in [12, 13]] == [
        384,
        384,
    ]


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
        sequence_type="nr_zc",
        sequence_id=7,
        group_hopping="enabled",
        sequence_hopping="enabled",
        cyclic_shift_indices=(0, 6),
    )

    first = build_srs_resource(cfg, num_subcarriers=24, num_ports=2, default_num_prb=2)
    second = build_srs_resource(cfg, num_subcarriers=24, num_ports=2, default_num_prb=2)

    np.testing.assert_allclose(first.pilot_symbols, second.pilot_symbols)
    active = first.pilot_symbols[first.pilot_symbols != 0]
    np.testing.assert_allclose(np.abs(active), 1.0, atol=1e-6)
    assert not np.allclose(first.pilot_symbols[0], first.pilot_symbols[1])
    assert first.zc_root_indices.tolist() != [first.zc_root_indices[0]] * 2


def test_srs_explicit_bwp_overflow_fails():
    cfg = NRSRSResourceConfig(bwp_start_prb=1, bwp_num_prb=3)

    with pytest.raises(ValueError, match="exceeds"):
        build_srs_resource(cfg, num_subcarriers=24, num_ports=2, default_num_prb=2)


def test_srs_flatless_config_legacy_fallback_expands_symbols_for_ports():
    phy = SimpleNamespace(num_ofdm_symbols=1)

    cfg = resolve_srs_resource_config(phy, num_subcarriers=24, num_ports=4)

    assert cfg.num_srs_symbols == 4
    assert cfg.start_symbol == 10


def test_srs_hopping_builds_per_symbol_prb_map():
    cfg = NRSRSResourceConfig(
        start_symbol=10,
        num_srs_symbols=2,
        comb_size=2,
        bwp_start_prb=4,
        bwp_num_prb=4,
        hopping=NRSRSHoppingConfig(
            enabled=True,
            frequency_offsets_prb=(0, 8),
            bandwidth_num_prb=(4, 2),
        ),
    )

    resource = build_srs_resource(cfg, num_subcarriers=240, num_tx_ant=2, default_num_prb=20)

    assert resource.prb_start_per_symbol.tolist() == [4, 12]
    assert resource.prb_count_per_symbol.tolist() == [4, 2]
    assert int(np.sum(resource.re_symbol_indices == 10)) == 24
    assert int(np.sum(resource.re_symbol_indices == 11)) == 12


def test_srs_antenna_switching_requires_all_tx_antennas_covered():
    cfg = NRSRSResourceConfig(
        start_symbol=10,
        num_srs_symbols=2,
        ports=NRSRSPortsConfig(
            num_srs_ports=1,
            mapping="antenna_switching",
            port_tx_ant_map=((0, 0),),
        ),
    )

    with pytest.raises(ValueError, match="every TX antenna"):
        build_srs_resource(cfg, num_subcarriers=24, num_tx_ant=2, default_num_prb=2)


def test_srs_time_multiplexing_keeps_symbol_count_guard():
    cfg = NRSRSResourceConfig(num_srs_symbols=1, cyclic_shift_multiplexing="time")

    with pytest.raises(ValueError, match="num_srs_symbols"):
        build_srs_resource(cfg, num_subcarriers=24, num_tx_ant=2, default_num_prb=2)
