from types import SimpleNamespace

import numpy as np

from sionna_measurement_sim.domain.link import LinkConfig
from sionna_measurement_sim.phy.impairments import ImpairmentConfig
from sionna_measurement_sim.phy.nr_srs_observation import run_nr_srs_observation


def _phy_config(**overrides):
    values = {
        "seed": 1,
        "observation_seed": 2,
        "snr_db": 300.0,
        "subcarrier_spacing_khz": 30,
        "num_ofdm_symbols": 2,
        "tx_power_dbm": 0.0,
        "receiver_failure_policy": "mark_invalid",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _srs_config(**overrides):
    values = {
        "slot_length_symbols": 14,
        "start_symbol": 12,
        "num_srs_symbols": 2,
        "comb_size": 2,
        "comb_offset": 0,
        "bwp_start_prb": 0,
        "bwp_num_prb": None,
        "trigger_mode": "aperiodic",
        "periodicity_slots": 1,
        "slot_offset": 0,
        "slot_number": 0,
        "sequence_type": "zc_like",
        "sequence_id": 0,
        "group_hopping": "disabled",
        "sequence_hopping": "disabled",
        "cyclic_shift_multiplexing": "cyclic_shift",
        "cyclic_shift_indices": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_nr_srs_full_band_ls_recovers_truth_without_noise():
    rng = np.random.default_rng(123)
    truth = (
        rng.standard_normal((1, 1, 2, 4, 8), dtype=np.float32)
        + 1j * rng.standard_normal((1, 1, 2, 4, 8), dtype=np.float32)
    ).astype(np.complex64)

    result = run_nr_srs_observation(
        truth,
        LinkConfig(),
            _phy_config(
                srs_config=_srs_config(
                    start_symbol=10,
                    num_srs_symbols=4,
                    comb_size=1,
                    cyclic_shift_multiplexing="time",
                )
            ),
        SimpleNamespace(num_subcarriers=8),
        has_signal=np.ones((1, 1), dtype=np.bool_),
    )

    np.testing.assert_allclose(result["observation"].cfr_est, truth[np.newaxis, ...], atol=1e-5)
    assert result["nr_waveform_spec"].standard == "nr_srs"
    assert result["waveform_grids"]["tx_grid"].shape == (1, 1, 1, 4, 14, 8)
    assert result["waveform_grids"]["rx_grid"].shape == (1, 1, 1, 2, 14, 8)
    assert result["waveform_grids"]["noise_variance"].shape == (1, 1, 1)
    assert result["waveform_grids"]["srs_resource_mask"].shape == (14, 8)
    assert result["waveform_grids"]["srs_pilot_symbols"].shape == (4, 14, 8)
    assert result["waveform_grids"]["srs_re_subcarrier_indices"].tolist() == list(range(8)) * 4
    expected_symbols = [10] * 8 + [11] * 8 + [12] * 8 + [13] * 8
    assert result["waveform_grids"]["srs_re_symbol_indices"].tolist() == expected_symbols
    assert result["waveform_grids"]["cfr_est_resource"].shape == (1, 1, 1, 2, 4, 32)
    assert result["evaluation"].ber == 0.0
    assert result["evaluation"].bler == 0.0


def test_nr_srs_time_mode_fails_when_configured_symbols_cannot_separate_ports():
    truth = np.ones((1, 1, 4, 2, 8), dtype=np.complex64)

    with np.testing.assert_raises_regex(ValueError, "num_srs_symbols"):
        run_nr_srs_observation(
            truth,
            LinkConfig(),
            _phy_config(
                srs_config=_srs_config(
                    start_symbol=12,
                    num_srs_symbols=1,
                    comb_size=1,
                    cyclic_shift_multiplexing="time",
                )
            ),
            SimpleNamespace(num_subcarriers=8),
        )


def test_nr_srs_cyclic_shift_recovers_flat_channels():
    truth = np.zeros((1, 1, 2, 4, 24), dtype=np.complex64)
    values = np.asarray([1.0 + 1.0j, 2.0 - 1.0j, -0.5 + 0.2j, 0.3 - 0.7j], dtype=np.complex64)
    for tx_ant, value in enumerate(values):
        truth[0, 0, :, tx_ant, :] = value

    result = run_nr_srs_observation(
        truth,
        LinkConfig(),
        _phy_config(
            srs_config=_srs_config(
                start_symbol=12,
                num_srs_symbols=2,
                comb_size=1,
                sequence_type="nr_zc",
                group_hopping="enabled",
                sequence_hopping="enabled",
            )
        ),
        SimpleNamespace(num_subcarriers=24),
    )

    np.testing.assert_allclose(result["observation"].cfr_est, truth[np.newaxis, ...], atol=1e-5)
    assert result["waveform_grids"]["srs_cyclic_shift_indices"].tolist() == [0, 3, 6, 9]


def test_nr_srs_time_comb_resource_ls_recovers_truth_on_srs_re():
    rng = np.random.default_rng(456)
    truth = (
        rng.standard_normal((1, 1, 2, 2, 24), dtype=np.float32)
        + 1j * rng.standard_normal((1, 1, 2, 2, 24), dtype=np.float32)
    ).astype(np.complex64)

    result = run_nr_srs_observation(
        truth,
        LinkConfig(),
        _phy_config(srs_config=_srs_config(comb_size=2, cyclic_shift_multiplexing="time")),
        SimpleNamespace(num_subcarriers=24),
    )

    re_indices = result["waveform_grids"]["srs_re_subcarrier_indices"]
    np.testing.assert_allclose(
        result["waveform_grids"]["cfr_est_resource"],
        truth[np.newaxis, ..., re_indices],
        atol=1e-5,
    )


def test_nr_srs_common_impairments_change_estimate_and_metadata():
    rng = np.random.default_rng(321)
    truth = (
        rng.standard_normal((1, 1, 2, 2, 32), dtype=np.float32)
        + 1j * rng.standard_normal((1, 1, 2, 2, 32), dtype=np.float32)
    ).astype(np.complex64)

    baseline = run_nr_srs_observation(
        truth,
        LinkConfig(),
        _phy_config(snr_db=300.0, srs_config=_srs_config(comb_size=1)),
        SimpleNamespace(num_subcarriers=32),
    )
    impaired = run_nr_srs_observation(
        truth,
        LinkConfig(),
        _phy_config(
            snr_db=300.0,
            impairment_config=ImpairmentConfig(
                random_seed=5,
                cfo_hz=500.0,
                timing_offset_samples=2.0,
            ),
            srs_config=_srs_config(comb_size=1),
        ),
        SimpleNamespace(num_subcarriers=32),
    )

    assert not np.allclose(
        impaired["observation"].cfr_est,
        baseline["observation"].cfr_est,
        atol=1e-6,
    )
    assert np.all(impaired["observation"].cfo_hz == 500.0)
    assert np.all(impaired["observation"].timing_offset_samples == 2.0)


def test_nr_srs_tx_power_scales_waveform_rssi_but_receiver_descales_cfr():
    truth = np.ones((1, 1, 1, 1, 8), dtype=np.complex64)
    common = dict(
        srs_config=_srs_config(
            start_symbol=12,
            num_srs_symbols=1,
            comb_size=1,
            cyclic_shift_multiplexing="cyclic_shift",
        ),
        snr_db=300.0,
    )

    result_0 = run_nr_srs_observation(
        truth,
        LinkConfig(),
        _phy_config(tx_power_dbm=0.0, **common),
        SimpleNamespace(num_subcarriers=8),
    )
    result_23 = run_nr_srs_observation(
        truth,
        LinkConfig(),
        _phy_config(tx_power_dbm=23.0, **common),
        SimpleNamespace(num_subcarriers=8),
    )

    tx_delta_db = 20.0 * np.log10(
        np.max(np.abs(result_23["waveform_grids"]["tx_grid"]))
        / np.max(np.abs(result_0["waveform_grids"]["tx_grid"]))
    )
    rx_delta_db = 20.0 * np.log10(
        np.max(np.abs(result_23["waveform_grids"]["rx_grid"]))
        / np.max(np.abs(result_0["waveform_grids"]["rx_grid"]))
    )
    rssi_delta_db = float(
        result_23["observation"].rssi_dbm[0, 0, 0]
        - result_0["observation"].rssi_dbm[0, 0, 0]
    )

    np.testing.assert_allclose(tx_delta_db, 23.0, atol=1e-4)
    np.testing.assert_allclose(rx_delta_db, 23.0, atol=1e-4)
    np.testing.assert_allclose(rssi_delta_db, 23.0, atol=1e-4)
    np.testing.assert_allclose(result_23["observation"].cfr_est, truth[np.newaxis, ...], atol=1e-5)
    np.testing.assert_allclose(
        result_23["waveform_grids"]["tx_power_scale_linear"],
        10.0 ** (23.0 / 20.0),
    )
