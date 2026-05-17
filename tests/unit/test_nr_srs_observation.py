from types import SimpleNamespace

import numpy as np

from sionna_measurement_sim.domain.link import LinkConfig
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


def test_nr_srs_full_band_ls_recovers_truth_without_noise():
    rng = np.random.default_rng(123)
    truth = (
        rng.standard_normal((1, 1, 2, 4, 8), dtype=np.float32)
        + 1j * rng.standard_normal((1, 1, 2, 4, 8), dtype=np.float32)
    ).astype(np.complex64)

    result = run_nr_srs_observation(
        truth,
        LinkConfig(),
        _phy_config(),
        SimpleNamespace(num_subcarriers=8),
        has_signal=np.ones((1, 1), dtype=np.bool_),
    )

    np.testing.assert_allclose(result["observation"].cfr_est, truth[np.newaxis, ...], atol=1e-5)
    assert result["nr_waveform_spec"].standard == "nr_srs"
    assert result["waveform_grids"]["srs_tx_grid"].shape == (1, 1, 1, 2, 2, 8)
    assert result["waveform_grids"]["srs_rx_grid"].shape == (1, 1, 1, 4, 2, 8)
    assert result["evaluation"].ber == 0.0
    assert result["evaluation"].bler == 0.0


def test_nr_srs_uses_enough_symbols_for_tx_antenna_orthogonality():
    truth = np.ones((1, 1, 4, 2, 8), dtype=np.complex64)

    result = run_nr_srs_observation(
        truth,
        LinkConfig(),
        _phy_config(num_ofdm_symbols=1),
        SimpleNamespace(num_subcarriers=8),
    )

    assert result["nr_waveform_spec"].num_ofdm_symbols == 4
    assert result["waveform_grids"]["srs_pilot_code"].shape == (4, 4)
