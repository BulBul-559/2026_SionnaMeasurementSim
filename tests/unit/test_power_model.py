from types import SimpleNamespace

import numpy as np
import pytest

from sionna_measurement_sim.phy.power import (
    amplitude_scale_from_dbm,
    compute_uplink_power,
    dbm_to_mw,
    mw_to_dbm,
    thermal_noise_dbm,
)


def test_dbm_mw_and_amplitude_scale_reference_zero_dbm():
    assert float(dbm_to_mw(0.0)) == pytest.approx(1.0)
    assert float(dbm_to_mw(23.0)) == pytest.approx(199.526, rel=1e-3)
    assert float(mw_to_dbm(1.0)) == pytest.approx(0.0)
    assert float(amplitude_scale_from_dbm(23.0)) == pytest.approx(np.sqrt(199.526), rel=1e-3)


def test_fixed_tx_power_scales_grid_even_without_power_control():
    result = compute_uplink_power(
        path_power_db=None,
        snapshot_count=1,
        tx_count=2,
        rx_count=1,
        port_count=1,
        fixed_tx_power_dbm=23.0,
        power_config=None,
    )

    np.testing.assert_allclose(result.tx_power_dbm, [[[23.0], [23.0]]])
    np.testing.assert_allclose(result.power_scale_linear, np.sqrt(199.526), rtol=1e-3)
    assert not np.any(result.clipped_flag)


def test_open_loop_closed_loop_control_selects_serving_rx_and_clips():
    power_config = SimpleNamespace(
        reference_tx_power_dbm=0.0,
        apply_tx_power_to_grid=True,
        uplink_control=SimpleNamespace(
            enabled=True,
            serving_rx_policy="strongest_path",
            open_loop_enabled=True,
            p0_dbm=-80.0,
            alpha=1.0,
            closed_loop_enabled=True,
            tpc_offset_db=3.0,
            accumulation_db=2.0,
            min_tx_power_dbm=-20.0,
            max_tx_power_dbm=10.0,
        ),
    )

    result = compute_uplink_power(
        path_power_db=np.asarray([[-90.0, -70.0], [-50.0, -60.0]], dtype=np.float32),
        snapshot_count=1,
        tx_count=2,
        rx_count=2,
        port_count=2,
        fixed_tx_power_dbm=0.0,
        power_config=power_config,
    )

    assert result.serving_rx_index.tolist() == [[1, 0]]
    np.testing.assert_allclose(result.path_loss_db, [[70.0, 50.0]])
    np.testing.assert_allclose(result.closed_loop_db, [[[5.0, 5.0], [5.0, 5.0]]])
    np.testing.assert_allclose(result.tx_power_dbm, [[[-5.0, -5.0], [-20.0, -20.0]]])
    assert result.clipped_flag.tolist() == [[[False, False], [True, True]]]


def test_thermal_noise_power_uses_ktb_noise_figure():
    assert thermal_noise_dbm(
        bandwidth_hz=1.0,
        noise_figure_db=0.0,
        temperature_k=290.0,
    ) == pytest.approx(-173.975187, abs=1e-6)
    assert thermal_noise_dbm(
        bandwidth_hz=1.0e6,
        noise_figure_db=7.0,
        temperature_k=290.0,
    ) == pytest.approx(-106.975187, abs=1e-6)
