from types import SimpleNamespace

import numpy as np

from sionna_measurement_sim.phy.nr_srs_power_control import compute_srs_power_control


def test_srs_power_control_selects_serving_rx_and_clips_power():
    result = compute_srs_power_control(
        path_power_db=np.asarray([[-90.0, -70.0], [-50.0, -60.0]], dtype=np.float32),
        snapshot_count=1,
        tx_count=2,
        rx_count=2,
        num_srs_ports=2,
        base_tx_power_dbm=0.0,
        config=SimpleNamespace(
            enabled=True,
            p0_dbm=-80.0,
            alpha=1.0,
            min_tx_power_dbm=-20.0,
            max_tx_power_dbm=10.0,
            serving_rx_policy="strongest_path",
        ),
    )

    assert result.serving_rx_index.tolist() == [[1, 0]]
    np.testing.assert_allclose(result.path_loss_db, [[70.0, 50.0]])
    np.testing.assert_allclose(result.tx_power_dbm, [[[-10.0, -10.0], [-20.0, -20.0]]])
    assert not np.allclose(result.power_scale_linear, 1.0)
