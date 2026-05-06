"""Phase 6 statistical tests for Doppler and motion."""

from pathlib import Path

import h5py
import numpy as np
import pytest


class TestStaticDoppler:
    def test_static_device_state_has_zero_velocity(self):
        from sionna_measurement_sim.domain.results import DeviceState

        devices = DeviceState.static(snapshots=1, tx=2, rx=3)
        assert np.all(devices.tx_velocity_mps == 0.0)
        assert np.all(devices.rx_velocity_mps == 0.0)
        assert devices.tx_velocity_mps.shape == (1, 2, 3)
        assert devices.rx_velocity_mps.shape == (1, 3, 3)

    def test_device_state_with_velocity(self):
        from sionna_measurement_sim.domain.results import DeviceState

        tx_v = np.array([[[1.0, 0.0, 0.0]]], dtype=np.float32)
        rx_v = np.array([[[0.0, 2.0, 0.0]]], dtype=np.float32)
        devices = DeviceState(
            tx_velocity_mps=tx_v,
            rx_velocity_mps=rx_v,
            tx_orientation_rad=np.zeros_like(tx_v),
            rx_orientation_rad=np.zeros_like(rx_v),
        )
        assert devices.tx_velocity_mps[0, 0, 0] == 1.0
        assert devices.rx_velocity_mps[0, 0, 1] == 2.0


class TestMotionTimestampMonotonic:
    def test_single_snapshot_ok(self):
        from sionna_measurement_sim.domain.motion import MotionSpec

        m = MotionSpec(
            snapshot_id=np.array([0], dtype=np.int64),
            timestamp_s=np.array([0.0], dtype=np.float64),
            sampling_frequency_hz=0.0,
            num_time_steps=1,
        )
        assert m.timestamp_s.shape == (1,)

    def test_multi_snapshot_monotonic(self):
        from sionna_measurement_sim.domain.motion import MotionSpec

        m = MotionSpec.doppler_synthetic(num_time_steps=20, sampling_frequency_hz=200.0)
        assert np.all(np.diff(m.timestamp_s) > 0)
        assert m.timestamp_s[-1] == pytest.approx(19 / 200.0)


class TestDopplerFieldPresence:
    def test_doppler_hz_field_exists_in_phase3_output(self):
        results_path = Path("outputs/phase3_paths/results.h5")
        if not results_path.exists():
            pytest.skip("Phase 3 output not available")
        with h5py.File(results_path, "r") as h5:
            doppler = h5["paths/samples/doppler_hz"]
            assert doppler is not None
            assert np.all(np.isfinite(doppler[()]))
