"""Phase 6 statistical tests for Doppler and motion."""

from pathlib import Path

import h5py
import numpy as np
import pytest


class TestStaticDoppler:
    def test_static_scene_yields_zero_doppler(self):
        """Without velocity, Sionna RT must not produce Doppler shifts."""
        results_path = Path("outputs/phase3_paths/results.h5")
        if not results_path.exists():
            pytest.skip("Phase 3 output not available")
        with h5py.File(results_path, "r") as h5:
            doppler = h5["paths/samples/doppler_hz"][()]
            assert np.all(np.isfinite(doppler))
            assert float(np.max(np.abs(doppler))) < 1e-3


class TestMovingDoppler:
    def test_moving_tx_yields_nonzero_doppler(self):
        """Configured velocity must produce non-zero path Doppler."""
        results_path = Path("outputs/e2e_doppler_test/results.h5")
        if not results_path.exists():
            pytest.skip("Doppler output not available; run e2e with velocity")
        with h5py.File(results_path, "r") as h5:
            doppler = h5["paths/samples/doppler_hz"][()]
            assert np.any(np.abs(doppler) > 1e-6), (
                "Non-zero velocity must produce non-zero Doppler"
            )


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
    def test_doppler_hz_field_exists_and_finite(self):
        results_path = Path("outputs/phase3_paths/results.h5")
        if not results_path.exists():
            pytest.skip("Phase 3 output not available")
        with h5py.File(results_path, "r") as h5:
            doppler = h5["paths/samples/doppler_hz"]
            assert doppler is not None
            assert np.all(np.isfinite(doppler[()]))
