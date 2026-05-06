import numpy as np
import pytest

from sionna_measurement_sim.domain.motion import MotionSpec


class TestMotionSpec:
    def test_static_single(self):
        m = MotionSpec.static_single()
        assert m.num_time_steps == 1
        assert m.sampling_frequency_hz == 0.0
        assert m.mobility_mode == "static"
        assert m.snapshot_id.shape == (1,)
        assert m.timestamp_s.shape == (1,)

    def test_doppler_synthetic(self):
        m = MotionSpec.doppler_synthetic(num_time_steps=10, sampling_frequency_hz=100.0)
        assert m.num_time_steps == 10
        assert m.sampling_frequency_hz == 100.0
        assert m.mobility_mode == "doppler_synthetic"
        assert m.snapshot_id.shape == (10,)
        assert np.array_equal(m.snapshot_id, np.arange(10, dtype=np.int64))
        # timestamps: 0, 0.01, 0.02, ...
        assert m.timestamp_s[0] == 0.0
        assert m.timestamp_s[1] == pytest.approx(0.01)

    def test_timestamp_monotonic(self):
        m = MotionSpec.doppler_synthetic(num_time_steps=5, sampling_frequency_hz=50.0)
        assert np.all(np.diff(m.timestamp_s) > 0)

    def test_timestamp_non_increasing_rejected(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            MotionSpec(
                snapshot_id=np.array([0, 1], dtype=np.int64),
                timestamp_s=np.array([1.0, 0.5], dtype=np.float64),
                sampling_frequency_hz=10.0,
                num_time_steps=2,
            )
