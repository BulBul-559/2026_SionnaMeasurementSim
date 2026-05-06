"""Motion domain models for multi-snapshot / Doppler support."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sionna_measurement_sim.domain.validation import require_shape


@dataclass(frozen=True)
class MotionSpec:
    """HDF5 `/motion` group fields."""

    snapshot_id: np.ndarray  # int64 [snapshot]
    timestamp_s: np.ndarray  # float64 [snapshot]
    sampling_frequency_hz: float
    num_time_steps: int
    mobility_mode: str = "static"

    def __post_init__(self) -> None:
        snapshot_id = np.asarray(self.snapshot_id, dtype=np.int64)
        timestamp_s = np.asarray(self.timestamp_s, dtype=np.float64)
        require_shape("snapshot_id", snapshot_id, (None,))
        require_shape("timestamp_s", timestamp_s, snapshot_id.shape)
        if snapshot_id.size > 1:
            if np.any(np.diff(timestamp_s) <= 0):
                msg = "timestamp_s must be strictly increasing"
                raise ValueError(msg)

        object.__setattr__(self, "snapshot_id", snapshot_id)
        object.__setattr__(self, "timestamp_s", timestamp_s)

    @classmethod
    def static_single(cls) -> MotionSpec:
        return cls(
            snapshot_id=np.array([0], dtype=np.int64),
            timestamp_s=np.array([0.0], dtype=np.float64),
            sampling_frequency_hz=0.0,
            num_time_steps=1,
            mobility_mode="static",
        )

    @classmethod
    def doppler_synthetic(
        cls,
        num_time_steps: int,
        sampling_frequency_hz: float,
    ) -> MotionSpec:
        snapshot_id = np.arange(num_time_steps, dtype=np.int64)
        timestamp_s = snapshot_id.astype(np.float64) / sampling_frequency_hz
        return cls(
            snapshot_id=snapshot_id,
            timestamp_s=timestamp_s,
            sampling_frequency_hz=float(sampling_frequency_hz),
            num_time_steps=num_time_steps,
            mobility_mode="doppler_synthetic",
        )
