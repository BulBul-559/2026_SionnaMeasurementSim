"""Frequency grid domain model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sionna_measurement_sim.domain.validation import require_finite, require_shape


@dataclass(frozen=True)
class FrequencyGrid:
    """Subcarrier frequencies in Hz."""

    center_frequency_hz: float
    bandwidth_hz: float
    frequencies_hz: np.ndarray

    def __post_init__(self) -> None:
        frequencies = np.asarray(self.frequencies_hz, dtype=np.float64)
        require_shape("frequencies_hz", frequencies, (None,))
        require_finite("frequencies_hz", frequencies)
        if frequencies.size == 0:
            msg = "frequencies_hz must not be empty"
            raise ValueError(msg)
        if np.any(np.diff(frequencies) <= 0):
            msg = "frequencies_hz must be strictly increasing"
            raise ValueError(msg)
        if self.bandwidth_hz <= 0:
            msg = "bandwidth_hz must be positive"
            raise ValueError(msg)

        object.__setattr__(self, "center_frequency_hz", float(self.center_frequency_hz))
        object.__setattr__(self, "bandwidth_hz", float(self.bandwidth_hz))
        object.__setattr__(self, "frequencies_hz", frequencies)

    @classmethod
    def from_center_bandwidth(
        cls,
        center_frequency_hz: float,
        bandwidth_hz: float,
        num_subcarriers: int,
    ) -> FrequencyGrid:
        if num_subcarriers <= 0:
            msg = "num_subcarriers must be positive"
            raise ValueError(msg)

        spacing = bandwidth_hz / num_subcarriers
        offsets = (np.arange(num_subcarriers, dtype=np.float64) - (num_subcarriers - 1) / 2.0)
        frequencies = center_frequency_hz + offsets * spacing
        return cls(center_frequency_hz, bandwidth_hz, frequencies)

    @property
    def num_subcarriers(self) -> int:
        return int(self.frequencies_hz.shape[0])

    @property
    def subcarrier_spacing_hz(self) -> float:
        if self.num_subcarriers == 1:
            return float(self.bandwidth_hz)
        return float(np.diff(self.frequencies_hz).mean())
