"""Domain models for SionnaMeasurementSim.

The domain layer is intentionally free of Sionna imports. Adapter code converts
external simulator objects into these stable data structures before IO or app
layers consume them.
"""

from sionna_measurement_sim.domain.antenna import AntennaSpec
from sionna_measurement_sim.domain.channel import RTTruthResult
from sionna_measurement_sim.domain.cir import CIRTruth
from sionna_measurement_sim.domain.frequency import FrequencyGrid
from sionna_measurement_sim.domain.link import LinkConfig
from sionna_measurement_sim.domain.observation import (
    EvaluationResult,
    ImpairmentSpec,
    ObservationResult,
    ReceiverSpec,
    WaveformSpec,
)
from sionna_measurement_sim.domain.path import PathSamples, PathTable
from sionna_measurement_sim.domain.results import (
    DeviceState,
    InputSpec,
    MeasurementSimulationResult,
    Metadata,
    RuntimeInfo,
    SceneSpec,
)
from sionna_measurement_sim.domain.topology import Topology

__all__ = [
    "AntennaSpec",
    "CIRTruth",
    "DeviceState",
    "EvaluationResult",
    "FrequencyGrid",
    "ImpairmentSpec",
    "InputSpec",
    "LinkConfig",
    "MeasurementSimulationResult",
    "Metadata",
    "PathSamples",
    "PathTable",
    "ObservationResult",
    "ReceiverSpec",
    "RTTruthResult",
    "RuntimeInfo",
    "SceneSpec",
    "Topology",
    "WaveformSpec",
]
