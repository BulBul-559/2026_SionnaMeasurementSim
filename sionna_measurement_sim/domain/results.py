"""Top-level simulation result domain model."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np

from sionna_measurement_sim import __version__
from sionna_measurement_sim.domain.antenna import AntennaSpec
from sionna_measurement_sim.domain.channel import RTTruthResult
from sionna_measurement_sim.domain.constants import (
    CONTRACT_NAME,
    INDEX_ORDER,
    PRODUCER,
    SCHEMA_VERSION,
    UNIT_CONVENTION,
)
from sionna_measurement_sim.domain.frequency import FrequencyGrid
from sionna_measurement_sim.domain.motion import MotionSpec
from sionna_measurement_sim.domain.observation import (
    EvaluationResult,
    ImpairmentSpec,
    ObservationResult,
    ReceiverSpec,
    WaveformSpec,
)
from sionna_measurement_sim.domain.path import PathSamples, PathTable
from sionna_measurement_sim.domain.topology import Topology
from sionna_measurement_sim.domain.validation import require_shape


@dataclass(frozen=True)
class Metadata:
    """HDF5 `/meta` fields."""

    run_id: str
    random_seed: int
    config_snapshot: str
    schema_version: str = SCHEMA_VERSION
    contract_name: str = CONTRACT_NAME
    producer: str = PRODUCER
    created_at: str = field(
        default_factory=lambda: datetime.now(UTC).replace(microsecond=0).isoformat()
    )
    git_commit: str = ""
    coordinate_system: str = "right_handed_xyz"
    unit_convention: str = UNIT_CONVENTION
    index_order: str = INDEX_ORDER
    truth_branch_enabled: bool = True
    observation_branch_enabled: bool = False
    measurement_realism_level: str = "phase1_schema_only"
    software_versions: str = field(default_factory=lambda: json.dumps({"package": __version__}))


@dataclass(frozen=True)
class InputSpec:
    """HDF5 `/input` fields."""

    label_file: str
    scene_file: str
    input_dataset_id: str = "phase1_minimal"
    input_schema: str = "manual_minimal_v1"


@dataclass(frozen=True)
class DeviceState:
    """Static or snapshot device state in SI units."""

    tx_velocity_mps: np.ndarray
    rx_velocity_mps: np.ndarray
    tx_orientation_rad: np.ndarray
    rx_orientation_rad: np.ndarray

    def __post_init__(self) -> None:
        tx_velocity = np.asarray(self.tx_velocity_mps, dtype=np.float32)
        rx_velocity = np.asarray(self.rx_velocity_mps, dtype=np.float32)
        tx_orientation = np.asarray(self.tx_orientation_rad, dtype=np.float32)
        rx_orientation = np.asarray(self.rx_orientation_rad, dtype=np.float32)

        require_shape("tx_velocity_mps", tx_velocity, (None, None, 3))
        require_shape("rx_velocity_mps", rx_velocity, (tx_velocity.shape[0], None, 3))
        require_shape("tx_orientation_rad", tx_orientation, tx_velocity.shape)
        require_shape("rx_orientation_rad", rx_orientation, rx_velocity.shape)

        object.__setattr__(self, "tx_velocity_mps", tx_velocity)
        object.__setattr__(self, "rx_velocity_mps", rx_velocity)
        object.__setattr__(self, "tx_orientation_rad", tx_orientation)
        object.__setattr__(self, "rx_orientation_rad", rx_orientation)

    @classmethod
    def static(cls, snapshots: int, tx: int, rx: int) -> DeviceState:
        return cls(
            tx_velocity_mps=np.zeros((snapshots, tx, 3), dtype=np.float32),
            rx_velocity_mps=np.zeros((snapshots, rx, 3), dtype=np.float32),
            tx_orientation_rad=np.zeros((snapshots, tx, 3), dtype=np.float32),
            rx_orientation_rad=np.zeros((snapshots, rx, 3), dtype=np.float32),
        )


@dataclass(frozen=True)
class SceneSpec:
    """HDF5 `/scene` fields."""

    scene_name: str
    scene_file: str
    material_policy: str = "phase1_not_loaded"


@dataclass(frozen=True)
class RuntimeInfo:
    """Dependency and execution metadata for `/runtime`."""

    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    sionna_version: str = ""
    sionna_rt_version: str = ""
    torch_version: str = ""
    mitsuba_version: str = ""
    drjit_version: str = ""
    cuda_available: bool = False
    cuda_device_name: str = ""
    command_line: str = ""
    elapsed_seconds: float = 0.0


@dataclass(frozen=True)
class MeasurementSimulationResult:
    """Full truth-only measurement result used by the HDF5 writer."""

    metadata: Metadata
    input_spec: InputSpec
    topology: Topology
    devices: DeviceState
    antenna: AntennaSpec
    scene: SceneSpec
    frequency: FrequencyGrid
    truth: RTTruthResult
    path_samples: PathSamples
    runtime: RuntimeInfo
    path_table: PathTable | None = None
    waveform: WaveformSpec | None = None
    observation: ObservationResult | None = None
    impairments: ImpairmentSpec | None = None
    receiver: ReceiverSpec | None = None
    evaluation: EvaluationResult | None = None
    motion: MotionSpec | None = None

    def __post_init__(self) -> None:
        cfr = self.truth.cfr
        ndim = cfr.ndim
        if ndim == 5:
            tx, rx, rx_ant, tx_ant, subcarrier = cfr.shape
        elif ndim == 6:
            _, tx, rx, rx_ant, tx_ant, subcarrier = cfr.shape
        else:
            msg = f"truth cfr must have rank 5 or 6, got {cfr.shape}"
            raise ValueError(msg)
        if tx != self.topology.num_tx or rx != self.topology.num_rx:
            msg = "truth cfr tx/rx dimensions must match topology"
            raise ValueError(msg)
        if rx_ant != self.antenna.rx_num_ant or tx_ant != self.antenna.tx_num_ant:
            msg = "truth cfr antenna dimensions must match antenna spec"
            raise ValueError(msg)
        if subcarrier != self.frequency.num_subcarriers:
            msg = "truth cfr subcarrier dimension must match frequency grid"
            raise ValueError(msg)
        if self.devices.tx_velocity_mps.shape[1] != tx:
            msg = "device tx state dimension must match topology"
            raise ValueError(msg)
        if self.devices.rx_velocity_mps.shape[1] != rx:
            msg = "device rx state dimension must match topology"
            raise ValueError(msg)
        if self.motion is not None:
            if cfr.ndim == 6 and cfr.shape[0] != self.motion.num_time_steps:
                msg = "truth cfr snapshots must match motion num_time_steps"
                raise ValueError(msg)
        if self.observation is not None:
            if self.observation.cfr_est.shape[1:] != cfr.shape[-5:]:
                msg = "observation cfr_est shape[1:] must match truth cfr"
                raise ValueError(msg)
            if self.evaluation is None or self.waveform is None or self.receiver is None:
                msg = "observation results require waveform, receiver, and evaluation"
                raise ValueError(msg)
            if self.evaluation.nmse_db.shape != self.observation.valid_mask.shape:
                msg = "evaluation nmse_db shape must match observation link mask"
                raise ValueError(msg)


def create_phase1_minimal_result() -> MeasurementSimulationResult:
    """Build a deterministic no-Sionna fixture that satisfies the HDF5 contract."""

    topology = Topology(
        tx_positions_m=np.array([[0.0, 0.0, 1.5]], dtype=np.float32),
        rx_positions_m=np.array([[5.0, 0.0, 1.5]], dtype=np.float32),
        tx_labels=("tx0",),
        rx_labels=("rx0",),
    )
    antenna = AntennaSpec()
    frequency = FrequencyGrid.from_center_bandwidth(
        center_frequency_hz=3.5e9,
        bandwidth_hz=20e6,
        num_subcarriers=8,
    )
    truth_cfr = np.ones(
        (
            topology.num_tx,
            topology.num_rx,
            antenna.rx_num_ant,
            antenna.tx_num_ant,
            frequency.num_subcarriers,
        ),
        dtype=np.complex64,
    )

    return MeasurementSimulationResult(
        metadata=Metadata(
            run_id="phase1_minimal",
            random_seed=0,
            config_snapshot=json.dumps({"phase": 1, "purpose": "schema_fixture"}),
            software_versions=json.dumps(
                {
                    "package": __version__,
                    "python": platform.python_version(),
                    "sionna": "",
                    "torch": "",
                }
            ),
        ),
        input_spec=InputSpec(label_file="", scene_file=""),
        topology=topology,
        devices=DeviceState.static(snapshots=1, tx=topology.num_tx, rx=topology.num_rx),
        antenna=antenna,
        scene=SceneSpec(scene_name="phase1_minimal", scene_file=""),
        frequency=frequency,
        truth=RTTruthResult(
            cfr=truth_cfr,
            path_power_db=np.array([[0.0]], dtype=np.float32),
            has_geometric_signal=np.array([[True]], dtype=np.bool_),
        ),
        path_samples=PathSamples.empty(),
        runtime=RuntimeInfo(command_line="phase1 fixture"),
    )
