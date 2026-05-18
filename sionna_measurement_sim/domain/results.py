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
from sionna_measurement_sim.domain.cir import CIRTruth
from sionna_measurement_sim.domain.constants import (
    CONTRACT_NAME,
    INDEX_ORDER,
    PRODUCER,
    SCHEMA_VERSION,
    UNIT_CONVENTION,
)
from sionna_measurement_sim.domain.derived import DerivedLabels, build_derived_labels
from sionna_measurement_sim.domain.frequency import FrequencyGrid
from sionna_measurement_sim.domain.link import LinkConfig
from sionna_measurement_sim.domain.motion import MotionSpec
from sionna_measurement_sim.domain.observation import (
    CalibrationResult,
    DiagnosticsReport,
    EvaluationResult,
    ImpairmentSpec,
    ObservationResult,
    ReceiverSpec,
    WaveformSpec,
)
from sionna_measurement_sim.domain.path import (
    NLoSPathTruth,
    PathSamples,
    PathTable,
    build_nlos_path_truth,
)
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
class ShardSpec:
    """Requested topology shard selection."""

    shard_index: int
    shard_count: int
    axis: str = "ue"
    ue_start: int = 0
    ue_count: int | None = None
    ue_indices: tuple[int, ...] | None = None
    bs_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.shard_count < 1:
            msg = "shard_count must be positive"
            raise ValueError(msg)
        if self.shard_index < 0 or self.shard_index >= self.shard_count:
            msg = "shard_index must be in [0, shard_count)"
            raise ValueError(msg)
        if not self.axis:
            msg = "axis must not be empty"
            raise ValueError(msg)
        if self.axis != "ue":
            msg = "only UE sharding is supported"
            raise ValueError(msg)
        if self.ue_start < 0:
            msg = "ue_start must be non-negative"
            raise ValueError(msg)
        if self.ue_count is not None and self.ue_count < 1:
            msg = "ue_count must be positive when provided"
            raise ValueError(msg)
        if self.ue_indices is not None:
            _validate_non_negative_indices("ue_indices", self.ue_indices)
            object.__setattr__(self, "ue_indices", tuple(int(i) for i in self.ue_indices))
        if self.bs_indices is not None:
            _validate_non_negative_indices("bs_indices", self.bs_indices)
            object.__setattr__(self, "bs_indices", tuple(int(i) for i in self.bs_indices))


@dataclass(frozen=True)
class ShardMetadata:
    """HDF5 `/shard` metadata mapping local topology indices to global indices."""

    shard_index: int
    shard_count: int
    axis: str
    global_rx_start: int
    global_rx_indices: np.ndarray
    global_tx_indices: np.ndarray

    def __post_init__(self) -> None:
        if self.shard_count < 1:
            msg = "shard_count must be positive"
            raise ValueError(msg)
        if self.shard_index < 0 or self.shard_index >= self.shard_count:
            msg = "shard_index must be in [0, shard_count)"
            raise ValueError(msg)
        if not self.axis:
            msg = "axis must not be empty"
            raise ValueError(msg)
        if self.global_rx_start < 0:
            msg = "global_rx_start must be non-negative"
            raise ValueError(msg)

        rx_indices = np.asarray(self.global_rx_indices, dtype=np.int64)
        tx_indices = np.asarray(self.global_tx_indices, dtype=np.int64)
        require_shape("global_rx_indices", rx_indices, (None,))
        require_shape("global_tx_indices", tx_indices, (None,))
        if rx_indices.size == 0:
            msg = "global_rx_indices must not be empty"
            raise ValueError(msg)
        if tx_indices.size == 0:
            msg = "global_tx_indices must not be empty"
            raise ValueError(msg)
        if np.any(rx_indices < 0):
            msg = "global_rx_indices must be non-negative"
            raise ValueError(msg)
        if np.any(tx_indices < 0):
            msg = "global_tx_indices must be non-negative"
            raise ValueError(msg)

        object.__setattr__(self, "global_rx_indices", rx_indices)
        object.__setattr__(self, "global_tx_indices", tx_indices)

    @classmethod
    def from_spec(
        cls,
        spec: ShardSpec,
        *,
        global_rx_indices: np.ndarray,
        global_tx_indices: np.ndarray,
    ) -> ShardMetadata:
        rx_indices = np.asarray(global_rx_indices, dtype=np.int64)
        rx_start = int(rx_indices[0]) if rx_indices.size else 0
        return cls(
            shard_index=spec.shard_index,
            shard_count=spec.shard_count,
            axis=spec.axis,
            global_rx_start=rx_start,
            global_rx_indices=rx_indices,
            global_tx_indices=np.asarray(global_tx_indices, dtype=np.int64),
        )


def _validate_non_negative_indices(name: str, values: tuple[int, ...]) -> None:
    if not values:
        msg = f"{name} must not be empty"
        raise ValueError(msg)
    if any(int(value) < 0 for value in values):
        msg = f"{name} must be non-negative"
        raise ValueError(msg)


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
    scene_id: str = ""
    map_id: str = ""
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
    derived: DerivedLabels | None = None
    path_table: PathTable | None = None
    nlos_path_truth: NLoSPathTruth | None = None
    cir_truth: CIRTruth | None = None
    waveform: WaveformSpec | None = None
    observation: ObservationResult | None = None
    impairments: ImpairmentSpec | None = None
    receiver: ReceiverSpec | None = None
    evaluation: EvaluationResult | None = None
    motion: MotionSpec | None = None
    calibration: CalibrationResult | None = None
    diagnostics: DiagnosticsReport | None = None
    link: LinkConfig | None = None
    shard: ShardMetadata | None = None
    waveform_extras: dict | None = None
    array_outputs: dict | None = None

    def __post_init__(self) -> None:
        cfr = self.truth.cfr
        if cfr.ndim != 5:
            msg = f"truth cfr must be rank 5, got {cfr.shape}"
            raise ValueError(msg)
        tx, rx, rx_ant, tx_ant, subcarrier = cfr.shape
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
        if self.shard is not None:
            if self.shard.global_tx_indices.shape[0] != tx:
                msg = "shard global_tx_indices length must match topology"
                raise ValueError(msg)
            if self.shard.global_rx_indices.shape[0] != rx:
                msg = "shard global_rx_indices length must match topology"
                raise ValueError(msg)
        if self.motion is not None and self.truth.cfr_snapshots is not None:
            if self.truth.cfr_snapshots.shape[0] != self.motion.num_time_steps:
                msg = "cfr_snapshots must match motion num_time_steps"
                raise ValueError(msg)
        if self.cir_truth is not None:
            if (
                self.cir_truth.coefficients.shape[1] != tx
                or self.cir_truth.coefficients.shape[2] != rx
            ):
                msg = "cir_truth tx/rx dimensions must match topology"
                raise ValueError(msg)
            if (
                self.cir_truth.coefficients.shape[3] != rx_ant
                or self.cir_truth.coefficients.shape[4] != tx_ant
            ):
                msg = "cir_truth antenna dimensions must match antenna spec"
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
        if self.derived is None:
            object.__setattr__(
                self,
                "derived",
                build_derived_labels(
                    self.topology, self.truth, self.path_table, self.cir_truth
                ),
            )
        if self.nlos_path_truth is None:
            nlos_path_truth = (
                build_nlos_path_truth(self.path_table)
                if self.path_table is not None
                else NLoSPathTruth.empty(tx, rx, rx_ant, tx_ant)
            )
            object.__setattr__(
                self,
                "nlos_path_truth",
                nlos_path_truth,
            )


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
        scene=SceneSpec(scene_name="phase1_minimal", scene_file="", scene_id="phase1_minimal"),
        frequency=frequency,
        truth=RTTruthResult(
            cfr=truth_cfr,
            path_power_db=np.array([[0.0]], dtype=np.float32),
            has_geometric_signal=np.array([[True]], dtype=np.bool_),
            geometric_path_count=np.zeros((1, 1), dtype=np.int32),
            los_exists=np.zeros((1, 1), dtype=np.bool_),
            nlos_exists=np.zeros((1, 1), dtype=np.bool_),
        ),
        path_samples=PathSamples.empty(),
        runtime=RuntimeInfo(command_line="phase1 fixture"),
        cir_truth=CIRTruth.empty(
            num_snapshots=1,
            num_tx=topology.num_tx,
            num_rx=topology.num_rx,
            num_rx_ant=antenna.rx_num_ant,
            num_tx_ant=antenna.tx_num_ant,
        ),
        link=LinkConfig(),
    )
