"""Top-level simulation result domain model."""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np

from sionna_measurement_sim import __version__
from sionna_measurement_sim.domain.antenna import AntennaSpec
from sionna_measurement_sim.domain.channel import RTTruthResult
from sionna_measurement_sim.domain.cir import CIRTruth
from sionna_measurement_sim.domain.constants import (
    CONTRACT_NAME,
    INDEX_ORDER,
    IQ_LINK_LIBRARY_CONTRACT_NAME,
    OUTPUT_PRODUCT_DERIVED,
    OUTPUT_PRODUCT_IQ,
    OUTPUT_PRODUCT_LINK_LABELS,
    OUTPUT_PRODUCT_NLOS_PATH_TRUTH,
    PRODUCER,
    RT_LABELS_CONTRACT_NAME,
    SCHEMA_VERSION,
    UNIT_CONVENTION,
)
from sionna_measurement_sim.domain.derived import DerivedLabels, build_derived_labels
from sionna_measurement_sim.domain.frequency import FrequencyGrid
from sionna_measurement_sim.domain.iq import IQObservationResult
from sionna_measurement_sim.domain.link import LinkConfig
from sionna_measurement_sim.domain.motion import MotionSpec
from sionna_measurement_sim.domain.multiuser import MultiUserSRSResult
from sionna_measurement_sim.domain.observation import (
    CalibrationResult,
    DiagnosticsReport,
    EvaluationResult,
    ImpairmentSpec,
    ObservationResult,
    ReceiverSpec,
    WaveformSpec,
)
from sionna_measurement_sim.domain.output_plan import FULL_OUTPUT_PRODUCTS
from sionna_measurement_sim.domain.path import (
    NLoSPathTruth,
    PathSamples,
    PathTable,
    build_nlos_path_truth,
)
from sionna_measurement_sim.domain.topology import Topology
from sionna_measurement_sim.domain.validation import require_shape

if TYPE_CHECKING:
    from sionna_measurement_sim.ranging.result import RangingResult


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
    output_profile: str = "full"
    output_products: tuple[str, ...] = ()


@dataclass(frozen=True)
class InputSpec:
    """HDF5 `/input` fields."""

    label_file: str
    scene_file: str
    input_dataset_id: str = "phase1_minimal"
    input_schema: str = "standard_label_0.1.0"


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
    shard_id: str | None = None
    parent_shard_index: int | None = None
    fallback_level: int = 0
    fallback_reason: str = ""

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
        if self.fallback_level < 0:
            msg = "fallback_level must be non-negative"
            raise ValueError(msg)
        if self.parent_shard_index is not None and self.parent_shard_index < 0:
            msg = "parent_shard_index must be non-negative"
            raise ValueError(msg)
        if self.ue_indices is not None:
            _validate_non_negative_indices("ue_indices", self.ue_indices)
            object.__setattr__(self, "ue_indices", tuple(int(i) for i in self.ue_indices))
        if self.bs_indices is not None:
            _validate_non_negative_indices("bs_indices", self.bs_indices)
            object.__setattr__(self, "bs_indices", tuple(int(i) for i in self.bs_indices))
        if self.shard_id is None:
            object.__setattr__(self, "shard_id", f"{self.shard_index:03d}")


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
class RTCompactLinkLabels:
    """Flattened link-level labels for compact RT labels-only output."""

    link_index: np.ndarray
    tx_index: np.ndarray
    rx_index: np.ndarray
    global_tx_index: np.ndarray
    global_rx_index: np.ndarray
    tx_xy_m: np.ndarray
    rx_xy_m: np.ndarray
    link_valid_mask: np.ndarray
    geometric_distance_m: np.ndarray
    first_path_delay_s: np.ndarray
    first_path_propagation_range_m: np.ndarray
    strongest_path_delay_s: np.ndarray
    path_power_db: np.ndarray
    los_flag: np.ndarray
    nlos_flag: np.ndarray
    path_count: np.ndarray
    first_path_aoa_azimuth_rad: np.ndarray
    first_path_aoa_zenith_rad: np.ndarray
    tx_rx_bearing_rad: np.ndarray
    tx_rx_distance_m: np.ndarray

    def __post_init__(self) -> None:
        link_index = np.asarray(self.link_index, dtype=np.int64)
        require_shape("link_index", link_index, (None,))
        link_count = link_index.shape[0]
        object.__setattr__(self, "link_index", link_index)
        for name in (
            "tx_index",
            "rx_index",
            "global_tx_index",
            "global_rx_index",
            "path_count",
        ):
            object.__setattr__(
                self,
                name,
                np.asarray(getattr(self, name), dtype=np.int64),
            )
            require_shape(name, getattr(self, name), (link_count,))
        for name in ("link_valid_mask", "los_flag", "nlos_flag"):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=np.bool_))
            require_shape(name, getattr(self, name), (link_count,))
        for name in (
            "geometric_distance_m",
            "first_path_delay_s",
            "first_path_propagation_range_m",
            "strongest_path_delay_s",
            "path_power_db",
            "first_path_aoa_azimuth_rad",
            "first_path_aoa_zenith_rad",
            "tx_rx_bearing_rad",
            "tx_rx_distance_m",
        ):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=np.float32))
            require_shape(name, getattr(self, name), (link_count,))
        for name in ("tx_xy_m", "rx_xy_m"):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=np.float32))
            require_shape(name, getattr(self, name), (link_count, 2))

    @classmethod
    def from_topology(
        cls,
        topology: Topology,
        derived: DerivedLabels,
        *,
        shard: ShardMetadata | None = None,
    ) -> RTCompactLinkLabels:
        tx_count, rx_count = topology.num_tx, topology.num_rx
        tx_grid, rx_grid = np.indices((tx_count, rx_count), dtype=np.int64)
        if shard is not None:
            global_tx = shard.global_tx_indices[tx_grid]
            global_rx = shard.global_rx_indices[rx_grid]
        else:
            global_tx = tx_grid
            global_rx = rx_grid
        tx_xy = np.broadcast_to(
            topology.tx_positions_m[:, np.newaxis, :2],
            (tx_count, rx_count, 2),
        )
        rx_xy = np.broadcast_to(
            topology.rx_positions_m[np.newaxis, :, :2],
            (tx_count, rx_count, 2),
        )
        return cls(
            link_index=np.arange(tx_count * rx_count, dtype=np.int64),
            tx_index=tx_grid.reshape(-1),
            rx_index=rx_grid.reshape(-1),
            global_tx_index=global_tx.reshape(-1),
            global_rx_index=global_rx.reshape(-1),
            tx_xy_m=tx_xy.reshape(-1, 2),
            rx_xy_m=rx_xy.reshape(-1, 2),
            link_valid_mask=derived.link_valid_mask.reshape(-1),
            geometric_distance_m=derived.geometric_distance_m.reshape(-1),
            first_path_delay_s=derived.first_path_delay_s.reshape(-1),
            first_path_propagation_range_m=(
                derived.first_path_propagation_range_m.reshape(-1)
            ),
            strongest_path_delay_s=derived.strongest_path_delay_s.reshape(-1),
            path_power_db=derived.path_power_db.reshape(-1),
            los_flag=derived.los_flag.reshape(-1),
            nlos_flag=derived.nlos_flag.reshape(-1),
            path_count=derived.path_count.reshape(-1),
            first_path_aoa_azimuth_rad=derived.first_path_aoa_azimuth_rad.reshape(-1),
            first_path_aoa_zenith_rad=derived.first_path_aoa_zenith_rad.reshape(-1),
            tx_rx_bearing_rad=derived.tx_rx_bearing_rad.reshape(-1),
            tx_rx_distance_m=derived.tx_rx_distance_m.reshape(-1),
        )


@dataclass(frozen=True)
class RTLabelsOnlyResult:
    """Compact RT labels-only result used by the lightweight HDF5 writer."""

    metadata: Metadata
    input_spec: InputSpec
    topology: Topology
    devices: DeviceState
    antenna: AntennaSpec
    scene: SceneSpec
    frequency: FrequencyGrid
    runtime: RuntimeInfo
    derived: DerivedLabels
    link_labels: RTCompactLinkLabels
    link: LinkConfig | None = None
    shard: ShardMetadata | None = None

    def __post_init__(self) -> None:
        if self.metadata.contract_name != RT_LABELS_CONTRACT_NAME:
            msg = "RTLabelsOnlyResult metadata.contract_name must be RT labels contract"
            raise ValueError(msg)
        if self.metadata.output_profile != "rt_labels_only":
            msg = "RTLabelsOnlyResult metadata.output_profile must be rt_labels_only"
            raise ValueError(msg)
        link_shape = (self.topology.num_tx, self.topology.num_rx)
        require_shape("derived.link_valid_mask", self.derived.link_valid_mask, link_shape)
        require_shape(
            "link_labels.link_index",
            self.link_labels.link_index,
            (link_shape[0] * link_shape[1],),
        )


@dataclass(frozen=True)
class IQLinkLibraryResult:
    """Compact clean-IQ-per-link result for online non-cooperative mixing."""

    metadata: Metadata
    input_spec: InputSpec
    topology: Topology
    devices: DeviceState
    antenna: AntennaSpec
    scene: SceneSpec
    frequency: FrequencyGrid
    runtime: RuntimeInfo
    iq: IQObservationResult
    link: LinkConfig | None = None
    shard: ShardMetadata | None = None

    def __post_init__(self) -> None:
        if self.metadata.contract_name != IQ_LINK_LIBRARY_CONTRACT_NAME:
            msg = "IQLinkLibraryResult metadata.contract_name must be IQ link library contract"
            raise ValueError(msg)
        if self.metadata.output_profile != "iq_link_library":
            msg = "IQLinkLibraryResult metadata.output_profile must be iq_link_library"
            raise ValueError(msg)
        if self.iq.link is None or self.iq.link.is_empty:
            msg = "IQ link library requires at least one /iq/link clean capture"
            raise ValueError(msg)
        if self.iq.noncooperative is not None:
            msg = "IQ link library stores per-link clean IQ, not mixed noncooperative frames"
            raise ValueError(msg)
        if self.iq.link.frequency_observed is not None or self.iq.link.time_observed is not None:
            msg = "IQ link library must not contain observed/impaired IQ"
            raise ValueError(msg)
        if self.iq.link.frequency_clean is None and self.iq.link.time_clean is None:
            msg = "IQ link library requires frequency_clean or time_clean"
            raise ValueError(msg)
        link_prefix = (self.topology.num_tx, self.topology.num_rx, self.antenna.rx_num_ant)
        if self.iq.link.frequency_clean is not None:
            freq = np.asarray(self.iq.link.frequency_clean)
            if freq.ndim != 6 or freq.shape[1:4] != link_prefix:
                msg = "frequency_clean must have [snapshot,tx,rx,rx_ant,symbol,subcarrier]"
                raise ValueError(msg)
            if freq.shape[-1] != self.frequency.num_subcarriers:
                msg = "frequency_clean subcarrier dimension must match frequency grid"
                raise ValueError(msg)
        if self.iq.link.time_clean is not None:
            time = np.asarray(self.iq.link.time_clean)
            if time.ndim != 5 or time.shape[1:4] != link_prefix:
                msg = "time_clean must have [snapshot,tx,rx,rx_ant,sample]"
                raise ValueError(msg)


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
    runtime: RuntimeInfo
    truth: RTTruthResult | None = None
    path_samples: PathSamples | None = None
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
    ranging: RangingResult | None = None
    multiuser: MultiUserSRSResult | None = None
    iq: IQObservationResult | None = None
    link_labels: RTCompactLinkLabels | None = None

    def __post_init__(self) -> None:
        output_products = tuple(self.metadata.output_products or ())
        product_aware_full = bool(output_products) and output_products != FULL_OUTPUT_PRODUCTS
        tx = self.topology.num_tx
        rx = self.topology.num_rx
        rx_ant = self.antenna.rx_num_ant
        tx_ant = self.antenna.tx_num_ant
        subcarrier = self.frequency.num_subcarriers
        cfr_shape: tuple[int, int, int, int, int] | None = None
        if self.truth is not None:
            cfr = self.truth.cfr
            if cfr.ndim != 5:
                msg = f"truth cfr must be rank 5, got {cfr.shape}"
                raise ValueError(msg)
            cfr_shape = cfr.shape
            tx, rx, rx_ant, tx_ant, subcarrier = cfr.shape
        elif not product_aware_full:
            msg = "full MeasurementSimulationResult requires truth CFR"
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
        if self.shard is not None:
            if self.shard.global_tx_indices.shape[0] != tx:
                msg = "shard global_tx_indices length must match topology"
                raise ValueError(msg)
            if self.shard.global_rx_indices.shape[0] != rx:
                msg = "shard global_rx_indices length must match topology"
                raise ValueError(msg)
        if (
            self.motion is not None
            and self.truth is not None
            and self.truth.cfr_snapshots is not None
        ):
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
            if cfr_shape is not None and self.observation.cfr_est.shape[1:] != cfr_shape:
                msg = "observation cfr_est shape[1:] must match truth cfr"
                raise ValueError(msg)
            if self.evaluation is None or self.waveform is None or self.receiver is None:
                msg = "observation results require waveform, receiver, and evaluation"
                raise ValueError(msg)
            if self.evaluation.nmse_db.shape != self.observation.valid_mask.shape:
                msg = "evaluation nmse_db shape must match observation link mask"
                raise ValueError(msg)
        if self.ranging is not None:
            if self.observation is None:
                msg = "ranging results require observation cfr_est"
                raise ValueError(msg)
            link_shape = self.observation.valid_mask.shape
            for estimator in (self.ranging.pdp_peak, self.ranging.phase_slope):
                if estimator is None:
                    continue
                if estimator.toa_est_s.shape != link_shape:
                    msg = "ranging estimator link shape must match observation link mask"
                    raise ValueError(msg)
        if self.multiuser is not None:
            if self.observation is None:
                msg = "multiuser SRS results require PHY observation"
                raise ValueError(msg)
            if self.multiuser.rx_grid_shared.shape[0] != self.observation.cfr_est.shape[0]:
                msg = "multiuser snapshot dimension must match observation snapshot dimension"
                raise ValueError(msg)
            if self.multiuser.rx_grid_shared.shape[2] != rx:
                msg = "multiuser RX dimension must match topology RX dimension"
                raise ValueError(msg)
            if self.multiuser.rx_grid_shared.shape[3] != rx_ant:
                msg = "multiuser RX antenna dimension must match antenna spec"
                raise ValueError(msg)
            if self.multiuser.rx_grid_shared.shape[-1] != subcarrier:
                msg = "multiuser subcarrier dimension must match frequency grid"
                raise ValueError(msg)
        if (
            self.iq is not None
            and self.observation is None
            and OUTPUT_PRODUCT_IQ not in output_products
        ):
            msg = "IQ observations require PHY observation unless output product is iq"
            raise ValueError(msg)
        wants_legacy_full = not product_aware_full
        wants_derived = (
            wants_legacy_full
            or OUTPUT_PRODUCT_DERIVED in self.metadata.output_products
            or OUTPUT_PRODUCT_LINK_LABELS in self.metadata.output_products
        )
        if self.derived is None and wants_derived:
            if self.truth is None:
                msg = "derived output requires truth or explicit DerivedLabels"
                raise ValueError(msg)
            object.__setattr__(
                self,
                "derived",
                build_derived_labels(
                    self.topology, self.truth, self.path_table, self.cir_truth
                ),
            )
        wants_nlos_truth = (
            wants_legacy_full
            or OUTPUT_PRODUCT_NLOS_PATH_TRUTH in self.metadata.output_products
        )
        if self.nlos_path_truth is None and wants_nlos_truth:
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
        if (
            self.link_labels is None
            and OUTPUT_PRODUCT_LINK_LABELS in output_products
        ):
            if self.derived is None:
                msg = "link label output requires derived labels"
                raise ValueError(msg)
            object.__setattr__(
                self,
                "link_labels",
                RTCompactLinkLabels.from_topology(
                    self.topology,
                    self.derived,
                    shard=self.shard,
                ),
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
