"""Pydantic configuration schema for SionnaMeasurementSim.

Each config group defined here maps to a section in the YAML config file.
Validation fails early (before RT/PHY starts) for any schema violation.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from sionna_measurement_sim.visualization.config import (
    ALLOWED_VISUALIZATION_PLOTS,
    DEFAULT_VISUALIZATION_PLOTS,
)


# ── runtime ──────────────────────────────────────────────────────────
class RuntimeConfig(BaseModel):
    seed: int = Field(default=42, ge=0)
    device: str = Field(default="cpu")
    require_gpu: bool = False
    precision: str = Field(default="single")
    torch_deterministic: bool = False


class DebugConfig(BaseModel):
    enabled: bool = False
    hardware_interval_s: float = Field(default=1.0, gt=0)
    link_log_interval: int = Field(default=250, ge=1)
    torch_synchronize: bool = True
    write_hardware_samples: bool = True


# ── input ────────────────────────────────────────────────────────────
class InputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label_file: str = Field(default="tests/fixtures/scenes/test/test5.json")
    scene_file: str = Field(default="tests/fixtures/scenes/test/scene.xml")
    scene_id: str = ""
    map_id: str = ""
    label_schema: str = Field(default="simplesionna_v1")
    coordinate_system: str = Field(default="scene_local_xyz_m")
    max_bs: int = Field(default=6, ge=1)
    max_ue: int = Field(default=100, ge=1)

    @model_validator(mode="after")
    def default_scene_id(self) -> InputConfig:
        if not self.scene_id:
            object.__setattr__(self, "scene_id", Path(self.scene_file).stem)
        return self


# ── output ───────────────────────────────────────────────────────────
class ShardingFallbackConfig(BaseModel):
    enabled: bool = True
    min_shard_size: int = Field(default=1, ge=1)
    split_factor: int = Field(default=2, ge=2)
    retry_errors: list[str] = Field(
        default_factory=lambda: ["cuda_oom", "drjit_array_limit"]
    )
    failure_policy: str = Field(default="fail_run")

    @model_validator(mode="after")
    def check_fallback_values(self) -> ShardingFallbackConfig:
        allowed = {"fail_run"}
        if self.failure_policy not in allowed:
            raise ValueError("output.sharding.fallback.failure_policy must be fail_run")
        allowed_errors = {"cuda_oom", "drjit_array_limit"}
        unknown = sorted(set(self.retry_errors) - allowed_errors)
        if unknown:
            raise ValueError(
                "output.sharding.fallback.retry_errors contains unknown values: "
                + ", ".join(unknown)
            )
        return self


class OutputShardingConfig(BaseModel):
    enabled: bool = False
    axis: str = Field(default="ue")
    shard_size: int = Field(default=1000, ge=1)
    filename_pattern: str = Field(default="result_{shard_index:03d}.h5")
    results_dir: str = Field(default="results")
    manifest_dir: str = Field(default="manifest")
    parallel_workers: int = Field(default=1, ge=1)
    gpu_ids: list[int] = Field(default_factory=list)
    visualization_mode: str = Field(default="first_shard")
    fallback: ShardingFallbackConfig = Field(default_factory=ShardingFallbackConfig)

    @model_validator(mode="after")
    def check_sharding_values(self) -> OutputShardingConfig:
        if self.axis != "ue":
            raise ValueError("output.sharding.axis must be 'ue'")
        if "{shard_index" not in self.filename_pattern:
            raise ValueError("output.sharding.filename_pattern must include {shard_index...}")
        if not self.results_dir or Path(self.results_dir).is_absolute():
            raise ValueError("output.sharding.results_dir must be a relative path")
        if not self.manifest_dir or Path(self.manifest_dir).is_absolute():
            raise ValueError("output.sharding.manifest_dir must be a relative path")
        if self.visualization_mode not in ("none", "first_shard", "all_shards"):
            raise ValueError(
                "output.sharding.visualization_mode must be none/first_shard/all_shards"
            )
        return self


class OutputConfig(BaseModel):
    profile: str = Field(default="full")
    root_dir: str = Field(default="outputs")
    run_id_format: str = Field(default="{label_stem}_{timestamp}")
    hdf5_filename: str = Field(default="results.h5")
    compression: str = Field(default="gzip")
    save_full_paths: bool = False
    save_sampled_paths: bool = True
    save_raw_waveform: bool = False
    sharding: OutputShardingConfig = Field(default_factory=OutputShardingConfig)

    @model_validator(mode="after")
    def check_output_values(self) -> OutputConfig:
        if self.compression not in ("gzip", "lzf", "none"):
            raise ValueError("output.compression must be gzip/lzf/none")
        if self.profile not in ("full", "rt_lite", "rt_labels_only"):
            raise ValueError("output.profile must be full/rt_lite/rt_labels_only")
        return self


class SpectrumConfig(BaseModel):
    enabled: bool = False
    sources: list[str] = Field(default_factory=lambda: ["truth_cfr", "cfr_est", "rx_grid"])
    method: str = Field(default="bartlett")
    zenith_bins: int = Field(default=91, ge=2)
    azimuth_bins: int = Field(default=181, ge=2)
    zenith_min_rad: float = 0.0
    zenith_max_rad: float = 3.141592653589793
    azimuth_min_rad: float = -3.141592653589793
    azimuth_max_rad: float = 3.141592653589793
    normalize: str = Field(default="per_link_max")
    aggregate_subcarriers: str = Field(default="mean")
    aggregate_symbols: str = Field(default="mean")
    link_chunk_size: int = Field(default=512, ge=1)

    @model_validator(mode="after")
    def check_supported_values(self) -> SpectrumConfig:
        allowed_sources = {"truth_cfr", "cfr_est", "rx_grid"}
        unknown_sources = set(self.sources) - allowed_sources
        if unknown_sources:
            raise ValueError(f"Unsupported spectrum sources: {sorted(unknown_sources)}")
        if self.method != "bartlett":
            raise ValueError("Only spectrum method 'bartlett' is supported")
        if self.normalize != "per_link_max":
            raise ValueError("Only spectrum normalize 'per_link_max' is supported")
        if self.aggregate_subcarriers != "mean":
            raise ValueError("Only aggregate_subcarriers 'mean' is supported")
        if self.aggregate_symbols != "mean":
            raise ValueError("Only aggregate_symbols 'mean' is supported")
        if self.zenith_max_rad <= self.zenith_min_rad:
            raise ValueError("zenith_max_rad must be greater than zenith_min_rad")
        if self.azimuth_max_rad <= self.azimuth_min_rad:
            raise ValueError("azimuth_max_rad must be greater than azimuth_min_rad")
        return self


class ArrayConfig(BaseModel):
    spectrum: SpectrumConfig = Field(default_factory=SpectrumConfig)


# ── carrier / frequency ─────────────────────────────────────────────
class CarrierConfig(BaseModel):
    center_frequency_hz: float = Field(default=3.5e9, gt=0)
    bandwidth_hz: float = Field(default=20e6, gt=0)
    num_subcarriers: int = Field(default=64, ge=2)
    subcarrier_spacing_hz: float = 0.0  # derived if 0

    @model_validator(mode="after")
    def check_spacing_consistency(self) -> CarrierConfig:
        if self.subcarrier_spacing_hz == 0:
            object.__setattr__(self, "subcarrier_spacing_hz",
                               self.bandwidth_hz / self.num_subcarriers)
        # Allow small tolerance
        expected = self.bandwidth_hz / self.num_subcarriers
        if abs(self.subcarrier_spacing_hz - expected) / expected > 0.01:
            msg = (
                f"subcarrier_spacing_hz ({self.subcarrier_spacing_hz}) "
                f"inconsistent with bandwidth_hz/num_subcarriers ({expected})"
            )
            raise ValueError(msg)
        return self


# ── antenna ──────────────────────────────────────────────────────────
class ArraySpec(BaseModel):
    type: str = Field(default="planar")
    num_rows: int = Field(default=1, ge=1)
    num_cols: int = Field(default=1, ge=1)
    vertical_spacing_lambda: float = Field(default=0.5, gt=0)
    horizontal_spacing_lambda: float = Field(default=0.5, gt=0)
    pattern: str = Field(default="iso")
    polarization: str = Field(default="V")
    orientation_mode: str = Field(default="fixed")
    orientation_rad: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])


class AntennaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bs_array: ArraySpec = Field(default_factory=ArraySpec)
    ue_array: ArraySpec = Field(default_factory=ArraySpec)


# ── rt ───────────────────────────────────────────────────────────────
class RTConfig(BaseModel):
    engine: str = Field(default="sionna_rt")
    max_depth: int = Field(default=1, ge=0)
    los: bool = True
    specular_reflection: bool = True
    diffuse_reflection: bool = False
    refraction: bool = False
    diffraction: bool = False
    synthetic_array: bool = False
    normalize_cfr: bool = False
    normalize_delays: bool = False
    merge_shapes: bool = False


# ── phy ──────────────────────────────────────────────────────────────
class SRSHoppingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    frequency_offsets_prb: list[int] = Field(default_factory=list)
    bandwidth_num_prb: list[int] = Field(default_factory=list)

    @model_validator(mode="after")
    def check_hopping_values(self) -> SRSHoppingConfig:
        if any(int(value) < 0 for value in self.frequency_offsets_prb):
            raise ValueError("phy.srs.hopping.frequency_offsets_prb must be non-negative")
        if any(int(value) < 1 for value in self.bandwidth_num_prb):
            raise ValueError("phy.srs.hopping.bandwidth_num_prb must be positive")
        return self


class SRSPortsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_srs_ports: int | None = Field(default=None, ge=1)
    mapping: str = Field(default="one_to_one")
    port_tx_ant_map: list[list[int]] | None = None
    usage: str = Field(default="non_codebook")

    @model_validator(mode="after")
    def check_ports_values(self) -> SRSPortsConfig:
        if self.mapping not in ("one_to_one", "antenna_switching"):
            raise ValueError("phy.srs.ports.mapping must be one_to_one/antenna_switching")
        if self.usage not in ("codebook", "non_codebook"):
            raise ValueError("phy.srs.ports.usage must be codebook/non_codebook")
        if self.port_tx_ant_map is not None:
            if not self.port_tx_ant_map or any(not row for row in self.port_tx_ant_map):
                raise ValueError("phy.srs.ports.port_tx_ant_map must be a non-empty 2-D list")
            width = len(self.port_tx_ant_map[0])
            if any(len(row) != width for row in self.port_tx_ant_map):
                raise ValueError("phy.srs.ports.port_tx_ant_map rows must have equal length")
            if any(int(value) < -1 for row in self.port_tx_ant_map for value in row):
                raise ValueError("phy.srs.ports.port_tx_ant_map values must be >= -1")
        return self


class SRSPowerControlConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    p0_dbm: float = 0.0
    alpha: float = Field(default=0.8, ge=0.0, le=1.0)
    min_tx_power_dbm: float = -40.0
    max_tx_power_dbm: float = 23.0
    serving_rx_policy: str = Field(default="strongest_path")

    @model_validator(mode="after")
    def check_power_control_values(self) -> SRSPowerControlConfig:
        if self.min_tx_power_dbm > self.max_tx_power_dbm:
            raise ValueError("phy.srs.power_control min_tx_power_dbm must be <= max")
        if self.serving_rx_policy not in ("strongest_path", "first_rx"):
            raise ValueError(
                "phy.srs.power_control.serving_rx_policy must be strongest_path/first_rx"
            )
        return self


class SRSMultiUserConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    active_ue_count: int = Field(default=2, ge=1)
    resource_strategy: str = Field(default="comb_offset")
    frame_policy: str = Field(default="sequential")

    @model_validator(mode="after")
    def check_multiuser_values(self) -> SRSMultiUserConfig:
        if self.resource_strategy not in ("comb_offset", "prb_split"):
            raise ValueError("phy.srs.multiuser.resource_strategy must be comb_offset/prb_split")
        if self.frame_policy != "sequential":
            raise ValueError("phy.srs.multiuser.frame_policy currently supports sequential only")
        return self


class ThermalNoiseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature_k: float = Field(default=290.0, gt=0)
    noise_figure_db: float = Field(default=7.0, ge=0)
    bandwidth_hz: float | None = Field(default=None, gt=0)


class UplinkPowerControlConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    serving_rx_policy: str = Field(default="strongest_path")
    open_loop_enabled: bool = True
    p0_dbm: float = 0.0
    alpha: float = Field(default=0.8, ge=0.0, le=1.0)
    closed_loop_enabled: bool = False
    tpc_offset_db: float = 0.0
    accumulation_db: float = 0.0
    min_tx_power_dbm: float = -40.0
    max_tx_power_dbm: float = 23.0

    @model_validator(mode="after")
    def check_uplink_control_values(self) -> UplinkPowerControlConfig:
        if self.serving_rx_policy not in ("strongest_path", "first_rx"):
            raise ValueError(
                "phy.power.uplink_control.serving_rx_policy must be "
                "strongest_path/first_rx"
            )
        if self.min_tx_power_dbm > self.max_tx_power_dbm:
            raise ValueError("phy.power.uplink_control min_tx_power_dbm must be <= max")
        return self


class PHYPowerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference_tx_power_dbm: float = 0.0
    apply_tx_power_to_grid: bool = True
    noise_mode: str = Field(default="relative_snr")
    thermal_noise: ThermalNoiseConfig = Field(default_factory=ThermalNoiseConfig)
    uplink_control: UplinkPowerControlConfig = Field(default_factory=UplinkPowerControlConfig)

    @model_validator(mode="after")
    def check_power_values(self) -> PHYPowerConfig:
        if self.noise_mode not in ("relative_snr", "absolute_thermal"):
            raise ValueError("phy.power.noise_mode must be relative_snr/absolute_thermal")
        return self


class PHYIQConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    save_frequency_clean: bool = False
    save_frequency_observed: bool = False
    save_time_clean: bool = False
    save_time_observed: bool = False
    cp_length: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def check_iq_values(self) -> PHYIQConfig:
        if self.enabled and not any(
            (
                self.save_frequency_clean,
                self.save_frequency_observed,
                self.save_time_clean,
                self.save_time_observed,
            )
        ):
            raise ValueError("phy.iq.enabled=true requires at least one save_* flag")
        return self


class SRSConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot_length_symbols: int = Field(default=14, ge=1)
    start_symbol: int = Field(default=12, ge=0)
    num_srs_symbols: int = Field(default=2, ge=1)
    comb_size: int = Field(default=2, ge=1)
    comb_offset: int = Field(default=0, ge=0)
    bwp_start_prb: int = Field(default=0, ge=0)
    bwp_num_prb: int | None = Field(default=None, ge=1)
    trigger_mode: str = Field(default="aperiodic")
    periodicity_slots: int = Field(default=1, ge=1)
    slot_offset: int = Field(default=0, ge=0)
    slot_number: int = Field(default=0, ge=0)
    sequence_type: str = Field(default="zc_like")
    sequence_id: int = Field(default=0, ge=0)
    group_hopping: str = Field(default="disabled")
    sequence_hopping: str = Field(default="disabled")
    cyclic_shift_multiplexing: str = Field(default="cyclic_shift")
    cyclic_shift_indices: list[int] | None = None
    hopping: SRSHoppingConfig = Field(default_factory=SRSHoppingConfig)
    ports: SRSPortsConfig = Field(default_factory=SRSPortsConfig)
    power_control: SRSPowerControlConfig = Field(default_factory=SRSPowerControlConfig)
    multiuser: SRSMultiUserConfig = Field(default_factory=SRSMultiUserConfig)

    @model_validator(mode="after")
    def check_supported_values(self) -> SRSConfig:
        if self.start_symbol + self.num_srs_symbols > self.slot_length_symbols:
            raise ValueError("phy.srs SRS symbols must fit within the slot")
        if self.comb_size not in (1, 2, 4):
            raise ValueError("phy.srs.comb_size must be one of 1, 2, 4")
        if self.comb_offset >= self.comb_size:
            raise ValueError("phy.srs.comb_offset must be < comb_size")
        if self.trigger_mode not in ("aperiodic", "periodic", "semipersistent"):
            raise ValueError(
                "phy.srs.trigger_mode must be aperiodic/periodic/semipersistent"
            )
        if self.sequence_type not in ("zc_like", "nr_zc"):
            raise ValueError("phy.srs.sequence_type must be zc_like/nr_zc")
        if self.group_hopping not in ("disabled", "enabled"):
            raise ValueError("phy.srs.group_hopping must be disabled/enabled")
        if self.sequence_hopping not in ("disabled", "enabled"):
            raise ValueError("phy.srs.sequence_hopping must be disabled/enabled")
        if self.cyclic_shift_multiplexing not in ("time", "cyclic_shift"):
            raise ValueError("phy.srs.cyclic_shift_multiplexing must be time/cyclic_shift")
        if self.cyclic_shift_indices is not None:
            if any(int(value) < 0 or int(value) > 11 for value in self.cyclic_shift_indices):
                raise ValueError("phy.srs.cyclic_shift_indices must be in [0, 11]")
        return self


class PHYConfig(BaseModel):
    enabled: bool = True
    standard: str = Field(default="custom_ofdm")
    snr_db: float = Field(default=30.0)
    fft_size: int = Field(default=64, ge=2)
    cp_length: int = Field(default=0, ge=0)
    num_ofdm_symbols: int = Field(default=1, ge=1)
    pilot_pattern: str = Field(default="all_active_subcarriers")
    channel_estimator: str = Field(default="ls")
    interpolation: str = Field(default="none")
    tx_power_dbm: float = Field(default=0.0)

    # NR PUSCH parameters
    subcarrier_spacing_khz: int = 30
    num_prb: int = 16
    num_layers: int = 1
    num_antenna_ports: int = 4
    mcs_index: int = 14
    mcs_table: int = 1
    perfect_csi: bool = False
    ebno_db: float | None = None
    pusch_dmrs_config_type: int = 1
    pusch_dmrs_length: int = 1
    pusch_dmrs_additional_position: int = 1
    pusch_num_cdm_groups_without_data: int = 2
    # MIMO / receiver fields (used for NR PUSCH)
    mimo_mode: str = "su_mimo"
    channel_backend: str = "apply_ofdm"
    mimo_detector: str = "lmmse"
    channel_estimator: str = "pusch_ls"
    receiver_failure_policy: str = "fail_fast"
    su_mimo_link_batch_size: int = Field(default=1, ge=1)
    power: PHYPowerConfig = Field(default_factory=PHYPowerConfig)
    iq: PHYIQConfig = Field(default_factory=PHYIQConfig)
    srs: SRSConfig = Field(default_factory=SRSConfig)

    @model_validator(mode="after")
    def check_fft_consistent(self) -> PHYConfig:
        if self.fft_size < 2:
            raise ValueError("fft_size must be >= 2")
        return self


# ── impairments ──────────────────────────────────────────────────────
class AWGNImpairmentConfig(BaseModel):
    enabled: bool = True


class CFOImpairmentConfig(BaseModel):
    enabled: bool = True
    cfo_hz: float | None = 100.0


class SFOImpairmentConfig(BaseModel):
    enabled: bool = True
    sfo_ppm: float | None = 5.0


class PhaseNoiseConfig(BaseModel):
    enabled: bool = True
    phase_offset_rad: float | None = 0.5


class TimingOffsetConfig(BaseModel):
    enabled: bool = True
    timing_offset_samples: float | None = 2.0


class AGCADCConfig(BaseModel):
    enabled: bool = True
    agc_gain_db: float = 0.0
    clipping_threshold: float | None = 3.0


class ImpairmentsConfig(BaseModel):
    awgn: AWGNImpairmentConfig = Field(default_factory=AWGNImpairmentConfig)
    cfo: CFOImpairmentConfig = Field(default_factory=CFOImpairmentConfig)
    sfo: SFOImpairmentConfig = Field(default_factory=SFOImpairmentConfig)
    phase_noise: PhaseNoiseConfig = Field(default_factory=PhaseNoiseConfig)
    timing_offset: TimingOffsetConfig = Field(default_factory=TimingOffsetConfig)
    agc_adc: AGCADCConfig = Field(default_factory=AGCADCConfig)
    impairment_seed: int = Field(default=142)


# ── ranging ──────────────────────────────────────────────────────────
class PdpPeakRangingConfig(BaseModel):
    oversampling_factor: int = Field(default=8, ge=1)
    window: str = Field(default="hann")
    peak_policy: str = Field(default="earliest_above_relative_threshold")
    relative_threshold_db: float = Field(default=-12.0)
    min_peak_snr_db: float = Field(default=6.0)
    interpolation: str = Field(default="parabolic_log_power")
    max_delay_s: float | None = None

    @model_validator(mode="after")
    def check_supported_values(self) -> PdpPeakRangingConfig:
        if self.window not in ("hann", "rect"):
            raise ValueError("ranging.pdp_peak.window must be hann/rect")
        if self.peak_policy != "earliest_above_relative_threshold":
            raise ValueError(
                "ranging.pdp_peak.peak_policy must be "
                "earliest_above_relative_threshold"
            )
        if self.interpolation not in ("parabolic_log_power", "none"):
            raise ValueError(
                "ranging.pdp_peak.interpolation must be parabolic_log_power/none"
            )
        if self.max_delay_s is not None and self.max_delay_s <= 0:
            raise ValueError("ranging.pdp_peak.max_delay_s must be positive when set")
        return self


class PhaseSlopeRangingConfig(BaseModel):
    unwrap: bool = True
    aggregate: str = Field(default="power_weighted_median")
    min_mean_power: float = Field(default=1.0e-12, ge=0)

    @model_validator(mode="after")
    def check_supported_values(self) -> PhaseSlopeRangingConfig:
        if self.aggregate != "power_weighted_median":
            raise ValueError(
                "ranging.phase_slope.aggregate must be power_weighted_median"
            )
        return self


class RangingConfig(BaseModel):
    enabled: bool = False
    source: str = Field(default="cfr_est")
    estimators: list[str] = Field(default_factory=lambda: ["pdp_peak", "phase_slope"])
    default_estimator: str = Field(default="pdp_peak")
    write_rtt_equivalent: bool = True
    pdp_peak: PdpPeakRangingConfig = Field(default_factory=PdpPeakRangingConfig)
    phase_slope: PhaseSlopeRangingConfig = Field(default_factory=PhaseSlopeRangingConfig)

    @model_validator(mode="after")
    def check_supported_values(self) -> RangingConfig:
        supported = {"pdp_peak", "phase_slope"}
        unknown = sorted(set(self.estimators) - supported)
        if unknown:
            raise ValueError(f"Unsupported ranging estimators: {unknown}")
        if self.source != "cfr_est":
            raise ValueError("Only ranging.source='cfr_est' is supported")
        if self.default_estimator not in self.estimators:
            raise ValueError("ranging.default_estimator must be listed in estimators")
        return self


# ── receiver ─────────────────────────────────────────────────────────
class ReceiverConfig(BaseModel):
    estimator_type: str = Field(default="ls")
    channel_estimator: str = Field(default="pusch_ls")
    sync_method: str = Field(default="ideal")
    interpolation_method: str = Field(default="none")
    packet_detection_threshold: float = Field(default=0.0)
    failure_policy: str = Field(default="mark_invalid")
    mimo_detector: str = Field(default="lmmse")
    calibration_profile_id: str = Field(default="synthetic_default")


# ── motion ───────────────────────────────────────────────────────────
class MotionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    mobility_mode: str = Field(default="static")
    num_time_steps: int = Field(default=3, ge=1)
    sampling_frequency_hz: float = Field(default=100.0, ge=0)
    bs_velocity_mps: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    ue_velocity_mps: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])

    @model_validator(mode="after")
    def check_velocity_shape(self) -> MotionConfig:
        for name in ("bs_velocity_mps", "ue_velocity_mps"):
            v = getattr(self, name)
            if len(v) != 3:
                raise ValueError(f"{name} must have 3 components, got {len(v)}")
        return self


# ── link ─────────────────────────────────────────────────────────────
class LinkConfig(BaseModel):
    """Link direction configuration."""

    model_config = ConfigDict(extra="forbid")

    duplex_mode: str = Field(default="tdd")
    phy_link_direction: str = Field(default="uplink")

    @model_validator(mode="after")
    def check_link_values(self) -> LinkConfig:
        if self.duplex_mode != "tdd":
            raise ValueError("Only TDD duplex mode is supported")
        if self.phy_link_direction not in ("uplink", "downlink"):
            raise ValueError("link.phy_link_direction must be uplink/downlink")
        return self


# ── calibration ──────────────────────────────────────────────────────
class CalibrationConfig(BaseModel):
    enabled: bool = True
    profile_id: str = Field(default="synthetic_default")


class NonCooperativeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    signal_standard: str = Field(default="nr_srs")
    active_tx_count: int = Field(default=2, ge=1)
    frame_policy: str = Field(default="sequential")
    resource_strategy: str = Field(default="comb_offset")
    save_time_clean: bool = True
    save_time_observed: bool = True
    cp_length: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def check_noncooperative_values(self) -> NonCooperativeConfig:
        if self.signal_standard != "nr_srs":
            raise ValueError("noncooperative.signal_standard currently supports nr_srs")
        if self.frame_policy != "sequential":
            raise ValueError("noncooperative.frame_policy currently supports sequential")
        if self.resource_strategy not in ("comb_offset", "prb_split"):
            raise ValueError(
                "noncooperative.resource_strategy must be comb_offset/prb_split"
            )
        if self.enabled and not (self.save_time_clean or self.save_time_observed):
            raise ValueError(
                "noncooperative.enabled=true requires save_time_clean or "
                "save_time_observed"
            )
        return self


class VisualizationConfig(BaseModel):
    enabled: bool = False
    output_dir: str = Field(default="figures")
    sample_policy: str = Field(default="valid_links_first")
    random_seed: int = Field(default=42, ge=0)
    max_bs: int = Field(default=5, ge=1)
    sample_ue_count: int = Field(default=3, ge=1)
    max_ue: int = Field(default=5, ge=1)
    dpi: int = Field(default=140, ge=50)
    format: str = Field(default="png")
    plots: list[str] = Field(default_factory=lambda: list(DEFAULT_VISUALIZATION_PLOTS))
    radio_map_mode: str = Field(default="interpolated")
    radio_map_grid_resolution_m: float | None = Field(default=None, gt=0)
    radio_map_show_samples: bool = Field(default=False)

    @model_validator(mode="after")
    def check_visualization_values(self) -> VisualizationConfig:
        if self.sample_policy not in (
            "valid_links_first",
            "spatially_spread_valid_links",
            "random",
            "first",
        ):
            raise ValueError(
                "visualization.sample_policy must be "
                "valid_links_first/spatially_spread_valid_links/random/first"
            )
        if self.format != "png":
            raise ValueError("Only visualization.format='png' is supported")
        if self.radio_map_mode not in ("interpolated", "samples", "both"):
            raise ValueError(
                "visualization.radio_map_mode must be interpolated/samples/both"
            )
        unknown_plots = set(self.plots) - set(ALLOWED_VISUALIZATION_PLOTS)
        if unknown_plots:
            raise ValueError(f"Unsupported visualization plots: {sorted(unknown_plots)}")
        if self.sample_ue_count > self.max_ue:
            object.__setattr__(self, "sample_ue_count", self.max_ue)
        return self


# ── top-level ────────────────────────────────────────────────────────
class MeasurementConfig(BaseModel):
    """Complete measurement simulation configuration."""

    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)
    input: InputConfig = Field(default_factory=InputConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    carrier: CarrierConfig = Field(default_factory=CarrierConfig)
    antenna: AntennaConfig = Field(default_factory=AntennaConfig)
    rt: RTConfig = Field(default_factory=RTConfig)
    phy: PHYConfig = Field(default_factory=PHYConfig)
    array: ArrayConfig = Field(default_factory=ArrayConfig)
    impairments: ImpairmentsConfig = Field(default_factory=ImpairmentsConfig)
    ranging: RangingConfig = Field(default_factory=RangingConfig)
    receiver: ReceiverConfig = Field(default_factory=ReceiverConfig)
    motion: MotionConfig = Field(default_factory=MotionConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    link: LinkConfig = Field(default_factory=LinkConfig)
    noncooperative: NonCooperativeConfig = Field(default_factory=NonCooperativeConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)

    @model_validator(mode="after")
    def check_phy_requires_observation(self) -> MeasurementConfig:
        if self.phy.enabled:
            if self.phy.fft_size < 2:
                raise ValueError("phy.fft_size must be >= 2 when phy enabled")
        if (
            self.output.profile == "full"
            and self.ranging.enabled
            and not self.phy.enabled
        ):
            raise ValueError(
                "ranging.enabled=true requires phy.enabled=true and /observation/cfr_est"
            )
        if self.output.profile == "full" and self.phy.iq.enabled and not self.phy.enabled:
            raise ValueError("phy.iq.enabled=true requires phy.enabled=true")
        if self.output.profile == "full" and self.noncooperative.enabled:
            if not self.phy.enabled:
                raise ValueError("noncooperative.enabled=true requires phy.enabled=true")
            if self.phy.standard != "nr_srs":
                raise ValueError("noncooperative.enabled=true currently requires nr_srs")
        if self.motion.enabled and self.motion.mobility_mode == "doppler_synthetic":
            if self.motion.sampling_frequency_hz <= 0:
                raise ValueError(
                    "motion.sampling_frequency_hz required for doppler_synthetic"
                )
        return self
