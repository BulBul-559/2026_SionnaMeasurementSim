"""Pydantic configuration schema for SionnaMeasurementSim.

Each config group defined here maps to a section in the YAML config file.
Validation fails early (before RT/PHY starts) for any schema violation.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, model_validator

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
    label_file: str = Field(default="tests/fixtures/scenes/test/test5.json")
    scene_file: str = Field(default="tests/fixtures/scenes/test/scene.xml")
    scene_id: str = ""
    map_id: str = ""
    label_schema: str = Field(default="simplesionna_v1")
    coordinate_system: str = Field(default="scene_local_xyz_m")
    max_tx: int = Field(default=6, ge=1)
    max_rx: int = Field(default=100, ge=1)

    @model_validator(mode="after")
    def default_scene_id(self) -> InputConfig:
        if not self.scene_id:
            object.__setattr__(self, "scene_id", Path(self.scene_file).stem)
        return self


# ── output ───────────────────────────────────────────────────────────
class OutputShardingConfig(BaseModel):
    enabled: bool = False
    axis: str = Field(default="rx")
    shard_size: int = Field(default=1000, ge=1)
    filename_pattern: str = Field(default="result_{shard_index:03d}.h5")
    parallel_workers: int = Field(default=1, ge=1)
    gpu_ids: list[int] = Field(default_factory=list)
    visualization_mode: str = Field(default="first_shard")

    @model_validator(mode="after")
    def check_sharding_values(self) -> OutputShardingConfig:
        if self.axis not in ("rx", "ue"):
            raise ValueError("output.sharding.axis must be 'rx' or 'ue'")
        if "{shard_index" not in self.filename_pattern:
            raise ValueError("output.sharding.filename_pattern must include {shard_index...}")
        if self.visualization_mode not in ("none", "first_shard", "all_shards"):
            raise ValueError(
                "output.sharding.visualization_mode must be none/first_shard/all_shards"
            )
        return self


class OutputConfig(BaseModel):
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
        allowed_sources = {"truth_cfr", "cfr_est", "rx_grid", "srs_cfr_est"}
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
    tx_array: ArraySpec = Field(default_factory=ArraySpec)
    rx_array: ArraySpec = Field(default_factory=ArraySpec)


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
    enabled: bool = True
    mobility_mode: str = Field(default="static")
    num_time_steps: int = Field(default=3, ge=1)
    sampling_frequency_hz: float = Field(default=100.0, ge=0)
    tx_velocity_mps: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    rx_velocity_mps: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])

    @model_validator(mode="after")
    def check_velocity_shape(self) -> MotionConfig:
        for name in ("tx_velocity_mps", "rx_velocity_mps"):
            v = getattr(self, name)
            if len(v) != 3:
                raise ValueError(f"{name} must have 3 components, got {len(v)}")
        return self


# ── link ─────────────────────────────────────────────────────────────
class LinkConfig(BaseModel):
    """NR PUSCH link configuration for TDD reciprocity."""

    duplex_mode: str = Field(default="tdd")
    phy_link_direction: str = Field(default="uplink")
    rt_trace_direction: str = Field(default="bs_to_ue")
    reciprocity_mode: str = Field(default="transpose_rt_channel")
    reciprocity_applied: bool = True


# ── calibration ──────────────────────────────────────────────────────
class CalibrationConfig(BaseModel):
    enabled: bool = True
    profile_id: str = Field(default="synthetic_default")


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
    receiver: ReceiverConfig = Field(default_factory=ReceiverConfig)
    motion: MotionConfig = Field(default_factory=MotionConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    link: LinkConfig = Field(default_factory=LinkConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)

    @model_validator(mode="after")
    def check_phy_requires_observation(self) -> MeasurementConfig:
        if self.phy.enabled:
            if self.phy.fft_size < 2:
                raise ValueError("phy.fft_size must be >= 2 when phy enabled")
        if self.motion.enabled and self.motion.mobility_mode == "doppler_synthetic":
            if self.motion.sampling_frequency_hz <= 0:
                raise ValueError(
                    "motion.sampling_frequency_hz required for doppler_synthetic"
                )
        return self
