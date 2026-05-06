"""Pydantic configuration schema for SionnaMeasurementSim.

Each config group defined here maps to a section in the YAML config file.
Validation fails early (before RT/PHY starts) for any schema violation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


# ── runtime ──────────────────────────────────────────────────────────
class RuntimeConfig(BaseModel):
    seed: int = Field(default=42, ge=0)
    device: str = Field(default="cpu")
    require_gpu: bool = False
    precision: str = Field(default="single")
    torch_deterministic: bool = False


# ── input ────────────────────────────────────────────────────────────
class InputConfig(BaseModel):
    label_file: str = Field(default="data/scenes/test/test5.json")
    scene_file: str = Field(default="data/scenes/test/scene.xml")
    label_schema: str = Field(default="simplesionna_v1")
    coordinate_system: str = Field(default="scene_local_xyz_m")
    max_tx: int = Field(default=6, ge=1)
    max_rx: int = Field(default=100, ge=1)


# ── output ───────────────────────────────────────────────────────────
class OutputConfig(BaseModel):
    root_dir: str = Field(default="outputs")
    run_id_format: str = Field(default="{label_stem}_{timestamp}")
    hdf5_filename: str = Field(default="results.h5")
    compression: str = Field(default="gzip")
    save_full_paths: bool = False
    save_sampled_paths: bool = True
    save_raw_waveform: bool = False


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
    sync_method: str = Field(default="ideal")
    interpolation_method: str = Field(default="none")
    packet_detection_threshold: float = Field(default=0.0)
    failure_policy: str = Field(default="mark_invalid")
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


# ── calibration ──────────────────────────────────────────────────────
class CalibrationConfig(BaseModel):
    enabled: bool = True
    profile_id: str = Field(default="synthetic_default")


# ── top-level ────────────────────────────────────────────────────────
class MeasurementConfig(BaseModel):
    """Complete measurement simulation configuration."""

    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    input: InputConfig = Field(default_factory=InputConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    carrier: CarrierConfig = Field(default_factory=CarrierConfig)
    antenna: AntennaConfig = Field(default_factory=AntennaConfig)
    rt: RTConfig = Field(default_factory=RTConfig)
    phy: PHYConfig = Field(default_factory=PHYConfig)
    impairments: ImpairmentsConfig = Field(default_factory=ImpairmentsConfig)
    receiver: ReceiverConfig = Field(default_factory=ReceiverConfig)
    motion: MotionConfig = Field(default_factory=MotionConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)

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
