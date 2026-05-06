"""PHY observation domain models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sionna_measurement_sim.domain.validation import require_finite, require_shape


@dataclass(frozen=True)
class WaveformSpec:
    """Minimal custom OFDM waveform description."""

    standard: str
    sample_rate_hz: float
    fft_size: int
    cp_length: int
    num_ofdm_symbols: int
    pilot_indices: np.ndarray
    data_subcarrier_indices: np.ndarray
    pilot_symbols: np.ndarray
    tx_power_dbm: float

    def __post_init__(self) -> None:
        pilot_indices = np.asarray(self.pilot_indices, dtype=np.int32)
        data_indices = np.asarray(self.data_subcarrier_indices, dtype=np.int32)
        pilot_symbols = np.asarray(self.pilot_symbols, dtype=np.complex64)
        require_shape("pilot_indices", pilot_indices, (None,))
        require_shape("data_subcarrier_indices", data_indices, (None,))
        require_shape("pilot_symbols", pilot_symbols, pilot_indices.shape)
        require_finite("pilot_symbols", pilot_symbols)

        object.__setattr__(self, "pilot_indices", pilot_indices)
        object.__setattr__(self, "data_subcarrier_indices", data_indices)
        object.__setattr__(self, "pilot_symbols", pilot_symbols)


@dataclass(frozen=True)
class ObservationResult:
    """Estimated CFR and receiver diagnostics."""

    cfr_est: np.ndarray
    valid_mask: np.ndarray
    detection_success: np.ndarray
    estimation_success: np.ndarray
    snr_db: np.ndarray
    rssi_dbm: np.ndarray
    noise_power_dbm: np.ndarray
    cfo_hz: np.ndarray
    sfo_ppm: np.ndarray
    timing_offset_samples: np.ndarray
    phase_offset_rad: np.ndarray
    agc_gain_db: np.ndarray
    clipping_flag: np.ndarray

    def __post_init__(self) -> None:
        cfr_est = np.asarray(self.cfr_est, dtype=np.complex64)
        valid_mask = np.asarray(self.valid_mask, dtype=np.bool_)
        detection_success = np.asarray(self.detection_success, dtype=np.bool_)
        estimation_success = np.asarray(self.estimation_success, dtype=np.bool_)
        snr_db = np.asarray(self.snr_db, dtype=np.float32)
        rssi_dbm = np.asarray(self.rssi_dbm, dtype=np.float32)
        noise_power_dbm = np.asarray(self.noise_power_dbm, dtype=np.float32)
        cfo_hz = np.asarray(self.cfo_hz, dtype=np.float32)
        sfo_ppm = np.asarray(self.sfo_ppm, dtype=np.float32)
        timing_offset_samples = np.asarray(self.timing_offset_samples, dtype=np.float32)
        phase_offset_rad = np.asarray(self.phase_offset_rad, dtype=np.float32)
        agc_gain_db = np.asarray(self.agc_gain_db, dtype=np.float32)
        clipping_flag = np.asarray(self.clipping_flag, dtype=np.bool_)

        require_shape("cfr_est", cfr_est, (None, None, None, None, None, None))
        snapshot, tx, rx = cfr_est.shape[:3]
        link_shape = (snapshot, tx, rx)
        for name, value in (
            ("valid_mask", valid_mask),
            ("detection_success", detection_success),
            ("estimation_success", estimation_success),
            ("snr_db", snr_db),
            ("rssi_dbm", rssi_dbm),
            ("noise_power_dbm", noise_power_dbm),
            ("cfo_hz", cfo_hz),
            ("sfo_ppm", sfo_ppm),
            ("timing_offset_samples", timing_offset_samples),
            ("phase_offset_rad", phase_offset_rad),
            ("clipping_flag", clipping_flag),
        ):
            require_shape(name, value, link_shape)
        require_shape("agc_gain_db", agc_gain_db, (snapshot, rx))
        require_finite("cfr_est", cfr_est)
        require_finite("snr_db", snr_db)
        require_finite("rssi_dbm", rssi_dbm)
        require_finite("noise_power_dbm", noise_power_dbm)

        object.__setattr__(self, "cfr_est", cfr_est)
        object.__setattr__(self, "valid_mask", valid_mask)
        object.__setattr__(self, "detection_success", detection_success)
        object.__setattr__(self, "estimation_success", estimation_success)
        object.__setattr__(self, "snr_db", snr_db)
        object.__setattr__(self, "rssi_dbm", rssi_dbm)
        object.__setattr__(self, "noise_power_dbm", noise_power_dbm)
        object.__setattr__(self, "cfo_hz", cfo_hz)
        object.__setattr__(self, "sfo_ppm", sfo_ppm)
        object.__setattr__(self, "timing_offset_samples", timing_offset_samples)
        object.__setattr__(self, "phase_offset_rad", phase_offset_rad)
        object.__setattr__(self, "agc_gain_db", agc_gain_db)
        object.__setattr__(self, "clipping_flag", clipping_flag)


@dataclass(frozen=True)
class ImpairmentSpec:
    """Impairment configuration snapshot."""

    model_version: str
    random_seed: int
    awgn_config: str
    cfo_sfo_config: str = "{}"
    phase_noise_config: str = "{}"
    iq_imbalance_config: str = "{}"
    agc_adc_config: str = "{}"


@dataclass(frozen=True)
class ReceiverSpec:
    """Receiver algorithm configuration."""

    estimator_type: str = "ls"
    sync_method: str = "perfect"
    interpolation_method: str = "none"
    packet_detection_threshold: float = 0.0
    failure_policy: str = "mark_invalid"
    calibration_profile_id: str = "synthetic_default"


@dataclass(frozen=True)
class EvaluationResult:
    """Truth-vs-observation metrics."""

    nmse_db: np.ndarray
    amplitude_error_db: np.ndarray
    phase_error_rad: np.ndarray
    correlation: np.ndarray
    detection_rate: float
    estimation_failure_rate: float

    def __post_init__(self) -> None:
        nmse_db = np.asarray(self.nmse_db, dtype=np.float32)
        amplitude_error_db = np.asarray(self.amplitude_error_db, dtype=np.float32)
        phase_error_rad = np.asarray(self.phase_error_rad, dtype=np.float32)
        correlation = np.asarray(self.correlation, dtype=np.float32)
        require_shape("nmse_db", nmse_db, (None, None, None))
        for name, value in (
            ("amplitude_error_db", amplitude_error_db),
            ("phase_error_rad", phase_error_rad),
            ("correlation", correlation),
        ):
            require_shape(name, value, nmse_db.shape)
            require_finite(name, value)
        require_finite("nmse_db", nmse_db)

        object.__setattr__(self, "nmse_db", nmse_db)
        object.__setattr__(self, "amplitude_error_db", amplitude_error_db)
        object.__setattr__(self, "phase_error_rad", phase_error_rad)
        object.__setattr__(self, "correlation", correlation)


@dataclass(frozen=True)
class CalibrationResult:
    """Calibration profile and fitted parameters."""

    profile_id: str
    fitted_parameters: str  # JSON
    validation_metrics: str  # JSON

    @classmethod
    def synthetic_default(cls) -> CalibrationResult:
        import json
        from datetime import UTC, datetime

        return cls(
            profile_id="synthetic_default",
            fitted_parameters=json.dumps({"correction_mode": "none"}),
            validation_metrics=json.dumps(
                {
                    "applied_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
                    "method": "synthetic_identity",
                }
            ),
        )


@dataclass(frozen=True)
class DiagnosticsReport:
    """Aggregate diagnostics summary across all links."""

    median_nmse_db: float
    median_snr_db: float
    median_phase_error_rad: float
    detection_rate: float
    estimation_failure_rate: float
    worst_link_nmse_db: float
    worst_link_index: tuple[int, int, int]  # (snapshot, tx, rx)
    num_links: int
    num_failed_links: int

    def to_summary_dict(self) -> dict:
        return {
            "median_nmse_db": self.median_nmse_db,
            "median_snr_db": self.median_snr_db,
            "median_phase_error_rad": self.median_phase_error_rad,
            "detection_rate": self.detection_rate,
            "estimation_failure_rate": self.estimation_failure_rate,
            "worst_link_nmse_db": self.worst_link_nmse_db,
            "worst_link_index": list(self.worst_link_index),
            "num_links": self.num_links,
            "num_failed_links": self.num_failed_links,
        }

    @classmethod
    def from_evaluation(
        cls,
        evaluation: EvaluationResult,
        observation: ObservationResult,
    ) -> DiagnosticsReport:
        import numpy as np

        nmse = evaluation.nmse_db.ravel()
        valid = observation.valid_mask.ravel()
        snr = observation.snr_db.ravel()
        phase = evaluation.phase_error_rad.ravel()
        num_links = int(np.prod(evaluation.nmse_db.shape))
        num_failed = int(np.sum(~observation.estimation_success))
        worst_idx_flat = int(np.argmax(nmse))
        worst_idx = np.unravel_index(worst_idx_flat, evaluation.nmse_db.shape)
        return cls(
            median_nmse_db=float(np.median(nmse[valid])) if np.any(valid) else 0.0,
            median_snr_db=float(np.median(snr[valid])) if np.any(valid) else 0.0,
            median_phase_error_rad=float(np.median(np.abs(phase[valid]))) if np.any(valid) else 0.0,
            detection_rate=float(evaluation.detection_rate),
            estimation_failure_rate=float(evaluation.estimation_failure_rate),
            worst_link_nmse_db=float(nmse[worst_idx_flat]),
            worst_link_index=tuple(int(i) for i in worst_idx),
            num_links=num_links,
            num_failed_links=num_failed,
        )
