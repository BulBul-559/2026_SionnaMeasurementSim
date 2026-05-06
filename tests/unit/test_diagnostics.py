import json

import numpy as np

from sionna_measurement_sim.domain.observation import (
    CalibrationResult,
    DiagnosticsReport,
    EvaluationResult,
    ObservationResult,
    ReceiverSpec,
)


class TestCalibrationResult:
    def test_synthetic_default(self):
        c = CalibrationResult.synthetic_default()
        assert c.profile_id == "synthetic_default"
        fitted = json.loads(c.fitted_parameters)
        assert fitted["correction_mode"] == "none"
        metrics = json.loads(c.validation_metrics)
        assert metrics["method"] == "synthetic_identity"


class TestDiagnosticsReport:
    def test_from_evaluation_perfect_link(self):
        evaluation = EvaluationResult(
            nmse_db=np.array([[[-30.0]]], dtype=np.float32),
            nmse_db_total=np.array([[[-30.0]]], dtype=np.float32),
            amplitude_error_db=np.array([[[-30.0]]], dtype=np.float32),
            phase_error_rad=np.array([[[0.001]]], dtype=np.float32),
            correlation=np.array([[[0.99]]], dtype=np.float32),
            detection_rate=1.0,
            estimation_failure_rate=0.0,
        )
        observation = ObservationResult(
            cfr_est=np.ones((1, 1, 1, 1, 1, 8), dtype=np.complex64),
            valid_mask=np.ones((1, 1, 1), dtype=np.bool_),
            detection_success=np.ones((1, 1, 1), dtype=np.bool_),
            estimation_success=np.ones((1, 1, 1), dtype=np.bool_),
            snr_db=np.full((1, 1, 1), 40.0, dtype=np.float32),
            rssi_dbm=np.zeros((1, 1, 1), dtype=np.float32),
            noise_power_dbm=np.zeros((1, 1, 1), dtype=np.float32),
            cfo_hz=np.zeros((1, 1, 1), dtype=np.float32),
            sfo_ppm=np.zeros((1, 1, 1), dtype=np.float32),
            timing_offset_samples=np.zeros((1, 1, 1), dtype=np.float32),
            phase_offset_rad=np.zeros((1, 1, 1), dtype=np.float32),
            agc_gain_db=np.zeros((1, 1), dtype=np.float32),
            clipping_flag=np.zeros((1, 1, 1), dtype=np.bool_),
        )
        report = DiagnosticsReport.from_evaluation(evaluation, observation)
        assert report.median_nmse_db == -30.0
        assert report.detection_rate == 1.0
        assert report.estimation_failure_rate == 0.0
        assert report.num_links == 1
        assert report.num_failed_links == 0

    def test_to_summary_dict(self):
        report = DiagnosticsReport(
            median_nmse_db=-25.0,
            median_snr_db=30.0,
            median_phase_error_rad=0.01,
            detection_rate=0.95,
            estimation_failure_rate=0.05,
            worst_link_nmse_db=-10.0,
            worst_link_index=(0, 1, 2),
            num_links=100,
            num_failed_links=5,
        )
        d = report.to_summary_dict()
        assert d["median_nmse_db"] == -25.0
        assert d["detection_rate"] == 0.95
        assert d["worst_link_index"] == [0, 1, 2]

    def test_from_evaluation_with_failures(self):
        evaluation = EvaluationResult(
            nmse_db=np.array([[[-10.0, -5.0], [-20.0, -15.0]]], dtype=np.float32),
            nmse_db_total=np.array([[[-8.0, -3.0], [-18.0, -12.0]]], dtype=np.float32),
            amplitude_error_db=np.zeros((1, 2, 2), dtype=np.float32),
            phase_error_rad=np.zeros((1, 2, 2), dtype=np.float32),
            correlation=np.ones((1, 2, 2), dtype=np.float32),
            detection_rate=0.75,
            estimation_failure_rate=0.25,
        )
        observation = ObservationResult(
            cfr_est=np.ones((1, 2, 2, 1, 1, 8), dtype=np.complex64),
            valid_mask=np.ones((1, 2, 2), dtype=np.bool_),
            detection_success=np.ones((1, 2, 2), dtype=np.bool_),
            estimation_success=np.ones((1, 2, 2), dtype=np.bool_),
            snr_db=np.full((1, 2, 2), 30.0, dtype=np.float32),
            rssi_dbm=np.zeros((1, 2, 2), dtype=np.float32),
            noise_power_dbm=np.zeros((1, 2, 2), dtype=np.float32),
            cfo_hz=np.zeros((1, 2, 2), dtype=np.float32),
            sfo_ppm=np.zeros((1, 2, 2), dtype=np.float32),
            timing_offset_samples=np.zeros((1, 2, 2), dtype=np.float32),
            phase_offset_rad=np.zeros((1, 2, 2), dtype=np.float32),
            agc_gain_db=np.zeros((1, 2), dtype=np.float32),
            clipping_flag=np.zeros((1, 2, 2), dtype=np.bool_),
        )
        report = DiagnosticsReport.from_evaluation(evaluation, observation)
        assert report.worst_link_nmse_db == -5.0
        assert report.num_links == 4
        assert report.detection_rate == 0.75


class TestReceiverSpec:
    def test_calibration_profile_id_default(self):
        r = ReceiverSpec()
        assert r.calibration_profile_id == "synthetic_default"
