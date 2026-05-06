"""Analysis and diagnostics helpers."""

from __future__ import annotations

import numpy as np

from sionna_measurement_sim.domain.observation import (
    EvaluationResult,
    ObservationResult,
)


class DiagnosticsSummary:
    """Compact summary of link-level evaluation diagnostics."""

    median_nmse_db: float
    median_snr_db: float
    detection_rate: float
    estimation_failure_rate: float
    num_links: int
    num_failed_links: int

    def __init__(
        self,
        median_nmse_db: float,
        median_snr_db: float,
        detection_rate: float,
        estimation_failure_rate: float,
        num_links: int,
        num_failed_links: int,
    ) -> None:
        self.median_nmse_db = median_nmse_db
        self.median_snr_db = median_snr_db
        self.detection_rate = detection_rate
        self.estimation_failure_rate = estimation_failure_rate
        self.num_links = num_links
        self.num_failed_links = num_failed_links

    def to_dict(self) -> dict[str, float | int]:
        return {
            "median_nmse_db": self.median_nmse_db,
            "median_snr_db": self.median_snr_db,
            "detection_rate": self.detection_rate,
            "estimation_failure_rate": self.estimation_failure_rate,
            "num_links": self.num_links,
            "num_failed_links": self.num_failed_links,
        }

    @classmethod
    def from_evaluation_result(
        cls,
        evaluation: EvaluationResult,
        observation: ObservationResult,
    ) -> DiagnosticsSummary:
        """Build a summary from an evaluation and its observation."""
        nmse = evaluation.nmse_db.ravel()
        valid = observation.valid_mask.ravel()
        snr = observation.snr_db.ravel()
        num_links = int(np.prod(evaluation.nmse_db.shape))
        num_failed_links = int(np.sum(~observation.estimation_success))
        return cls(
            median_nmse_db=float(np.median(nmse[valid])) if np.any(valid) else 0.0,
            median_snr_db=float(np.median(snr[valid])) if np.any(valid) else 0.0,
            detection_rate=float(evaluation.detection_rate),
            estimation_failure_rate=float(evaluation.estimation_failure_rate),
            num_links=num_links,
            num_failed_links=num_failed_links,
        )


__all__ = [
    "DiagnosticsSummary",
]
