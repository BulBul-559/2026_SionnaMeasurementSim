"""Batch processing domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class BatchConfig:
    """Configuration for a batch experiment run."""

    enabled: bool = False
    total_batches: int = 1
    completed_batches: int = 0
    failed_batches: int = 0

    @classmethod
    def single_batch(cls) -> BatchConfig:
        return cls(enabled=False, total_batches=1, completed_batches=1, failed_batches=0)


@dataclass(frozen=True)
class BatchManifestEntry:
    """Per-batch result summary."""

    batch_index: int
    batch_id: str
    status: str  # "completed" | "failed"
    results_h5: str
    error_message: str = ""

    def to_dict(self) -> dict:
        return {
            "batch_index": self.batch_index,
            "batch_id": self.batch_id,
            "status": self.status,
            "results_h5": self.results_h5,
            "error_message": self.error_message,
        }


@dataclass
class BatchExperimentResult:
    """Aggregate result from a batch experiment."""

    batch_config: BatchConfig
    entries: list[BatchManifestEntry] = field(default_factory=list)
    base_output_dir: Path = Path("outputs/batch")

    @property
    def succeeded(self) -> int:
        return sum(1 for e in self.entries if e.status == "completed")

    @property
    def failed(self) -> int:
        return sum(1 for e in self.entries if e.status == "failed")

    def to_manifest_dict(self) -> dict:
        return {
            "batching": {
                "enabled": self.batch_config.enabled,
                "total_batches": self.batch_config.total_batches,
                "completed_batches": self.succeeded,
                "failed_batches": self.failed,
            },
            "batches": [e.to_dict() for e in self.entries],
        }
