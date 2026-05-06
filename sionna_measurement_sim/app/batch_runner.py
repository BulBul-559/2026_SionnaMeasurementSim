"""Batch experiment orchestrator for Phase 8."""

from __future__ import annotations

from pathlib import Path

from sionna_measurement_sim.domain.batch import (
    BatchConfig,
    BatchExperimentResult,
    BatchManifestEntry,
)
from sionna_measurement_sim.io.manifest import write_manifest
from sionna_measurement_sim.rt.truth_pipeline import RTTruthRunConfig, run_rt_truth_pipeline


def run_batch_experiment(
    base_config: RTTruthRunConfig,
    batch_config: BatchConfig,
) -> BatchExperimentResult:
    """Run a batch experiment, producing per-batch HDF5 and aggregate manifest."""

    result = BatchExperimentResult(
        batch_config=batch_config,
        base_output_dir=base_config.output_dir,
    )

    for batch_idx in range(batch_config.total_batches):
        batch_id = f"batch_{batch_idx:03d}"
        batch_dir = base_config.output_dir / batch_id
        batch_seed = base_config.seed + batch_idx * 1000

        batch_run_config = RTTruthRunConfig(
            label_file=base_config.label_file,
            scene_file=base_config.scene_file,
            output_dir=batch_dir,
            center_frequency_hz=base_config.center_frequency_hz,
            bandwidth_hz=base_config.bandwidth_hz,
            num_subcarriers=base_config.num_subcarriers,
            seed=batch_seed,
            max_tx=base_config.max_tx,
            max_rx=base_config.max_rx,
            max_depth=base_config.max_depth,
            specular_reflection=base_config.specular_reflection,
            observation_snr_db=base_config.observation_snr_db,
            observation_seed=base_config.observation_seed + batch_idx * 1000,
            impairment_config=base_config.impairment_config,
            num_time_steps=base_config.num_time_steps,
            sampling_frequency_hz=base_config.sampling_frequency_hz,
            tx_velocity_mps=base_config.tx_velocity_mps,
            rx_velocity_mps=base_config.rx_velocity_mps,
        )

        try:
            h5_path = run_rt_truth_pipeline(batch_run_config)
            entry = BatchManifestEntry(
                batch_index=batch_idx,
                batch_id=batch_id,
                status="completed",
                results_h5=str(h5_path),
            )
        except Exception as exc:
            entry = BatchManifestEntry(
                batch_index=batch_idx,
                batch_id=batch_id,
                status="failed",
                results_h5="",
                error_message=str(exc)[:500],
            )
        result.entries.append(entry)
        _cleanup()

    _write_batch_manifest(result)
    return result


def _write_batch_manifest(result: BatchExperimentResult) -> Path:
    manifest_path = result.base_output_dir / "batch_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    data = result.to_manifest_dict()
    data["summary"] = {
        "total": result.batch_config.total_batches,
        "succeeded": result.succeeded,
        "failed": result.failed,
    }
    write_manifest(manifest_path, data)
    return manifest_path


def _cleanup() -> None:
    """Release GPU memory between batches if available."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
