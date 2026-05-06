"""Phases 2-6 RT truth pipeline."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from sionna_measurement_sim.adapters.sionna_rt.rt_solver import (
    SionnaRTConfig,
    run_sionna_rt_truth,
)
from sionna_measurement_sim.domain.antenna import AntennaSpec
from sionna_measurement_sim.domain.frequency import FrequencyGrid
from sionna_measurement_sim.domain.motion import MotionSpec
from sionna_measurement_sim.domain.observation import (
    CalibrationResult,
    DiagnosticsReport,
)
from sionna_measurement_sim.domain.results import (
    DeviceState,
    InputSpec,
    MeasurementSimulationResult,
    Metadata,
    RuntimeInfo,
    SceneSpec,
)
from sionna_measurement_sim.domain.topology import Topology
from sionna_measurement_sim.io.hdf5_writer import write_measurement_result
from sionna_measurement_sim.io.label_parser import load_topology_from_label
from sionna_measurement_sim.io.manifest import write_manifest
from sionna_measurement_sim.io.schema_validator import validate_hdf5_contract
from sionna_measurement_sim.phy.impairments import ImpairmentConfig
from sionna_measurement_sim.phy.observation_pipeline import (
    AWGNObservationConfig,
    run_awgn_ls_observation,
)


@dataclass(frozen=True)
class RTTruthRunConfig:
    """User-facing config for a minimal RT truth run."""

    label_file: Path
    scene_file: Path
    output_dir: Path
    center_frequency_hz: float = 3.5e9
    bandwidth_hz: float = 20e6
    num_subcarriers: int = 8
    seed: int = 1
    max_tx: int = 1
    max_rx: int = 1
    max_depth: int = 1
    specular_reflection: bool = True
    observation_snr_db: float | None = None
    observation_seed: int = 11
    impairment_config: ImpairmentConfig | None = None
    num_time_steps: int = 1
    sampling_frequency_hz: float = 0.0
    tx_velocity_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rx_velocity_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)


def run_rt_truth_pipeline(config: RTTruthRunConfig) -> Path:
    """Run Phase 2 minimal RT truth and write HDF5, manifest, and log."""

    start = time.perf_counter()
    output_dir = config.output_dir
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    topology = load_topology_from_label(
        config.label_file,
        max_tx=config.max_tx,
        max_rx=config.max_rx,
    )
    antenna = AntennaSpec(tx_polarization="V", rx_polarization="V")
    frequency = FrequencyGrid.from_center_bandwidth(
        config.center_frequency_hz,
        config.bandwidth_hz,
        config.num_subcarriers,
    )
    adapter_result = run_sionna_rt_truth(
        topology=topology,
        antenna=antenna,
        frequency=frequency,
        config=SionnaRTConfig(
            scene_file=config.scene_file,
            seed=config.seed,
            max_depth=config.max_depth,
            specular_reflection=config.specular_reflection,
            num_time_steps=config.num_time_steps,
            sampling_frequency_hz=config.sampling_frequency_hz,
        ),
    )
    elapsed_seconds = time.perf_counter() - start
    observation_bundle = None
    if config.observation_snr_db is not None:
        observation_bundle = run_awgn_ls_observation(
            adapter_result.truth.cfr,
            AWGNObservationConfig(
                snr_db=config.observation_snr_db,
                random_seed=config.observation_seed,
                sample_rate_hz=config.bandwidth_hz,
                fft_size=config.num_subcarriers,
                impairment=config.impairment_config,
            ),
        )
    phase = 4 if observation_bundle is not None else 3 if config.max_depth > 0 else 2

    result = MeasurementSimulationResult(
        metadata=Metadata(
            run_id=output_dir.name,
            random_seed=config.seed,
            config_snapshot=json.dumps(_config_snapshot(config), sort_keys=True),
            measurement_realism_level=f"phase{phase}_rt_truth",
            observation_branch_enabled=observation_bundle is not None,
            software_versions=json.dumps(adapter_result.runtime_versions, sort_keys=True),
        ),
        input_spec=InputSpec(
            label_file=config.label_file.as_posix(),
            scene_file=config.scene_file.as_posix(),
            input_dataset_id="data/scenes/test",
            input_schema="test5_json",
        ),
        topology=topology,
        devices=_build_device_state(config, topology),
        motion=_build_motion_spec(config),
        antenna=antenna,
        scene=SceneSpec(
            scene_name=config.scene_file.stem,
            scene_file=config.scene_file.as_posix(),
            material_policy="sionna_rt_scene_materials",
        ),
        frequency=frequency,
        truth=adapter_result.truth,
        path_samples=adapter_result.path_samples,
        runtime=RuntimeInfo(
            sionna_version=adapter_result.runtime_versions["sionna"],
            sionna_rt_version=adapter_result.runtime_versions["sionna_rt"],
            torch_version=adapter_result.runtime_versions["torch"],
            mitsuba_version=adapter_result.runtime_versions["mitsuba"],
            drjit_version=adapter_result.runtime_versions["drjit"],
            command_line="run-rt-truth",
            elapsed_seconds=elapsed_seconds,
        ),
        path_table=adapter_result.path_table,
        waveform=observation_bundle.waveform if observation_bundle else None,
        observation=observation_bundle.observation if observation_bundle else None,
        impairments=observation_bundle.impairments if observation_bundle else None,
        receiver=observation_bundle.receiver if observation_bundle else None,
        evaluation=observation_bundle.evaluation if observation_bundle else None,
        calibration=(
            CalibrationResult.synthetic_default() if observation_bundle else None
        ),
        diagnostics=(
            DiagnosticsReport.from_evaluation(
                observation_bundle.evaluation, observation_bundle.observation
            )
            if observation_bundle
            else None
        ),
    )

    results_path = write_measurement_result(output_dir / "results.h5", result)
    validate_hdf5_contract(results_path)
    manifest_data = {
        "phase": phase,
        "results_h5": results_path.as_posix(),
        "label_file": config.label_file.as_posix(),
        "scene_file": config.scene_file.as_posix(),
        "config_snapshot": _config_snapshot(config),
        "software_versions": adapter_result.runtime_versions,
        "raw_cfr_shape": adapter_result.raw_cfr_shape,
        "internal_cfr_shape": adapter_result.internal_cfr_shape,
        "path_count": int(adapter_result.path_samples.path_count.sum()),
        "observation_snr_db": config.observation_snr_db,
        "elapsed_seconds": elapsed_seconds,
    }
    if result.diagnostics is not None:
        manifest_data["diagnostics"] = result.diagnostics.to_summary_dict()
    manifest_path = write_manifest(output_dir / "manifest.json", manifest_data)
    (logs_dir / "run.log").write_text(
        "\n".join(
            [
                "phase=2",
                f"results_h5={results_path.as_posix()}",
                f"manifest={manifest_path.as_posix()}",
                f"raw_cfr_shape={adapter_result.raw_cfr_shape}",
                f"internal_cfr_shape={adapter_result.internal_cfr_shape}",
                f"path_count={int(adapter_result.path_samples.path_count.sum())}",
                f"observation_snr_db={config.observation_snr_db}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return results_path


def _build_device_state(config: RTTruthRunConfig, topology: Topology) -> DeviceState:
    import numpy as np

    num_snap = max(config.num_time_steps, 1)
    v_tx = np.array(config.tx_velocity_mps, dtype=np.float32)
    v_rx = np.array(config.rx_velocity_mps, dtype=np.float32)
    tx_v = np.tile(v_tx.reshape(1, 1, 3), (num_snap, topology.num_tx, 1))
    rx_v = np.tile(v_rx.reshape(1, 1, 3), (num_snap, topology.num_rx, 1))
    return DeviceState(
        tx_velocity_mps=tx_v,
        rx_velocity_mps=rx_v,
        tx_orientation_rad=np.zeros_like(tx_v),
        rx_orientation_rad=np.zeros_like(rx_v),
    )


def _build_motion_spec(config: RTTruthRunConfig) -> MotionSpec | None:
    if config.num_time_steps <= 1 and config.sampling_frequency_hz <= 0:
        return None
    return MotionSpec.doppler_synthetic(
        num_time_steps=max(config.num_time_steps, 1),
        sampling_frequency_hz=config.sampling_frequency_hz or 1.0,
    )


def _config_snapshot(config: RTTruthRunConfig) -> dict[str, object]:
    return {
        "label_file": config.label_file.as_posix(),
        "scene_file": config.scene_file.as_posix(),
        "center_frequency_hz": config.center_frequency_hz,
        "bandwidth_hz": config.bandwidth_hz,
        "num_subcarriers": config.num_subcarriers,
        "seed": config.seed,
        "max_tx": config.max_tx,
        "max_rx": config.max_rx,
        "max_depth": config.max_depth,
        "specular_reflection": config.specular_reflection,
        "observation_snr_db": config.observation_snr_db,
        "observation_seed": config.observation_seed,
        "num_time_steps": config.num_time_steps,
        "sampling_frequency_hz": config.sampling_frequency_hz,
        "tx_velocity_mps": list(config.tx_velocity_mps),
        "rx_velocity_mps": list(config.rx_velocity_mps),
    }
