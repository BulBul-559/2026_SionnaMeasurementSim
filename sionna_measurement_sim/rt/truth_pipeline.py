"""Phase 2 RT truth pipeline."""

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
from sionna_measurement_sim.domain.results import (
    DeviceState,
    InputSpec,
    MeasurementSimulationResult,
    Metadata,
    RuntimeInfo,
    SceneSpec,
)
from sionna_measurement_sim.io.hdf5_writer import write_measurement_result
from sionna_measurement_sim.io.label_parser import load_topology_from_label
from sionna_measurement_sim.io.manifest import write_manifest
from sionna_measurement_sim.io.schema_validator import validate_hdf5_contract


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
        ),
    )
    elapsed_seconds = time.perf_counter() - start
    phase = 3 if config.max_depth > 0 else 2

    result = MeasurementSimulationResult(
        metadata=Metadata(
            run_id=output_dir.name,
            random_seed=config.seed,
            config_snapshot=json.dumps(_config_snapshot(config), sort_keys=True),
            measurement_realism_level=f"phase{phase}_rt_truth",
            software_versions=json.dumps(adapter_result.runtime_versions, sort_keys=True),
        ),
        input_spec=InputSpec(
            label_file=config.label_file.as_posix(),
            scene_file=config.scene_file.as_posix(),
            input_dataset_id="data/scenes/test",
            input_schema="test5_json",
        ),
        topology=topology,
        devices=DeviceState.static(snapshots=1, tx=topology.num_tx, rx=topology.num_rx),
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
    )

    results_path = write_measurement_result(output_dir / "results.h5", result)
    validate_hdf5_contract(results_path)
    manifest_path = write_manifest(
        output_dir / "manifest.json",
        {
            "phase": phase,
            "results_h5": results_path.as_posix(),
            "label_file": config.label_file.as_posix(),
            "scene_file": config.scene_file.as_posix(),
            "config_snapshot": _config_snapshot(config),
            "software_versions": adapter_result.runtime_versions,
            "raw_cfr_shape": adapter_result.raw_cfr_shape,
            "internal_cfr_shape": adapter_result.internal_cfr_shape,
            "path_count": int(adapter_result.path_samples.path_count.sum()),
            "elapsed_seconds": elapsed_seconds,
        },
    )
    (logs_dir / "run.log").write_text(
        "\n".join(
            [
                "phase=2",
                f"results_h5={results_path.as_posix()}",
                f"manifest={manifest_path.as_posix()}",
                f"raw_cfr_shape={adapter_result.raw_cfr_shape}",
                f"internal_cfr_shape={adapter_result.internal_cfr_shape}",
                f"path_count={int(adapter_result.path_samples.path_count.sum())}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return results_path


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
    }
