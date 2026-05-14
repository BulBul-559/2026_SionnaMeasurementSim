"""Phases 2-6 RT truth pipeline."""

from __future__ import annotations

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from sionna_measurement_sim.adapters.sionna_rt.rt_solver import (
    SionnaRTConfig,
    run_sionna_rt_truth,
)
from sionna_measurement_sim.domain.antenna import AntennaSpec
from sionna_measurement_sim.domain.array import ArraySpectrumConfig
from sionna_measurement_sim.domain.derived import build_derived_labels
from sionna_measurement_sim.domain.frequency import FrequencyGrid
from sionna_measurement_sim.domain.link import LinkConfig
from sionna_measurement_sim.domain.motion import MotionSpec
from sionna_measurement_sim.domain.observation import (
    CalibrationResult,
    DiagnosticsReport,
)
from sionna_measurement_sim.domain.path import build_nlos_path_truth
from sionna_measurement_sim.domain.results import (
    DeviceState,
    InputSpec,
    MeasurementSimulationResult,
    Metadata,
    RuntimeInfo,
    SceneSpec,
    ShardMetadata,
    ShardSpec,
)
from sionna_measurement_sim.domain.topology import Topology
from sionna_measurement_sim.io.hdf5_writer import write_measurement_result
from sionna_measurement_sim.io.label_parser import count_topology_points, load_topology_from_label
from sionna_measurement_sim.io.manifest import write_manifest
from sionna_measurement_sim.io.schema_validator import validate_hdf5_contract
from sionna_measurement_sim.perf import PerfTracer
from sionna_measurement_sim.phy.impairments import ImpairmentConfig
from sionna_measurement_sim.phy.observation_pipeline import (
    AWGNObservationConfig,
    PHYObservationBundle,
    run_awgn_ls_observation,
)
from sionna_measurement_sim.preflight.system import collect_basic_environment
from sionna_measurement_sim.visualization.config import VisualizationRunConfig
from sionna_measurement_sim.visualization.report import generate_visualization_report


@dataclass(frozen=True)
class RTTruthRunConfig:
    """User-facing config for a minimal RT truth run."""

    label_file: Path
    scene_file: Path
    output_dir: Path
    scene_id: str = ""
    map_id: str = ""
    center_frequency_hz: float = 3.5e9
    bandwidth_hz: float = 20e6
    num_subcarriers: int = 8
    seed: int = 1
    device: str = "cpu"
    max_tx: int = 1
    max_rx: int = 1
    max_depth: int = 1
    los: bool = True
    specular_reflection: bool = True
    diffuse_reflection: bool = False
    refraction: bool = False
    diffraction: bool = False
    synthetic_array: bool = False
    normalize_cfr: bool = False
    normalize_delays: bool = False
    observation_snr_db: float | None = None
    observation_seed: int = 11
    impairment_config: ImpairmentConfig | None = None
    num_time_steps: int = 1
    sampling_frequency_hz: float = 0.0
    tx_velocity_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rx_velocity_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Antenna config
    tx_num_rows: int = 1
    tx_num_cols: int = 1
    rx_num_rows: int = 1
    rx_num_cols: int = 1
    tx_polarization: str = "V"
    rx_polarization: str = "V"
    tx_orientation_mode: str = "fixed"
    tx_orientation_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rx_orientation_mode: str = "fixed"
    rx_orientation_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    tx_spacing_lambda: tuple[float, float] = (0.5, 0.5)
    rx_spacing_lambda: tuple[float, float] = (0.5, 0.5)
    tx_pattern: str = "iso"
    rx_pattern: str = "iso"
    merge_shapes: bool = False
    hdf5_filename: str = "results.h5"
    hdf5_compression: str = "gzip"
    save_full_paths: bool = False
    calibration_enabled: bool = True
    link_config: LinkConfig = LinkConfig()
    spectrum_config: ArraySpectrumConfig = field(default_factory=ArraySpectrumConfig)
    debug_config: Any | None = None
    output_sharding_config: Any | None = None
    shard_spec: ShardSpec | None = None
    phy_standard: str = "custom_ofdm"  # "custom_ofdm" | "nr_pusch"
    # NR PUSCH fields (used when phy_standard == "nr_pusch")
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
    tx_power_dbm: float = 0.0
    num_ofdm_symbols: int = 14
    cp_length: int = 0
    # MIMO / receiver configuration
    mimo_mode: str = "su_mimo"
    channel_backend: str = "apply_ofdm"
    mimo_detector: str = "lmmse"
    channel_estimator: str = "pusch_ls"
    receiver_failure_policy: str = "fail_fast"
    su_mimo_link_batch_size: int = 1
    visualization_config: VisualizationRunConfig = field(default_factory=VisualizationRunConfig)


def run_rt_truth_pipeline(config: RTTruthRunConfig) -> Path:
    """Run RT truth pipeline, optionally as UE/RX HDF5 shards."""

    if _should_run_sharded(config):
        return run_sharded_rt_truth_pipeline(config)
    return _run_rt_truth_pipeline_single(config)


def run_sharded_rt_truth_pipeline(config: RTTruthRunConfig) -> Path:
    """Run a UE/RX-sharded `run-full` and write an aggregate manifest."""

    sharding = config.output_sharding_config
    if sharding is None or not getattr(sharding, "enabled", False):
        return _run_rt_truth_pipeline_single(config)

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    shard_specs = _build_shard_specs(config)
    shard_count = len(shard_specs)
    shard_jobs = [
        _build_shard_run_config(config, spec)
        for spec in shard_specs
    ]

    parallel_workers = min(
        max(int(getattr(sharding, "parallel_workers", 1)), 1),
        shard_count,
    )
    gpu_ids = [int(gpu_id) for gpu_id in getattr(sharding, "gpu_ids", [])]
    shard_results: list[dict[str, object]] = []

    if parallel_workers <= 1:
        for index, shard_config in enumerate(shard_jobs):
            gpu_id = gpu_ids[index % len(gpu_ids)] if gpu_ids else None
            result_path = _run_shard_worker(shard_config, gpu_id)
            shard_results.append(_shard_result_summary(shard_config, result_path))
    else:
        with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
            future_to_config = {}
            for index, shard_config in enumerate(shard_jobs):
                gpu_id = gpu_ids[index % len(gpu_ids)] if gpu_ids else None
                future = executor.submit(_run_shard_worker, shard_config, gpu_id)
                future_to_config[future] = shard_config
            for future in as_completed(future_to_config):
                shard_config = future_to_config[future]
                result_path = future.result()
                shard_results.append(_shard_result_summary(shard_config, result_path))

    shard_results.sort(key=lambda item: int(item["shard_index"]))
    elapsed_seconds = time.perf_counter() - start
    aggregate_manifest = {
        "phase": "sharded_run_full",
        "results_h5": "",
        "results": shard_results,
        "label_file": config.label_file.as_posix(),
        "scene_file": config.scene_file.as_posix(),
        "scene_id": config.scene_id or config.scene_file.stem,
        "map_id": config.map_id,
        "elapsed_seconds": elapsed_seconds,
        "sharding": {
            "enabled": True,
            "axis": _normalize_shard_axis(getattr(sharding, "axis", "rx")),
            "requested_axis": str(getattr(sharding, "axis", "rx")),
            "shard_size": int(getattr(sharding, "shard_size", config.max_rx)),
            "shard_count": shard_count,
            "parallel_workers": parallel_workers,
            "gpu_ids": gpu_ids,
            "filename_pattern": getattr(
                sharding, "filename_pattern", "result_{shard_index:03d}.h5"
            ),
            "visualization_mode": getattr(sharding, "visualization_mode", "first_shard"),
        },
        "config_snapshot": _config_snapshot(config),
        "performance": _aggregate_shard_performance(output_dir, shard_results),
    }
    write_manifest(output_dir / "manifest.json", aggregate_manifest)
    return output_dir


def _run_rt_truth_pipeline_single(config: RTTruthRunConfig) -> Path:
    """Run Phase 2 minimal RT truth and write HDF5, manifest, and log."""

    start = time.perf_counter()
    output_dir = config.output_dir
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    worker_id = (
        f"shard_{config.shard_spec.shard_index:03d}"
        if config.shard_spec is not None
        else "main"
    )
    tracer = PerfTracer(output_dir, config.debug_config, worker_id=worker_id)
    tracer.start()

    with tracer.span("topology_load"):
        topology = load_topology_from_label(
            config.label_file,
            max_tx=config.max_tx,
            max_rx=config.max_rx,
            rx_start=config.shard_spec.rx_start if config.shard_spec else 0,
            rx_count=config.shard_spec.rx_count if config.shard_spec else None,
            rx_indices=config.shard_spec.rx_indices if config.shard_spec else None,
            tx_indices=config.shard_spec.tx_indices if config.shard_spec else None,
        )
    antenna = AntennaSpec(
        tx_num_rows=config.tx_num_rows,
        tx_num_cols=config.tx_num_cols,
        rx_num_rows=config.rx_num_rows,
        rx_num_cols=config.rx_num_cols,
        tx_polarization=config.tx_polarization,
        rx_polarization=config.rx_polarization,
        tx_spacing_lambda=config.tx_spacing_lambda,
        rx_spacing_lambda=config.rx_spacing_lambda,
        tx_pattern=config.tx_pattern,
        rx_pattern=config.rx_pattern,
        tx_orientation_mode=config.tx_orientation_mode,
        tx_orientation_rad=config.tx_orientation_rad,
        rx_orientation_mode=config.rx_orientation_mode,
        rx_orientation_rad=config.rx_orientation_rad,
    )
    frequency = FrequencyGrid.from_center_bandwidth(
        config.center_frequency_hz,
        config.bandwidth_hz,
        config.num_subcarriers,
    )
    with tracer.span("rt_solve"):
        adapter_result = run_sionna_rt_truth(
            topology=topology,
            antenna=antenna,
            frequency=frequency,
            config=SionnaRTConfig(
                scene_file=config.scene_file,
                seed=config.seed,
                max_depth=config.max_depth,
                los=config.los,
                specular_reflection=config.specular_reflection,
                diffuse_reflection=config.diffuse_reflection,
                refraction=config.refraction,
                diffraction=config.diffraction,
                synthetic_array=config.synthetic_array,
                normalize_cfr=config.normalize_cfr,
                normalize_delays=config.normalize_delays,
                num_time_steps=config.num_time_steps,
                sampling_frequency_hz=config.sampling_frequency_hz,
                tx_velocity=config.tx_velocity_mps,
                rx_velocity=config.rx_velocity_mps,
                merge_shapes=config.merge_shapes,
            ),
        )
    elapsed_seconds = time.perf_counter() - start
    environment = collect_basic_environment()
    observation_bundle = None
    nr_pusch_extra: dict = {}
    if config.observation_snr_db is not None:
        if config.phy_standard == "nr_pusch":
            with tracer.span("nr_pusch_observation"):
                observation_bundle, nr_pusch_extra = _run_nr_pusch_obs(config, adapter_result)
        else:
            with tracer.span("custom_ofdm_observation"):
                observation_bundle = _run_custom_ofdm_obs(config, adapter_result)
    phase = 7 if observation_bundle is not None else 3 if config.max_depth > 0 else 2

    scene_id = config.scene_id or config.scene_file.stem
    with tracer.span("derived_nlos"):
        derived = build_derived_labels(
            topology, adapter_result.truth, adapter_result.path_table, adapter_result.cir_truth
        )
        nlos_path_truth = build_nlos_path_truth(adapter_result.path_table)
    with tracer.span("array_outputs"):
        if nr_pusch_extra.get("waveform_extras"):
            cfr_est = (
                observation_bundle.observation.cfr_est
                if observation_bundle and observation_bundle.observation
                else None
            )
            _attach_nr_array_outputs(
                nr_pusch_extra, derived, config, adapter_result.truth.cfr, cfr_est
            )
        elif config.spectrum_config.enabled and "truth_cfr" in config.spectrum_config.sources:
            nr_pusch_extra["array_outputs"] = _build_truth_array_outputs(
                config, derived, adapter_result.truth.cfr
            )
    shard_metadata = _build_shard_metadata(config, topology)
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
        devices=_build_device_state(
            config, topology,
            tx_orientation_rad_scene=adapter_result.tx_orientation_rad,
            rx_orientation_rad_scene=adapter_result.rx_orientation_rad,
        ),
        motion=_build_motion_spec(config),
        antenna=antenna,
        scene=SceneSpec(
            scene_name=config.scene_file.stem,
            scene_file=config.scene_file.as_posix(),
            scene_id=scene_id,
            map_id=config.map_id,
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
            cuda_available=bool(environment["cuda_available"]),
            cuda_device_name=str(environment["cuda_device_name"]),
            command_line="run-rt-truth",
            elapsed_seconds=elapsed_seconds,
        ),
        cir_truth=adapter_result.cir_truth,
        derived=derived,
        path_table=adapter_result.path_table if config.save_full_paths else None,
        nlos_path_truth=nlos_path_truth,
        waveform=observation_bundle.waveform if observation_bundle else None,
        observation=observation_bundle.observation if observation_bundle else None,
        impairments=observation_bundle.impairments if observation_bundle else None,
        receiver=observation_bundle.receiver if observation_bundle else None,
        evaluation=observation_bundle.evaluation if observation_bundle else None,
        calibration=(
            CalibrationResult.synthetic_default()
            if (observation_bundle and config.calibration_enabled)
            else None
        ),
        link=config.link_config,
        shard=shard_metadata,
        waveform_extras=nr_pusch_extra.get("waveform_extras"),
        array_outputs=nr_pusch_extra.get("array_outputs"),
        diagnostics=(
            DiagnosticsReport.from_evaluation(
                observation_bundle.evaluation, observation_bundle.observation
            )
            if observation_bundle
            else None
        ),
    )

    with tracer.span("hdf5_write"):
        results_path = write_measurement_result(
            output_dir / config.hdf5_filename,
            result,
            compression=config.hdf5_compression,
        )
    with tracer.span("schema_validate"):
        validate_hdf5_contract(results_path)
    visualization_summary = None
    if config.visualization_config.enabled:
        with tracer.span("visualization"):
            visualization_summary = generate_visualization_report(
                results_path,
                output_dir / config.visualization_config.output_dir,
                config.visualization_config,
                mode="sample",
            )
    manifest_data = {
        "phase": phase,
        "results_h5": results_path.as_posix(),
        "label_file": config.label_file.as_posix(),
        "scene_file": config.scene_file.as_posix(),
        "scene_id": scene_id,
        "map_id": config.map_id,
        "config_snapshot": _config_snapshot(config),
        "software_versions": adapter_result.runtime_versions,
        "raw_cfr_shape": adapter_result.raw_cfr_shape,
        "internal_cfr_shape": adapter_result.internal_cfr_shape,
        "path_count": int(adapter_result.path_samples.path_count.sum()),
        "observation_snr_db": config.observation_snr_db,
        "elapsed_seconds": elapsed_seconds,
    }
    if nr_pusch_extra.get("batching_stats"):
        manifest_data["nr_pusch_batching"] = nr_pusch_extra["batching_stats"]
    if visualization_summary is not None:
        manifest_data["visualization"] = _manifest_visualization_summary(visualization_summary)
    if result.diagnostics is not None:
        manifest_data["diagnostics"] = result.diagnostics.to_summary_dict()
    if shard_metadata is not None:
        manifest_data["shard"] = _manifest_shard_summary(shard_metadata)
    manifest_filename = (
        f"manifest_{shard_metadata.shard_index:03d}.json"
        if shard_metadata is not None
        else "manifest.json"
    )
    with tracer.span("manifest_write"):
        manifest_path = write_manifest(output_dir / manifest_filename, manifest_data)
    perf_summary = tracer.finish({"results_h5": results_path.as_posix()})
    if perf_summary:
        manifest_data["performance"] = {
            "enabled": True,
            "summary_path": perf_summary["logs"]["events"].replace(
                "perf_events", "perf_summary"
            ).replace(".jsonl", ".json"),
            "stage_totals_s": perf_summary.get("stage_totals_s", {}),
        }
        manifest_path = write_manifest(output_dir / manifest_filename, manifest_data)

    run_log_name = (
        f"run_{shard_metadata.shard_index:03d}.log"
        if shard_metadata is not None
        else "run.log"
    )
    (logs_dir / run_log_name).write_text(
        "\n".join(
            [
                f"phase={phase}",
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


def _should_run_sharded(config: RTTruthRunConfig) -> bool:
    sharding = config.output_sharding_config
    return (
        sharding is not None
        and bool(getattr(sharding, "enabled", False))
        and config.shard_spec is None
    )


def _build_shard_specs(config: RTTruthRunConfig) -> list[ShardSpec]:
    sharding = config.output_sharding_config
    if sharding is None:
        return []
    axis = _normalize_shard_axis(getattr(sharding, "axis", "rx"))
    shard_size = int(getattr(sharding, "shard_size", config.max_rx))
    if shard_size < 1:
        msg = "output.sharding.shard_size must be positive"
        raise ValueError(msg)
    available_tx, available_rx = count_topology_points(config.label_file)
    effective_rx_count = min(config.max_rx, available_rx)
    if min(config.max_tx, available_tx) < 1 or effective_rx_count < 1:
        msg = "Label file must contain at least one selected BS and UE"
        raise ValueError(msg)
    shard_count = (effective_rx_count + shard_size - 1) // shard_size
    return [
        ShardSpec(
            shard_index=shard_index,
            shard_count=shard_count,
            axis=axis,
            rx_start=shard_index * shard_size,
            rx_count=min(shard_size, effective_rx_count - shard_index * shard_size),
        )
        for shard_index in range(shard_count)
    ]


def _normalize_shard_axis(axis: object) -> str:
    axis_str = str(axis)
    if axis_str in ("rx", "ue"):
        return "rx"
    msg = f"Only rx/ue sharding is supported, got {axis_str!r}"
    raise ValueError(msg)


def _build_shard_run_config(config: RTTruthRunConfig, spec: ShardSpec) -> RTTruthRunConfig:
    sharding = config.output_sharding_config
    filename_pattern = (
        getattr(sharding, "filename_pattern", "result_{shard_index:03d}.h5")
        if sharding is not None
        else "result_{shard_index:03d}.h5"
    )
    hdf5_filename = filename_pattern.format(
        shard_index=spec.shard_index,
        shard_count=spec.shard_count,
    )
    visualization_config = _shard_visualization_config(config, spec)
    return replace(
        config,
        hdf5_filename=hdf5_filename,
        shard_spec=spec,
        visualization_config=visualization_config,
    )


def _shard_visualization_config(
    config: RTTruthRunConfig,
    spec: ShardSpec,
) -> VisualizationRunConfig:
    sharding = config.output_sharding_config
    mode = (
        getattr(sharding, "visualization_mode", "first_shard")
        if sharding is not None
        else "first_shard"
    )
    vis = config.visualization_config
    if mode == "none":
        return replace(vis, enabled=False)
    if mode == "first_shard" and spec.shard_index != 0:
        return replace(vis, enabled=False)
    if mode == "all_shards":
        return replace(vis, output_dir=f"{vis.output_dir}_{spec.shard_index:03d}")
    return vis


def _run_shard_worker(config: RTTruthRunConfig, gpu_id: int | None) -> Path:
    if gpu_id is not None and str(config.device).startswith("cuda"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return _run_rt_truth_pipeline_single(config)


def _shard_result_summary(config: RTTruthRunConfig, result_path: Path) -> dict[str, object]:
    spec = config.shard_spec
    if spec is None:
        msg = "shard result summary requires shard_spec"
        raise ValueError(msg)
    manifest_path = config.output_dir / f"manifest_{spec.shard_index:03d}.json"
    shard_manifest = _read_json_if_exists(manifest_path)
    shard_summary = shard_manifest.get("shard", {})
    rx_indices = [
        int(index)
        for index in shard_summary.get(
            "global_rx_indices",
            list(range(spec.rx_start, spec.rx_start + int(spec.rx_count or 0))),
        )
    ]
    tx_indices = [
        int(index)
        for index in shard_summary.get("global_tx_indices", list(range(config.max_tx)))
    ]
    return {
        "shard_index": spec.shard_index,
        "shard_count": spec.shard_count,
        "axis": str(shard_summary.get("axis", spec.axis)),
        "result_h5": Path(result_path).as_posix(),
        "manifest": manifest_path.as_posix(),
        "global_rx_start": int(shard_summary.get("global_rx_start", spec.rx_start)),
        "global_rx_count": len(rx_indices),
        "global_rx_indices": rx_indices,
        "global_tx_indices": tx_indices,
        "nr_pusch_batching": shard_manifest.get("nr_pusch_batching", {}),
    }


def _read_json_if_exists(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _aggregate_shard_performance(
    output_dir: Path,
    shard_results: list[dict[str, object]],
) -> dict[str, object]:
    summaries: list[dict[str, object]] = []
    stage_totals: dict[str, float] = {}
    total_durations: list[float] = []
    for shard in shard_results:
        shard_index = int(shard["shard_index"])
        summary_path = output_dir / "logs" / f"perf_summary_shard_{shard_index:03d}.json"
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        summaries.append(
            {
                "shard_index": shard_index,
                "summary_path": summary_path.as_posix(),
                "total_duration_s": summary.get("total_duration_s"),
                "stage_totals_s": summary.get("stage_totals_s", {}),
            }
        )
        total_duration = summary.get("total_duration_s")
        if isinstance(total_duration, (int, float)):
            total_durations.append(float(total_duration))
        for name, value in dict(summary.get("stage_totals_s", {})).items():
            stage_totals[str(name)] = stage_totals.get(str(name), 0.0) + float(value)
    return {
        "enabled": bool(summaries),
        "shard_summaries": summaries,
        "stage_totals_s": stage_totals,
        "max_shard_duration_s": max(total_durations) if total_durations else None,
        "sum_shard_duration_s": sum(total_durations) if total_durations else None,
    }


def _build_shard_metadata(
    config: RTTruthRunConfig,
    topology: Topology,
) -> ShardMetadata | None:
    spec = config.shard_spec
    if spec is None:
        return None
    rx_indices = (
        np.asarray(spec.rx_indices, dtype=np.int64)
        if spec.rx_indices is not None
        else np.arange(spec.rx_start, spec.rx_start + topology.num_rx, dtype=np.int64)
    )
    tx_indices = (
        np.asarray(spec.tx_indices, dtype=np.int64)
        if spec.tx_indices is not None
        else np.arange(topology.num_tx, dtype=np.int64)
    )
    return ShardMetadata.from_spec(
        spec,
        global_rx_indices=rx_indices,
        global_tx_indices=tx_indices,
    )


def _manifest_shard_summary(shard: ShardMetadata) -> dict[str, object]:
    return {
        "shard_index": int(shard.shard_index),
        "shard_count": int(shard.shard_count),
        "axis": shard.axis,
        "global_rx_start": int(shard.global_rx_start),
        "global_rx_indices": shard.global_rx_indices.astype(int).tolist(),
        "global_tx_indices": shard.global_tx_indices.astype(int).tolist(),
    }


def _build_device_state(
    config: RTTruthRunConfig,
    topology: Topology,
    tx_orientation_rad_scene: np.ndarray | None = None,
    rx_orientation_rad_scene: np.ndarray | None = None,
) -> DeviceState:
    num_snap = max(config.num_time_steps, 1)
    v_tx = np.array(config.tx_velocity_mps, dtype=np.float32)
    v_rx = np.array(config.rx_velocity_mps, dtype=np.float32)
    tx_v = np.tile(v_tx.reshape(1, 1, 3), (num_snap, topology.num_tx, 1))
    rx_v = np.tile(v_rx.reshape(1, 1, 3), (num_snap, topology.num_rx, 1))

    if tx_orientation_rad_scene is not None:
        tx_orient = tx_orientation_rad_scene.reshape(1, topology.num_tx, 3)
        tx_o = np.tile(tx_orient, (num_snap, 1, 1)).astype(np.float32, copy=False)
    else:
        tx_orient = np.array(config.tx_orientation_rad, dtype=np.float32).reshape(1, 1, 3)
        tx_o = np.tile(tx_orient, (num_snap, topology.num_tx, 1))

    if rx_orientation_rad_scene is not None:
        rx_orient = rx_orientation_rad_scene.reshape(1, topology.num_rx, 3)
        rx_o = np.tile(rx_orient, (num_snap, 1, 1)).astype(np.float32, copy=False)
    else:
        rx_orient = np.array(config.rx_orientation_rad, dtype=np.float32).reshape(1, 1, 3)
        rx_o = np.tile(rx_orient, (num_snap, topology.num_rx, 1))

    return DeviceState(
        tx_velocity_mps=tx_v,
        rx_velocity_mps=rx_v,
        tx_orientation_rad=tx_o,
        rx_orientation_rad=rx_o,
    )


def _build_motion_spec(config: RTTruthRunConfig) -> MotionSpec | None:
    if config.num_time_steps <= 1 and config.sampling_frequency_hz <= 0:
        return None
    return MotionSpec.doppler_synthetic(
        num_time_steps=max(config.num_time_steps, 1),
        sampling_frequency_hz=config.sampling_frequency_hz or 1.0,
    )


def _run_custom_ofdm_obs(config, adapter_result):
    return run_awgn_ls_observation(
        adapter_result.truth.cfr,
        AWGNObservationConfig(
            snr_db=config.observation_snr_db,
            random_seed=config.observation_seed,
            sample_rate_hz=config.bandwidth_hz,
            fft_size=config.num_subcarriers,
            impairment=config.impairment_config,
        ),
        has_signal=adapter_result.truth.has_geometric_signal,
        cfr_snapshots=adapter_result.truth.cfr_snapshots,
    )


def _run_nr_pusch_obs(config, adapter_result):
    from sionna_measurement_sim.phy.nr_pusch_observation import (
        run_nr_pusch_observation,
    )

    nr_result = run_nr_pusch_observation(
        cir_coefficients=adapter_result.cir_truth.coefficients,
        cir_delays=adapter_result.cir_truth.delays_s,
        link_config=config.link_config,
        phy_config=config,  # RTTruthRunConfig acts as phy_config
        carrier_config=config,  # RTTruthRunConfig acts as carrier_config
    )
    # Map NR result into existing bundle format
    bundle = PHYObservationBundle(
        waveform=nr_result["nr_waveform_spec"],
        observation=nr_result["observation"],
        impairments=nr_result["impairments"],
        receiver=nr_result["receiver_spec"],
        evaluation=nr_result["evaluation"],
    )
    return bundle, {
        "pusch_config": nr_result["pusch_config"],
        "waveform_extras": {
            "num_prb": config.num_prb,
            "subcarrier_spacing_khz": config.subcarrier_spacing_khz,
            "subcarrier_spacing_hz": config.subcarrier_spacing_khz * 1000.0,
            "slot_number": 0,
            "cyclic_prefix": "normal",
            "target_coderate": 0.54,
            "modulation": "16QAM",
            "num_layers": config.num_layers,
            "num_antenna_ports": config.num_antenna_ports,
            "mcs_index": config.mcs_index,
            "mcs_table": config.mcs_table,
            "dmrs_config_type": config.pusch_dmrs_config_type,
            "dmrs_length": config.pusch_dmrs_length,
            "dmrs_additional_position": config.pusch_dmrs_additional_position,
            "num_cdm_groups_without_data": config.pusch_num_cdm_groups_without_data,
            **nr_result["waveform_grids"],
        },
        "array_outputs": nr_result["array_outputs"],
        "batching_stats": nr_result.get("batching_stats", {}),
    }


def _attach_nr_array_outputs(
    nr_pusch_extra: dict,
    derived,
    config,
    truth_cfr: np.ndarray,
    cfr_est: np.ndarray | None,
) -> None:
    waveform_extras = nr_pusch_extra.get("waveform_extras") or {}
    rx_grid = waveform_extras.get("rx_grid")
    if rx_grid is None:
        return
    from sionna_measurement_sim.phy.nr_pusch_observation import (
        build_array_outputs_from_waveform,
    )
    from sionna_measurement_sim.phy.spatial_spectrum import (
        project_cfr_to_ul_receiver_samples,
    )

    rx_grid = np.asarray(rx_grid)
    num_snap = rx_grid.shape[0]
    # NR PUSCH waveform grids use UL convention:
    # ul_tx == DL rx (UE), ul_rx == DL tx (BS).
    aoa_2d = np.stack(
        (
            derived.first_path_aoa_zenith_rad.T,
            derived.first_path_aoa_azimuth_rad.T,
        ),
        axis=-1,
    ).astype(np.float32, copy=False)
    aoa = np.broadcast_to(aoa_2d[np.newaxis, ...], (num_snap, *aoa_2d.shape))
    nr_pusch_extra["array_outputs"] = build_array_outputs_from_waveform(
        rx_grid,
        aoa_label_rad=aoa,
        spectrum_config=config.spectrum_config,
        rx_num_rows=config.tx_num_rows,
        rx_num_cols=config.tx_num_cols,
        rx_spacing_lambda=config.tx_spacing_lambda,
        truth_spectrum_samples=project_cfr_to_ul_receiver_samples(truth_cfr),
        cfr_est_spectrum_samples=(
            project_cfr_to_ul_receiver_samples(cfr_est) if cfr_est is not None else None
        ),
    )


def _build_truth_array_outputs(config, derived, truth_cfr: np.ndarray) -> dict:
    from sionna_measurement_sim.phy.spatial_spectrum import (
        build_angle_grid_rad,
        build_aoa_heatmap_label,
        build_bartlett_spectrum,
        project_cfr_to_ul_receiver_samples,
    )

    samples = project_cfr_to_ul_receiver_samples(truth_cfr)
    link_shape = samples.shape[:3]
    aoa_2d = np.stack(
        (
            derived.first_path_aoa_zenith_rad.T,
            derived.first_path_aoa_azimuth_rad.T,
        ),
        axis=-1,
    ).astype(np.float32, copy=False)
    aoa = np.broadcast_to(aoa_2d[np.newaxis, ...], (link_shape[0], *aoa_2d.shape))
    angle_grid = build_angle_grid_rad(config.spectrum_config)
    labels, heatmap = build_aoa_heatmap_label(aoa, angle_grid, link_shape)
    outputs: dict[str, object] = {
        "aoa_label_rad": labels,
        "aoa_heatmap_label": heatmap,
        "spatial_spectrum_label": heatmap,
        "angle_grid_rad": angle_grid,
        "spectrum_policy": config.spectrum_config.policy,
    }
    outputs["spatial_spectrum_truth"] = build_bartlett_spectrum(
        samples,
        rx_num_rows=config.tx_num_rows,
        rx_num_cols=config.tx_num_cols,
        rx_spacing_lambda=config.tx_spacing_lambda,
        config=config.spectrum_config,
    )
    return outputs

def _config_snapshot(config: RTTruthRunConfig) -> dict[str, object]:
    return {
        "label_file": config.label_file.as_posix(),
        "scene_file": config.scene_file.as_posix(),
        "scene_id": config.scene_id or config.scene_file.stem,
        "map_id": config.map_id,
        "center_frequency_hz": config.center_frequency_hz,
        "bandwidth_hz": config.bandwidth_hz,
        "num_subcarriers": config.num_subcarriers,
        "device": config.device,
        "max_depth": config.max_depth,
        "los": config.los,
        "specular_reflection": config.specular_reflection,
        "diffuse_reflection": config.diffuse_reflection,
        "refraction": config.refraction,
        "diffraction": config.diffraction,
        "synthetic_array": config.synthetic_array,
        "normalize_cfr": config.normalize_cfr,
        "normalize_delays": config.normalize_delays,
        "merge_shapes": config.merge_shapes,
        "tx_num_rows": config.tx_num_rows,
        "tx_num_cols": config.tx_num_cols,
        "rx_num_rows": config.rx_num_rows,
        "rx_num_cols": config.rx_num_cols,
        "tx_spacing_lambda": list(config.tx_spacing_lambda),
        "rx_spacing_lambda": list(config.rx_spacing_lambda),
        "visualization": {
            "enabled": config.visualization_config.enabled,
            "output_dir": config.visualization_config.output_dir,
            "sample_policy": config.visualization_config.sample_policy,
            "random_seed": config.visualization_config.random_seed,
            "max_bs": config.visualization_config.max_bs,
            "sample_ue_count": config.visualization_config.sample_ue_count,
            "max_ue": config.visualization_config.max_ue,
            "dpi": config.visualization_config.dpi,
            "format": config.visualization_config.format,
            "plots": list(config.visualization_config.plots),
        },
        "spectrum_config": {
            "enabled": config.spectrum_config.enabled,
            "sources": list(config.spectrum_config.sources),
            "method": config.spectrum_config.method,
            "zenith_bins": config.spectrum_config.zenith_bins,
            "azimuth_bins": config.spectrum_config.azimuth_bins,
            "zenith_min_rad": config.spectrum_config.zenith_min_rad,
            "zenith_max_rad": config.spectrum_config.zenith_max_rad,
            "azimuth_min_rad": config.spectrum_config.azimuth_min_rad,
            "azimuth_max_rad": config.spectrum_config.azimuth_max_rad,
            "normalize": config.spectrum_config.normalize,
            "aggregate_subcarriers": config.spectrum_config.aggregate_subcarriers,
            "aggregate_symbols": config.spectrum_config.aggregate_symbols,
            "link_chunk_size": config.spectrum_config.link_chunk_size,
        },
        "tx_pattern": config.tx_pattern,
        "rx_pattern": config.rx_pattern,
        "tx_polarization": config.tx_polarization,
        "rx_polarization": config.rx_polarization,
        "tx_orientation_mode": config.tx_orientation_mode,
        "tx_orientation_rad": list(config.tx_orientation_rad),
        "rx_orientation_mode": config.rx_orientation_mode,
        "rx_orientation_rad": list(config.rx_orientation_rad),
        "num_time_steps": config.num_time_steps,
        "sampling_frequency_hz": config.sampling_frequency_hz,
        "tx_velocity_mps": list(config.tx_velocity_mps),
        "rx_velocity_mps": list(config.rx_velocity_mps),
        "seed": config.seed,
        "max_tx": config.max_tx,
        "max_rx": config.max_rx,
        "ebno_db": config.ebno_db,
        "observation_snr_db": config.observation_snr_db,
        "observation_seed": config.observation_seed,
        "link_config": {
            "duplex_mode": config.link_config.duplex_mode,
            "phy_link_direction": config.link_config.phy_link_direction,
            "rt_trace_direction": config.link_config.rt_trace_direction,
            "reciprocity_mode": config.link_config.reciprocity_mode,
            "reciprocity_applied": config.link_config.reciprocity_applied,
        },
        "mimo_mode": config.mimo_mode,
        "channel_backend": config.channel_backend,
        "mimo_detector": config.mimo_detector,
        "channel_estimator": config.channel_estimator,
        "receiver_failure_policy": config.receiver_failure_policy,
        "su_mimo_link_batch_size": config.su_mimo_link_batch_size,
        "hdf5_compression": config.hdf5_compression,
        "debug": {
            "enabled": bool(getattr(config.debug_config, "enabled", False)),
            "hardware_interval_s": float(
                getattr(config.debug_config, "hardware_interval_s", 1.0)
            ),
            "link_log_interval": int(
                getattr(config.debug_config, "link_log_interval", 250)
            ),
            "torch_synchronize": bool(
                getattr(config.debug_config, "torch_synchronize", True)
            ),
            "write_hardware_samples": bool(
                getattr(config.debug_config, "write_hardware_samples", True)
            ),
        },
        "sharding": {
            "enabled": bool(getattr(config.output_sharding_config, "enabled", False)),
            "axis": str(getattr(config.output_sharding_config, "axis", "rx")),
            "shard_size": int(
                getattr(config.output_sharding_config, "shard_size", config.max_rx)
            ),
            "filename_pattern": str(
                getattr(
                    config.output_sharding_config,
                    "filename_pattern",
                    "result_{shard_index:03d}.h5",
                )
            ),
            "parallel_workers": int(
                getattr(config.output_sharding_config, "parallel_workers", 1)
            ),
            "gpu_ids": [
                int(gpu_id)
                for gpu_id in getattr(config.output_sharding_config, "gpu_ids", [])
            ],
            "visualization_mode": str(
                getattr(config.output_sharding_config, "visualization_mode", "first_shard")
            ),
        },
    }


def _manifest_visualization_summary(summary: dict[str, object]) -> dict[str, object]:
    return {
        "enabled": True,
        "output_dir": summary["output_dir"],
        "index_path": summary["index_path"],
        "selected_bs_indices": summary["selected_bs_indices"],
        "selected_ue_indices": summary["selected_ue_indices"],
        "generated_files": summary["generated_files"],
        "skipped_plots": summary["skipped_plots"],
    }
