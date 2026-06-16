"""Phases 2-6 RT truth pipeline."""

from __future__ import annotations

import gc
import json
import multiprocessing as mp
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
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
from sionna_measurement_sim.domain.output_plan import RTOutputPlan, build_rt_output_plan
from sionna_measurement_sim.domain.path import build_nlos_path_truth
from sionna_measurement_sim.domain.results import (
    DeviceState,
    InputSpec,
    IQLinkLibraryResult,
    MeasurementSimulationResult,
    Metadata,
    RTCompactLinkLabels,
    RTLabelsOnlyResult,
    RuntimeInfo,
    SceneSpec,
    ShardMetadata,
    ShardSpec,
)
from sionna_measurement_sim.domain.topology import (
    LinkRoleMapping,
    RoleTopology,
    Topology,
    resolve_link_roles,
    resolve_role_pair,
    resolve_role_topology,
    resolved_global_indices,
)
from sionna_measurement_sim.io.hdf5_writer import (
    write_iq_link_library_result,
    write_measurement_result,
    write_rt_labels_result,
)
from sionna_measurement_sim.io.label_parser import (
    STANDARD_LABEL_SCHEMA_VERSION,
    count_topology_points,
    load_role_topology_from_label,
)
from sionna_measurement_sim.io.manifest import write_manifest
from sionna_measurement_sim.io.schema_validator import validate_hdf5_contract
from sionna_measurement_sim.perf import PerfTracer
from sionna_measurement_sim.phy.impairments import ImpairmentConfig
from sionna_measurement_sim.phy.iq_observation import build_iq_observation
from sionna_measurement_sim.phy.modules import PHYContext, get_phy_module
from sionna_measurement_sim.preflight.system import collect_basic_environment
from sionna_measurement_sim.ranging.config import RangingConfig
from sionna_measurement_sim.ranging.runner import run_ranging_observation
from sionna_measurement_sim.visualization.config import VisualizationRunConfig
from sionna_measurement_sim.visualization.radio_map import (
    RadioMapRenderConfig,
    generate_radio_map_heatmaps,
)
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
    max_bs: int = 1
    max_ue: int = 1
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
    bs_velocity_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ue_velocity_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Role-view antenna config
    bs_num_rows: int = 1
    bs_num_cols: int = 1
    ue_num_rows: int = 1
    ue_num_cols: int = 1
    bs_polarization: str = "V"
    ue_polarization: str = "V"
    bs_orientation_mode: str = "fixed"
    bs_orientation_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ue_orientation_mode: str = "fixed"
    ue_orientation_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bs_spacing_lambda: tuple[float, float] = (0.5, 0.5)
    ue_spacing_lambda: tuple[float, float] = (0.5, 0.5)
    bs_pattern: str = "iso"
    ue_pattern: str = "iso"
    merge_shapes: bool = False
    hdf5_filename: str = "results.h5"
    hdf5_compression: str = "gzip"
    hdf5_gzip_level: int = 4
    output_profile: str = "full"
    output_products: tuple[str, ...] | None = None
    output_plan: RTOutputPlan | None = None
    save_full_paths: bool = False
    calibration_enabled: bool = True
    link_config: LinkConfig = LinkConfig()
    spectrum_config: ArraySpectrumConfig = field(default_factory=ArraySpectrumConfig)
    debug_config: Any | None = None
    output_sharding_config: Any | None = None
    shard_spec: ShardSpec | None = None
    phy_standard: str = "custom_ofdm"  # "custom_ofdm" | "nr_pusch" | "nr_srs"
    # NR-family fields (used by nr_pusch and nr_srs where applicable)
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
    power_config: Any | None = None
    iq_config: Any | None = None
    srs_config: Any | None = None
    noncooperative_config: Any | None = None
    ranging_config: RangingConfig = field(default_factory=RangingConfig)
    visualization_config: VisualizationRunConfig = field(default_factory=VisualizationRunConfig)


def run_rt_truth_pipeline(config: RTTruthRunConfig) -> Path:
    """Run RT truth pipeline, optionally as UE HDF5 shards."""

    config = _normalize_output_profile_config(config)
    if config.ranging_config.enabled and config.observation_snr_db is None:
        msg = "ranging.enabled=true requires PHY observation with /observation/cfr_est"
        raise ValueError(msg)
    if _should_run_sharded(config):
        return run_sharded_rt_truth_pipeline(config)
    return _run_rt_truth_pipeline_single(config)


def _normalize_output_profile_config(config: RTTruthRunConfig) -> RTTruthRunConfig:
    plan = config.output_plan or build_rt_output_plan(
        config.output_profile,
        products=config.output_products,
        array_sources=config.spectrum_config.sources,
    )
    updates: dict[str, object] = {
        "output_profile": plan.profile,
        "output_plan": plan,
    }
    if plan.profile == "rt_labels_only":
        updates.update(
            {
                "observation_snr_db": None,
                "calibration_enabled": False,
                "spectrum_config": replace(config.spectrum_config, enabled=False),
                "ranging_config": replace(config.ranging_config, enabled=False),
                "iq_config": None,
                "noncooperative_config": None,
                "visualization_config": replace(config.visualization_config, enabled=False),
                "save_full_paths": False,
            }
        )
    if plan.is_product_aware_full:
        if not plan.requires_phy_observation:
            updates.update(
                {
                    "observation_snr_db": None,
                    "iq_config": None,
                    "noncooperative_config": None,
                }
            )
        elif config.observation_snr_db is None:
            msg = "Selected output.products require PHY observation; enable phy.enabled"
            raise ValueError(msg)
        if plan.write_iq:
            if config.phy_standard == "custom_ofdm":
                msg = (
                    "output.products includes 'iq', which requires a PHY module that "
                    "exports waveform grids; use phy.standard='nr_srs' or 'nr_pusch'"
                )
                raise ValueError(msg)
            updates["iq_config"] = _link_iq_product_config(
                config.iq_config,
                cp_length=config.cp_length,
            )
        if not plan.write_array_outputs:
            updates["spectrum_config"] = replace(config.spectrum_config, enabled=False)
        else:
            updates["spectrum_config"] = replace(config.spectrum_config, enabled=True)
        if not plan.write_ranging:
            updates["ranging_config"] = replace(config.ranging_config, enabled=False)
        else:
            updates["ranging_config"] = replace(config.ranging_config, enabled=True)
        if not plan.write_iq:
            updates["iq_config"] = None
            updates["noncooperative_config"] = None
        if plan.write_multiuser:
            if config.phy_standard != "nr_srs":
                msg = (
                    "output.products includes 'multiuser', which currently requires "
                    "phy.standard='nr_srs'"
                )
                raise ValueError(msg)
            updates["srs_config"] = _srs_multiuser_product_config(config.srs_config)
        if not plan.write_calibration:
            updates["calibration_enabled"] = False
        if plan.write_visualization:
            updates["visualization_config"] = replace(
                config.visualization_config,
                enabled=True,
            )
        else:
            updates["visualization_config"] = replace(
                config.visualization_config,
                enabled=False,
            )
        if plan.write_path_full:
            updates["save_full_paths"] = True
        else:
            updates["save_full_paths"] = False
    if plan.profile == "iq_link_library":
        if config.phy_standard != "nr_srs":
            msg = "output.profile='iq_link_library' currently requires phy.standard='nr_srs'"
            raise ValueError(msg)
        updates.update(
            {
                "calibration_enabled": False,
                "spectrum_config": replace(config.spectrum_config, enabled=False),
                "ranging_config": replace(config.ranging_config, enabled=False),
                "iq_config": _clean_iq_link_library_config(
                    config.iq_config,
                    cp_length=config.cp_length,
                ),
                "noncooperative_config": None,
                "visualization_config": replace(config.visualization_config, enabled=False),
                "save_full_paths": False,
            }
        )
    return replace(config, **updates)


def _config_values(config: Any | None) -> dict[str, Any]:
    if config is None:
        return {}
    if hasattr(config, "model_dump"):
        return dict(config.model_dump())
    if hasattr(config, "__dict__"):
        return {
            name: value
            for name, value in vars(config).items()
            if not name.startswith("_")
        }
    return {
        name: getattr(config, name)
        for name in dir(config)
        if not name.startswith("_") and not callable(getattr(config, name))
    }


def _copy_config_with_updates(
    config: Any | None,
    updates: dict[str, Any],
    *,
    drop_fields: tuple[str, ...] = (),
) -> Any:
    if config is None:
        return SimpleNamespace(**updates)
    if hasattr(config, "model_copy"):
        return config.model_copy(update=updates)
    values = _config_values(config)
    for field_name in drop_fields:
        values.pop(field_name, None)
    values.update(updates)
    return SimpleNamespace(**values)


def _link_iq_product_config(iq_config: Any | None, *, cp_length: int) -> Any:
    clean_output = getattr(iq_config, "clean_output", None)
    save_frequency_observed = bool(
        getattr(iq_config, "save_frequency_observed", False)
    )
    save_time_observed = bool(getattr(iq_config, "save_time_observed", False))
    if clean_output is None and not (save_frequency_observed or save_time_observed):
        clean_output = "time"
    if clean_output is not None and clean_output not in ("time", "frequency", "both"):
        raise ValueError("phy.iq.clean_output must be time/frequency/both")
    resolved_cp_length = getattr(iq_config, "cp_length", None)
    if resolved_cp_length is None:
        resolved_cp_length = cp_length
    return _copy_config_with_updates(
        iq_config,
        {
            "enabled": True,
            "clean_output": clean_output,
            "save_frequency_observed": save_frequency_observed,
            "save_time_observed": save_time_observed,
            "cp_length": resolved_cp_length,
        },
        drop_fields=("save_frequency_clean", "save_time_clean"),
    )


def _clean_iq_link_library_config(iq_config: Any | None, *, cp_length: int) -> Any:
    clean_output = getattr(iq_config, "clean_output", None)
    if clean_output is None:
        clean_output = "time"
    if clean_output not in ("time", "frequency", "both"):
        raise ValueError("phy.iq.clean_output must be time/frequency/both")
    resolved_cp_length = getattr(iq_config, "cp_length", None)
    if resolved_cp_length is None:
        resolved_cp_length = cp_length
    updates = {
        "enabled": True,
        "clean_output": clean_output,
        "save_frequency_observed": False,
        "save_time_observed": False,
        "cp_length": resolved_cp_length,
    }
    return _copy_config_with_updates(
        iq_config,
        updates,
        drop_fields=("save_frequency_clean", "save_time_clean"),
    )


def _srs_multiuser_product_config(srs_config: Any | None) -> Any:
    multi_cfg = getattr(srs_config, "multiuser", None)
    multi_values = {
        "active_ue_count": int(getattr(multi_cfg, "active_ue_count", 2)),
        "resource_strategy": str(getattr(multi_cfg, "resource_strategy", "comb_offset")),
        "frame_policy": str(getattr(multi_cfg, "frame_policy", "sequential")),
        "enabled": True,
    }
    multi_cfg = _copy_config_with_updates(multi_cfg, multi_values)
    updates: dict[str, Any] = {"multiuser": multi_cfg}
    defaults = {
        "slot_length_symbols": 14,
        "start_symbol": 12,
        "num_srs_symbols": 2,
        "comb_size": 2,
        "comb_offset": 0,
        "bwp_start_prb": 0,
        "bwp_num_prb": None,
        "trigger_mode": "aperiodic",
        "periodicity_slots": 1,
        "slot_offset": 0,
        "slot_number": 0,
        "sequence_type": "zc_like",
        "sequence_id": 0,
        "group_hopping": "disabled",
        "sequence_hopping": "disabled",
        "cyclic_shift_multiplexing": "cyclic_shift",
        "cyclic_shift_indices": None,
    }
    values = _config_values(srs_config)
    for name, value in defaults.items():
        if name not in values:
            updates[name] = value
    return _copy_config_with_updates(srs_config, updates)


def run_sharded_rt_truth_pipeline(config: RTTruthRunConfig) -> Path:
    """Run a UE-sharded `run-full` and write an aggregate manifest."""

    sharding = config.output_sharding_config
    if sharding is None or not getattr(sharding, "enabled", False):
        return _run_rt_truth_pipeline_single(config)

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = _shard_manifest_dir(config)
    results_dir = _shard_results_dir(config)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    shard_specs = _build_shard_specs(config)
    shard_count = len(shard_specs)

    parallel_workers = min(
        max(int(getattr(sharding, "parallel_workers", 1)), 1),
        shard_count,
    )
    gpu_ids = [int(gpu_id) for gpu_id in getattr(sharding, "gpu_ids", [])]
    shard_results: list[dict[str, object]] = []
    shard_attempts: list[dict[str, object]] = []

    if parallel_workers <= 1:
        for index, spec in enumerate(shard_specs):
            gpu_id = gpu_ids[index % len(gpu_ids)] if gpu_ids else None
            outcome = _run_shard_spec_with_fallback(config, spec, gpu_id)
            shard_results.extend(outcome["results"])
            shard_attempts.extend(outcome["attempts"])
    else:
        with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
            future_to_spec = {}
            for index, spec in enumerate(shard_specs):
                gpu_id = gpu_ids[index % len(gpu_ids)] if gpu_ids else None
                future = executor.submit(_run_shard_spec_with_fallback, config, spec, gpu_id)
                future_to_spec[future] = spec
            for future in as_completed(future_to_spec):
                outcome = future.result()
                shard_results.extend(outcome["results"])
                shard_attempts.extend(outcome["attempts"])

    shard_results.sort(
        key=lambda item: (
            min([int(i) for i in item.get("global_ue_indices", [])] or [0]),
            str(item.get("shard_id", "")),
        )
    )
    shard_attempts.sort(key=lambda item: str(item.get("shard_id", "")))
    elapsed_seconds = time.perf_counter() - start
    config_snapshot_path = write_manifest(
        manifest_dir / "config_snapshot.json",
        _config_snapshot(config),
    )
    attempts_path = _write_shard_attempts(manifest_dir, shard_attempts)
    aggregate_manifest = {
        "phase": "sharded_run_full",
        "results_h5": "",
        "results": shard_results,
        "results_dir": results_dir.as_posix(),
        "manifest_dir": manifest_dir.as_posix(),
        "config_snapshot_path": config_snapshot_path.as_posix(),
        "shard_attempts_path": attempts_path.as_posix() if attempts_path else "",
        "label_file": config.label_file.as_posix(),
        "scene_file": config.scene_file.as_posix(),
        "scene_id": config.scene_id or config.scene_file.stem,
        "map_id": config.map_id,
        "elapsed_seconds": elapsed_seconds,
        "sharding": {
            "enabled": True,
            "axis": _normalize_shard_axis(getattr(sharding, "axis", "ue")),
            "requested_axis": str(getattr(sharding, "axis", "ue")),
            "shard_size": int(getattr(sharding, "shard_size", config.max_ue)),
            "planned_shard_count": shard_count,
            "result_file_count": len(shard_results),
            "parallel_workers": parallel_workers,
            "gpu_ids": gpu_ids,
            "filename_pattern": getattr(
                sharding, "filename_pattern", "result_{shard_index:03d}.h5"
            ),
            "results_dir": getattr(sharding, "results_dir", "results"),
            "manifest_dir": getattr(sharding, "manifest_dir", "manifest"),
            "visualization_mode": getattr(sharding, "visualization_mode", "first_shard"),
            "fallback": _sharding_fallback_summary(sharding, shard_attempts),
        },
        "config_snapshot": _config_snapshot(config),
        "performance": _aggregate_shard_performance(output_dir, shard_results),
    }
    write_manifest(manifest_dir / "manifest.json", aggregate_manifest)
    _generate_sharded_radio_maps_if_requested(config, output_dir)
    return output_dir


def _run_rt_truth_pipeline_single(config: RTTruthRunConfig) -> Path:
    """Run Phase 2 minimal RT truth and write HDF5, manifest, and log."""

    start = time.perf_counter()
    output_dir = config.output_dir
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    worker_id = (
        f"shard_{_shard_spec_id(config.shard_spec)}"
        if config.shard_spec is not None
        else "main"
    )
    tracer = PerfTracer(output_dir, config.debug_config, worker_id=worker_id)
    tracer.start()
    try:
        return _run_rt_truth_pipeline_single_impl(
            config,
            tracer,
            start=start,
            output_dir=output_dir,
            logs_dir=logs_dir,
        )
    except Exception as exc:
        tracer.finish(
            {"output_dir": output_dir.as_posix()},
            status="failed",
            exception=exc,
        )
        raise


def _run_rt_truth_pipeline_single_impl(
    config: RTTruthRunConfig,
    tracer: PerfTracer,
    *,
    start: float,
    output_dir: Path,
    logs_dir: Path,
) -> Path:
    output_plan = config.output_plan or build_rt_output_plan(
        config.output_profile,
        products=config.output_products,
        array_sources=config.spectrum_config.sources,
    )
    mapping = resolve_link_roles(config.link_config.phy_link_direction)
    link_config = replace(
        config.link_config,
        tx_role=mapping.tx_role,
        rx_role=mapping.rx_role,
    )
    with tracer.span("topology_load"):
        role_topology = load_role_topology_from_label(
            config.label_file,
            max_bs=config.max_bs,
            max_ue=config.max_ue,
            ue_start=config.shard_spec.ue_start if config.shard_spec else 0,
            ue_count=config.shard_spec.ue_count if config.shard_spec else None,
            ue_indices=config.shard_spec.ue_indices if config.shard_spec else None,
            bs_indices=config.shard_spec.bs_indices if config.shard_spec else None,
        )
        topology = resolve_role_topology(role_topology, mapping)
    antenna = _build_resolved_antenna(config, mapping)
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
                tx_velocity=_resolve_tx_rx_values(
                    config.bs_velocity_mps,
                    config.ue_velocity_mps,
                    mapping,
                )[0],
                rx_velocity=_resolve_tx_rx_values(
                    config.bs_velocity_mps,
                    config.ue_velocity_mps,
                    mapping,
                )[1],
                merge_shapes=config.merge_shapes,
                compute_cfr=output_plan.compute_cfr,
                compute_cir=output_plan.compute_cir,
                compute_path_samples=output_plan.compute_path_samples,
            ),
        )
    elapsed_seconds = time.perf_counter() - start
    environment = collect_basic_environment()
    observation_bundle = None
    phy_waveform = None
    phy_extra: dict = {}
    should_run_phy = config.observation_snr_db is not None or output_plan.write_iq_link_library
    if should_run_phy:
        if adapter_result.truth is None:
            msg = "PHY observation requires RT output plan with compute_cfr=true"
            raise ValueError(msg)
        phy_module = get_phy_module(config.phy_standard)
        with tracer.span(f"{phy_module.standard}_observation"):
            phy_result = phy_module.run(
                PHYContext(
                    config=config,
                    adapter_result=adapter_result,
                    tracer=tracer,
                )
            )
        observation_bundle = phy_result.to_bundle()
        phy_waveform = phy_result.waveform
        phy_extra = {
            "waveform_extras": phy_result.waveform_extras,
            "array_outputs": phy_result.array_outputs,
            "diagnostics": phy_result.diagnostics,
            "metadata": phy_result.metadata,
            "multiuser": phy_result.multiuser,
        }
    phase = (
        5
        if output_plan.write_iq_link_library and phy_waveform is not None
        else 7 if observation_bundle is not None else 3 if config.max_depth > 0 else 2
    )

    scene_id = config.scene_id or config.scene_file.stem
    truth_summary = adapter_result.truth or adapter_result.link_summary
    with tracer.span("derived_nlos"):
        derived = build_derived_labels(
            topology,
            truth_summary,
            adapter_result.path_table,
            adapter_result.cir_truth,
            link_config=link_config,
        )
        nlos_path_truth = (
            build_nlos_path_truth(adapter_result.path_table)
            if output_plan.compute_nlos_truth
            else None
        )
    with tracer.span("array_outputs"):
        truth_cfr = adapter_result.truth.cfr if adapter_result.truth is not None else None
        cfr_est = (
            observation_bundle.observation.cfr_est
            if observation_bundle and observation_bundle.observation
            else None
        )
        if phy_extra.get("waveform_extras"):
            _attach_phy_array_outputs(
                phy_extra,
                derived,
                config,
                truth_cfr,
                cfr_est,
                rx_orientation_rad=adapter_result.rx_orientation_rad,
            )
        elif (
            truth_cfr is not None
            and config.spectrum_config.enabled
            and "truth_cfr" in config.spectrum_config.sources
        ):
            phy_extra["array_outputs"] = _build_truth_array_outputs(
                config,
                derived,
                truth_cfr,
                rx_orientation_rad=adapter_result.rx_orientation_rad,
            )
        elif (
            cfr_est is not None
            and config.spectrum_config.enabled
            and "cfr_est" in config.spectrum_config.sources
        ):
            phy_extra["array_outputs"] = _build_cfr_est_array_outputs(
                config,
                derived,
                cfr_est,
                rx_orientation_rad=adapter_result.rx_orientation_rad,
            )
    with tracer.span("ranging"):
        ranging_result = run_ranging_observation(
            observation=observation_bundle.observation if observation_bundle else None,
            frequency=frequency,
            derived=derived,
            config=config.ranging_config,
        )
    shard_metadata = _build_shard_metadata(config, role_topology, mapping)
    with tracer.span("iq_observation"):
        iq_result = (
            build_iq_observation(
                iq_config=config.iq_config,
                noncooperative_config=config.noncooperative_config,
                waveform_extras=phy_extra.get("waveform_extras"),
                multiuser=phy_extra.get("multiuser"),
                topology=topology,
                shard=shard_metadata,
                sample_rate_hz=(
                    phy_waveform.sample_rate_hz
                    if phy_waveform
                    else frequency.bandwidth_hz
                ),
                fft_size=(
                    phy_waveform.fft_size
                    if phy_waveform
                    else frequency.num_subcarriers
                ),
                cp_length=(
                    phy_waveform.cp_length
                    if phy_waveform
                    else config.cp_length
                ),
                num_ofdm_symbols=(
                    phy_waveform.num_ofdm_symbols
                    if phy_waveform
                    else config.num_ofdm_symbols
                ),
            )
            if phy_waveform is not None
            else None
        )
    common_metadata = Metadata(
        run_id=output_dir.name,
        random_seed=config.seed,
        config_snapshot=json.dumps(_config_snapshot(config), sort_keys=True),
        contract_name=output_plan.contract_name,
        output_profile=output_plan.profile,
        measurement_realism_level=f"phase{phase}_rt_truth",
        observation_branch_enabled=(
            observation_bundle is not None and output_plan.write_cfr_observation
        ),
        software_versions=json.dumps(adapter_result.runtime_versions, sort_keys=True),
        output_products=output_plan.products,
    )
    input_spec = InputSpec(
        label_file=config.label_file.as_posix(),
        scene_file=config.scene_file.as_posix(),
        input_dataset_id=_input_dataset_id(config.label_file),
        input_schema=f"standard_label_{STANDARD_LABEL_SCHEMA_VERSION}",
    )
    devices = _build_device_state(
        config,
        topology,
        mapping,
        tx_orientation_rad_scene=adapter_result.tx_orientation_rad,
        rx_orientation_rad_scene=adapter_result.rx_orientation_rad,
    )
    scene = SceneSpec(
        scene_name=config.scene_file.stem,
        scene_file=config.scene_file.as_posix(),
        scene_id=scene_id,
        map_id=config.map_id,
        material_policy="sionna_rt_scene_materials",
    )
    runtime = RuntimeInfo(
        sionna_version=adapter_result.runtime_versions["sionna"],
        sionna_rt_version=adapter_result.runtime_versions["sionna_rt"],
        torch_version=adapter_result.runtime_versions["torch"],
        mitsuba_version=adapter_result.runtime_versions["mitsuba"],
        drjit_version=adapter_result.runtime_versions["drjit"],
        cuda_available=bool(environment["cuda_available"]),
        cuda_device_name=str(environment["cuda_device_name"]),
        command_line="run-rt-truth",
        elapsed_seconds=elapsed_seconds,
    )
    if output_plan.write_iq_link_library:
        if iq_result is None:
            msg = "iq_link_library output requires clean link IQ capture"
            raise ValueError(msg)
        iq_library_result = IQLinkLibraryResult(
            metadata=common_metadata,
            input_spec=input_spec,
            topology=topology,
            devices=devices,
            antenna=antenna,
            scene=scene,
            frequency=frequency,
            runtime=runtime,
            iq=iq_result,
            link=link_config,
            shard=shard_metadata,
        )
        with tracer.span("hdf5_write"):
            results_path = write_iq_link_library_result(
                output_dir / config.hdf5_filename,
                iq_library_result,
                compression=config.hdf5_compression,
                gzip_level=config.hdf5_gzip_level,
                tracer=tracer,
            )
        with tracer.span("schema_validate"):
            validate_hdf5_contract(results_path)
        manifest_path = _write_single_manifest(
            config=config,
            output_dir=output_dir,
            scene_id=scene_id,
            adapter_result=adapter_result,
            results_path=results_path,
            phase=phase,
            elapsed_seconds=elapsed_seconds,
            shard_metadata=shard_metadata,
            tracer=tracer,
        )
        _write_run_log(
            logs_dir,
            config=config,
            shard_metadata=shard_metadata,
            phase=phase,
            results_path=results_path,
            manifest_path=manifest_path,
            adapter_result=adapter_result,
            path_count=int(adapter_result.link_summary.geometric_path_count.sum()),
        )
        return results_path
    if output_plan.write_compact_link_labels:
        labels_result = RTLabelsOnlyResult(
            metadata=common_metadata,
            input_spec=input_spec,
            topology=topology,
            devices=devices,
            antenna=antenna,
            scene=scene,
            frequency=frequency,
            runtime=runtime,
            derived=derived,
            link_labels=RTCompactLinkLabels.from_topology(
                topology,
                derived,
                shard=shard_metadata,
            ),
            link=link_config,
            shard=shard_metadata,
        )
        with tracer.span("hdf5_write"):
            results_path = write_rt_labels_result(
                output_dir / config.hdf5_filename,
                labels_result,
                compression=config.hdf5_compression,
                gzip_level=config.hdf5_gzip_level,
                tracer=tracer,
            )
        with tracer.span("schema_validate"):
            validate_hdf5_contract(results_path)
        manifest_path = _write_single_manifest(
            config=config,
            output_dir=output_dir,
            scene_id=scene_id,
            adapter_result=adapter_result,
            results_path=results_path,
            phase=phase,
            elapsed_seconds=elapsed_seconds,
            shard_metadata=shard_metadata,
            tracer=tracer,
        )
        _write_run_log(
            logs_dir,
            config=config,
            shard_metadata=shard_metadata,
            phase=phase,
            results_path=results_path,
            manifest_path=manifest_path,
            adapter_result=adapter_result,
            path_count=int(adapter_result.link_summary.geometric_path_count.sum()),
        )
        return results_path
    if output_plan.write_cfr_truth and adapter_result.truth is None:
        msg = "Selected cfr_truth output requires RT output plan with compute_cfr=true"
        raise ValueError(msg)
    if output_plan.write_path_samples and adapter_result.path_samples is None:
        msg = "Selected path_samples output requires RT path sample extraction"
        raise ValueError(msg)
    needs_observation_result = (
        output_plan.write_cfr_observation
        or output_plan.write_ranging
        or output_plan.write_iq
        or output_plan.write_multiuser
    )
    result = MeasurementSimulationResult(
        metadata=common_metadata,
        input_spec=input_spec,
        topology=topology,
        devices=devices,
        motion=_build_motion_spec(config),
        antenna=antenna,
        scene=scene,
        frequency=frequency,
        runtime=runtime,
        truth=adapter_result.truth,
        path_samples=(
            adapter_result.path_samples if output_plan.write_path_samples else None
        ),
        cir_truth=adapter_result.cir_truth if output_plan.write_cir_truth else None,
        derived=(
            derived
            if (output_plan.write_derived or output_plan.write_link_labels)
            else None
        ),
        path_table=(
            adapter_result.path_table
            if (config.save_full_paths and output_plan.write_path_full)
            else None
        ),
        nlos_path_truth=nlos_path_truth if output_plan.write_nlos_path_truth else None,
        waveform=(
            observation_bundle.waveform
            if (observation_bundle and needs_observation_result)
            else None
        ),
        observation=(
            observation_bundle.observation
            if (observation_bundle and needs_observation_result)
            else None
        ),
        impairments=(
            observation_bundle.impairments
            if (observation_bundle and needs_observation_result)
            else None
        ),
        receiver=(
            observation_bundle.receiver
            if (observation_bundle and needs_observation_result)
            else None
        ),
        evaluation=(
            observation_bundle.evaluation
            if (observation_bundle and needs_observation_result)
            else None
        ),
        calibration=(
            CalibrationResult.synthetic_default()
            if (
                observation_bundle
                and config.calibration_enabled
                and output_plan.write_calibration
            )
            else None
        ),
        link=link_config,
        shard=shard_metadata,
        waveform_extras=(
            phy_extra.get("waveform_extras") if output_plan.write_cfr_observation else None
        ),
        array_outputs=phy_extra.get("array_outputs") if output_plan.write_array_outputs else None,
        ranging=ranging_result if output_plan.write_ranging else None,
        multiuser=phy_extra.get("multiuser") if output_plan.write_multiuser else None,
        iq=iq_result if output_plan.write_iq else None,
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
            gzip_level=config.hdf5_gzip_level,
            tracer=tracer,
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
        "output_profile": config.output_profile,
        "output_products": list(output_plan.products),
        "results_h5": results_path.as_posix(),
        "label_file": config.label_file.as_posix(),
        "scene_file": config.scene_file.as_posix(),
        "scene_id": scene_id,
        "map_id": config.map_id,
        "config_snapshot": _config_snapshot(config),
        "software_versions": adapter_result.runtime_versions,
        "raw_cfr_shape": adapter_result.raw_cfr_shape,
        "internal_cfr_shape": adapter_result.internal_cfr_shape,
        "path_count": _adapter_path_count(adapter_result),
        "observation_snr_db": config.observation_snr_db,
        "elapsed_seconds": elapsed_seconds,
    }
    batching_stats = dict(phy_extra.get("diagnostics", {})).get("batching_stats", {})
    if batching_stats:
        manifest_data["nr_pusch_batching"] = batching_stats
    if visualization_summary is not None:
        manifest_data["visualization"] = _manifest_visualization_summary(visualization_summary)
    if result.diagnostics is not None:
        manifest_data["diagnostics"] = result.diagnostics.to_summary_dict()
    if result.ranging is not None:
        manifest_data["ranging"] = _manifest_ranging_summary(result.ranging)
    if shard_metadata is not None:
        manifest_data["shard"] = _manifest_shard_summary(shard_metadata)
    manifest_path = (
        _shard_manifest_path(config)
        if shard_metadata is not None
        else output_dir / "manifest.json"
    )
    with tracer.span("manifest_write"):
        manifest_path = write_manifest(manifest_path, manifest_data)
    perf_summary = tracer.finish({"results_h5": results_path.as_posix()})
    if perf_summary:
        manifest_data["performance"] = {
            "enabled": True,
            "summary_path": perf_summary["logs"]["events"].replace(
                "perf_events", "perf_summary"
            ).replace(".jsonl", ".json"),
            "stage_totals_s": perf_summary.get("stage_totals_s", {}),
            "hardware_summary": perf_summary.get("hardware_summary", {}),
            "dataset_write_summary": perf_summary.get("dataset_write_summary", {}),
        }
        manifest_path = write_manifest(manifest_path, manifest_data)

    run_log_name = (
        f"run_{_shard_spec_id(config.shard_spec)}.log"
        if shard_metadata is not None
        else "run.log"
    )
    (logs_dir / run_log_name).write_text(
        "\n".join(
            [
                f"phase={phase}",
                f"output_profile={config.output_profile}",
                f"output_products={','.join(output_plan.products)}",
                f"results_h5={results_path.as_posix()}",
                f"manifest={manifest_path.as_posix()}",
                f"raw_cfr_shape={adapter_result.raw_cfr_shape}",
                f"internal_cfr_shape={adapter_result.internal_cfr_shape}",
                f"path_count={_adapter_path_count(adapter_result)}",
                f"observation_snr_db={config.observation_snr_db}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return results_path


def _adapter_path_count(adapter_result: Any) -> int:
    path_samples = getattr(adapter_result, "path_samples", None)
    if path_samples is not None:
        return int(path_samples.path_count.sum())
    link_summary = getattr(adapter_result, "link_summary", None)
    if link_summary is not None:
        return int(link_summary.geometric_path_count.sum())
    return 0


def _write_single_manifest(
    *,
    config: RTTruthRunConfig,
    output_dir: Path,
    scene_id: str,
    adapter_result: Any,
    results_path: Path,
    phase: int,
    elapsed_seconds: float,
    shard_metadata: ShardMetadata | None,
    tracer: PerfTracer,
    visualization_summary: Any | None = None,
    diagnostics: DiagnosticsReport | None = None,
    ranging_result: Any | None = None,
    batching_stats: dict | None = None,
) -> Path:
    path_count = (
        int(adapter_result.path_samples.path_count.sum())
        if adapter_result.path_samples is not None
        else int(adapter_result.link_summary.geometric_path_count.sum())
    )
    manifest_data = {
        "phase": phase,
        "output_profile": config.output_profile,
        "output_products": list(config.output_plan.products)
        if config.output_plan is not None
        else list(
            build_rt_output_plan(
                config.output_profile,
                products=config.output_products,
                array_sources=config.spectrum_config.sources,
            ).products
        ),
        "results_h5": results_path.as_posix(),
        "label_file": config.label_file.as_posix(),
        "scene_file": config.scene_file.as_posix(),
        "scene_id": scene_id,
        "map_id": config.map_id,
        "config_snapshot": _config_snapshot(config),
        "software_versions": adapter_result.runtime_versions,
        "raw_cfr_shape": adapter_result.raw_cfr_shape,
        "internal_cfr_shape": adapter_result.internal_cfr_shape,
        "path_count": path_count,
        "observation_snr_db": config.observation_snr_db,
        "elapsed_seconds": elapsed_seconds,
    }
    if batching_stats:
        manifest_data["nr_pusch_batching"] = batching_stats
    if visualization_summary is not None:
        manifest_data["visualization"] = _manifest_visualization_summary(
            visualization_summary
        )
    if diagnostics is not None:
        manifest_data["diagnostics"] = diagnostics.to_summary_dict()
    if ranging_result is not None:
        manifest_data["ranging"] = _manifest_ranging_summary(ranging_result)
    if shard_metadata is not None:
        manifest_data["shard"] = _manifest_shard_summary(shard_metadata)
    manifest_path = (
        _shard_manifest_path(config)
        if shard_metadata is not None
        else output_dir / "manifest.json"
    )
    with tracer.span("manifest_write"):
        manifest_path = write_manifest(manifest_path, manifest_data)
    perf_summary = tracer.finish({"results_h5": results_path.as_posix()})
    if perf_summary:
        manifest_data["performance"] = {
            "enabled": True,
            "summary_path": perf_summary["logs"]["events"].replace(
                "perf_events", "perf_summary"
            ).replace(".jsonl", ".json"),
            "stage_totals_s": perf_summary.get("stage_totals_s", {}),
            "hardware_summary": perf_summary.get("hardware_summary", {}),
            "dataset_write_summary": perf_summary.get("dataset_write_summary", {}),
        }
        manifest_path = write_manifest(manifest_path, manifest_data)
    return manifest_path


def _write_run_log(
    logs_dir: Path,
    *,
    config: RTTruthRunConfig,
    shard_metadata: ShardMetadata | None,
    phase: int,
    results_path: Path,
    manifest_path: Path,
    adapter_result: Any,
    path_count: int,
) -> None:
    run_log_name = (
        f"run_{_shard_spec_id(config.shard_spec)}.log"
        if shard_metadata is not None
        else "run.log"
    )
    (logs_dir / run_log_name).write_text(
        "\n".join(
            [
                f"phase={phase}",
                f"output_profile={config.output_profile}",
                f"output_products={','.join(config.output_products or ())}",
                f"results_h5={results_path.as_posix()}",
                f"manifest={manifest_path.as_posix()}",
                f"raw_cfr_shape={adapter_result.raw_cfr_shape}",
                f"internal_cfr_shape={adapter_result.internal_cfr_shape}",
                f"path_count={path_count}",
                f"observation_snr_db={config.observation_snr_db}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


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
    axis = _normalize_shard_axis(getattr(sharding, "axis", "ue"))
    shard_size = int(getattr(sharding, "shard_size", config.max_ue))
    if shard_size < 1:
        msg = "output.sharding.shard_size must be positive"
        raise ValueError(msg)
    available_bs, available_ue = count_topology_points(config.label_file)
    effective_ue_count = min(config.max_ue, available_ue)
    if min(config.max_bs, available_bs) < 1 or effective_ue_count < 1:
        msg = "Label file must contain at least one selected BS and UE"
        raise ValueError(msg)
    shard_count = (effective_ue_count + shard_size - 1) // shard_size
    return [
        ShardSpec(
            shard_index=shard_index,
            shard_count=shard_count,
            axis=axis,
            ue_start=shard_index * shard_size,
            ue_count=min(shard_size, effective_ue_count - shard_index * shard_size),
        )
        for shard_index in range(shard_count)
    ]


def _normalize_shard_axis(axis: object) -> str:
    axis_str = str(axis)
    if axis_str in ("rx", "ue"):
        return "ue"
    msg = f"Only rx/ue sharding is supported, got {axis_str!r}"
    raise ValueError(msg)


def _shard_results_dir_name(config: RTTruthRunConfig) -> Path:
    sharding = config.output_sharding_config
    return Path(str(getattr(sharding, "results_dir", "results")))


def _shard_manifest_dir_name(config: RTTruthRunConfig) -> Path:
    sharding = config.output_sharding_config
    return Path(str(getattr(sharding, "manifest_dir", "manifest")))


def _shard_results_dir(config: RTTruthRunConfig) -> Path:
    return config.output_dir / _shard_results_dir_name(config)


def _shard_manifest_dir(config: RTTruthRunConfig) -> Path:
    return config.output_dir / _shard_manifest_dir_name(config)


def _shard_spec_id(spec: ShardSpec) -> str:
    return str(spec.shard_id or f"{spec.shard_index:03d}")


def _shard_manifest_path(config: RTTruthRunConfig) -> Path:
    spec = config.shard_spec
    if spec is None:
        return config.output_dir / "manifest.json"
    return _shard_manifest_dir(config) / f"manifest_{_shard_spec_id(spec)}.json"


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
    if spec.shard_id and spec.shard_id != f"{spec.shard_index:03d}":
        path = Path(hdf5_filename)
        index_token = f"{spec.shard_index:03d}"
        fallback_stem = path.stem.replace(index_token, spec.shard_id, 1)
        if fallback_stem == path.stem:
            fallback_stem = f"{path.stem}_{spec.shard_id}"
        hdf5_filename = f"{fallback_stem}{path.suffix}"
    hdf5_filename = (_shard_results_dir_name(config) / hdf5_filename).as_posix()
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
    vis = _without_aggregate_visualization_plots(vis)
    if mode == "first_shard" and spec.shard_index != 0:
        return replace(vis, enabled=False)
    if mode == "all_shards":
        return replace(vis, output_dir=f"{vis.output_dir}_{spec.shard_index:03d}")
    return vis


def _without_aggregate_visualization_plots(
    config: VisualizationRunConfig,
) -> VisualizationRunConfig:
    """Remove plots that are generated once from the aggregate manifest."""

    plots = tuple(plot for plot in config.plots if plot != "radio_map")
    if plots == config.plots:
        return config
    if not plots:
        return replace(config, enabled=False, plots=plots)
    return replace(config, plots=plots)


def _generate_sharded_radio_maps_if_requested(
    config: RTTruthRunConfig,
    output_dir: Path,
) -> None:
    vis = config.visualization_config
    if not vis.enabled or "radio_map" not in vis.plots:
        return
    generate_radio_map_heatmaps(
        output_dir,
        output_dir / vis.output_dir / "heatmaps",
        config=RadioMapRenderConfig(
            render_mode=vis.radio_map_mode,
            grid_resolution_m=vis.radio_map_grid_resolution_m,
            dpi=vis.dpi,
            show_samples_on_interpolated=vis.radio_map_show_samples,
        ),
    )


def _run_shard_worker(config: RTTruthRunConfig, gpu_id: int | None) -> Path:
    if gpu_id is not None and str(config.device).startswith("cuda"):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    return _run_rt_truth_pipeline_single(config)


def _run_shard_worker_attempt(config: RTTruthRunConfig, gpu_id: int | None) -> Path:
    """Run one shard attempt, isolating fallback-enabled work in a fresh process."""

    fallback = getattr(getattr(config, "output_sharding_config", None), "fallback", None)
    if bool(getattr(fallback, "enabled", False)):
        return _run_shard_worker_in_isolated_process(config, gpu_id)
    return _run_shard_worker(config, gpu_id)


def _run_shard_worker_in_isolated_process(
    config: RTTruthRunConfig,
    gpu_id: int | None,
) -> Path:
    """Run a shard attempt in a fresh process so Dr.Jit/CUDA OOM state is released."""

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as executor:
        future = executor.submit(_run_shard_worker, config, gpu_id)
        return future.result()


def _run_shard_spec_with_fallback(
    config: RTTruthRunConfig,
    spec: ShardSpec,
    gpu_id: int | None,
) -> dict[str, list[dict[str, object]]]:
    return _run_shard_spec_attempt(config, spec, gpu_id, reason="")


def _run_shard_spec_attempt(
    config: RTTruthRunConfig,
    spec: ShardSpec,
    gpu_id: int | None,
    *,
    reason: str,
) -> dict[str, list[dict[str, object]]]:
    shard_config = _build_shard_run_config(config, spec)
    attempt: dict[str, object] = {
        "shard_id": spec.shard_id or f"{spec.shard_index:03d}",
        "shard_index": spec.shard_index,
        "parent_shard_index": (
            spec.parent_shard_index
            if spec.parent_shard_index is not None
            else spec.shard_index
        ),
        "fallback_level": spec.fallback_level,
        "ue_start": spec.ue_start,
        "ue_count": spec.ue_count,
        "reason": reason or spec.fallback_reason,
        "status": "started",
    }
    try:
        result_path = _run_shard_worker_attempt(shard_config, gpu_id)
    except Exception as exc:
        retry_error = _classify_retryable_shard_error(exc)
        attempt.update(
            {
                "status": "failed",
                "retry_error": retry_error or "",
                "error": str(exc),
            }
        )
        if not _can_fallback_shard(config, spec, retry_error):
            raise
        _clear_accelerator_caches()
        split_factor = int(
            getattr(
                getattr(config.output_sharding_config, "fallback", None),
                "split_factor",
                2,
            )
        )
        children = _split_shard_spec_for_fallback(
            spec,
            retry_error or "retryable",
            split_factor=split_factor,
        )
        attempt.update(
            {
                "status": "split",
                "children": [child.shard_id for child in children],
            }
        )
        results: list[dict[str, object]] = []
        attempts: list[dict[str, object]] = [attempt]
        for child in children:
            _clear_accelerator_caches()
            outcome = _run_shard_spec_attempt(
                config,
                child,
                gpu_id,
                reason=retry_error or "retryable",
            )
            results.extend(outcome["results"])
            attempts.extend(outcome["attempts"])
        return {"results": results, "attempts": attempts}
    attempt["status"] = "succeeded"
    return {
        "results": [_shard_result_summary(shard_config, result_path)],
        "attempts": [attempt],
    }


def _clear_accelerator_caches() -> None:
    """Release best-effort accelerator caches before retrying a smaller shard."""

    gc.collect()
    try:
        import drjit as dr

        dr.flush_malloc_cache()
        dr.flush_kernel_cache()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _classify_retryable_shard_error(exc: BaseException) -> str | None:
    message = "".join(traceback.format_exception(exc)).lower()
    if "jit_var_counter" in message or "exceeds the limit of 2^32" in message:
        return "drjit_array_limit"
    if (
        "cuda out of memory" in message
        or "out of memory" in message
        or "jit_malloc(): out of memory" in message
        or "oom" in message
    ):
        return "cuda_oom"
    return None


def _can_fallback_shard(
    config: RTTruthRunConfig,
    spec: ShardSpec,
    retry_error: str | None,
) -> bool:
    sharding = config.output_sharding_config
    fallback = getattr(sharding, "fallback", None)
    if fallback is None or not bool(getattr(fallback, "enabled", False)):
        return False
    if retry_error not in set(getattr(fallback, "retry_errors", [])):
        return False
    count = _shard_ue_count(spec)
    min_size = int(getattr(fallback, "min_shard_size", 1))
    return count > min_size


def _shard_ue_count(spec: ShardSpec) -> int:
    if spec.ue_indices is not None:
        return len(spec.ue_indices)
    return int(spec.ue_count or 0)


def _split_shard_spec_for_fallback(
    spec: ShardSpec,
    reason: str,
    *,
    split_factor: int = 2,
) -> list[ShardSpec]:
    count = _shard_ue_count(spec)
    if count <= 1:
        return []
    split_factor = max(int(split_factor), 2)
    base = count // split_factor
    remainder = count % split_factor
    sizes = [
        base + (1 if index < remainder else 0)
        for index in range(split_factor)
        if base + (1 if index < remainder else 0) > 0
    ]
    parent_id = _shard_spec_id(spec)
    children: list[ShardSpec] = []
    offset = 0
    for child_index, size in enumerate(sizes):
        if spec.ue_indices is None:
            ue_start = spec.ue_start + offset
            ue_indices = None
        else:
            selected = spec.ue_indices[offset : offset + size]
            ue_start = int(selected[0])
            ue_indices = tuple(int(index) for index in selected)
        children.append(
            replace(
                spec,
                ue_start=ue_start,
                ue_count=size,
                ue_indices=ue_indices,
                shard_id=f"{parent_id}_{child_index:02d}",
                parent_shard_index=(
                    spec.parent_shard_index
                    if spec.parent_shard_index is not None
                    else spec.shard_index
                ),
                fallback_level=spec.fallback_level + 1,
                fallback_reason=reason,
            )
        )
        offset += size
    return children


def _shard_result_summary(config: RTTruthRunConfig, result_path: Path) -> dict[str, object]:
    spec = config.shard_spec
    if spec is None:
        msg = "shard result summary requires shard_spec"
        raise ValueError(msg)
    manifest_path = _shard_manifest_path(config)
    shard_manifest = _read_json_if_exists(manifest_path)
    shard_summary = shard_manifest.get("shard", {})
    rx_indices = [
        int(index)
        for index in shard_summary.get(
            "global_rx_indices",
            list(range(int(config.max_bs))),
        )
    ]
    tx_indices = [
        int(index)
        for index in shard_summary.get(
            "global_tx_indices",
            list(range(spec.ue_start, spec.ue_start + int(spec.ue_count or 0))),
        )
    ]
    ue_indices = [
        int(index)
        for index in shard_summary.get("global_ue_indices", tx_indices)
    ]
    bs_indices = [
        int(index)
        for index in shard_summary.get("global_bs_indices", rx_indices)
    ]
    return {
        "shard_id": spec.shard_id or f"{spec.shard_index:03d}",
        "shard_index": spec.shard_index,
        "shard_count": spec.shard_count,
        "parent_shard_index": (
            spec.parent_shard_index
            if spec.parent_shard_index is not None
            else spec.shard_index
        ),
        "fallback_level": spec.fallback_level,
        "fallback_reason": spec.fallback_reason,
        "axis": str(shard_summary.get("axis", spec.axis)),
        "result_h5": Path(result_path).as_posix(),
        "manifest": manifest_path.as_posix(),
        "ue_start": spec.ue_start,
        "ue_count": int(spec.ue_count or len(tx_indices)),
        "global_rx_start": int(
            shard_summary.get("global_rx_start", rx_indices[0] if rx_indices else 0)
        ),
        "global_rx_count": len(rx_indices),
        "global_rx_indices": rx_indices,
        "global_tx_indices": tx_indices,
        "global_ue_count": len(ue_indices),
        "global_ue_indices": ue_indices,
        "global_bs_indices": bs_indices,
        "nr_pusch_batching": shard_manifest.get("nr_pusch_batching", {}),
    }


def _read_json_if_exists(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_shard_attempts(
    manifest_dir: Path,
    shard_attempts: list[dict[str, object]],
) -> Path | None:
    if not shard_attempts:
        return None
    path = manifest_dir / "shard_attempts.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in shard_attempts) + "\n",
        encoding="utf-8",
    )
    return path


def _sharding_fallback_summary(
    sharding: object,
    shard_attempts: list[dict[str, object]],
) -> dict[str, object]:
    fallback = getattr(sharding, "fallback", None)
    split_attempts = [
        item for item in shard_attempts if str(item.get("status", "")) == "split"
    ]
    return {
        "enabled": bool(getattr(fallback, "enabled", False)),
        "min_shard_size": int(getattr(fallback, "min_shard_size", 1)),
        "split_factor": int(getattr(fallback, "split_factor", 2)),
        "retry_errors": list(getattr(fallback, "retry_errors", [])),
        "split_count": len(split_attempts),
        "split_shard_ids": [str(item.get("shard_id", "")) for item in split_attempts],
    }


def _aggregate_shard_performance(
    output_dir: Path,
    shard_results: list[dict[str, object]],
) -> dict[str, object]:
    summaries: list[dict[str, object]] = []
    stage_totals: dict[str, float] = {}
    total_durations: list[float] = []
    for shard in shard_results:
        shard_id = str(shard.get("shard_id", f"{int(shard['shard_index']):03d}"))
        shard_index = int(shard["shard_index"])
        summary_path = output_dir / "logs" / f"perf_summary_shard_{shard_id}.json"
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
                "hardware_summary": summary.get("hardware_summary", {}),
                "dataset_write_summary": summary.get("dataset_write_summary", {}),
            }
        )
        total_duration = summary.get("total_duration_s")
        if isinstance(total_duration, (int, float)):
            total_durations.append(float(total_duration))
        for name, value in dict(summary.get("stage_totals_s", {})).items():
            stage_totals[str(name)] = stage_totals.get(str(name), 0.0) + float(value)
    hardware = _aggregate_perf_hardware(summaries)
    dataset_writes = _aggregate_perf_dataset_writes(summaries)
    return {
        "enabled": bool(summaries),
        "shard_summaries": summaries,
        "stage_totals_s": stage_totals,
        "max_shard_duration_s": max(total_durations) if total_durations else None,
        "sum_shard_duration_s": sum(total_durations) if total_durations else None,
        "hardware_summary": hardware,
        "dataset_write_summary": dataset_writes,
    }


def _aggregate_perf_hardware(summaries: list[dict[str, object]]) -> dict[str, object]:
    hardware_items = [
        dict(item.get("hardware_summary", {}))
        for item in summaries
        if isinstance(item.get("hardware_summary"), dict)
    ]
    if not hardware_items:
        return {}
    return {
        "sample_count": sum(int(item.get("sample_count", 0) or 0) for item in hardware_items),
        "gpu_sample_count": sum(
            int(item.get("gpu_sample_count", 0) or 0) for item in hardware_items
        ),
        "peak_rss_mb": _max_optional(item.get("peak_rss_mb") for item in hardware_items),
        "peak_gpu_mem_used_mb": _max_optional(
            item.get("peak_gpu_mem_used_mb") for item in hardware_items
        ),
        "max_gpu_util_percent": _max_optional(
            item.get("max_gpu_util_percent") for item in hardware_items
        ),
        "mean_gpu_util_percent": _mean_optional(
            item.get("mean_gpu_util_percent") for item in hardware_items
        ),
    }


def _aggregate_perf_dataset_writes(
    summaries: list[dict[str, object]],
) -> dict[str, object]:
    items = [
        dict(item.get("dataset_write_summary", {}))
        for item in summaries
        if isinstance(item.get("dataset_write_summary"), dict)
    ]
    if not items:
        return {}
    total_raw = sum(int(item.get("total_raw_bytes", 0) or 0) for item in items)
    storage_values = [
        int(item.get("total_storage_bytes", 0) or 0)
        for item in items
        if item.get("total_storage_bytes") is not None
    ]
    total_storage = sum(storage_values)
    return {
        "dataset_count": sum(int(item.get("dataset_count", 0) or 0) for item in items),
        "total_raw_bytes": total_raw,
        "total_storage_bytes": total_storage if storage_values else None,
        "storage_to_raw_ratio": (
            float(total_storage) / float(total_raw)
            if total_raw > 0 and storage_values
            else None
        ),
        "raw_to_storage_ratio": (
            float(total_raw) / float(total_storage)
            if total_storage > 0 and storage_values
            else None
        ),
    }


def _max_optional(values: object) -> float | None:
    finite: list[float] = []
    for value in values:
        if isinstance(value, (int, float)):
            finite.append(float(value))
    return max(finite) if finite else None


def _mean_optional(values: object) -> float | None:
    finite: list[float] = []
    for value in values:
        if isinstance(value, (int, float)):
            finite.append(float(value))
    return sum(finite) / len(finite) if finite else None


def _build_shard_metadata(
    config: RTTruthRunConfig,
    role_topology: RoleTopology,
    mapping: LinkRoleMapping,
) -> ShardMetadata | None:
    spec = config.shard_spec
    if spec is None:
        return None
    tx_indices, rx_indices = resolved_global_indices(role_topology, mapping)
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
    mapping: LinkRoleMapping,
    tx_orientation_rad_scene: np.ndarray | None = None,
    rx_orientation_rad_scene: np.ndarray | None = None,
) -> DeviceState:
    num_snap = max(config.num_time_steps, 1)
    tx_velocity, rx_velocity = _resolve_tx_rx_values(
        config.bs_velocity_mps,
        config.ue_velocity_mps,
        mapping,
    )
    v_tx = np.array(tx_velocity, dtype=np.float32)
    v_rx = np.array(rx_velocity, dtype=np.float32)
    tx_v = np.tile(v_tx.reshape(1, 1, 3), (num_snap, topology.num_tx, 1))
    rx_v = np.tile(v_rx.reshape(1, 1, 3), (num_snap, topology.num_rx, 1))

    if tx_orientation_rad_scene is not None:
        tx_orient = tx_orientation_rad_scene.reshape(1, topology.num_tx, 3)
        tx_o = np.tile(tx_orient, (num_snap, 1, 1)).astype(np.float32, copy=False)
    else:
        tx_orientation, _rx_orientation = _resolve_tx_rx_values(
            config.bs_orientation_rad,
            config.ue_orientation_rad,
            mapping,
        )
        tx_orient = np.array(tx_orientation, dtype=np.float32).reshape(1, 1, 3)
        tx_o = np.tile(tx_orient, (num_snap, topology.num_tx, 1))

    if rx_orientation_rad_scene is not None:
        rx_orient = rx_orientation_rad_scene.reshape(1, topology.num_rx, 3)
        rx_o = np.tile(rx_orient, (num_snap, 1, 1)).astype(np.float32, copy=False)
    else:
        _tx_orientation, rx_orientation = _resolve_tx_rx_values(
            config.bs_orientation_rad,
            config.ue_orientation_rad,
            mapping,
        )
        rx_orient = np.array(rx_orientation, dtype=np.float32).reshape(1, 1, 3)
        rx_o = np.tile(rx_orient, (num_snap, topology.num_rx, 1))

    return DeviceState(
        tx_velocity_mps=tx_v,
        rx_velocity_mps=rx_v,
        tx_orientation_rad=tx_o,
        rx_orientation_rad=rx_o,
    )


def _resolve_tx_rx_values(
    bs_value,
    ue_value,
    mapping: LinkRoleMapping,
):
    return resolve_role_pair(bs_value=bs_value, ue_value=ue_value, mapping=mapping)


def _build_resolved_antenna(
    config: RTTruthRunConfig,
    mapping: LinkRoleMapping,
) -> AntennaSpec:
    (tx_rows, rx_rows) = _resolve_tx_rx_values(
        config.bs_num_rows,
        config.ue_num_rows,
        mapping,
    )
    (tx_cols, rx_cols) = _resolve_tx_rx_values(
        config.bs_num_cols,
        config.ue_num_cols,
        mapping,
    )
    (tx_pol, rx_pol) = _resolve_tx_rx_values(
        config.bs_polarization,
        config.ue_polarization,
        mapping,
    )
    (tx_spacing, rx_spacing) = _resolve_tx_rx_values(
        config.bs_spacing_lambda,
        config.ue_spacing_lambda,
        mapping,
    )
    (tx_pattern, rx_pattern) = _resolve_tx_rx_values(
        config.bs_pattern,
        config.ue_pattern,
        mapping,
    )
    (tx_orientation_mode, rx_orientation_mode) = _resolve_tx_rx_values(
        config.bs_orientation_mode,
        config.ue_orientation_mode,
        mapping,
    )
    (tx_orientation, rx_orientation) = _resolve_tx_rx_values(
        config.bs_orientation_rad,
        config.ue_orientation_rad,
        mapping,
    )
    return AntennaSpec(
        tx_num_rows=int(tx_rows),
        tx_num_cols=int(tx_cols),
        rx_num_rows=int(rx_rows),
        rx_num_cols=int(rx_cols),
        tx_polarization=str(tx_pol),
        rx_polarization=str(rx_pol),
        tx_spacing_lambda=tuple(float(v) for v in tx_spacing),
        rx_spacing_lambda=tuple(float(v) for v in rx_spacing),
        tx_pattern=str(tx_pattern),
        rx_pattern=str(rx_pattern),
        tx_orientation_mode=str(tx_orientation_mode),
        tx_orientation_rad=tuple(float(v) for v in tx_orientation),
        rx_orientation_mode=str(rx_orientation_mode),
        rx_orientation_rad=tuple(float(v) for v in rx_orientation),
        synthetic_array=bool(config.synthetic_array),
    )


def _build_motion_spec(config: RTTruthRunConfig) -> MotionSpec | None:
    if config.num_time_steps <= 1 and config.sampling_frequency_hz <= 0:
        return None
    return MotionSpec.doppler_synthetic(
        num_time_steps=max(config.num_time_steps, 1),
        sampling_frequency_hz=config.sampling_frequency_hz or 1.0,
    )


def _attach_phy_array_outputs(
    phy_extra: dict,
    derived,
    config,
    truth_cfr: np.ndarray,
    cfr_est: np.ndarray | None,
    *,
    rx_orientation_rad: np.ndarray | None = None,
) -> None:
    waveform_extras = phy_extra.get("waveform_extras") or {}
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
    aoa_2d = np.stack(
        (
            derived.first_path_aoa_zenith_rad,
            derived.first_path_aoa_azimuth_rad,
        ),
        axis=-1,
    ).astype(np.float32, copy=False)
    aoa = np.broadcast_to(aoa_2d[np.newaxis, ...], (num_snap, *aoa_2d.shape))
    resolved_antenna = _build_resolved_antenna(
        config,
        resolve_link_roles(config.link_config.phy_link_direction),
    )
    phy_extra["array_outputs"] = build_array_outputs_from_waveform(
        rx_grid,
        aoa_label_rad=aoa,
        spectrum_config=config.spectrum_config,
        rx_num_rows=resolved_antenna.rx_num_rows,
        rx_num_cols=resolved_antenna.rx_num_cols,
        rx_spacing_lambda=tuple(float(v) for v in resolved_antenna.rx_spacing_lambda),
        rx_orientation_rad=rx_orientation_rad,
        truth_spectrum_samples=project_cfr_to_ul_receiver_samples(truth_cfr),
        cfr_est_spectrum_samples=(
            project_cfr_to_ul_receiver_samples(cfr_est) if cfr_est is not None else None
        ),
    )


def _build_truth_array_outputs(
    config,
    derived,
    truth_cfr: np.ndarray,
    *,
    rx_orientation_rad: np.ndarray | None = None,
) -> dict:
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
            derived.first_path_aoa_zenith_rad,
            derived.first_path_aoa_azimuth_rad,
        ),
        axis=-1,
    ).astype(np.float32, copy=False)
    aoa = np.broadcast_to(aoa_2d[np.newaxis, ...], (link_shape[0], *aoa_2d.shape))
    angle_grid = build_angle_grid_rad(config.spectrum_config)
    labels, heatmap = build_aoa_heatmap_label(aoa, angle_grid, link_shape)
    outputs: dict[str, object] = {
        "aoa_label_rad": labels,
        "aoa_heatmap_label": heatmap,
        "angle_grid_rad": angle_grid,
        "spectrum_policy": config.spectrum_config.policy,
    }
    resolved_antenna = _build_resolved_antenna(
        config,
        resolve_link_roles(config.link_config.phy_link_direction),
    )
    outputs["spatial_spectrum_truth"] = build_bartlett_spectrum(
        samples,
        rx_num_rows=resolved_antenna.rx_num_rows,
        rx_num_cols=resolved_antenna.rx_num_cols,
        rx_spacing_lambda=tuple(float(v) for v in resolved_antenna.rx_spacing_lambda),
        rx_orientation_rad=rx_orientation_rad,
        config=config.spectrum_config,
    )
    return outputs


def _build_cfr_est_array_outputs(
    config,
    derived,
    cfr_est: np.ndarray,
    *,
    rx_orientation_rad: np.ndarray | None = None,
) -> dict:
    from sionna_measurement_sim.phy.spatial_spectrum import (
        build_angle_grid_rad,
        build_aoa_heatmap_label,
        build_bartlett_spectrum,
        project_cfr_to_ul_receiver_samples,
    )

    samples = project_cfr_to_ul_receiver_samples(cfr_est)
    link_shape = samples.shape[:3]
    aoa_2d = np.stack(
        (
            derived.first_path_aoa_zenith_rad,
            derived.first_path_aoa_azimuth_rad,
        ),
        axis=-1,
    ).astype(np.float32, copy=False)
    aoa = np.broadcast_to(aoa_2d[np.newaxis, ...], (link_shape[0], *aoa_2d.shape))
    angle_grid = build_angle_grid_rad(config.spectrum_config)
    labels, heatmap = build_aoa_heatmap_label(aoa, angle_grid, link_shape)
    outputs: dict[str, object] = {
        "aoa_label_rad": labels,
        "aoa_heatmap_label": heatmap,
        "angle_grid_rad": angle_grid,
        "spectrum_policy": config.spectrum_config.policy,
    }
    resolved_antenna = _build_resolved_antenna(
        config,
        resolve_link_roles(config.link_config.phy_link_direction),
    )
    outputs["spatial_spectrum_cfr_est"] = build_bartlett_spectrum(
        samples,
        rx_num_rows=resolved_antenna.rx_num_rows,
        rx_num_cols=resolved_antenna.rx_num_cols,
        rx_spacing_lambda=tuple(float(v) for v in resolved_antenna.rx_spacing_lambda),
        rx_orientation_rad=rx_orientation_rad,
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
        "output_profile": config.output_profile,
        "output_products": list(
            config.output_plan.products
            if config.output_plan is not None
            else config.output_products or ()
        ),
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
        "bs_num_rows": config.bs_num_rows,
        "bs_num_cols": config.bs_num_cols,
        "ue_num_rows": config.ue_num_rows,
        "ue_num_cols": config.ue_num_cols,
        "bs_spacing_lambda": list(config.bs_spacing_lambda),
        "ue_spacing_lambda": list(config.ue_spacing_lambda),
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
            "radio_map_mode": config.visualization_config.radio_map_mode,
            "radio_map_grid_resolution_m": (
                config.visualization_config.radio_map_grid_resolution_m
            ),
            "radio_map_show_samples": config.visualization_config.radio_map_show_samples,
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
        "bs_pattern": config.bs_pattern,
        "ue_pattern": config.ue_pattern,
        "bs_polarization": config.bs_polarization,
        "ue_polarization": config.ue_polarization,
        "bs_orientation_mode": config.bs_orientation_mode,
        "bs_orientation_rad": list(config.bs_orientation_rad),
        "ue_orientation_mode": config.ue_orientation_mode,
        "ue_orientation_rad": list(config.ue_orientation_rad),
        "num_time_steps": config.num_time_steps,
        "sampling_frequency_hz": config.sampling_frequency_hz,
        "bs_velocity_mps": list(config.bs_velocity_mps),
        "ue_velocity_mps": list(config.ue_velocity_mps),
        "seed": config.seed,
        "max_bs": config.max_bs,
        "max_ue": config.max_ue,
        "ebno_db": config.ebno_db,
        "observation_snr_db": config.observation_snr_db,
        "observation_seed": config.observation_seed,
        "phy_standard": config.phy_standard,
        "tx_power_dbm": config.tx_power_dbm,
        "power_config": _power_config_snapshot(config.power_config),
        "iq_config": _plain_config_snapshot(config.iq_config),
        "noncooperative": _plain_config_snapshot(config.noncooperative_config),
        "srs_config": _srs_config_snapshot(config.srs_config),
        "ranging": _ranging_config_snapshot(config.ranging_config),
        "link_config": {
            "duplex_mode": config.link_config.duplex_mode,
            "phy_link_direction": config.link_config.phy_link_direction,
            "tx_role": config.link_config.tx_role,
            "rx_role": config.link_config.rx_role,
        },
        "mimo_mode": config.mimo_mode,
        "channel_backend": config.channel_backend,
        "mimo_detector": config.mimo_detector,
        "channel_estimator": config.channel_estimator,
        "receiver_failure_policy": config.receiver_failure_policy,
        "su_mimo_link_batch_size": config.su_mimo_link_batch_size,
        "hdf5_compression": config.hdf5_compression,
        "hdf5_gzip_level": config.hdf5_gzip_level,
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
            "axis": str(getattr(config.output_sharding_config, "axis", "ue")),
            "shard_size": int(
                getattr(config.output_sharding_config, "shard_size", config.max_ue)
            ),
            "filename_pattern": str(
                getattr(
                    config.output_sharding_config,
                    "filename_pattern",
                    "result_{shard_index:03d}.h5",
                )
            ),
            "results_dir": str(
                getattr(config.output_sharding_config, "results_dir", "results")
            ),
            "manifest_dir": str(
                getattr(config.output_sharding_config, "manifest_dir", "manifest")
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
            "fallback": {
                "enabled": bool(
                    getattr(
                        getattr(config.output_sharding_config, "fallback", None),
                        "enabled",
                        False,
                    )
                ),
                "min_shard_size": int(
                    getattr(
                        getattr(config.output_sharding_config, "fallback", None),
                        "min_shard_size",
                        1,
                    )
                ),
                "split_factor": int(
                    getattr(
                        getattr(config.output_sharding_config, "fallback", None),
                        "split_factor",
                        2,
                    )
                ),
                "retry_errors": list(
                    getattr(
                        getattr(config.output_sharding_config, "fallback", None),
                        "retry_errors",
                        [],
                    )
                ),
            },
        },
    }


def _input_dataset_id(label_file: Path) -> str:
    label_dir = label_file.parent
    if label_dir.name == "label":
        return label_dir.parent.as_posix()
    return label_dir.as_posix()


def _plain_config_snapshot(config: Any | None) -> dict[str, object]:
    if config is None:
        return {}
    if hasattr(config, "model_dump"):
        return dict(config.model_dump(mode="json"))
    if hasattr(config, "__dict__"):
        snapshot: dict[str, object] = {}
        for key, value in vars(config).items():
            if hasattr(value, "model_dump"):
                snapshot[key] = dict(value.model_dump(mode="json"))
            elif hasattr(value, "__dict__"):
                snapshot[key] = dict(vars(value))
            else:
                snapshot[key] = value
        return snapshot
    return {}


def _ranging_config_snapshot(config: RangingConfig) -> dict[str, object]:
    return {
        "enabled": config.enabled,
        "source": config.source,
        "estimators": list(config.estimators),
        "default_estimator": config.default_estimator,
        "write_rtt_equivalent": config.write_rtt_equivalent,
        "pdp_peak": {
            "oversampling_factor": config.pdp_peak.oversampling_factor,
            "window": config.pdp_peak.window,
            "peak_policy": config.pdp_peak.peak_policy,
            "relative_threshold_db": config.pdp_peak.relative_threshold_db,
            "min_peak_snr_db": config.pdp_peak.min_peak_snr_db,
            "interpolation": config.pdp_peak.interpolation,
            "max_delay_s": config.pdp_peak.max_delay_s,
        },
        "phase_slope": {
            "unwrap": config.phase_slope.unwrap,
            "aggregate": config.phase_slope.aggregate,
            "min_mean_power": config.phase_slope.min_mean_power,
        },
    }


def _power_config_snapshot(config: Any | None) -> dict[str, object]:
    if config is None:
        return {
            "reference_tx_power_dbm": 0.0,
            "apply_tx_power_to_grid": True,
            "noise_mode": "relative_snr",
            "thermal_noise": {
                "temperature_k": 290.0,
                "noise_figure_db": 7.0,
                "bandwidth_hz": None,
            },
            "uplink_control": {
                "enabled": False,
                "serving_rx_policy": "strongest_path",
                "open_loop_enabled": True,
                "p0_dbm": 0.0,
                "alpha": 0.8,
                "closed_loop_enabled": False,
                "tpc_offset_db": 0.0,
                "accumulation_db": 0.0,
                "min_tx_power_dbm": -40.0,
                "max_tx_power_dbm": 23.0,
            },
        }
    if hasattr(config, "model_dump"):
        return dict(config.model_dump())
    if hasattr(config, "__dict__"):
        snapshot: dict[str, object] = {}
        for key, value in vars(config).items():
            if hasattr(value, "model_dump"):
                snapshot[key] = dict(value.model_dump())
            elif hasattr(value, "__dict__"):
                snapshot[key] = dict(vars(value))
            else:
                snapshot[key] = value
        return snapshot
    return {}


def _srs_config_snapshot(config: Any | None) -> dict[str, object]:
    if config is None:
        return {}
    if hasattr(config, "model_dump"):
        return dict(config.model_dump())
    fields = (
        "slot_length_symbols",
        "start_symbol",
        "num_srs_symbols",
        "comb_size",
        "comb_offset",
        "bwp_start_prb",
        "bwp_num_prb",
        "trigger_mode",
        "periodicity_slots",
        "slot_offset",
        "slot_number",
        "sequence_type",
        "sequence_id",
        "group_hopping",
        "sequence_hopping",
        "cyclic_shift_multiplexing",
        "cyclic_shift_indices",
        "hopping",
        "ports",
        "power_control",
        "multiuser",
    )
    snapshot: dict[str, object] = {}
    for field_name in fields:
        if not hasattr(config, field_name):
            continue
        value = getattr(config, field_name)
        if hasattr(value, "model_dump"):
            snapshot[field_name] = dict(value.model_dump())
        elif hasattr(value, "__dict__"):
            snapshot[field_name] = dict(vars(value))
        else:
            snapshot[field_name] = value
    return snapshot


def _manifest_ranging_summary(ranging: object) -> dict[str, object]:
    summary = {"default_estimator": str(ranging.default_estimator)}
    for name in ("pdp_peak", "phase_slope"):
        result = getattr(ranging, name, None)
        if result is None:
            continue
        success = np.asarray(result.detection_success, dtype=np.bool_)
        abs_error = np.abs(np.asarray(result.range_error_m, dtype=np.float32))
        finite_error = np.isfinite(abs_error) & success
        summary[name] = {
            "finite_rate": float(np.mean(finite_error)) if finite_error.size else 0.0,
            "mean_abs_range_error_m": (
                float(np.mean(abs_error[finite_error])) if np.any(finite_error) else float("nan")
            ),
            "median_abs_range_error_m": (
                float(np.median(abs_error[finite_error])) if np.any(finite_error) else float("nan")
            ),
        }
    return summary


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
