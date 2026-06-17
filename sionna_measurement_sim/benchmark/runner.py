"""Reusable RT/write/spectrum benchmark implementations."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from sionna_measurement_sim.config.schema import OutputBundleConfig, OutputShardingConfig
from sionna_measurement_sim.domain.antenna import AntennaSpec
from sionna_measurement_sim.domain.array import ArraySpectrumConfig
from sionna_measurement_sim.domain.channel import RTTruthResult
from sionna_measurement_sim.domain.frequency import FrequencyGrid
from sionna_measurement_sim.domain.link import LinkConfig
from sionna_measurement_sim.domain.observation import (
    EvaluationResult,
    ImpairmentSpec,
    ObservationResult,
    ReceiverSpec,
    WaveformSpec,
)
from sionna_measurement_sim.domain.path import PathSamples
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
from sionna_measurement_sim.io.hdf5_bundle_writer import HDF5ResultBundleWriter
from sionna_measurement_sim.io.hdf5_reader import iter_manifest_dataset
from sionna_measurement_sim.io.hdf5_writer import write_measurement_result
from sionna_measurement_sim.io.schema_validator import validate_hdf5_contract
from sionna_measurement_sim.perf import PerfTracer
from sionna_measurement_sim.phy.spatial_spectrum import build_bartlett_spectrum
from sionna_measurement_sim.preflight.system import collect_basic_environment


@dataclass(frozen=True)
class BenchmarkDebugConfig:
    """Minimal debug config shape consumed by `PerfTracer`."""

    enabled: bool = True
    hardware_interval_s: float = 1.0
    link_log_interval: int = 250
    torch_synchronize: bool = True
    write_hardware_samples: bool = True


@dataclass(frozen=True)
class BenchmarkOptions:
    """Common benchmark options."""

    output_dir: Path
    seed: int = 1
    repeat: int = 1
    warmup: int = 0
    device: str = "cpu"
    debug_hardware_interval_s: float = 1.0
    write_hardware_samples: bool = True
    summary_name: str = "benchmark_summary"

    def __post_init__(self) -> None:
        if self.repeat < 1:
            msg = "repeat must be >= 1"
            raise ValueError(msg)
        if self.warmup < 0:
            msg = "warmup must be >= 0"
            raise ValueError(msg)


def run_rt_benchmark(options: BenchmarkOptions, parameters: dict[str, Any]) -> Path:
    """Run RT-only benchmark iterations and write JSON/CSV artifacts."""

    from dataclasses import replace

    from sionna_measurement_sim.adapters.sionna_rt.rt_solver import (
        SionnaRTConfig,
        run_sionna_rt_truth,
    )
    from sionna_measurement_sim.config.loader import load_config_or_exit
    from sionna_measurement_sim.domain.topology import (
        resolve_link_roles,
        resolve_role_topology,
    )
    from sionna_measurement_sim.io.label_parser import load_role_topology_from_label
    from sionna_measurement_sim.rt.truth_pipeline import (
        RTTruthRunConfig,
        _build_resolved_antenna,
        _resolve_tx_rx_values,
    )

    output_dir = options.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config = _rt_run_config_from_parameters(options, parameters)
    config_path = parameters.get("config")
    if config_path:
        cfg = load_config_or_exit(str(config_path))
        run_config = RTTruthRunConfig(
            label_file=Path(parameters.get("label_file") or cfg.input.label_file),
            scene_file=Path(parameters.get("scene_file") or cfg.input.scene_file),
            output_dir=output_dir,
            scene_id=cfg.input.scene_id,
            map_id=cfg.input.map_id,
            center_frequency_hz=cfg.carrier.center_frequency_hz,
            bandwidth_hz=cfg.carrier.bandwidth_hz,
            num_subcarriers=int(
                parameters.get("num_subcarriers") or cfg.carrier.num_subcarriers
            ),
            seed=options.seed,
            device=options.device or cfg.runtime.device,
            max_bs=int(parameters.get("max_bs") or cfg.input.max_bs),
            max_ue=int(parameters.get("max_ue") or cfg.input.max_ue),
            max_depth=int(parameters.get("max_depth") or cfg.rt.max_depth),
            los=_optional_bool(parameters.get("los"), cfg.rt.los),
            specular_reflection=_optional_bool(
                parameters.get("specular_reflection"),
                cfg.rt.specular_reflection,
            ),
            diffuse_reflection=_optional_bool(
                parameters.get("diffuse_reflection"),
                cfg.rt.diffuse_reflection,
            ),
            refraction=_optional_bool(parameters.get("refraction"), cfg.rt.refraction),
            diffraction=_optional_bool(parameters.get("diffraction"), cfg.rt.diffraction),
            synthetic_array=_optional_bool(
                parameters.get("synthetic_array"),
                cfg.rt.synthetic_array,
            ),
            merge_shapes=cfg.rt.merge_shapes,
            normalize_cfr=cfg.rt.normalize_cfr,
            normalize_delays=cfg.rt.normalize_delays,
            bs_num_rows=cfg.antenna.bs_array.num_rows,
            bs_num_cols=cfg.antenna.bs_array.num_cols,
            ue_num_rows=cfg.antenna.ue_array.num_rows,
            ue_num_cols=cfg.antenna.ue_array.num_cols,
            bs_polarization=cfg.antenna.bs_array.polarization,
            ue_polarization=cfg.antenna.ue_array.polarization,
            bs_pattern=cfg.antenna.bs_array.pattern,
            ue_pattern=cfg.antenna.ue_array.pattern,
            bs_orientation_mode=cfg.antenna.bs_array.orientation_mode,
            bs_orientation_rad=tuple(cfg.antenna.bs_array.orientation_rad),
            ue_orientation_mode=cfg.antenna.ue_array.orientation_mode,
            ue_orientation_rad=tuple(cfg.antenna.ue_array.orientation_rad),
            bs_spacing_lambda=(
                cfg.antenna.bs_array.vertical_spacing_lambda,
                cfg.antenna.bs_array.horizontal_spacing_lambda,
            ),
            ue_spacing_lambda=(
                cfg.antenna.ue_array.vertical_spacing_lambda,
                cfg.antenna.ue_array.horizontal_spacing_lambda,
            ),
            link_config=LinkConfig(
                duplex_mode=cfg.link.duplex_mode,
                phy_link_direction=cfg.link.phy_link_direction,
            ),
        )

    tracer = _start_tracer(options, "rt")
    rows: list[dict[str, Any]] = []
    status = "success"
    exception: BaseException | None = None
    try:
        for iteration, is_warmup in _iteration_plan(options):
            iter_config = replace(run_config, seed=options.seed + iteration)
            row: dict[str, Any] = {
                "iteration": iteration,
                "warmup": is_warmup,
                "benchmark_type": "rt",
            }
            iter_start = time.perf_counter()
            mapping = resolve_link_roles(iter_config.link_config.phy_link_direction)
            with tracer.span("rt.topology_load", iteration=iteration, warmup=is_warmup):
                role_topology = load_role_topology_from_label(
                    iter_config.label_file,
                    max_bs=iter_config.max_bs,
                    max_ue=iter_config.max_ue,
                )
                topology = resolve_role_topology(role_topology, mapping)
            antenna = _build_resolved_antenna(iter_config, mapping)
            frequency = FrequencyGrid.from_center_bandwidth(
                iter_config.center_frequency_hz,
                iter_config.bandwidth_hz,
                iter_config.num_subcarriers,
            )
            with tracer.span("rt.rt_solve", iteration=iteration, warmup=is_warmup):
                adapter_result = run_sionna_rt_truth(
                    topology=topology,
                    antenna=antenna,
                    frequency=frequency,
                    config=SionnaRTConfig(
                        scene_file=iter_config.scene_file,
                        seed=iter_config.seed,
                        max_depth=iter_config.max_depth,
                        los=iter_config.los,
                        specular_reflection=iter_config.specular_reflection,
                        diffuse_reflection=iter_config.diffuse_reflection,
                        refraction=iter_config.refraction,
                        diffraction=iter_config.diffraction,
                        synthetic_array=iter_config.synthetic_array,
                        normalize_cfr=iter_config.normalize_cfr,
                        normalize_delays=iter_config.normalize_delays,
                        num_time_steps=iter_config.num_time_steps,
                        sampling_frequency_hz=iter_config.sampling_frequency_hz,
                        tx_velocity=_resolve_tx_rx_values(
                            iter_config.bs_velocity_mps,
                            iter_config.ue_velocity_mps,
                            mapping,
                        )[0],
                        rx_velocity=_resolve_tx_rx_values(
                            iter_config.bs_velocity_mps,
                            iter_config.ue_velocity_mps,
                            mapping,
                        )[1],
                        merge_shapes=iter_config.merge_shapes,
                    ),
                )
            row.update(
                {
                    "status": "success",
                    "wall_time_s": time.perf_counter() - iter_start,
                    "rt_solve_s": _stage_duration(
                        tracer, "rt.rt_solve", iteration=iteration
                    ),
                    "tx_count": topology.num_tx,
                    "rx_count": topology.num_rx,
                    "path_count": int(adapter_result.path_samples.path_count.sum()),
                    "los_rate": float(np.mean(adapter_result.truth.los_exists)),
                    "nlos_rate": float(np.mean(adapter_result.truth.nlos_exists)),
                    "truth_cfr_shape": list(adapter_result.truth.cfr.shape),
                    "truth_cfr_bytes": int(adapter_result.truth.cfr.nbytes),
                }
            )
            rows.append(row)
    except Exception as exc:  # pragma: no cover - exercised through callers
        status = "failed"
        exception = exc
        raise
    finally:
        perf_summary = tracer.finish(
            {"benchmark_type": "rt"},
            status=status,
            exception=exception,
        )
        _write_benchmark_outputs(
            output_dir=output_dir,
            summary_name=options.summary_name,
            benchmark_type="rt",
            status=status,
            parameters={**parameters, **_options_snapshot(options)},
            rows=rows,
            perf_summary=perf_summary,
            artifacts={},
        )
    return output_dir / f"{options.summary_name}.json"


def run_write_benchmark(options: BenchmarkOptions, parameters: dict[str, Any]) -> Path:
    """Run synthetic HDF5 writer benchmark iterations."""

    output_dir = options.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tracer = _start_tracer(options, "write")
    rows: list[dict[str, Any]] = []
    status = "success"
    exception: BaseException | None = None
    try:
        for iteration, is_warmup in _iteration_plan(options):
            if int(parameters.get("bundle_shards", 0)) > 0:
                rows.extend(
                    _run_write_bundle_comparison_iteration(
                        output_dir,
                        tracer,
                        options=options,
                        parameters=parameters,
                        iteration=iteration,
                        is_warmup=is_warmup,
                    )
                )
                continue
            rows.append(
                _run_write_single_iteration(
                    output_dir,
                    tracer,
                    options=options,
                    parameters=parameters,
                    iteration=iteration,
                    is_warmup=is_warmup,
                )
            )
    except Exception as exc:  # pragma: no cover - exercised through callers
        status = "failed"
        exception = exc
        raise
    finally:
        perf_summary = tracer.finish(
            {"benchmark_type": "write"},
            status=status,
            exception=exception,
        )
        _write_benchmark_outputs(
            output_dir=output_dir,
            summary_name=options.summary_name,
            benchmark_type="write",
            status=status,
            parameters={**parameters, **_options_snapshot(options)},
            rows=rows,
            perf_summary=perf_summary,
            artifacts={"output_dir": output_dir.as_posix()},
        )
    return output_dir / f"{options.summary_name}.json"


def run_sharding_benchmark(options: BenchmarkOptions, parameters: dict[str, Any]) -> Path:
    """Run real sharded pipeline outputs as shard files and appendable bundles."""

    output_dir = options.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tracer = _start_tracer(options, "sharding")
    rows: list[dict[str, Any]] = []
    status = "success"
    exception: BaseException | None = None
    try:
        for iteration, is_warmup in _iteration_plan(options):
            for write_mode in ("shard_files", "bundle_append"):
                rows.append(
                    _run_sharding_comparison_iteration(
                        output_dir,
                        tracer,
                        options=options,
                        parameters=parameters,
                        iteration=iteration,
                        is_warmup=is_warmup,
                        write_mode=write_mode,
                    )
                )
    except Exception as exc:  # pragma: no cover - exercised through callers
        status = "failed"
        exception = exc
        raise
    finally:
        perf_summary = tracer.finish(
            {"benchmark_type": "sharding"},
            status=status,
            exception=exception,
        )
        _write_benchmark_outputs(
            output_dir=output_dir,
            summary_name=options.summary_name,
            benchmark_type="sharding",
            status=status,
            parameters={**parameters, **_options_snapshot(options)},
            rows=rows,
            perf_summary=perf_summary,
            artifacts={"output_dir": output_dir.as_posix()},
        )
    return output_dir / f"{options.summary_name}.json"


def _run_sharding_comparison_iteration(
    output_dir: Path,
    tracer: PerfTracer,
    *,
    options: BenchmarkOptions,
    parameters: dict[str, Any],
    iteration: int,
    is_warmup: bool,
    write_mode: str,
) -> dict[str, Any]:
    from sionna_measurement_sim.rt.truth_pipeline import run_rt_truth_pipeline

    run_dir = output_dir / f"sharding_iter_{iteration:03d}_{write_mode}"
    config = _sharding_run_config(
        options,
        parameters,
        output_dir=run_dir,
        seed=options.seed + iteration,
        bundle_enabled=write_mode == "bundle_append",
    )
    row: dict[str, Any] = {
        "iteration": iteration,
        "warmup": is_warmup,
        "benchmark_type": "sharding",
        "write_mode": write_mode,
    }
    iter_start = time.perf_counter()
    with tracer.span(
        "sharding.pipeline_run",
        iteration=iteration,
        warmup=is_warmup,
        write_mode=write_mode,
    ):
        run_rt_truth_pipeline(config)
    manifest_path = run_dir / "manifest" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_paths = _sharding_artifact_paths(manifest)
    stage_totals = dict(manifest.get("performance", {}).get("stage_totals_s", {}))
    dataset_write_summary = dict(
        manifest.get("performance", {}).get("dataset_write_summary", {})
    )
    readback_summary = _measure_sharding_readback(
        tracer,
        manifest_path,
        dataset_path=str(parameters.get("readback_dataset") or "channel/truth/cfr"),
        iteration=iteration,
        is_warmup=is_warmup,
        write_mode=write_mode,
    )
    row.update(
        {
            "status": "success",
            "wall_time_s": time.perf_counter() - iter_start,
            "pipeline_stage_s": _stage_duration(
                tracer,
                "sharding.pipeline_run",
                iteration=iteration,
            ),
            "rt_solve_s": float(stage_totals.get("rt_solve", 0.0)),
            "hdf5_write_s": float(stage_totals.get("hdf5_write", 0.0)),
            "hdf5_bundle_write_s": float(stage_totals.get("hdf5_bundle_write", 0.0)),
            "hdf5_bundle_append_s": float(stage_totals.get("hdf5_bundle_append", 0.0)),
            "schema_validate_s": float(stage_totals.get("schema_validate", 0.0)),
            "planned_shard_count": int(
                manifest.get("sharding", {}).get("planned_shard_count", 0)
            ),
            "fragment_count": len(manifest.get("results", [])),
            "file_count": len(artifact_paths),
            "file_size_bytes": sum(path.stat().st_size for path in artifact_paths),
            "dataset_write_count": int(dataset_write_summary.get("dataset_count", 0)),
            "dataset_raw_bytes": int(dataset_write_summary.get("total_raw_bytes", 0)),
            "dataset_storage_bytes": int(
                dataset_write_summary.get("total_storage_bytes", 0)
            ),
            **readback_summary,
            "result_dir": run_dir.as_posix(),
            "manifest_path": manifest_path.as_posix(),
            "artifact_paths": [path.as_posix() for path in artifact_paths],
        }
    )
    if write_mode == "bundle_append":
        row["bundle_count"] = len(manifest.get("bundles", []))
        row["bundle_max_planned_shards"] = int(
            parameters.get("bundle_max_planned_shards", 2)
        )
    return row


def _sharding_run_config(
    options: BenchmarkOptions,
    parameters: dict[str, Any],
    *,
    output_dir: Path,
    seed: int,
    bundle_enabled: bool,
):
    from sionna_measurement_sim.rt.truth_pipeline import RTTruthRunConfig

    default_label_file = Path("tests/fixtures/scenes/test/test5.json")
    default_scene_file = Path("tests/fixtures/scenes/test/scene.xml")

    return RTTruthRunConfig(
        label_file=Path(parameters.get("label_file") or default_label_file),
        scene_file=Path(parameters.get("scene_file") or default_scene_file),
        output_dir=output_dir,
        num_subcarriers=int(parameters.get("num_subcarriers") or 8),
        seed=seed,
        device=options.device,
        max_depth=int(parameters.get("max_depth") or 1),
        max_bs=int(parameters.get("max_bs") or 1),
        max_ue=int(parameters.get("max_ue") or 3),
        output_products=("cfr_truth",),
        hdf5_compression=str(parameters.get("compression", "mixed")),
        hdf5_gzip_level=int(parameters.get("gzip_level", 1)),
        debug_config=BenchmarkDebugConfig(
            hardware_interval_s=options.debug_hardware_interval_s,
            write_hardware_samples=options.write_hardware_samples,
        ),
        output_sharding_config=OutputShardingConfig(
            enabled=True,
            shard_size=int(parameters.get("shard_size") or 1),
            parallel_workers=int(parameters.get("parallel_workers") or 1),
            visualization_mode="none",
            bundle=OutputBundleConfig(
                enabled=bundle_enabled,
                max_planned_shards_per_bundle=max(
                    int(parameters.get("bundle_max_planned_shards") or 2),
                    1,
                ),
                validate_schema=True,
            ),
        ),
    )


def _sharding_artifact_paths(manifest: dict[str, Any]) -> list[Path]:
    if manifest.get("phase") == "bundled_sharded_run_full":
        paths = [Path(str(item["bundle_h5"])) for item in manifest.get("bundles", [])]
    else:
        paths = [Path(str(item["result_h5"])) for item in manifest.get("results", [])]
    return sorted(set(paths), key=lambda path: path.as_posix())


def _measure_sharding_readback(
    tracer: PerfTracer,
    manifest_path: Path,
    *,
    dataset_path: str,
    iteration: int,
    is_warmup: bool,
    write_mode: str,
) -> dict[str, Any]:
    start = time.perf_counter()
    bytes_read = 0
    fragment_count = 0
    finite_rates: list[float] = []
    global_ue_count = 0
    with tracer.span(
        "sharding.dataset_readback",
        iteration=iteration,
        warmup=is_warmup,
        write_mode=write_mode,
        dataset_path=dataset_path,
    ):
        for fragment in iter_manifest_dataset(manifest_path, dataset_path):
            array = np.asarray(fragment.data)
            fragment_count += 1
            bytes_read += int(array.nbytes)
            global_ue_count += len(fragment.global_ue_indices)
            if array.size:
                finite_rates.append(float(np.mean(np.isfinite(array))))
    return {
        "readback_dataset": dataset_path,
        "readback_s": time.perf_counter() - start,
        "readback_fragment_count": fragment_count,
        "readback_bytes": bytes_read,
        "readback_global_ue_count": global_ue_count,
        "readback_finite_rate_min": min(finite_rates) if finite_rates else 1.0,
    }


def _run_write_single_iteration(
    output_dir: Path,
    tracer: PerfTracer,
    *,
    options: BenchmarkOptions,
    parameters: dict[str, Any],
    iteration: int,
    is_warmup: bool,
) -> dict[str, Any]:
    result = _build_synthetic_write_result(options, parameters, seed=options.seed + iteration)
    h5_path = output_dir / f"write_iter_{iteration:03d}.h5"
    row = {
        "iteration": iteration,
        "warmup": is_warmup,
        "benchmark_type": "write",
        "write_mode": "single_file",
    }
    iter_start = time.perf_counter()
    with tracer.span("write.hdf5_write", iteration=iteration, warmup=is_warmup):
        write_start = time.perf_counter()
        write_measurement_result(
            h5_path,
            result,
            compression=str(parameters["compression"]),
            gzip_level=int(parameters.get("gzip_level", 4)),
            tracer=tracer,
        )
        writer_s = time.perf_counter() - write_start
    schema_validate_s = 0.0
    if parameters["validate_schema"]:
        validate_start = time.perf_counter()
        with tracer.span(
            "write.schema_validate",
            iteration=iteration,
            warmup=is_warmup,
        ):
            validate_hdf5_contract(h5_path)
        schema_validate_s = time.perf_counter() - validate_start
    row.update(
        {
            "status": "success",
            "wall_time_s": time.perf_counter() - iter_start,
            "writer_s": writer_s,
            "schema_validate_s": schema_validate_s,
            "file_size_bytes": h5_path.stat().st_size,
            "file_count": 1,
            "fragment_count": 1,
            "ue_count": int(parameters["tx_count"]),
            "path": h5_path.as_posix(),
        }
    )
    return row


def _run_write_bundle_comparison_iteration(
    output_dir: Path,
    tracer: PerfTracer,
    *,
    options: BenchmarkOptions,
    parameters: dict[str, Any],
    iteration: int,
    is_warmup: bool,
) -> list[dict[str, Any]]:
    shard_count = int(parameters["bundle_shards"])
    if shard_count < 1:
        msg = "bundle_shards must be >= 1 when bundle comparison is enabled"
        raise ValueError(msg)
    tx_count = int(parameters["tx_count"])
    fragments = [
        _with_synthetic_shard_metadata(
            _build_synthetic_write_result(
                options,
                parameters,
                seed=options.seed + iteration * 1000 + shard_index,
            ),
            shard_index=shard_index,
            shard_count=shard_count,
            tx_count=tx_count,
            rx_count=int(parameters["rx_count"]),
        )
        for shard_index in range(shard_count)
    ]
    shard_specs = [
        ShardSpec(
            shard_index=shard_index,
            shard_count=shard_count,
            axis="ue",
            ue_start=shard_index * tx_count,
            ue_count=tx_count,
        )
        for shard_index in range(shard_count)
    ]
    shard_row = _write_synthetic_shard_files(
        output_dir,
        tracer,
        parameters=parameters,
        fragments=fragments,
        iteration=iteration,
        is_warmup=is_warmup,
    )
    bundle_row = _write_synthetic_bundles(
        output_dir,
        tracer,
        parameters=parameters,
        fragments=fragments,
        shard_specs=shard_specs,
        iteration=iteration,
        is_warmup=is_warmup,
    )
    return [shard_row, bundle_row]


def _write_synthetic_shard_files(
    output_dir: Path,
    tracer: PerfTracer,
    *,
    parameters: dict[str, Any],
    fragments: list[MeasurementSimulationResult],
    iteration: int,
    is_warmup: bool,
) -> dict[str, Any]:
    iter_dir = output_dir / f"write_iter_{iteration:03d}_shards"
    iter_dir.mkdir(parents=True, exist_ok=True)
    row = _write_comparison_row(
        iteration=iteration,
        is_warmup=is_warmup,
        write_mode="shard_files",
        fragment_count=len(fragments),
        ue_count=len(fragments) * int(parameters["tx_count"]),
    )
    iter_start = time.perf_counter()
    paths: list[str] = []
    with tracer.span("write.shards_hdf5_write", iteration=iteration, warmup=is_warmup):
        write_start = time.perf_counter()
        for shard_index, result in enumerate(fragments):
            path = iter_dir / f"result_{shard_index:03d}.h5"
            write_measurement_result(
                path,
                result,
                compression=str(parameters["compression"]),
                gzip_level=int(parameters.get("gzip_level", 4)),
                tracer=tracer,
            )
            paths.append(path.as_posix())
        writer_s = time.perf_counter() - write_start
    schema_validate_s = 0.0
    if parameters["validate_schema"]:
        validate_start = time.perf_counter()
        with tracer.span(
            "write.shards_schema_validate",
            iteration=iteration,
            warmup=is_warmup,
        ):
            for path in paths:
                validate_hdf5_contract(path)
        schema_validate_s = time.perf_counter() - validate_start
    row.update(
        {
            "status": "success",
            "wall_time_s": time.perf_counter() - iter_start,
            "writer_s": writer_s,
            "schema_validate_s": schema_validate_s,
            "file_size_bytes": sum(Path(path).stat().st_size for path in paths),
            "file_count": len(paths),
            "path": iter_dir.as_posix(),
            "artifact_paths": paths,
        }
    )
    return row


def _write_synthetic_bundles(
    output_dir: Path,
    tracer: PerfTracer,
    *,
    parameters: dict[str, Any],
    fragments: list[MeasurementSimulationResult],
    shard_specs: list[ShardSpec],
    iteration: int,
    is_warmup: bool,
) -> dict[str, Any]:
    iter_dir = output_dir / f"write_iter_{iteration:03d}_bundles"
    iter_dir.mkdir(parents=True, exist_ok=True)
    max_planned = max(int(parameters.get("bundle_max_planned_shards", 10)), 1)
    row = _write_comparison_row(
        iteration=iteration,
        is_warmup=is_warmup,
        write_mode="bundle_append",
        fragment_count=len(fragments),
        ue_count=len(fragments) * int(parameters["tx_count"]),
    )
    iter_start = time.perf_counter()
    paths: list[str] = []
    with tracer.span("write.bundle_hdf5_write", iteration=iteration, warmup=is_warmup):
        write_start = time.perf_counter()
        for bundle_index, start in enumerate(range(0, len(fragments), max_planned)):
            bundle_path = iter_dir / f"bundle_{bundle_index:03d}.h5"
            with HDF5ResultBundleWriter(
                bundle_path,
                compression=str(parameters["compression"]),
                gzip_level=int(parameters.get("gzip_level", 4)),
                tracer=tracer,
            ) as writer:
                for result, spec in zip(
                    fragments[start : start + max_planned],
                    shard_specs[start : start + max_planned],
                    strict=True,
                ):
                    writer.append_result(result, shard_spec=spec)
            paths.append(bundle_path.as_posix())
        writer_s = time.perf_counter() - write_start
    schema_validate_s = 0.0
    if parameters["validate_schema"]:
        validate_start = time.perf_counter()
        with tracer.span(
            "write.bundle_schema_validate",
            iteration=iteration,
            warmup=is_warmup,
        ):
            for path in paths:
                validate_hdf5_contract(path)
        schema_validate_s = time.perf_counter() - validate_start
    row.update(
        {
            "status": "success",
            "wall_time_s": time.perf_counter() - iter_start,
            "writer_s": writer_s,
            "schema_validate_s": schema_validate_s,
            "file_size_bytes": sum(Path(path).stat().st_size for path in paths),
            "file_count": len(paths),
            "path": iter_dir.as_posix(),
            "artifact_paths": paths,
            "bundle_max_planned_shards": max_planned,
        }
    )
    return row


def _write_comparison_row(
    *,
    iteration: int,
    is_warmup: bool,
    write_mode: str,
    fragment_count: int,
    ue_count: int,
) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "warmup": is_warmup,
        "benchmark_type": "write",
        "write_mode": write_mode,
        "fragment_count": fragment_count,
        "ue_count": ue_count,
    }


def _build_synthetic_write_result(
    options: BenchmarkOptions,
    parameters: dict[str, Any],
    *,
    seed: int,
) -> MeasurementSimulationResult:
    return build_synthetic_measurement_result(
        seed=seed,
        tx_count=int(parameters["tx_count"]),
        rx_count=int(parameters["rx_count"]),
        rx_ant=int(parameters["rx_ant"]),
        tx_ant=int(parameters["tx_ant"]),
        subcarriers=int(parameters["subcarriers"]),
        snapshots=int(parameters["snapshots"]),
        include_waveform=bool(parameters["include_waveform"]),
        include_array=bool(parameters["include_array"]),
        include_ranging=bool(parameters["include_ranging"]),
    )


def _with_synthetic_shard_metadata(
    result: MeasurementSimulationResult,
    *,
    shard_index: int,
    shard_count: int,
    tx_count: int,
    rx_count: int,
) -> MeasurementSimulationResult:
    global_tx_start = shard_index * tx_count
    global_tx_indices = np.arange(
        global_tx_start,
        global_tx_start + tx_count,
        dtype=np.int64,
    )
    topology = replace(
        result.topology,
        tx_positions_m=result.topology.tx_positions_m
        + np.asarray([float(global_tx_start), 0.0, 0.0], dtype=np.float32),
        tx_labels=tuple(f"tx{index}" for index in global_tx_indices),
    )
    return replace(
        result,
        topology=topology,
        shard=ShardMetadata(
            shard_index=shard_index,
            shard_count=shard_count,
            axis="ue",
            global_rx_start=0,
            global_rx_indices=np.arange(rx_count, dtype=np.int64),
            global_tx_indices=global_tx_indices,
        ),
    )


def run_spectrum_benchmark(options: BenchmarkOptions, parameters: dict[str, Any]) -> Path:
    """Run synthetic Bartlett spatial-spectrum benchmark iterations."""

    output_dir = options.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    tracer = _start_tracer(options, "spectrum")
    rows: list[dict[str, Any]] = []
    status = "success"
    exception: BaseException | None = None
    try:
        for iteration, is_warmup in _iteration_plan(options):
            rng = np.random.default_rng(options.seed + iteration)
            samples_by_source = _synthetic_spectrum_samples(parameters, rng)
            config = ArraySpectrumConfig(
                enabled=True,
                sources=tuple(parameters["sources"]),
                zenith_bins=int(parameters["zenith_bins"]),
                azimuth_bins=int(parameters["azimuth_bins"]),
                link_chunk_size=int(parameters["link_chunk_size"]),
            )
            row = {
                "iteration": iteration,
                "warmup": is_warmup,
                "benchmark_type": "spectrum",
            }
            iter_start = time.perf_counter()
            total_output_bytes = 0
            finite_rates: list[float] = []
            for source, samples in samples_by_source.items():
                with tracer.span(
                    f"spectrum.{source}",
                    iteration=iteration,
                    warmup=is_warmup,
                ):
                    spectrum = build_bartlett_spectrum(
                        samples,
                        rx_num_rows=1,
                        rx_num_cols=int(parameters["rx_ant"]),
                        rx_spacing_lambda=(0.5, 0.5),
                        config=config,
                    )
                total_output_bytes += int(spectrum.nbytes)
                finite_rates.append(float(np.mean(np.isfinite(spectrum))))
                row[f"{source}_time_s"] = _stage_duration(
                    tracer,
                    f"spectrum.{source}",
                    iteration=iteration,
                )
                row[f"{source}_shape"] = list(spectrum.shape)
            row.update(
                {
                    "status": "success",
                    "wall_time_s": time.perf_counter() - iter_start,
                    "source_count": len(samples_by_source),
                    "output_bytes": total_output_bytes,
                    "finite_rate_min": min(finite_rates) if finite_rates else 1.0,
                    "chunk_count": int(
                        np.ceil(
                            int(parameters["tx_count"]) * int(parameters["rx_count"])
                            / int(parameters["link_chunk_size"])
                        )
                    ),
                }
            )
            rows.append(row)
    except Exception as exc:  # pragma: no cover - exercised through callers
        status = "failed"
        exception = exc
        raise
    finally:
        perf_summary = tracer.finish(
            {"benchmark_type": "spectrum"},
            status=status,
            exception=exception,
        )
        _write_benchmark_outputs(
            output_dir=output_dir,
            summary_name=options.summary_name,
            benchmark_type="spectrum",
            status=status,
            parameters={**parameters, **_options_snapshot(options)},
            rows=rows,
            perf_summary=perf_summary,
            artifacts={"output_dir": output_dir.as_posix()},
        )
    return output_dir / f"{options.summary_name}.json"


def build_synthetic_measurement_result(
    *,
    seed: int,
    tx_count: int,
    rx_count: int,
    rx_ant: int,
    tx_ant: int,
    subcarriers: int,
    snapshots: int,
    include_waveform: bool,
    include_array: bool,
    include_ranging: bool,
) -> MeasurementSimulationResult:
    """Create a deterministic representative result without RT/Sionna."""

    rng = np.random.default_rng(seed)
    topology = Topology(
        tx_positions_m=_positions(tx_count, 0.0),
        rx_positions_m=_positions(rx_count, 10.0),
        tx_labels=tuple(f"tx{i}" for i in range(tx_count)),
        rx_labels=tuple(f"rx{i}" for i in range(rx_count)),
    )
    antenna = AntennaSpec(
        tx_num_rows=1,
        tx_num_cols=tx_ant,
        rx_num_rows=1,
        rx_num_cols=rx_ant,
    )
    frequency = FrequencyGrid.from_center_bandwidth(
        center_frequency_hz=3.5e9,
        bandwidth_hz=20e6,
        num_subcarriers=subcarriers,
    )
    cfr = _complex_normal(rng, (tx_count, rx_count, rx_ant, tx_ant, subcarriers))
    link_shape = (tx_count, rx_count)
    result = MeasurementSimulationResult(
        metadata=Metadata(
            run_id="benchmark_write",
            random_seed=seed,
            config_snapshot=json.dumps({"benchmark": "write"}, sort_keys=True),
            observation_branch_enabled=include_waveform,
            measurement_realism_level="benchmark_synthetic",
        ),
        input_spec=InputSpec(label_file="", scene_file="", input_dataset_id="benchmark"),
        topology=topology,
        devices=DeviceState.static(snapshots=snapshots, tx=tx_count, rx=rx_count),
        antenna=antenna,
        scene=SceneSpec(scene_name="benchmark", scene_file="", scene_id="benchmark"),
        frequency=frequency,
        truth=RTTruthResult(
            cfr=cfr,
            path_power_db=np.full(link_shape, -60.0, dtype=np.float32),
            has_geometric_signal=np.ones(link_shape, dtype=np.bool_),
            geometric_path_count=np.ones(link_shape, dtype=np.int32),
            los_exists=np.ones(link_shape, dtype=np.bool_),
            nlos_exists=np.zeros(link_shape, dtype=np.bool_),
        ),
        path_samples=PathSamples.empty(),
        runtime=RuntimeInfo(command_line="benchmark write"),
        link=LinkConfig(),
    )
    if not include_waveform:
        return _with_optional_array_and_ranging(
            result,
            rng,
            include_array=include_array,
            include_ranging=False,
            snapshots=snapshots,
        )

    obs_shape = (snapshots, tx_count, rx_count, rx_ant, tx_ant, subcarriers)
    cfr_est = _complex_normal(rng, obs_shape)
    obs_link_shape = (snapshots, tx_count, rx_count)
    observation = ObservationResult(
        cfr_est=cfr_est,
        valid_mask=np.ones(obs_link_shape, dtype=np.bool_),
        detection_success=np.ones(obs_link_shape, dtype=np.bool_),
        estimation_success=np.ones(obs_link_shape, dtype=np.bool_),
        snr_db=np.full(obs_link_shape, 30.0, dtype=np.float32),
        rssi_dbm=np.full(obs_link_shape, -45.0, dtype=np.float32),
        noise_power_dbm=np.full(obs_link_shape, -95.0, dtype=np.float32),
        cfo_hz=np.zeros(obs_link_shape, dtype=np.float32),
        sfo_ppm=np.zeros(obs_link_shape, dtype=np.float32),
        timing_offset_samples=np.zeros(obs_link_shape, dtype=np.float32),
        phase_offset_rad=np.zeros(obs_link_shape, dtype=np.float32),
        agc_gain_db=np.zeros((snapshots, rx_count), dtype=np.float32),
        clipping_flag=np.zeros(obs_link_shape, dtype=np.bool_),
    )
    evaluation = EvaluationResult(
        nmse_db=np.zeros(obs_link_shape, dtype=np.float32),
        nmse_db_total=np.zeros(obs_link_shape, dtype=np.float32),
        amplitude_error_db=np.zeros(obs_link_shape, dtype=np.float32),
        phase_error_rad=np.zeros(obs_link_shape, dtype=np.float32),
        correlation=np.ones(obs_link_shape, dtype=np.float32),
        detection_rate=1.0,
        estimation_failure_rate=0.0,
    )
    waveform = WaveformSpec(
        standard="custom_ofdm",
        sample_rate_hz=20e6,
        fft_size=subcarriers,
        cp_length=0,
        num_ofdm_symbols=1,
        pilot_indices=np.arange(subcarriers, dtype=np.int32),
        data_subcarrier_indices=np.zeros((0,), dtype=np.int32),
        pilot_symbols=np.ones((subcarriers,), dtype=np.complex64),
        tx_power_dbm=0.0,
    )
    result = MeasurementSimulationResult(
        **{
            **result.__dict__,
            "waveform": waveform,
            "observation": observation,
            "impairments": ImpairmentSpec(
                model_version="benchmark",
                random_seed=seed,
                awgn_config=json.dumps({"synthetic": True}),
            ),
            "receiver": ReceiverSpec(receiver_type="benchmark"),
            "evaluation": evaluation,
        }
    )
    return _with_optional_array_and_ranging(
        result,
        rng,
        include_array=include_array,
        include_ranging=include_ranging,
        snapshots=snapshots,
    )


def _with_optional_array_and_ranging(
    result: MeasurementSimulationResult,
    rng: np.random.Generator,
    *,
    include_array: bool,
    include_ranging: bool,
    snapshots: int,
) -> MeasurementSimulationResult:
    updates: dict[str, Any] = {}
    if include_array:
        link_shape = (snapshots, result.topology.num_tx, result.topology.num_rx)
        config = ArraySpectrumConfig(enabled=True, zenith_bins=9, azimuth_bins=13)
        updates["array_outputs"] = {
            "aoa_label_rad": np.zeros((*link_shape, 2), dtype=np.float32),
            "aoa_heatmap_label": np.zeros((*link_shape, 9, 13), dtype=np.float32),
            "spatial_spectrum_truth": rng.random((*link_shape, 9, 13), dtype=np.float32),
            "angle_grid_rad": np.zeros((9, 13, 2), dtype=np.float32),
            "spectrum_policy": config.policy,
        }
    if include_ranging and result.observation is not None:
        from sionna_measurement_sim.ranging.result import (
            PdpPeakResult,
            PhaseSlopeResult,
            RangingResult,
        )

        link_shape = result.observation.valid_mask.shape
        zeros = np.zeros(link_shape, dtype=np.float32)
        success = np.ones(link_shape, dtype=np.bool_)
        updates["ranging"] = RangingResult(
            default_estimator="pdp_peak",
            pdp_peak=PdpPeakResult(
                toa_est_s=zeros,
                one_way_range_est_m=zeros,
                rtt_equiv_s=zeros,
                range_error_m=zeros,
                detection_success=success,
                selected_delay_bin=np.zeros(link_shape, dtype=np.int32),
                peak_power_linear=np.ones(link_shape, dtype=np.float32),
                peak_snr_db=np.ones(link_shape, dtype=np.float32),
            ),
            phase_slope=PhaseSlopeResult(
                toa_est_s=zeros,
                one_way_range_est_m=zeros,
                rtt_equiv_s=zeros,
                range_error_m=zeros,
                detection_success=success,
                fit_residual_rad=zeros,
            ),
        )
    if not updates:
        return result
    return MeasurementSimulationResult(**{**result.__dict__, **updates})


def _write_benchmark_outputs(
    *,
    output_dir: Path,
    summary_name: str,
    benchmark_type: str,
    status: str,
    parameters: dict[str, Any],
    rows: list[dict[str, Any]],
    perf_summary: dict[str, Any],
    artifacts: dict[str, Any],
) -> None:
    measured_rows = [row for row in rows if not row.get("warmup")]
    summary = {
        "benchmark_type": benchmark_type,
        "status": status,
        "parameters": parameters,
        "environment": collect_basic_environment(),
        "iterations": rows,
        "aggregate": _aggregate_rows(measured_rows),
        "aggregate_by_write_mode": _aggregate_rows_by_key(measured_rows, "write_mode"),
        "perf_summary": perf_summary,
        "artifacts": {
            **artifacts,
            "summary_json": (output_dir / f"{summary_name}.json").as_posix(),
            "rows_csv": (output_dir / "benchmark_rows.csv").as_posix(),
            "config_snapshot_json": (output_dir / "config_snapshot.json").as_posix(),
            "perf_summary_json": _perf_summary_path(perf_summary),
        },
    }
    (output_dir / f"{summary_name}.json").write_text(
        json.dumps(_jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "config_snapshot.json").write_text(
        json.dumps(_jsonable(parameters), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_rows_csv(output_dir / "benchmark_rows.csv", rows)


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {"count": len(rows)}
    if not rows:
        return aggregate
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float)) and key not in {"iteration", "warmup"}
        }
    )
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
        if not values:
            continue
        aggregate[f"{key}_mean"] = float(np.mean(values))
        aggregate[f"{key}_median"] = float(np.median(values))
        aggregate[f"{key}_max"] = float(np.max(values))
    return aggregate


def _aggregate_rows_by_key(
    rows: list[dict[str, Any]],
    key: str,
) -> dict[str, dict[str, Any]]:
    values = sorted({str(row[key]) for row in rows if key in row})
    return {
        value: _aggregate_rows([row for row in rows if str(row.get(key, "")) == value])
        for value in values
    }


def _write_rows_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    if not fieldnames:
        fieldnames = ["benchmark_type", "status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})


def _start_tracer(options: BenchmarkOptions, worker_id: str) -> PerfTracer:
    tracer = PerfTracer(
        options.output_dir,
        BenchmarkDebugConfig(
            hardware_interval_s=options.debug_hardware_interval_s,
            write_hardware_samples=options.write_hardware_samples,
        ),
        worker_id=worker_id,
    )
    tracer.start()
    return tracer


def _perf_summary_path(perf_summary: dict[str, Any]) -> str:
    events = str(dict(perf_summary.get("logs", {})).get("events", ""))
    if not events:
        return ""
    return events.replace("perf_events", "perf_summary").replace(".jsonl", ".json")


def _iteration_plan(options: BenchmarkOptions) -> list[tuple[int, bool]]:
    total = options.warmup + options.repeat
    return [(idx, idx < options.warmup) for idx in range(total)]


def _stage_duration(tracer: PerfTracer, name: str, *, iteration: int) -> float:
    stages = getattr(tracer, "_stages", [])
    for stage in reversed(stages):
        if stage.get("name") == name and int(stage.get("iteration", -1)) == iteration:
            return float(stage.get("duration_s", 0.0))
    return 0.0


def _rt_run_config_from_parameters(
    options: BenchmarkOptions,
    parameters: dict[str, Any],
) -> Any:
    from sionna_measurement_sim.rt.truth_pipeline import RTTruthRunConfig

    return RTTruthRunConfig(
        label_file=Path(parameters.get("label_file") or "tests/fixtures/scenes/test/test5.json"),
        scene_file=Path(parameters.get("scene_file") or "tests/fixtures/scenes/test/scene.xml"),
        output_dir=options.output_dir,
        num_subcarriers=int(parameters.get("num_subcarriers") or 8),
        seed=options.seed,
        device=options.device,
        max_depth=int(parameters.get("max_depth") or 1),
        max_bs=int(parameters.get("max_bs") or 1),
        max_ue=int(parameters.get("max_ue") or 2),
        los=_optional_bool(parameters.get("los"), True),
        specular_reflection=_optional_bool(parameters.get("specular_reflection"), True),
        diffuse_reflection=_optional_bool(parameters.get("diffuse_reflection"), False),
        refraction=_optional_bool(parameters.get("refraction"), False),
        diffraction=_optional_bool(parameters.get("diffraction"), False),
        synthetic_array=_optional_bool(parameters.get("synthetic_array"), False),
    )


def _optional_bool(value: Any, default: bool) -> bool:
    return default if value is None else bool(value)


def _synthetic_spectrum_samples(
    parameters: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    tx_count = int(parameters["tx_count"])
    rx_count = int(parameters["rx_count"])
    rx_ant = int(parameters["rx_ant"])
    tx_ant = int(parameters["tx_ant"])
    subcarriers = int(parameters["subcarriers"])
    ofdm_symbols = int(parameters["ofdm_symbols"])
    snapshots = int(parameters["snapshots"])
    shape_tail = (tx_ant, subcarriers)
    samples: dict[str, np.ndarray] = {}
    for source in parameters["sources"]:
        if source == "truth_cfr":
            samples[source] = _complex_normal(
                rng,
                (1, tx_count, rx_count, rx_ant, *shape_tail),
            )
        elif source == "cfr_est":
            samples[source] = _complex_normal(
                rng,
                (snapshots, tx_count, rx_count, rx_ant, *shape_tail),
            )
        elif source == "rx_grid":
            samples[source] = _complex_normal(
                rng,
                (snapshots, tx_count, rx_count, rx_ant, ofdm_symbols, subcarriers),
            )
    return samples


def _positions(count: int, x_offset: float) -> np.ndarray:
    coords = np.zeros((count, 3), dtype=np.float32)
    coords[:, 0] = np.arange(count, dtype=np.float32) + np.float32(x_offset)
    coords[:, 2] = 1.5
    return coords


def _complex_normal(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    real = rng.standard_normal(shape).astype(np.float32)
    imag = rng.standard_normal(shape).astype(np.float32)
    return ((real + 1j * imag) / np.sqrt(np.float32(2.0))).astype(np.complex64)


def _options_snapshot(options: BenchmarkOptions) -> dict[str, Any]:
    return {
        "output_dir": options.output_dir.as_posix(),
        "seed": options.seed,
        "repeat": options.repeat,
        "warmup": options.warmup,
        "device": options.device,
        "debug_hardware_interval_s": options.debug_hardware_interval_s,
        "write_hardware_samples": options.write_hardware_samples,
        "summary_name": options.summary_name,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(_jsonable(value), sort_keys=True)
    return value
