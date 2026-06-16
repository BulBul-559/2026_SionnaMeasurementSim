"""Reusable RT/write/spectrum benchmark implementations."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

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
)
from sionna_measurement_sim.domain.topology import Topology
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
            result = build_synthetic_measurement_result(
                seed=options.seed + iteration,
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
            h5_path = output_dir / f"write_iter_{iteration:03d}.h5"
            row = {
                "iteration": iteration,
                "warmup": is_warmup,
                "benchmark_type": "write",
            }
            iter_start = time.perf_counter()
            with tracer.span("write.hdf5_write", iteration=iteration, warmup=is_warmup):
                write_measurement_result(
                    h5_path,
                    result,
                    compression=str(parameters["compression"]),
                    gzip_level=int(parameters.get("gzip_level", 4)),
                    tracer=tracer,
                )
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
                    "writer_s": _stage_duration(
                        tracer, "write.hdf5_write", iteration=iteration
                    ),
                    "schema_validate_s": schema_validate_s,
                    "file_size_bytes": h5_path.stat().st_size,
                    "path": h5_path.as_posix(),
                }
            )
            rows.append(row)
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
