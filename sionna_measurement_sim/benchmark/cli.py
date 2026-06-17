"""Argparse wiring for benchmark subcommands."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from sionna_measurement_sim.benchmark.runner import (
    BenchmarkOptions,
    run_rt_benchmark,
    run_sharding_benchmark,
    run_spectrum_benchmark,
    run_write_benchmark,
)


def add_benchmark_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register `benchmark rt|write|spectrum` under the main CLI."""

    benchmark = subparsers.add_parser(
        "benchmark",
        help="Run isolated RT/write/spectrum performance benchmarks.",
    )
    modes = benchmark.add_subparsers(dest="benchmark_mode", required=True)
    _add_rt_parser(modes)
    _add_write_parser(modes)
    _add_sharding_parser(modes)
    _add_spectrum_parser(modes)


def run_benchmark_from_args(args: argparse.Namespace) -> Path:
    """Dispatch parsed benchmark CLI arguments."""

    options = BenchmarkOptions(
        output_dir=Path(args.output_dir),
        seed=args.seed,
        repeat=args.repeat,
        warmup=args.warmup,
        device=args.device,
        debug_hardware_interval_s=args.debug_hardware_interval_s,
        write_hardware_samples=args.write_hardware_samples,
        summary_name=args.summary_name,
    )
    if args.benchmark_mode == "rt":
        return run_rt_benchmark(options, _rt_parameters(args))
    if args.benchmark_mode == "write":
        return run_write_benchmark(options, _write_parameters(args))
    if args.benchmark_mode == "sharding":
        return run_sharding_benchmark(options, _sharding_parameters(args))
    if args.benchmark_mode == "spectrum":
        return run_spectrum_benchmark(options, _spectrum_parameters(args))
    msg = f"Unsupported benchmark mode: {args.benchmark_mode!r}"
    raise ValueError(msg)


def _add_common(parser: argparse.ArgumentParser, *, default_output: str) -> None:
    parser.add_argument("--output-dir", default=default_output)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--debug-hardware-interval-s", type=float, default=1.0)
    parser.add_argument(
        "--write-hardware-samples",
        dest="write_hardware_samples",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-write-hardware-samples",
        dest="write_hardware_samples",
        action="store_false",
    )
    parser.add_argument("--summary-name", default="benchmark_summary")


def _add_rt_parser(modes: argparse._SubParsersAction) -> None:
    parser = modes.add_parser("rt", help="Benchmark RT solve without PHY/HDF5 output.")
    _add_common(parser, default_output="outputs/benchmark_rt")
    parser.add_argument("--config", dest="benchmark_config", default=None)
    parser.add_argument("--label-file", default=None)
    parser.add_argument("--scene-file", default=None)
    parser.add_argument("--max-bs", type=int, default=None)
    parser.add_argument("--max-ue", type=int, default=None)
    parser.add_argument("--num-subcarriers", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.set_defaults(
        los=None,
        specular_reflection=None,
        diffuse_reflection=None,
        refraction=None,
        diffraction=None,
        synthetic_array=None,
    )
    _add_bool_pair(parser, "los")
    _add_bool_pair(parser, "specular-reflection")
    _add_bool_pair(parser, "diffuse-reflection")
    _add_bool_pair(parser, "refraction")
    _add_bool_pair(parser, "diffraction")
    _add_bool_pair(parser, "synthetic-array")


def _add_write_parser(modes: argparse._SubParsersAction) -> None:
    parser = modes.add_parser("write", help="Benchmark HDF5 writer with synthetic arrays.")
    _add_common(parser, default_output="outputs/benchmark_write")
    parser.add_argument("--tx-count", type=int, default=1)
    parser.add_argument("--rx-count", type=int, default=2)
    parser.add_argument("--rx-ant", type=int, default=2)
    parser.add_argument("--tx-ant", type=int, default=1)
    parser.add_argument("--subcarriers", type=int, default=16)
    parser.add_argument("--snapshots", type=int, default=1)
    parser.add_argument("--include-waveform", action="store_true")
    parser.add_argument("--include-array", action="store_true")
    parser.add_argument("--include-ranging", action="store_true")
    parser.add_argument(
        "--bundle-shards",
        type=int,
        default=0,
        help=(
            "If >0, benchmark independent shard HDF5 files versus appendable "
            "bundle HDF5 files with this many synthetic shard fragments."
        ),
    )
    parser.add_argument(
        "--bundle-max-planned-shards",
        type=int,
        default=10,
        help="Maximum planned shard fragments per synthetic bundle file.",
    )
    parser.add_argument(
        "--compression",
        default="gzip",
        choices=["gzip", "lzf", "none", "mixed"],
    )
    parser.add_argument("--gzip-level", type=int, default=4)
    parser.add_argument(
        "--validate-schema",
        dest="validate_schema",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-validate-schema",
        dest="validate_schema",
        action="store_false",
    )


def _add_sharding_parser(modes: argparse._SubParsersAction) -> None:
    parser = modes.add_parser(
        "sharding",
        help="Benchmark real sharded HDF5 files versus appendable bundle HDF5.",
    )
    _add_common(parser, default_output="outputs/benchmark_sharding")
    parser.add_argument("--label-file", default=None)
    parser.add_argument("--scene-file", default=None)
    parser.add_argument("--max-bs", type=int, default=1)
    parser.add_argument("--max-ue", type=int, default=3)
    parser.add_argument("--num-subcarriers", type=int, default=8)
    parser.add_argument("--max-depth", type=int, default=1)
    parser.add_argument("--shard-size", type=int, default=1)
    parser.add_argument("--parallel-workers", type=int, default=1)
    parser.add_argument("--bundle-max-planned-shards", type=int, default=2)
    parser.add_argument(
        "--readback-dataset",
        default="channel/truth/cfr",
        help="Dataset to read back through the manifest after each sharding run.",
    )
    parser.add_argument(
        "--compression",
        default="mixed",
        choices=["gzip", "lzf", "none", "mixed"],
    )
    parser.add_argument("--gzip-level", type=int, default=1)


def _add_spectrum_parser(modes: argparse._SubParsersAction) -> None:
    parser = modes.add_parser("spectrum", help="Benchmark Bartlett spectrum core.")
    _add_common(parser, default_output="outputs/benchmark_spectrum")
    parser.add_argument("--links", type=int, default=None)
    parser.add_argument("--tx-count", type=int, default=4)
    parser.add_argument("--rx-count", type=int, default=1)
    parser.add_argument("--rx-ant", type=int, default=4)
    parser.add_argument("--tx-ant", type=int, default=1)
    parser.add_argument("--subcarriers", type=int, default=32)
    parser.add_argument("--ofdm-symbols", type=int, default=2)
    parser.add_argument("--snapshots", type=int, default=1)
    parser.add_argument("--zenith-bins", type=int, default=9)
    parser.add_argument("--azimuth-bins", type=int, default=13)
    parser.add_argument("--sources", default="truth_cfr,cfr_est,rx_grid")
    parser.add_argument("--link-chunk-size", type=int, default=512)


def _add_bool_pair(parser: argparse.ArgumentParser, name: str) -> None:
    dest = name.replace("-", "_")
    parser.add_argument(f"--{name}", dest=dest, action="store_true")
    parser.add_argument(f"--no-{name}", dest=dest, action="store_false")


def _rt_parameters(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "config": args.benchmark_config or getattr(args, "config", None),
        "label_file": args.label_file,
        "scene_file": args.scene_file,
        "max_bs": args.max_bs,
        "max_ue": args.max_ue,
        "num_subcarriers": args.num_subcarriers,
        "max_depth": args.max_depth,
        "los": args.los,
        "specular_reflection": args.specular_reflection,
        "diffuse_reflection": args.diffuse_reflection,
        "refraction": args.refraction,
        "diffraction": args.diffraction,
        "synthetic_array": args.synthetic_array,
    }


def _write_parameters(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "tx_count": args.tx_count,
        "rx_count": args.rx_count,
        "rx_ant": args.rx_ant,
        "tx_ant": args.tx_ant,
        "subcarriers": args.subcarriers,
        "snapshots": args.snapshots,
        "include_waveform": args.include_waveform,
        "include_array": args.include_array,
        "include_ranging": args.include_ranging,
        "bundle_shards": args.bundle_shards,
        "bundle_max_planned_shards": args.bundle_max_planned_shards,
        "compression": args.compression,
        "gzip_level": args.gzip_level,
        "validate_schema": args.validate_schema,
    }


def _sharding_parameters(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "label_file": args.label_file,
        "scene_file": args.scene_file,
        "max_bs": args.max_bs,
        "max_ue": args.max_ue,
        "num_subcarriers": args.num_subcarriers,
        "max_depth": args.max_depth,
        "shard_size": args.shard_size,
        "parallel_workers": args.parallel_workers,
        "bundle_max_planned_shards": args.bundle_max_planned_shards,
        "readback_dataset": args.readback_dataset,
        "compression": args.compression,
        "gzip_level": args.gzip_level,
    }


def _spectrum_parameters(args: argparse.Namespace) -> dict[str, Any]:
    tx_count = args.tx_count
    rx_count = args.rx_count
    if args.links is not None:
        tx_count = args.links
        rx_count = 1
    return {
        "tx_count": tx_count,
        "rx_count": rx_count,
        "rx_ant": args.rx_ant,
        "tx_ant": args.tx_ant,
        "subcarriers": args.subcarriers,
        "ofdm_symbols": args.ofdm_symbols,
        "snapshots": args.snapshots,
        "zenith_bins": args.zenith_bins,
        "azimuth_bins": args.azimuth_bins,
        "sources": tuple(
            source.strip() for source in args.sources.split(",") if source.strip()
        ),
        "link_chunk_size": args.link_chunk_size,
    }
