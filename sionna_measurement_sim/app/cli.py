"""Command line entry point for SionnaMeasurementSim."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from sionna_measurement_sim import __version__
from sionna_measurement_sim.preflight.system import collect_basic_environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sionna-measurement-sim",
        description="SionnaMeasurementSim command line interface.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "preflight",
        help="Print basic local environment information.",
    )
    rt_truth = subparsers.add_parser(
        "run-rt-truth",
        help="Run the Phase 2 minimal Sionna RT truth pipeline.",
    )
    rt_truth.add_argument("--label-file", default="data/scenes/test/test5.json")
    rt_truth.add_argument("--scene-file", default="data/scenes/test/scene.xml")
    rt_truth.add_argument("--output-dir", default="outputs/phase2_rt_truth")
    rt_truth.add_argument("--num-subcarriers", type=int, default=8)
    rt_truth.add_argument("--seed", type=int, default=1)
    rt_truth.add_argument("--max-depth", type=int, default=1)
    rt_truth.add_argument("--no-specular-reflection", action="store_true")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "preflight":
        for key, value in collect_basic_environment().items():
            print(f"{key}: {value}")
        return 0

    if args.command == "run-rt-truth":
        from pathlib import Path

        from sionna_measurement_sim.rt.truth_pipeline import RTTruthRunConfig, run_rt_truth_pipeline

        output_path = run_rt_truth_pipeline(
            RTTruthRunConfig(
                label_file=Path(args.label_file),
                scene_file=Path(args.scene_file),
                output_dir=Path(args.output_dir),
                num_subcarriers=args.num_subcarriers,
                seed=args.seed,
                max_depth=args.max_depth,
                specular_reflection=not args.no_specular_reflection,
            )
        )
        print(output_path)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
