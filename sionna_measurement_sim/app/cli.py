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
    rt_truth.add_argument("--num-time-steps", type=int, default=1)
    rt_truth.add_argument("--sampling-frequency-hz", type=float, default=0.0)
    motion = subparsers.add_parser(
        "run-motion",
        help="Run Phase 6 multi-snapshot RT truth with motion / Doppler.",
    )
    motion.add_argument("--label-file", default="data/scenes/test/test5.json")
    motion.add_argument("--scene-file", default="data/scenes/test/scene.xml")
    motion.add_argument("--output-dir", default="outputs/phase6_motion")
    motion.add_argument("--num-subcarriers", type=int, default=8)
    motion.add_argument("--seed", type=int, default=1)
    motion.add_argument("--num-time-steps", type=int, default=3)
    motion.add_argument("--sampling-frequency-hz", type=float, default=100.0)
    observation = subparsers.add_parser(
        "run-observation",
        help="Run the Phase 4 minimal RT + AWGN/LS observation pipeline.",
    )
    observation.add_argument("--label-file", default="data/scenes/test/test5.json")
    observation.add_argument("--scene-file", default="data/scenes/test/scene.xml")
    observation.add_argument("--output-dir", default="outputs/phase4_observation")
    observation.add_argument("--num-subcarriers", type=int, default=8)
    observation.add_argument("--seed", type=int, default=1)
    observation.add_argument("--snr-db", type=float, default=40.0)
    observation.add_argument("--cfo-hz", type=float, default=None)
    observation.add_argument("--sfo-ppm", type=float, default=None)
    observation.add_argument("--phase-offset-rad", type=float, default=None)
    observation.add_argument("--timing-offset-samples", type=float, default=None)
    observation.add_argument("--clipping-threshold", type=float, default=None)
    observation.add_argument("--impairment-seed", type=int, default=11)

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
                num_time_steps=args.num_time_steps,
                sampling_frequency_hz=args.sampling_frequency_hz,
            )
        )
        print(output_path)
        return 0

    if args.command == "run-motion":
        from pathlib import Path

        from sionna_measurement_sim.rt.truth_pipeline import RTTruthRunConfig, run_rt_truth_pipeline

        output_path = run_rt_truth_pipeline(
            RTTruthRunConfig(
                label_file=Path(args.label_file),
                scene_file=Path(args.scene_file),
                output_dir=Path(args.output_dir),
                num_subcarriers=args.num_subcarriers,
                seed=args.seed,
                max_depth=1,
                specular_reflection=True,
                num_time_steps=args.num_time_steps,
                sampling_frequency_hz=args.sampling_frequency_hz,
            )
        )
        print(output_path)
        return 0

    if args.command == "run-observation":
        from pathlib import Path

        from sionna_measurement_sim.phy.impairments import ImpairmentConfig
        from sionna_measurement_sim.rt.truth_pipeline import RTTruthRunConfig, run_rt_truth_pipeline

        impairment = None
        if any(
            v is not None
            for v in (args.cfo_hz, args.sfo_ppm, args.phase_offset_rad, args.timing_offset_samples, args.clipping_threshold)  # noqa: E501
        ):
            impairment = ImpairmentConfig(
                random_seed=args.impairment_seed,
                cfo_hz=args.cfo_hz,
                sfo_ppm=args.sfo_ppm,
                phase_offset_rad=args.phase_offset_rad,
                timing_offset_samples=args.timing_offset_samples,
                clipping_threshold=args.clipping_threshold,
            )
        output_path = run_rt_truth_pipeline(
            RTTruthRunConfig(
                label_file=Path(args.label_file),
                scene_file=Path(args.scene_file),
                output_dir=Path(args.output_dir),
                num_subcarriers=args.num_subcarriers,
                seed=args.seed,
                max_depth=1,
                specular_reflection=True,
                observation_snr_db=args.snr_db,
                impairment_config=impairment,
            )
        )
        print(output_path)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
