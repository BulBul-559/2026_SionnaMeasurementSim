"""Command line entry point for SionnaMeasurementSim."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from sionna_measurement_sim import __version__
from sionna_measurement_sim.preflight.system import collect_basic_environment

_DEFAULT_LABEL = "data/scenes/test/test5.json"
_DEFAULT_SCENE = "data/scenes/test/scene.xml"
_SEED_OFFSET_IMPAIRMENT = 100
_SEED_OFFSET_OBSERVATION = 200
_SEED_OFFSET_BATCH = 1000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sionna-measurement-sim",
        description="SionnaMeasurementSim command line interface.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", default=None, help="YAML config file path")

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "preflight",
        help="Print basic local environment information.",
    )
    rt_truth = subparsers.add_parser(
        "run-rt-truth",
        help="Run the Phase 2 minimal Sionna RT truth pipeline.",
    )
    rt_truth.add_argument("--label-file", default=_DEFAULT_LABEL)
    rt_truth.add_argument("--scene-file", default=_DEFAULT_SCENE)
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
    motion.add_argument("--label-file", default=_DEFAULT_LABEL)
    motion.add_argument("--scene-file", default=_DEFAULT_SCENE)
    motion.add_argument("--output-dir", default="outputs/phase6_motion")
    motion.add_argument("--num-subcarriers", type=int, default=8)
    motion.add_argument("--seed", type=int, default=1)
    motion.add_argument("--num-time-steps", type=int, default=3)
    motion.add_argument("--sampling-frequency-hz", type=float, default=100.0)
    observation = subparsers.add_parser(
        "run-observation",
        help="Run the Phase 4 minimal RT + AWGN/LS observation pipeline.",
    )
    observation.add_argument("--label-file", default=_DEFAULT_LABEL)
    observation.add_argument("--scene-file", default=_DEFAULT_SCENE)
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

    batch = subparsers.add_parser(
        "run-batch",
        help="Run Phase 8 batch experiment across multiple seeds/SNRs.",
    )
    batch.add_argument("--label-file", default=_DEFAULT_LABEL)
    batch.add_argument("--scene-file", default=_DEFAULT_SCENE)
    batch.add_argument("--output-dir", default="outputs/phase8_batch")
    batch.add_argument("--num-subcarriers", type=int, default=8)
    batch.add_argument("--seed", type=int, default=1)
    batch.add_argument("--snr-db", type=float, default=40.0)
    batch.add_argument("--batch-count", type=int, default=2)

    full = subparsers.add_parser(
        "run-full",
        help="Full e2e: RT truth + paths + impairments + observation + motion + calibration.",
    )
    full.add_argument("--label-file", default=_DEFAULT_LABEL)
    full.add_argument("--scene-file", default=_DEFAULT_SCENE)
    full.add_argument("--output-dir", default="outputs/e2e_full")
    full.add_argument("--num-subcarriers", type=int, default=64)
    full.add_argument("--seed", type=int, default=42)
    full.add_argument("--snr-db", type=float, default=30.0)
    full.add_argument("--cfo-hz", type=float, default=100.0)
    full.add_argument("--sfo-ppm", type=float, default=5.0)
    full.add_argument("--phase-offset-rad", type=float, default=0.5)
    full.add_argument("--timing-offset-samples", type=float, default=2.0)
    full.add_argument("--clipping-threshold", type=float, default=3.0)
    full.add_argument("--num-time-steps", type=int, default=3)
    full.add_argument("--sampling-frequency-hz", type=float, default=100.0)
    full.add_argument("--max-tx", type=int, default=6)
    full.add_argument("--max-rx", type=int, default=30)

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

    if args.command == "run-full":
        from pathlib import Path

        from sionna_measurement_sim.phy.impairments import ImpairmentConfig
        from sionna_measurement_sim.rt.truth_pipeline import RTTruthRunConfig, run_rt_truth_pipeline

        if args.config:
            from sionna_measurement_sim.config.loader import load_config_or_exit

            cfg = load_config_or_exit(args.config)
            # CLI overrides (non-default wins)
            _max_rx = (args.max_rx if hasattr(args, 'max_rx')
                       and args.max_rx != 30 else cfg.input.max_rx)
            _max_tx = (args.max_tx if hasattr(args, 'max_tx')
                       and args.max_tx != 6 else cfg.input.max_tx)
            _snr = (args.snr_db if hasattr(args, 'snr_db')
                    and args.snr_db != 30.0 else cfg.phy.snr_db)
            _nsub = (args.num_subcarriers if hasattr(args, 'num_subcarriers')
                     and args.num_subcarriers != 64 else cfg.carrier.num_subcarriers)
            _outdir = (args.output_dir if hasattr(args, 'output_dir')
                       and args.output_dir != "outputs/e2e_full"
                       else cfg.output.root_dir)

            # Impairments: respect .enabled flags
            imp = cfg.impairments
            impairment = ImpairmentConfig(
                random_seed=imp.impairment_seed,
                cfo_hz=imp.cfo.cfo_hz if imp.cfo.enabled else None,
                sfo_ppm=imp.sfo.sfo_ppm if imp.sfo.enabled else None,
                phase_offset_rad=imp.phase_noise.phase_offset_rad
                if imp.phase_noise.enabled else None,
                timing_offset_samples=imp.timing_offset.timing_offset_samples
                if imp.timing_offset.enabled else None,
                agc_gain_db=imp.agc_adc.agc_gain_db if imp.agc_adc.enabled else 0.0,
                clipping_threshold=imp.agc_adc.clipping_threshold
                if imp.agc_adc.enabled else None,
            )
            # PHY: only enable if cfg.phy.enabled
            phy_enabled = cfg.phy.enabled
            obs_snr = _snr if phy_enabled else None
            # Motion: respect enabled flag
            mot = cfg.motion
            motion_enabled = mot.enabled
            run_config = RTTruthRunConfig(
                label_file=Path(cfg.input.label_file),
                scene_file=Path(cfg.input.scene_file),
                output_dir=Path(_outdir),
                center_frequency_hz=cfg.carrier.center_frequency_hz,
                bandwidth_hz=cfg.carrier.bandwidth_hz,
                num_subcarriers=_nsub,
                seed=cfg.runtime.seed,
                max_depth=cfg.rt.max_depth,
                los=cfg.rt.los,
                specular_reflection=cfg.rt.specular_reflection,
                diffuse_reflection=cfg.rt.diffuse_reflection,
                refraction=cfg.rt.refraction,
                diffraction=cfg.rt.diffraction,
                synthetic_array=cfg.rt.synthetic_array,
                normalize_cfr=cfg.rt.normalize_cfr,
                normalize_delays=cfg.rt.normalize_delays,
                observation_snr_db=obs_snr,
                impairment_config=impairment,
                num_time_steps=mot.num_time_steps if motion_enabled else 1,
                sampling_frequency_hz=(
                    mot.sampling_frequency_hz if motion_enabled else 0.0
                ),
                max_tx=_max_tx,
                max_rx=_max_rx,
                tx_num_rows=cfg.antenna.tx_array.num_rows,
                tx_num_cols=cfg.antenna.tx_array.num_cols,
                rx_num_rows=cfg.antenna.rx_array.num_rows,
                rx_num_cols=cfg.antenna.rx_array.num_cols,
                tx_polarization=cfg.antenna.tx_array.polarization,
                rx_polarization=cfg.antenna.rx_array.polarization,
                hdf5_filename=cfg.output.hdf5_filename,
                save_full_paths=cfg.output.save_full_paths,
                calibration_enabled=cfg.calibration.enabled,
                tx_velocity_mps=(
                    cfg.motion.tx_velocity_mps[0],
                    cfg.motion.tx_velocity_mps[1],
                    cfg.motion.tx_velocity_mps[2],
                ) if motion_enabled else (0.0, 0.0, 0.0),
                rx_velocity_mps=(
                    cfg.motion.rx_velocity_mps[0],
                    cfg.motion.rx_velocity_mps[1],
                    cfg.motion.rx_velocity_mps[2],
                ) if motion_enabled else (0.0, 0.0, 0.0),
            )
        else:
            impairment = ImpairmentConfig(
                random_seed=args.seed + _SEED_OFFSET_IMPAIRMENT,
                cfo_hz=args.cfo_hz,
                sfo_ppm=args.sfo_ppm,
                phase_offset_rad=args.phase_offset_rad,
                timing_offset_samples=args.timing_offset_samples,
                clipping_threshold=args.clipping_threshold,
            )
            run_config = RTTruthRunConfig(
                label_file=Path(args.label_file),
                scene_file=Path(args.scene_file),
                output_dir=Path(args.output_dir),
                num_subcarriers=args.num_subcarriers,
                seed=args.seed,
                max_depth=1,
                specular_reflection=True,
                observation_snr_db=args.snr_db,
                observation_seed=args.seed + _SEED_OFFSET_OBSERVATION,
                impairment_config=impairment,
                num_time_steps=args.num_time_steps,
                sampling_frequency_hz=args.sampling_frequency_hz,
                max_tx=args.max_tx,
                max_rx=args.max_rx,
            )
        output_path = run_rt_truth_pipeline(run_config)
        print(output_path)
        return 0

    if args.command == "run-batch":
        from pathlib import Path

        from sionna_measurement_sim.app.batch_runner import run_batch_experiment
        from sionna_measurement_sim.domain.batch import BatchConfig
        from sionna_measurement_sim.rt.truth_pipeline import RTTruthRunConfig

        batch_config = BatchConfig(
            enabled=True,
            total_batches=args.batch_count,
            completed_batches=0,
            failed_batches=0,
        )
        base_config = RTTruthRunConfig(
            label_file=Path(args.label_file),
            scene_file=Path(args.scene_file),
            output_dir=Path(args.output_dir),
            num_subcarriers=args.num_subcarriers,
            seed=args.seed,
            max_depth=1,
            specular_reflection=True,
            observation_snr_db=args.snr_db,
        )
        result = run_batch_experiment(base_config, batch_config)
        print(
            f"Batch experiment complete: {result.succeeded}/{result.batch_config.total_batches}"
            f" succeeded, {result.failed} failed"
        )
        print(f"Manifest: {result.base_output_dir / 'batch_manifest.json'}")
        return 0 if result.failed == 0 else 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
