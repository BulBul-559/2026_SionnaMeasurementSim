"""Command line entry point for SionnaMeasurementSim."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from sionna_measurement_sim import __version__
from sionna_measurement_sim.preflight.system import collect_basic_environment

_DEFAULT_LABEL = "tests/fixtures/scenes/test/test5.json"
_DEFAULT_SCENE = "tests/fixtures/scenes/test/scene.xml"
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
    full.add_argument("--output-dir", default=None)
    full.add_argument("--num-subcarriers", type=int, default=None)
    full.add_argument("--seed", type=int, default=None)
    full.add_argument("--snr-db", type=float, default=None)
    full.add_argument("--cfo-hz", type=float, default=None)
    full.add_argument("--sfo-ppm", type=float, default=None)
    full.add_argument("--phase-offset-rad", type=float, default=None)
    full.add_argument("--timing-offset-samples", type=float, default=None)
    full.add_argument("--clipping-threshold", type=float, default=None)
    full.add_argument("--num-time-steps", type=int, default=None)
    full.add_argument("--sampling-frequency-hz", type=float, default=None)
    full.add_argument("--max-bs", type=int, default=None)
    full.add_argument("--max-ue", type=int, default=None)
    full.add_argument("--phy-standard", default=None,
                      choices=["custom_ofdm", "nr_pusch", "nr_srs"])

    visualize = subparsers.add_parser(
        "visualize",
        help="Generate PNG visualization reports from an existing results.h5.",
    )
    visualize.add_argument("--hdf5", required=True, help="Input HDF5 results file")
    visualize.add_argument("--output-dir", required=True, help="Output directory for PNG/index")
    visualize.add_argument(
        "--mode",
        default="sample",
        choices=["sample", "selected", "full", "dataset"],
    )
    visualize.add_argument("--bs-indices", default="", help="Comma-separated BS indices")
    visualize.add_argument("--ue-indices", default="", help="Comma-separated UE indices")
    visualize.add_argument("--max-bs", type=int, default=5, help="Maximum BS count to plot")
    visualize.add_argument(
        "--sample-ue-count",
        type=int,
        default=3,
        help="Number of UEs to sample in sample mode",
    )
    visualize.add_argument("--max-ue", type=int, default=5, help="Maximum UE count to plot")
    visualize.add_argument(
        "--sample-policy",
        default="valid_links_first",
        choices=["valid_links_first", "spatially_spread_valid_links", "random", "first"],
        help="UE sampling policy for sample mode.",
    )
    visualize.add_argument("--plots", default="", help="Comma-separated plot names")
    visualize.add_argument(
        "--dataset-path",
        default=None,
        help="HDF5 dataset path for dataset mode",
    )
    visualize.add_argument(
        "--plot-type",
        default="auto",
        choices=["auto", "line", "heatmap", "hist"],
    )

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
            from sionna_measurement_sim.domain.array import ArraySpectrumConfig
            from sionna_measurement_sim.domain.link import LinkConfig as DomainLinkConfig
            from sionna_measurement_sim.visualization.config import VisualizationRunConfig

            cfg = load_config_or_exit(args.config)
            # CLI overrides: None means "not provided"; any explicit value wins.
            _max_ue = args.max_ue if args.max_ue is not None else cfg.input.max_ue
            _max_bs = args.max_bs if args.max_bs is not None else cfg.input.max_bs
            _seed = args.seed if args.seed is not None else cfg.runtime.seed
            _snr = args.snr_db if args.snr_db is not None else cfg.phy.snr_db
            _nsub = (
                args.num_subcarriers
                if args.num_subcarriers is not None
                else cfg.carrier.num_subcarriers
            )
            _outdir = args.output_dir if args.output_dir is not None else cfg.output.root_dir
            _phy_standard = (
                args.phy_standard if args.phy_standard is not None else cfg.phy.standard
            )

            # Impairments: respect .enabled flags
            imp = cfg.impairments
            impairment = ImpairmentConfig(
                random_seed=imp.impairment_seed,
                cfo_hz=(
                    args.cfo_hz
                    if args.cfo_hz is not None
                    else imp.cfo.cfo_hz if imp.cfo.enabled else None
                ),
                sfo_ppm=(
                    args.sfo_ppm
                    if args.sfo_ppm is not None
                    else imp.sfo.sfo_ppm if imp.sfo.enabled else None
                ),
                phase_offset_rad=imp.phase_noise.phase_offset_rad
                if args.phase_offset_rad is None and imp.phase_noise.enabled
                else args.phase_offset_rad,
                timing_offset_samples=imp.timing_offset.timing_offset_samples
                if args.timing_offset_samples is None and imp.timing_offset.enabled
                else args.timing_offset_samples,
                agc_gain_db=imp.agc_adc.agc_gain_db if imp.agc_adc.enabled else 0.0,
                clipping_threshold=imp.agc_adc.clipping_threshold
                if args.clipping_threshold is None and imp.agc_adc.enabled
                else args.clipping_threshold,
            )
            # PHY: only enable if cfg.phy.enabled
            phy_enabled = cfg.phy.enabled
            obs_snr = _snr if phy_enabled else None
            # Motion: respect enabled flag
            mot = cfg.motion
            motion_cli_override = (
                args.num_time_steps is not None
                or args.sampling_frequency_hz is not None
            )
            motion_enabled = mot.enabled or motion_cli_override
            run_config = RTTruthRunConfig(
                label_file=Path(cfg.input.label_file),
                scene_file=Path(cfg.input.scene_file),
                output_dir=Path(_outdir),
                scene_id=cfg.input.scene_id,
                map_id=cfg.input.map_id,
                center_frequency_hz=cfg.carrier.center_frequency_hz,
                bandwidth_hz=cfg.carrier.bandwidth_hz,
                num_subcarriers=_nsub,
                seed=_seed,
                device=cfg.runtime.device,
                max_depth=cfg.rt.max_depth,
                los=cfg.rt.los,
                specular_reflection=cfg.rt.specular_reflection,
                diffuse_reflection=cfg.rt.diffuse_reflection,
                refraction=cfg.rt.refraction,
                diffraction=cfg.rt.diffraction,
                synthetic_array=cfg.rt.synthetic_array,
                merge_shapes=cfg.rt.merge_shapes,
                normalize_cfr=cfg.rt.normalize_cfr,
                normalize_delays=cfg.rt.normalize_delays,
                observation_snr_db=obs_snr,
                impairment_config=impairment,
                phy_standard=_phy_standard,
                subcarrier_spacing_khz=cfg.phy.subcarrier_spacing_khz,
                num_prb=cfg.phy.num_prb,
                num_layers=cfg.phy.num_layers,
                num_antenna_ports=cfg.phy.num_antenna_ports,
                mcs_index=cfg.phy.mcs_index,
                mcs_table=cfg.phy.mcs_table,
                perfect_csi=cfg.phy.perfect_csi,
                pusch_dmrs_config_type=cfg.phy.pusch_dmrs_config_type,
                pusch_dmrs_length=cfg.phy.pusch_dmrs_length,
                pusch_dmrs_additional_position=cfg.phy.pusch_dmrs_additional_position,
                pusch_num_cdm_groups_without_data=cfg.phy.pusch_num_cdm_groups_without_data,
                tx_power_dbm=cfg.phy.tx_power_dbm,
                su_mimo_link_batch_size=cfg.phy.su_mimo_link_batch_size,
                num_ofdm_symbols=cfg.phy.num_ofdm_symbols,
                cp_length=cfg.phy.cp_length,
                num_time_steps=(
                    args.num_time_steps
                    if args.num_time_steps is not None
                    else mot.num_time_steps if motion_enabled else 1
                ),
                sampling_frequency_hz=(
                    args.sampling_frequency_hz
                    if args.sampling_frequency_hz is not None
                    else mot.sampling_frequency_hz if motion_enabled else 0.0
                ),
                max_tx=_max_bs,
                max_rx=_max_ue,
                tx_num_rows=cfg.antenna.bs_array.num_rows,
                tx_num_cols=cfg.antenna.bs_array.num_cols,
                rx_num_rows=cfg.antenna.ue_array.num_rows,
                rx_num_cols=cfg.antenna.ue_array.num_cols,
                tx_polarization=cfg.antenna.bs_array.polarization,
                rx_polarization=cfg.antenna.ue_array.polarization,
                tx_pattern=cfg.antenna.bs_array.pattern,
                rx_pattern=cfg.antenna.ue_array.pattern,
                tx_orientation_mode=cfg.antenna.bs_array.orientation_mode,
                tx_orientation_rad=tuple(cfg.antenna.bs_array.orientation_rad),
                rx_orientation_mode=cfg.antenna.ue_array.orientation_mode,
                rx_orientation_rad=tuple(cfg.antenna.ue_array.orientation_rad),
                tx_spacing_lambda=(
                    cfg.antenna.bs_array.vertical_spacing_lambda,
                    cfg.antenna.bs_array.horizontal_spacing_lambda,
                ),
                rx_spacing_lambda=(
                    cfg.antenna.ue_array.vertical_spacing_lambda,
                    cfg.antenna.ue_array.horizontal_spacing_lambda,
                ),
                hdf5_filename=cfg.output.hdf5_filename,
                hdf5_compression=cfg.output.compression,
                save_full_paths=cfg.output.save_full_paths,
                calibration_enabled=cfg.calibration.enabled,
                link_config=DomainLinkConfig(
                    duplex_mode=cfg.link.duplex_mode,
                    phy_link_direction=cfg.link.phy_link_direction,
                ),
                debug_config=cfg.debug,
                output_sharding_config=cfg.output.sharding,
                visualization_config=VisualizationRunConfig(
                    enabled=cfg.visualization.enabled,
                    output_dir=cfg.visualization.output_dir,
                    sample_policy=cfg.visualization.sample_policy,
                    random_seed=cfg.visualization.random_seed,
                    max_bs=cfg.visualization.max_bs,
                    sample_ue_count=cfg.visualization.sample_ue_count,
                    max_ue=cfg.visualization.max_ue,
                    dpi=cfg.visualization.dpi,
                    format=cfg.visualization.format,
                    plots=tuple(cfg.visualization.plots),
                ),
                spectrum_config=ArraySpectrumConfig(
                    enabled=cfg.array.spectrum.enabled,
                    sources=tuple(cfg.array.spectrum.sources),
                    method=cfg.array.spectrum.method,
                    zenith_bins=cfg.array.spectrum.zenith_bins,
                    azimuth_bins=cfg.array.spectrum.azimuth_bins,
                    zenith_min_rad=cfg.array.spectrum.zenith_min_rad,
                    zenith_max_rad=cfg.array.spectrum.zenith_max_rad,
                    azimuth_min_rad=cfg.array.spectrum.azimuth_min_rad,
                    azimuth_max_rad=cfg.array.spectrum.azimuth_max_rad,
                    normalize=cfg.array.spectrum.normalize,
                    aggregate_subcarriers=cfg.array.spectrum.aggregate_subcarriers,
                    aggregate_symbols=cfg.array.spectrum.aggregate_symbols,
                    link_chunk_size=cfg.array.spectrum.link_chunk_size,
                ),
                tx_velocity_mps=(
                    cfg.motion.bs_velocity_mps[0],
                    cfg.motion.bs_velocity_mps[1],
                    cfg.motion.bs_velocity_mps[2],
                ) if motion_enabled else (0.0, 0.0, 0.0),
                rx_velocity_mps=(
                    cfg.motion.ue_velocity_mps[0],
                    cfg.motion.ue_velocity_mps[1],
                    cfg.motion.ue_velocity_mps[2],
                ) if motion_enabled else (0.0, 0.0, 0.0),
                mimo_mode=cfg.phy.mimo_mode,
                channel_backend=cfg.phy.channel_backend,
                mimo_detector=cfg.phy.mimo_detector,
                channel_estimator=cfg.phy.channel_estimator,
                receiver_failure_policy=cfg.phy.receiver_failure_policy,
            )
        else:
            seed = args.seed if args.seed is not None else 42
            impairment = ImpairmentConfig(
                random_seed=seed + _SEED_OFFSET_IMPAIRMENT,
                cfo_hz=args.cfo_hz if args.cfo_hz is not None else 100.0,
                sfo_ppm=args.sfo_ppm if args.sfo_ppm is not None else 5.0,
                phase_offset_rad=(
                    args.phase_offset_rad
                    if args.phase_offset_rad is not None
                    else 0.5
                ),
                timing_offset_samples=(
                    args.timing_offset_samples
                    if args.timing_offset_samples is not None
                    else 2.0
                ),
                clipping_threshold=(
                    args.clipping_threshold
                    if args.clipping_threshold is not None
                    else 3.0
                ),
            )
            run_config = RTTruthRunConfig(
                label_file=Path(args.label_file),
                scene_file=Path(args.scene_file),
                output_dir=Path(args.output_dir or "outputs/e2e_full"),
                num_subcarriers=args.num_subcarriers or 64,
                seed=seed,
                max_depth=1,
                specular_reflection=True,
                observation_snr_db=args.snr_db if args.snr_db is not None else 30.0,
                observation_seed=seed + _SEED_OFFSET_OBSERVATION,
                impairment_config=impairment,
                phy_standard=args.phy_standard or "custom_ofdm",
                num_time_steps=args.num_time_steps if args.num_time_steps is not None else 3,
                sampling_frequency_hz=(
                    args.sampling_frequency_hz
                    if args.sampling_frequency_hz is not None
                    else 100.0
                ),
                max_tx=args.max_bs if args.max_bs is not None else 6,
                max_rx=args.max_ue if args.max_ue is not None else 30,
            )
        output_path = run_rt_truth_pipeline(run_config)
        print(output_path)
        return 0

    if args.command == "visualize":
        from pathlib import Path

        from sionna_measurement_sim.visualization.config import VisualizationRunConfig
        from sionna_measurement_sim.visualization.report import generate_visualization_report

        report = generate_visualization_report(
            Path(args.hdf5),
            Path(args.output_dir),
            VisualizationRunConfig(
                enabled=True,
                sample_policy=args.sample_policy,
                max_bs=args.max_bs,
                sample_ue_count=args.sample_ue_count,
                max_ue=args.max_ue,
            ),
            mode=args.mode,
            bs_indices=_parse_csv_ints(args.bs_indices),
            ue_indices=_parse_csv_ints(args.ue_indices),
            plots=_parse_csv_strings(args.plots),
            dataset_path=args.dataset_path,
            plot_type=args.plot_type,
        )
        print(report["index_path"])
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


def _parse_csv_ints(value: str) -> list[int] | None:
    if not value:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_csv_strings(value: str) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
