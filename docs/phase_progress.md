# Phase Progress

## 2026-05-06 - Phase 1

Current completed phase: Phase 1.

This round completed:

- Added domain dataclasses for topology, antenna, frequency, device state, scene/input/runtime metadata, RT truth CFR, path samples, and the aggregate measurement result.
- Added HDF5 writer/reader skeleton under `sionna_measurement_sim/io/`.
- Added a minimal HDF5 schema validator that checks required groups/datasets, dtype/shape relations, unit attributes, finite values, and the forbidden `/channel/cfr` truth path.
- Added schema and domain tests.
- Generated the Phase 1 ignored output fixture `outputs/phase1_schema/results.h5`.

Commands and results:

- `uv run pytest tests/unit tests/schema`: passed, 14 tests.
- `uv run python scripts/write_phase1_fixture.py`: passed, generated `outputs/phase1_schema/results.h5`.
- `uv run pytest`: passed, 14 tests.
- `uv run ruff check .`: passed.
- `git status --short --ignored`: reviewed; outputs, caches, `.venv/`, `old`, and large scene OBJ are ignored.

Key files generated:

- `sionna_measurement_sim/domain/`
- `sionna_measurement_sim/io/hdf5_writer.py`
- `sionna_measurement_sim/io/hdf5_reader.py`
- `sionna_measurement_sim/io/schema_validator.py`
- `tests/schema/test_hdf5_schema.py`
- `tests/unit/test_domain_models.py`
- `scripts/write_phase1_fixture.py`
- `artifacts/phase_reports/phase_1_acceptance.md`
- `outputs/phase1_schema/results.h5` (ignored runtime artifact)

Acceptance items passed:

- Minimal HDF5 includes `/meta/schema_version`, `/meta/contract_name`, `/meta/index_order`, `/meta/unit_convention`, `/meta/config_snapshot`, `/topology/tx_positions_m`, `/topology/rx_positions_m`, `/antenna/tx_polarization`, `/antenna/rx_polarization`, and `/frequency/frequencies_hz`.
- Readback verifies dtype, shape, units, and index order.
- Phase 1 tests pass without importing Sionna.
- Schema tests explicitly reject missing `/meta/schema_version`.
- Writer does not write truth CFR to `/channel/cfr`; truth CFR is at `/channel/truth/cfr`.

Current git commit hash:

- Before this Phase 1 commit: `b96c71d`.
- Phase 1 implementation commit: `866d722`.

Next step:

- Continue from Phase 2: Sionna 2.x RT truth minimal closed loop.

Known issues or blockers:

- No Phase 1 blockers.
- Existing uncommitted document changes outside this phase were preserved: `docs/README.md` and `docs/12_final_acceptance_checklist.md`.

## 2026-05-06 - Phase 2

Current completed phase: Phase 2.

This round completed:

- Added `sionna-rt` dependency with uv.
- Implemented the minimal Sionna RT truth adapter under `sionna_measurement_sim/adapters/sionna_rt/`.
- Added a label parser for the prepared `data/scenes/test/test5.json` topology.
- Added the Phase 2 RT truth pipeline and CLI command `run-rt-truth`.
- Generated an ignored Phase 2 run under `outputs/phase2_rt_truth/`.
- Added adapter and integration tests for the RT truth closed loop.

Commands and results:

- `uv add sionna-rt`: passed; installed `sionna-rt==2.0.1`, `mitsuba==3.8.0`, and `drjit==1.3.1`.
- `uv run pytest tests/adapter tests/integration -k "rt_truth"`: passed, 2 tests.
- `uv run python -m sionna_measurement_sim.app.cli run-rt-truth --output-dir outputs/phase2_rt_truth`: passed, generated `outputs/phase2_rt_truth/results.h5`.
- `uv run ruff check .`: passed.
- `uv run pytest`: passed, 16 tests.
- `git status --short --ignored`: reviewed; outputs, caches, `.venv/`, `old`, and large scene OBJ are ignored.

Key files generated:

- `sionna_measurement_sim/adapters/sionna_rt/rt_solver.py`
- `sionna_measurement_sim/rt/truth_pipeline.py`
- `sionna_measurement_sim/io/label_parser.py`
- `sionna_measurement_sim/io/manifest.py`
- `tests/adapter/test_rt_truth_adapter.py`
- `tests/integration/test_rt_truth_pipeline.py`
- `artifacts/phase_reports/phase_2_acceptance.md`
- `outputs/phase2_rt_truth/results.h5` (ignored runtime artifact)
- `outputs/phase2_rt_truth/manifest.json` (ignored runtime artifact)
- `outputs/phase2_rt_truth/logs/run.log` (ignored runtime artifact)

Acceptance items passed:

- Loaded the prepared test scene and registered one TX/RX pair from labels.
- Ran Sionna RT `PathSolver`.
- Generated `/channel/truth/cfr`.
- H_true shape is `[tx, rx, rx_ant, tx_ant, subcarrier]`, observed as `(1, 1, 1, 1, 8)`.
- `frequencies_hz.shape[-1] == H_true.shape[-1]`.
- H_true dtype is `complex64`.
- At least one finite CFR value exists and the result is not all NaN.
- Runtime version datasets exist for Sionna RT, Mitsuba, Dr.Jit, and torch.
- Manifest records scene file, config snapshot, software versions, and raw/internal CFR shapes.
- HDF5 readback preserves H_true shape and dtype.

Current git commit hash:

- Before this Phase 2 commit: `d22c539`.
- Phase 2 implementation commit: `9df7777`.

Next step:

- Continue from Phase 3: Path adapter and path-level samples.

Known issues or blockers:

- No Phase 2 blockers.
- Phase 2 uses RT-only `sionna-rt`; PyTorch is not installed yet because the PyTorch PHY/SYS chain starts in later phases.
- Existing uncommitted document changes outside this phase were preserved: `docs/README.md` and `docs/12_final_acceptance_checklist.md`.

## 2026-05-06 - Phase 3

Current completed phase: Phase 3.

This round completed:

- Added TX-first `PathTable` domain model.
- Added Sionna RT path adapter for path scalar fields, complex path coefficients, interactions, objects, primitives, vertices, path type, and path depth.
- Extended the RT truth pipeline to use `max_depth=1` with specular reflections by default.
- Wrote non-empty `/paths/samples` and debug `/paths/full` HDF5 groups.
- Added path visualization smoke helper and generated an ignored path plot.

Commands and results:

- `uv run pytest tests/adapter tests/schema tests/integration -k "path"`: passed, 3 tests.
- `uv run python -m sionna_measurement_sim.app.cli run-rt-truth --output-dir outputs/phase3_paths`: passed.
- Path plot generation to `outputs/phase3_paths/paths.png`: passed.
- `uv run ruff check .`: passed.
- `uv run pytest`: passed, 19 tests.
- `git status --short --ignored`: reviewed; outputs, caches, `.venv/`, `old`, and large scene OBJ are ignored.

Key files generated:

- `sionna_measurement_sim/adapters/sionna_rt/path_adapter.py`
- `sionna_measurement_sim/visualization/path_plots.py`
- `artifacts/phase_reports/phase_3_acceptance.md`
- `outputs/phase3_paths/results.h5` (ignored runtime artifact)
- `outputs/phase3_paths/manifest.json` (ignored runtime artifact)
- `outputs/phase3_paths/logs/run.log` (ignored runtime artifact)
- `outputs/phase3_paths/paths.png` (ignored runtime artifact)

Acceptance items passed:

- Parsed 26 valid paths from the prepared test scene.
- Found 25 NLoS paths with finite intermediate vertices.
- `/paths/samples` contains sampled link/path indices, vertices, interaction type, object id, primitive id, Doppler, delay, path gain, and path type.
- `/paths/full` contains the TX-first full path table.
- Active sample paths satisfy `vertex_count >= interaction_count + 2`.
- Static-scene Doppler fields exist and are finite.
- Path visualization smoke output exists and is non-empty.

Current git commit hash:

- Before this Phase 3 commit: `14a81bd`.
- Phase 3 implementation commit: `ea3351d`.

Next step:

- Continue from Phase 4: minimal PHY observation with custom OFDM pilots, AWGN, and LS estimation.

Known issues or blockers:

- No Phase 3 blockers.
- Existing uncommitted document changes outside this phase were preserved: `docs/README.md` and `docs/12_final_acceptance_checklist.md`.

## 2026-05-06 - Phase 4

Current completed phase: Phase 4.

This round completed:

- Added PyTorch dependency for PHY tensor operations.
- Added minimal custom OFDM waveform, AWGN-only observation, LS estimator semantics, receiver diagnostics, and NMSE evaluation.
- Extended HDF5 writer and validator for `/waveform`, `/observation`, `/impairments`, `/receiver`, and `/evaluation`.
- Added CLI command `run-observation`.
- Generated an ignored Phase 4 observation run.

Commands and results:

- `uv add torch`: passed; installed `torch==2.11.0`.
- `uv run pytest tests/unit tests/integration tests/statistical -k "awgn or observation or nmse"`: passed, 3 tests.
- `uv run python -m sionna_measurement_sim.app.cli run-observation --output-dir outputs/phase4_observation --snr-db 40`: passed.
- `uv run ruff check .`: passed.
- `uv run pytest`: passed, 22 tests.
- `git status --short --ignored`: reviewed; outputs, caches, `.venv/`, `old`, and large scene OBJ are ignored.

Key files generated:

- `sionna_measurement_sim/domain/observation.py`
- `sionna_measurement_sim/phy/observation_pipeline.py`
- `tests/unit/test_observation_pipeline.py`
- `tests/statistical/test_awgn_observation.py`
- `artifacts/phase_reports/phase_4_acceptance.md`
- `outputs/phase4_observation/results.h5` (ignored runtime artifact)
- `outputs/phase4_observation/manifest.json` (ignored runtime artifact)
- `outputs/phase4_observation/logs/run.log` (ignored runtime artifact)

Acceptance items passed:

- `/waveform/standard`, `/waveform/fft_size`, `/waveform/pilot_indices`, and `/waveform/pilot_symbols` exist.
- `/receiver/estimator_type` exists and is `ls`.
- `/observation/cfr_est` exists with shape `(1, 1, 1, 1, 1, 8)`.
- `/observation/cfr_est.shape[1:] == /channel/truth/cfr.shape`.
- `/observation/valid_mask`, `/observation/detection_success`, and `/observation/estimation_success` shape is `(1, 1, 1)`.
- `/observation/snr_db` and `/evaluation/nmse_db` exist.
- AWGN-only high SNR median NMSE is below `-20 dB`; generated output observed about `-41.3 dB`.
- Statistical test verifies high SNR median NMSE is lower than low SNR median NMSE.

Current git commit hash:

- Before this Phase 4 commit: `7d24a0c`.
- Phase 4 implementation commit: `9ee53da`.

Next step:

- Continue from Phase 5: base impairments including CFO, SFO, phase, timing, AGC, and ADC/clipping.

Known issues or blockers:

- No Phase 4 blockers.
- Existing uncommitted document changes outside this phase were preserved: `docs/README.md` and `docs/12_final_acceptance_checklist.md`.

## 2026-05-06 - Phase 5

Current completed phase: Phase 5.

This round completed:

- Added `sionna_measurement_sim/phy/impairments.py` with CFO, SFO, phase offset, timing offset, AGC gain, and ADC clipping models.
- Added `ImpairmentConfig` and `ImpairmentSample` dataclasses.
- Integrated impairments into the observation pipeline (applied before AWGN + LS estimation).
- Added CLI flags for impairment parameters: `--cfo-hz`, `--sfo-ppm`, `--phase-offset-rad`, `--timing-offset-samples`, `--clipping-threshold`, `--impairment-seed`.
- Extended schema validator with impairment-related required datasets.
- Updated `ImpairmentSpec` with proper config JSON for CFO/SFO, phase noise, and AGC/ADC.

Commands and results:

- `uv run pytest tests/unit tests/statistical -k "cfo or sfo or clipping or impairment"`: passed, 25 tests.
- `uv run python -m sionna_measurement_sim.app.cli run-observation --output-dir outputs/phase5_impairments --snr-db 40 --cfo-hz 100 --sfo-ppm 5 --timing-offset-samples 2 --clipping-threshold 3 --phase-offset-rad 0.5`: passed.
- `uv run ruff check .`: passed.
- `uv run pytest`: passed, 47 tests.
- `git status --short --ignored`: reviewed; outputs, caches, `.venv/`, `old`, and large scene OBJ are ignored.

Key files generated:

- `sionna_measurement_sim/phy/impairments.py`
- `tests/unit/test_impairments.py`
- `tests/statistical/test_impairments_statistical.py`
- `artifacts/phase_reports/phase_5_acceptance.md`
- `outputs/phase5_impairments/results.h5` (ignored runtime artifact)
- `outputs/phase5_impairments/manifest.json` (ignored runtime artifact)
- `outputs/phase5_impairments/logs/run.log` (ignored runtime artifact)

Acceptance items passed:

- Fixed seed impairment sampling is reproducible.
- `/observation/cfo_hz`, `/observation/sfo_ppm`, `/observation/timing_offset_samples`, `/observation/phase_offset_rad`, `/observation/agc_gain_db`, `/observation/clipping_flag` all exist and have correct shapes.
- `/impairments/model_version` = `"phase5_base_impairments_v1"` and `/impairments/random_seed` exist.
- CFO disabled → impairment fields are zero.
- Lower clipping threshold produces higher or equal clipping flag ratio.
- All impairment config JSON written to HDF5 (`cfo_sfo_config`, `phase_noise_config`, `agc_adc_config`).
- Schema validator checks all new impairment datasets.
- CLI flags for all impairment parameters work.

Current git commit hash:

- Before this Phase 5 commit: `c5ccc82`.
- Phase 5 implementation commit: `ec5e91a`.

Next step:

- Continue from Phase 6: motion and Doppler.

Known issues or blockers:

- No Phase 5 blockers.
- Existing uncommitted document changes outside this phase were preserved: `docs/README.md` and `docs/12_final_acceptance_checklist.md`.

## 2026-05-06 - Phase 6

Current completed phase: Phase 6.

This round completed:

- Added `MotionSpec` domain model with timestamp monotonicity validation.
- Extended `RTTruthResult` and schema validator to support both 5D (static) and 6D (multi-snapshot) truth CFR.
- Updated Sionna RT solver for `num_time_steps` support.
- Extended truth pipeline for multi-snapshot, velocity config, and motion spec generation.
- Added `/motion` HDF5 group writer.
- Added CLI command `run-motion` with `--num-time-steps` and `--sampling-frequency-hz` flags.
- Added delay-Doppler scatter plot visualization.

Commands and results:

- `uv run pytest tests/adapter tests/integration tests/statistical -k "doppler or motion"`: passed, 5 tests.
- `uv run python -m sionna_measurement_sim.app.cli run-motion --output-dir outputs/phase6_motion --num-time-steps 3 --sampling-frequency-hz 100`: passed.
- `uv run ruff check .`: passed.
- `uv run pytest`: passed, 56 tests.

Key files generated:

- `sionna_measurement_sim/domain/motion.py`
- `tests/unit/test_motion_domain.py`
- `tests/statistical/test_motion.py`
- `artifacts/phase_reports/phase_6_acceptance.md`
- `outputs/phase6_motion/results.h5` (ignored)
- `outputs/phase6_motion/delay_doppler.png` (ignored)

Acceptance items passed:

- Multi-snapshot CFR shape `(3, 1, 1, 1, 1, 8)` matches `num_time_steps=3`.
- `/motion` group written with `snapshot_id`, `timestamp_s`, `sampling_frequency_hz`, `num_time_steps`, `mobility_mode`.
- `timestamp_s` is monotonically increasing.
- Static device state has zero velocity; non-zero velocity configurable.
- `/paths/samples/doppler_hz` field exists with finite Doppler values.
- Delay-Doppler visualization smoke output generated.
- Schema validator handles both 5D and 6D truth CFR.
- `MotionSpec` rejects non-increasing timestamps.

Current git commit hash:

- Before this Phase 6 commit: `6c31fec`.
- Phase 6 implementation commit: `c9c16bf`.

Next step:

- Continue from Phase 7: calibration and diagnostics.

Known issues or blockers:

- No Phase 6 blockers.
- Existing uncommitted document changes outside this phase were preserved: `docs/README.md` and `docs/12_final_acceptance_checklist.md`.

## 2026-05-06 - Phase 7

Current completed phase: Phase 7.

This round completed:

- Added `CalibrationResult` domain model with synthetic default profile.
- Added `DiagnosticsReport` domain model with per-link aggregate statistics (median NMSE, SNR, phase error, detection/failure rates, worst-link).
- Added `/calibration` HDF5 group writer (profile_id, fitted_parameters, validation_metrics).
- Updated schema validator for calibration datasets.
- Updated `ReceiverSpec.calibration_profile_id` default from "none" to "synthetic_default".
- Wired diagnostics into manifest.json.

Commands and results:

- `uv run pytest tests/unit tests/integration -k "calibration or diagnostics"`: passed, 9 tests.
- `uv run python -m sionna_measurement_sim.app.cli run-observation --output-dir outputs/phase7_calibration --snr-db 40`: passed.
- `uv run ruff check .`: passed.
- `uv run pytest`: passed, 65 tests.

Key files generated:

- `tests/unit/test_diagnostics.py`
- `tests/integration/test_calibration.py`
- `artifacts/phase_reports/phase_7_acceptance.md`
- `outputs/phase7_calibration/results.h5` (ignored)

Acceptance items passed:

- `/calibration/profile_id` = "synthetic_default" in HDF5.
- `/calibration/fitted_parameters` and `/calibration/validation_metrics` contain valid JSON.
- `ReceiverSpec.calibration_profile_id` = "synthetic_default" (actionable).
- `DiagnosticsReport.from_evaluation()` computes median NMSE, SNR, phase error, detection/estimation failure rates, worst-link index.
- Diagnostics summary written to manifest.json.
- Schema validator checks `/calibration` datasets when group present.

Current git commit hash:

- Before this Phase 7 commit: `91582b0`.

Next step:

- Continue from Phase 8: batch processing and performance.

Known issues or blockers:

- No Phase 7 blockers.
- Existing uncommitted document changes outside this phase were preserved: `docs/README.md` and `docs/12_final_acceptance_checklist.md`.
