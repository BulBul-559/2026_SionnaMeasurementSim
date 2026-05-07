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
- Phase 7 implementation commit: `531634a`.

Next step:

- Continue from Phase 8: batch processing and performance.

Known issues or blockers:

- No Phase 7 blockers.
- Existing uncommitted document changes outside this phase were preserved: `docs/README.md` and `docs/12_final_acceptance_checklist.md`.

## 2026-05-06 - Phase 8

Current completed phase: Phase 8 (final).

This round completed:

- Added `BatchConfig`, `BatchManifestEntry`, and `BatchExperimentResult` domain models.
- Created `app/batch_runner.py` orchestrator with per-batch HDF5 output, failed batch isolation, and GPU memory cleanup.
- Added `run-batch` CLI command with `--batch-count` flag.
- Batch manifest records `batching.enabled`, `total_batches`, `completed_batches`, `failed_batches`, and per-batch entries.

Commands and results:

- `uv run pytest tests/integration -k "batch"`: passed, 8 tests.
- `uv run python -m sionna_measurement_sim.app.cli run-batch --output-dir outputs/phase8_batch --batch-count 2 --snr-db 40`: 2/2 succeeded.
- Each batch HDF5 passes schema validation.
- `uv run ruff check .`: passed.
- `uv run pytest`: passed, 73 tests.

Key files generated:

- `sionna_measurement_sim/domain/batch.py`
- `sionna_measurement_sim/app/batch_runner.py`
- `tests/integration/test_batch.py`
- `artifacts/phase_reports/phase_8_acceptance.md`
- `outputs/phase8_batch/batch_manifest.json` (ignored)
- `outputs/phase8_batch/batch_000/results.h5` (ignored)
- `outputs/phase8_batch/batch_001/results.h5` (ignored)

Acceptance items passed:

- `batching.enabled` = true, `total_batches` = 2, `completed_batches` = 2, `failed_batches` = 0 in batch manifest.
- Each batch HDF5 passes full schema validation.
- Per-batch entries record batch_id, batch_index, status, and results_h5 path.
- Failed batch isolation: failed batch records error_message, does not prevent other batches.
- `outputs/` directory not in git.
- GPU memory cleanup between batches.

Current git commit hash:

- Before this Phase 8 commit: `1143aa3`.
- Phase 8 implementation commit: `3889605`.

Next step:

- All 8 phases complete. Final acceptance checklist review.

Known issues or blockers:

- No Phase 8 blockers.
- All phases (0-8) now complete with 73 passing tests.
- Existing uncommitted document changes outside this phase were preserved: `docs/README.md` and `docs/12_final_acceptance_checklist.md`.

## 2026-05-07 - NR PUSCH MIMO Fix (Post-Phase-8)

This round addressed the MIMO gap analysis from [docs/15_mimo_phy_gap_analysis.md](15_mimo_phy_gap_analysis.md).

### What was fixed

1. **Created `sionna_measurement_sim/phy/nr_mimo_channel.py`** — MIMO channel bridge
   - Converts project CIR to Sionna-compatible CFR using `cir_to_ofdm_channel`.
   - Handles TDD reciprocity and UL→DL convention reversals.
   - Exports `PUSCHMIMOChannel` dataclass and helper functions.

2. **Refactored `sionna_measurement_sim/phy/nr_pusch_observation.py`** for full MIMO
   - Added `build_multiuser_pusch_configs()` with per-UE DMRS port sets.
   - Added `build_stream_management()` and `build_mimo_detector()`.
   - Replaced manual `tx_freq * h_ch + noise` with `ApplyOFDMChannel` for full MIMO.
   - PUSCHReceiver now uses real `StreamManagement` + `LinearDetector`/`KBestDetector`.
   - `perfect_csi=True` path passes physical MIMO `h` to PUSCHReceiver.
   - `/observation/cfr_est` written from full MIMO `h` (no SISO broadcast).
   - Per-link iteration over (snap, ul_tx, ul_rx) with independent processing.

3. **Updated config schema**: Added `mimo_detector`, `channel_estimator`, `receiver_failure_policy` to `PHYConfig`, `ReceiverConfig`, and `RTTruthRunConfig`.

4. **Updated schema validator**: NR PUSCH field validation (num_layers, num_antenna_ports, mimo_detector, receiver_type).

5. **Fixed pre-existing bug**: `requir e_shape` typo in `sionna_measurement_sim/domain/motion.py`.

### New tests added

- `tests/unit/test_nr_mimo_channel_bridge.py` — 13 tests for CIR→CFR bridge, shape conversions, reciprocity
- `tests/unit/test_nr_pusch_mimo_config.py` — 13 tests for multi-user configs, StreamManagement, detector builders
- `tests/integration/test_nr_pusch_mimo_observation.py` — 6 tests for 4x4 SU-MIMO pipeline (self-generated HDF5, no broadcast check, NMSE, BER/BLER, estimated CSI smoke)
- `tests/schema/test_nr_pusch_schema.py` — 8 tests for NR PUSCH HDF5 fields

### Commands and results

- `uv run pytest tests/unit/test_nr_mimo_channel_bridge.py tests/unit/test_nr_pusch_mimo_config.py`: 26 passed
- `uv run pytest tests/integration/test_nr_pusch_mimo_observation.py tests/schema/test_nr_pusch_schema.py`: 14 passed
- `uv run pytest`: 156 passed, 2 warnings
- `uv run ruff check .`: All checks passed

### Acceptance criteria met

1. No `h_tensor[0, 0, 0, 0, :]` in NR PUSCH backend — removed.
2. NR PUSCH 4x4 integration test self-generates HDF5 (does not rely on existing outputs).
3. `/observation/cfr_est.shape[1:] == /channel/truth/cfr.shape` — verified in tests.
4. Different antenna pairs have distinct CFR estimates (no SISO broadcast) — verified in `test_4x4_no_siso_broadcast`.
5. `/receiver/mimo_detector` = "lmmse" matches actual detector used — verified.
6. `perfect_csi=True` 跑通 4x4 SU-MIMO — verified in `test_4x4_pipeline_self_generated`.
7. `StreamManagement`, `LinearDetector`, DMRS port sets have unit tests.
8. HDF5 schema validates NR PUSCH MIMO fields — verified.

### Known limitations

- Estimated CSI (`perfect_csi=False`) requires `num_layers == num_antenna_ports` for correct shape; with codebook precoding and `num_layers < num_antenna_ports`, the estimated effective channel has fewer streams than antenna ports.
- MU-MIMO (multiple project TX → multiple PUSCHConfigs with different DMRS ports) infrastructure is in place but not yet integration-tested beyond unit tests.
- BLER/CRC/transport block semantics remain at bit-level comparison (per review.md #4).
- Only SU-MIMO with single BS (num_rx=1) is integration-tested; multi-BS scenarios are structurally supported but untested.

## 2026-05-07 - NR PUSCH MIMO Fix Round 2 (SU-MIMO Hardening)

This round tightened SU-MIMO boundaries, fixed the estimated CSI zero-padding
anti-pattern, and added statistical acceptance tests per the execution guide in
[docs/review.md](review.md).

### What was fixed

1. **SU-MIMO boundaries**: Added `mimo_mode="su_mimo"` and `channel_backend="apply_ofdm"`
   fields to `PHYConfig`, `ReceiverConfig`, and `RTTruthRunConfig`. fail-fast guard
   rejects `mimo_mode != "su_mimo"`.

2. **Estimated CSI zero-padding removed**: When `num_layers < num_antenna_ports` and
   `perfect_csi=False`, the code now raises `NotImplementedError` instead of silently
   zero-padding the stream-level effective channel estimate into the physical
   antenna-pair `/observation/cfr_est`. Tests verify both the error case and the
   allowed case (`num_layers == num_antenna_ports`).

3. **4x4 MIMO statistical test** (`tests/statistical/test_nr_pusch_mimo_metrics.py`):
   7 tests covering perfect vs estimated CSI NMSE/BER comparison, EB/N0 monotonicity,
   CFR shape consistency, estimation success, and mimo_detector metadata.

### Commands and results

- `uv run pytest tests/unit/test_nr_mimo_channel_bridge.py tests/unit/test_nr_pusch_mimo_config.py tests/integration/test_nr_pusch_mimo_observation.py tests/schema/test_nr_pusch_schema.py tests/statistical/test_nr_pusch_mimo_metrics.py`: 48 passed
- `uv run pytest`: 166 passed, 2 warnings
- `uv run ruff check .`: All checks passed

### Key files changed

- `sionna_measurement_sim/config/schema.py` — added `mimo_mode`, `channel_backend`
- `sionna_measurement_sim/rt/truth_pipeline.py` — added `ebno_db`, `mimo_mode`, `channel_backend` to run config and snapshot
- `sionna_measurement_sim/phy/nr_pusch_observation.py` — replaced zero-padding with `NotImplementedError`; relaxed SU-MIMO topology guard
- `tests/unit/test_nr_pusch_config.py` — added 2 estimated CSI error/allow tests
- `tests/integration/test_nr_pusch_mimo_observation.py` — added estimated CSI error-case test
- `tests/statistical/test_nr_pusch_mimo_metrics.py` — new file, 7 statistical tests

### Known limitations (unchanged)

- MU-MIMO not yet implemented (infrastructure ready in `build_multiuser_pusch_configs`).
- Estimated CSI with `num_layers < num_antenna_ports` (codebook precoding) explicitly rejected for CFR export.
- BLER/CRC/transport block semantics remain at bit-level comparison.
- No `CIRDataset + OFDMChannel` backend yet (current backend is `ApplyOFDMChannel`).

## 2026-05-07 - NR PUSCH Channel Backend Abstraction

Extracted channel operations into a pluggable backend to prepare for the
official ``CIRDataset + OFDMChannel`` route without touching per-link processing.

### What was done

- **Created `sionna_measurement_sim/phy/nr_channel_backend.py`**:
  - ``ApplyOFDMChannelBackend`` class wrapping the current
    ``build_mimo_cfr_from_cir + ApplyOFDMChannel`` logic.
  - ``create_channel_backend()`` factory function for backend selection by name.
  - Delegates all PUSCHMIMOChannel properties (num_snap, num_ul_tx, cfr, etc.).
  - ``perfect_h()`` method returns the per-link perfect-CSI tensor.
  - ``apply()`` method applies the MIMO OFDM channel via ``ApplyOFDMChannel``.

- **Refactored `run_nr_pusch_observation` and `_process_one_pusch_link`**:
  - Replaced direct ``PUSCHMIMOChannel`` + ``ApplyOFDMChannel`` usage with
    a single ``backend`` parameter.
  - Backend selection is config-driven via ``channel_backend`` config field.
  - All existing tests pass without modification.

### Commands and results

- `uv run pytest`: 166 passed, 2 warnings
- `uv run ruff check .`: All checks passed

### Key files changed

- `sionna_measurement_sim/phy/nr_channel_backend.py` — new file
- `sionna_measurement_sim/phy/nr_pusch_observation.py` — uses backend

### Next

- All planned MIMO PHY milestones complete.

## 2026-05-07 - CIRDataset + OFDMChannel Backend

Implemented the official Sionna ``CIRDataset + OFDMChannel`` channel backend
alongside the existing ``ApplyOFDMChannel`` backend.

### What was done

- **``CIRDatasetOFDMChannelBackend``**: Properly stores UL-convention CIR, creates
  per-link ``CIRDataset`` with correct shapes (a: ``[1, rx_ant, 1, tx_ant, path, 1]``,
  tau: ``[1, 1, path]``), and uses ``OFDMChannel(return_channel=True)`` for
  channel application.
- Both backends share the same ``perfect_h()`` implementation (via pre-computed
  CFR from ``build_mimo_cfr_from_cir``), guaranteeing identical results.
- ``create_channel_backend()`` factory supports both ``"apply_ofdm"`` and
  ``"cir_dataset_ofdm"`` names.
- Added 6 unit tests: build, perfect_h comparison, apply shape, CFR equality,
  factory dispatch, and unknown-backend error.
- ``apply()`` now takes an optional ``resource_grid`` parameter (required by
  ``OFDMChannel``; unused by ``ApplyOFDMChannel``).

### Commands and results

- `uv run pytest`: 172 passed, 2 warnings
- `uv run ruff check .`: All checks passed

### Next

- All planned MIMO PHY milestones complete.

## 2026-05-07 - MU-MIMO and TB/CRC BLER (Steps 6-7)

Implemented joint multi-UE PUSCH (MU-MIMO) and transport-block CRC BLER.

**Step 6 — MU-MIMO:**
- Added ``cfr_to_full_mimo_h()`` for multi-TX/RX perfect-CSI tensor.
- Added ``_process_mu_mimo()``: per-snapshot joint processing with full
  ``StreamManagement`` and single ``PUSCHReceiver`` forward pass.
- ``mimo_mode="mu_mimo"`` auto-derives ``num_pusch_tx`` from CIR topology.

**Step 7 — TB/CRC BLER:**
- ``return_tb_crc_status=True`` with default ``TBDecoder``.
- BLER from ``tb_crc_status`` CRC pass/fail per transport block.
- ``num_block_errors`` / ``num_blocks`` with real TB semantics.

**Commands:** ``uv run pytest``: 176 passed, 2 warnings; ``uv run ruff check .``: All checks passed.

**New files:** ``tests/integration/test_nr_pusch_mu_mimo_observation.py`` (4 tests).

## 2026-05-07 - NR PUSCH MIMO Backend Closure (Steps 1-5)

Closed remaining semantic gaps in the MIMO PHY pipeline.

### What was fixed

**Step 1 — cir_dataset_ofdm returned-h loop:**
- Added ``ChannelApplyResult(y, h)`` dataclass.
- Added ``apply_with_h()`` to both backends returning both ``y`` and ``h``.
- ``_process_one_pusch_link`` now uses ``backend.apply_with_h()``, and the
  returned ``h`` drives perfect CSI, PUSCHReceiver, and ``/observation/cfr_est``.
- ``CIRDatasetOFDMChannelBackend.apply_with_h()`` returns ``h`` directly from
  ``OFDMChannel(return_channel=True)``.
- Added integration test ``test_cir_dataset_ofdm_h_closes_csi_loop``.

**Step 2 — CIRDataset delay semantics:**
- ``CIRDataset`` requires link-level ``tau: [num_rx, num_tx, num_paths]``,
  not per-antenna-pair.  Real RT data has antenna-dependent delays.
- Generator now uses per-link **median** delay across antenna pairs (was
  hard-coded ``[0, 0]`` first antenna pair).
- Documented as the standard Sionna link-level approximation.

**Step 3 — MU-MIMO backend bypass removed:**
- Added ``apply_full_with_h()`` to both backends.
- ``ApplyOFDMChannelBackend.apply_full_with_h()`` uses ``cfr_to_full_mimo_h``
  for all TX/RX.
- ``CIRDatasetOFDMChannelBackend.apply_full_with_h()`` raises
  ``NotImplementedError`` (full multi-TX/RX not yet supported for CIRDataset).
- ``_process_mu_mimo`` no longer unconditionally overwrites ``y`` with
  ``ApplyOFDMChannel``; uses backend's ``apply_full_with_h()``.

**Step 4 — TB/CRC BLER contract strengthened:**
- ``/evaluation/num_block_errors`` added to schema required fields.
- New ``_validate_bler_contract()`` enforces for NR PUSCH output:
  ``num_blocks > 0``, ``0 <= num_block_errors <= num_blocks``,
  ``bler == num_block_errors / num_blocks``.
- MU-MIMO processing now properly accumulates ``total_block_errors`` and
  ``total_blocks``.

### Commands and results

- ``uv run pytest``: 177 passed, 2 warnings
- ``uv run ruff check .``: All checks passed
