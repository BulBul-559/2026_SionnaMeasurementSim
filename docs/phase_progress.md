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
- Phase 2 implementation commit: created immediately after this progress entry.

Next step:

- Continue from Phase 3: Path adapter and path-level samples.

Known issues or blockers:

- No Phase 2 blockers.
- Phase 2 uses RT-only `sionna-rt`; PyTorch is not installed yet because the PyTorch PHY/SYS chain starts in later phases.
- Existing uncommitted document changes outside this phase were preserved: `docs/README.md` and `docs/12_final_acceptance_checklist.md`.
