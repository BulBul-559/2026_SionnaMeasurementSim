# Phase 2 Acceptance

Status: passed

Scope:

- Added `sionna-rt==2.0.1` through uv.
- Added a minimal Sionna RT adapter isolated under `sionna_measurement_sim/adapters/sionna_rt/`.
- Loaded `data/scenes/test/scene.xml`, registered one TX and one RX from `test5.json`, ran `PathSolver`, and generated TX-first truth CFR.
- Added RT truth pipeline that writes `results.h5`, `manifest.json`, and `logs/run.log`.
- Added adapter and integration tests for the Phase 2 RT truth path.

Commands run:

```bash
uv add sionna-rt
uv run pytest tests/adapter tests/integration -k "rt_truth"
uv run python -m sionna_measurement_sim.app.cli run-rt-truth --output-dir outputs/phase2_rt_truth
uv run ruff check .
uv run pytest
git status --short --ignored
```

Results:

- `uv add sionna-rt`: passed; installed `sionna-rt==2.0.1`, `mitsuba==3.8.0`, and `drjit==1.3.1`.
- `uv run pytest tests/adapter tests/integration -k "rt_truth"`: passed, 2 tests.
- `uv run python -m sionna_measurement_sim.app.cli run-rt-truth --output-dir outputs/phase2_rt_truth`: passed; generated `outputs/phase2_rt_truth/results.h5`.
- `uv run ruff check .`: passed.
- `uv run pytest`: passed, 16 tests.

Acceptance items:

- H_true shape is `[tx, rx, rx_ant, tx_ant, subcarrier]`, observed as `(1, 1, 1, 1, 8)`.
- `frequencies_hz.shape[-1] == H_true.shape[-1]`.
- `H_true` dtype is `complex64`.
- At least one finite, non-NaN CFR value exists.
- Runtime fields include Sionna RT, Mitsuba, Dr.Jit, and torch version datasets.
- Manifest records scene file, config snapshot, software versions, and shape conversion details.
- HDF5 readback preserves H_true shape and dtype.

Notes:

- Phase 2 is RT-only and uses `sionna-rt`; `torch_version` is present as a runtime dataset and remains empty until the PyTorch PHY/SYS chain is introduced.
