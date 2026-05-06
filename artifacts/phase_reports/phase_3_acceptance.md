# Phase 3 Acceptance

Status: passed

Scope:

- Added TX-first `PathTable` domain model.
- Added Sionna RT path adapter for `valid`, `a`, `tau`, Doppler, AoA/AoD, interactions, objects, primitives, vertices, path type, and path depth.
- Wrote non-empty `/paths/samples` and debug `/paths/full`.
- Added path visualization smoke helper and generated an ignored path plot.

Commands run:

```bash
uv run pytest tests/adapter tests/schema tests/integration -k "path"
uv run python -m sionna_measurement_sim.app.cli run-rt-truth --output-dir outputs/phase3_paths
uv run python - <<'PY'
from sionna_measurement_sim.visualization.path_plots import plot_path_samples
print(plot_path_samples('outputs/phase3_paths/results.h5', 'outputs/phase3_paths/paths.png'))
PY
uv run ruff check .
uv run pytest
git status --short --ignored
```

Results:

- `uv run pytest tests/adapter tests/schema tests/integration -k "path"`: passed, 3 tests.
- CLI path run: passed; generated `outputs/phase3_paths/results.h5`.
- Path plot generation: passed; generated `outputs/phase3_paths/paths.png`.
- `uv run ruff check .`: passed.
- `uv run pytest`: passed, 19 tests.

Acceptance items:

- At least one valid path parsed: 26 valid paths.
- At least one NLoS path has finite intermediate vertices: 25 NLoS paths.
- `interaction_type`, `object_id`, and `primitive_id` depth dimensions align.
- `doppler_hz` and `tau_s` exist and contain finite values.
- Sample `vertices_m` includes TX/RX endpoints; active samples satisfy `vertex_count >= interaction_count + 2`.
- Path visualization smoke output exists and is non-empty.
