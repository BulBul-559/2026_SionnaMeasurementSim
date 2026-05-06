# Phase 1 Acceptance

Status: passed

Scope:

- Domain dataclasses for topology, antenna, frequency, device state, RT truth, path samples, and full truth-only result.
- HDF5 writer/reader skeleton that consumes only domain models.
- HDF5 schema validator for minimal truth files.
- Schema tests for required datasets, dtype/shape readback, missing schema version rejection, and forbidden `/channel/cfr`.

Commands run:

```bash
uv run pytest tests/unit tests/schema
uv run python scripts/write_phase1_fixture.py
uv run pytest
uv run ruff check .
git status --short --ignored
```

Results:

- `uv run pytest tests/unit tests/schema`: passed, 14 tests.
- `uv run python scripts/write_phase1_fixture.py`: passed; generated `outputs/phase1_schema/results.h5`.
- `uv run pytest`: passed, 14 tests.
- `uv run ruff check .`: passed.
- `git status --short --ignored`: reviewed; `outputs/`, `.venv/`, caches, `old`, and large `scene.obj` are ignored.

Acceptance items:

- Minimal HDF5 contains required Phase 1 fields.
- Readback verifies dtype, shape, unit attributes, and index order.
- Tests pass without importing Sionna.
- Schema validator rejects files missing `/meta/schema_version`.
- Writer does not create `/channel/cfr`.
