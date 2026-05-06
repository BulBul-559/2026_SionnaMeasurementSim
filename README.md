# SionnaMeasurementSim

New measurement-oriented simulation system for Sionna 2.x RT truth generation, PHY observations, and HDF5 outputs.

This repository follows the phase roadmap in `docs/08_roadmap_milestones_acceptance.md`. Phase 0 establishes the uv-managed Python package, CLI entry point, and test framework only; RT, HDF5 schema writing, PHY observation, impairments, and calibration are implemented in later phases.

## Development

```bash
uv sync
uv run pytest
uv run ruff check .
uv run python -m sionna_measurement_sim.app.cli --help
```

The git repository root is this directory, `SionnaMeasurementSim/`. The historical project under `old/` is reference-only and must not be imported by new system code.
