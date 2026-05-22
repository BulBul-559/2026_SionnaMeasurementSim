---
name: sionna-dev-workflow
description: Project-specific development discipline for SionnaMeasurementSim. Use when modifying code, configs, docs, schema/HDF5, tests, scripts, simulation pipelines, or git state in this repository.
---

# Sionna Dev Workflow

## Scope

Use this workflow for repository work that changes behavior, configuration, documentation, tests, generated schemas, simulation outputs, analysis scripts, or git state. It captures the project habits that keep SionnaMeasurementSim reproducible: current docs first, small branches and commits, synchronized schema/docs/config updates, validation before merge, and careful handling of local large-data paths.

## Start Every Task

- Read the current handoff and system docs before changing behavior: `docs/agent_handoff.md`, `docs/sys/README.md`, `README.md`, `config/README.md`, and the task-relevant files under `docs/sys/`.
- Treat `docs/agent_handoff.md` and `docs/sys/` as current project truth. Treat `docs/performance/` as historical experiment records unless the user asks otherwise. Treat `docs/old/` as archival.
- Do not recursively scan or operate on `data/` or `outputs/`; they are ignored local large-data paths and may be symlinks. Touch them only through explicitly scoped files or smoke-test output directories.
- Run `git status --short --branch` before edits. Preserve unrelated user changes and untracked local analysis scripts.
- Prefer `rg` and targeted reads over broad scans.

## Git Discipline

- For large features, refactors, schema changes, simulation-output changes, or risky work, create a task branch from the current `main`, usually `codex/<feature-name>`.
- Keep commits coherent and timely. Commit code, tests, and docs for the same behavior change together after validation.
- Never stage unrelated local files. In this repository, analysis helpers such as `scripts/plot_cfr_similarity_*.py` may be intentionally untracked unless the user asks to include them.
- Use non-destructive git commands. Do not reset, checkout, or revert user changes unless explicitly requested.
- When the user asks to merge back, verify status, commit intended files, fast-forward merge to `main` when possible, and report the branch and commit hash.

## Implementation Discipline

- Follow existing architecture before inventing new abstractions: config validates inputs, domain models stay lightweight, PHY standards plug into common link/result layers, IO owns HDF5 writing and schema validation, and docs describe the contract.
- Keep BS/UE role-view semantics distinct from TX/RX link-view semantics when touching simulation, PHY, HDF5, or analysis code.
- For HDF5 or schema changes, update the writer, validator, tests, config examples, docs, and handoff in the same change. Bump schema versions only with explicit intent.
- For PHY changes, keep shared channel/impairment/result logic decoupled from standard-specific waveform builders and receivers.
- For ranging/derived labels, keep truth fields and observation estimates explicitly separated in names, docs, and tests.
- Keep edits scoped. Avoid opportunistic refactors unless they remove real duplication or unblock the requested change.

## Documentation Contract

- Update affected documentation before considering the task complete. Typical files are `README.md`, `config/README.md`, `docs/agent_handoff.md`, relevant `docs/sys/*.md`, TODO documents, and validation notes.
- After substantial behavior, schema, config, pipeline, simulation-output, or architecture changes, use the project `sionna-doc-maintenance` skill to audit whether README, config docs/templates, sys docs, TODO docs, handoff, validation notes, or legacy docs need updates.
- If defaults or config semantics change, update templates and config documentation together.
- If HDF5 fields, schema versions, or output groups change, update `docs/sys/07_config_and_h5_format.md`, schema tests, and handoff.
- If a feature remains partial, document exactly what is implemented, what is not, and the next-stage TODOs.

## Validation

- Run focused tests while developing, then run broader validation before commit or merge. The usual merge-ready baseline is:
  - `uv run ruff check sionna_measurement_sim tests scripts`
  - `uv run pytest -q`
- For simulation or data-output behavior changes, run a small real smoke test into an ignored `outputs/<named_smoke>/` directory, validate HDF5 schema, and produce a short numeric sanity summary.
- For analysis scripts, generate outputs with clear names and avoid overwriting prior result directories unless the user requests it.
- Report commands, pass/fail status, smoke output paths, and any skipped or blocked checks.

## Multi-Agent Coordination

- Use multiple agents only when the user explicitly asks for or authorizes delegation, parallel agents, or multi-agent work.
- Split parallel work by non-overlapping ownership: separate modules, independent docs, read-only exploration, validation, or analysis. Avoid overlapping write sets.
- Do critical blocking work locally when the next step depends on it. Delegate sidecar tasks that can advance in parallel.
- Tell worker agents they are not alone in the codebase, must preserve others' edits, and must list changed files.
- Integrate agent output by reviewing changes, running validation, and reconciling docs/tests before commit.

## Finish Criteria

- No required shell sessions are still running.
- `git status --short --branch` has been checked and unrelated files are called out.
- The final response states what changed, where it lives, validation results, branch/commit/merge state, and remaining TODOs or risks.
