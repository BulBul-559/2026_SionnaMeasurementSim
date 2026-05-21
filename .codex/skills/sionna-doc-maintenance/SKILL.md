---
name: sionna-doc-maintenance
description: Use after substantial SionnaMeasurementSim changes to audit and update project documentation, including README, config docs/templates, docs/sys, docs/todo, docs/agent_handoff, validation notes, and docs/legacy migration decisions.
---

# Sionna Documentation Maintenance

Use this with `sionna-dev-workflow` after large or user-visible project changes. If the change
creates, completes, or reclassifies TODOs, also use `sionna-todo-docs`.

## Goal

Keep documentation synchronized with the current system after behavior, schema, config, workflow,
simulation output, or architecture changes. Decide whether existing docs need updating, a new doc is
needed, or stale docs should move to `docs/legacy/`.

## Required Context

Before editing docs, read only the relevant current-truth files:

- `docs/agent_handoff.md`
- `docs/sys/README.md`
- `README.md`
- `config/README.md`
- Task-relevant `docs/sys/*.md`
- `docs/todo/README.md` if TODOs are affected

Treat `docs/sys/` and `docs/agent_handoff.md` as current truth. Treat `docs/performance/` as
historical experiment records. Use `docs/legacy/` for obsolete docs that should remain available for
manual review.

## Documentation Impact Checklist

For any substantial change, check these surfaces:

- Top-level `README.md`: current capabilities, quickstart, docs index, HDF5/schema summary.
- `config/README.md`: effective config fields, defaults, warnings, production guidance.
- Config templates in `config/defaults/` and `config/perf/`: comments, defaults, paths, schema fields.
- `docs/agent_handoff.md`: current project state, baseline guidance, common pitfalls, doc map.
- `docs/sys/README.md`: docs index and current facts.
- `docs/sys/00_project_overview.md`: architecture/dataflow when major components move.
- `docs/sys/01_app_and_config.md`: CLI, config loading, sharding, visualization entrypoints.
- `docs/sys/04_rt_pipeline.md`: BS/UE vs TX/RX mapping, pipeline order, shard behavior.
- `docs/sys/05_phy_observation.md`: PHY modules, common link, impairments, waveform/receiver behavior.
- `docs/sys/06_io_and_testing.md`: writer/validator/manifest/test expectations.
- `docs/sys/07_config_and_h5_format.md`: schema version, HDF5 groups/datasets, shapes, attrs.
- `docs/sys/phy_module_development.md`: extension rules after PHY/link architecture changes.
- `docs/sys/indoor_fr1_100mhz_validation.md`: current production baseline and cost guidance.
- `docs/todo/*`: active follow-ups, completed history, counts, categories.
- `docs/check/*`: update only when the task is a health-check/report task.
- `docs/performance/*`: update status banners or links only when historical docs point to stale current truth.

## Decide Update vs New Doc vs Legacy

Update an existing doc when:

- The concept already has a clear home.
- The change modifies current behavior, fields, defaults, commands, or caveats.
- A small status note prevents old experimental conclusions from being mistaken for defaults.

Create a new doc when:

- The feature has a distinct workflow or contract too large for an existing page.
- A new module family needs an implementation guide or validation note.
- A new repeated procedure would otherwise be scattered across README/sys/config docs.

Move a doc to `docs/legacy/` when:

- It is no longer current truth and would mislead agents/users even with a short status banner.
- Its contents are superseded by current docs and only worth keeping for manual inspection.
- It is not useful as a historical performance record.

When moving to legacy:

- Preserve enough filename context; prefix with date or topic if needed.
- Add or update links in current docs so users know the replacement location.
- Do not move `docs/performance/` experiment records just because they are old; only move them if they
  are actively misleading and not valuable as historical reports.

## HDF5 / Schema Changes

If schema, HDF5 paths, shapes, attrs, or output groups change:

1. Update schema version docs and any config/schema references.
2. Update `docs/sys/07_config_and_h5_format.md`.
3. Update `docs/sys/06_io_and_testing.md` if validator/test rules changed.
4. Update README HDF5 summary if the change is user-visible.
5. Update `docs/agent_handoff.md` with new current truth.
6. Check schema tests and writer/validator docs references.

## Config / Defaults Changes

If config fields, defaults, templates, or CLI override behavior changed:

1. Update `config/README.md`.
2. Update affected YAML comments/templates.
3. Update `docs/sys/01_app_and_config.md` if the behavior is structural.
4. Update README quickstart if commands or templates changed.
5. Update handoff if the change affects recommended baseline.

## PHY / Pipeline Changes

If PHY, RT pipeline, role semantics, array spectrum, ranging, visualization, or shard behavior changed:

1. Update the relevant `docs/sys/*.md` owner page.
2. Update `docs/agent_handoff.md` if it affects current project truth or common pitfalls.
3. Update `docs/todo/*` for remaining or completed follow-ups.
4. Update README capability bullets if user-visible.
5. Update config templates/docs if defaults or knobs changed.

## TODO Handling

Use `sionna-todo-docs` rules when:

- A change leaves follow-up work.
- A TODO is completed and should move to history.
- A TODO changes category or priority.
- A new TODO category is needed.

Do not leave active TODO lists in `docs/sys/` or `docs/performance/`.

## Audit Commands

Use targeted searches; do not recurse into `data/` or `outputs/`.

```bash
git status --short --branch
rg -n "old_name|old_path|schema 1\\.|SCHEMA_VERSION|deprecated|legacy|TODO" README.md config docs sionna_measurement_sim tests scripts -g '!data/**' -g '!outputs/**'
rg -n "docs/sys/.*todo|docs/performance/.*todo|rtt_like|pilot_code|srs_cfr_est" README.md config docs sionna_measurement_sim tests scripts -g '!data/**' -g '!outputs/**'
find docs -maxdepth 2 -type f | sort
```

## Validation

For docs-only changes:

```bash
git diff --check
```

Also run `rg` checks for renamed paths/fields. Run broader tests when docs changes accompany code,
schema, config, or generated-output changes:

```bash
uv run ruff check sionna_measurement_sim tests scripts
uv run pytest -q
```

## Final Response

Report:

- Which docs were updated, created, or moved to legacy.
- Any TODO updates and active-count changes.
- Validation commands and results.
- Any docs intentionally left unchanged and why.
- Unrelated local files preserved.
