---
name: sionna-todo-docs
description: Use when creating, updating, completing, reclassifying, sorting, or auditing SionnaMeasurementSim TODO documents under docs/todo; enforces category routing, short category filenames, priority lists, detail sections, README counts, and history archiving.
---

# Sionna TODO Docs

Use this with `sionna-dev-workflow` whenever the task touches project TODOs.

## Canonical Files

- `docs/todo/README.md`: index, maintenance rules, category table, active counts.
- `docs/todo/feature.md`: new features, standards completeness, algorithms, research capability.
- `docs/todo/structure.md`: contracts, readers, benchmark APIs, legacy modules, output structure.
- `docs/todo/performance.md`: runtime, memory, write speed, RT cost, GPU scheduling, visualization cost.
- `docs/todo/bug.md`: confirmed defects and regressions only.
- `docs/todo/history.md`: completed TODO archive.

Do not keep active TODO lists in `docs/sys/` or `docs/performance/`. Those docs may link to
`docs/todo/`, but `docs/todo/` is the active source of truth.

## Classification

Choose an existing category first:

- `feature`: expands what the simulator can model, estimate, validate, or study.
- `structure`: changes how data, interfaces, contracts, readers, docs, or legacy paths are organized.
- `performance`: improves cost, throughput, memory, scaling, profiling, or large-run ergonomics.
- `bug`: fixes a confirmed wrong behavior or regression with reproducible evidence.

If none fits, create a new `docs/todo/<short-noun>.md`:

- Filename: lowercase ASCII, short, descriptive, no long phrases, e.g. `quality.md`, `dataset.md`.
- Heading: `# <Title> TODO`.
- Same required structure as existing category docs.
- Add it to `docs/todo/README.md` category table and counts.

## Active TODO Format

Every active category doc must have:

1. Short intro explaining the category.
2. `## Priority List` table sorted by current importance:

   ```markdown
   | 顺位 | ID | TODO | 简述 |
   |---:|---|---|---|
   | 1 | FEAT-XXX-001 | Short title | One sentence describing what to do. |
   ```

3. `## Details` with one section per ID:

   ```markdown
   ### FEAT-XXX-001: Short title

   目的：...

   涉及模块：...

   验收标准：...

   重点提醒：...
   ```

Keep detail length proportional to uncertainty and task size. TODOs may be incomplete by nature,
but they must be precise enough to locate affected modules and know what “done” roughly means.

## ID Rules

Use stable IDs. Never renumber existing IDs just because priority order changes.

- `FEAT-*` for `feature.md`
- `STR-*` for `structure.md`
- `PERF-*` for `performance.md`
- `BUG-*` for `bug.md`
- For new categories, use a short uppercase prefix derived from the filename.

For a new TODO in an existing theme, continue that theme sequence if present, e.g.
`FEAT-SRS-009`, `FEAT-RNG-005`. For a new theme, create a clear prefix segment, e.g.
`FEAT-WIFI-001`.

## Creating Or Updating A TODO

1. Read `docs/todo/README.md`, the likely category doc, and `docs/todo/history.md`.
2. Search existing TODOs to avoid duplicates:

   ```bash
   rg -n "keyword|module|concept" docs/todo docs/sys README.md config sionna_measurement_sim tests scripts
   ```

3. Decide category and ID.
4. Insert or update the priority row.
5. Add or update the detail section.
6. Re-sort only the `Priority List` by importance. Do not reorder detail sections unless it improves
   readability; stable IDs matter more than visual order.
7. Update `docs/todo/README.md` active counts and category table if needed.
8. If links in `docs/sys/`, README, configs, or performance docs should point to the new TODO, update
   those references.

## Completing A TODO

1. Confirm the requested TODO is actually complete from code/docs/tests or the user's explicit statement.
2. Remove its row from the active category `Priority List`.
3. Remove its detail section from the active category doc.
4. Append one concise row to `docs/todo/history.md` under the matching category title:

   ```markdown
   | YYYY-MM-DD | One sentence describing the completed TODO. |
   ```

   Use the current date from the environment/user context.
5. Re-sort the remaining active priority list if completion changes importance.
6. Update `docs/todo/README.md` counts.
7. Remove or update stale links that directly referenced the old active section.

## Auditing TODOs

For audits, report:

- Active counts by category and total.
- Any category count mismatch with `docs/todo/README.md`.
- Duplicate or overlapping TODOs.
- Completed items still left in active docs.
- Active TODOs outside `docs/todo/`.
- Broken references to removed TODO docs.

Useful commands:

```bash
find docs/todo -maxdepth 1 -type f -print | sort
rg -n "TODO|FIXME|待办|后续|下一阶段" docs README.md config sionna_measurement_sim tests scripts -g '!data/**' -g '!outputs/**'
rg -n "nr_srs_standard_todo|ranging_observation_todo|nr_pusch_performance_optimization_todo" docs README.md config
```

## Validation

After edits:

```bash
git diff --check
rg -n "old-path-or-removed-id" docs README.md config sionna_measurement_sim tests scripts -g '!data/**' -g '!outputs/**'
```

Run broader tests only if code, config behavior, or generated outputs changed. For docs/skills-only
changes, `git diff --check` is usually enough; run `ruff` if Python files or scripts were touched.
