---
name: sionna-project-health-check
description: Use when evaluating SionnaMeasurementSim project structure, architecture health, long-term maintainability, protocol extensibility, responsibility boundaries, duplicated logic, technical debt, or P0-P5 engineering risks; creates or updates Chinese docs/check health-check reports using the project scorecard.
---

# Sionna Project Health Check

Use this with `sionna-dev-workflow` for architecture and maintainability reviews of this
repository.

## Workflow

1. Read current project truth before scoring:
   - `docs/agent_handoff.md`
   - `docs/sys/README.md`
   - `README.md`
   - `config/README.md`
   - task-relevant `docs/sys/*.md`
2. Inspect the repo with targeted, non-destructive commands:
   - `git status --short --branch`
   - `find sionna_measurement_sim tests docs/sys config .codex/skills -maxdepth ...`
   - `rg` for interfaces, TODOs, duplicated names, schema versions, registry entries, and role semantics.
3. Run a concept ownership / single-source-of-truth pass before scoring.
4. Do not recursively scan or operate on `data/` or `outputs/`.
5. If the user asks for a formal health check, create a dated report from
   `docs/check/project_health_scorecard_template_v1.md` named
   `docs/check/YYYY-MM-DD_project_health_check.md`.
6. Write formal health-check reports in Chinese unless the user explicitly asks for another
   language. Keep file paths, commands, dataset names, maturity levels, and P0-P5 priorities in
   their original form when clearer.
7. If the user only asks for analysis, explain findings without creating a report unless they
   request persistence.

## Evidence To Collect

- Directory and module layout, including unusually large files or concentrated responsibilities.
- Dependency direction and whether `domain/` and `io/` stay free of Sionna/runtime coupling.
- PHY registry, common link abstractions, protocol-private logic boundaries, and extension points.
- Config schema, HDF5 writer, schema validator, and docs alignment.
- BS/UE role-view vs TX/RX link-view handling.
- Tests by category: unit, integration, schema, statistical, smoke, and failure-path tests.
- Documentation freshness: handoff, sys docs, config README, TODO docs, and local skills.
- Git state, untracked files, recent commits, and validation commands already available.

## Concept Ownership Pass

This pass is mandatory for formal reports. It catches structural drift that plain file-size,
test-count, or layer checks miss.

1. List core concepts likely to cross module boundaries:
   - config models, schema version, HDF5 dataset paths, PHY standard names, registry keys,
     BS/UE vs TX/RX semantics, ranging estimators, SRS/PUSCH resources, manifest shard identity,
     array spectrum labels, and visualization source names.
2. For each concept, identify:
   - source of truth
   - runtime representation
   - mapper or adapter
   - output contract
   - validator
   - docs
   - tests
3. Classify the concept:
   - `Single source`: one owner; other code imports/references it.
   - `Adapter boundary`: multiple representations are intentional and protected by a mapper plus
     equivalence tests.
   - `Duplicate definition`: fields/defaults/constants/validation are repeated without a clear
     mapper/test.
   - `Drift observed`: repeated definitions disagree or have caused behavior/output differences.
4. If this pass is skipped, say so in blind spots and cap related scorecard items at `Partial`.

Useful searches:

```bash
rg -n "class .*Config|SCHEMA_VERSION|PHY_REGISTRY|RangingConfig|/observation|/waveform" sionna_measurement_sim tests docs
rg -n "\"nr_srs\"|\"nr_pusch\"|\"custom_ofdm\"|srs_|dmrs_|ranging" sionna_measurement_sim tests docs
```

## Scoring Rules

- Use `docs/check/project_health_scorecard_template_v1.md`.
- Keep the total at 100 points and preserve v1 weights unless the user explicitly asks for a new
  template version.
- Score every subitem as one of: `Missing`, `Weak`, `Partial`, `Good`, `Excellent`.
- Convert each subitem using the template fractions: `0%`, `25%`, `50%`, `75%`, `100%`.
- Write concrete evidence and confidence for every module.
- Separate facts from judgment. Mention file paths, commands, or observed structure before drawing
  conclusions.
- Do not force severe risks. If there are no P0/P1 risks, state that directly.

## Risk Rules

Use the template's P0-P5 scale:

- `P0`: wrong data, wrong scientific conclusions, main-flow breakage, or destructive operations.
- `P1`: core experiment validity risk or near-term blocker.
- `P2`: structural debt that blocks extension or causes duplicated development.
- `P3`: local maintainability, testing, performance, or documentation gap.
- `P4`: minor cleanup, readability, or workflow improvement.
- `P5`: idea, long-term enhancement, or unproven concern.

Each risk must include: `ID`, `Priority`, `Area`, `Evidence`, `Impact`, `Likelihood`,
`Recommendation`, `Owner or Trigger`, and `Status`.

## Output Expectations

- For a formal report, save the report under `docs/check/` and summarize the total score, strongest
  areas, weakest areas, and highest-priority risks in Chinese in the final response.
- For a non-persistent analysis, still use the same rubric mentally and mention that no report was
  written.
- Preserve unrelated local files, especially ignored or untracked analysis scripts.
- Run `git diff --check` after editing docs/skills. Run broader tests only if code or behavior
  changes.
