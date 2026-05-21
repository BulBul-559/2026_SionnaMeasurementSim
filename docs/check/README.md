# Project Health Check

`docs/check/` stores reusable project health-check material and dated health reports.
Use it to evaluate architecture quality, maintainability, protocol extensibility, and
engineering risk over time.

This scorecard is about structural health and future evolvability. It does not replace
functional tests, HDF5 schema validation, simulation smoke tests, or research-result
analysis.

## Files

- `project_health_scorecard_template_v1.md`: reusable v1 scorecard template.
- `YYYY-MM-DD_project_health_check.md`: dated health-check reports created from the template.

Do not overwrite old reports. If the scoring rubric changes, create a new template version
instead of editing prior reports in place.

## Required Report Content

Every formal health-check report must include:

1. Scope, date, branch/commit, and reviewer.
2. Score summary with module scores and total score out of 100.
3. Detailed score table with evidence, confidence, and improvement notes for each item.
4. Risk list graded P0 through P5.
5. Recommendations grouped by priority or workstream.
6. Explicit exclusions and blind spots.

Formal health-check reports must be written in Chinese unless the user explicitly asks for
another language. Technical identifiers such as file paths, commands, dataset names, maturity
levels, and P0-P5 risk priorities may stay in their original English form.

## Scoring Scale

Each subitem is scored with one of these maturity levels, then multiplied by the subitem
weight:

| Level | Fraction | Meaning |
|---|---:|---|
| Missing | 0% | No meaningful implementation or evidence. |
| Weak | 25% | Exists, but fragmented, fragile, or mostly manual. |
| Partial | 50% | Usable, but has clear gaps or inconsistent application. |
| Good | 75% | Mostly healthy with limited known gaps. |
| Excellent | 100% | Clear, tested, documented, and consistently applied. |

Scores must be evidence-based. Prefer a lower score with clear evidence over an optimistic
score based on intent.

## Risk Priority

| Priority | Meaning |
|---|---|
| P0 | Emergency issue that can cause wrong data, wrong scientific conclusions, main-flow breakage, or destructive operations. |
| P1 | High-probability issue affecting core experiment validity or blocking near-term mainline work. |
| P2 | Structural technical debt that will block future extension or cause duplicated development. |
| P3 | Local maintainability, testing, performance, or documentation gap that should be scheduled. |
| P4 | Minor cleanup, readability, or workflow improvement. |
| P5 | Idea, long-term improvement, or unproven concern to revisit later. |

Do not invent severe risks to fill the table. If no P0 or P1 risk exists, say so.

## Evidence Rules

- Use current truth from `docs/agent_handoff.md` and `docs/sys/`.
- Use targeted `rg`, `find`, `git status`, and focused file reads.
- Do not recursively scan or operate on `data/` or `outputs/`.
- Keep facts and judgment separate: cite files, commands, or observed structure before scoring.
- Record confidence as `High`, `Medium`, or `Low` for each module and risk.
