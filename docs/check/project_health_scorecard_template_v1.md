# Project Health Check Template v1

Report date: `YYYY-MM-DD`
Branch / commit: `<branch> / <commit>`
Reviewer: `<name or agent>`
Scope: `<repo-wide / selected subsystems>`
Template version: `v1`
Report language: `中文`

> 正式健康体检报告必须使用中文撰写，除非用户明确要求使用其他语言。文件路径、命令、
> HDF5 dataset 名称、成熟度等级和 P0-P5 风险优先级可以保留英文原文。

## Summary

Total score: `<0-100> / 100`

| Module | Weight | Score | Confidence | Short rationale |
|---|---:|---:|---|---|
| Architecture Layering and Responsibility Boundaries | 18 | TBD | TBD | TBD |
| Protocol and Feature Extensibility | 14 | TBD | TBD | TBD |
| Data Contracts and Reproducibility | 14 | TBD | TBD | TBD |
| Code Maintainability and Complexity | 16 | TBD | TBD | TBD |
| Testing and Validation System | 14 | TBD | TBD | TBD |
| Documentation and Onboarding | 10 | TBD | TBD | TBD |
| Runtime and Experiment Engineering | 8 | TBD | TBD | TBD |
| Change Governance | 6 | TBD | TBD | TBD |
| **Total** | **100** | **TBD** |  |  |

## Scoring Scale

Score each subitem using exactly one maturity level, then multiply by the subitem weight.

| Level | Fraction | Meaning |
|---|---:|---|
| Missing | 0% | No meaningful implementation or evidence. |
| Weak | 25% | Exists, but fragmented, fragile, or mostly manual. |
| Partial | 50% | Usable, but has clear gaps or inconsistent application. |
| Good | 75% | Mostly healthy with limited known gaps. |
| Excellent | 100% | Clear, tested, documented, and consistently applied. |

## Evidence Inventory

Record the concrete inputs used for this report.

| Evidence type | Items checked | Notes |
|---|---|---|
| Git state | `git status --short --branch`, recent commits | TBD |
| Architecture docs | `docs/agent_handoff.md`, `docs/sys/*.md` | TBD |
| Code structure | targeted `find` / `rg`, key module reads | TBD |
| Tests and validation | test tree, recent command output if run | TBD |
| Config and schema | config templates, schema/writer/validator docs | TBD |
| Known TODOs | `docs/sys/*todo*.md`, inline TODO search | TBD |

## Detailed Scorecard

### 1. Architecture Layering and Responsibility Boundaries (18)

| Item | Weight | Level | Score | Evidence | Confidence | Improvement notes |
|---|---:|---|---:|---|---|---|
| Dependency direction is clear and mostly one-way | 3 | TBD | TBD | TBD | TBD | TBD |
| Domain models stay pure and free of external framework dependencies | 3 | TBD | TBD | TBD | TBD | TBD |
| Modules follow single responsibility and cohesive ownership | 3 | TBD | TBD | TBD | TBD | TBD |
| Registry/plugin boundaries isolate swappable implementations | 3 | TBD | TBD | TBD | TBD | TBD |
| Public contracts are stable and documented | 3 | TBD | TBD | TBD | TBD | TBD |
| Directory layout is discoverable for new contributors | 3 | TBD | TBD | TBD | TBD | TBD |

### 2. Protocol and Feature Extensibility (14)

| Item | Weight | Level | Score | Evidence | Confidence | Improvement notes |
|---|---:|---|---:|---|---|---|
| Shared abstractions prevent protocol-specific duplication | 3 | TBD | TBD | TBD | TBD | TBD |
| Config extension model is clear and validated | 3 | TBD | TBD | TBD | TBD | TBD |
| Protocol-private waveform/receiver logic is isolated | 3 | TBD | TBD | TBD | TBD | TBD |
| Schema and migration strategy can handle new outputs | 3 | TBD | TBD | TBD | TBD | TBD |
| Defaults and feature flags support incremental adoption | 2 | TBD | TBD | TBD | TBD | TBD |

### 3. Data Contracts and Reproducibility (14)

| Item | Weight | Level | Score | Evidence | Confidence | Improvement notes |
|---|---:|---|---:|---|---|---|
| HDF5 schema, writer, and validator remain aligned | 3 | TBD | TBD | TBD | TBD | TBD |
| Dimension semantics and BS/UE vs TX/RX roles are explicit | 3 | TBD | TBD | TBD | TBD | TBD |
| Manifest and config snapshot preserve shard provenance | 2 | TBD | TBD | TBD | TBD | TBD |
| Random seeds and small experiments are reproducible | 2 | TBD | TBD | TBD | TBD | TBD |
| `data/` and `outputs/` are safely isolated from repo state | 2 | TBD | TBD | TBD | TBD | TBD |
| Units, attrs, and metadata are consistently written | 2 | TBD | TBD | TBD | TBD | TBD |

### 4. Code Maintainability and Complexity (16)

| Item | Weight | Level | Score | Evidence | Confidence | Improvement notes |
|---|---:|---|---:|---|---|---|
| Large files and long functions are controlled or justified | 3 | TBD | TBD | TBD | TBD | TBD |
| Repeated logic is extracted into appropriate helpers | 3 | TBD | TBD | TBD | TBD | TBD |
| Error handling is explicit and fail-fast where needed | 2 | TBD | TBD | TBD | TBD | TBD |
| Types and dataclasses express subsystem boundaries | 2 | TBD | TBD | TBD | TBD | TBD |
| Performance/resource concerns are visible in code paths | 2 | TBD | TBD | TBD | TBD | TBD |
| TODOs and technical debt are tracked in durable places | 2 | TBD | TBD | TBD | TBD | TBD |
| Dependencies and global state are kept under control | 2 | TBD | TBD | TBD | TBD | TBD |

### 5. Testing and Validation System (14)

| Item | Weight | Level | Score | Evidence | Confidence | Improvement notes |
|---|---:|---|---:|---|---|---|
| Unit tests cover pure logic and shape contracts | 3 | TBD | TBD | TBD | TBD | TBD |
| Integration and smoke tests cover main workflows | 3 | TBD | TBD | TBD | TBD | TBD |
| Schema, statistical, and regression tests catch contract drift | 3 | TBD | TBD | TBD | TBD | TBD |
| Real small-experiment summaries validate data products | 2 | TBD | TBD | TBD | TBD | TBD |
| Failure paths and invalid configs are tested | 2 | TBD | TBD | TBD | TBD | TBD |
| Standard developer commands are documented and current | 1 | TBD | TBD | TBD | TBD | TBD |

### 6. Documentation and Onboarding (10)

| Item | Weight | Level | Score | Evidence | Confidence | Improvement notes |
|---|---:|---|---:|---|---|---|
| Handoff and system docs describe current truth | 2 | TBD | TBD | TBD | TBD | TBD |
| Config README and templates stay synchronized | 2 | TBD | TBD | TBD | TBD | TBD |
| Architecture and API docs explain extension points | 2 | TBD | TBD | TBD | TBD | TBD |
| TODOs and limitations are explicit and actionable | 2 | TBD | TBD | TBD | TBD | TBD |
| Skills/runbooks capture recurring engineering workflows | 2 | TBD | TBD | TBD | TBD | TBD |

### 7. Runtime and Experiment Engineering (8)

| Item | Weight | Level | Score | Evidence | Confidence | Improvement notes |
|---|---:|---|---:|---|---|---|
| Sharding and multi-GPU execution are operationally clear | 2 | TBD | TBD | TBD | TBD | TBD |
| I/O volume and output layout are managed intentionally | 2 | TBD | TBD | TBD | TBD | TBD |
| Profiling and logs support debugging slow or failed runs | 1 | TBD | TBD | TBD | TBD | TBD |
| Local large-data paths are handled safely | 2 | TBD | TBD | TBD | TBD | TBD |
| Visualization/diagnostics support small-sample inspection | 1 | TBD | TBD | TBD | TBD | TBD |

### 8. Change Governance (6)

| Item | Weight | Level | Score | Evidence | Confidence | Improvement notes |
|---|---:|---|---:|---|---|---|
| Branch and commit discipline keeps changes reviewable | 2 | TBD | TBD | TBD | TBD | TBD |
| Docs/config/schema updates land with behavior changes | 2 | TBD | TBD | TBD | TBD | TBD |
| Compatibility and migration impact are stated clearly | 1 | TBD | TBD | TBD | TBD | TBD |
| Review, validation, and follow-up closure are visible | 1 | TBD | TBD | TBD | TBD | TBD |

## Risk List

Use P0-P5 only when evidence supports the priority. It is acceptable for a report to have no P0
or P1 risks.

| ID | Priority | Area | Evidence | Impact | Likelihood | Recommendation | Owner or Trigger | Status |
|---|---|---|---|---|---|---|---|---|
| R-001 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### Priority Definitions

| Priority | Meaning |
|---|---|
| P0 | Emergency issue that can cause wrong data, wrong scientific conclusions, main-flow breakage, or destructive operations. |
| P1 | High-probability issue affecting core experiment validity or blocking near-term mainline work. |
| P2 | Structural technical debt that will block future extension or cause duplicated development. |
| P3 | Local maintainability, testing, performance, or documentation gap that should be scheduled. |
| P4 | Minor cleanup, readability, or workflow improvement. |
| P5 | Idea, long-term improvement, or unproven concern to revisit later. |

## Recommendations

| Priority | Workstream | Recommendation | Expected benefit | Suggested validation |
|---|---|---|---|---|
| TBD | TBD | TBD | TBD | TBD |

## Exclusions and Blind Spots

- `<List files, subsystems, generated outputs, or runtime behavior not inspected.>`
- `<List tests or experiments not run.>`

## Follow-Up

- Next recommended review date: `<date or trigger>`
- Template version changes needed: `<none / proposed v2 changes>`
