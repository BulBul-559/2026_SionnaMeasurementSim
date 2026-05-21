# 项目健康体检报告

报告日期：`2026-05-22`
分支 / 提交：`codex/docs-todo-consolidation / 1f0c93a`
评审者：`Codex`
范围：`repo-wide`
模板版本：`v1`
报告语言：`中文`

## 总结

总分：`90.25 / 100`

整体判断：项目结构健康度较高，当前已经形成清晰的 `app / config / domain / rt / phy / io / ranging / visualization` 分层，PUSCH/SRS 共享通用 clean channel + impairment/AWGN 链路，HDF5 schema 与测试体系也比较成熟。主要扣分集中在长期维护成本：若干核心模块过大、writer/schema validator 与协议字段耦合较强、多 shard 下游 reader 尚未产品化，以及 `custom_ofdm` legacy 路径仍需决策。

本次未发现需要立即处理的 P0/P1 风险。最高风险为 P2，属于结构性技术债，不会阻止当前生产运行，但会影响下一阶段协议扩展和下游训练/分析效率。

| 模块 | 权重 | 得分 | 置信度 | 简要原因 |
|---|---:|---:|---|---|
| 架构分层与职责边界 | 18 | 13.50 | High | 分层和 domain 纯度好，但多个核心文件职责集中。 |
| 协议与功能扩展性 | 14 | 10.50 | High | common link、registry、ranging runner 已成型；新增协议仍需改 CLI/config/writer/schema 多处。 |
| 数据契约与可复现性 | 14 | 12.25 | High | schema 1.4.0、TX/RX 语义、manifest 和大数据路径约束清晰。 |
| 代码可维护性与复杂度 | 16 | 11.75 | Medium | 大文件、长函数和 writer/schema 重复校验是主要技术债。 |
| 测试与验证体系 | 14 | 12.25 | High | 单元/schema/statistical/integration 测试覆盖强，完整 pytest 通过。 |
| 文档与上手成本 | 10 | 9.00 | High | handoff/sys/config/todo/skills 较完整，少数历史/实验口径需持续清理。 |
| 运行与实验工程 | 8 | 6.50 | Medium | shard、多 GPU、profiling 和 visualization 均具备，但性能 benchmark 入口仍在 TODO。 |
| 变更治理 | 6 | 4.50 | Medium | 分支/提交/文档同步习惯较好；本地 untracked 分析脚本和未推送分支需继续留意。 |
| **总计** | **100** | **90.25** |  |  |

## 证据清单

| 证据类型 | 已检查内容 | 备注 |
|---|---|---|
| Git 状态 | `git status --short --branch`, `git log --oneline --decorate -8` | 当前在 `codex/docs-todo-consolidation`；存在 4 个未跟踪 CFR similarity 分析脚本。 |
| 当前事实文档 | `docs/agent_handoff.md`, `docs/sys/README.md`, `README.md`, `config/README.md` | 当前 schema `1.4.0`；`docs/todo/` 是 active TODO 入口。 |
| 架构文档 | `docs/sys/00_project_overview.md`, `docs/sys/05_phy_observation.md`, `docs/sys/phy_module_development.md` | 明确分层、TX/RX 语义、PHY registry、common link 和新增 PHY module 规范。 |
| 代码结构 | targeted `find`, line-count script, AST long-function scan | 未递归扫描 `data/` 或 `outputs/`。 |
| 依赖方向 | AST import scan for `domain/` and `io/` | 未发现 `domain/` 或 `io/` 直接 import `sionna`、`torch`、`tensorflow`、`mitsuba`、`drjit`。 |
| 测试分布 | AST test count | `unit=255`, `integration=68`, `schema=43`, `statistical=43`, `adapter=29` test functions。 |
| 验证命令 | `uv run ruff check sionna_measurement_sim tests scripts`; `uv run pytest -q` | ruff 通过；pytest `249 passed, 19 skipped, 1 warning in 53.41s`。 |
| TODO | `docs/todo/README.md`, `feature.md`, `structure.md`, `performance.md`, `bug.md` | active TODO 共 24 个；bug TODO 当前为 0。 |

## 详细评分

### 1. 架构分层与职责边界 (18)

| 条目 | 权重 | 等级 | 得分 | 证据 | 置信度 | 改进建议 |
|---|---:|---|---:|---|---|---|
| 依赖方向清晰且基本单向 | 3 | Good | 2.25 | `docs/sys/00_project_overview.md` 定义 app/config/domain/rt/phy/io 分层；AST 检查未发现 domain/io 直接 import Sionna/Torch。 | High | 后续可加 import-boundary 测试，防止回归。 |
| domain 模型保持纯净 | 3 | Excellent | 3.00 | `domain/` 未直接 import `sionna`、`torch` 等框架；文档要求 domain 零 Sionna 依赖。 | High | 维持现状。 |
| 模块满足单一职责和内聚性 | 3 | Partial | 1.50 | 大文件集中：`nr_pusch_observation.py` 1728 行、`truth_pipeline.py` 1537 行、`visualization/report.py` 1256 行、`schema_validator.py` 1095 行、`hdf5_writer.py` 924 行；AST 显示多个 150 行以上函数/类。 | High | 优先拆 orchestration、writer/schema validator、visualization report 和 PUSCH processing path。 |
| registry/plugin 边界隔离可替换实现 | 3 | Good | 2.25 | `phy/modules.py` 提供 `PHY_REGISTRY`；`phy_module_development.md` 说明新增 PHY module 流程。 | High | 把 writer/schema 的协议字段注册化，减少新增协议时的跨文件改动。 |
| 公共契约稳定且有文档 | 3 | Good | 2.25 | HDF5 schema、TX/RX shape、role-view 语义在 README、sys docs、config README 中均有说明。 | High | 增加 machine-readable schema 摘要或契约测试生成文档，降低手工同步成本。 |
| 目录布局对新贡献者可发现 | 3 | Good | 2.25 | `docs/sys/00_project_overview.md` 有代码目录到文档映射；`docs/agent_handoff.md` 有深读地图。 | High | 补一页“从需求到落盘字段”的快速导航会更友好。 |

小结：`13.50 / 18`。架构边界总体好，主要弱点是职责集中。

### 2. 协议与功能扩展性 (14)

| 条目 | 权重 | 等级 | 得分 | 证据 | 置信度 | 改进建议 |
|---|---:|---|---:|---|---|---|
| 共享抽象减少协议重复 | 3 | Good | 2.25 | PUSCH/SRS 共享 `ObservationImpairmentChain`；ranging 在 pipeline 级读取统一 `/observation/cfr_est`。 | High | 把 custom OFDM 后续命运收敛，避免第三套链路长期存在。 |
| 配置扩展模型清晰且被校验 | 3 | Good | 2.25 | `config/schema.py` 使用 Pydantic；SRS 有嵌套 `phy.srs` 配置；旧字段会被拒绝。 | High | 配置 schema 文件已 609 行，可按子树拆分以降低演进成本。 |
| 协议私有 waveform/receiver 逻辑隔离 | 3 | Good | 2.25 | SRS resource/sequence 在 `nr_srs_resources.py`，PUSCH 在 `nr_pusch_observation.py`；公共链路在 `common_link.py`。 | High | PUSCH 主文件还承担多路径处理、receiver、array output glue，建议继续拆分。 |
| schema/迁移策略可承载新输出 | 3 | Good | 2.25 | `SCHEMA_VERSION = "1.4.0"`；schema tests 覆盖旧字段拒绝和新字段要求。 | High | writer/validator 仍是集中式条件逻辑，新增协议会持续增大复杂度。 |
| defaults/feature flags 支持增量采用 | 2 | Good | 1.50 | `array.spectrum`、`visualization`、`ranging`、sharding/fallback、SRS feature 子项均可配置。 | Medium | 对重型功能补成本矩阵和推荐 preset。 |

小结：`10.50 / 14`。扩展基础已具备，但新增协议仍会触发多处同步修改。

### 3. 数据契约与可复现性 (14)

| 条目 | 权重 | 等级 | 得分 | 证据 | 置信度 | 改进建议 |
|---|---:|---|---:|---|---|---|
| HDF5 schema、writer、validator 对齐 | 3 | Good | 2.25 | schema tests 通过；docs 标记当前 schema `1.4.0`；`schema_validator.py` 强校验关键 group。 | High | 将 protocol-specific schema 片段模块化，降低 validator 单文件膨胀。 |
| 维度语义和 BS/UE vs TX/RX 明确 | 3 | Excellent | 3.00 | handoff、README、config README、sys docs 均说明 role-view/link-view；可视化也按 `/link/tx_role` 和 `/link/rx_role` 解析。 | High | 维持并加 reader 层统一索引 API。 |
| manifest/config snapshot 保留 shard provenance | 2 | Excellent | 2.00 | README/config/sys docs 说明 `manifest/manifest.json`、`config_snapshot.json`、fallback 子 shard。 | High | 下游 reader 未完成前，继续提醒不要假设文件连续。 |
| 随机种子和小实验可复现 | 2 | Good | 1.50 | `runtime.seed`、smoke summary、baseline 文档存在；完整 pytest 通过。 | Medium | 将 smoke summary 生成标准化为 CI/脚本入口。 |
| `data/` 和 `outputs/` 安全隔离 | 2 | Excellent | 2.00 | handoff/config/skill 均要求不递归扫描；`.gitignore` 忽略大数据路径。 | High | 维持现状。 |
| unit/attrs/metadata 写入一致 | 2 | Good | 1.50 | docs/sys/07 详细列出 unit/index_order；writer/schema tests 覆盖关键字段。 | Medium | 清理 array alias 与 custom OFDM waveform TODO，减少字段冗余和例外。 |

小结：`12.25 / 14`。数据契约是项目强项，最大缺口是 shard-aware reader 尚未产品化。

### 4. 代码可维护性与复杂度 (16)

| 条目 | 权重 | 等级 | 得分 | 证据 | 置信度 | 改进建议 |
|---|---:|---|---:|---|---|---|
| 大文件和长函数受控或有合理解释 | 3 | Partial | 1.50 | 前五大代码文件均超过 900 行；`app/cli.py:main` 410 行、`truth_pipeline._run_rt_truth_pipeline_single` 288 行、`nr_pusch_observation._run_nr_pusch_observation_impl` 284 行。 | High | 分阶段拆出 CLI command handlers、pipeline stages、writer sections、validator sections。 |
| 重复逻辑提取到合适 helper | 3 | Good | 2.25 | common link、SRS resource helpers、ranging runner 已减少重复；PUSCH batching/fallback 有 helper。 | Medium | writer/validator/visualization 中仍有按字段分支和绘图重复模式，可抽象数据源描述表。 |
| 错误处理明确且必要时 fail-fast | 2 | Good | 1.50 | 配置校验、unknown backend、unsupported MU-MIMO backend、SRS 越界/旧字段拒绝均有测试。 | High | 对跳过/降级路径增加统一 error code 或 manifest 字段。 |
| 类型和 dataclass 表达边界 | 2 | Good | 1.50 | domain dataclass 丰富；config 使用 Pydantic；PHYContext/PHYModuleResult 存在。 | Medium | 部分 backend 是 duck-typed，可考虑 Protocol 类型。 |
| 性能/资源关注点在代码路径中可见 | 2 | Good | 1.50 | shard fallback、batch size、debug profiling、GPU binding、link_chunk_size 均有配置和文档。 | High | 建立 write-only/RT-only benchmark。 |
| TODO/技术债有耐久跟踪 | 2 | Excellent | 2.00 | `docs/todo/` active TODO 入口，合计 24 个；bug TODO 当前为 0。 | High | 每次完成后继续迁移到 history。 |
| 依赖和全局状态可控 | 2 | Good | 1.50 | 依赖集中在 pyproject；无 TensorFlow；registry 是少数全局状态。 | Medium | 对随机状态、GPU device、Sionna runtime 再补统一上下文说明。 |

小结：`11.75 / 16`。代码能跑且测试强，但维护成本会随着协议扩展继续上升。

### 5. 测试与验证体系 (14)

| 条目 | 权重 | 等级 | 得分 | 证据 | 置信度 | 改进建议 |
|---|---:|---|---:|---|---|---|
| 单元测试覆盖纯逻辑和 shape contract | 3 | Excellent | 3.00 | `tests/unit` 22 文件、255 个 test functions；覆盖 config、common_link、SRS resources、spatial_spectrum 等。 | High | 维持。 |
| 集成和 smoke 覆盖主流程 | 3 | Good | 2.25 | `tests/integration` 7 文件、68 个 test functions；部分依赖历史输出的测试 skipped。 | High | 把需要外部输出的 skipped 测试拆成 fixture-friendly smoke。 |
| schema/statistical/regression 能捕获契约漂移 | 3 | Excellent | 3.00 | `tests/schema` 43 个 test functions；`tests/statistical` 43 个 test functions；pytest 全量通过。 | High | 维持。 |
| 真实小实验摘要验证数据产品 | 2 | Good | 1.50 | handoff 记录 median SRS baseline、dense PUSCH rerun、40UE smoke 等。 | Medium | 将 summary 脚本输出格式标准化并纳入报告模板。 |
| 失败路径和非法配置有测试 | 2 | Good | 1.50 | unknown backend、legacy 字段拒绝、unsupported MIMO/SRS 配置等有测试；fallback 有单测。 | High | 对 writer/schema 的失败消息再增加快照或结构化断言。 |
| 标准开发命令当前有效 | 1 | Excellent | 1.00 | `uv run ruff check sionna_measurement_sim tests scripts` 通过；`uv run pytest -q` 通过。 | High | 维持。 |

小结：`12.25 / 14`。测试体系扎实，是当前架构演进的主要安全网。

### 6. 文档与上手成本 (10)

| 条目 | 权重 | 等级 | 得分 | 证据 | 置信度 | 改进建议 |
|---|---:|---|---:|---|---|---|
| handoff 和 sys docs 描述当前事实 | 2 | Excellent | 2.00 | handoff 与 sys index 反映 schema 1.4.0、SRS v2、TODO 入口、large-data 规则。 | High | 维持。 |
| config README 与模板同步 | 2 | Good | 1.50 | config README 覆盖 SRS/PUSCH、sharding、array spectrum、visualization 等；模板齐全。 | High | 对 768-subcarrier / 64PRB 局部仿真这类新运行口径可补专门说明。 |
| 架构/API 文档解释扩展点 | 2 | Good | 1.50 | `phy_module_development.md` 给出新 PHY module 流程；sys docs 覆盖 pipeline/io/PHY。 | High | 增加 writer/schema 新字段接入 checklist。 |
| TODO 和限制明确可执行 | 2 | Excellent | 2.00 | `docs/todo/` 明确 24 个 active TODO；bug TODO 为 0；feature/structure/performance 分类清楚。 | High | 维持。 |
| skills/runbooks 捕获重复工程流程 | 2 | Excellent | 2.00 | 本地已有 `sionna-dev-workflow` 和 `sionna-project-health-check`。 | High | 后续可增加 benchmark/smoke skill。 |

小结：`9.00 / 10`。文档质量高，已能支撑多轮复杂演进。

### 7. 运行与实验工程 (8)

| 条目 | 权重 | 等级 | 得分 | 证据 | 置信度 | 改进建议 |
|---|---:|---|---:|---|---|---|
| sharding 和多 GPU 执行清楚 | 2 | Good | 1.50 | config README 和 sys docs 描述 UE shard、多 worker、GPU IDs、fallback；baseline 有多 GPU 记录。 | High | 增加动态负载均衡和 GPU 忙碌场景测试。 |
| I/O 体积和输出布局被有意管理 | 2 | Good | 1.50 | 多文件 shard + manifest 是推荐方式；大输出不入 git。 | High | HDF5 写入优化仍是 PERF-001。 |
| profiling/logs 支持定位慢或失败运行 | 1 | Good | 0.75 | debug profiling 有配置和 logs 目录；perf events/hardware samples 有文档。 | Medium | 输出统一 summary schema，便于跨实验对比。 |
| 本地大数据路径处理安全 | 2 | Excellent | 2.00 | handoff、config README、skills 均禁止递归扫描/提交 `data/` 和 `outputs/`。 | High | 维持。 |
| visualization/diagnostics 支持小样本检查 | 1 | Good | 0.75 | visualization 支持 topology/CFR/waveform/AoA/spectrum/path；近期修正 role-aware 和 scene frame。 | Medium | PERF-004 仍需降低开销。 |

小结：`6.50 / 8`。运行工程已可生产使用，但性能与 benchmark 仍有可改空间。

### 8. 变更治理 (6)

| 条目 | 权重 | 等级 | 得分 | 证据 | 置信度 | 改进建议 |
|---|---:|---|---:|---|---|---|
| 分支和提交纪律保持可 review | 2 | Good | 1.50 | 当前在 `codex/docs-todo-consolidation`；近期提交按主题拆分；仍有 4 个未跟踪分析脚本。 | Medium | 定期确认未跟踪脚本是否归档、提交或保留本地。 |
| docs/config/schema 与行为变更同步 | 2 | Good | 1.50 | 近期 SRS stage2、AoA、path_samples、health skill 均有 docs/tests 同步。 | High | 对 schema breaking change 加 migration checklist。 |
| 兼容性和迁移影响说明清楚 | 1 | Good | 0.75 | docs 说明旧字段拒绝、legacy fallback、custom OFDM legacy。 | Medium | reader alias/migration guide 可减少下游脚本破坏。 |
| review、验证和后续闭环可见 | 1 | Good | 0.75 | 每轮有 ruff/pytest/smoke 记录；TODO history 已建立。 | Medium | 把每次正式 smoke summary 固化到 `docs/check` 或 `docs/performance` 索引。 |

小结：`4.50 / 6`。治理习惯健康，主要是 formal checklist 还可以更机器化。

## 风险列表

| ID | Priority | Area | Evidence | Impact | Likelihood | Recommendation | Owner or Trigger | Status |
|---|---|---|---|---|---|---|---|---|
| R-001 | P2 | 下游数据读取 | `docs/todo/structure.md` STR-001：多 `result_xxx.h5` 输出缺少统一 shard-aware reader / dataset loader。 | 下游训练/分析需要手写遍历 manifest，容易错误假设 shard 文件连续或忽略 fallback 子 shard。 | High | 优先实现 manifest-aware reader，支持全局 UE/BS 索引和 shard 迭代。 | 下一轮训练/分析管线接入前 | Open |
| R-002 | P2 | 模块复杂度 | `nr_pusch_observation.py` 1728 行、`truth_pipeline.py` 1537 行、`report.py` 1256 行、`schema_validator.py` 1095 行、`hdf5_writer.py` 924 行；多个函数/类超过 150 行。 | 新协议和新输出字段会继续放大修改面，增加回归风险。 | High | 按 pipeline stage、protocol schema section、visualization plot source 做渐进拆分。 | 下一次新增协议或大字段迁移前 | Open |
| R-003 | P2 | legacy 路径 | `docs/todo/structure.md` STR-003 与 `hdf5_writer.py` TODO 指向 custom OFDM legacy waveform grid 问题。 | 长期保留第三套口径会干扰公共链路抽象，也容易让 schema/docs 出现例外承诺。 | Medium | 决定迁移到 common link 或正式移除/标记 legacy，并同步 schema/docs/tests。 | custom OFDM 再次被用于生产或新增协议前 | Open |
| R-004 | P2 | 标准声明风险 | `docs/todo/feature.md` FEAT-SRS-001；当前 `nr_srs` 明确是 standards-shaped v2 subset，不是 3GPP-compliant。 | 若论文或对外文档误称标准完整，会影响实验可解释性和审稿可信度。 | Medium | 先做 38.211/38.213 reference validation，再决定是否提升声明级别。 | 对外写作或标准完整性实验前 | Open |
| R-005 | P3 | writer/schema 耦合 | `hdf5_writer.py` 与 `schema_validator.py` 大文件集中处理多个协议字段。 | 新增 WiFi-like/6G-like 时需要同步修改集中式 writer/validator，容易遗漏 attrs/shape 校验。 | Medium | 引入协议扩展描述表或 per-protocol writer/schema hooks。 | 新增第四个 PHY module 时 | Open |
| R-006 | P3 | 性能成本 | `docs/todo/performance.md` PERF-001/003/004/005/006/008；HDF5 write、空间谱、visualization 和多 GPU 调度仍待系统 benchmark。 | 大规模运行成本和尾部耗时可能不稳定，影响生产效率。 | Medium | 建立 RT-only、write-only、spectrum/visualization 开关矩阵 benchmark。 | 下一轮大规模全量实验前 | Open |
| R-007 | P4 | 本地工作区卫生 | `git status` 显示 4 个未跟踪 `scripts/plot_cfr_similarity_*.py`。 | 不影响当前项目运行，但容易在提交时误纳入或长期遗忘。 | Low | 明确这些脚本是保留本地、纳入 repo、还是移动到 legacy/analysis 目录。 | 下次整理 scripts 或提交前 | Open |

当前没有确认中的 P0/P1 风险。

## 建议

| 优先级 | 工作流 | 建议 | 预期收益 | 建议验收 |
|---|---|---|---|---|
| P2 | 数据读取 | 实现 shard-aware reader / dataset loader。 | 降低训练/分析侧重复代码和 shard/fallback 误读风险。 | fixture + 真实 manifest smoke；支持全局 UE/BS 索引。 |
| P2 | 复杂度治理 | 拆分 `truth_pipeline.py`、`nr_pusch_observation.py`、`hdf5_writer.py`、`schema_validator.py`。 | 降低新增协议和字段迁移的修改面。 | 拆分后 ruff/pytest 全通过，外部 API 不变。 |
| P2 | 标准声明 | 做 SRS reference validation。 | 支撑论文/文档中更稳妥的标准一致性描述。 | 38.211/38.213 reference cases + 文档声明矩阵。 |
| P3 | 性能工程 | 建立 write-only / RT-only / spectrum-only benchmark 入口。 | 让性能优化不再依赖端到端大实验。 | 输出 JSON/CSV summary，纳入 docs/performance 或 docs/todo 结果更新。 |
| P3 | Schema 扩展 | 设计 per-protocol writer/schema hook 或 declarative field spec。 | 减少新增 PHY module 时的重复字段逻辑。 | 新增一个 toy PHY/schema fixture 验证 hook。 |
| P4 | 工作区整理 | 处理未跟踪 CFR similarity 脚本归属。 | 降低误提交和上下文噪音。 | git status 清晰或文档说明这些脚本为何保留本地。 |

## 未覆盖范围和盲点

- 未递归扫描或读取 `data/`、`outputs/`。
- 未运行新的真实仿真 smoke；本报告使用现有文档记录和完整单元/集成/schema/statistical 测试作为证据。
- 未逐行审查所有算法正确性；重点是结构健康、扩展性、职责边界和工程风险。
- 未评估外部下游训练代码，因为当前仓库内尚未有正式 shard-aware dataset loader。
- 当前报告基于分支 `codex/docs-todo-consolidation` 的状态，不等同于远端 `origin/main`。

## 后续

- 下次建议复查时间：完成 shard-aware reader 或新增第四个 PHY module 前。
- 模板版本建议：v1 暂不需要修改；如果后续引入自动度量脚本，可升级到 v2 并加入自动指标附录。
