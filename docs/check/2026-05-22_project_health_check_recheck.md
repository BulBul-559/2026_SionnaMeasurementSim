# 项目健康体检复检报告

报告日期：`2026-05-22`
分支 / 提交：`codex/docs-todo-consolidation / 8d70e31`
评审者：`Codex`
范围：`repo-wide`
模板版本：`v1`
报告语言：`中文`

## 总结

总分：`78.75 / 100`

整体判断：项目当前仍处在可持续演进区间，但不是“几乎没有结构债”的状态。核心分层清楚，`domain` 基本保持纯净，SRS/PUSCH 已共享 `common_link` 观测链路，schema `1.4.0`、TX/RX 语义、sharding/manifest、ranging、array spectrum 和文档体系都比较成熟。`ruff` 与全量 `pytest` 均通过，说明近期重构没有明显破坏现有契约。

本次复检严格按 v1 模板的子项权重逐项累加，并执行 Concept Ownership / Single Source of Truth Pass。主要扣分原因不是功能不可用，而是发现多个概念存在重复表达或集中式耦合：`ranging` 有全局 Pydantic config 和算法包 dataclass 双定义，`aoa_heatmap_label` / `spatial_spectrum_label` 是兼容 alias，`hdf5_writer.py` / `schema_validator.py` 仍集中承载多个协议字段，若继续扩展 WiFi-like 或 6G-like，会增加漂移和重复开发风险。

当前没有确认的 P0/P1 风险。最高风险为 P2，集中在 shard-aware reader 缺失、核心模块复杂度、legacy `custom_ofdm` 路径和 SRS 标准声明边界。

| 模块 | 权重 | 得分 | 置信度 | 简要原因 |
|---|---:|---:|---|---|
| 架构分层与职责边界 | 18 | 13.50 | High | 分层清楚、domain 纯净，但核心 orchestration、writer、validator、report 仍偏大。 |
| 协议与功能扩展性 | 14 | 9.75 | High | common link 和 PHY registry 已成型；新增协议仍需同步改 config/CLI/writer/schema/docs。 |
| 数据契约与可复现性 | 14 | 12.25 | High | schema、TX/RX 语义、manifest、config snapshot 和大数据隔离都比较稳。 |
| 代码可维护性与复杂度 | 16 | 11.00 | High | 大文件、长函数、重复概念表达和集中式 schema/writer 是主要扣分点。 |
| 测试与验证体系 | 14 | 12.25 | High | 单元、集成、schema、statistical 测试扎实，全量测试通过。 |
| 文档与上手成本 | 10 | 9.00 | High | handoff/sys/config/todo/skills 完整，仍需持续清理历史 alias 和实验口径。 |
| 运行与实验工程 | 8 | 6.50 | Medium | sharding、多 GPU、profiling、visualization 可用，性能 benchmark 入口仍在 TODO。 |
| 变更治理 | 6 | 4.50 | Medium | 分支/提交/文档同步习惯好，但本地仍有未跟踪分析脚本。 |
| **总计** | **100** | **78.75** |  |  |

## 证据清单

| 证据类型 | 已检查内容 | 备注 |
|---|---|---|
| Git 状态 | `git status --short --branch`, `git log --oneline --decorate -8` | 当前分支 `codex/docs-todo-consolidation`；HEAD `8d70e31`；存在 4 个未跟踪 CFR similarity 分析脚本。 |
| 当前事实文档 | `docs/agent_handoff.md`, `docs/sys/README.md`, `README.md`, `config/README.md` | 当前系统事实为 schema `1.4.0`，active TODO 位于 `docs/todo/`。 |
| 架构文档 | `docs/sys/00_project_overview.md`, `docs/sys/05_phy_observation.md`, `docs/sys/phy_module_development.md` | 覆盖分层、TX/RX 语义、PHY registry、common link、ranging 和新增模块规范。 |
| 代码结构 | targeted `find`, line-count script, AST long-function scan | 未递归扫描 `data/` 或 `outputs/`。 |
| 概念归属 | `rg` 搜索 schema/registry/config/path/alias；AST duplicate class scan | 发现 `RangingConfig` 双定义、array label alias、writer/schema 集中耦合等结构性风险。 |
| 依赖方向 | AST import scan for `domain/` and `io/` | 未发现 `domain/` 或 `io/` 直接 import `sionna`、`torch`、`tensorflow`、`mitsuba`、`drjit`。 |
| 测试分布 | AST test count | `unit=255`, `integration=68`, `schema=43`, `statistical=43`, `adapter=29` test functions。 |
| 验证命令 | `uv run ruff check sionna_measurement_sim tests scripts`; `uv run pytest -q` | ruff 通过；pytest `249 passed, 19 skipped, 1 warning in 53.26s`。 |
| TODO | `docs/todo/README.md`, `feature.md`, `structure.md`, `performance.md`, `bug.md` | active TODO 共 24 个：feature 12、structure 4、performance 8、bug 0。 |

## Concept Ownership / Single Source of Truth

本节是本次复检的重点。它检查跨模块概念是否有明确 owner，以及多个表示之间是否有 mapper、validator、docs 和 tests 保护。

| 概念 | Source of truth | Runtime representation | Mapper / adapter | Output contract | Validator | Docs | Tests | 状态 | 风险链接 |
|---|---|---|---|---|---|---|---|---|---|
| HDF5 schema version | `sionna_measurement_sim/domain/constants.py::SCHEMA_VERSION = "1.4.0"` | `MeasurementSimulationResult.metadata.schema_version` | writer 读取 metadata/constants | `/metadata/schema_version` | `io/schema_validator.py` | `docs/sys/07_config_and_h5_format.md` | schema tests | Single source | 无 |
| PHY standard registry | `sionna_measurement_sim/phy/modules.py::PHY_REGISTRY` | `CustomOFDMModule`, `NRPUSCHModule`, `NRSRSModule` | `get_phy_module()` | `metadata/phy_standard`, protocol groups | schema validator + module tests | `docs/sys/phy_module_development.md` | unit/integration tests | Adapter boundary | R-005 |
| TX/RX role semantics | `domain/topology.py::resolve_link_roles`, `domain/link.py::LinkConfig` | `/link/tx_role`, `/link/rx_role`, resolved topology | role topology resolver | `/channel/truth/cfr`, `/observation/cfr_est`, visualization axes | schema validator | README, handoff, sys docs | domain/schema/integration tests | Single source | 无 |
| `/observation/cfr_est` contract | `domain/observation.py::ObservationResult` + schema docs | SRS/PUSCH observation output | protocol receiver -> common `ObservationResult` | `[snapshot,tx,rx,rx_ant,tx_ant,subcarrier]` | `schema_validator.py` | `docs/sys/07_config_and_h5_format.md` | schema/statistical tests | Adapter boundary | R-005 |
| Ranging config | `config/schema.py` 和 `ranging/config.py` 同时定义 | CLI 从全局 config 显式构造算法 config | 手写转换，尚未有集中 mapper/equivalence tests | `/ranging/*` | config validator + schema validator | config README, sys docs | ranging unit tests | Duplicate definition | R-007 |
| NR SRS resource metadata | `phy/nr_srs_resources.py` + `config/schema.py::SRSConfig` | `srs_resource_mask`, flattened RE, ports, power scale | SRS resource builder | `/waveform/srs_*`, `/observation/cfr_est_resource` | SRS schema validator | handoff, sys docs | SRS resource/schema tests | Adapter boundary | R-004/R-005 |
| Array label alias | `array/aoa_heatmap_label` 和 `array/spatial_spectrum_label` 兼容语义 | writer 同时写两个 label dataset | 兼容 alias，无 reader 层统一抽象 | `/array/aoa_heatmap_label`, `/array/spatial_spectrum_label` | schema validator 检查二者 | sys docs, TODO STR-004 | schema tests | Duplicate definition | R-008 |
| Spectrum source alias | `domain/array.py` 允许 `srs_cfr_est` | `srs_cfr_est` 实际指向 `/observation/cfr_est` | visualization/PHY 输出侧兼容处理 | `/array/spatial_spectrum_srs` | schema validator | sys docs | SRS schema tests | Adapter boundary | R-009 |
| Manifest shard identity | `domain/results.py::ShardSpec` | `manifest/manifest.json` + per-shard HDF5 | sharding pipeline/fallback | `result_{shard_index}.h5`, fallback child shards | integration tests and manifest checks | `docs/sys/04_rt_pipeline.md` | sharding tests | Single source, but consumer incomplete | R-001 |
| Custom OFDM legacy status | docs/TODO 标记 legacy | `CustomOFDMModule` 仍在 registry | legacy branch | legacy waveform exceptions | schema tests allow exceptions | TODO + sys docs | legacy tests | Adapter boundary with debt | R-003 |

结论：项目不是缺少抽象，而是部分概念已有多层表达后，缺少统一的“mapper/equivalence test”来证明它们不会漂移。后续治理应优先把重复定义升级为明确的 adapter boundary，而不是为每个新发现的问题追加特判。

## 详细评分

### 1. 架构分层与职责边界 (18)

| 条目 | 权重 | 等级 | 得分 | 证据 | 置信度 | 改进建议 |
|---|---:|---|---:|---|---|---|
| 依赖方向清晰且基本单向 | 3 | Good | 2.25 | `docs/sys/00_project_overview.md` 定义 app/config/domain/rt/phy/io 分层；AST 检查未发现 domain/io 直接 import Sionna/Torch。 | High | 增加 import-boundary test，防止未来回归。 |
| domain 模型保持纯净 | 3 | Excellent | 3.00 | `domain/` 未直接 import `sionna`、`torch` 等框架；domain 仅表达结构、契约和 shape。 | High | 维持现状。 |
| 模块满足单一职责和内聚性 | 3 | Partial | 1.50 | 大文件集中：`nr_pusch_observation.py` 1728 行、`truth_pipeline.py` 1537 行、`visualization/report.py` 1256 行、`schema_validator.py` 1095 行、`hdf5_writer.py` 924 行。 | High | 优先拆 orchestration、writer/schema validator、visualization report 和 PUSCH processing path。 |
| registry/plugin 边界隔离可替换实现 | 3 | Good | 2.25 | `PHY_REGISTRY` 提供模块入口；`phy_module_development.md` 说明新增 PHY module 流程。 | High | writer/schema 仍需 per-protocol hook，避免 registry 只管运行、不管落盘契约。 |
| 公共契约有显式 ownership 和 single-source 边界 | 3 | Good | 2.25 | schema version、TX/RX role、`cfr_est` 等核心契约 owner 清晰；ranging config 和 array label alias 仍是重复表达。 | High | 对重复表达概念补 mapper/equivalence tests，并在 TODO 中追踪。 |
| 目录布局对新贡献者可发现 | 3 | Good | 2.25 | `docs/sys/00_project_overview.md` 有代码目录到文档映射；`docs/agent_handoff.md` 有深读地图。 | High | 增加“从配置到 HDF5 字段”的贡献者导航。 |

小结：`13.50 / 18`。分层基础好，但职责集中和重复概念边界需要治理。

### 2. 协议与功能扩展性 (14)

| 条目 | 权重 | 等级 | 得分 | 证据 | 置信度 | 改进建议 |
|---|---:|---|---:|---|---|---|
| 共享抽象减少协议重复 | 3 | Good | 2.25 | PUSCH/SRS 共享 `ObservationImpairmentChain`；ranging 在 pipeline 级读取统一 `/observation/cfr_est`。 | High | 后续 WiFi-like/6G-like 应直接复用 common link，不再开第三套链路。 |
| 配置扩展模型有一个输入 contract 和显式 runtime mapper | 3 | Partial | 1.50 | `config/schema.py` 与 `ranging/config.py` 同名配置类重复；当前转换分散在 CLI/pipeline glue。 | High | 建立全局 YAML config 到 domain/algorithm config 的集中 mapper，并补 equivalence tests。 |
| 协议私有 waveform/receiver 逻辑隔离 | 3 | Good | 2.25 | SRS resource/sequence 在 `nr_srs_resources.py`，PUSCH receiver/backend 逻辑在 phy 侧，公共链路在 `common_link.py`。 | High | 继续拆 PUSCH 主文件，减少 protocol-private glue 混杂。 |
| schema/迁移策略可承载新输出 | 3 | Good | 2.25 | `SCHEMA_VERSION = "1.4.0"`；schema tests 覆盖旧字段拒绝和新字段要求。 | High | 将 SRS/PUSCH/custom OFDM schema 片段插件化或描述表化。 |
| defaults/feature flags 支持增量采用 | 2 | Good | 1.50 | `array.spectrum`、`visualization`、`ranging`、sharding、SRS 子功能都有开关。 | Medium | 对重型功能补成本矩阵和推荐 preset。 |

小结：`9.75 / 14`。扩展基础可用，但新增协议仍会触发多文件同步。

### 3. 数据契约与可复现性 (14)

| 条目 | 权重 | 等级 | 得分 | 证据 | 置信度 | 改进建议 |
|---|---:|---|---:|---|---|---|
| HDF5 schema、writer、validator 对齐 | 3 | Good | 2.25 | schema tests 通过；docs 标记 schema `1.4.0`；validator 强校验关键 group。 | High | 降低 writer/validator 集中式条件逻辑。 |
| 维度语义和 BS/UE vs TX/RX 明确 | 3 | Excellent | 3.00 | handoff、README、config README、sys docs 均说明 role-view/link-view。 | High | 维持，并让未来 reader 默认按 link role 解释。 |
| manifest/config snapshot 保留 shard provenance | 2 | Excellent | 2.00 | docs 说明 `manifest/manifest.json`、`config_snapshot.json` 和 fallback 子 shard。 | High | 下游 reader 完成前，继续避免手写 shard 假设。 |
| 随机种子和小实验可复现 | 2 | Good | 1.50 | `runtime.seed`、smoke summary、baseline 文档存在；全量 pytest 通过。 | Medium | 将 smoke summary 生成标准化。 |
| `data/` 和 `outputs/` 安全隔离 | 2 | Excellent | 2.00 | handoff、config README、skills 均要求不递归扫描；`.gitignore` 忽略大数据路径。 | High | 维持。 |
| units/attrs/metadata 写入一致 | 2 | Good | 1.50 | docs/sys/07 详细列出 unit/index_order；writer/schema tests 覆盖关键字段。 | Medium | 清理 array alias 和 custom OFDM waveform 例外。 |

小结：`12.25 / 14`。数据契约是强项，最大短板是消费侧 reader。

### 4. 代码可维护性与复杂度 (16)

| 条目 | 权重 | 等级 | 得分 | 证据 | 置信度 | 改进建议 |
|---|---:|---|---:|---|---|---|
| 大文件和长函数受控或有合理解释 | 3 | Partial | 1.50 | 前五大代码文件均超过 900 行；`app/cli.py:main` 410 行，`truth_pipeline._run_rt_truth_pipeline_single` 288 行。 | High | 分阶段拆出 CLI handlers、pipeline stages、writer sections、validator sections。 |
| 重复逻辑提取到合适 helper | 3 | Partial | 1.50 | common link 已减少 PHY 重复；但 ranging config 双定义、array label alias、writer/schema 字段分支仍显示重复 ownership。 | High | 对重复概念建立 source-of-truth matrix、mapper 和等价测试。 |
| 错误处理明确且必要时 fail-fast | 2 | Good | 1.50 | config 校验、unknown backend、unsupported MIMO/SRS 配置、旧字段拒绝均有测试。 | High | 对跳过/降级路径增加统一 error code 或 manifest 字段。 |
| 类型和 dataclass 表达边界 | 2 | Good | 1.50 | domain dataclass 丰富；config 使用 Pydantic；PHYContext/PHYModuleResult 存在。 | Medium | 部分 backend 是 duck-typed，可考虑 Protocol 类型。 |
| 性能/资源关注点在代码路径中可见 | 2 | Good | 1.50 | shard fallback、batch size、profiling、GPU binding、link_chunk_size 均有配置和文档。 | High | 建立 write-only/RT-only benchmark。 |
| TODO/技术债有耐久跟踪 | 2 | Excellent | 2.00 | `docs/todo/` active TODO 入口，合计 24 个，bug TODO 为 0。 | High | 完成后继续迁移到 history。 |
| 依赖和全局状态可控 | 2 | Good | 1.50 | 依赖集中在 pyproject；无 TensorFlow；registry 是少数全局状态。 | Medium | 对随机状态、GPU device、Sionna runtime 再补统一上下文说明。 |

小结：`11.00 / 16`。这里是本次复检主要降分项。

### 5. 测试与验证体系 (14)

| 条目 | 权重 | 等级 | 得分 | 证据 | 置信度 | 改进建议 |
|---|---:|---|---:|---|---|---|
| 单元测试覆盖纯逻辑和 shape contract | 3 | Excellent | 3.00 | `tests/unit` 22 文件、255 个 test functions，覆盖 config、common_link、SRS resources、ranging、spatial_spectrum 等。 | High | 维持。 |
| 集成和 smoke 覆盖主流程 | 3 | Good | 2.25 | `tests/integration` 7 文件、68 个 test functions；部分依赖历史输出的测试 skipped。 | High | 把依赖外部输出的 skipped 测试拆成 fixture-friendly smoke。 |
| schema/statistical/regression 能捕获契约漂移 | 3 | Excellent | 3.00 | `tests/schema` 43 个 test functions；`tests/statistical` 43 个 test functions；pytest 全量通过。 | High | 维持。 |
| 真实小实验摘要验证数据产品 | 2 | Good | 1.50 | handoff 记录 median SRS baseline、dense PUSCH rerun、40UE smoke 等。 | Medium | 将 summary 脚本输出格式标准化。 |
| 失败路径和非法配置有测试 | 2 | Good | 1.50 | unknown backend、legacy 字段拒绝、unsupported MIMO/SRS 配置等有测试；fallback 有单测。 | High | 对 config mapper/equivalence 增加失败路径测试。 |
| 标准开发命令当前有效 | 1 | Excellent | 1.00 | `ruff` 和全量 `pytest` 均通过。 | High | 维持。 |

小结：`12.25 / 14`。测试体系是项目强项。

### 6. 文档与上手成本 (10)

| 条目 | 权重 | 等级 | 得分 | 证据 | 置信度 | 改进建议 |
|---|---:|---|---:|---|---|---|
| handoff 和 sys docs 描述当前事实 | 2 | Excellent | 2.00 | handoff 与 sys index 反映 schema 1.4.0、SRS v2、TODO 入口、large-data 规则。 | High | 维持。 |
| config README 与模板同步 | 2 | Good | 1.50 | config README 覆盖 SRS/PUSCH、sharding、array spectrum、visualization 等。 | High | 对 64PRB 局部仿真、ranging config ownership 补说明。 |
| 架构/API 文档解释扩展点 | 2 | Good | 1.50 | `phy_module_development.md` 给出新 PHY module 流程。 | High | 增加 writer/schema 新字段接入 checklist。 |
| TODO 和限制明确可执行 | 2 | Excellent | 2.00 | `docs/todo/` 明确 24 个 active TODO。 | High | 维持。 |
| skills/runbooks 捕获重复工程流程 | 2 | Excellent | 2.00 | 本地已有 `sionna-dev-workflow`、`sionna-project-health-check` 和 TODO 维护 skill。 | High | 后续可增加 benchmark/smoke skill。 |

小结：`9.00 / 10`。文档体系足以支撑后续迭代。

### 7. 运行与实验工程 (8)

| 条目 | 权重 | 等级 | 得分 | 证据 | 置信度 | 改进建议 |
|---|---:|---|---:|---|---|---|
| sharding 和多 GPU 执行清楚 | 2 | Good | 1.50 | config README 和 sys docs 描述 UE shard、多 worker、GPU IDs、fallback。 | High | 增加动态负载均衡和 GPU 忙碌场景测试。 |
| I/O 体积和输出布局被有意管理 | 2 | Good | 1.50 | 多文件 shard + manifest 是推荐方式；大输出不入 git。 | High | HDF5 写入优化仍是 PERF-001。 |
| profiling/logs 支持定位慢或失败运行 | 1 | Good | 0.75 | debug profiling 有配置和 logs 目录；perf events/hardware samples 有文档。 | Medium | 输出统一 summary schema。 |
| 本地大数据路径处理安全 | 2 | Excellent | 2.00 | handoff、config README、skills 均禁止递归扫描/提交 `data/` 和 `outputs/`。 | High | 维持。 |
| visualization/diagnostics 支持小样本检查 | 1 | Good | 0.75 | visualization 支持 topology/CFR/waveform/AoA/spectrum/path；近期修正 role-aware 和 scene frame。 | Medium | PERF-004 仍需降低开销。 |

小结：`6.50 / 8`。生产运行可用，性能工程还可更系统。

### 8. 变更治理 (6)

| 条目 | 权重 | 等级 | 得分 | 证据 | 置信度 | 改进建议 |
|---|---:|---|---:|---|---|---|
| 分支和提交纪律保持可 review | 2 | Good | 1.50 | 当前在 `codex/docs-todo-consolidation`；近期提交按主题拆分；仍有 4 个未跟踪分析脚本。 | Medium | 定期确认未跟踪脚本是否归档、提交或保留本地。 |
| docs/config/schema 与行为变更同步 | 2 | Good | 1.50 | 近期 SRS、AoA、path_samples、health skill、TODO skill 均有 docs/tests 同步。 | High | 对 schema breaking change 加 migration checklist。 |
| 兼容性和迁移影响说明清楚 | 1 | Good | 0.75 | docs 说明旧字段拒绝、legacy fallback、custom OFDM legacy。 | Medium | reader alias/migration guide 可减少下游脚本破坏。 |
| review、验证和后续闭环可见 | 1 | Good | 0.75 | 每轮有 ruff/pytest/smoke 记录；TODO history 已建立。 | Medium | 把每次正式 smoke summary 固化到 docs/performance 或 docs/check 索引。 |

小结：`4.50 / 6`。治理习惯健康，仍可进一步机器化。

## 风险列表

| ID | Priority | Area | Evidence | Impact | Likelihood | Recommendation | Owner or Trigger | Status |
|---|---|---|---|---|---|---|---|---|
| R-001 | P2 | 下游数据读取 | `docs/todo/structure.md` STR-001：多 `result_xxx.h5` 输出缺少统一 shard-aware reader / dataset loader。 | 下游训练/分析需要手写遍历 manifest，容易错误假设 shard 连续或忽略 fallback 子 shard。 | High | 优先实现 manifest-aware reader，支持全局 UE/BS 索引和 shard 迭代。 | 下一轮训练/分析管线接入前 | Open |
| R-002 | P2 | 模块复杂度 | 多个核心文件超过 900 行，且多个函数/类超过 150 行。 | 新协议和新输出字段会继续放大修改面，增加回归风险。 | High | 按 pipeline stage、protocol schema section、visualization plot source 做渐进拆分。 | 下一次新增协议或大字段迁移前 | Open |
| R-003 | P2 | legacy 路径 | `docs/todo/structure.md` STR-003 与 writer TODO 指向 custom OFDM legacy waveform grid 问题。 | 长期保留第三套口径会干扰公共链路抽象，也容易让 schema/docs 出现例外承诺。 | Medium | 决定迁移到 common link 或正式移除/标记 legacy，并同步 schema/docs/tests。 | custom OFDM 再次被用于生产或新增协议前 | Open |
| R-004 | P2 | 标准声明风险 | `docs/todo/feature.md` FEAT-SRS-001；当前 `nr_srs` 是 standards-shaped v2 subset，不是 3GPP-compliant。 | 若论文或对外文档误称标准完整，会影响实验可解释性和审稿可信度。 | Medium | 做 38.211/38.213 reference validation，再决定声明级别。 | 对外写作或标准完整性实验前 | Open |
| R-005 | P3 | writer/schema 耦合 | `hdf5_writer.py` 与 `schema_validator.py` 集中处理多个协议字段。 | 新增 WiFi-like/6G-like 时需要同步修改集中式 writer/validator，容易遗漏 attrs/shape 校验。 | Medium | 引入协议扩展描述表或 per-protocol writer/schema hooks。 | 新增第四个 PHY module 时 | Open |
| R-006 | P3 | 性能成本 | `docs/todo/performance.md` PERF-001/003/004/005/006/008。 | 大规模运行成本和尾部耗时可能不稳定，影响生产效率。 | Medium | 建立 RT-only、write-only、spectrum/visualization 开关矩阵 benchmark。 | 下一轮大规模全量实验前 | Open |
| R-007 | P3 | 配置单一事实源 | `config/schema.py` 和 `ranging/config.py` 分别定义同名 ranging 配置类与重复校验。 | 新增 ranging estimator 或字段时可能只改一边，导致 YAML schema、runtime estimator 和测试夹层漂移。 | Medium | 建立单一事实源或明确 adapter boundary：集中 mapper、equivalence tests、非法值测试。 | 下一次扩展 ranging estimator/config 前 | Open |
| R-008 | P3 | array label alias | `aoa_heatmap_label` 和 `spatial_spectrum_label` 语义兼容，`docs/todo/structure.md` STR-004 已追踪。 | 输出体积和字段语义重复，长期会让 reader/训练代码选择口径不一致。 | Medium | 确定唯一权威字段，另一个用兼容 alias、hard link 或 migration guide。 | 下游 reader 实现或 schema 1.5 前 | Open |
| R-009 | P4 | spectrum source alias | `srs_cfr_est` 是历史 source 名称，但实际指向统一 `/observation/cfr_est`。 | 不太会导致错误数据，但会增加新用户理解成本。 | Low | 在 reader/config 层将其标记 deprecated，并给出迁移提示。 | 清理 array source 兼容项时 | Open |
| R-010 | P4 | 本地工作区卫生 | `git status` 显示 4 个未跟踪 `scripts/plot_cfr_similarity_*.py`。 | 不影响当前运行，但容易误提交或长期遗忘。 | Low | 明确这些脚本是保留本地、纳入 repo、还是移动到 legacy/analysis 目录。 | 下次整理 scripts 或提交前 | Open |

当前没有确认中的 P0/P1 风险。

## 建议

| 优先级 | 工作流 | 建议 | 预期收益 | 建议验收 |
|---|---|---|---|---|
| P2 | 数据读取 | 实现 shard-aware reader / dataset loader。 | 降低训练/分析侧重复代码和 shard/fallback 误读风险。 | fixture + 真实 manifest smoke；支持全局 UE/BS 索引。 |
| P2 | 复杂度治理 | 拆分 `truth_pipeline.py`、`nr_pusch_observation.py`、`hdf5_writer.py`、`schema_validator.py`。 | 降低新增协议和字段迁移的修改面。 | 拆分后 ruff/pytest 全通过，外部 API 不变。 |
| P2 | 标准声明 | 做 SRS reference validation。 | 支撑论文/文档中更稳妥的标准一致性描述。 | 38.211/38.213 reference cases + 文档声明矩阵。 |
| P3 | 配置治理 | 整理 ranging 双配置模型，补全局 schema 到算法 config 的 mapper/equivalence tests。 | 降低 estimator/config 字段漂移风险，同时保持 ranging 算法包解耦。 | 测试覆盖默认值、非法值和 YAML to domain config 等价性。 |
| P3 | Schema 扩展 | 设计 per-protocol writer/schema hook 或 declarative field spec。 | 减少新增 PHY module 时的重复字段逻辑。 | 新增 toy PHY/schema fixture 验证 hook。 |
| P3 | 字段别名清理 | 处理 `aoa_heatmap_label` / `spatial_spectrum_label` 与 `srs_cfr_est` 兼容别名。 | 降低 HDF5 输出口径和 reader 选择成本。 | schema/docs/reader 共同说明权威字段和兼容策略。 |
| P3 | 性能工程 | 建立 write-only / RT-only / spectrum-only benchmark 入口。 | 让性能优化不再依赖端到端大实验。 | 输出 JSON/CSV summary，纳入 docs/performance 或 TODO history。 |
| P4 | 工作区整理 | 处理未跟踪 CFR similarity 脚本归属。 | 降低误提交和上下文噪音。 | git status 清晰或文档说明这些脚本为何保留本地。 |

## 未覆盖范围和盲点

- 未递归扫描或读取 `data/`、`outputs/`。
- 未运行新的真实仿真 smoke；本报告使用当前文档、代码结构和完整测试作为证据。
- 未逐行审查所有算法正确性；重点是结构健康、扩展性、职责边界、单一事实源和工程风险。
- 未评估外部下游训练代码，因为当前仓库内尚未有正式 shard-aware dataset loader。
- 当前报告基于分支 `codex/docs-todo-consolidation` 的状态，不等同于远端 `origin/main`。

## 后续

- 下次建议复查触发点：完成 shard-aware reader、整理 ranging config、或新增第四个 PHY module 前。
- 模板版本建议：v1 仍可继续使用。若要升级 v2，建议加入自动指标附录和概念 ownership 自动扫描脚本。
