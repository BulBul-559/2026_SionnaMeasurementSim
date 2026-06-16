# 项目健康体检报告

报告日期：`2026-06-16`
分支 / 提交：`codex/noncooperative-iq-observation / 14f72c5`
评审者：`Codex`
范围：`repo-wide`
模板版本：`v1`
报告语言：`中文`

## 总结

总分：`78.75 / 100`

整体判断：项目仍处在可持续演进区间，自动化契约和文档同步做得比较扎实；但最近新增的
`output.profile`、IQ observation、`iq_link_library` compact contract、noncooperative IQ、
multi-UE SRS、benchmark 和 schema `2.0.0` 让系统能力显著扩张，同时也把核心
orchestration、writer、validator、visualization 的集中式复杂度继续推高。

本次体检没有发现确认的 P0/P1 风险。最高风险仍是 P2，集中在：核心模块过大、输出
contract/profile 矩阵需要更模块化的 writer/validator 边界、shard-aware reader 缺失、
IQ/noncooperative 观测语义容易在论文或下游使用中被误读、SRS 仍缺 reference validation。

相比上一次复检，项目能力更强、测试数量更多、`ranging` 双配置问题已通过 mapper 和单测
从“重复定义”缓解为“adapter boundary”；但新增能力没有降低复杂度债，因此总分基本持平。

| 模块 | 权重 | 得分 | 置信度 | 简要原因 |
|---|---:|---:|---|---|
| 架构分层与职责边界 | 18 | 13.50 | High | 分层和 domain 纯度仍好，但核心文件继续膨胀。 |
| 协议与功能扩展性 | 14 | 10.50 | High | common link、output plan、IQ 层增强扩展性；新增 profile 仍需多处同步。 |
| 数据契约与可复现性 | 14 | 12.25 | High | schema `2.0.0`、compact contracts、TX/RX 语义、manifest/config snapshot 明确。 |
| 代码可维护性与复杂度 | 16 | 11.00 | High | 大文件、长函数和集中式 writer/validator 是主要扣分点。 |
| 测试与验证体系 | 14 | 11.50 | High | `ruff`、全量 `pytest` 通过；真实输出依赖测试仍有 skipped。 |
| 文档与上手成本 | 10 | 9.00 | High | handoff/sys/config/todo/skills 同步较好，历史口径和实验口径还需持续整理。 |
| 运行与实验工程 | 8 | 6.50 | Medium | sharding、多 GPU、benchmark、profiling、可视化齐全，但性能债仍在。 |
| 变更治理 | 6 | 4.50 | Medium | 分支/提交/文档同步习惯良好；新增能力变更面较大，需更强迁移闭环。 |
| **总计** | **100** | **78.75** |  |  |

## 证据清单

| 证据类型 | 已检查内容 | 备注 |
|---|---|---|
| Git 状态 | `git status --short --branch`, `git rev-parse --short HEAD`, `git log --oneline --decorate -8` | 当前分支 `codex/noncooperative-iq-observation`，HEAD `14f72c5`，工作区开始时干净。 |
| 近期变更 | `git diff --stat main..HEAD` | 相对 `main` 有 42 文件变化，`2438 insertions(+), 60 deletions(-)`。 |
| 当前事实文档 | `docs/agent_handoff.md`, `docs/sys/README.md`, `README.md`, `config/README.md` | 当前系统事实为 schema/project version `2.0.0`，支持 full/rt_lite/rt_labels_only/iq_link_library profile。 |
| 架构文档 | `docs/sys/04_rt_pipeline.md`, `docs/sys/05_phy_observation.md`, `docs/sys/06_io_and_testing.md`, `docs/sys/07_config_and_h5_format.md` | 覆盖 output plan、IQ observation、compact contracts、multiuser、ranging、schema 和 benchmark。 |
| 代码结构 | targeted `find` / line-count / AST long-function scan | 未递归扫描或操作 `data/`、`outputs/`。 |
| Concept ownership | duplicate class/function scan，source-of-truth matrix，config mapper 检查 | `ranging` 双 config 已有 `config/mappers.py` 与单测保护；仍需把该模式推广。 |
| 测试分布 | AST test count | `unit=197`, `integration=43`, `schema=32`, `statistical=22`, `adapter=16` test functions。 |
| 验证命令 | `uv run ruff check sionna_measurement_sim tests scripts`; `uv run pytest -q` | ruff 通过；pytest `292 passed, 19 skipped, 1 warning in 66.75s`。 |
| TODO | `docs/todo/README.md`, `feature.md`, `structure.md`, `performance.md`, `bug.md` | active TODO 共 22 个：feature 12、structure 3、performance 7、bug 0。 |

### 结构指标

| 指标 | 当前值 |
|---|---:|
| `sionna_measurement_sim` | 75 files / 22680 lines |
| `tests` | 47 files / 7717 lines |
| `scripts` | 13 files / 4299 lines |
| `docs/sys` | 11 files / 3324 lines |
| `docs/todo` | 6 files / 426 lines |
| `docs/performance` | 13 files / 2282 lines |
| `config` | 17 files / 3873 lines |
| `.codex/skills` | 4 files / 480 lines |

最大代码文件：

| Lines | File |
|---:|---|
| 2286 | `sionna_measurement_sim/visualization/report.py` |
| 2118 | `sionna_measurement_sim/rt/truth_pipeline.py` |
| 1923 | `sionna_measurement_sim/phy/nr_pusch_observation.py` |
| 1790 | `sionna_measurement_sim/io/schema_validator.py` |
| 1374 | `sionna_measurement_sim/io/hdf5_writer.py` |
| 961 | `sionna_measurement_sim/phy/nr_srs_observation.py` |
| 828 | `sionna_measurement_sim/benchmark/runner.py` |
| 773 | `sionna_measurement_sim/config/schema.py` |

最长函数/类片段：

| Lines | Location |
|---:|---|
| 480 | `sionna_measurement_sim/app/cli.py:187 main` |
| 455 | `sionna_measurement_sim/rt/truth_pipeline.py:383 _run_rt_truth_pipeline_single_impl` |
| 378 | `sionna_measurement_sim/phy/nr_channel_backend.py:310 CIRDatasetOFDMChannelBackend` |
| 315 | `sionna_measurement_sim/phy/nr_pusch_observation.py:249 _run_nr_pusch_observation_impl` |
| 282 | `sionna_measurement_sim/phy/nr_srs_observation.py:36 run_nr_srs_observation` |
| 246 | `sionna_measurement_sim/phy/nr_pusch_observation.py:1134 _process_mu_mimo` |
| 236 | `sionna_measurement_sim/io/hdf5_writer.py:441 _write_waveform` |
| 231 | `sionna_measurement_sim/phy/common_link.py:58 ObservationImpairmentChain` |

## Concept Ownership / Single Source of Truth

本节是正式体检的必查项。它不只看“有没有功能”，而是看一个概念是否有明确 owner，
多个表示之间是否有 mapper、adapter、validator、docs 和 tests 防止漂移。

| 概念 | Source of truth | Runtime representation | Mapper / adapter | Output contract | Validator | Docs | Tests | 状态 | 风险链接 |
|---|---|---|---|---|---|---|---|---|---|
| Schema version / contracts | `domain/constants.py`：`SCHEMA_VERSION="2.0.0"`，full/RT labels/IQ link contract names | `MeasurementSimulationResult.metadata`，writer meta fields | writer 从 metadata/constants 写入 | `/meta/schema_version`, `/meta/contract_name`, `/meta/output_profile` | `io/schema_validator.py` | `docs/sys/07_config_and_h5_format.md` | `tests/schema/test_hdf5_schema.py` | Single source | 无 |
| Output profile | `domain/output_plan.py::build_rt_output_plan` + `config/schema.py::OutputConfig.profile` | `RTOutputPlan` 控制 compute/write profile | app/pipeline 构建 plan，truth pipeline 执行 | full、`rt_lite`、`rt_labels_only`、`iq_link_library` | schema validator + profile tests | README, config README, sys docs | `tests/unit/test_output_plan.py`, schema tests | Adapter boundary | R-002 |
| PHY standard registry | `phy/modules.py::PHY_REGISTRY` | `CustomOFDMModule`, `NRPUSCHModule`, `NRSRSModule` | `get_phy_module()` | `meta/phy_standard`, protocol groups | schema validator | `docs/sys/05_phy_observation.md`, `phy_module_development.md` | unit/integration tests | Adapter boundary | R-006 |
| Common PHY observation link | `phy/common_link.py` | `WaveformGrid`, `CleanChannelResult`, `ObservationImpairmentChain` | SRS/PUSCH module adapters | `/waveform/tx_grid`, `/waveform/rx_grid`, `/waveform/noise_variance`, `/observation/cfr_est` | schema validator | sys docs | common link + SRS/PUSCH tests | Single source with protocol adapters | R-001 |
| TX/RX vs BS/UE role semantics | `domain/topology.py`, `domain/link.py` | resolved topology, `LinkConfig` | topology/role resolver | `/link/tx_role`, `/link/rx_role`, channel/observation shapes | schema validator | README, handoff, sys docs | domain/schema/integration tests | Single source | 无 |
| IQ observation | `domain/iq.py` + `phy/iq_observation.py` | `LinkIQCapture`, `NonCooperativeIQCapture`, `IQObservationResult` | PHY waveform extras and multiuser source -> IQ builder | `/iq/link`, `/iq/noncooperative` | schema validator | `docs/sys/05`, `docs/sys/07`, config README | SRS schema and visualization tests | Adapter boundary | R-004 |
| IQ link library profile | `domain/output_plan.py` + `io/hdf5_writer.py::write_iq_link_library_result` | compact clean IQ-only result | clean IQ config adapter in truth pipeline | contract `sionna_measurement_iq_link_library` with clean `/iq/link` | schema validator | README, config README, sys docs | output plan/schema tests | Adapter boundary | R-002/R-004 |
| Multi-UE SRS | `domain/multiuser.py` + `phy/nr_srs_observation.py` | `MultiUserSRSResult` | SRS module optional extra | `/multiuser/*` | schema validator | sys docs/config docs | unit/schema/visualization tests | Adapter boundary | R-002 |
| Ranging config | YAML config in `config/schema.py`; runtime config in `ranging/config.py` | `ranging.runner` consumes runtime dataclasses | `config/mappers.py::to_domain_ranging_config` | `/ranging/*` | config + schema validator | config README, sys docs | `tests/unit/test_config_mappers.py`, ranging tests | Adapter boundary, previously duplicate risk mitigated | R-007 |
| NR SRS resource metadata | `phy/nr_srs_resources.py` + `config/schema.py::SRSConfig` | resource plan, port map, power scale, sequence metadata | SRS waveform builder/receiver | `/waveform/srs_*`, `/observation/cfr_est_resource` | SRS schema validator | sys docs, TODO | SRS resource/schema tests | Adapter boundary | R-005 |
| Shard manifest identity | `domain/results.py::ShardSpec` + manifest writer | per-shard result files + manifest | sharding pipeline/fallback | `manifest/manifest.json`, `results/result_*.h5` | integration/schema checks | sys docs/config docs | sharding tests | Single source, consumer incomplete | R-003 |
| Benchmark harness | `benchmark/cli.py`, `benchmark/runner.py` | `benchmark rt/write/spectrum` | CLI dispatch + reusable runner | JSON/CSV artifacts, not HDF5 schema | command tests/docs | config README, sys docs, performance docs | integration coverage | Adapter boundary, TODO needs refresh | R-008 |
| Custom OFDM legacy | docs/TODO mark legacy | `CustomOFDMModule` remains registry entry | legacy path | legacy waveform exceptions | schema tests allow exceptions | sys docs/TODO | legacy tests | Adapter boundary with debt | R-006 |

结论：现在项目已经具备比较明确的 concept ownership 体系，尤其是 `output_plan`、`domain/iq.py`、
`config/mappers.py` 让新增概念没有完全散落。但输出 contract/profile 的组合仍然集中在
truth pipeline、writer、validator、docs/tests 多处同步；后续每新增一个协议或观测类型，
都应该先做 concept ownership 表，再写代码。

## 详细评分

### 1. 架构分层与职责边界 (18)

| 条目 | 权重 | 等级 | 得分 | 证据 | 置信度 | 改进建议 |
|---|---:|---|---:|---|---|---|
| 依赖方向清晰且基本单向 | 3 | Good | 2.25 | `docs/sys` 分层清楚；AST 检查未发现 `domain/` 或 `io/` 直接 import Sionna/Torch/TensorFlow/Mitsuba。 | High | 增加 import-boundary test，防止未来回归。 |
| domain 模型保持纯净 | 3 | Excellent | 3.00 | `domain/` 承载 constants、output plan、IQ/multiuser/result dataclasses，未耦合外部仿真框架。 | High | 维持。 |
| 模块满足单一职责和内聚性 | 3 | Partial | 1.50 | 多个核心文件超过 1300 行，`truth_pipeline`、`report`、`writer`、`validator`、PUSCH/SRS observation 均承载多职责。 | High | 分阶段拆 orchestration、writer/schema sections、visualization plot builders。 |
| registry/plugin 边界隔离可替换实现 | 3 | Good | 2.25 | PHY registry、common link 和 protocol module 边界存在；IQ/ranging 不直接嵌进 SRS/PUSCH receiver。 | High | writer/schema 也需要类似 per-profile/per-protocol hook。 |
| 公共契约有显式 ownership 和 single-source 边界 | 3 | Good | 2.25 | schema version、output profile、IQ result、ranging mapper、TX/RX role 均有 owner；compact contracts 仍要求多点同步。 | High | 新增概念前强制填 concept ownership 表。 |
| 目录布局对新贡献者可发现 | 3 | Good | 2.25 | `README`、handoff、sys docs 和 skills 描述主要入口；新增 feature docs 同步较快。 | High | 增加“新增 output contract/profile”贡献者 checklist。 |

小结：`13.50 / 18`。分层基础依然好，问题不是方向错，而是核心模块越来越大。

### 2. 协议与功能扩展性 (14)

| 条目 | 权重 | 等级 | 得分 | 证据 | 置信度 | 改进建议 |
|---|---:|---|---:|---|---|---|
| 共享抽象减少协议重复 | 3 | Good | 2.25 | SRS/PUSCH 共享 common link 和 impairment chain；IQ observation 独立于具体 receiver；ranging 读取统一 `cfr_est`。 | High | 后续 WiFi-like/6G-like 应复用 common link 和 IQ/ranging runner，不重写观测链路。 |
| 配置扩展模型有一个输入 contract 和显式 runtime mapper | 3 | Good | 2.25 | 全局 YAML/Pydantic config 是输入 contract；ranging 已有 `config/mappers.py` 和 mapper 单测。 | High | 把 mapper/equivalence-test 模式推广到未来 algorithm-private config。 |
| 协议私有 waveform/receiver 逻辑隔离 | 3 | Good | 2.25 | SRS resource、PUSCH receiver/backend、IQ builder、ranging estimator 各自独立。 | High | 拆分 PUSCH/SRS 主函数，减少私有逻辑中的 orchestration 噪音。 |
| schema/迁移策略可承载新输出 | 3 | Good | 2.25 | schema `2.0.0` 支持 full、RT labels、IQ link library contracts，并拒绝不该出现的 group。 | High | writer/validator 需要描述式 field spec 或 hook，降低新增 contract 的修改面。 |
| defaults/feature flags 支持增量采用 | 2 | Good | 1.50 | `output.profile`、`phy.iq`、`noncooperative`、`srs.multiuser`、ranging、array、visualization 均可开关。 | High | 对重型组合补推荐 preset 和成本矩阵。 |

小结：`10.50 / 14`。扩展能力比上次更强，但仍以集中式 profile/contract 分支为代价。

### 3. 数据契约与可复现性 (14)

| 条目 | 权重 | 等级 | 得分 | 证据 | 置信度 | 改进建议 |
|---|---:|---|---:|---|---|---|
| HDF5 schema、writer、validator 对齐 | 3 | Good | 2.25 | schema tests 通过；full/compact contracts 写入和校验已覆盖；schema version 为 `2.0.0`。 | High | 将大型 writer/validator 拆为 profile/protocol sections。 |
| 维度语义和 BS/UE vs TX/RX 明确 | 3 | Excellent | 3.00 | handoff、README、config README、sys docs 均说明 role-view/link-view；前序 AoA 方向问题已修正到 scene frame。 | High | 维持，并让 reader 默认暴露 role-aware API。 |
| manifest/config snapshot 保留 shard provenance | 2 | Excellent | 2.00 | full 和 compact contracts 均记录 run/config/runtime；sharding/fallback 规则文档明确。 | High | reader 完成前不要让下游脚本绕过 manifest。 |
| 随机种子和小实验可复现 | 2 | Good | 1.50 | `runtime.seed`、smoke 记录、config templates、pytest fixtures 存在。 | Medium | 标准化 smoke summary JSON/CSV，减少人工口头验收。 |
| `data/` 和 `outputs/` 安全隔离 | 2 | Excellent | 2.00 | docs/skills 明确禁止递归扫描，git ignore 大数据路径。 | High | 维持。 |
| units/attrs/metadata 写入一致 | 2 | Good | 1.50 | docs/sys/07 覆盖 unit/index_order；schema validator 校验核心字段。 | Medium | 对 IQ time-domain convention、noncooperative 标签、compact contracts 增加 reader-facing examples。 |

小结：`12.25 / 14`。数据契约是强项，消费侧 reader 仍是主要缺口。

### 4. 代码可维护性与复杂度 (16)

| 条目 | 权重 | 等级 | 得分 | 证据 | 置信度 | 改进建议 |
|---|---:|---|---:|---|---|---|
| 大文件和长函数受控或有合理解释 | 3 | Partial | 1.50 | 最大文件已到 2286/2118/1923/1790 行；`app.main`、truth pipeline、PUSCH/SRS 主函数偏长。 | High | 优先拆 `truth_pipeline.py`、`visualization/report.py`、`schema_validator.py`、`hdf5_writer.py`。 |
| 重复逻辑提取到合适 helper | 3 | Partial | 1.50 | common link/output plan/ranging mapper 是进步；但 writer/schema/profile/visualization 分支仍集中重复。 | High | 引入 profile writer hook、schema field spec、visualization plot registry。 |
| 错误处理明确且必要时 fail-fast | 2 | Good | 1.50 | unsupported profile/config、IQ link library 约束、noncooperative 约束、schema forbidden groups 有检查。 | High | 对 fail-fast 错误补统一错误信息格式，方便 CLI/日志定位。 |
| 类型和 dataclass 表达边界 | 2 | Good | 1.50 | `domain/iq.py`、`domain/multiuser.py`、`domain/output_plan.py`、ranging dataclasses 表达边界清楚。 | High | `PHYModuleResult.multiuser: Any` 这类字段可逐步收紧。 |
| 性能/资源关注点在代码路径中可见 | 2 | Good | 1.50 | benchmark runner、PerfTracer、sharding、多 GPU、link chunk、output profile 都存在。 | Medium | 用 benchmark 数据反向驱动默认配置。 |
| TODO/技术债有耐久跟踪 | 2 | Excellent | 2.00 | `docs/todo/` 当前 active TODO 22 个，bug 0。 | High | 更新已部分完成的 benchmark TODO，避免 TODO drift。 |
| 依赖和全局状态可控 | 2 | Good | 1.50 | 依赖集中在 `pyproject.toml`；registry 和 runtime context 是主要全局入口。 | Medium | 对 GPU/runtime/global seed 再补边界测试。 |

小结：`11.00 / 16`。这是当前最大短板：功能增长速度快于结构拆分速度。

### 5. 测试与验证体系 (14)

| 条目 | 权重 | 等级 | 得分 | 证据 | 置信度 | 改进建议 |
|---|---:|---|---:|---|---|---|
| 单元测试覆盖纯逻辑和 shape contract | 3 | Excellent | 3.00 | `tests/unit` 27 文件，覆盖 config mapper、output plan、SRS resources、IQ visualization、ranging 等。 | High | 维持。 |
| 集成和 smoke 覆盖主流程 | 3 | Good | 2.25 | `tests/integration` 8 文件；全量 pytest 通过，但 19 个依赖历史输出或大输出的测试 skipped。 | High | 将依赖外部输出的 skipped 用小 fixture 替代。 |
| schema、statistical、regression 能捕获契约漂移 | 3 | Good | 2.25 | schema/statistical/adapter 测试存在，schema `2.0.0` 与 compact contracts 已有覆盖。 | High | 对 IQ link library 和 noncooperative 增加更多 failure-path schema tests。 |
| 真实小实验摘要验证数据产品 | 2 | Good | 1.50 | handoff 记录 64PRB、10UE、formal runs 等；本轮未重新跑真实仿真。 | Medium | 固化 small-experiment summary 生成和存档位置。 |
| 失败路径和非法配置有测试 | 2 | Good | 1.50 | unknown profile、ranging mapper 非法值、IQ profile 约束等有测试。 | High | 对 output.profile 与 visualization/array/ranging 的组合矩阵补负例。 |
| 标准开发命令当前有效 | 1 | Excellent | 1.00 | `ruff` 通过；pytest `292 passed, 19 skipped, 1 warning`。 | High | 维持。 |

小结：`11.50 / 14`。自动化质量健康，但跳过的真实输出依赖测试仍应逐步 fixture 化。

### 6. 文档与上手成本 (10)

| 条目 | 权重 | 等级 | 得分 | 证据 | 置信度 | 改进建议 |
|---|---:|---|---:|---|---|---|
| handoff 和 sys docs 描述当前事实 | 2 | Excellent | 2.00 | handoff/sys docs 覆盖 schema `2.0.0`、output profiles、IQ、multiuser、benchmark。 | High | 维持。 |
| config README 与模板同步 | 2 | Good | 1.50 | `config/README.md` 覆盖 `iq_link_library`、`phy.iq`、`noncooperative`、SRS multiuser。 | High | 对每个 profile 增加最小可运行命令和输出 group 快览。 |
| 架构/API 文档解释扩展点 | 2 | Good | 1.50 | `docs/sys/05`、`06`、`07` 解释 PHY/IQ/schema；skills 记录工程流程。 | High | 增加 output contract/profile 新增 checklist。 |
| TODO 和限制明确可执行 | 2 | Excellent | 2.00 | feature/structure/performance/bug 分类清楚，active TODO 22 个。 | High | 清理 STR-002 与已存在 benchmark 的重叠描述。 |
| skills/runbooks 捕获重复工程流程 | 2 | Excellent | 2.00 | 本地已有 dev workflow、doc maintenance、todo docs、health check skills。 | High | 若 benchmark/smoke 流程稳定，可新增专门 skill。 |

小结：`9.00 / 10`。文档是强项，主要要防止 TODO 和代码事实不同步。

### 7. 运行与实验工程 (8)

| 条目 | 权重 | 等级 | 得分 | 证据 | 置信度 | 改进建议 |
|---|---:|---|---:|---|---|---|
| sharding 和多 GPU 执行清楚 | 2 | Good | 1.50 | config/sys docs 描述 shard、parallel workers、GPU IDs、fallback。 | High | 测 2/4/6/8 GPU 扩展性和尾 shard 调度。 |
| I/O 体积和输出布局被有意管理 | 2 | Good | 1.50 | `rt_lite`、`rt_labels_only`、`iq_link_library` 明确降低输出体积。 | High | 用 benchmark write 系统化比较 profile 写盘成本。 |
| profiling/logs 支持定位慢或失败运行 | 1 | Good | 0.75 | `PerfTracer`、`benchmark rt/write/spectrum`、debug profiling 存在。 | Medium | 将 benchmark 结果纳入 docs/performance 索引和推荐阈值。 |
| 本地大数据路径处理安全 | 2 | Excellent | 2.00 | docs/skills 明确不递归扫描 `data/`、`outputs/`。 | High | 维持。 |
| visualization/diagnostics 支持小样本检查 | 1 | Good | 0.75 | visualization 支持标准图、multiuser、IQ、RSS radio map；`report.py` 很大且仍有性能 TODO。 | Medium | 拆 plot registry，降低单文件复杂度和 first-shard 可视化长尾。 |

小结：`6.50 / 8`。运行工程已经有工具箱，但性能数据还需要系统化闭环。

### 8. 变更治理 (6)

| 条目 | 权重 | 等级 | 得分 | 证据 | 置信度 | 改进建议 |
|---|---:|---|---:|---|---|---|
| 分支和提交纪律保持可 review | 2 | Good | 1.50 | 当前 feature branch 4 个相关 commits；工作区初始干净。 | High | 大功能继续按阶段提交，避免把数据输出混入。 |
| docs/config/schema 与行为变更同步 | 2 | Good | 1.50 | diff 中 README/config/sys/docs/tests 与代码一同更新。 | High | 对 breaking schema/profile 增加 migration checklist。 |
| 兼容性和迁移影响说明清楚 | 1 | Good | 0.75 | docs 描述 compact contracts、forbidden groups、legacy custom OFDM。 | Medium | 为下游训练/reader 提供 profile 迁移表。 |
| review、验证和后续闭环可见 | 1 | Good | 0.75 | ruff/pytest 结果明确，TODO 持续维护。 | Medium | 将每次真实 smoke summary 按固定路径归档。 |

小结：`4.50 / 6`。治理纪律不错，下一步需要把大变更的迁移影响机器化。

## 风险列表

| ID | Priority | Area | Evidence | Impact | Likelihood | Recommendation | Owner or Trigger | Status |
|---|---|---|---|---|---|---|---|---|
| R-001 | P2 | 核心复杂度 | `visualization/report.py` 2286 行，`truth_pipeline.py` 2118 行，`nr_pusch_observation.py` 1923 行，`schema_validator.py` 1790 行，`hdf5_writer.py` 1374 行；多个函数/类超过 200 行。 | 新增协议、profile、schema 字段时修改面过大，容易回归。 | High | 按 pipeline stage、profile writer、protocol schema section、plot registry 渐进拆分。 | 下一次新增协议或 output contract 前 | Open |
| R-002 | P2 | Output contract/profile 矩阵 | full、`rt_lite`、`rt_labels_only`、`iq_link_library` 同时牵动 config、output plan、truth pipeline、writer、validator、docs/tests。 | 新增 profile 或字段时容易漏写/漏校验/漏文档，造成 contract drift。 | High | 引入 profile-specific writer/validator hooks 或 declarative field spec，并让 output plan 成为更强 owner。 | 下一次 profile/schema breaking change 前 | Open |
| R-003 | P2 | 下游数据读取 | `docs/todo/structure.md` STR-001：多 `result_xxx.h5` 输出缺少统一 shard-aware reader / dataset loader。 | 训练/分析侧容易手写遍历、误读 fallback shard 或忽略 global UE/BS index。 | High | 优先实现 manifest-aware reader，支持 profile-aware group discovery 和 shard iteration。 | 下一轮大规模训练/分析接入前 | Open |
| R-004 | P2 | IQ/noncooperative 科学语义 | `/iq/noncooperative` 当前是基于 NR SRS shared frequency grid 合成的 BS 侧 time IQ；TODO FEAT-IQ-001 仍需 raw-IQ 前端模型。 | 若下游把它当作真实连续 RF/ADC raw IQ，可能得出过强或错误结论。 | Medium | 在 docs/reader/report 中强标 v1 语义；后续实现 per-UE async/CFO/timing/ADC/collision raw-IQ frontend。 | 使用非合作 IQ 写论文或训练模型前 | Open |
| R-005 | P2 | SRS 标准声明边界 | `docs/todo/feature.md` FEAT-SRS-001：当前是 standards-shaped v2 subset，不是 3GPP-compliant。 | 对外论文/文档若误称标准完整，会影响可信度。 | Medium | 做 38.211/38.213 reference validation，并输出声明矩阵。 | 对外写作或标准完整性实验前 | Open |
| R-006 | P3 | legacy custom OFDM | `docs/todo/structure.md` STR-003；`custom_ofdm` 仍在 registry，且不走新 common link/output profile 主线。 | 长期保留例外会拖累 schema/docs 和新协议抽象。 | Medium | 决定迁移到 common link 或正式 legacy/remove，并清理 schema/docs/tests 例外。 | custom OFDM 再次被用于生产前 | Open |
| R-007 | P3 | Config adapter 治理 | `RangingConfig`、`PdpPeakRangingConfig`、`PhaseSlopeRangingConfig` 在 YAML schema 和 runtime dataclass 中双表示；目前已有 mapper 和单测。 | 该问题已缓解，但未来 algorithm-private config 容易复现同类漂移。 | Medium | 把 `config/mappers.py` + equivalence tests 作为所有 runtime config 的标准模式写进开发 checklist。 | 新增 ranging estimator 或独立算法包 config 前 | Open |
| R-008 | P3 | TODO/benchmark 状态漂移 | `benchmark rt/write/spectrum` 已存在，但 STR-002 仍写“建立 RT-only、PHY-only、write-only 等稳定 CLI/API”。 | TODO 与现实部分重叠，会降低 TODO 作为路线图的可信度。 | Medium | 更新 STR-002：关闭已完成部分，拆出缺失的 PHY-only/API/summary 标准化子项。 | 下一次 TODO 整理 | Open |
| R-009 | P3 | 性能与可视化成本 | `docs/todo/performance.md` 仍有 HDF5 write、RT 参数、空间谱、visualization、多 GPU 扩展性等 7 个 active TODO。 | 大规模仿真可能受写盘、空间谱、可视化长尾和 GPU 调度影响。 | Medium | 用 `benchmark rt/write/spectrum` 建立基线和推荐配置，并把结果写入 docs/performance。 | 下一轮正式全量仿真前 | Open |
| R-010 | P4 | Profile 迁移体验 | compact contracts 与 full contract 差异已文档化，但尚无统一 reader 或迁移表。 | 新用户或下游脚本可能不知道哪个 profile 有哪些 group。 | Medium | 在 config/sys docs 增加 profile-to-groups 表，并在未来 reader 中暴露 profile-aware API。 | reader 实现或新下游接入时 | Open |

当前没有确认中的 P0/P1 风险。

## 建议

| 优先级 | 工作流 | 建议 | 预期收益 | 建议验收 |
|---|---|---|---|---|
| P2 | 架构拆分 | 拆分 `truth_pipeline.py`、`hdf5_writer.py`、`schema_validator.py`、`visualization/report.py` 的 profile/protocol/plot 分支。 | 降低新增协议和 contract 的修改面。 | 拆分后外部 CLI/HDF5 契约不变，ruff/pytest 全过。 |
| P2 | 数据消费 | 实现 shard-aware、profile-aware reader / dataset loader。 | 让训练/分析不再手写 HDF5 shard 遍历。 | fixture + 真实 manifest smoke，覆盖 fallback shard 和 compact contracts。 |
| P2 | Contract 治理 | 设计 output profile writer/validator hook 或 declarative field spec。 | 减少 full/compact/profile 矩阵的重复条件逻辑。 | 新增 toy profile/schema fixture 验证 hook。 |
| P2 | IQ 语义 | 为 `/iq/noncooperative` v1 增加更醒目的语义说明，并规划 raw-IQ frontend。 | 避免论文/模型误用观测语义。 | docs + schema attrs + visualization title 明确 v1 convention；后续 FEAT-IQ-001 验收。 |
| P2 | 标准声明 | 做 NR SRS reference validation。 | 支撑 standards-shaped subset 的可信边界。 | 38.211/38.213 reference cases + 文档声明矩阵。 |
| P3 | Config 模式 | 将 `config/mappers.py` 模式固化为开发规范。 | 防止 future config/runtime dataclass 漂移。 | 新增 mapper tests 模板，health check concept ownership 表持续检查。 |
| P3 | TODO 维护 | 更新 STR-002 benchmark TODO。 | 保持 TODO 作为工程路线图可信。 | 已完成项归档或拆分，README counts 更新。 |
| P3 | 性能工程 | 用 benchmark 体系建立 write/spectrum/visualization/gpu 调度基线。 | 大规模仿真前可预估瓶颈。 | JSON/CSV summary + docs/performance 记录 + 推荐配置。 |

## 未覆盖范围和盲点

- 未递归扫描或读取 `data/`、`outputs/`。
- 未运行新的真实仿真 smoke；本报告使用当前文档、代码结构、git diff、测试和现有 TODO 作为证据。
- 未逐行审查全部算法正确性；重点是结构健康、扩展性、职责边界、单一事实源和工程风险。
- 未评估仓库外下游训练代码；当前仍缺正式 shard-aware reader。
- 本报告基于分支 `codex/noncooperative-iq-observation`，不是远端主分支快照。

## 后续

- 下次建议复查触发点：完成 shard-aware reader、拆分 writer/validator、实现 raw-IQ frontend、
  或新增 WiFi-like/6G-like PHY module 前。
- 模板版本建议：v1 仍可继续使用；若升级 v2，建议加入自动指标附录、profile/contract 矩阵
  和概念 ownership 自动扫描脚本。
