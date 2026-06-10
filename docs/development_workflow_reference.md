# 项目开发规范参考

本文档整理一套可迁移到其他项目的开发协作规范。它适合中长期迭代、包含代码、配置、数据产物、文档和实验记录的工程项目。使用时可以按目标项目的目录结构替换示例路径。

## 1. 基本原则

1. **先读当前事实，再动手修改**  
   每次开始任务前先阅读项目交接文档、系统说明、README、配置说明和任务相关模块文档。不要只依赖记忆或旧实验记录。

2. **代码、配置、文档、测试同步演进**  
   行为变化不能只改代码。涉及配置、输出格式、数据 schema、CLI、实验流程或用户可见行为时，要同步更新 README、配置文档、系统文档、TODO 和交接说明。

3. **小步提交，提交语义清楚**  
   每个提交应围绕一个明确目的，避免把无关脚本、临时结果、格式化噪声和核心改动混在一起。

4. **保留用户和他人的改动**  
   工作区可能已有别人或用户的改动。不要随意 reset、checkout 或覆盖。修改前检查状态，编辑时只碰任务相关文件。

5. **大数据和运行产物默认不入库**  
   `data/`、`outputs/`、临时缓存、模型权重、实验结果等路径应默认 ignored。需要读取时只使用明确路径，不递归扫描大数据目录。

## 2. 开始任务的固定流程

每个开发任务建议按以下顺序开始：

1. 检查分支和工作区状态：

   ```bash
   git status --short --branch
   ```

2. 阅读当前事实文档，例如：

   - `README.md`
   - `config/README.md`
   - `docs/agent_handoff.md`
   - `docs/sys/README.md`
   - 与本任务相关的 `docs/sys/*.md`

3. 明确任务类型：

   - 代码行为修复
   - 新功能
   - 配置或 schema 变化
   - 性能工程
   - 文档整理
   - 实验/仿真运行
   - 数据分析脚本

4. 对大改动单拉分支：

   ```bash
   git switch main
   git pull --ff-only
   git switch -c codex/<task-name>
   ```

5. 识别不应纳入提交的本地文件，例如临时配置、输出目录、未跟踪分析结果。

## 3. 分支与提交规范

### 分支

- 大功能、重构、schema 变化、性能工程、输出格式变化都应单独建分支。
- 分支名使用清楚的任务语义，例如：

  ```text
  codex/unified-power-model
  codex/schema-output-cleanup
  codex/perf-benchmark-harness
  ```

### 提交

- 及时提交已经完成并验证的工作。
- 提交信息使用动词开头，说明实际变化：

  ```text
  Add compact labels-only output contract
  Refactor uplink power scaling into common module
  Update docs for schema 1.6.0
  ```

- 不要提交：

  - `data/`
  - `outputs/`
  - 临时日志
  - 本地实验缓存
  - 与任务无关的格式化改动

## 4. 代码修改规范

1. **优先沿用现有架构**  
   先理解项目已有模块边界，再决定是否新增抽象。

2. **共享能力放在通用层**  
   如果功能未来会被多个协议、模型、数据格式或 pipeline 复用，应放在协议无关或业务无关模块中。具体标准或场景只做轻量适配。

3. **领域模型、配置模型、IO 模型分层**  
   配置负责校验用户输入；领域模型负责运行时语义；IO/writer 负责落盘契约。不要让算法模块直接依赖 YAML/Pydantic/HDF5。

4. **新增字段要有明确语义**  
   字段名应能看出是 truth、observation、estimate、metadata 还是 derived label。避免模糊命名导致后续误用。

5. **破坏式变更要显式标识**  
   修改输出格式、schema 或字段含义时，应升级版本号并写迁移说明。

## 5. 配置与 Schema 规范

涉及配置变化时，至少检查：

- 配置模型
- 默认模板
- 示例配置
- `config/README.md`
- CLI 参数或启动脚本
- 测试中的配置 fixture

涉及 HDF5、JSON、数据库表或其他输出契约变化时，至少检查：

- writer/exporter
- schema validator
- reader
- 下游分析脚本
- 可视化脚本
- schema 测试
- `docs/sys` 中的数据格式说明
- 交接文档

新增字段时建议记录：

- shape
- dtype
- unit
- index order
- 是否 optional
- 失败值策略，例如 `NaN`、`false`、空数组
- 与旧字段的迁移关系

## 6. 文档维护规范

完成较大改动后，应主动检查以下文档是否需要更新：

- `README.md`：项目入口、核心能力、快速开始
- `config/README.md`：配置项、默认模板、常用运行方式
- `docs/sys/`：系统事实、架构、数据格式、模块职责
- `docs/agent_handoff.md`：当前状态、重要约定、下一步提醒
- `docs/todo/`：未完成能力、技术债、后续计划
- `docs/performance/`：性能实验和 benchmark 记录
- `docs/legacy/`：已经过时但暂不删除的历史文档

如果发现文档已经过时：

1. 小范围过时：直接更新。
2. 大段内容已不再代表当前事实：移动到 `docs/legacy/`。
3. 需要保留历史实验：放在 `docs/performance/` 或相应历史目录，并明确它不是当前默认配置。

## 7. TODO 管理规范

TODO 应集中管理，不散落在多个临时文档中。建议目录：

```text
docs/todo/
  README.md
  feature.md
  bug.md
  performance.md
  structure.md
  history.md
```

每个分类文档建议包含：

1. 当前该类 TODO 的重要程度排序列表。
2. 每个 TODO 的简述。
3. 详细描述：
   - 涉及模块
   - 当前问题
   - 目标效果
   - 粗略验收标准
   - 重点提醒或参考方向

完成 TODO 时：

- 从 active 分类文档移除或标记完成。
- 在 `history.md` 按分类记录完成时间和简述。
- 检查排序是否需要调整。

## 8. 测试与验证规范

根据风险选择验证范围：

- 小工具或文档：格式检查、目标脚本 smoke 即可。
- 单模块逻辑：新增或更新 unit test。
- 配置/schema/IO：必须覆盖 writer、reader、validator 和文档。
- pipeline 或仿真行为：跑小规模真实 smoke，输出到 ignored 目录。
- 合并前常规检查：

  ```bash
  uv run ruff check <project> tests scripts
  uv run pytest -q
  git diff --check
  ```

如果测试无法运行，要在最终说明中明确：

- 没跑什么
- 为什么没跑
- 剩余风险是什么

## 9. 数据、输出与长任务规范

1. **输出路径要清晰**  
   每次实验输出到独立目录，目录名体现数据集、场景、配置和关键参数。

2. **运行配置随结果保存**  
   最终生效的 config snapshot 应写到对应输出目录，例如：

   ```text
   outputs/<run_name>/
     config_resolved.yaml
     summary.json
     logs/run.log
   ```

3. **长任务放后台并可监控**  
   大规模运行建议用 `tmux` 或任务队列，并写日志到运行目录。

4. **任务异常要留下记录**  
   如果出现卡住、fallback、OOM、设备占用等问题，记录：

   - 时间
   - 命令/配置
   - 失败现象
   - 初步判断
   - 应对策略

5. **不要污染仓库根目录**  
   日志、summary、临时配置应写入具体 run 目录，而不是散落在项目根目录或 `outputs/` 根目录。

## 10. 多 Agent 协作规范

可以并行开发时再使用多 agent，适合：

- 独立模块实现
- 文档审查
- 测试补全
- 只读分析
- 不同实验结果汇总

不适合并行的情况：

- 多个 agent 修改同一文件
- schema/writer/reader 强耦合改动无人统一收口
- 需要连续推理和调试的关键路径

多 agent 协作时需要明确：

- 每个 agent 的文件边界
- 不允许覆盖他人改动
- 输出 changed files 列表
- 主 agent 最终统一 review、测试和提交

## 11. 交接规范

长任务或阶段性工作结束后，应更新交接信息，至少包括：

- 当前分支
- 最近提交
- 已完成事项
- 未完成事项
- 重要输出路径
- 已知问题
- 下一步建议
- 不能触碰或需要谨慎处理的本地路径

交接文档应尽量描述当前事实，而不是历史计划。

## 12. 合并规范

合并回主分支前：

1. 确认工作区只包含本任务改动。
2. 运行必要测试。
3. 更新文档和 TODO。
4. 提交当前分支。
5. 切回主分支并 fast-forward 合并：

   ```bash
   git switch main
   git pull --ff-only
   git merge --ff-only codex/<task-name>
   ```

6. 合并后再次检查：

   ```bash
   git status --short --branch
   ```

## 13. 可复制的项目约定模板

新项目可以在仓库中建立类似文件：

```text
docs/
  agent_handoff.md
  sys/
    README.md
    00_project_overview.md
    01_app_and_config.md
    02_data_format.md
    03_pipeline.md
    04_io_and_testing.md
  todo/
    README.md
    feature.md
    bug.md
    performance.md
    structure.md
    history.md
  legacy/
    README.md
  performance/
    README.md
```

也可以增加项目级 agent/skill 约束：

- 开发 workflow skill
- 文档维护 skill
- TODO 管理 skill
- 项目健康检查 skill

这些规范的核心不是目录本身，而是让项目长期保持：

- 当前事实有地方查
- 过时内容有地方归档
- TODO 不散落
- 输出和配置可复现
- 代码、文档、测试一起演进
- 每次合并都能解释清楚系统现在是什么状态
