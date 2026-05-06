# 07. 项目目录、uv 与 Git 工作流

本文定义新项目工程规范。实现阶段必须与 [08_roadmap_milestones_acceptance.md](08_roadmap_milestones_acceptance.md) 的里程碑配合，每个阶段通过测试后及时提交 git。

## 1. 仓库位置

git 仓库根目录是：

```text
SionnaMeasurementSim/
```

Python 包目录是：

```text
SionnaMeasurementSim/sionna_measurement_sim/
```

不要把 git 仓库 init 在 Python 包目录内部。

## 2. 推荐根目录

```text
SionnaMeasurementSim/
  .git/
  README.md
  pyproject.toml
  uv.lock
  .gitignore

  sionna_measurement_sim/
  config/
  docs/
  tests/
  scripts/
  data/
    examples/
    scenes/
      test/
    measurements/
    calibration/
  outputs/
  artifacts/
  old/
```

## 3. old 目录

旧项目放入：

```text
SionnaMeasurementSim/old/SimpleSionna/
```

新代码不得 import `old/` 中的模块。旧项目只能作为：

- 设计参考。
- 迁移对照。
- 输出格式对比。
- 临时人工查阅。

迁移规则见 [10_old_system_reuse_and_migration.md](10_old_system_reuse_and_migration.md)。

## 4. 数据目录

测试场景约定：

```text
SionnaMeasurementSim/data/scenes/test/
```

用途：

- 小规模 RT 集成测试。
- adapter shape 验证。
- HDF5 readback。
- 可视化 smoke test。

大型数据默认不提交 git。

## 5. .gitignore 建议

```gitignore
.venv/
__pycache__/
.pytest_cache/
.ruff_cache/

outputs/
artifacts/
data/measurements/
data/scenes/**/large/

*.h5
*.hdf5
*.npy
*.npz
*.pt
*.pth
```

如果需要提交小型 HDF5 fixture，应放在：

```text
tests/fixtures/
```

并在 PR/commit 说明中解释大小和用途。

## 6. uv 环境管理

新项目使用 `uv`。

官方参考：

- https://docs.astral.sh/uv/guides/projects/
- https://docs.astral.sh/uv/reference/cli/

初始化：

```bash
uv init
uv python pin 3.11
uv add numpy h5py pyyaml pydantic matplotlib
uv add --dev pytest ruff
```

Sionna/PyTorch 依赖应按官方安装要求确认后加入。涉及 CUDA/PyTorch 时，必须在安装文档中记录平台和命令。

Sionna 官方安装参考：

- https://nvlabs.github.io/sionna/installation.html

约束：

- `uv.lock` 必须提交 git。
- `.venv/` 必须加入 `.gitignore`。
- CI 或阶段验收应使用 `uv sync` 和 `uv run`，避免依赖手工激活环境。
- 不允许手工编辑 `uv.lock`。

常用命令：

```bash
uv sync
uv run pytest
uv run ruff check .
uv run python -m sionna_measurement_sim.app.cli --help
```

## 7. pyproject 约束

建议启用：

- `ruff`
- `pytest`
- `src` 或 flat package 统一一种布局
- Python 版本锁定

如果使用 flat package：

```text
sionna_measurement_sim/
```

作为根目录下直接包即可。

## 8. Git 分支规范

推荐：

```text
main
feat/phase-0-skeleton
feat/phase-1-rt-truth
feat/phase-2-path-adapter
feat/phase-3-phy-observation
```

每个 phase 对应一个或多个小分支，但不要把多个阶段混在一个巨型提交中。

## 9. 提交规范

提交粒度：

- 一个可验证功能。
- 一个 schema 变更。
- 一个 adapter 能力。
- 一个测试补充。

提交信息建议：

```text
phase1: add rt truth result dataclasses
phase1: write hdf5 truth cfr schema
phase2: extract sionna path vertices and interactions
test: add hdf5 readback contract tests
```

## 10. 每阶段提交要求

每个阶段结束必须：

1. 运行该阶段要求的测试。
2. 更新或确认文档。
3. 确认 `git status` 中没有意外大文件。
4. 提交代码。
5. 在 commit message 或阶段记录中写明通过的验收项。

阶段定义见 [08_roadmap_milestones_acceptance.md](08_roadmap_milestones_acceptance.md)。

## 11. 大文件策略

禁止直接提交：

- 全量仿真输出。
- 大型 HDF5。
- 大型场景资产。
- 实测数据。

如需版本化大文件，后续再决定使用 Git LFS、DVC 或外部对象存储。
