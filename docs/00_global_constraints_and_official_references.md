# 00. 全局约束与官方参考

本文定义所有后续开发文档和实现必须遵守的全局约束，并列出开发时优先参考的官方文档。若本文与其他文档冲突，以本文和 [03_data_contract_hdf5.md](03_data_contract_hdf5.md) 为准；若官方 Sionna 2.x API 与本文假设冲突，必须先更新 adapter 设计文档 [04_sionna_rt_adapter_and_path_data.md](04_sionna_rt_adapter_and_path_data.md)，再修改代码。

## 1. 约束关键词

本文档群使用以下语义：

- 必须：实现和验收都不能跳过。
- 禁止：不得在新系统中出现。
- 应该：默认采用；若不采用，必须在设计记录或 commit message 中解释原因。
- 可选：不阻塞当前阶段，但 schema 或扩展点应预留。

## 2. 平台约束

必须：

- 新系统使用 Sionna 2.x。
- 新系统使用 Python 3.11+。
- 新系统使用 PyTorch 作为 PHY/SYS 方向的主张量后端。
- 新系统使用 `uv` 管理项目环境、依赖和 lockfile。
- 新系统 git 根目录为 `SionnaMeasurementSim/`。
- 旧项目只能放在 `SionnaMeasurementSim/old/` 中作为参考，不得被新系统 import。

禁止：

- 在新核心链路中以 TensorFlow 作为主依赖。
- 业务层直接读取 Sionna `Paths` 属性。
- HDF5 writer 直接消费 Sionna 原生对象。
- 新 writer 将 truth CFR 主路径写为 `/channel/cfr`。
- 提交大型仿真输出、实测数据或全量 HDF5 到 git。

## 3. 数据契约约束

必须：

- HDF5 内包含 `/meta/schema_version`。
- HDF5 内包含 `/meta/index_order`。
- HDF5 内包含 `/meta/unit_convention`。
- truth CFR 使用 `/channel/truth/cfr`。
- observed CFR/CSI 使用 `/observation/cfr_est`。
- truth CFR 维度为 `[tx, rx, rx_ant, tx_ant, subcarrier]`。
- observed CFR 维度为 `[snapshot, tx, rx, rx_ant, tx_ant, subcarrier]`。
- Sionna 原生 rx-first shape 必须由 adapter 转为 TX-first。
- 路径级数据必须至少保存 samples；debug/标定阶段应该保存 full paths。

路径级 samples 必须至少包含：

```text
sampled_link_indices
vertices_m
interaction_type
object_id
primitive_id
doppler_hz
tau_s
path_gain_db
path_type
```

## 4. 官方参考链接

### 4.1 Sionna 安装与平台

开发环境和版本约束优先参考：

- Sionna installation: https://nvlabs.github.io/sionna/installation.html

该页面说明 Sionna 由 RT、PHY、SYS 三个模块组成，并说明 PHY/SYS 需要 Python 3.11+ 和 PyTorch 2.9+；RT 基于 Mitsuba 3 和 Dr.Jit。

### 4.2 Sionna RT 入门

RT 场景加载、设备注册、PathSolver、CIR/CFR 基础用法优先参考：

- Introduction to Sionna RT: https://nvlabs.github.io/sionna/rt/tutorials/Introduction.html

### 4.3 Paths API

路径对象字段、`cfr()`、`cir()`、`taps()`、`a`、`tau`、`doppler`、`interactions`、`objects`、`primitives`、`vertices` 等优先参考：

- Paths API: https://nvlabs.github.io/sionna/rt/api/paths.html

### 4.4 Paths 对象理解

如何查看 path coefficient、delay、AoA/AoD、Doppler、interactions、objects、vertices、primitive normal，优先参考：

- Understanding the Paths Object: https://nvlabs.github.io/sionna/rt/developer/dev_understanding_paths.html

### 4.5 Mobility 与 Doppler

Doppler、速度、time-varying CFR/CIR、Delay-Doppler 分析优先参考：

- Tutorial on Mobility: https://nvlabs.github.io/sionna/rt/tutorials/Mobility.html

### 4.6 uv

项目初始化、`pyproject.toml`、`.python-version`、`.venv`、`uv.lock`、`uv run`、`uv sync` 以官方 uv 文档为准：

- Working on projects: https://docs.astral.sh/uv/guides/projects/
- CLI reference: https://docs.astral.sh/uv/reference/cli/

uv 官方文档说明 `uv.lock` 应提交到版本控制，用于跨机器可复现安装；`uv run` 会基于 lockfile 和项目环境运行命令。

## 5. 变更控制

任何改动如果影响以下内容，必须同步更新文档和测试：

- HDF5 group 或 dataset 名称。
- 维度顺序。
- 单位。
- Sionna adapter shape 转换。
- `H_true`/`H_obs` 语义。
- 配置 schema。
- 阶段验收标准。

不允许“先实现、以后补文档”。文档、实现、测试必须同一阶段闭环。

