# SionnaMeasurementSim

基于 [Sionna 2.x](https://nvlabs.github.io/sionna/) 的信道测量仿真系统。生成射线追踪真值（H_true），施加 PHY 损伤和信道估计，输出仿真观测（H_obs），所有数据按 HDF5 契约落盘。

**当前能力：**
- Sionna RT 射线追踪（LoS / 反射 / 折射 / 绕射 / 漫反射）
- 多快照运动与多普勒
- Custom OFDM + AWGN + LS 估计 + 全链路 impairment（CFO/SFO/相偏/定时偏/AGC/削波）
- **NR PUSCH 4x4 SU-MIMO**（perfect CSI + DMRS LS 估计，LMMSE/KBest 检测器，StreamManagement）
- **NR PUSCH MU-MIMO**（多 UE 联合 PUSCH，DMRS port set 分配）
- 两个可插拔信道后端：`ApplyOFDMChannel`（稳定）+ `CIRDataset + OFDMChannel`（官方 API）
- TB/CRC 语义的 BLER（transport block CRC pass/fail）
- TDD 互易性（DL RT trace → UL PUSCH）
- HDF5 schema 强校验（包括 NR PUSCH 专有字段）
- 批量实验（多 seed/SNR 分批）
- 176 个测试覆盖（单元 / schema / adapter / 集成 / 统计）

## 快速开始

```bash
# 安装环境
uv sync

# 运行 custom OFDM 端到端仿真
uv run python -m sionna_measurement_sim.app.cli run-full \
    --config config/defaults/measurement_mvp.yaml \
    --output-dir outputs/my_run

# 运行 NR PUSCH 4x4 SU-MIMO perfect CSI
uv run python -m sionna_measurement_sim.app.cli run-full \
    --config config/defaults/nr_pusch_mvp.yaml \
    --phy-standard nr_pusch \
    --output-dir outputs/nr_pusch_4x4
```

## 命令行

| 命令 | 说明 |
|------|------|
| `preflight` | 打印本地环境信息（Python/Sionna/Torch/Mitsuba/GPU） |
| `run-rt-truth` | 仅运行 RT 真值（射线追踪 + 路径提取 + HDF5） |
| `run-motion` | RT 真值 + 多快照运动/多普勒 |
| `run-observation` | RT 真值 + PHY 观测（AWGN + LS 估计 + 损伤） |
| `run-full` | **全功能端到端**：RT + 路径 + 损伤 + 观测 + 运动 + 校准 + 诊断 |
| `run-batch` | 批量实验（多个种子/SNR 自动分批） |

```bash
# 查看所有参数
uv run python -m sionna_measurement_sim.app.cli --help
uv run python -m sionna_measurement_sim.app.cli run-full --help
```

### NR PUSCH 常用参数

```bash
# 4x4 SU-MIMO perfect CSI
uv run python -m sionna_measurement_sim.app.cli run-full \
    --phy-standard nr_pusch --perfect-csi \
    --num-layers 4 --num-antenna-ports 4 \
    --mimo-detector lmmse \
    --reciprocity-applied

# 4x4 SU-MIMO estimated CSI (要求 num_layers == num_antenna_ports)
uv run python -m sionna_measurement_sim.app.cli run-full \
    --phy-standard nr_pusch \
    --num-layers 4 --num-antenna-ports 4 \
    --ebno-db 30

# MU-MIMO: 1 BS, 2 UEs, 各 2 天线
uv run python -m sionna_measurement_sim.app.cli run-full \
    --phy-standard nr_pusch --mimo-mode mu_mimo \
    --max-tx 1 --max-rx 2 \
    --num-layers 1 --num-antenna-ports 2
```

## 配置文件

所有仿真参数通过 YAML 文件控制，详见 [config/README.md](config/README.md)。

| 模板 | 用途 |
|------|------|
| `config/defaults/measurement_mvp.yaml` | 通用 custom OFDM + impairment + motion |
| `config/defaults/nr_pusch_mvp.yaml` | NR PUSCH 4x4 SU-MIMO TDD uplink |

配置包含 11 个分组：`runtime`、`input`、`output`、`carrier`、`antenna`、`rt`、`link`、`phy`、`impairments`、`receiver`、`motion`、`calibration`。

配置加载时自动进行 pydantic schema 校验，不合规会在 RT/PHY 启动前报错退出。

## HDF5 输出结构

每次运行生成 `outputs/<run_dir>/results.h5`，包含以下顶层 group：

| Group | 内容 |
|-------|------|
| `/meta` | schema 版本、运行 ID、随机种子、配置快照 |
| `/input` | 标签文件、场景文件路径 |
| `/topology` | TX/RX 三维位置 |
| `/devices` | TX/RX 速度、朝向 |
| `/antenna` | 阵列类型、极化、天线数 |
| `/scene` | 场景名称、文件、材质策略 |
| `/frequency` | 中心频率、带宽、子载波列表 |
| `/channel/truth` | 真值 CFR（`[tx, rx, rx_ant, tx_ant, subcarrier]`）、路径功率、CIR、LoS/NLoS |
| `/paths/samples` | 采样路径：顶点、交互类型、对象/primitive ID、多普勒、延迟、增益 |
| `/paths/full` | 全量路径表（`save_full_paths: true` 时） |
| `/link` | 双工模式、链路方向、互易性 |
| `/waveform` | OFDM/NR PUSCH 波形参数（standard、num_prb、num_layers、num_antenna_ports、DMRS 等） |
| `/observation` | 估计 CFR `[snap, tx, rx, rx_ant, tx_ant, subcarrier]`、SNR、CFO、SFO、相偏、定时偏、AGC、削波 |
| `/impairments` | 损伤模型版本和配置 |
| `/receiver` | 估计器类型、MIMO 检测器、同步方法、输入域 |
| `/evaluation` | NMSE、BER、BLER（TB CRC）、幅度/相位误差、检测率 |
| `/calibration` | 校准 profile 和参数 |
| `/motion` | 快照 ID、时间戳、运动模式 |
| `/runtime` | 软件版本、耗时 |

完整数据契约见 [docs/03_data_contract_hdf5.md](docs/03_data_contract_hdf5.md)。

## 开发

```bash
# 环境
uv sync                  # 安装依赖
uv run ruff check .      # 代码检查
uv run pytest            # 运行全部测试 (176 tests)

# 仅运行特定测试
uv run pytest tests/unit -k "mimo"         # MIMO 单元测试
uv run pytest tests/integration -k "mimo"  # MIMO 集成测试
uv run pytest tests/statistical            # 统计测试
```

### 项目结构

```
SionnaMeasurementSim/
  sionna_measurement_sim/
    app/              CLI 和 pipeline 编排
    config/           配置 schema (pydantic) 和 YAML 模板
    domain/           领域模型（dataclass，纯 numpy，无 Sionna 依赖）
    adapters/         Sionna RT/PHY API 适配层
      sionna_rt/      RT 场景/路径/材质适配
      sionna_phy/     PHY OFDM/信道适配
    rt/               RT 真值 pipeline
    phy/              PHY 观测 pipeline + 损伤 + NR PUSCH + MIMO 信道
    impairments/      损伤模型（re-export from phy）
    analysis/         诊断分析
    io/               HDF5 读写、schema validator、manifest、label 解析
    visualization/    拓扑图、路径图、CFR 图、NMSE 诊断图
    preflight/        环境检查
  config/defaults/    默认 YAML 配置
  data/scenes/test/   测试场景
  tests/
    unit/            单元测试（domain, config, impairments, MIMO channel/config）
    schema/          HDF5 schema 测试（truth, CIR, NR PUSCH）
    adapter/         Sionna adapter shape 测试
    integration/     RT truth, 4x4 SU-MIMO, MU-MIMO, batch, calibration
    statistical/     AWGN, impairments, motion, NR PUSCH link metrics, MIMO metrics
  docs/               设计文档
  outputs/            仿真输出（gitignore）
  old/                旧 SimpleSionna 项目（仅参考，不可 import）
```

## 文档索引

| 文档 | 内容 |
|------|------|
| [00_global_constraints](docs/00_global_constraints_and_official_references.md) | 全局约束与官方参考 |
| [02_architecture](docs/02_architecture.md) | 系统架构与分层原则 |
| [03_data_contract_hdf5](docs/03_data_contract_hdf5.md) | HDF5 数据契约 |
| [04_sionna_rt_adapter](docs/04_sionna_rt_adapter_and_path_data.md) | Sionna RT 适配与路径数据 |
| [05_phy_observation](docs/05_phy_observation_and_impairments.md) | PHY 观测与损伤 |
| [06_config_schema](docs/06_config_and_experiment_schema.md) | 配置与实验 schema |
| [08_roadmap](docs/08_roadmap_milestones_acceptance.md) | 路线图与验收标准 |
| [09_testing](docs/09_testing_and_quality_gates.md) | 测试与质量门 |
| [15_mimo_phy_gap_analysis](docs/15_mimo_phy_gap_analysis.md) | NR PUSCH MIMO 分析与修复 |
| [review](docs/review.md) | MIMO 修复复核与执行指南 |
| [phase_progress](docs/phase_progress.md) | 各阶段开发记录 |

## 约束

- Python 3.11+，`uv` 管理环境
- Sionna 2.x（RT + PHY）+ PyTorch（张量后端）
- **禁止**业务层直接 import Sionna；所有 Sionna 调用集中在 `adapters/` 和 `phy/` 层
- **禁止**新系统 import `old/` 代码
- **禁止** HDF5 writer 直接消费 Sionna 原生对象
- **禁止**将 truth CFR 写为 `/channel/cfr`（必须用 `/channel/truth/cfr`）
- **禁止**引入 TensorFlow 到核心链路
- 大型输出不入 git
