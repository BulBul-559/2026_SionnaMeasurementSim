# SionnaMeasurementSim

基于 [Sionna 2.x RT](https://nvlabs.github.io/sionna/) 的信道测量仿真系统。生成射线追踪真值（H_true），施加 PHY 损伤和信道估计，输出仿真观测（H_obs），所有数据按 HDF5 契约落盘。

## 快速开始

```bash
# 安装环境
uv sync

# 运行全量端到端仿真（使用默认配置）
uv run python -m sionna_measurement_sim.app.cli run-full \
    --config config/defaults/measurement_mvp.yaml \
    --output-dir outputs/my_run

# 使用命令行参数覆盖配置
uv run python -m sionna_measurement_sim.app.cli run-full \
    --output-dir outputs/quick_test \
    --num-subcarriers 64 --snr-db 30 \
    --max-tx 6 --max-rx 100
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

## 配置文件

所有仿真参数通过 YAML 文件控制，详见 [config/README.md](config/README.md)。

```bash
# 默认配置模板
config/defaults/measurement_mvp.yaml
```

配置包含 11 个分组：`runtime`（运行环境）、`input`（输入数据）、`output`（输出控制）、`carrier`（载波频率）、`antenna`（天线阵列）、`rt`（射线追踪）、`phy`（物理层）、`impairments`（损伤模型）、`receiver`（接收机）、`motion`（运动多普勒）、`calibration`（校准）。

配置加载时自动进行 pydantic schema 校验，不合规会在 RT/PHY 启动前报错退出。

## HDF5 输出结构

每次运行生成 `outputs/<run_dir>/results.h5`，包含以下顶层 group：

| Group | 内容 |
|-------|------|
| `/meta` | schema 版本、运行 ID、随机种子、配置快照 |
| `/input` | 标签文件、场景文件路径 |
| `/topology` | TX/RX 三维位置 |
| `/devices` | TX/RX 速度、朝向 |
| `/antenna` | 阵列类型、极化 |
| `/frequency` | 中心频率、带宽、子载波列表 |
| `/channel/truth` | 真值 CFR、路径功率、几何信号标记、路径计数、LoS/NLoS 存在性 |
| `/paths/samples` | 采样路径：顶点坐标、交互类型、对象 ID、多普勒、延迟、增益 |
| `/waveform` | OFDM 波形参数、导频配置 |
| `/observation` | 估计 CFR、SNR、CFO、SFO、相偏、定时偏、AGC、削波标记 |
| `/impairments` | 损伤模型版本和配置 |
| `/receiver` | 估计器类型、同步方法 |
| `/evaluation` | NMSE、幅度/相位误差、检测率、失败率 |
| `/calibration` | 校准 profile 和参数 |
| `/motion` | 快照 ID、时间戳、运动模式 |
| `/runtime` | 软件版本、耗时 |

完整数据契约见 [docs/03_data_contract_hdf5.md](docs/03_data_contract_hdf5.md)。

## 开发

```bash
# 环境
uv sync                  # 安装依赖
uv run ruff check .      # 代码检查
uv run pytest            # 运行全部测试

# 仅运行特定测试
uv run pytest tests/unit tests/statistical -k "impairment"
uv run pytest tests/integration -k "batch"
```

### 项目结构

```
SionnaMeasurementSim/
  sionna_measurement_sim/
    app/              CLI 和 pipeline 编排
    config/           配置 schema (pydantic) 和 YAML 模板
    domain/           领域模型（dataclass，纯 numpy，无 Sionna 依赖）
    adapters/sionna_rt/  Sionna RT API 适配层
    rt/               RT 真值 pipeline
    phy/              PHY 观测 pipeline + 损伤模型
    impairments/      损伤模型命名空间（re-export from phy）
    analysis/         诊断分析
    io/               HDF5 读写、manifest、label 解析
    visualization/    拓扑图、路径图、CFR 图、NMSE 诊断图
    preflight/        环境检查
  config/defaults/    默认 YAML 配置
  data/scenes/test/   测试场景
  tests/              测试（unit/schema/adapter/integration/statistical）
  docs/               设计文档
  outputs/            仿真输出（gitignore）
  old/                旧 SimpleSionna 项目（仅参考，不可 import）
```

## 文档索引

| 文档 | 内容 |
|------|------|
| [00_global_constraints](docs/00_global_constraints_and_official_references.md) | 全局约束与官方参考 |
| [03_data_contract_hdf5](docs/03_data_contract_hdf5.md) | HDF5 数据契约 |
| [04_sionna_rt_adapter](docs/04_sionna_rt_adapter_and_path_data.md) | Sionna RT 适配与路径数据 |
| [05_phy_observation](docs/05_phy_observation_and_impairments.md) | PHY 观测与损伤 |
| [06_config_schema](docs/06_config_and_experiment_schema.md) | 配置与实验 schema |
| [08_roadmap](docs/08_roadmap_milestones_acceptance.md) | 路线图与验收标准 |
| [09_testing](docs/09_testing_and_quality_gates.md) | 测试与质量门 |
| [12_final_checklist](docs/12_final_acceptance_checklist.md) | 最终验收清单 |

## 约束

- Python 3.11+，`uv` 管理环境
- Sionna 2.x（RT）+ PyTorch（PHY 张量）
- 禁止业务层直接 import Sionna；所有 Sionna 调用集中在 `adapters/sionna_rt/`
- 禁止新系统 import `old/` 代码
- 禁止 HDF5 writer 直接消费 Sionna 原生对象
- 禁止将 truth CFR 写为 `/channel/cfr`（必须用 `/channel/truth/cfr`）
- 大型输出不入 git
