# SionnaMeasurementSim

基于 [Sionna 2.x](https://nvlabs.github.io/sionna/) 的信道测量仿真系统。生成射线追踪真值（H_true），施加 PHY 损伤和信道估计，输出仿真观测（H_obs），所有数据按 HDF5 契约落盘。

**当前能力：**
- Sionna RT 射线追踪（LoS / 反射 / 折射 / 绕射 / 漫反射）
- 多快照运动与多普勒
- Custom OFDM + AWGN + LS 估计 + 全链路 impairment（CFO/SFO/相偏/定时偏/AGC/削波）
- NR PUSCH 4x4 SU-MIMO（perfect CSI + DMRS LS 估计，LMMSE/KBest 检测器）
- NR PUSCH MU-MIMO（多 UE 联合 PUSCH，独立 DMRS port set）
- 两个可插拔信道后端：`ApplyOFDMChannel` + `CIRDataset + OFDMChannel`
- TB/CRC 语义的 BLER（transport block CRC pass/fail）
- TDD 互易性（DL RT trace → UL PUSCH）
- NR PUSCH SU-MIMO link batching（配置项 `phy.su_mimo_link_batch_size`）
- `run-full` UE/RX shard 输出（`result_000.h5` 风格，多进程不共享 HDF5 写句柄）
- 配置驱动 debug profiling（阶段耗时、GPU/CPU/RSS 采样、每 shard summary）
- HDF5 schema 强校验（含 NR PUSCH MIMO 必填字段）
- 批量实验（多 seed/SNR 自动分批）
- 220 个测试收集项（单元 / schema / adapter / 集成 / 统计）

## 快速开始

```bash
# 安装环境
uv sync

# 验证环境
uv run python -m sionna_measurement_sim.app.cli preflight

# 运行 custom OFDM 端到端仿真（6 BS × 100 UE 全场景）
uv run python -m sionna_measurement_sim.app.cli run-full \
    --config config/defaults/measurement_mvp.yaml \
    --output-dir outputs/my_run

# 运行 NR PUSCH 4x4 SU-MIMO（需使用专用配置模板）
uv run python -m sionna_measurement_sim.app.cli run-full \
    --config config/defaults/nr_pusch_mvp.yaml \
    --output-dir outputs/nr_pusch_4x4

# 查看所有参数
uv run python -m sionna_measurement_sim.app.cli run-full --help
```

项目默认锁定 PyTorch `2.10.0+cu128`，通过官方 PyTorch CUDA 12.8 wheel 源安装；在 NVIDIA driver 支持 CUDA 12.8 的机器上，`uv sync` 后即可使用 GPU。需要启用 NR PUSCH GPU 执行时，将 YAML 中 `runtime.device` 改为 `"cuda"` 或 `"cuda:0"`。

> `--config` 参数会加载 YAML 配置文件；CLI 的 `--snr-db`、`--max-tx`等参数可覆盖 YAML 中的对应值。

## 命令行

| 命令 | 说明 |
|------|------|
| `preflight` | 打印本地环境（Python / Sionna / Torch / Mitsuba / GPU） |
| `run-rt-truth` | 仅 RT 真值（射线追踪 + 路径提取 + HDF5） |
| `run-motion` | RT 真值 + 多快照运动/多普勒 |
| `run-observation` | RT 真值 + PHY 观测（AWGN + LS 估计 + 损伤） |
| `run-full` | 全功能端到端：RT + 路径 + 损伤 + 观测 + 运动 + 校准 + 诊断 |
| `run-batch` | 批量实验（多 seed/SNR 分批） |

## NR PUSCH MIMO 使用

NR PUSCH 的 MIMO 参数（天线数、层数、检测器、backend 等）通过 YAML 配置文件控制，CLI 不提供独立开关。推荐使用专用模板：

```bash
# 4x4 SU-MIMO estimated CSI（使用 nr_pusch_mvp.yaml 默认配置）
uv run python -m sionna_measurement_sim.app.cli run-full \
    --config config/defaults/nr_pusch_mvp.yaml \
    --output-dir outputs/nr_pusch_su_mimo

# 4x4 SU-MIMO perfect CSI：修改 YAML 中 phy.perfect_csi = true
# MU-MIMO：设置 phy.mimo_mode = "mu_mimo" 且 input.max_rx > 1
```

也可直接用 Python API：

```python
from pathlib import Path
from sionna_measurement_sim.rt.truth_pipeline import RTTruthRunConfig, run_rt_truth_pipeline

config = RTTruthRunConfig(
    label_file=Path("tests/fixtures/scenes/test/test5.json"),
    scene_file=Path("tests/fixtures/scenes/test/scene.xml"),
    output_dir=Path("outputs/my_run"),
    num_subcarriers=48, seed=42,
    max_tx=1, max_rx=1,
    tx_num_rows=2, tx_num_cols=2,      # BS 4 antennas
    rx_num_rows=2, rx_num_cols=2,      # UE 4 antennas
    max_depth=3, los=True, specular_reflection=True,
    observation_snr_db=40.0,
    phy_standard="nr_pusch", num_prb=4,
    num_layers=4, num_antenna_ports=4,
    perfect_csi=True,
    channel_backend="apply_ofdm",
    receiver_failure_policy="fail_fast",
)
path = run_rt_truth_pipeline(config)
```

## 配置文件

| 模板 | 用途 |
|------|------|
| `config/defaults/measurement_mvp.yaml` | 通用 custom OFDM + impairment（默认） |
| `config/defaults/nr_pusch_mvp.yaml` | NR PUSCH 4x4 SU-MIMO TDD uplink |
| `config/perf/nr_pusch_3x3000_sharded.yaml` | 3 BS × 3000 UE shard 性能回归模板 |
| `config/perf/nr_pusch_6x8884_sharded.yaml` | 6 BS × 8884 UE 4 GPU shard 验收模板 |

完整字段说明见 [config/README.md](config/README.md)。配置加载时自动进行 pydantic schema 校验。

## HDF5 输出结构

普通运行生成 `outputs/<run_dir>/results.h5`。开启 `output.sharding.enabled=true` 时，输出目录下直接生成 `result_000.h5`、`result_001.h5` 等多个自包含 HDF5 文件，并由根目录 `manifest.json` 记录全局 UE/RX 覆盖范围。

| Group | 内容 |
|-------|------|
| `/meta` | schema 版本、运行 ID、随机种子、配置快照 |
| `/shard` | shard 模式下的局部到全局 UE/RX/BS 索引映射 |
| `/input` | 标签文件、场景文件 |
| `/topology` | TX/RX 三维位置 |
| `/scene` | 场景文件、`scene_id`、`map_id` |
| `/antenna` | 阵列类型、极化、天线数 |
| `/frequency` | 中心频率、带宽、子载波 |
| `/channel/truth` | 真值 CFR `[tx, rx, rx_ant, tx_ant, subcarrier]`、CIR、LoS/NLoS |
| `/derived` | 距离、ToA/RTT-like、AoA、LoS/NLoS、link mask、TX/RX 平面几何量 |
| `/paths/samples` | 路径采样：顶点、交互、对象 ID、多普勒、延迟 |
| `/link` | 双工模式、互易性 |
| `/waveform` | OFDM/NR PUSCH 波形；NR PUSCH 额外保存频域 `tx_grid`、`rx_grid`、`noise_variance` |
| `/array` | NR PUSCH 阵列 snapshot、AoA 标签、空间谱标签 |
| `/observation` | 估计 CFR `[snap, tx, rx, rx_ant, tx_ant, subcarrier]`、SNR、CFO 等 |
| `/receiver` | 估计器类型、MIMO 检测器 |
| `/evaluation` | NMSE、BER、BLER（TB CRC）、num_block_errors/num_blocks |
| `/runtime` | 软件版本、耗时 |

完整数据契约见 [docs/sys/07_config_and_h5_format.md](docs/sys/07_config_and_h5_format.md)。

## GPU 与大规模 PUSCH

`runtime.device: "cuda"` 会让 NR PUSCH 的 PyTorch/Sionna 频域链路在 GPU 上运行。SU-MIMO PUSCH 支持把多个独立 `(snapshot, UE, BS)` link 合成 batch，配置项为 `phy.su_mimo_link_batch_size`；schema 默认保守值是 1，NR PUSCH 与性能模板使用 64。

已验证的参考规模：

| 场景 | 结果 |
|------|------|
| `3 BS × 3000 UE × 4x4 PUSCH` | shard+batch64 运行完成，3 个 HDF5 schema 通过，端到端约 178 s |
| `6 BS × 8884 UE × 4x4 PUSCH` | 4 GPU shard+batch64 运行完成，9 个 HDF5 schema 通过，端到端约 279 s |

生产级大规模 PUSCH 推荐开启 UE/RX shard：

```bash
uv run python -m sionna_measurement_sim.app.cli run-full \
    --config config/perf/nr_pusch_6x8884_sharded.yaml \
    --output-dir outputs/nr_pusch_6x8884_sharded
```

shard 模式不会把多个进程写进同一个 HDF5；每个进程写自己的 `result_xxx.h5`，根目录 `manifest.json` 汇总所有 shard 的全局索引、可视化摘要和 debug 性能日志路径。

## 开发

```bash
uv sync                  # 安装依赖（含 PyTorch CUDA 12.8 wheel）
uv run ruff check .      # 代码检查
uv run pytest            # 运行全部测试

# 按类别运行
uv run pytest tests/unit -q                        # 单元测试
uv run pytest tests/integration -k "mimo" -q      # MIMO 集成测试
uv run pytest tests/statistical -q                 # 统计测试
```

### 项目结构

```
SionnaMeasurementSim/
  sionna_measurement_sim/
    app/              CLI 和 pipeline
    config/           配置 schema (pydantic)
    domain/           领域模型（dataclass，纯 numpy）
    adapters/sionna_rt/  Sionna RT API 适配
    rt/               RT 真值 pipeline
    phy/              PHY 观测 + 损伤 + NR PUSCH + MIMO 信道 + backend
    io/               HDF5 读写、schema validator、manifest、label 解析
    analysis/         诊断分析
    visualization/    拓扑/路径/CFR/NMSE/空间谱图
  config/defaults/    默认 YAML 配置
  data/               本地数据占位目录（真实场景/输出不进 git）
  tests/fixtures/scenes/test/   测试场景
  tests/
    unit/ domain / config / impairments / MIMO channel / MIMO config
    schema/           HDF5 truth / CIR / NR PUSCH
    adapter/          Sionna adapter shape
    integration/      RT truth / 4x4 SU-MIMO / MU-MIMO / batch / calibration
    statistical/      AWGN / impairments / motion / NR PUSCH MIMO metrics
  docs/               设计文档
  outputs/            仿真输出（gitignore，可为本地 symlink）
```

## 文档索引

| 文档 | 内容 |
|------|------|
| [00_project_overview](docs/sys/00_project_overview.md) | 项目总览 |
| [01_app_and_config](docs/sys/01_app_and_config.md) | CLI、配置加载和批量实验 |
| [02_domain_models](docs/sys/02_domain_models.md) | 领域模型 |
| [03_adapters](docs/sys/03_adapters.md) | Sionna RT 适配 |
| [04_rt_pipeline](docs/sys/04_rt_pipeline.md) | RT pipeline |
| [05_phy_observation](docs/sys/05_phy_observation.md) | PHY 观测与 NR PUSCH |
| [06_io_and_testing](docs/sys/06_io_and_testing.md) | HDF5 I/O、schema 和测试 |
| [07_config_and_h5_format](docs/sys/07_config_and_h5_format.md) | 配置与 HDF5 数据契约 |

## 约束

- Python 3.11+，`uv` 管理环境
- Sionna 2.x（RT + PHY）+ PyTorch（张量后端）
- 禁止业务层直接 import Sionna
- 禁止将 truth CFR 写为 `/channel/cfr`（必须 `/channel/truth/cfr`）
- 禁止引入 TensorFlow 到核心链路
- 大型输出不入 git
