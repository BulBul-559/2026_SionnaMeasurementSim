# 00. 项目总览

## 项目定位

**SionnaMeasurementSim** 是一个基于 [Sionna 2.x](https://nvlabs.github.io/sionna/) 的信道测量仿真系统。它的核心任务是：

1. 使用 Sionna RT（Ray Tracing）在三维场景中生成信道真值（`/channel/truth/cfr`）
2. 在真值之上施加 PHY 层 OFDM 调制、AWGN、各类损伤、信道估计，生成仿真观测值（`/observation/cfr_est`）
3. 计算真值与观测值之间的评估指标（NMSE、BER、BLER 等）
4. 将所有数据按 HDF5 契约落盘，供后续分析使用
5. 输出 `scene_id` 与 `/derived` 物理派生标签，便于和外部平面图/地图系统按场景对齐

项目支持两种 PHY 标准：
- **custom_ofdm**：简化 OFDM + LS 估计 + 完整 impairment 链
- **nr_pusch**：5G NR PUSCH 上行链路，支持 4x4 SU-MIMO 和 MU-MIMO

## 顶层架构

```
┌──────────────────────────────────────────────────────────┐
│                    app (CLI + Pipeline)                   │
├──────────────────────────────────────────────────────────┤
│  config (Pydantic Schema)  │  domain (Dataclasses)       │
├──────────────────────────────────────────────────────────┤
│  rt/truth_pipeline     │  phy/observation + NR PUSCH     │
├──────────────────────────────────────────────────────────┤
│  adapters/sionna_rt/   │  io/ (HDF5 Writer/Reader)      │
├──────────────────────────────────────────────────────────┤
│              Sionna 2.x (RT + PHY) + PyTorch             │
└──────────────────────────────────────────────────────────┘
```

**分层原则**（详见本目录的分层文档）：
- **app** 层只做编排，不直接访问 Sionna API
- **domain** 层定义纯 Python 数据模型，零 Sionna 依赖
- **adapters** 层封装所有 Sionna 调用，输出 domain 对象
- **phy** 层实现 PHY 链路（OFDM、PUSCH、信道估计、MIMO 检测）
- **io** 层负责 HDF5 读写和 schema 校验

## 数据流

```
Label JSON + Scene XML + Config YAML
        │
        ▼
  ┌─────────────┐
  │ Label Parser │ ──→ Topology (TX/RX positions)
  └─────────────┘
        │
        ▼
  ┌──────────────────┐
  │ Sionna RT Solver │ ──→ RTTruthResult (H_true CFR)
  │  + Path Adapter  │ ──→ PathSamples, CIRTruth
  └──────────────────┘
        │
        ▼
  ┌─────────────────────────┐
  │ PHY Observation Pipeline │
  │  custom_ofdm: AWGN + LS  │
  │  nr_pusch: PUSCH + MIMO  │──→ ObservationResult (H_obs CFR)
  └─────────────────────────┘      EvaluationResult (NMSE/BER/BLER)
        │
        ▼
  ┌──────────────┐
  │ HDF5 Writer  │ ──→ results.h5
  │ + Validator  │
  └──────────────┘
```

## 代码目录与文档映射

| 代码目录 | 文档 |
|----------|------|
| `app/` | [01_app_and_config.md](01_app_and_config.md) |
| `domain/` | [02_domain_models.md](02_domain_models.md) |
| `adapters/sionna_rt/` | [03_adapters.md](03_adapters.md) |
| `rt/` | [04_rt_pipeline.md](04_rt_pipeline.md) |
| `phy/` | [05_phy_observation.md](05_phy_observation.md) |
| `io/` | [06_io_and_testing.md](06_io_and_testing.md) |
| `config/` | [07_config_and_h5_format.md](07_config_and_h5_format.md) |
| `tests/` | [06_io_and_testing.md](06_io_and_testing.md) |
| HDF5 数据格式 | [07_config_and_h5_format.md](07_config_and_h5_format.md) |

## 关键维度约定

项目内部所有张量采用 **TX-first** 维度顺序（与 Sionna 原生 rx-first 不同，adapter 负责转换）：

```
Truth CFR:     [tx, rx, rx_ant, tx_ant, subcarrier]
Obs CFR:       [snapshot, tx, rx, rx_ant, tx_ant, subcarrier]
CIR:           [snapshot, tx, rx, rx_ant, tx_ant, path]
Path scalars:  [tx, rx, rx_ant, tx_ant, path]
Derived:       [tx, rx]
```

完整约定见 [07_config_and_h5_format.md](07_config_and_h5_format.md)。

## 核心约束

- Python 3.11+，`uv` 管理依赖
- Sionna 2.x（RT + PHY）+ PyTorch 张量后端
- 禁止在 `domain/` 和 `io/` 中 import Sionna
- 禁止将 truth CFR 写为 `/channel/cfr`（必须 `/channel/truth/cfr`）
- 禁止引入 TensorFlow
- 大型输出不入 git

## 当前能力矩阵

| 能力 | 状态 |
|------|------|
| RT 射线追踪 (LoS / 反射 / 折射 / 绕射) | ✅ |
| 多快照运动与多普勒 | ✅ |
| Custom OFDM + AWGN + LS 估计 | ✅ |
| 全链路 impairment (CFO/SFO/相偏/定时偏/AGC/削波) | ✅ |
| NR PUSCH 4x4 SU-MIMO perfect CSI | ✅ |
| NR PUSCH 4x4 SU-MIMO estimated CSI (需 num_layers == num_antenna_ports) | ✅ |
| NR PUSCH MU-MIMO (多 UE 联合 PUSCH) | ✅ |
| LMMSE / KBest MIMO 检测器 | ✅ |
| `ApplyOFDMChannel` channel backend | ✅ |
| `CIRDataset + OFDMChannel` channel backend (per-link, shared delay median) | ✅ |
| TB/CRC 语义 BLER | ✅ |
| TDD 互易性 | ✅ |
| `scene_id` / `map_id` 对齐字段 | ✅ |
| `/derived` 距离、ToA/RTT-like、AoA、LoS/NLoS 标签 | ✅ |
| NR PUSCH 频域 tx/rx grid 与 `/array` 标签 | ✅ |
| HDF5 schema 强校验 | ✅ |
| 批量实验 | ✅ |
| 测试覆盖 (190 collected / 188 passed / 2 skipped) | ✅ |

## 外部参考

- [Sionna 官方文档](https://nvlabs.github.io/sionna/)
- [Sionna RT API](https://nvlabs.github.io/sionna/rt/api/paths.html)
- [Sionna 5G NR PUSCH Tutorial](https://nvlabs.github.io/sionna/phy/tutorials/notebooks/5G_NR_PUSCH.html)
- [Sionna RT Link-Level Tutorial](https://nvlabs.github.io/sionna/phy/tutorials/notebooks/Link_Level_Simulations_with_RT.html)
- [uv 官方文档](https://docs.astral.sh/uv/)
