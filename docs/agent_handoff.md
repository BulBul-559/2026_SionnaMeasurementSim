# Agent Handoff

这份文档用于让新的 Codex/agent 快速理解当前项目状态。新对话开始时，建议先读：

- `docs/agent_handoff.md`
- `docs/sys/README.md`
- `README.md`
- `config/README.md`
- `docs/sys/07_config_and_h5_format.md`
- 当前任务相关的 `docs/performance/*.md`

除非用户明确要求，不要递归扫描 `data/` 和 `outputs/`，它们是本地大数据路径，可能是 symlink。
`docs/sys/` 是当前系统设计和接口说明的主参考；`docs/performance/` 是历史实验记录，
其中的参数反映当时实验事实，不一定代表当前默认配置。

## 项目定位

SionnaMeasurementSim 是一个基于 Sionna RT 的室内无线仿真数据生成系统。当前重点是：

- 从场景/label 生成 RT truth、CIR/CFR、路径真值、AoA、NLoS path truth。
- 生成 PHY 观测数据，包括 NR PUSCH-DMRS CSI proxy 和 NR SRS-like full-band uplink sounding。
- 支持频域 waveform grid、array/空间谱、HDF5 多文件 shard 输出和 manifest 汇总。
- 为后续定位、场重建、CSI/embedding 学习生成可复现数据。

当前工作分支为 `codex/common-phy-link-impairments`。本分支已把 NR PUSCH 与
NR SRS-like 的 clean channel、基带损伤、AWGN 和 observation metadata 统一到
`sionna_measurement_sim/phy/common_link.py`；`custom_ofdm` 保留为 legacy 路径。
当前 HDF5 schema 版本是 `1.1.0`。

## 深读文档地图

| 目标 | 建议阅读 |
|---|---|
| 快速了解系统分层 | `docs/sys/00_project_overview.md` |
| 查 CLI/config/shard/debug 配置 | `docs/sys/01_app_and_config.md`, `config/README.md` |
| 查 pipeline 编排和 BS/UE→TX/RX 映射 | `docs/sys/04_rt_pipeline.md` |
| 查 PUSCH/SRS/custom OFDM 实现口径 | `docs/sys/05_phy_observation.md` |
| 查 HDF5 字段、shape、manifest | `docs/sys/07_config_and_h5_format.md` |
| 查 SRS baseline 和 shard size 依据 | `docs/sys/indoor_fr1_100mhz_validation.md` |
| 新增 PHY module | `docs/sys/phy_module_development.md` |
| 标准 NR SRS 后续工作 | `docs/sys/nr_srs_standard_todo.md` |

## 核心语义

配置和 label 层使用物理角色：

- `BS`
- `UE`

仿真和 HDF5 落盘层使用链路视角：

- `TX`
- `RX`

映射只由 `link.phy_link_direction` 决定：

| `phy_link_direction` | TX | RX |
|---|---|---|
| `uplink` | UE | BS |
| `downlink` | BS | UE |

HDF5 里保留 `/link/tx_role` 和 `/link/rx_role`。当前 SRS 生产口径使用真实 direct uplink，因此 `tx_role="ue"`、`rx_role="bs"`。

`/channel/truth/cfr` 的维度语义是：

```text
[tx, rx, rx_ant, tx_ant, subcarrier]
```

在 uplink SRS 中就是：

```text
[ue, bs, bs_ant, ue_ant, subcarrier]
```

## 当前 PHY 模块

PHY 已经模块化，新增链路应走 registry/module 方式。

当前模块：

- `custom_ofdm`
- `nr_pusch`
- `nr_srs`

`nr_pusch` 和 `nr_srs` 共享通用链路：

```text
tx_grid -> clean channel apply -> rx_grid_clean
         -> CFO/SFO/phase/timing/AGC/ADC + AWGN
         -> rx_grid -> standard receiver/estimator
```

common chain 统一写 `/waveform/tx_grid`、`/waveform/rx_grid`、
`/waveform/noise_variance` 以及 `/observation/cfo_hz` 等损伤观测字段。
SRS-like 另写 `/waveform/pilot_code`。schema `1.1.0` 后不再写
`/waveform/srs_*` 或 `/observation/srs_cfr_est`。

`nr_srs` 当前是 SRS-like full-band sounding，不是完整 3GPP NR SRS。它使用全带宽已知 pilot，经过信道后做 LS：

```text
H_hat = Y / X
```

标准 SRS 的 comb、sequence、cyclic shift、hopping 等还在 TODO 中，见 `docs/sys/nr_srs_standard_todo.md`。

## 数据目录

`data/` 与 `outputs/` 都是 ignored 本地路径，可以是 symlink。

当前本地 `data/` 下主要有：

```text
data/dense/
data/median/
data/sparse/
```

每个场景目录已统一命名，例如：

```text
data/median/median_0000/
```

每个场景有三种 label：

| 文件 | UE 采样间隔 | 当前 UE 数 |
|---|---:|---:|
| `label0p1.json` | 0.1 m | 10360 |
| `label0p2.json` | 0.2 m | 2583 |
| `label0p4.json` | 0.4 m | 654 |

`label0p2.json` 是当前更适合作为 baseline 的默认规模。`label0p1.json` 成本很高，先不要默认跑全量。

## 当前推荐 SRS 配置

默认模板：

```text
config/defaults/nr_srs_indoor_positioning_fr1_100mhz.yaml
```

当前推荐口径：

- `runtime.device: "cuda"`
- `link.phy_link_direction: "uplink"`
- `phy.standard: "nr_srs"`
- `rt.synthetic_array: false`
- `rt.max_depth: 4`
- `rt.los: true`
- `rt.specular_reflection: true`
- `rt.diffuse_reflection: false`
- `rt.refraction: true`
- `rt.diffraction: true`
- `array.spectrum.enabled: false`
- `visualization.enabled: false`
- `output.sharding.enabled: true`
- `output.sharding.axis: "ue"`
- `output.sharding.shard_size: 20`

模板里的 `input.label_file`、`input.scene_file`、`scene_id` 和 `output.root_dir`
需要按目标场景复制后修改。推荐把临时运行配置放在 ignored `outputs/local_configs/` 下。

`shard_size=20` 是当前生产建议。历史报告里出现的 `25` 是旧实验记录，不再作为默认生产值。

## 最近重要验证

### `median_0000 label0p2` SRS baseline

输出目录：

```text
outputs/nr_srs_median_0000_label0p2_full_baseline_shard20
```

实际路径通常是：

```text
/data/sunmeiyuan/projects/sionna/outputs/nr_srs_median_0000_label0p2_full_baseline_shard20
```

结果：

| 项 | 值 |
|---|---:|
| 场景 | `median_0000` |
| label | `label0p2.json` |
| BS / UE | 7 / 2583 |
| shard size | 20 |
| shard count | 130 |
| GPU | `[5, 6, 7]` |
| wall time | 1274.72 s, 约 21.2 min |
| output size | 约 52 GB |

检查结论：

- `results/result_000.h5` 到 `results/result_129.h5` 连续存在。
- UE 覆盖 `0..2582`，无缺失、无重复。
- BS 覆盖 `[0,1,2,3,4,5,6]`。
- `/link/tx_role = "ue"`。
- `/link/rx_role = "bs"`。
- 没有 `/waveform/tx_time` 或 `/waveform/rx_time`。
- 使用统一 waveform 字段 `/waveform/tx_grid`、`/waveform/rx_grid`、
  `/waveform/noise_variance` 和 `/waveform/pilot_code`。
- `manifest/manifest.json` 已生成。

### `dense_0001 label0p4` PUSCH common-AWGN rerun

输出目录：

```text
outputs/dense_0001_label0p4_pusch_fixed_snr_shard10
```

用途：验证 common link 接入后 PUSCH estimated CSI 的噪声口径。早先同场景结果中
PUSCH 把 `snr_db` 误当作绝对 noise variance，导致 effective SNR 约 `-19 dB`、
BER 接近随机；修正后 normal `snr_db` 由 common chain 按 clean `rx_grid` 功率计算。

结果：

| 项 | 值 |
|---|---:|
| 场景 | `dense_0001` |
| label | `label0p4.json` |
| UE | 654 |
| planned shard / result file | 66 / 101 |
| fallback split | 35 |
| GPU | `[0,1,2,3,5,6,7]` |
| wall time | 642.35 s |
| schema | 101 / 101 passed |
| effective SNR | median/mean 30.0 dB |
| NMSE | mean -30.283 dB, median -30.300 dB |
| global BER / BLER | 0.00293 / 0.03298 |

可视化在：

```text
outputs/dense_0001_label0p4_pusch_fixed_snr_shard10/figures/result_000
```

这批结果说明 PUSCH 链路当前是通的：`perfect_csi=false` 时 receiver 使用
PUSCHReceiver 内部 DMRS LS；导出的 `/observation/cfr_est` 来自外部
`PUSCHLSChannelEstimator` 对同一个 impaired `rx_grid` 的估计。`perfect_csi=true`
才把 clean backend 返回的 oracle `h` 传给 receiver。

### `shard_size=25` 的问题

同一 `median_0000 label0p2` 用 `shard_size=25` 在后段 shard 失败：

```text
paths.cfr()
Dr.Jit single-array entry count > 2^32
```

这不是普通显存 OOM，而是底层 Dr.Jit 单数组 entry 数限制。当前默认模板已改为 `shard_size=20`。

失败的 partial 输出曾位于：

```text
outputs/nr_srs_median_0000_label0p2_full_baseline
```

它没有完整 manifest，不能作为 baseline 使用。

## 常用命令

查看配置：

```bash
uv run python -m sionna_measurement_sim.app.cli --config config/defaults/nr_srs_indoor_positioning_fr1_100mhz.yaml run-full --help
```

运行 SRS 全流程时，建议复制模板到 `outputs/local_configs/`，再改 label/output/gpu，避免把本地绝对路径提交进 git。

基础检查：

```bash
uv run ruff check .
uv run pytest
```

只检查配置加载：

```bash
uv run pytest tests/unit/test_config_loader.py -q
```

查看 GPU：

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits
```

## 重要坑点

- 不要默认跑 `label0p1.json` 全量，它是 10360 UE，成本约为 `label0p2` 的 4 倍。
- `data/` 和 `outputs/` 不进 git，里面的真实数据和仿真输出都应视为本地大文件。
- 历史性能报告中的 `shard_size=25` 是历史记录；当前 SRS 生产模板推荐 `20`。
- direct uplink 下 UE 是 source，UE block 大小比 BS 数更容易触发 RT/Dr.Jit 限制。
- 多文件 shard 是当前推荐输出方式，不建议为了训练强行合并成单个巨大 HDF5。
- 空间谱和 visualization 默认关闭；它们适合小样本诊断，不适合默认全量生产。
- HDF5 下游读取应通过 `manifest/manifest.json` 和其中记录的 `results/result_xxx.h5` shard 列表，不要假设只有 `results.h5`，也不要假设文件名连续；fallback 可能产生 `result_089_00.h5` 这类子 shard。
- `nr_srs` 是 SRS-like，不要在论文或文档里称为 standards-complete 3GPP SRS。

## 新任务建议流程

1. 先确认分支和工作区：

   ```bash
   git branch --show-current
   git status --short
   ```

2. 读当前任务相关文档，避免重复扫描大目录。
3. 如果需要仿真，先确认 label 粒度、BS/UE 数量、shard size、GPU 空闲和输出目录。
4. 大规模仿真先跑小 shard smoke，再跑全量。
5. 完成配置或代码修改后，至少跑：

   ```bash
   git diff --check
   uv run pytest tests/unit/test_config_loader.py -q
   ```

6. 如果涉及代码逻辑，最终跑：

   ```bash
   uv run ruff check .
   uv run pytest
   ```
