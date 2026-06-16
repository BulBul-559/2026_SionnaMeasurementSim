# SionnaMeasurementSim

基于 [Sionna 2.x](https://nvlabs.github.io/sionna/) 的信道测量仿真系统。生成射线追踪真值（H_true），施加 PHY 损伤和信道估计，输出仿真观测（H_obs），所有数据按 HDF5 契约落盘。

**当前能力：**
- Sionna RT 射线追踪（LoS / 反射 / 折射 / 绕射 / 漫反射）
- 多快照运动与多普勒
- Custom OFDM + AWGN + LS 估计（legacy 路径，保留测试/兼容用途）
- SRS/PUSCH 共享的通用 clean channel → impairment/AWGN → receiver 链路
- NR PUSCH 4x4 SU-MIMO（perfect CSI + DMRS LS 估计，LMMSE/KBest 检测器）
- NR PUSCH MU-MIMO（多 UE 联合 PUSCH，独立 DMRS port set）
- NR SRS standards-shaped v2 subset（comb/BWP、NR-ZC-like sequence、group/sequence hopping、cyclic-shift port multiplexing、frequency/bandwidth hopping、port/antenna switching、resource LS + full-band interpolation；可选 multi-UE 正交 SRS shared observation；暂非完整 3GPP SRS）
- 协议无关 uplink power/RSSI 模型：`tx_power_dbm` 生效到 SRS/PUSCH TX grid，支持 relative-SNR 与 absolute-thermal 噪声口径
- 协议无关 IQ observation 层：可选保存合作式 per-link clean/observed 频域 IQ、时域 IQ；支持 `iq_link_library` compact profile 只保存可在线混合的 per-link clean IQ；并支持非合作 multi-UE shared time-domain IQ 输出
- 波形级 ranging observation：从 `/observation/cfr_est` 估计 ToA/one-way range，支持 PDP peak 与 phase-slope estimator
- PHY module registry：`custom_ofdm`、`nr_pusch`、`nr_srs` 通过统一接口接入 pipeline
- 两个可插拔信道后端：`ApplyOFDMChannel` + `CIRDataset + OFDMChannel`
- TB/CRC 语义的 BLER（transport block CRC pass/fail）
- BS/UE role-view 配置到 TX/RX link-view 仿真的显式映射
- NR PUSCH SU-MIMO link batching（配置项 `phy.su_mimo_link_batch_size`）
- `run-full` UE shard 输出（`result_000.h5` 风格，多进程不共享 HDF5 写句柄）
- 配置驱动 debug profiling（阶段耗时、GPU/CPU/RSS 采样、HDF5 dataset 写入聚合、失败运行 summary、每 shard summary）
- RSS radio map 可视化：按 BS 聚合 `/observation/rssi_dbm`，在 floorplan 上输出插值或原始采样点热力图
- `benchmark rt/write/spectrum` 性能工程入口，用于隔离 RT solve、HDF5 writer/schema validate 和 Bartlett 空间谱成本
- `output.profile` 只保留三种真实输出契约：`full`、`rt_labels_only`、`iq_link_library`；`full` 可通过 `output.products` 选择关键产物并裁剪 RT/PHY/下游链路
- HDF5 schema `2.3.0` 强校验（full contract、RT labels-only contract、clean IQ link-library contract、product-aware full contract、NR SRS v2 resource/port/power datasets、multi-UE SRS `/multiuser` 输出、协议无关 `/iq` 输出、统一 waveform/power 字段、array 旧别名移除、ranging 与 truth range 语义拆开）
- 批量实验（多 seed/SNR 自动分批）
- 测试套件覆盖单元 / schema / adapter / 集成 / 统计；最近全量结果以本地 `uv run pytest -q` 为准

## 快速开始

```bash
# 安装环境
uv sync

# 验证环境
uv run python -m sionna_measurement_sim.app.cli preflight

# 运行 custom OFDM 端到端仿真（6 BS × 100 UE 全场景）
uv run python -m sionna_measurement_sim.app.cli \
    --config config/defaults/measurement_mvp.yaml \
    run-full \
    --output-dir outputs/my_run

# 运行 NR PUSCH 4x4 SU-MIMO（需使用专用配置模板）
uv run python -m sionna_measurement_sim.app.cli \
    --config config/defaults/nr_pusch_mvp.yaml \
    run-full \
    --output-dir outputs/nr_pusch_4x4

# 运行室内 FR1 100 MHz NR SRS subset uplink sounding
uv run python -m sionna_measurement_sim.app.cli \
    --config config/defaults/nr_srs_indoor_positioning_fr1_100mhz.yaml \
    run-full \
    --output-dir outputs/bistro_0000_nr_srs

# 只生成紧凑 RT link-level 标签，不写 CFR/CIR/path samples/PHY
uv run python -m sionna_measurement_sim.app.cli \
    --config config/defaults/rt_labels_only.yaml \
    run-full \
    --output-dir outputs/my_rt_labels

# 只生成可在线混合的 clean per-link IQ 库，不写 CFR estimate/损伤/空间谱
# 模板默认保存 time IQ；可用 phy.iq.clean_output 切换为 time/frequency/both
uv run python -m sionna_measurement_sim.app.cli \
    --config config/defaults/iq_link_library_nr_srs.yaml \
    run-full \
    --output-dir outputs/my_iq_link_library

# 只保存 CFR truth 和必要元数据，跳过 CIR/path samples/PHY/ranging/空间谱
uv run python -m sionna_measurement_sim.app.cli \
    --config config/defaults/cfr_truth_only.yaml \
    run-full \
    --output-dir outputs/my_cfr_truth_only

# 查看所有参数
uv run python -m sionna_measurement_sim.app.cli run-full --help

# 运行最小 write-only benchmark（输出 JSON/CSV 到 ignored outputs/）
uv run python -m sionna_measurement_sim.app.cli benchmark write \
    --output-dir outputs/benchmark_write_smoke \
    --tx-count 1 --rx-count 2 --rx-ant 2 --subcarriers 16
```

项目默认锁定 PyTorch `2.10.0+cu128`，通过官方 PyTorch CUDA 12.8 wheel 源安装；在 NVIDIA driver 支持 CUDA 12.8 的机器上，`uv sync` 后即可使用 GPU。需要启用 NR PUSCH GPU 执行时，将 YAML 中 `runtime.device` 改为 `"cuda"` 或 `"cuda:0"`。

> `--config` 参数会加载 YAML 配置文件；CLI 的 `--snr-db`、`--max-bs`、`--max-ue` 等参数可覆盖 YAML 中的对应值。

## 输入数据约定

`input.label_file` 使用标准 label `0.1.0` JSON。pipeline 读取顶层
`bs_points` 和 `ue_points` 作为全场景 BS/UE 点集；`groups` 只作为房间、区域或生成策略
元数据保留，当前不会默认选第一个 group，也不需要配置 `label_group_policy`。点坐标可以
写成 `position: [x, y, z]` 或显式 `x/y/z`，单位为米。可选 floorplan 使用
`floorplan_<height>.png` 命名，例如 `floorplan_1p60.png` 表示 `z=1.60 m` 截断高度，
像素/真实尺寸转换由 `floorplan/meta.json` 提供。

## 命令行

| 命令 | 说明 |
|------|------|
| `preflight` | 打印本地环境（Python / Sionna / Torch / Mitsuba / GPU） |
| `run-rt-truth` | 仅 RT 真值（射线追踪 + 路径提取 + HDF5） |
| `run-motion` | RT 真值 + 多快照运动/多普勒 |
| `run-observation` | RT 真值 + PHY 观测（AWGN + LS 估计 + 损伤） |
| `run-full` | 全功能端到端：RT + 路径 + 损伤 + 观测 + 运动 + 校准 + 诊断 |
| `run-batch` | 批量实验（多 seed/SNR 分批） |
| `benchmark rt` | 仅测 RT solve，不跑 PHY/HDF5/可视化 |
| `benchmark write` | 合成 `MeasurementSimulationResult`，仅测 HDF5 writer、compression（含 `mixed` 策略）和 schema validate |
| `benchmark spectrum` | 合成 array samples，仅测 Bartlett 空间谱核心 |

## NR PUSCH MIMO 使用

NR PUSCH 的 MIMO 参数（天线数、层数、检测器、backend 等）通过 YAML 配置文件控制，CLI 不提供独立开关。推荐使用专用模板：

```bash
# 4x4 SU-MIMO estimated CSI（使用 nr_pusch_mvp.yaml 默认配置）
uv run python -m sionna_measurement_sim.app.cli \
    --config config/defaults/nr_pusch_mvp.yaml \
    run-full \
    --output-dir outputs/nr_pusch_su_mimo

# 4x4 SU-MIMO perfect CSI：修改 YAML 中 phy.perfect_csi = true
# MU-MIMO：设置 phy.mimo_mode = "mu_mimo" 且 input.max_ue > 1
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
    max_bs=1, max_ue=1,
    bs_num_rows=2, bs_num_cols=2,      # BS 4 antennas
    ue_num_rows=2, ue_num_cols=2,      # UE 4 antennas
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

## NR SRS Subset Sounding

`phy.standard: "nr_srs"` 当前实现 standards-shaped NR SRS v2 subset：按 `phy.srs`
生成 comb/BWP/hopping resource plan，在完整 14-symbol slot 中只填 SRS symbols，支持
`zc_like` 与 deterministic `nr_zc` sequence、group/sequence hopping、同 symbol
cyclic-shift port multiplexing、antenna switching 口径和通用 uplink power/RSSI 标定。
receiver 先在 flattened SRS RE 上做 despread/LS，再插值到 full-band
`/observation/cfr_est`。它和 PUSCH 一样走通用 clean channel →
impairment/AWGN 链路，写出统一 waveform 字段 `/waveform/tx_grid`、
`/waveform/rx_grid`、`/waveform/noise_variance`、通用
`/waveform/tx_power_dbm_per_port` 和 `/waveform/tx_power_scale_linear`，以及 SRS 专属
`/waveform/srs_resource_mask`、`/waveform/srs_pilot_symbols`、
`/waveform/srs_re_symbol_indices`、`/waveform/srs_re_subcarrier_indices`、
`/waveform/srs_port_tx_ant_map` 和 SRS power metadata。schema `1.5.0`
后不再写 `/waveform/pilot_code` 或 `/waveform/srs_port_index`；schema `1.6.0`
后 `tx_power_dbm` 生效到 SRS/PUSCH 发射 grid，并写入通用 power/RSSI 字段。

可选 `phy.srs.multiuser.enabled=true` 会生成一个静态 snapshot 的 multi-UE SRS
shared observation。当前 v1 只建模“资源完全正交、理想 OFDM 下 UE 间不强烈互扰”的
理想情况：多个 UE 按 sequential frame 分组，在同一 OFDM symbol 上使用互不重叠的
`comb_offset` 或 PRB 段发射，先在 BS 侧叠加为 `/multiuser/rx_grid_shared`，再从各自
resource 中提取 `cfr_est_resource` 和 `cfr_est_allocated`。其中实际 SRS RE 上的
`cfr_est_resource` 是最权威观测；`comb_offset` 场景若补 full allocated band，未观测
子载波仍来自插值/补全，不应当成真实全频观测。

可选 `phy.iq.enabled=true` 会额外写协议无关 `/iq/link`：从现有 SRS/PUSCH 波形导出
per-link clean/observed 频域 IQ 和/或由 OFDM IFFT 生成的时域 IQ。可选
`noncooperative.enabled=true` 会基于 NR SRS multi-UE shared source 写
`/iq/noncooperative/rx_time_*`，只保存 BS 侧 shared time-domain IQ 和 active UE
标签，供非合作多目标检测/定位实验使用。

如果目标是构建“离线逐链路、在线组合”的非合作 IQ 数据库，推荐使用
`output.profile: "iq_link_library"`。该 profile 当前要求 `phy.standard: "nr_srs"`，
只执行 RT CFR、SRS resource/tx grid 和 clean channel apply，直接从 `rx_grid_clean`
按 `phy.iq.clean_output: "time" | "frequency" | "both"` 生成 `/iq/link/time_clean`
和/或 `/iq/link/frequency_clean`；不会运行 common
impairment/AWGN、SRS receiver、`/observation/cfr_est`、ranging、array spectrum 或
可视化，也不会写 `/channel`、`/paths`、`/waveform` 等 full-contract 重型组。这样保存的
clean IQ 满足线性叠加前提，后续可在训练管线中按 UE 组合在线混合，并在混合后统一加入
噪声、同步偏差或前端损伤。

`phy.tx_power_dbm` 当前会影响 SRS/PUSCH 的发射 grid 幅度。默认
`phy.power.noise_mode="relative_snr"` 保持历史 SNR 口径：TX power 上调时 RSSI 和
noise power 同步上移，SNR 基本不变。切到 `"absolute_thermal"` 后，噪声按 kTB + NF
固定，TX power 会直接改变 SNR。

这一路径适合做室内定位 CSI 基线和 PUSCH-DMRS proxy 对比，但仍不能称为
完整 3GPP NR SRS：本项目的 hopping/sequence/power control 是可解释的 v2 subset，
尚未做 38.211/38.213 reference 对齐或认证级一致性测试，见
[Feature TODO](docs/todo/feature.md)。

室内 100 MHz SRS 模板当前默认 `rt.synthetic_array=false`、direct uplink、
UE shard `20`。已完成 `median_0000 label0p2` 的 `7 BS × 2583 UE` direct
uplink baseline；历史 micro-sweep 和旧 shard 参数只作为实验记录，见
[Indoor FR1 validation](docs/sys/indoor_fr1_100mhz_validation.md)。

同一场景的 PUSCH 与 SRS 输出可用轻量脚本对比：

```bash
uv run python scripts/compare_phy_csi_outputs.py \
    outputs/bistro_0000_pusch \
    outputs/bistro_0000_nr_srs \
    --label-left pusch_dmrs \
    --label-right srs_like
```

## 配置文件

| 模板 | 用途 |
|------|------|
| `config/defaults/measurement_mvp.yaml` | 通用 custom OFDM + impairment（默认） |
| `config/defaults/cfr_truth_only.yaml` | 最小 `full` + `products: ["cfr_truth"]` 输出；只写 `/channel/truth/cfr` 和必要元数据 |
| `config/defaults/rt_labels_only.yaml` | 紧凑 RT link-level 标签；写 `/labels/link/*`，跳过 CFR/CIR/path samples/PHY |
| `config/defaults/nr_pusch_mvp.yaml` | NR PUSCH 4x4 SU-MIMO TDD uplink |
| `config/defaults/nr_pusch_indoor_positioning_fr1_100mhz.yaml` | Bistro 室内 FR1 100 MHz PUSCH-DMRS 定位模板 |
| `config/defaults/nr_srs_indoor_positioning_fr1_100mhz.yaml` | Bistro 室内 FR1 100 MHz NR SRS subset 定位模板 |
| `config/perf/nr_pusch_3x3000_sharded.yaml` | 3 BS × 3000 UE shard 性能回归模板 |
| `config/perf/nr_pusch_6x8884_sharded.yaml` | 6 BS × 8884 UE 4 GPU shard 验收模板 |

完整字段说明见 [config/README.md](config/README.md)。配置加载时自动进行 pydantic schema 校验。

## HDF5 输出结构

普通运行生成 `outputs/<run_dir>/results.h5`。开启 `output.sharding.enabled=true` 时，输出目录会按职责分层：

```text
outputs/<run_dir>/
  run_config.yaml       # run-full 写入的最终 YAML 配置
  summary.json          # 可选队列/验收脚本汇总，不属于 HDF5 schema
  results/              # result_xxx.h5，自包含 HDF5 shard
  manifest/             # aggregate manifest、per-shard manifest、config snapshot
  logs/                 # run.log、heatmap.log、debug/perf 日志
  figures/              # 可选采样可视化；index.json + standard/multiuser/iq/heatmaps 子目录
```

`run_config.yaml` 保存 YAML 加载和 CLI 覆盖后的最终运行配置，便于把单个输出目录作为
可复现实验单元。`manifest/manifest.json` 是 shard 数据集入口，记录全局 UE 覆盖范围、
resolved TX/RX 索引、fallback 拆分记录和 `manifest/config_snapshot.json` 路径。
本地队列、验收和 heatmap 包装脚本也应把运行日志、heatmap 日志和汇总 JSON 放回同一个
run 目录，例如 `logs/run.log`、`logs/heatmap.log` 和 `summary.json`，避免在
`outputs/` 根目录生成 sidecar 文件。

可视化入口会把普通采样诊断图写到 `figures/standard/`，RSS radio map 写到
`figures/heatmaps/`，multi-UE SRS shared observation 图写到 `figures/multiuser/`。
`multiuser_srs` 图集用于检查资源占用、BS 侧混合 RX grid、per-UE CFR 折线与幅度/相位
热力图、拆分误差和 shared/separated 空间谱。IQ 图写到 `figures/iq/`，展示
频域 IQ 的 I/Q/幅度/相位和时域 IQ 波形，以及非合作帧中的 active target 位置。

| Group | 内容 |
|-------|------|
| `/meta` | schema 版本、运行 ID、随机种子、配置快照、`output_profile`、`output_products` |
| `/shard` | shard 模式下的局部 TX/RX 到全局 BS/UE 索引映射 |
| `/input` | 标签文件、场景文件 |
| `/topology` | TX/RX 三维位置 |
| `/scene` | 场景文件、`scene_id`、`map_id` |
| `/antenna` | 阵列类型、极化、天线数 |
| `/frequency` | 中心频率、带宽、子载波 |
| `/channel/truth` | 真值 CFR `[tx, rx, rx_ant, tx_ant, subcarrier]`、CIR、LoS/NLoS |
| `/derived` | 几何距离、first-path truth delay/range、AoA、LoS/NLoS、link mask、TX/RX 平面几何量 |
| `/paths/samples` | 路径采样：顶点、交互、对象 ID、多普勒、延迟 |
| `/link` | 双工模式、PHY 方向、resolved `tx_role`/`rx_role` |
| `/waveform` | OFDM/NR 波形；NR PUSCH/SRS 统一保存 `tx_grid/rx_grid/noise_variance` 和 power/RSSI metadata，NR SRS 另写 resource/sequence/port datasets |
| `/multiuser` | 可选 NR SRS shared observation；多个 UE 正交 SRS 叠加后的 `rx_grid_shared`、资源分配、collision mask 和 per-UE resource/allocated CFR |
| `/iq` | 可选协议无关 IQ observation；`/iq/link` 保存合作式 per-link 频域/时域 IQ，`/iq/noncooperative` 保存非合作 shared time-domain IQ 和 active UE 标签 |
| `/array` | 阵列 snapshot、`aoa_heatmap_label`、truth/estimated/RX-grid Bartlett 空间谱；旧 `spatial_spectrum_label` 和 `spatial_spectrum_srs` 不再写入 |
| `/observation` | 估计 CFR `[snap, tx, rx, rx_ant, tx_ant, subcarrier]`、SNR、CFO 等 |
| `/ranging` | 从受损后 `cfr_est` 估计的 ToA/range observation；含 `pdp_peak` 与 `phase_slope` |
| `/receiver` | 估计器类型、MIMO 检测器 |
| `/evaluation` | NMSE、BER、BLER（TB CRC）、num_block_errors/num_blocks |
| `/runtime` | 软件版本、耗时 |

`output.profile="rt_labels_only"` 会写 `contract_name="sionna_measurement_rt_labels"`，
只保留 topology、derived 和 `/labels/link/*` compact table；不会写 `/channel`、
`/paths`、`/waveform`、`/observation`、`/array` 或 `/ranging`。
`output.profile="iq_link_library"` 会写
`contract_name="sionna_measurement_iq_link_library"`，只保存可在线混合的 clean
`/iq/link` 和必要元数据。
`output.profile="full"` 可选 `output.products`，并用 `/meta/output_products` 记录
解析后的产品列表；validator 只要求所选产品对应的 group/dataset。例如
`products: ["cfr_truth"]` 不会写 `/paths`、`/derived`、`/waveform`、`/observation`、
`/array` 或 `/ranging`。`array`、`ranging`、`iq` 和 `multiuser` 作为 full product
时会自动启用对应下游模块，但内部计算依赖不会隐式写出未选择的重型 group。

schema `2.3.0` 起，历史 `rt_lite` 和 `custom` profile 已破坏式移除：轻量 full-contract
输出请使用 `profile: "full"` + `products`，紧凑标签输出请使用 `rt_labels_only`。

完整数据契约见 [docs/sys/07_config_and_h5_format.md](docs/sys/07_config_and_h5_format.md)。

## GPU 与大规模 PUSCH

`runtime.device: "cuda"` 会让 NR PUSCH 的 PyTorch/Sionna 频域链路在 GPU 上运行。SU-MIMO PUSCH 支持把多个独立 `(snapshot, UE, BS)` link 合成 batch，配置项为 `phy.su_mimo_link_batch_size`；schema 默认保守值是 1，NR PUSCH 与性能模板使用 64。

已验证的参考规模：

| 场景 | 结果 |
|------|------|
| `3 BS × 3000 UE × 4x4 PUSCH` | shard+batch64 运行完成，3 个 HDF5 schema 通过，端到端约 178 s |
| `6 BS × 8884 UE × 4x4 PUSCH` | 4 GPU shard+batch64 运行完成，9 个 HDF5 schema 通过，端到端约 279 s |

生产级大规模 PUSCH 推荐开启 UE shard：

```bash
uv run python -m sionna_measurement_sim.app.cli \
    --config config/perf/nr_pusch_6x8884_sharded.yaml \
    run-full \
    --output-dir outputs/nr_pusch_6x8884_sharded
```

shard 模式不会把多个进程写进同一个 HDF5；每个进程写自己的 `results/result_xxx.h5`，`manifest/manifest.json` 汇总所有 shard 的全局索引、可视化摘要和 debug 性能日志路径。若某个 shard 触发 CUDA OOM 或 Dr.Jit 单数组 2^32 上限，默认会按 UE 继续二分重试，直到 `min_shard_size` 或成功为止。

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
    phy/              PHY module registry + common link + 损伤 + NR PUSCH/SRS + backend
    benchmark/        RT/write/spectrum 性能 benchmark harness
    io/               HDF5 读写、schema validator、manifest、label 解析
    analysis/         诊断分析
    visualization/    拓扑/路径/CFR/NMSE/空间谱图
  config/defaults/    默认 YAML 配置
  data/               本地数据路径（gitignore，可为本地 symlink）
  tests/fixtures/scenes/test/   测试场景
  tests/
    unit/ domain / config / impairments / MIMO channel / MIMO config
    schema/           HDF5 truth / CIR / NR PUSCH
    adapter/          Sionna adapter shape
    integration/      RT truth / 4x4 SU-MIMO / MU-MIMO / batch / calibration
    statistical/      AWGN / impairments / motion / NR PUSCH MIMO metrics
  docs/               设计文档
    sys/              当前系统事实和接口契约
    todo/             当前 active TODO 与已完成 TODO history
    performance/      历史性能实验记录
    legacy/           过时文档临时归档，供人工复核
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
| [phy_module_development](docs/sys/phy_module_development.md) | 新 PHY module 接入指南 |
| [todo](docs/todo/README.md) | 当前 active TODO 总入口 |
| [todo/feature](docs/todo/feature.md) | 功能、标准完整性、算法增强和研究能力 TODO |
| [todo/structure](docs/todo/structure.md) | 数据契约、reader、benchmark 入口和 legacy 模块 TODO |
| [todo/performance](docs/todo/performance.md) | 大规模运行、写盘、RT、空间谱和 GPU 调度 TODO |
| [todo/bug](docs/todo/bug.md) | 已确认缺陷和回归 TODO |
| [benchmark_harness](docs/performance/benchmark_harness.md) | benchmark rt/write/spectrum 输出格式和推荐命令 |
| [indoor_fr1_100mhz_validation](docs/sys/indoor_fr1_100mhz_validation.md) | Bistro FR1 100 MHz probe 与全量成本估算 |
| [performance](docs/performance/README.md) | 历史性能实验记录索引和 legacy 审查状态 |

## 约束

- Python 3.11+，`uv` 管理环境
- Sionna 2.x（RT + PHY）+ PyTorch（张量后端）
- 禁止业务层直接 import Sionna
- 禁止将 truth CFR 写为 `/channel/cfr`（必须 `/channel/truth/cfr`）
- 禁止引入 TensorFlow 到核心链路
- 大型输出不入 git
