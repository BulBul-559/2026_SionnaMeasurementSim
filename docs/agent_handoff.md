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
其中的参数反映当时实验事实，不一定代表当前默认配置。`docs/legacy/` 用于放置已经过时、
但暂时保留给人工复核的文档。

## 项目定位

SionnaMeasurementSim 是一个基于 Sionna RT 的室内无线仿真数据生成系统。当前重点是：

- 从场景/label 生成 RT truth、CIR/CFR、路径真值、AoA、NLoS path truth。
- 生成 PHY 观测数据，包括 NR PUSCH-DMRS CSI proxy 和 NR SRS standards-shaped v2 subset uplink sounding。
- 可选生成 waveform-level ranging observation，从 `/observation/cfr_est` 估计 ToA/one-way range。
- 可选生成 RSS radio map 可视化，从 `/observation/rssi_dbm` 按 BS 聚合并覆盖到 floorplan。
- 支持频域 waveform grid、array/空间谱、HDF5 多文件 shard 输出和 manifest 汇总；
  AoA label、angle grid 与 Bartlett 空间谱统一使用 scene/global 角度，空间谱会用
  `/devices/rx_orientation_rad` 适配接收阵列旋转。
- 为后续定位、场重建、CSI/embedding 学习生成可复现数据。

当前 `main` 已包含 NR PUSCH 与 NR SRS 的 clean channel、基带损伤、AWGN、
uplink power/RSSI 和 observation metadata 统一链路；`custom_ofdm` 保留为 legacy 路径。
NR SRS 已升级为 standards-shaped v2 subset，并可选生成 multi-UE 正交 SRS shared
observation。输入 label 解析已切到标准 label `0.1.0` 的全场景顶层
`bs_points`/`ue_points` 口径。
当前 HDF5 schema 版本是 `2.0.0`。`output.profile` 支持 `full`、`rt_lite`、
`rt_labels_only` 和 `iq_link_library`：`rt_lite` 是保留 full contract 的轻量 preset，会关闭
PHY/ranging/spectrum/viz/calibration/full paths；`rt_labels_only` 使用独立
`sionna_measurement_rt_labels` compact contract，只写 topology、derived 和
`/labels/link/*` link-level 标签，不写 CFR/CIR/path samples/waveform/observation/array/ranging；
`iq_link_library` 使用独立 `sionna_measurement_iq_link_library` compact contract，只写
topology/device/antenna/frequency/link/runtime 和 clean `/iq/link`，用于离线逐链路、在线混合
多 UE IQ，不写 `/channel`、`/paths`、`/waveform`、`/observation`、`/array` 或 `/ranging`。
新输出不再写 array 兼容别名
`/array/spatial_spectrum_label` 或 `/array/spatial_spectrum_srs`；AoA 监督 heatmap
统一使用 `/array/aoa_heatmap_label`，SRS/估计 CFR 空间谱统一使用
`/array/spatial_spectrum_cfr_est`。

## 深读文档地图

| 目标 | 建议阅读 |
|---|---|
| 快速了解系统分层 | `docs/sys/00_project_overview.md` |
| 查 CLI/config/shard/debug 配置 | `docs/sys/01_app_and_config.md`, `config/README.md` |
| 查 pipeline 编排和 BS/UE→TX/RX 映射 | `docs/sys/04_rt_pipeline.md` |
| 查 PUSCH/SRS/custom OFDM 实现口径 | `docs/sys/05_phy_observation.md` |
| 查 HDF5 字段、shape、manifest | `docs/sys/07_config_and_h5_format.md` |
| 查性能 benchmark 入口 | `docs/performance/benchmark_harness.md` |
| 查 SRS baseline 和 shard size 依据 | `docs/sys/indoor_fr1_100mhz_validation.md` |
| 新增 PHY module | `docs/sys/phy_module_development.md` |
| 当前 TODO 总入口 | `docs/todo/README.md` |
| 功能 / 标准 / ranging 后续工作 | `docs/todo/feature.md` |
| 性能和规模化后续工作 | `docs/todo/performance.md` |
| 结构和数据契约后续工作 | `docs/todo/structure.md` |
| 历史性能实验索引 | `docs/performance/README.md` |

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
`/waveform/noise_variance`、`/waveform/tx_power_dbm_per_port`、
`/waveform/tx_power_scale_linear` 以及 `/observation/cfo_hz` 等损伤观测字段。
`phy.tx_power_dbm` 默认以 `reference_tx_power_dbm=0 dBm` 标定单位幅度 TX grid；
`tx_power_dbm=23` 时发射幅度乘约 `sqrt(200)`。默认 `phy.power.noise_mode="relative_snr"`
保持历史 SNR 口径，RSSI/noise power 随 TX power 同步平移；`"absolute_thermal"` 则用
kTB + noise figure 固定噪声，TX power 会改变 SNR。
NR SRS 另写 `/waveform/srs_resource_mask`、`/waveform/srs_pilot_symbols`、
`/waveform/srs_re_symbol_indices`、`/waveform/srs_re_subcarrier_indices`、
`/waveform/srs_port_tx_ant_map`、per-symbol PRB/cyclic-shift/sequence/power metadata
和 `/observation/cfr_est_resource`。schema `1.5.0` 后不再写 `/waveform/pilot_code`、
`/waveform/srs_port_index`、`/observation/srs_cfr_est`、`/array/spatial_spectrum_label`
或 `/array/spatial_spectrum_srs`。

schema `1.2.0` 后 `/derived/rtt_like_m` 和 `/derived/rtt_like_s` 已移除。
truth range 写为 `/derived/first_path_propagation_range_m`。估计型 ToA/range 写到
`/ranging/pdp_peak` 和 `/ranging/phase_slope`；其中 `rtt_equiv_s=2*toa_est_s`
只是 two-way equivalent，不是完整协议 RTT。

`nr_srs` 当前是 standards-shaped v2 subset，不是完整 3GPP NR SRS。它按 `phy.srs`
生成 full-slot 14-symbol resource grid，支持 comb/BWP/hopping、`zc_like`/`nr_zc`、
group/sequence hopping、cyclic-shift port multiplexing、简化 antenna switching 和
通用 uplink power/RSSI metadata；SRS RE 上做 LS/despread，然后插值到 full-band
`/observation/cfr_est`：

```text
resource LS -> cfr_est_resource -> frequency interpolation -> cfr_est
```

v2 仍不能称为 3GPP-compliant：38.211/38.213 reference 对齐、完整 antenna switching
procedure、闭环功控和标准一致性验证仍在 TODO 中，见 `docs/todo/feature.md`。

可选 `phy.srs.multiuser.enabled=true` 会额外写 `/multiuser` group，用于研究同一
静态 snapshot 下多个 UE 正交 SRS 同时发射的 shared observation。当前只支持理想 OFDM
正交资源：`comb_offset` 或 `prb_split`，不新增 per-UE CFO/timing，也不建模非正交碰撞。
`/multiuser/rx_grid_shared` 是 BS 侧混合观测，`cfr_est_resource` 是每个 UE 实际 SRS
RE 上的权威 CFR estimate，`cfr_est_allocated` 是分配频带上的 CFR estimate。若使用
`comb_offset`，未 sounding 子载波的补全只应视作插值派生，不能当成真实全频观测。

可选 `phy.iq.enabled=true` 会额外写协议无关 `/iq/link`，用于在正常 SRS/PUSCH
链路中保存 clean/observed 频域 IQ 或由 OFDM IFFT 合成的时域 IQ。IQ 层只读
waveform 产物，不参与 receiver、CFR 估计、ranging 或 BER/NMSE 计算。时域 IQ 的
约定是 `ofdm_ifft_per_symbol_cp_appended_contiguous_symbols`，不是连续 RF passband
或真实 ADC 流。

如果只需要可在线相加的 clean IQ 链路库，使用
`config/defaults/iq_link_library_nr_srs.yaml` 或 `output.profile: "iq_link_library"`。
该 profile 当前仅支持 NR SRS：pipeline 仍需要 RT CFR 来计算 `H*x`，但 SRS 模块会在
`rx_grid_clean` 后直接返回，不再执行 common impairment/AWGN、SRS receiver、resource
LS、full-band `cfr_est`、ranging、空间谱或可视化。模板默认只保存
`/iq/link/time_clean`；可用 `phy.iq.clean_output: "time" | "frequency" | "both"`
选择只保存时域 clean IQ、只保存频域 clean IQ 或两者都保存。未设置 `clean_output`
时仍兼容底层 `save_frequency_clean/save_time_clean` 开关。

可选 `noncooperative.enabled=true` 会基于 NR SRS multi-UE shared source 写
`/iq/noncooperative/rx_time_clean` 和/或 `rx_time_observed`，并保存每个 frame 的
active UE index、全局 index、位置标签和 shared resource occupancy/collision mask。
第一版只支持 `signal_standard="nr_srs"`，只保存时域 shared IQ；它仍假设静态 snapshot
和理想 OFDM 正交资源，不建模 per-UE 独立 CFO/timing/asynchrony 或随机接入式非正交碰撞。

配置层注意：`config/schema.py` 是 YAML/Pydantic validation model，ranging 算法使用
`ranging/config.py` dataclass；两者通过 `config/mappers.py` 集中转换，不要在 CLI 或
pipeline 里重新手写 ranging 字段拷贝。

性能工程注意：debug profiling 在失败运行中也会尽量写 `logs/perf_summary*.json`，
并汇总 hardware peak 与 HDF5 dataset 写入统计。隔离 benchmark 已进入主 CLI：

```text
benchmark rt       # RT-only，不跑 PHY/HDF5/可视化
benchmark write    # synthetic MeasurementSimulationResult -> HDF5 writer/schema
benchmark spectrum # synthetic array samples -> Bartlett spectrum core
```

benchmark 输出是 ignored `outputs/` 下的 JSON/CSV/log artifact，不是正式 HDF5 schema 数据。

RT labels-only 输出可用以下模板生成：

```text
config/defaults/rt_labels_only.yaml
```

它适合大规模场景的视觉预训练或 link-level 标签筛选；若需要 CFR、CIR、path samples、
path 可视化、PHY observation、空间谱或 ranging，应使用 `output.profile="full"`。

可视化注意：`visualization.plots` 可加入 `radio_map`。该图把 uplink 中 UE 发射、BS 接收的
`/observation/rssi_dbm` 解释为“每个 BS 在 UE 位置的 RSS 代表值”，每个 BS 输出一张图到
`figures/heatmaps/`。`visualization.radio_map_mode` 默认 `interpolated`，可设为
`samples` 或 `both`；shard 模式下 radio map 在 aggregate manifest 写完后聚合所有 shard 生成。
无有效 RSS 的 UE-BS 位置在绘图层按全局最小 RSS 渲染为最弱信号，避免插值补出虚假的覆盖。
普通采样诊断图写到 `figures/standard/`，multi-UE SRS shared observation 图写到
`figures/multiuser/`。`multiuser_srs` plot 只在 `/multiuser` group 存在时生成，包含
resource ownership、shared RX grid、per-UE resource/allocated CFR 折线与幅度/相位热力图、
误差摘要以及
shared/separated 空间谱，用于检查多 UE 正交资源拆分和 BS 侧混合观测。
`iq` plot 只在 `/iq` group 存在时生成，输出到 `figures/iq/`；当前包含 per-link
频域 IQ、per-link 时域 IQ、非合作 shared time IQ 和 active UE/BS 示意图。

## 数据目录

`data/` 与 `outputs/` 都是 ignored 本地路径，可以是 symlink。

标准场景目录应提供 `scene.xml`、标准 label `0.1.0` JSON 和可选 floorplan 资源：

```text
data/<dataset>/<scene_id>/
├── scene.xml
├── label/
│   └── <label_variant>.json
└── floorplan/
    ├── floorplan_1p60.png
    └── meta.json
```

pipeline 只读取 label 顶层 `bs_points` 和 `ue_points` 作为全场景默认点集；`groups`
保留为房间、区域或生成策略子集元数据，当前没有 `label_group_policy`。点坐标单位为米，
支持 `position: [x, y, z]` 或显式 `x/y/z`。floorplan 命名中的 `1p60` 表示截断高度
`1.60 m`，图像与真实坐标转换来自 `floorplan/meta.json`。

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
需要按目标场景复制后修改。推荐把运行配置放在目标输出目录的 `run_config.yaml`；
运行 `run-full` 时，CLI 会把 YAML 加载和命令行覆盖后的最终配置写回输出目录根部的
`run_config.yaml`，使每个结果目录自包含。
本地 tmux 队列、验收统计和 heatmap 包装脚本的输出也必须放回对应 run 目录：
`logs/run.log`、`logs/heatmap.log` 和 `summary.json`。不要再在 `outputs/` 根目录写
`<run_name>.run.log`、`<run_name>.heatmap.log` 或 `<run_name>_summary.json`。

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
- schema `1.5.0` 后使用统一 waveform 字段 `/waveform/tx_grid`、`/waveform/rx_grid`、
  `/waveform/noise_variance`，schema `1.6.0` 后另写通用 power 字段
  `tx_power_dbm_per_port`、`tx_power_scale_linear`，以及 NR SRS resource 字段 `srs_resource_mask`、
  `srs_pilot_symbols`、`srs_re_symbol_indices`、`srs_re_subcarrier_indices`、
  `srs_port_tx_ant_map` 和 SRS power metadata。
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

运行 SRS 全流程时，建议复制模板到目标输出目录的 `run_config.yaml`，再改
label/output/gpu。也可以用临时 YAML 启动；`run-full` 会把最终有效配置写入输出目录的
`run_config.yaml`，避免后续复现实验时再去追散落的临时配置文件。
如果脚本还会跑 radio-map heatmap 或 schema/验收汇总，附加 artifact 也写在该输出目录下：
`logs/heatmap.log`、`summary.json`。

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

- 不要默认跑最高密度 label 全量；先用中等或稀疏采样做 smoke，再按 shard 扩大。
- 标准 label 中不要再默认取第一个 `groups`；全场景点集必须来自顶层
  `bs_points`/`ue_points`。
- `data/` 和 `outputs/` 不进 git，里面的真实数据和仿真输出都应视为本地大文件。
- 历史性能报告中的 `shard_size=25` 是历史记录；当前 SRS 生产模板推荐 `20`。
- direct uplink 下 UE 是 source，UE block 大小比 BS 数更容易触发 RT/Dr.Jit 限制。
- 多文件 shard 是当前推荐输出方式，不建议为了训练强行合并成单个巨大 HDF5。
- 空间谱和 visualization 默认关闭；它们适合小样本诊断，不适合默认全量生产。
- `path_samples` 可视化只画当前采样选择中的第一个 UE-BS 链路，避免多个链路路径几何混叠。
- HDF5 下游读取应通过 `manifest/manifest.json` 和其中记录的 `results/result_xxx.h5` shard 列表，不要假设只有 `results.h5`，也不要假设文件名连续；fallback 可能产生 `result_089_00.h5` 这类子 shard。
- `nr_srs` 是 standards-shaped v2 subset，不要在论文或文档里称为 standards-complete 3GPP SRS。

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
