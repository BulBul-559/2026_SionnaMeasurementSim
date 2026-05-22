# 07. 配置系统与 HDF5 数据格式

## 一、配置系统

### 1.1 设计概览

配置系统基于 **Pydantic v2 + YAML**，在仿真启动前完整校验，避免 RT/PHY 运行到一半才发现参数错误。

```
YAML/JSON 文件
    │
    ▼
config/loader.py  →  解析 + 校验  →  MeasurementConfig (Pydantic)
    │
    ▼
app/cli.py  →  覆盖 CLI 参数  →  RTTruthRunConfig (dataclass)
    │
    ▼
rt/truth_pipeline.py  →  编排执行
```

文件：
- `sionna_measurement_sim/config/schema.py` — 全量 Pydantic schema
- `sionna_measurement_sim/config/loader.py` — YAML/JSON 加载与校验
- `sionna_measurement_sim/app/cli.py` — CLI 入口，参数覆盖
- `config/defaults/` — 配置模板

### 1.2 配置顶层结构

`MeasurementConfig` 包含这些顶层分组：

```yaml
runtime:       # 运行环境 (seed, device, precision)
debug:         # 可选性能 profiling 日志
input:         # 输入数据 (label_file, scene_file, max_bs, max_ue)
output:        # 输出控制 (root_dir, hdf5_filename, compression, sharding)
carrier:       # 载波频率 (center_frequency_hz, bandwidth_hz, num_subcarriers)
antenna:       # 天线阵列 (bs_array, ue_array)
rt:            # 射线追踪 (max_depth, los, specular_reflection, ...)
link:          # 链路配置 (duplex_mode, phy_link_direction)
phy:           # 物理层 (standard, snr_db, nr_pusch fields)
impairments:   # 损伤模型 (awgn, cfo, sfo, phase_noise, timing, agc_adc)
ranging:       # 波形级 ToA/range observation (source=cfr_est)
receiver:      # 接收机 (estimator_type, mimo_detector, failure_policy)
motion:        # 运动/多普勒 (mobility_mode, num_time_steps, velocity)
calibration:   # 校准 (profile_id)
visualization: # 采样可视化
```

### 1.3 各组字段详表

#### `runtime` — 运行环境

| 字段 | 类型 | 默认值 | 约束 | 说明 |
|------|------|--------|------|------|
| `seed` | int | 42 | ≥0 | 全局随机种子 |
| `device` | str | `"cpu"` | — | PyTorch 设备；NR PUSCH 支持 `"cpu"`、`"cuda"`、`"cuda:0"` 等 |
| `require_gpu` | bool | false | — | 是否要求 GPU |
| `precision` | str | `"single"` | — | 浮点精度 |
| `torch_deterministic` | bool | false | — | PyTorch 确定性模式 |

当前依赖锁定 PyTorch `2.10.0+cu128`，通过官方 PyTorch CUDA 12.8 wheel 源安装。`runtime.device: "cuda"` 会驱动 NR PUSCH 的 PyTorch/Sionna 频域链路在 CUDA 设备上执行；如果 CUDA 不可用则报错，不自动回退到 CPU。

当前 pipeline 实际驱动设备选择的是 `device` 字段；`require_gpu`、`precision`、`torch_deterministic` 是 schema 保留字段，尚未完整接入所有执行路径。

#### `debug` — 性能 profiling

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | false | 是否写性能日志 |
| `hardware_interval_s` | float | 1.0 | GPU/CPU/RSS 采样间隔 |
| `link_log_interval` | int | 250 | link chunk 汇总间隔 |
| `torch_synchronize` | bool | true | 阶段计时前后同步 CUDA |
| `write_hardware_samples` | bool | true | 是否写硬件采样 CSV |

#### `input` — 输入数据

| 字段 | 类型 | 默认值 | 约束 | 说明 |
|------|------|--------|------|------|
| `label_file` | str | `"tests/fixtures/scenes/test/test5.json"` | — | BS/UE 位置标签 JSON |
| `scene_file` | str | `"tests/fixtures/scenes/test/scene.xml"` | — | Mitsuba 场景 XML |
| `scene_id` | str | scene 文件名 stem | — | 与地图/平面图系统对齐的稳定场景 ID |
| `map_id` | str | `""` | — | 可选地图版本 ID |
| `label_schema` | str | `"simplesionna_v1"` | — | 标签 schema 版本 |
| `coordinate_system` | str | `"scene_local_xyz_m"` | — | 坐标系统 |
| `max_bs` | int | 6 | ≥1 | BS 数量上限 |
| `max_ue` | int | 100 | ≥1 | UE 数量上限 |

#### `output` — 输出控制

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `root_dir` | str | `"outputs"` | 输出根目录 |
| `run_id_format` | str | `"{label_stem}_{timestamp}"` | 输出子目录命名模板 |
| `hdf5_filename` | str | `"results.h5"` | HDF5 文件名 |
| `compression` | str | `"gzip"` | HDF5 压缩算法 |
| `save_full_paths` | bool | false | 是否保存 `/paths/full`（全量路径表） |
| `save_sampled_paths` | bool | true | 是否保存 `/paths/samples` |
| `save_raw_waveform` | bool | false | 是否保存原始波形 |

`compression` 可选 `gzip`、`lzf`、`none`。大规模性能排查时可用 `none` 或 `lzf` 分离写盘压缩成本。

`output.sharding` 子段控制 UE shard：`enabled`、`axis`、`shard_size`、`filename_pattern`、`results_dir`、`manifest_dir`、`parallel_workers`、`gpu_ids`、`visualization_mode` 和 `fallback`。当前生产化的是 UE range shard，`axis: "ue"`，输出文件写到 `results/result_000.h5`，aggregate manifest 和 config snapshot 写到 `manifest/`。

`fallback` 默认开启，针对 CUDA OOM 与 Dr.Jit 单数组 2^32 上限自动把失败 shard 按 UE 二分重试。比如计划 shard `089` 覆盖 20 个 UE 时失败，会落成 `results/result_089_00.h5` 与 `results/result_089_01.h5`；下游只需要读取 `manifest/manifest.json`，不应假设文件名连续或每个计划 shard 只对应一个 HDF5。

#### `carrier` — 载波与频率

| 字段 | 类型 | 默认值 | 约束 | 说明 |
|------|------|--------|------|------|
| `center_frequency_hz` | float | 3.5e9 | >0 | 中心频率 Hz |
| `bandwidth_hz` | float | 20e6 | >0 | 带宽 Hz |
| `num_subcarriers` | int | 64 | ≥2 | 子载波数 |
| `subcarrier_spacing_hz` | float | 0（自动推导） | — | 子载波间隔 Hz |

`subcarrier_spacing_hz = bandwidth_hz / num_subcarriers`，若显式指定则校验一致性（容忍 1% 误差）。

> **NR PUSCH 约束**：NR PUSCH 链路实际子载波数 = `num_prb × 12`，必须与 `num_subcarriers` 一致。`nr_pusch_mvp.yaml` 模板中已对齐（`num_prb=4, num_subcarriers=48`）。

#### `antenna` — 天线阵列

每侧天线包含 `bs_array` 和 `ue_array`，均为 `ArraySpec`。进入 RT/PHY 时再根据
`link.phy_link_direction` 解析成 TX/RX 阵列：

| 字段 | 类型 | 默认值 | 约束 | 说明 |
|------|------|--------|------|------|
| `type` | str | `"planar"` | — | 阵列类型 |
| `num_rows` | int | 1 | ≥1 | 行数 |
| `num_cols` | int | 1 | ≥1 | 列数 |
| `vertical_spacing_lambda` | float | 0.5 | >0 | 垂直间距（波长） |
| `horizontal_spacing_lambda` | float | 0.5 | >0 | 水平间距（波长） |
| `pattern` | str | `"iso"` | — | 天线方向图 |
| `polarization` | str | `"V"` | — | 极化 (V/H) |
| `orientation_mode` | str | `"fixed"` | — | 朝向模式 |
| `orientation_rad` | [float×3] | [0,0,0] | — | 朝向角 rad |

天线数 = `num_rows × num_cols`。4x4 MIMO 需 BS 和 UE 两侧均为 2×2。

#### `array.spectrum` — 空间谱输出

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | false | 是否生成 Bartlett 空间谱；默认关闭以避免大规模输出膨胀 |
| `sources` | list[str] | `["truth_cfr", "cfr_est", "rx_grid"]` | 可选 `truth_cfr`、`cfr_est`、`rx_grid`；schema 1.5.0 后不再接受历史 `srs_cfr_est` source |
| `method` | str | `"bartlett"` | 第一版仅支持 Bartlett 扫描 |
| `zenith_bins` | int | 91 | zenith 方向网格数 |
| `azimuth_bins` | int | 181 | azimuth 方向网格数 |
| `zenith_min_rad/max_rad` | float | `[0, pi]` | zenith 扫描范围，默认全空间 |
| `azimuth_min_rad/max_rad` | float | `[-pi, pi]` | azimuth 扫描范围，默认全向 |
| `normalize` | str | `"per_link_max"` | 每条 link 最大值归一到 1 |
| `aggregate_subcarriers` | str | `"mean"` | 子载波聚合口径 |
| `aggregate_symbols` | str | `"mean"` | OFDM symbol 聚合口径 |
| `link_chunk_size` | int | 512 | Bartlett 空间谱按 link chunk 向量化的 chunk 大小 |

这里的“全向”指扫描角度范围覆盖 scene/global 方向域；天线方向图仍由
`antenna.*.pattern` 控制，默认模板使用 `iso`。Bartlett steering vector 先按
Sionna `PlanarArray` 的本地 y-z 平面布局生成：top-left 起、column-first 编号、
第一行 z 为正；随后用每个 RX 的 `/devices/rx_orientation_rad` 旋转到 scene
坐标，因此 `/array/angle_grid_rad`、AoA label 和各类 `spatial_spectrum_*` 都统一
以 scene/global zenith、azimuth 表示。可视化读取时会根据 `/link/tx_role` 和
`/link/rx_role` 把 link-view 轴映射回 BS/UE 语义。

#### `visualization` — 采样可视化

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | false（schema）/ true（默认模板） | 是否在 pipeline 结束后生成采样 PNG |
| `output_dir` | str | `"figures"` | 相对 run 输出目录 |
| `sample_policy` | str | `"valid_links_first"` | UE 采样策略；可选 `valid_links_first`、`spatially_spread_valid_links`、`random`、`first` |
| `random_seed` | int | 42 | 可视化采样随机种子 |
| `max_bs` | int | 5 | 自动图中最多 BS 数 |
| `sample_ue_count` | int | 3 | 自动采样 UE 数 |
| `max_ue` | int | 5 | 自动图中最多 UE 数 |
| `dpi` | int | 140 | PNG DPI |
| `format` | str | `"png"` | 第一版仅支持 PNG |
| `plots` | list[str] | 核心诊断集 | topology、link、CFR、waveform、AoA/NLoS、空间谱、NMSE、path 图 |

`sample_policy` 说明：

- `valid_links_first`：优先从 `/derived/link_valid_mask=True` 的 UE 中随机采样。
- `spatially_spread_valid_links`：优先有效链路，并用 UE 的 XY 坐标做远点采样，让采样位置更分散。
- `random`：从所有 UE 中随机采样。
- `first`：取前 N 个 UE。

绘图输出约定：

- 涉及子载波的热力图统一把 subcarrier 放在纵轴；CFR 折线图例外，把 subcarrier 放在横轴。
- 热力图绘制时显式关闭显示插值，使用原始采样网格直接画图。
- `cfr_lines` 输出 `cfr_lines_magnitude.png` 和 `cfr_lines_phase.png`。
- `cfr_heatmap` 输出 `cfr_heatmap_magnitude.png` 和 `cfr_heatmap_phase.png`；热力图轴为 `[subcarrier, antenna_pair]`。
- `cfr_error` 输出 `cfr_error_magnitude.png` 和 `cfr_error_phase.png`；幅度误差为估计 CFR 幅度相对 truth CFR 幅度的 dB 差，相位误差为 wrap 到 `[-pi, pi]` 的相位差。
- `waveform_grid` 输出 `waveform_rx_grid.png`，轴为 `[subcarrier, OFDM symbol]`，颜色为接收 grid 的天线聚合功率。
- `path_samples` 只绘制当前采样选择中的第一个 UE-BS 链路，避免多个链路的路径几何叠到同一张 3D 图里；可通过多次指定不同 BS/UE 生成多链路对比。
- `spatial_spectrum` 按数据源分开输出 `spatial_spectrum_aoa_heatmap_label.png`、
  `spatial_spectrum_truth.png`、`spatial_spectrum_cfr_est.png`、
  `spatial_spectrum_observation.png`，标题中标明 AoA heatmap label、truth CFR Bartlett、estimated CFR Bartlett 或 RX grid Bartlett。
- `spatial_spectrum` 同时输出对应 `*_polar.png`：每个 link 的 polar 图左右并排，
  左侧上半球半径为 zenith，右侧下半球半径为 `pi - zenith`，两者外圈均表示
  zenith `90°` 的水平面；原始矩形图仍保留。
- 空间谱矩形图和 polar 图共享“同一个 UE 内的选中 BS”局部颜色尺度；polar 图不放
  colorbar，避免在多 UE/BS 采样图中遮挡子图。

pipeline 可视化只做少量采样示意图。独立 `visualize` CLI 的 `full` 模式表示
全量聚合统计，不逐 link 生成海量细节图。

#### `rt` — 射线追踪

| 字段 | 类型 | 默认值 | 约束 | 说明 |
|------|------|--------|------|------|
| `engine` | str | `"sionna_rt"` | — | 引擎 |
| `max_depth` | int | 1 | ≥0 | 最大交互深度 |
| `los` | bool | true | — | 直射路径 |
| `specular_reflection` | bool | true | — | 镜面反射 |
| `diffuse_reflection` | bool | false | — | 漫反射 |
| `refraction` | bool | false | — | 折射 |
| `diffraction` | bool | false | — | 绕射 |
| `synthetic_array` | bool | false | — | 合成阵列 |
| `normalize_cfr` | bool | false | — | 归一化 CFR |
| `normalize_delays` | bool | false | — | 归一化延迟 |
| `merge_shapes` | bool | false | — | 合并几何体 |

#### `link` — 链路配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `duplex_mode` | str | `"tdd"` | 双工模式 |
| `phy_link_direction` | str | `"uplink"` | PHY 链路方向 |

`phy_link_direction="uplink"` 时 resolved TX 为 UE、RX 为 BS；`downlink` 时 resolved
TX 为 BS、RX 为 UE。旧用户配置字段 `rt_trace_direction`、`reciprocity_mode`、
`reciprocity_applied` 已移除；HDF5 通过 `/link/tx_role` 和 `/link/rx_role` 记录解析结果。

#### `phy` — 物理层

**通用字段**（custom OFDM 和 NR PUSCH 共享）：

| 字段 | 类型 | 默认值 | 约束 | 说明 |
|------|------|--------|------|------|
| `enabled` | bool | true | — | 是否启用 PHY 观测 |
| `standard` | str | `"custom_ofdm"` | `"custom_ofdm"`\|`"nr_pusch"`\|`"nr_srs"` | 波形标准 |
| `snr_db` | float | 30.0 | — | 信噪比 dB |
| `fft_size` | int | 64 | ≥2 | FFT 大小 |
| `cp_length` | int | 0 | ≥0 | 循环前缀长度 |
| `num_ofdm_symbols` | int | 1 | ≥1 | OFDM 符号数 |
| `tx_power_dbm` | float | 0.0 | — | 发射功率 dBm |

**NR PUSCH 专有字段**：

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `subcarrier_spacing_khz` | int | 30 | 子载波间隔 kHz (15/30/60) |
| `num_prb` | int | 16 | PRB 数量 |
| `num_layers` | int | 1 | 每 UE 空间流数 |
| `num_antenna_ports` | int | 4 | 每 UE 天线端口数 |
| `mcs_index` | int | 14 | MCS 索引 (0-28) |
| `mcs_table` | int | 1 | MCS 表 (0=256QAM, 1=64QAM) |
| `perfect_csi` | bool | false | true 时 PUSCH receiver 使用 clean backend 返回的 oracle CSI；false 时使用 PUSCHReceiver 内部 DMRS LS |
| `ebno_db` | float\|null | null | Eb/N0 dB；非 null 优先于 snr_db |
| `pusch_dmrs_config_type` | int | 1 | DMRS 配置类型 |
| `pusch_dmrs_length` | int | 1 | DMRS 长度 |
| `pusch_dmrs_additional_position` | int | 1 | 附加 DMRS 位置 |
| `pusch_num_cdm_groups_without_data` | int | 2 | CDM 组数 |
| `mimo_mode` | str | `"su_mimo"` | `"su_mimo"`\|`"mu_mimo"` |
| `channel_backend` | str | `"apply_ofdm"` | `"apply_ofdm"`\|`"cir_dataset_ofdm"` |
| `mimo_detector` | str | `"lmmse"` | `"lmmse"`\|`"kbest"` |
| `channel_estimator` | str | `"pusch_ls"` | `"pusch_ls"`\|`"perfect"` |
| `receiver_failure_policy` | str | `"fail_fast"` | `"fail_fast"`\|`"mark_invalid"` |
| `su_mimo_link_batch_size` | int | 1 | SU-MIMO 独立 link batching；NR PUSCH 模板设为 64 |

**NR SRS subset 字段**：

`standard == "nr_srs"` 复用 `subcarrier_spacing_khz`、`num_prb`、`tx_power_dbm`
等字段，并通过 `phy.srs` 控制 resource。v2 支持 full-slot time allocation、
comb/BWP/hopping resource、`zc_like`/`nr_zc` pilot、group/sequence hopping、
同 symbol cyclic-shift port multiplexing、port/antenna switching 口径、简化
SRS power scaling、resource LS/despread 和 full-band interpolation。当前实现仍不是
完整 3GPP NR SRS，reference 对齐与闭环功控见 `docs/todo/feature.md`。

NR PUSCH 和 NR SRS 都通过 `common_link.py` 统一施加 CFO/SFO/timing/phase/
AGC/ADC 与 AWGN。普通 `snr_db` 下噪声方差按 clean `rx_grid` 每条 link 的平均功率
计算；仅当 PUSCH `ebno_db` 非空时使用 Sionna `ebnodb2no` 作为 override。

#### `impairments` — 损伤模型

每个子项有独立的 `enabled` 开关：

| 分组 | 字段 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `awgn` | `enabled` | bool | true | AWGN 噪声 |
| `cfo` | `enabled` | bool | true | 载波频偏开关 |
| `cfo` | `cfo_hz` | float\|null | 100.0 | 频偏值 Hz |
| `sfo` | `enabled` | bool | true | 采样频偏开关 |
| `sfo` | `sfo_ppm` | float\|null | 5.0 | 频偏值 ppm |
| `phase_noise` | `enabled` | bool | true | 相位偏移开关 |
| `phase_noise` | `phase_offset_rad` | float\|null | 0.5 | 偏移值 rad |
| `timing_offset` | `enabled` | bool | true | 定时偏移开关 |
| `timing_offset` | `timing_offset_samples` | float\|null | 2.0 | 偏移值（采样点） |
| `agc_adc` | `enabled` | bool | true | AGC/ADC 开关 |
| `agc_adc` | `agc_gain_db` | float | 0.0 | AGC 增益 dB |
| `agc_adc` | `clipping_threshold` | float\|null | 3.0 | ADC 削波阈值 |
| — | `impairment_seed` | int | 142 | 损伤随机种子 |

`null` 值表示不施加该损伤（即便 `enabled=true`）。损伤施加顺序：IFFT → CFO（时域）→ FFT → SFO → 相偏 → 定时偏 → AGC/ADC 削波。

#### `ranging` — 波形级 ToA/range 观测

`ranging` 在 PHY observation 之后运行，不属于某个标准模块私有逻辑。v1 只支持
`source: "cfr_est"`，即从 `/observation/cfr_est` 估计 ToA/range；开启时必须有
`phy.enabled=true`。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | false | 是否写 `/ranging` group |
| `source` | str | `"cfr_est"` | v1 仅支持该值 |
| `estimators` | list[str] | `["pdp_peak", "phase_slope"]` | 启用 estimator |
| `default_estimator` | str | `"pdp_peak"` | 下游默认 estimator |
| `write_rtt_equivalent` | bool | true | 写 `rtt_equiv_s=2*toa_est_s`；不是协议 RTT |
| `pdp_peak.oversampling_factor` | int | 8 | PDP IFFT zero-padding 倍数 |
| `pdp_peak.window` | str | `"hann"` | `hann` 或 `rect` |
| `pdp_peak.relative_threshold_db` | float | -12.0 | 相对最强峰的首径候选阈值 |
| `pdp_peak.min_peak_snr_db` | float | 6.0 | 峰值检测最小 SNR |
| `pdp_peak.interpolation` | str | `"parabolic_log_power"` | 峰值亚 bin 插值 |
| `pdp_peak.max_delay_s` | float\|null | null | 可选最大搜索 delay |
| `phase_slope.unwrap` | bool | true | phase 斜率拟合前是否 unwrap |
| `phase_slope.aggregate` | str | `"power_weighted_median"` | 天线 pair 聚合方式 |
| `phase_slope.min_mean_power` | float | 1.0e-12 | pair 级最小平均功率 |

#### `receiver` — 接收机

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `estimator_type` | str | `"ls"` | HDF5 记录的估计器类型 |
| `channel_estimator` | str | `"pusch_ls"` | NR PUSCH 信道估计器 |
| `sync_method` | str | `"ideal"` | 同步方法 |
| `interpolation_method` | str | `"none"` | 插值方法 |
| `packet_detection_threshold` | float | 0.0 | 包检测阈值 |
| `failure_policy` | str | `"mark_invalid"` | 失败处理策略 |
| `mimo_detector` | str | `"lmmse"` | MIMO 检测器 |
| `calibration_profile_id` | str | `"synthetic_default"` | 校准 profile |

#### `motion` — 运动与多普勒

| 字段 | 类型 | 默认值 | 约束 | 说明 |
|------|------|--------|------|------|
| `enabled` | bool | true | — | 是否启用 |
| `mobility_mode` | str | `"static"` | `"static"`\|`"doppler_synthetic"` | 移动模式 |
| `num_time_steps` | int | 3 | ≥1 | 时间快照数 |
| `sampling_frequency_hz` | float | 100.0 | — | 采样频率 Hz |
| `bs_velocity_mps` | [float×3] | [0,0,0] | 必须 3 分量 | BS 速度 m/s |
| `ue_velocity_mps` | [float×3] | [0,0,0] | 必须 3 分量 | UE 速度 m/s |

#### `calibration` — 校准

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | true | 是否写 `/calibration` group |
| `profile_id` | str | `"synthetic_default"` | 校准 profile ID |

### 1.4 Pydantic 校验规则

主要校验在 `model_validator` 中自动执行：

- `carrier.subcarrier_spacing_hz` 与 `bandwidth_hz / num_subcarriers` 一致性（1% 容差）
- `phy.fft_size >= 2`
- `ranging.enabled=true` 时必须 `phy.enabled=true`
- `motion.bs_velocity_mps` 和 `motion.ue_velocity_mps` 必须是 3 分量
- `motion.mobility_mode == "doppler_synthetic"` 时 `sampling_frequency_hz > 0`

### 1.5 配置模板

| 文件 | 用途 | 关键差异 |
|------|------|----------|
| `config/defaults/measurement_mvp.yaml` | custom OFDM + 全 impairment + motion | `standard: "custom_ofdm"`, fft_size=64, num_subcarriers=64 |
| `config/defaults/nr_pusch_mvp.yaml` | NR PUSCH 4x4 SU-MIMO TDD uplink | `standard: "nr_pusch"`, 4×4 天线, num_prb=4, num_subcarriers=48 |
| `config/defaults/nr_pusch_indoor_positioning_fr1_100mhz.yaml` | 室内 FR1 100 MHz PUSCH-DMRS 定位模板 | `standard: "nr_pusch"`, 273 PRB, shard size 5；已验证 6x5 probe |
| `config/defaults/nr_srs_indoor_positioning_fr1_100mhz.yaml` | 室内 FR1 100 MHz NR SRS subset 定位模板 | `standard: "nr_srs"`, direct uplink, `synthetic_array=false`, UE shard `shard_size=20`；空间谱/可视化默认关闭，按需显式开启 |
| `config/perf/nr_srs_7x500_sharded.yaml` | 室内 FR1 100 MHz NR SRS shard 历史确认测试 | `7 BS x 500 UE`, `shard_size=25` 的历史实验配置，验证 `result_xxx.h5` 和 aggregate manifest；当前生产模板使用 `20` |

### 1.6 输入数据格式

#### Label JSON（`input.label_file`）

```json
{
  "label_schema": "0.1.0",
  "scene_file": "scene.xml",
  "bs_points": [
    {"position": [2.0, 4.0, 2.4], "label": "BS0"},
    ...
  ],
  "ue_points": [
    {"x": 2.0, "y": 4.0, "z": 1.6, "label": "UE0"},
    ...
  ],
  "groups": [
    {
      "name": "room_or_region_name",
      "bs_points": [...],
      "ue_points": [...]
    }
  ]
}
```

当前标准 label 格式版本为 `0.1.0`。`label_schema`/`label_version` 目前只作为信息字段
保留，pipeline 不做强校验；解析器直接读取顶层 `bs_points` 和 `ue_points` 作为全场景
默认点集。`groups` 只表示房间、区域或生成策略等子集元数据，当前 pipeline 不根据
`groups` 选择局部场景，也没有 `label_group_policy`。

- 取顶层 `bs_points[:max_bs]` 和 `ue_points[:max_ue]`
- BS/UE 点坐标可以写成 `position: [x, y, z]`，也可以写成显式 `x/y/z`
- 坐标单位为米（场景本地坐标系）
- 解析器：`sionna_measurement_sim/io/label_parser.py`

标准 floorplan 文件名使用截断高度编码，例如 `floorplan_1p60.png` 表示
`z=1.60 m` 的平面图。对应 `floorplan/meta.json` 应提供 `origin_xy_m`、
`extent_xy_m`、`resolution_m_per_pixel`、`grid_shape` 和 `z_levels_m`，用于真实坐标
与图像像素之间转换。

#### Scene XML（`input.scene_file`）

Mitsuba 3 场景描述文件，引用 `.obj` 网格和材质：

```xml
<scene version="3.0.0">
  <bsdf type="diffuse" id="mat-itu_concrete">
    <rgb name="reflectance" value="0.75, 0.75, 0.75"/>
  </bsdf>
  <shape type="obj" id="scene_mesh">
    <string name="filename" value="scene.obj"/>
    <ref id="mat-itu_concrete"/>
  </shape>
</scene>
```

### 1.7 配置加载流程

```python
# config/loader.py
def load_config(path) -> MeasurementConfig       # YAML/JSON → Pydantic
def load_config_or_exit(path) -> MeasurementConfig  # 失败时打印错误并 sys.exit(1)
```

CLI 中 `--config <path> run-full` 加载 YAML 后，可用 CLI 参数覆盖部分字段（如 `--snr-db`、`--phy-standard`、`--output-dir`）。

### 1.8 MIMO 配置速查

| 场景 | 关键配置 |
|------|----------|
| 4x4 SU-MIMO perfect CSI | `mimo_mode="su_mimo"`, `num_ant=4`, `num_layers=4`, `perfect_csi=true`, `channel_estimator="perfect"` |
| 4x4 SU-MIMO estimated CSI | `perfect_csi=false`, `num_layers=4`, `num_antenna_ports=4` (必须等秩) |
| MU-MIMO | `mimo_mode="mu_mimo"`, `max_ue >= 2`, `dmrs_port_set` 自动不重叠 |

MIMO backend 支持矩阵：

| mimo_mode | apply_ofdm | cir_dataset_ofdm |
|-----------|-----------|-----------------|
| `su_mimo` | 支持 | 支持 (per-link) |
| `mu_mimo` | 支持 | 不支持 (入口拒绝) |

### 1.9 禁止事项

- 配置加载失败后不进行任何 RT/PHY 运算
- 不在代码中硬编码配置默认值（应使用 Pydantic Field default）
- NR PUSCH 场景 `num_subcarriers` 必须等于 `num_prb × 12`

---

## 二、HDF5 数据格式

### 2.1 设计原则

- **契约驱动**：所有 HDF5 输出必须通过 `schema_validator.py` 校验
- **纯 Python 写入**：writer 只接受 domain 对象，不导入 Sionna
- **自描述**：每个 dataset 标注 `unit` 和 `index_order` attribute
- **大数组压缩**：ndim > 0 且 size > 0 的数组按 `output.compression` 启用 `gzip`/`lzf` + shuffle，也可配置为 `none`

### 2.2 Group 层级总览

```
results.h5
├── /meta                          # 元数据
├── /shard                         # shard 模式下的全局索引映射
├── /input                         # 输入引用
├── /topology                      # TX/RX 位置与标签
├── /devices                       # 设备状态（速度、朝向）
├── /antenna                       # 天线阵列规格
├── /scene                         # 场景引用
├── /frequency                     # 频率网格
├── /derived                       # 几何距离、first-path truth delay/range、AoA、LoS/NLoS
├── /channel
│   └── /truth
│       ├── cfr                    # CFR 真值 [tx, rx, rx_ant, tx_ant, subcarrier]
│       ├── cfr_snapshots          # 多快照 CFR [snap, tx, rx, rx_ant, tx_ant, subcarrier]
│       ├── cir_coefficients       # CIR 系数 [snap, tx, rx, rx_ant, tx_ant, path]
│       ├── cir_delays_s           # CIR 延迟 [snap, tx, rx, rx_ant, tx_ant, path]
│       ├── cir_valid              # CIR 有效性 [snap, tx, rx, rx_ant, tx_ant, path]
│       └── ... (scalar diagnostics)
├── /paths
│   ├── /samples                   # 采样路径（可视化）
│   └── /full                      # 全量路径表（save_full_paths=true）
├── /link                          # 链路配置
├── /waveform                      # 波形参数（PHY 启用时）
├── /array                         # NR PUSCH 阵列观测和 AoA/空间谱标签
├── /observation                   # 观测 CFR（PHY 启用时）
├── /ranging                       # 从 cfr_est 估计的 ToA/range observation（可选）
├── /impairments                   # 损伤配置（PHY 启用时）
├── /receiver                      # 接收机配置（PHY 启用时）
├── /evaluation                    # 评估指标（PHY 启用时）
├── /calibration                   # 校准结果
├── /motion                        # 运动/多普勒参数
└── /runtime                       # 运行环境/版本/耗时
```

### 2.3 `/meta` — 元数据

| Dataset | 类型 | 说明 |
|---------|------|------|
| `schema_version` | string | Schema 版本号，当前为 `1.5.0` |
| `contract_name` | string | 契约名称 |
| `producer` | string | 生成器标识 |
| `created_at` | string | 创建时间戳 |
| `run_id` | string | 运行 ID |
| `git_commit` | string | Git commit hash |
| `random_seed` | int64 | 全局随机种子 |
| `coordinate_system` | string | 坐标系统 |
| `unit_convention` | string | 单位约定 |
| `index_order` | string | 维度顺序约定 |
| `truth_branch_enabled` | bool | RT 真值分支是否启用 |
| `observation_branch_enabled` | bool | PHY 观测分支是否启用 |
| `measurement_realism_level` | string | 测量真实度等级 |
| `config_snapshot` | string | 完整 YAML 配置快照 |
| `software_versions` | string | 软件版本摘要 |

### 2.4 `/shard` — Shard 元数据

普通单文件运行不写 `/shard`。开启 `output.sharding.enabled=true` 时，每个 `results/result_xxx.h5` 自包含局部数据，并写入以下映射：

| Dataset | 类型/Shape | 说明 |
|---------|------------|------|
| `shard_index` | int32 | 当前 shard 序号 |
| `shard_count` | int32 | 总 shard 数 |
| `axis` | string | 实际 shard 维度，当前为 `"ue"` |
| `global_rx_start` | int64 | 当前 shard 第一个 resolved RX 全局索引 |
| `global_rx_indices` | [local_rx] int64 | 本文件局部 RX 对应的全局角色索引；配合 `/link/rx_role` 解释 |
| `global_tx_indices` | [local_tx] int64 | 本文件局部 TX 对应的全局角色索引；配合 `/link/tx_role` 解释 |

`manifest/manifest.json` 会汇总每个 shard 的文件路径、全局索引覆盖范围、可视化摘要和性能日志路径。`manifest/config_snapshot.json` 保存 resolved 运行配置，便于数据目录脱离外部临时 YAML 后仍可复现。发生 fallback 时，`manifest/shard_attempts.jsonl` 保存失败原因与拆分链路。

### 2.5 `/input` — 输入引用

| Dataset | 类型 | 说明 |
|---------|------|------|
| `label_file` | string | 标签文件路径 |
| `scene_file` | string | 场景文件路径 |
| `input_dataset_id` | string | 数据集标识；标准 `label/` 子目录布局下取场景目录路径 |
| `input_schema` | string | 标签 schema 版本，当前写 `standard_label_0.1.0` |

### 2.6 `/topology` — 拓扑

| Dataset | Shape | Unit | 说明 |
|---------|-------|------|------|
| `tx_positions_m` | [num_tx, 3] | m | TX 三维位置 float32 |
| `rx_positions_m` | [num_rx, 3] | m | RX 三维位置 float32 |
| `tx_labels` | [num_tx] | — | TX 标签 string |
| `rx_labels` | [num_rx] | — | RX 标签 string |

### 2.7 `/devices` — 设备状态

| Dataset | Shape | Unit | 说明 |
|---------|-------|------|------|
| `tx_velocity_mps` | [num_tx, 3] | m/s | TX 速度向量 float32 |
| `rx_velocity_mps` | [num_rx, 3] | m/s | RX 速度向量 float32 |
| `tx_orientation_rad` | [num_tx, 3] | rad | TX 朝向角 float32 |
| `rx_orientation_rad` | [num_rx, 3] | rad | RX 朝向角 float32 |

### 2.8 `/antenna` — 天线

| Dataset | 类型 | 说明 |
|---------|------|------|
| `tx_array_type` | string | TX 阵列类型 |
| `rx_array_type` | string | RX 阵列类型 |
| `tx_num_rows` | int32 | TX 行数 |
| `tx_num_cols` | int32 | TX 列数 |
| `rx_num_rows` | int32 | RX 行数 |
| `rx_num_cols` | int32 | RX 列数 |
| `tx_num_ant` | int32 | TX 总天线数 (rows×cols) |
| `rx_num_ant` | int32 | RX 总天线数 |
| `tx_spacing_lambda` | [2] float | TX 间距（波长） |
| `rx_spacing_lambda` | [2] float | RX 间距（波长） |
| `tx_polarization` | string | TX 极化 (V/H) |
| `rx_polarization` | string | RX 极化 |
| `tx_pattern` | string | TX 方向图 |
| `rx_pattern` | string | RX 方向图 |
| `synthetic_array` | bool | 合成阵列 |
| `tx_orientation_mode` | string | TX 朝向模式 |
| `rx_orientation_mode` | string | RX 朝向模式 |

### 2.9 `/scene` — 场景

| Dataset | 类型 | 说明 |
|---------|------|------|
| `scene_name` | string | 场景名称 |
| `scene_file` | string | 场景文件路径 |
| `scene_id` | string | 跨仿真/平面图系统对齐 ID |
| `map_id` | string | 可选地图版本 ID |
| `material_policy` | string | 材质策略 |

### 2.10 `/frequency` — 频率网格

| Dataset | 类型 | Unit | 说明 |
|---------|------|------|------|
| `center_frequency_hz` | float64 | — | 中心频率 |
| `bandwidth_hz` | float64 | — | 带宽 |
| `num_subcarriers` | int32 | — | 子载波数 |
| `subcarrier_spacing_hz` | float64 | — | 子载波间隔 |
| `frequencies_hz` | [num_subcarriers] float64 | Hz | 子载波频率数组（严格递增） |

### 2.11 `/channel/truth` — 信道真值

#### 核心 datasets

| Dataset | Shape | Dtype | Unit | Index Order |
|---------|-------|-------|------|-------------|
| `cfr` | [tx, rx, rx_ant, tx_ant, subcarrier] | complex64 | `linear_complex` | `tx,rx,rx_ant,tx_ant,subcarrier` |
| `cfr_snapshots` | [snap, tx, rx, rx_ant, tx_ant, subcarrier] | complex64 | `linear_complex` | `snapshot,tx,rx,rx_ant,tx_ant,subcarrier` |
| `cir_coefficients` | [snap, tx, rx, rx_ant, tx_ant, path] | complex64 | `linear_complex` | — |
| `cir_delays_s` | [snap, tx, rx, rx_ant, tx_ant, path] | float32 | `s` | — |
| `cir_valid` | [snap, tx, rx, rx_ant, tx_ant, path] | bool | — | — |

> `cfr_snapshots` 仅 motion 启用时存在，其 `shape[1:] == cfr.shape`。

#### 辅助 datasets

| Dataset | Shape | Dtype | Unit | 说明 |
|---------|-------|-------|------|------|
| `path_power_db` | [tx, rx] | float32 | dB | per-link 路径功率 |
| `has_geometric_signal` | [tx, rx] | bool | — | 是否有几何路径信号 |
| `los_exists` | [tx, rx] | bool | — | 是否存在 LoS |
| `nlos_exists` | [tx, rx] | bool | — | 是否存在 NLoS |
| `geometric_path_count` | [tx, rx] | int32 | — | 几何路径数 |

### 2.12 `/derived` — 派生物理标签

这些字段始终写入，供定位 baseline 和数据转换器复用统一口径。`/paths/full` 是否落盘仍由 `output.save_full_paths` 控制，但派生标签在 pipeline 内部使用完整路径表计算。

| Dataset | Shape | Unit | 说明 |
|---------|-------|------|------|
| `geometric_distance_m` | [tx, rx] | m | TX/RX 三维欧氏距离 |
| `los_distance_m` | [tx, rx] | m | LoS 路径传播距离；无 LoS 为 NaN |
| `first_path_delay_s` | [tx, rx] | s | 最早有效路径 delay；无路径为 NaN |
| `first_path_propagation_range_m` | [tx, rx] | m | `first_path_delay_s * c`，最早路径 truth propagation range |
| `strongest_path_delay_s` | [tx, rx] | s | 最强路径 delay；无路径为 NaN |
| `los_aoa_azimuth_rad` / `los_aoa_zenith_rad` | [tx, rx] | rad | LoS 到达角；无 LoS 为 NaN |
| `strongest_aoa_azimuth_rad` / `strongest_aoa_zenith_rad` | [tx, rx] | rad | 最强路径 AoA |
| `first_path_aoa_azimuth_rad` / `first_path_aoa_zenith_rad` | [tx, rx] | rad | 最早路径 AoA |
| `los_flag` / `nlos_flag` | [tx, rx] | — | LoS/NLoS 标记 |
| `path_count` | [tx, rx] | — | 有效几何路径数 |
| `path_power_db` | [tx, rx] | dB | 总路径功率 |
| `link_valid_mask` | [tx, rx] | — | 是否有有效几何链路 |
| `tx_rx_midpoint_m` | [tx, rx, 2] | m | TX/RX 的 XY 平面中点 |
| `tx_rx_bearing_rad` | [tx, rx] | rad | TX→RX 的 XY 方位角 |
| `tx_rx_distance_m` | [tx, rx] | m | TX/RX 的 XY 平面距离 |
| `path_selection_policy` | scalar string | — | first/strongest/LoS 路径选择口径 |

### 2.13 `/paths/samples` — 采样路径

轻量级路径采样，用于可视化和快速分析。

| Dataset | Shape | Dtype | Unit | 说明 |
|---------|-------|-------|------|------|
| `sampled_link_indices` | [sample, 2] | int64 | — | 采样的 link (tx, rx) 索引 |
| `sampled_rx_ant_indices` | [sample] | int64 | — | 采样的 RX 天线索引 |
| `sampled_tx_ant_indices` | [sample] | int64 | — | 采样的 TX 天线索引 |
| `sampled_path_indices` | [sample, sample_path] | int64 | — | 采样落地路径索引 |
| `path_count` | [sample] | int64 | — | 每个采样点的路径数 |
| `path_gain_db` | [sample, sample_path] | float32 | dB | 路径增益 |
| `path_type` | [sample, sample_path] | string | — | `"LoS"` / `"NLoS"` |
| `vertices_m` | [sample, sample_path, max_vertices, 3] | float32 | m | 路径交互点坐标（含 TX/RX 端点） |
| `vertex_count` | [sample, sample_path] | int64 | — | 每个路径的顶点数 |
| `interaction_type` | [sample, sample_path, max_depth] | uint32 | — | 每级交互类型 |
| `object_id` | [sample, sample_path, max_depth] | uint32 | — | 命中物体 ID |
| `primitive_id` | [sample, sample_path, max_depth] | uint32 | — | 命中图元 ID |
| `doppler_hz` | [sample, sample_path] | float32 | Hz | 多普勒频移 |
| `tau_s` | [sample, sample_path] | float32 | s | 路径延迟 |

约束：`vertex_count >= interaction_type_nonzero_count + 2`（包含 TX/RX 端点）。

### 2.14 `/paths/full` — 全量路径表

仅当 `output.save_full_paths = true` 时写入。与 `PathTable` domain 模型一一对应：

| Dataset | Shape | Dtype | Unit | 说明 |
|---------|-------|-------|------|------|
| `valid` | [tx, rx, rx_ant, tx_ant, path] | bool | — | 路径有效性 |
| `a` | [tx, rx, rx_ant, tx_ant, path] | complex64 | `linear_complex` | 路径系数 |
| `tau_s` | [tx, rx, rx_ant, tx_ant, path] | float32 | s | 延迟 |
| `doppler_hz` | [tx, rx, rx_ant, tx_ant, path] | float32 | Hz | 多普勒频移 |
| `theta_t_rad` | [tx, rx, rx_ant, tx_ant, path] | float32 | rad | AoD 天顶角 |
| `phi_t_rad` | [tx, rx, rx_ant, tx_ant, path] | float32 | rad | AoD 方位角 |
| `theta_r_rad` | [tx, rx, rx_ant, tx_ant, path] | float32 | rad | AoA 天顶角 |
| `phi_r_rad` | [tx, rx, rx_ant, tx_ant, path] | float32 | rad | AoA 方位角 |
| `interaction_type` | [tx, rx, rx_ant, tx_ant, path, depth] | uint32 | — | 交互类型 |
| `object_id` | [tx, rx, rx_ant, tx_ant, path, depth] | uint32 | — | 命中物体 ID |
| `primitive_id` | [tx, rx, rx_ant, tx_ant, path, depth] | uint32 | — | 命中图元 ID |
| `vertices_m` | [tx, rx, rx_ant, tx_ant, path, depth, 3] | float32 | m | 交互点坐标 |
| `path_type` | [tx, rx, rx_ant, tx_ant, path] | string | — | 路径类型 |
| `path_depth` | [tx, rx, rx_ant, tx_ant, path] | int32 | — | 有效交互数 |

### 2.15 `/paths/nlos_truth` — NLoS 路径 AoA/AoD 真值

始终写入，独立于 `output.save_full_paths`。它是面向空间谱/多径监督的轻量路径真值，
只保留 NLoS path；LoS 或 invalid 位置的角度、功率、延迟写 `NaN`，`path_type` 写
`"invalid"`。

| Dataset | Shape | Dtype | Unit | 说明 |
|---------|-------|-------|------|------|
| `valid` | [tx, rx, rx_ant, tx_ant, path] | bool | — | `PathTable.valid & path_type != "los"` |
| `aoa_zenith_rad` | [tx, rx, rx_ant, tx_ant, path] | float32 | rad | NLoS AoA 天顶角 |
| `aoa_azimuth_rad` | [tx, rx, rx_ant, tx_ant, path] | float32 | rad | NLoS AoA 方位角 |
| `aod_zenith_rad` | [tx, rx, rx_ant, tx_ant, path] | float32 | rad | NLoS AoD 天顶角 |
| `aod_azimuth_rad` | [tx, rx, rx_ant, tx_ant, path] | float32 | rad | NLoS AoD 方位角 |
| `path_power_db` | [tx, rx, rx_ant, tx_ant, path] | float32 | dB | `10*log10(abs(a)^2)` |
| `delay_s` | [tx, rx, rx_ant, tx_ant, path] | float32 | s | 路径延迟 |
| `path_depth` | [tx, rx, rx_ant, tx_ant, path] | int32 | — | 有效交互数 |
| `path_type` | [tx, rx, rx_ant, tx_ant, path] | string | — | NLoS 类型或 `"invalid"` |

### 2.16 `/link` — 链路配置

| Dataset | 类型 | 说明 |
|---------|------|------|
| `duplex_mode` | string | 双工模式 (tdd/fdd) |
| `phy_link_direction` | string | PHY 方向 (uplink/downlink) |
| `tx_role` | string | resolved TX 角色，`"ue"` 或 `"bs"` |
| `rx_role` | string | resolved RX 角色，`"ue"` 或 `"bs"` |

### 2.17 `/waveform` — 波形参数（PHY 启用时）

| Dataset | 类型 | 说明 |
|---------|------|------|
| `standard` | string | `"custom_ofdm"` / `"nr_pusch"` / `"nr_srs"` |
| `sample_rate_hz` | float64 | 采样率 Hz |
| `fft_size` | int32 | FFT 大小 |
| `cp_length` | int32 | 循环前缀长度 |
| `num_ofdm_symbols` | int32 | OFDM 符号数 |
| `pilot_indices` | int array | 导频子载波索引 |
| `data_subcarrier_indices` | int array | 数据子载波索引 |
| `pilot_symbols` | complex array | 导频符号 (unit: `linear_complex`) |
| `tx_power_dbm` | float32 | 发射功率 dBm |

**NR PUSCH 专有**（仅 `standard == "nr_pusch"`）：

| Dataset | 类型 | 说明 |
|---------|------|------|
| `num_prb` | int32 | PRB 数量 ≥1 |
| `subcarrier_spacing_khz` | int32 | 子载波间隔 kHz |
| `num_layers` | int32 | 空间流数 ≥1 |
| `num_antenna_ports` | int32 | 天线端口数 ≥ num_layers |
| `mcs_index` | int32 | MCS 索引 |
| `mcs_table` | int32 | MCS 表 (0/1) |
| `dmrs_config_type` | int32 | DMRS 配置类型 |
| `dmrs_length` | int32 | DMRS 长度 |
| `dmrs_additional_position` | int32 | 附加 DMRS 位置 |
| `num_cdm_groups_without_data` | int32 | CDM 组数 |
| `subcarrier_spacing_hz` | float64 | 子载波间隔 Hz |
| `slot_number` | int32 | 时隙号 |
| `cyclic_prefix` | string | 循环前缀类型 |
| `target_coderate` | string | 目标码率 |
| `modulation` | string | 调制方式 |
| `tx_grid` | complex64 [snap, ul_tx, ul_rx, ul_tx_ant, ofdm_symbol, subcarrier] | 实际 NR PUSCH 频域发送 grid |
| `rx_grid` | complex64 [snap, ul_tx, ul_rx, ul_rx_ant, ofdm_symbol, subcarrier] | 实际 NR PUSCH 频域接收 grid |
| `noise_variance` | float32 [snap, ul_tx, ul_rx] | common impairment chain 添加 AWGN 时使用的噪声方差 |

**NR SRS subset 使用的统一 waveform 字段**（仅 `standard == "nr_srs"`）：

| Dataset | 类型 | 说明 |
|---------|------|------|
| `tx_grid` | complex64 [snap, ul_tx, ul_rx, ul_tx_ant, ofdm_symbol, subcarrier] | 完整 slot 发送 grid；非 SRS symbol/subcarrier 为 0 |
| `rx_grid` | complex64 [snap, ul_tx, ul_rx, ul_rx_ant, ofdm_symbol, subcarrier] | clean channel + common impairment/AWGN 后的接收 grid |
| `noise_variance` | float32 [snap, ul_tx, ul_rx] | common impairment chain 添加 AWGN 时使用的噪声方差 |
| `srs_resource_mask` | bool [ofdm_symbol, subcarrier] | SRS resource RE mask |
| `srs_pilot_symbols` | complex64 [srs_port, ofdm_symbol, subcarrier] | 每个 SRS port 的 pilot 符号，非 SRS RE 为 0 |
| `srs_re_symbol_indices` | int32 [srs_re] | flattened SRS RE 的 OFDM symbol 索引 |
| `srs_re_subcarrier_indices` | int32 [srs_re] | flattened SRS RE 的子载波索引 |
| `srs_symbol_indices` | int32 [srs_symbol] | SRS symbol 索引 |
| `srs_port_tx_ant_map` | int32 [srs_port, srs_symbol] | 每个 SRS port 在每个 SRS symbol 使用的 UE TX antenna；`-1` 表示 inactive |
| `srs_prb_start_per_symbol` | int32 [srs_symbol] | 每个 SRS symbol 的 PRB 起点 |
| `srs_prb_count_per_symbol` | int32 [srs_symbol] | 每个 SRS symbol 的 PRB 数 |
| `srs_cyclic_shift_indices` | int32 [srs_port] | port-specific cyclic shift 配置 |
| `srs_tx_power_dbm` | float32 [snap, ul_tx, srs_port] | SRS open-loop power-control 后的发射功率 |
| `srs_power_scale_linear` | float32 [snap, ul_tx, srs_port] | 相对 `phy.tx_power_dbm` 的发射幅度缩放 |

不保存 `/waveform/tx_time` 或 `/waveform/rx_time`；custom OFDM 暂不写 fake grid，后续另行适配。
schema `1.5.0` 后 NR SRS 不再写 `/waveform/pilot_code`、`/waveform/srs_tx_grid`、
`/waveform/srs_port_index`、`/observation/srs_cfr_est`、`/array/spatial_spectrum_srs`
或 `/array/spatial_spectrum_label`。resource LS/despread 写到
`/observation/cfr_est_resource [snapshot,tx,rx,rx_ant,srs_port,srs_re]`，full-band
插值结果仍写 `/observation/cfr_est [snapshot,tx,rx,rx_ant,tx_ant,subcarrier]`。

大规模 NR PUSCH/SRS 输出建议按 shard 生成：开启 `output.sharding.enabled=true` 后，`run-full` 会按 UE/RX 范围直接写多个 `results/result_xxx.h5`，并由 `manifest/manifest.json` 汇总全局索引和每个 shard 的 schema/debug 信息。NR PUSCH 的 `6 BS × 8884 UE × 4x4` 已通过 4 GPU shard + batch64 全量验收；NR SRS direct uplink 模板默认 `shard_size=20`，已完成 `median_0000 label0p2` 的 `7 BS × 2583 UE` baseline。下游训练或分析应优先通过 manifest 按 shard 读取，而不是假设只有单个 `results.h5`，也不要假设 `result_xxx.h5` 文件名严格连续。

### 2.18 `/array` — 阵列观测与标签

| Dataset | Shape | Unit | 说明 |
|---------|-------|------|------|
| `rx_snapshot_matrix` | [snap, ul_tx, ul_rx, ul_rx_ant, ul_rx_ant] | linear_complex | 由 NR PUSCH/SRS `rx_grid` 聚合的接收阵列协方差/快照矩阵 |
| `aoa_label_rad` | [snap, ul_tx, ul_rx, 2] | rad | `[zenith, azimuth]` scene/global PHY 接收侧 AoA 标签；direct uplink 中 BS 是 RT receiver，因此使用 receiver-side AoA；只有 legacy reverse fallback 才会用原 RT AoD |
| `aoa_heatmap_label` | [snap, ul_tx, ul_rx, zenith_bins, azimuth_bins] | linear | 从真值 AoA 画出的监督 heatmap |
| `spatial_spectrum_truth` | [snap, ul_tx, ul_rx, zenith_bins, azimuth_bins] | linear | `array.spectrum.enabled=true` 且 source 包含 `truth_cfr` 时写入，由 truth CFR Bartlett 在 scene/global 角度网格上扫描得到 |
| `spatial_spectrum_cfr_est` | [snap, ul_tx, ul_rx, zenith_bins, azimuth_bins] | linear | `array.spectrum.enabled=true` 且 source 包含 `cfr_est` 时写入，由 `/observation/cfr_est` Bartlett 扫描得到 |
| `spatial_spectrum_observation` | [snap, ul_tx, ul_rx, zenith_bins, azimuth_bins] | linear | NR PUSCH 且 source 包含 `rx_grid` 时写入，由实际接收 grid Bartlett 扫描得到 |
| `angle_grid_rad` | [zenith_bins, azimuth_bins, 2] | rad | scene/global 角度网格；默认 zenith `[0, pi]`，azimuth `[-pi, pi]`，可配置分辨率 |
| `spectrum_policy` | scalar string | — | 记录 method、source、角度范围、归一化与聚合口径 |

默认即使 `array.spectrum.enabled=false`，NR PUSCH/SRS 仍写 `aoa_heatmap_label`
作为轻量监督标签；真实 Bartlett 谱只在显式开启时写入。schema 1.5.0 的迁移：
旧 `/array/spatial_spectrum_label` 改用 `/array/aoa_heatmap_label`；旧
`array.spectrum.sources: ["srs_cfr_est"]` 和 `/array/spatial_spectrum_srs`
改用 `cfr_est` source 与 `/array/spatial_spectrum_cfr_est`。

### 2.19 `/observation` — 观测结果（PHY 启用时）

| Dataset | Shape | Dtype | Unit | Index Order |
|---------|-------|-------|------|-------------|
| `cfr_est` | [snap, tx, rx, rx_ant, tx_ant, subcarrier] | complex64 | `linear_complex` | `snapshot,tx,rx,rx_ant,tx_ant,subcarrier` |
| `valid_mask` | [snap, tx, rx] | bool | — | 快照有效性 |
| `detection_success` | [snap, tx, rx] | bool | — | 检测成功标志 |
| `estimation_success` | [snap, tx, rx] | bool | — | 估计成功标志 |
| `snr_db` | [snap, tx, rx] | float | dB | SNR |
| `rssi_dbm` | [snap, tx, rx] | float | dBm | RSSI |
| `noise_power_dbm` | [snap, tx, rx] | float | dBm | 噪声功率 |
| `cfo_hz` | [snap, tx, rx] | float | Hz | 载波频偏 |
| `sfo_ppm` | [snap, tx, rx] | float | ppm | 采样频偏 |
| `timing_offset_samples` | [snap, tx, rx] | float | sample | 定时偏移 |
| `phase_offset_rad` | [snap, tx, rx] | float | rad | 相位偏移 |
| `agc_gain_db` | [snap, rx] | float | dB | 接收侧 AGC 增益 |
| `clipping_flag` | [snap, tx, rx] | bool | — | ADC 削波标志 |

> `cfr_est.shape[-5:] == truth.cfr.shape`，即观测 CFR 的 TX/RX/天线/子载波维度与真值一致，仅在前面多一维 snapshot。schema `1.5.0` 后 NR SRS 不再写 `/observation/srs_cfr_est`，统一使用 `/observation/cfr_est`，resource LS/despread 另写 `/observation/cfr_est_resource`。

### 2.20 `/ranging` — 波形级 ToA/range 观测（可选）

仅当 `ranging.enabled=true` 时写入。`/ranging` 是 observation，不覆盖 `/derived`
truth。当前 v1 从 `/observation/cfr_est` 估计上行波形 ToA/one-way range：

| Dataset | Shape/类型 | Unit | 说明 |
|---------|------------|------|------|
| `default_estimator` | scalar string | — | 默认 estimator 名称 |
| `pdp_peak/toa_est_s` | [snap, tx, rx] float32 | s | PDP 峰值 ToA 估计；失败为 NaN |
| `pdp_peak/one_way_range_est_m` | [snap, tx, rx] float32 | m | `toa_est_s * c`；失败为 NaN |
| `pdp_peak/rtt_equiv_s` | [snap, tx, rx] float32 | s | `2 * toa_est_s`，two-way equivalent，不是协议 RTT |
| `pdp_peak/range_error_m` | [snap, tx, rx] float32 | m | `one_way_range_est_m - /derived/first_path_propagation_range_m` |
| `pdp_peak/detection_success` | [snap, tx, rx] bool | — | PDP 峰值是否通过检测门限 |
| `pdp_peak/selected_delay_bin` | [snap, tx, rx] int32 | — | 选中 delay bin；失败为 -1 |
| `pdp_peak/peak_power_linear` | [snap, tx, rx] float32 | linear | 选中峰功率 |
| `pdp_peak/peak_snr_db` | [snap, tx, rx] float32 | dB | 峰值相对噪底 SNR |
| `phase_slope/toa_est_s` | [snap, tx, rx] float32 | s | phase-slope ToA 估计；失败为 NaN |
| `phase_slope/one_way_range_est_m` | [snap, tx, rx] float32 | m | `toa_est_s * c` |
| `phase_slope/rtt_equiv_s` | [snap, tx, rx] float32 | s | two-way equivalent |
| `phase_slope/range_error_m` | [snap, tx, rx] float32 | m | 相对 first-path truth range 的误差 |
| `phase_slope/detection_success` | [snap, tx, rx] bool | — | 是否得到有效相位斜率拟合 |
| `phase_slope/fit_residual_rad` | [snap, tx, rx] float32 | rad | 加权线性拟合残差 |

schema 校验要求成功位置的 range/toa/error 为 finite，失败位置为 NaN。

### 2.21 `/impairments` — 损伤配置（PHY 启用时）

| Dataset | 类型 | 说明 |
|---------|------|------|
| `model_version` | string | 损伤模型版本 |
| `random_seed` | int64 | 损伤随机种子 |
| `awgn_config` | string | AWGN 配置摘要 |
| `cfo_sfo_config` | string | CFO/SFO 配置摘要 |
| `phase_noise_config` | string | 相位噪声配置摘要 |
| `iq_imbalance_config` | string | IQ 不平衡配置摘要 |
| `agc_adc_config` | string | AGC/ADC 配置摘要 |

### 2.22 `/receiver` — 接收机配置（PHY 启用时）

| Dataset | 类型 | 说明 |
|---------|------|------|
| `receiver_type` | string | 接收机类型 (`"pusch_receiver"` / `"srs_ls_receiver"` / `"generic"`) |
| `estimator_type` | string | 估计器类型 |
| `sync_method` | string | 同步方法 |
| `mimo_detector` | string | MIMO 检测器 (`"lmmse"` / `"kbest"`；SRS 为 `"none"`) |
| `input_domain` | string | 输入域 |
| `interpolation_method` | string | 插值方法 |
| `packet_detection_threshold` | float32 | 包检测阈值 |
| `failure_policy` | string | 失败策略 |
| `calibration_profile_id` | string | 校准 profile ID |

> NR PUSCH 场景下 `receiver_type` 必须是 `"pusch_receiver"`，`mimo_detector` 必须是 `"lmmse"` 或 `"kbest"`。

### 2.23 `/evaluation` — 评估指标（PHY 启用时）

| Dataset | Shape | Dtype | Unit | 说明 |
|---------|-------|-------|------|------|
| `nmse_db` | [snap, tx, rx] | float | dB | 主指标: NMSE(H_est, H_true) |
| `nmse_db_total` | [snap, tx, rx] | float | dB | 诊断指标: NMSE vs impaired 信道 |
| `amplitude_error_db` | [snap, tx, rx] | float | dB | 幅度误差 |
| `phase_error_rad` | [snap, tx, rx] | float | rad | 相位误差 |
| `correlation` | [snap, tx, rx] | float | — | 相关系数 |
| `detection_rate` | scalar | float32 | — | 检测成功率 |
| `estimation_failure_rate` | scalar | float32 | — | 估计失败率 |
| `ber` | scalar | float32 | — | 误比特率 (NR PUSCH) |
| `bler` | scalar | float32 | — | 误块率 TB CRC (NR PUSCH) |
| `num_bit_errors` | scalar | int64 | — | 比特错误数 |
| `num_bits` | scalar | int64 | — | 总比特数 |
| `num_block_errors` | scalar | int64 | — | TB CRC 错误块数 |
| `num_blocks` | scalar | int64 | — | 总块数 |

> `nmse_db` 计算公式：`10*log10(||H_est - H_true||² / ||H_true||²)`，越小越好。

**NR PUSCH BLER 契约**（仅 `standard == "nr_pusch"` 时严格校验）：
- `num_blocks > 0`
- `0 <= num_block_errors <= num_blocks`
- `bler == num_block_errors / num_blocks`

### 2.24 `/calibration` — 校准

| Dataset | 类型 | 说明 |
|---------|------|------|
| `profile_id` | string | 校准 profile ID |
| `fitted_parameters` | string | 拟合参数 |
| `validation_metrics` | string | 校验指标 |

### 2.24 `/motion` — 运动/多普勒

| Dataset | Shape | Unit | 说明 |
|---------|-------|------|------|
| `snapshot_id` | [num_time_steps] int64 | — | 快照序号 |
| `timestamp_s` | [num_time_steps] float64 | s | 时间戳 |
| `sampling_frequency_hz` | scalar float64 | — | 采样频率 |
| `num_time_steps` | scalar int32 | — | 快照数 |
| `mobility_mode` | scalar string | — | `"static"` / `"doppler_synthetic"` |

### 2.25 `/runtime` — 运行环境

| Dataset | 类型 | 说明 |
|---------|------|------|
| `python_version` | string | Python 版本 |
| `sionna_version` | string | Sionna 版本 |
| `sionna_rt_version` | string | Sionna RT 版本 |
| `torch_version` | string | PyTorch 版本 |
| `mitsuba_version` | string | Mitsuba 版本 |
| `drjit_version` | string | Dr.Jit 版本 |
| `cuda_available` | bool | CUDA 可用性 |
| `cuda_device_name` | string | CUDA 设备名 |
| `command_line` | string | 完整命令行 |
| `elapsed_seconds` | float64 | 运行耗时（秒） |

### 2.26 核心维度约定

项目内部所有张量采用 **TX-first** 维度顺序（Sionna 原生为 rx-first，adapter 负责转换）：

| 数据 | Shape | 索引顺序 |
|------|-------|----------|
| Truth CFR | `[tx, rx, rx_ant, tx_ant, subcarrier]` | 5-D |
| Truth CFR snapshots | `[snap, tx, rx, rx_ant, tx_ant, subcarrier]` | 6-D |
| Obs CFR | `[snap, tx, rx, rx_ant, tx_ant, subcarrier]` | 6-D |
| CIR | `[snap, tx, rx, rx_ant, tx_ant, path]` | 6-D |
| Path scalars (full) | `[tx, rx, rx_ant, tx_ant, path]` | 5-D |
| Path scalars (samples) | `[sample, sample_path, ...]` | 2-D+ |
| per-link scalars | `[snap, tx, rx]` | 3-D |
| Snapshot scalars | `[snap]` | 1-D |

### 2.27 Schema 校验规则

`validate_hdf5_contract()` 在每次写入后自动执行，检查以下内容：

**结构检查：**
- `/channel/cfr` 必须不存在（truth CFR 必须在 `/channel/truth/cfr`）
- 必填 group 和 dataset 是否存在（见 2.2 节各组）
- CIR 三件套（`cir_coefficients`、`cir_delays_s`、`cir_valid`）同时存在或同时不存在
- PHY 启用时 observation/receiver/evaluation/waveform 各组字段完整性

**Shape 关系：**
- `truth.cfr.ndim == 5`，dtype 为 complex
- `truth.cfr_snapshots.ndim == 6`，`shape[1:] == truth.cfr.shape`
- `cfr_est.ndim == 6`，`shape[-5:] == truth.cfr.shape`
- `cfr.shape[0] == tx_positions.shape[0]`，`cfr.shape[1] == rx_positions.shape[0]`
- `cfr.shape[-1] == frequencies_hz.shape[-1]`
- CIR 三件套 shape 一致，均为 6-D
- per-link 指标 shape 为 `[snap, tx, rx]`

**Path samples shape 约束：**
- `sampled_link_indices.shape == [sample, 2]`
- `vertices_m.ndim == 4`，最后一维为 3
- `doppler_hz`、`tau_s` shape 为 `[sample, sample_path]`
- `interaction_type`、`object_id`、`primitive_id` 的 `shape[:2]` 均为 `[sample, sample_path]`

**数值有效性：**
- CFR 至少一个有限值
- 频率严格递增且有限
- 路径数据有限值
- 观测数据有限值
- `vertex_count[active] >= interaction_count + 2`（包含端点）

**NR PUSCH 专有检查**（`waveform/standard == "nr_pusch"`）：
- `num_prb`、`num_layers`、`num_antenna_ports`、`mimo_detector` 等字段存在
- `num_layers >= 1`，`num_antenna_ports >= num_layers`
- `mimo_detector` 为 `"lmmse"` 或 `"kbest"`
- `receiver_type` 为 `"pusch_receiver"`
- BLER 契约：`num_blocks > 0`，`0 <= num_block_errors <= num_blocks`，`bler == num_block_errors / num_blocks`

**Unit 属性检查：**
所有主要 dataset 必须有 `unit` attribute（见上文各 dataset 表的 Unit 列）。

### 2.28 禁止事项

- 禁止将 truth CFR 写为 `/channel/cfr`（必须在 `/channel/truth/cfr`）
- 禁止在 writer 中 import Sionna
- 禁止 dataset 无 `unit` attribute
- 禁止在 HDF5 中存储 Sionna 原生对象（Paths、Tensor 等）
- 禁止在 `domain/` 和 `io/` 中 import Sionna
- 大文件不入 git
