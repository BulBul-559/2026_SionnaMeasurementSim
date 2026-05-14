# 配置说明

SionnaMeasurementSim 使用 YAML 配置文件控制仿真参数。配置加载时自动进行 pydantic schema 校验，不合规会在 RT/PHY 启动前报错退出。

## 使用方式

```bash
# 使用默认配置运行
uv run python -m sionna_measurement_sim.app.cli run-full \
    --config config/defaults/measurement_mvp.yaml \
    --output-dir outputs/my_run

# 使用 NR PUSCH MIMO 配置
uv run python -m sionna_measurement_sim.app.cli run-full \
    --config config/defaults/nr_pusch_mvp.yaml \
    --phy-standard nr_pusch \
    --output-dir outputs/my_nr_pusch_run
```

## 配置模板

| 模板 | 用途 |
|------|------|
| `config/defaults/measurement_mvp.yaml` | 通用 custom OFDM + impairment + motion |
| `config/defaults/nr_pusch_mvp.yaml` | NR PUSCH 4x4 SU-MIMO TDD uplink |
| `config/perf/nr_pusch_3x3000_sharded.yaml` | 3×3000 NR PUSCH shard 性能回归 |
| `config/perf/nr_pusch_6x8884_sharded.yaml` | 6×8884 NR PUSCH 4 GPU shard 验收 |

## 有效配置项

以下仅列出当前 pipeline 中实际生效的字段。

### `runtime` — 运行环境

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `seed` | int (>=0) | 42 | 全局随机种子 |
| `device` | str | "cpu" | PyTorch 设备；NR PUSCH 支持 `"cpu"`、`"cuda"`、`"cuda:0"` 等 PyTorch 设备字符串 |

项目依赖锁定 PyTorch `2.10.0+cu128`，`uv sync` 会从官方 PyTorch CUDA 12.8 wheel 源安装。若配置为 `runtime.device: "cuda"` 但当前 PyTorch 无法初始化 CUDA，NR PUSCH 会直接报错，避免误以为使用了 GPU。

大规模 NR PUSCH 支持 SU-MIMO link batching 和 UE/RX shard。生产运行建议使用 `config/perf/nr_pusch_6x8884_sharded.yaml` 这类模板，让多个进程分别绑定 GPU、分别写 `result_xxx.h5`，再由根目录 `manifest.json` 汇总。

### `debug` — 性能日志

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | false | 是否启用性能 profiling 日志 |
| `hardware_interval_s` | float | 1.0 | GPU/CPU/RSS 采样间隔 |
| `link_log_interval` | int | 250 | 预留的 link chunk 汇总间隔 |
| `torch_synchronize` | bool | true | 阶段计时前后是否同步 CUDA，避免异步 kernel 扭曲耗时 |
| `write_hardware_samples` | bool | true | 是否写 `logs/hardware_samples*.csv` |

开启后会在输出目录 `logs/` 下写入 `perf_events*.jsonl`、`hardware_samples*.csv`、`perf_summary*.json`。shard 模式下文件名带 `shard_000` 后缀，避免多进程互相覆盖。

### `input` — 输入数据

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `label_file` | str | data/scenes/test/test5.json | TX/RX 位置标签 JSON |
| `scene_file` | str | data/scenes/test/scene.xml | Mitsuba 场景 XML |
| `scene_id` | str | scene 文件名 stem | 与平面图/地图系统对齐的稳定场景 ID |
| `map_id` | str | "" | 可选地图版本 ID |
| `max_tx` | int (>=1) | 6 | TX 数量上限 |
| `max_rx` | int (>=1) | 100 | RX 数量上限 |

### `output` — 输出控制

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `root_dir` | str | "outputs" | 输出根目录 |
| `hdf5_filename` | str | "results.h5" | HDF5 文件名 |
| `compression` | str | "gzip" | HDF5 大数组压缩；可选 `gzip`、`lzf`、`none` |
| `save_full_paths` | bool | false | 是否保存全量路径表 `/paths/full` |

#### `output.sharding`

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | false | 是否启用 shard 输出 |
| `axis` | str | "rx" | shard 维度；当前支持 `"rx"`，`"ue"` 会作为别名归一为 `"rx"` |
| `shard_size` | int | 1000 | 每个 shard 的 UE/RX 数量 |
| `filename_pattern` | str | `result_{shard_index:03d}.h5` | shard HDF5 文件名模板 |
| `parallel_workers` | int | 1 | 并行 worker 数 |
| `gpu_ids` | list[int] | [] | shard worker 轮询绑定的 GPU ID |
| `visualization_mode` | str | "first_shard" | `none`、`first_shard`、`all_shards` |

shard 模式下，`run-full` 返回输出目录，根目录 `manifest.json` 汇总所有 `result_xxx.h5`。每个 HDF5 内有 `/shard` group，记录局部索引对应的全局 UE/RX/BS 索引。本项目不把 shard 物理合并成单个巨大 HDF5。

### `carrier` — 载波与频率

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `center_frequency_hz` | float (>0) | 3.5e9 | 中心频率 Hz |
| `bandwidth_hz` | float (>0) | 20e6 | 带宽 Hz |
| `num_subcarriers` | int (>=2) | 64 | 子载波数 |

子载波间隔自动推导 = `bandwidth_hz / num_subcarriers`。

> **NR PUSCH 注意**：NR PUSCH 链路的实际子载波数 = `num_prb * 12`，必须与 `num_subcarriers` 一致。使用 `nr_pusch_mvp.yaml` 模板时两者已对齐（`num_prb=4, num_subcarriers=48`）。

### `antenna` — 天线阵列

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `tx_array.num_rows` | int (>=1) | 1 | TX 行数 |
| `tx_array.num_cols` | int (>=1) | 1 | TX 列数 |
| `tx_array.vertical_spacing_lambda` | float (>0) | 0.5 | 垂直间距（波长） |
| `tx_array.horizontal_spacing_lambda` | float (>0) | 0.5 | 水平间距（波长） |
| `tx_array.polarization` | str | "V" | TX 极化 (V/H) |
| `tx_array.pattern` | str | "iso" | TX 天线方向图 |
| `tx_array.orientation_mode` | str | "fixed" | 朝向模式 |
| `rx_array.num_rows` | int (>=1) | 1 | RX 行数 |
| `rx_array.num_cols` | int (>=1) | 1 | RX 列数 |
| `rx_array.polarization` | str | "V" | RX 极化 |
| `rx_array.pattern` | str | "iso" | RX 天线方向图 |

> 4x4 MIMO：设置 `tx_num_rows=2, tx_num_cols=2, rx_num_rows=2, rx_num_cols=2` 得到 4 TX 天线 × 4 RX 天线。

### `array.spectrum` — 空间谱输出

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | false | 是否生成 Bartlett 空间谱；默认关闭以控制 HDF5 体积 |
| `sources` | list[str] | ["truth_cfr", "cfr_est", "rx_grid"] | `truth_cfr` 生成真值信道谱；`cfr_est` 生成估计信道谱；`rx_grid` 生成 NR PUSCH 接收信号谱 |
| `method` | str | "bartlett" | 第一版仅支持 Bartlett |
| `zenith_bins` | int | 91 | zenith 分辨率 |
| `azimuth_bins` | int | 181 | azimuth 分辨率 |
| `zenith_min_rad/max_rad` | float | [0, pi] | zenith 默认全空间扫描 |
| `azimuth_min_rad/max_rad` | float | [-pi, pi] | azimuth 默认全向扫描 |
| `normalize` | str | "per_link_max" | 每条 link 最大值归一化 |
| `aggregate_subcarriers` | str | "mean" | 子载波聚合方式 |
| `aggregate_symbols` | str | "mean" | OFDM symbol 聚合方式 |
| `link_chunk_size` | int | 512 | Bartlett 空间谱按 link chunk 向量化时的 chunk 大小 |

`/paths/nlos_truth` 默认始终保存所有 NLoS path 的 AoA/AoD、功率、延迟和类型；
`/paths/full` 仍只由 `output.save_full_paths` 控制。

### `visualization` — 采样可视化

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | true（模板） | run-full 后自动生成少量采样 PNG；schema 默认 false |
| `output_dir` | str | "figures" | 相对 run 输出目录的图像目录 |
| `sample_policy` | str | "valid_links_first" | UE 采样策略；可选 `valid_links_first`、`spatially_spread_valid_links`、`random`、`first` |
| `random_seed` | int | 42 | 采样随机种子 |
| `max_bs` | int | 5 | 自动图中最多绘制的 BS 数 |
| `sample_ue_count` | int | 3 | 自动图中随机采样的 UE 数 |
| `max_ue` | int | 5 | 自动图中最多绘制的 UE 数 |
| `dpi` | int | 140 | PNG 分辨率 |
| `format` | str | "png" | 第一版仅支持 PNG |
| `plots` | list[str] | 核心诊断集 | topology/link/CFR/waveform/AoA/NLoS/spectrum/NMSE/path 图 |

`sample_policy` 含义：

- `valid_links_first`：先从任一选中 BS 有效的 UE 中随机采样，不足时从全体 UE 补齐。
- `spatially_spread_valid_links`：先过滤有效 UE，再按 UE 的 XY 坐标做远点采样，适合让示意图中的 UE 跨度更明显。
- `random`：从全体 UE 中随机采样。
- `first`：取前 N 个 UE，便于复现和调试。

绘图约定：

- 所有涉及子载波的热力图统一把 subcarrier 放在纵轴；CFR 折线图例外，使用 subcarrier 横轴，便于直接看频域曲线。
- Matplotlib 热力图显式使用 `interpolation="none"`，不做显示层插值或平滑。
- `cfr_lines`、`cfr_heatmap`、`cfr_error` 都会输出幅度和相位两张图，文件名分别带
  `_magnitude` / `_phase` 后缀；其中 CFR error 的幅度图是估计幅度相对真值幅度的 dB 误差，相位图是 wrap 到 `[-pi, pi]` 的相位误差。
- `spatial_spectrum` 会按数据来源分开输出：
  `spatial_spectrum_label.png`、`spatial_spectrum_truth.png`、
  `spatial_spectrum_cfr_est.png`、`spatial_spectrum_observation.png`。
  同时会额外输出对应的
  `*_polar.png`，每个 link 的 polar 图左右并排：左侧上半球半径为 zenith，
  右侧下半球半径为 `pi - zenith`，外圈都表示水平面。缺失的数据源会跳过，不会用其他谱混画。
- 空间谱矩形图和 polar 图都按“同一个 UE 内的选中 BS”做局部颜色尺度归一；
  polar 图不额外放 colorbar，避免多 link 图中互相遮挡。

嵌入 pipeline 的可视化只做示意采样，不做逐链路全量出图。独立入口支持：

```bash
uv run python -m sionna_measurement_sim.app.cli visualize \
  --hdf5 outputs/run/results.h5 \
  --output-dir outputs/run/figures_manual \
  --mode selected \
  --bs-indices 0,1 \
  --ue-indices 10,20 \
  --plots cfr_lines,spatial_spectrum
```

需要在独立可视化入口使用空间分散采样时：

```bash
uv run python -m sionna_measurement_sim.app.cli visualize \
  --hdf5 outputs/run/results.h5 \
  --output-dir outputs/run/figures_spread \
  --mode sample \
  --sample-policy spatially_spread_valid_links \
  --sample-ue-count 3 \
  --max-bs 5
```

### `rt` — 射线追踪

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_depth` | int (>=0) | 1 | 最大交互深度 |
| `los` | bool | true | 直射路径 |
| `specular_reflection` | bool | true | 镜面反射 |
| `diffuse_reflection` | bool | false | 漫反射 |
| `refraction` | bool | false | 折射 |
| `diffraction` | bool | false | 绕射 |
| `synthetic_array` | bool | false | 合成阵列 |
| `normalize_cfr` | bool | false | 归一化 CFR |
| `normalize_delays` | bool | false | 归一化延迟 |

### `link` — 链路配置（NR PUSCH）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `duplex_mode` | str | "tdd" | 双工模式 |
| `phy_link_direction` | str | "uplink" | PHY 链路方向 |
| `rt_trace_direction` | str | "bs_to_ue" | RT 追踪方向 |
| `reciprocity_mode` | str | "transpose_rt_channel" | 互易性模式 |
| `reciprocity_applied` | bool | true | 是否应用 TDD 互易性 |

### `phy` — 物理层观测

#### 通用字段（custom OFDM 和 NR PUSCH 共享）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | true | 是否启用 PHY 观测 |
| `standard` | str | "custom_ofdm" | `"custom_ofdm"` \| `"nr_pusch"` |
| `snr_db` | float | 30.0 | 信噪比 dB |
| `tx_power_dbm` | float | 0.0 | 发射功率 dBm |

#### custom OFDM 字段

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `fft_size` | int (>=2) | 64 | FFT 大小 |
| `cp_length` | int | 0 | 循环前缀长度 |
| `num_ofdm_symbols` | int | 1 | OFDM 符号数 |

#### NR PUSCH 字段

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `mimo_mode` | str | "su_mimo" | MIMO 模式：`"su_mimo"` \| `"mu_mimo"` |
| `channel_backend` | str | "apply_ofdm" | 信道后端：`"apply_ofdm"` \| `"cir_dataset_ofdm"` |
| `perfect_csi` | bool | false | 完美 CSI（true=接收机用真实信道） |
| `ebno_db` | float\|null | null | Eb/N0 dB；非 null 时优先于 snr_db |
| `num_prb` | int | 16 | PRB 数量 |
| `num_layers` | int | 1 | 每 UE 空间流数 |
| `num_antenna_ports` | int | 4 | 每 UE 天线端口数 (1/2/4) |
| `mcs_index` | int | 14 | MCS 索引 (0-28) |
| `mcs_table` | int | 1 | MCS 表 (0=256QAM, 1=64QAM) |
| `subcarrier_spacing_khz` | int | 30 | 子载波间隔 kHz (15/30/60) |
| `num_ofdm_symbols` | int | 14 | 每时隙 OFDM 符号数 |
| `pusch_dmrs_config_type` | int | 1 | DMRS 配置类型 (1/2) |
| `pusch_dmrs_length` | int | 1 | DMRS 长度 (1/2) |
| `pusch_dmrs_additional_position` | int | 1 | 附加 DMRS 位置 (0-3) |
| `pusch_num_cdm_groups_without_data` | int | 2 | CDM 组数 (1/2/3) |
| `mimo_detector` | str | "lmmse" | MIMO 检测器：`"lmmse"` \| `"kbest"` |
| `channel_estimator` | str | "pusch_ls" | 信道估计：`"pusch_ls"` \| `"perfect"` |
| `receiver_failure_policy` | str | "fail_fast" | 失败策略：`"fail_fast"` \| `"mark_invalid"` |
| `su_mimo_link_batch_size` | int | 1 | SU-MIMO 独立 link batching 大小；NR PUSCH 模板和性能模板设为 64 |

> **MIMO 配置提示：**
> - 4x4 SU-MIMO: `mimo_mode="su_mimo"`, `num_layers=4`, `num_antenna_ports=4`
> - 4x4 SU-MIMO perfect CSI: 加 `perfect_csi=true`, `channel_estimator="perfect"`
> - 4x4 estimated CSI: `perfect_csi=false`, `num_layers=4`, `num_antenna_ports=4` (必须等秩)
> - MU-MIMO: `mimo_mode="mu_mimo"`, 且 `max_rx > 1` (多 UE)
> - 天线数需匹配：`num_antenna_ports` 应等于 `rx_array.num_rows * rx_array.num_cols`

### `impairments` — 损伤模型

每个子项均有 `enabled` 开关。`enabled: false` 时不施加该损伤。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `awgn.enabled` | bool | true | AWGN 噪声 |
| `cfo.enabled` | bool | true | CFO 开关 |
| `cfo.cfo_hz` | float\|null | 100.0 | 载波频偏 Hz |
| `sfo.enabled` | bool | true | SFO 开关 |
| `sfo.sfo_ppm` | float\|null | 5.0 | 采样频偏 ppm |
| `phase_noise.enabled` | bool | true | 相位偏移开关 |
| `phase_noise.phase_offset_rad` | float\|null | 0.5 | 相位偏移 rad |
| `timing_offset.enabled` | bool | true | 定时偏移开关 |
| `timing_offset.timing_offset_samples` | float\|null | 2.0 | 定时偏移（采样点） |
| `agc_adc.enabled` | bool | true | AGC/ADC 开关 |
| `agc_adc.agc_gain_db` | float | 0.0 | AGC 增益 dB |
| `agc_adc.clipping_threshold` | float\|null | 3.0 | ADC 削波阈值 |
| `impairment_seed` | int | 142 | 损伤随机种子 |

### `receiver` — 接收机

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `estimator_type` | str | "ls" | HDF5 `/receiver/estimator_type` |
| `channel_estimator` | str | "pusch_ls" | 信道估计器类型 |
| `mimo_detector` | str | "lmmse" | MIMO 检测器 |
| `sync_method` | str | "ideal" | 同步方法 |
| `interpolation_method` | str | "none" | 插值方法 |
| `failure_policy` | str | "mark_invalid" | 失败策略 |
| `calibration_profile_id` | str | "synthetic_default" | 校准 profile ID |

### `motion` — 运动与多普勒

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | true | 是否启用运动 |
| `mobility_mode` | str | "static" | `"static"` \| `"doppler_synthetic"` |
| `num_time_steps` | int (>=1) | 3 | 时间快照数 |
| `sampling_frequency_hz` | float | 100.0 | 多普勒采样频率 Hz |
| `tx_velocity_mps` | [float×3] | [0,0,0] | TX 速度 m/s |
| `rx_velocity_mps` | [float×3] | [0,0,0] | RX 速度 m/s |

### `calibration` — 校准

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | true | 是否写 `/calibration` group |
| `profile_id` | str | "synthetic_default" | 校准 profile ID |

## MIMO Backend 支持矩阵

| mimo_mode | channel_backend | 状态 |
|-----------|----------------|------|
| `su_mimo` | `apply_ofdm` | 稳定支持 |
| `su_mimo` | `cir_dataset_ofdm` | 支持（per-link，shared delay median） |
| `mu_mimo` | `apply_ofdm` | 支持 |
| `mu_mimo` | `cir_dataset_ofdm` | 不支持（入口拒绝，提示改用 apply_ofdm） |

## MIMO 场景配置速查

### 4x4 SU-MIMO NR PUSCH (perfect CSI)

```yaml
input:
  max_tx: 1
  max_rx: 1
antenna:
  tx_array: { num_rows: 2, num_cols: 2, polarization: "V" }
  rx_array: { num_rows: 2, num_cols: 2, polarization: "V" }
phy:
  standard: "nr_pusch"
  mimo_mode: "su_mimo"
  channel_backend: "apply_ofdm"
  perfect_csi: true
  num_prb: 4
  num_layers: 4
  num_antenna_ports: 4
  channel_estimator: "perfect"
  receiver_failure_policy: "fail_fast"
link:
  reciprocity_applied: true
```

### 4x4 SU-MIMO NR PUSCH (estimated CSI)

```yaml
phy:
  standard: "nr_pusch"
  perfect_csi: false
  num_layers: 4            # must equal num_antenna_ports for estimated CSI
  num_antenna_ports: 4
  channel_estimator: "pusch_ls"
```

### 2-UE MU-MIMO NR PUSCH

```yaml
input:
  max_tx: 1          # 1 BS
  max_rx: 2          # 2 UEs
phy:
  standard: "nr_pusch"
  mimo_mode: "mu_mimo"
  num_layers: 1
  num_antenna_ports: 2    # 2 antennas per UE
```
