# 配置说明

SionnaMeasurementSim 使用 YAML 配置文件控制仿真参数。配置加载时自动进行 pydantic schema 校验，不合规会在 RT/PHY 启动前报错退出。

## 使用方式

```bash
# 使用默认配置运行
uv run python -m sionna_measurement_sim.app.cli \
    --config config/defaults/measurement_mvp.yaml \
    run-full \
    --output-dir outputs/my_run

# 使用 NR PUSCH MIMO 配置
uv run python -m sionna_measurement_sim.app.cli \
    --config config/defaults/nr_pusch_mvp.yaml \
    run-full \
    --phy-standard nr_pusch \
    --output-dir outputs/my_nr_pusch_run

# 使用 NR SRS standards-shaped v2 subset 配置
uv run python -m sionna_measurement_sim.app.cli \
    --config config/defaults/nr_srs_indoor_positioning_fr1_100mhz.yaml \
    run-full \
    --phy-standard nr_srs \
    --output-dir outputs/my_nr_srs_run
```

## 配置模板

| 模板 | 用途 |
|------|------|
| `config/defaults/measurement_mvp.yaml` | 通用 custom OFDM + impairment + motion |
| `config/defaults/nr_pusch_mvp.yaml` | NR PUSCH 4x4 SU-MIMO TDD uplink |
| `config/defaults/nr_pusch_indoor_positioning_fr1_100mhz.yaml` | 室内 FR1 100 MHz NR uplink 定位主实验模板 |
| `config/defaults/nr_srs_indoor_positioning_fr1_100mhz.yaml` | 室内 FR1 100 MHz NR SRS subset uplink sounding 模板 |
| `config/perf/nr_pusch_3x3000_sharded.yaml` | 3×3000 NR PUSCH shard 性能回归 |
| `config/perf/nr_pusch_6x8884_sharded.yaml` | 6×8884 NR PUSCH 4 GPU shard 验收 |
| `config/perf/nr_srs_7x500_sharded.yaml` | 7×500 NR SRS direct uplink shard 历史确认测试 |
| `config/perf/nr_srs_6x5_rt_refraction_*.yaml` | NR SRS 100 MHz RT 参数 sweep 模板，用于对比 refraction/diffuse/max_depth |

> Bistro FR1 100 MHz 模板使用 3276 个 active subcarrier。NR SRS 模板当前默认
> direct uplink、`rt.synthetic_array=false`、UE shard `shard_size: 20`，适合少量
> BS、大量 UE 的生产数据生成。`max_ue: 2500` 是目标数据规模，不是提交前必跑验收；
> 7×500 shard 确认测试见 `config/perf/nr_srs_7x500_sharded.yaml` 和
> `docs/performance/nr_srs_7x500_sharded_confirmation.md`。

仓库里的 `data/` 与 `outputs/` 一样是 gitignore 的本地运行路径。生产场景、floor-plan、label 和 HDF5 不进入 git；可以把 `data` 本身做成本地 symlink 指向共享场景目录，也可以在 YAML 中直接使用共享存储的绝对路径。测试用的小场景固定放在 `tests/fixtures/scenes/test/`。

## 有效配置项

以下仅列出当前 pipeline 中实际生效的字段。

### `runtime` — 运行环境

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `seed` | int (>=0) | 42 | 全局随机种子 |
| `device` | str | "cpu" | PyTorch 设备；NR PUSCH 支持 `"cpu"`、`"cuda"`、`"cuda:0"` 等 PyTorch 设备字符串 |

项目依赖锁定 PyTorch `2.10.0+cu128`，`uv sync` 会从官方 PyTorch CUDA 12.8 wheel 源安装。若配置为 `runtime.device: "cuda"` 但当前 PyTorch 无法初始化 CUDA，NR PUSCH 会直接报错，避免误以为使用了 GPU。

大规模 NR PUSCH 支持 SU-MIMO link batching 和 UE shard。生产运行建议使用 `config/perf/nr_pusch_6x8884_sharded.yaml` 这类模板，让多个进程分别绑定 GPU、分别写 `results/result_xxx.h5`，再由 `manifest/manifest.json` 汇总。

NR SRS 100 MHz 在 `rt.synthetic_array=false` 下会显著增加 Sionna RT 的底层显存需求。
当前 pipeline 使用 `link.phy_link_direction` 直接解析 BS/UE 到 TX/RX：uplink 为
`UE -> BS`，downlink 为 `BS -> UE`。在 `bistro_0000` 当前 RT 配置下，已验证
`7 BS x 30 UE` 单 shard 可运行、`7 BS x 35 UE` 会在 `PathSolver` 阶段 OOM。
在 `median_0000` 的 `label0p2` 全量 baseline 中，`shard_size: 25` 会在
`paths.cfr()` 阶段触发 Dr.Jit 单数组 entry 数超过 `2^32` 的限制；`shard_size: 20`
已完成 `7 BS x 2583 UE` 全量 SRS direct uplink。因此默认 NR SRS 模板使用
`shard_size: 20` 作为当前生产值。多个 shard 会分别写 `results/result_xxx.h5`，
`manifest/manifest.json` 汇总全局 UE/BS 索引。

## Role View vs Link View

配置和 label 层使用物理角色：`BS`、`UE`。仿真和 HDF5 层使用当前链路方向下的
`TX`、`RX`。映射只由 `link.phy_link_direction` 决定：

| `phy_link_direction` | TX role | RX role |
|----------------------|---------|---------|
| `uplink` | UE | BS |
| `downlink` | BS | UE |

因此配置里写 `input.max_bs/max_ue`、`antenna.bs_array/ue_array`、
`motion.bs_velocity_mps/ue_velocity_mps`；HDF5 仍输出
`/channel/truth/cfr [tx, rx, rx_ant, tx_ant, subcarrier]`，并在 `/link/tx_role`、
`/link/rx_role` 记录 resolved role。旧的 `max_tx/max_rx`、`tx_array/rx_array`、
`rt_trace_direction`、`reciprocity_*` YAML 字段是 breaking change 后的废弃字段，
配置加载会拒绝它们。

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
| `label_file` | str | tests/fixtures/scenes/test/test5.json | BS/UE 位置标签 JSON |
| `scene_file` | str | tests/fixtures/scenes/test/scene.xml | Mitsuba 场景 XML |
| `scene_id` | str | scene 文件名 stem | 与平面图/地图系统对齐的稳定场景 ID |
| `map_id` | str | "" | 可选地图版本 ID |
| `max_bs` | int (>=1) | 6 | BS 数量上限 |
| `max_ue` | int (>=1) | 100 | UE 数量上限 |

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
| `axis` | str | "ue" | shard 维度；当前仅支持 `"ue"` |
| `shard_size` | int | 1000 | 每个 shard 的 UE 数量 |
| `filename_pattern` | str | `result_{shard_index:03d}.h5` | shard HDF5 文件名模板 |
| `results_dir` | str | `results` | shard HDF5 文件子目录 |
| `manifest_dir` | str | `manifest` | aggregate/per-shard manifest 与 config snapshot 子目录 |
| `parallel_workers` | int | 1 | 并行 worker 数 |
| `gpu_ids` | list[int] | [] | shard worker 轮询绑定的 GPU ID |
| `visualization_mode` | str | "first_shard" | `none`、`first_shard`、`all_shards` |
| `fallback.enabled` | bool | true | 单个 shard 失败时是否自动拆小重试 |
| `fallback.min_shard_size` | int | 1 | fallback 最小 UE shard 大小 |
| `fallback.split_factor` | int | 2 | 每次失败拆成几个子 shard |
| `fallback.retry_errors` | list[str] | `["cuda_oom", "drjit_array_limit"]` | 允许自动回退的错误类型 |
| `fallback.failure_policy` | str | `fail_run` | 到最小 shard 仍失败时终止运行 |

shard 模式下，`run-full` 返回输出目录，`results/` 保存所有 `result_xxx.h5`，`manifest/manifest.json` 汇总所有 result 文件。每个 HDF5 内有 `/shard` group，记录局部 TX/RX 索引对应的全局 BS/UE 索引。本项目不把 shard 物理合并成单个巨大 HDF5。`manifest/config_snapshot.json` 会保存 resolved 运行配置，保证数据目录自包含。

### `carrier` — 载波与频率

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `center_frequency_hz` | float (>0) | 3.5e9 | 中心频率 Hz |
| `bandwidth_hz` | float (>0) | 20e6 | 带宽 Hz |
| `num_subcarriers` | int (>=2) | 64 | 子载波数 |

子载波间隔自动推导 = `bandwidth_hz / num_subcarriers`。

> **NR PUSCH 注意**：NR PUSCH 链路的实际子载波数 = `num_prb * 12`，必须与 `num_subcarriers` 一致。使用 `nr_pusch_mvp.yaml` 模板时两者已对齐（`num_prb=4, num_subcarriers=48`）。

> 室内定位模板使用名义 100 MHz NR channel bandwidth、30 kHz SCS、273 PRB。当前代码的 `carrier.bandwidth_hz` 用于生成 active subcarrier 频率栅格，因此模板写为 active occupied bandwidth `273*12*30 kHz = 98.28 MHz`，以保证频率间隔严格等于 30 kHz。

### `antenna` — 天线阵列

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `bs_array.num_rows` | int (>=1) | 1 | BS 阵列行数 |
| `bs_array.num_cols` | int (>=1) | 1 | BS 阵列列数 |
| `bs_array.vertical_spacing_lambda` | float (>0) | 0.5 | BS 垂直间距（波长） |
| `bs_array.horizontal_spacing_lambda` | float (>0) | 0.5 | BS 水平间距（波长） |
| `bs_array.polarization` | str | "V" | BS 极化 (V/H) |
| `bs_array.pattern` | str | "iso" | BS 天线方向图 |
| `bs_array.orientation_mode` | str | "fixed" | BS 朝向模式 |
| `ue_array.num_rows` | int (>=1) | 1 | UE 阵列行数 |
| `ue_array.num_cols` | int (>=1) | 1 | UE 阵列列数 |
| `ue_array.polarization` | str | "V" | UE 极化 |
| `ue_array.pattern` | str | "iso" | UE 天线方向图 |

> 4x4 MIMO：设置 `bs_array.num_rows=2, bs_array.num_cols=2, ue_array.num_rows=2, ue_array.num_cols=2`，在 uplink 下得到 4 UE TX 天线 × 4 BS RX 天线。

### `array.spectrum` — 空间谱输出

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | false | 是否生成 Bartlett 空间谱；默认关闭以控制 HDF5 体积 |
| `sources` | list[str] | ["truth_cfr", "cfr_est", "rx_grid"] | `truth_cfr` 生成真值信道谱；`cfr_est` 生成估计信道谱；`rx_grid` 生成 NR PUSCH/SRS 接收信号谱；`srs_cfr_est` 是历史兼容别名，仍指向 NR SRS 的 `/observation/cfr_est` |
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
- `path_samples` 只绘制当前采样选择中的第一个 UE-BS 链路，避免多个链路路径叠加后难以判断几何关系；如需对比多条链路，应分别指定 BS/UE 后多次可视化。
- `cfr_lines`、`cfr_heatmap`、`cfr_error` 都会输出幅度和相位两张图，文件名分别带
  `_magnitude` / `_phase` 后缀；其中 CFR error 的幅度图是估计幅度相对真值幅度的 dB 误差，相位图是 wrap 到 `[-pi, pi]` 的相位误差。
- `spatial_spectrum` 会按数据来源分开输出：
  `spatial_spectrum_label.png`、`spatial_spectrum_truth.png`、
  `spatial_spectrum_cfr_est.png`、`spatial_spectrum_observation.png`、
  `spatial_spectrum_srs.png`。
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

### `link` — 链路配置

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `duplex_mode` | str | "tdd" | 双工模式 |
| `phy_link_direction` | str | "uplink" | PHY 链路方向；`uplink` 为 UE→BS，`downlink` 为 BS→UE |

### `phy` — 物理层观测

#### 通用字段（custom OFDM、NR PUSCH 和 NR SRS 共享）

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | true | 是否启用 PHY 观测 |
| `standard` | str | "custom_ofdm" | `"custom_ofdm"` \| `"nr_pusch"` \| `"nr_srs"` |
| `snr_db` | float | 30.0 | 信噪比 dB |
| `tx_power_dbm` | float | 0.0 | 发射功率 dBm |

NR PUSCH 与 NR SRS 共享 `common_link.py` 的 clean channel → impairment/AWGN
链路。`snr_db` 口径下 AWGN 方差按 clean `rx_grid` 每条 link 的平均功率计算；
仅当 `phy.ebno_db` 非空时，PUSCH 使用 Sionna `ebnodb2no` 结果作为 receiver/noise
override。`/observation/cfo_hz`、`sfo_ppm`、`timing_offset_samples`、
`phase_offset_rad`、`agc_gain_db`、`clipping_flag` 均来自 common impairment chain。

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
| `perfect_csi` | bool | false | 完美 CSI（true=接收机使用 clean channel 返回的 oracle CSI；false=PUSCHReceiver 内部 DMRS LS 估计） |
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

#### NR SRS subset 字段

`standard: "nr_srs"` 复用 `subcarrier_spacing_khz`、`num_prb`、`tx_power_dbm`、
`receiver_failure_policy` 等通用/NR-family 字段，并通过 `phy.srs` 控制 SRS resource。
当前实现是 standards-shaped v2 subset：完整 14-symbol slot 中只在 SRS symbols
填充 comb/BWP/hopping resource，支持 deterministic `nr_zc`、group/sequence
hopping、cyclic-shift port multiplexing、antenna switching 口径和简化 uplink power
scaling。receiver 从 flattened SRS RE 做 despread/LS，再插值到 full-band
`/observation/cfr_est`。它不产生 BER/BLER，只输出 NMSE、幅度/相位误差、
correlation、estimation success 和 SRS resource quality 指标。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `srs.slot_length_symbols` | int | 14 | 当前单 slot 的 OFDM symbol 数 |
| `srs.start_symbol` | int | 12 | SRS 起始 symbol |
| `srs.num_srs_symbols` | int | 2 | SRS 占用 symbol 数；`cyclic_shift_multiplexing="time"` 时需不小于 SRS port 数 |
| `srs.comb_size` | int | 2 | SRS comb size，支持 1/2/4 |
| `srs.comb_offset` | int | 0 | comb offset，需小于 comb size |
| `srs.bwp_start_prb` | int | 0 | SRS BWP 起始 PRB |
| `srs.bwp_num_prb` | int\|null | null | null 时使用 `phy.num_prb`，并按 carrier 子载波数裁剪 |
| `srs.trigger_mode` | str | "aperiodic" | `"aperiodic"` \| `"periodic"` \| `"semipersistent"` |
| `srs.periodicity_slots` | int | 1 | periodic/semipersistent 调度周期 |
| `srs.slot_offset` | int | 0 | 调度 slot offset |
| `srs.slot_number` | int | 0 | 当前单 slot 编号；若未被调度则 fail-fast |
| `srs.sequence_type` | str | "zc_like" | `"zc_like"` \| `"nr_zc"`；`nr_zc` 用于 v2 low-PAPR/ZC-like sounding |
| `srs.sequence_id` | int | 0 | sequence seed/id |
| `srs.group_hopping` | str | "disabled" | `"disabled"` \| `"enabled"` |
| `srs.sequence_hopping` | str | "disabled" | `"disabled"` \| `"enabled"` |
| `srs.cyclic_shift_multiplexing` | str | "cyclic_shift" | `"cyclic_shift"` 同 symbol port 复用；`"time"` 为 time-symbol orthogonality fallback |
| `srs.cyclic_shift_indices` | list[int]\|null | null | null 时按 port 均匀分配 0..11；`cyclic_shift` 模式要求唯一 |
| `srs.hopping.enabled` | bool | false | 是否启用每 SRS symbol 的 frequency/bandwidth hopping |
| `srs.hopping.frequency_offsets_prb` | list[int] | [] | 空或长度等于 `num_srs_symbols`；每 symbol PRB offset |
| `srs.hopping.bandwidth_num_prb` | list[int] | [] | 空或长度等于 `num_srs_symbols`；每 symbol SRS PRB 数 |
| `srs.ports.num_srs_ports` | int\|null | null | null 时使用 UE TX 天线数 |
| `srs.ports.mapping` | str | "one_to_one" | `"one_to_one"` \| `"antenna_switching"` |
| `srs.ports.port_tx_ant_map` | list[list[int]]\|null | null | `antenna_switching` 时为 `[srs_port,srs_symbol]`，`-1` 表示该 symbol 不发 |
| `srs.ports.usage` | str | "non_codebook" | `"codebook"` \| `"non_codebook"` metadata |
| `srs.power_control.enabled` | bool | false | 是否启用 pathloss-compensated SRS power scaling |
| `srs.power_control.p0_dbm` | float | 0.0 | open-loop power control 的基准功率 |
| `srs.power_control.alpha` | float | 0.8 | pathloss compensation 系数 |
| `srs.power_control.min_tx_power_dbm/max_tx_power_dbm` | float | -40.0 / 23.0 | 发射功率裁剪范围 |
| `srs.power_control.serving_rx_policy` | str | "strongest_path" | `"strongest_path"` \| `"first_rx"` |

当前仍不能称为完整 3GPP NR SRS；v2 是 standards-shaped subset，尚未做 38.211/38.213
reference 对齐、真实闭环功控或认证级一致性测试。

> **MIMO 配置提示：**
> - 4x4 SU-MIMO: `mimo_mode="su_mimo"`, `num_layers=4`, `num_antenna_ports=4`
> - 4x4 SU-MIMO perfect CSI: 加 `perfect_csi=true`, `channel_estimator="perfect"`
> - 4x4 estimated CSI: `perfect_csi=false`, `num_layers=4`, `num_antenna_ports=4` (必须等秩)
> - MU-MIMO: `mimo_mode="mu_mimo"`, 且 `max_ue > 1` (多 UE)
> - uplink 下天线数需匹配：`num_antenna_ports` 应等于 `ue_array.num_rows * ue_array.num_cols`

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

### `ranging` — 波形级 ToA/range 观测

`ranging` 是独立于 SRS/PUSCH 的 observation 后处理模块。开启后必须同时启用
`phy.enabled=true`，并从 `/observation/cfr_est` 估计 ToA/one-way range；它不会改写
`/derived` truth 字段。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | false | 是否写 `/ranging` group |
| `source` | str | `"cfr_est"` | v1 仅支持从 `/observation/cfr_est` 估计 |
| `estimators` | list[str] | `["pdp_peak", "phase_slope"]` | 启用的 estimator |
| `default_estimator` | str | `"pdp_peak"` | 下游默认读取的 estimator 名称 |
| `write_rtt_equivalent` | bool | true | 是否写 `rtt_equiv_s = 2 * toa_est_s`；它不是协议 RTT |
| `pdp_peak.oversampling_factor` | int | 8 | IFFT PDP zero-padding 倍数 |
| `pdp_peak.window` | str | `"hann"` | PDP 窗函数；可选 `hann`、`rect` |
| `pdp_peak.peak_policy` | str | `"earliest_above_relative_threshold"` | 选择最早可检测峰 |
| `pdp_peak.relative_threshold_db` | float | -12.0 | 相对最强峰的首径检测阈值 |
| `pdp_peak.min_peak_snr_db` | float | 6.0 | 峰值相对噪底的最小检测 SNR |
| `pdp_peak.interpolation` | str | `"parabolic_log_power"` | 峰值亚 bin 插值 |
| `pdp_peak.max_delay_s` | float\|null | null | 可选最大搜索 delay |
| `phase_slope.unwrap` | bool | true | phase vs frequency 是否 unwrap |
| `phase_slope.aggregate` | str | `"power_weighted_median"` | 多天线 pair 聚合方式 |
| `phase_slope.min_mean_power` | float | 1.0e-12 | pair 级最小平均功率 |

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
| `bs_velocity_mps` | [float×3] | [0,0,0] | BS 速度 m/s |
| `ue_velocity_mps` | [float×3] | [0,0,0] | UE 速度 m/s |

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
  max_bs: 1
  max_ue: 1
antenna:
  bs_array: { num_rows: 2, num_cols: 2, polarization: "V" }
  ue_array: { num_rows: 2, num_cols: 2, polarization: "V" }
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
  phy_link_direction: "uplink"
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
  max_bs: 1          # 1 BS
  max_ue: 2          # 2 UEs
phy:
  standard: "nr_pusch"
  mimo_mode: "mu_mimo"
  num_layers: 1
  num_antenna_ports: 2    # 2 antennas per UE
```

### 1x2 UE NR SRS subset sounding

```yaml
array:
  spectrum:
    enabled: true
    sources: ["truth_cfr", "cfr_est"]  # "srs_cfr_est" 仍是兼容别名
antenna:
  bs_array: { num_rows: 4, num_cols: 4 }  # BS receiver array in uplink
  ue_array: { num_rows: 1, num_cols: 2 }  # UE sounding antennas
phy:
  standard: "nr_srs"
  subcarrier_spacing_khz: 30
  num_ofdm_symbols: 2
  channel_estimator: "srs_ls"
```
