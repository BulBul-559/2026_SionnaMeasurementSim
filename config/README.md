# 配置说明

SionnaMeasurementSim 使用 YAML 配置文件控制仿真参数。配置加载时自动进行 pydantic schema 校验，不合规会在 RT/PHY 启动前报错退出。

## 使用方式

```bash
uv run python -m sionna_measurement_sim.app.cli --config config/defaults/measurement_mvp.yaml run-full
```

## 有效配置项

以下仅列出当前 pipeline 中实际生效的字段。

### `runtime` — 运行环境

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `seed` | int (>=0) | 42 | 全局随机种子，影响 RT 路径追踪和 PHY 噪声 |
| `device` | str | "cpu" | PyTorch 设备 |

### `input` — 输入数据

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `label_file` | str | data/scenes/test/test5.json | TX/RX 位置标签 JSON |
| `scene_file` | str | data/scenes/test/scene.xml | Mitsuba 场景 XML |
| `max_tx` | int (>=1) | 6 | TX 数量（从标签文件前 N 个取） |
| `max_rx` | int (>=1) | 100 | RX 数量 |

### `output` — 输出控制

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `root_dir` | str | "outputs" | 输出根目录 |
| `hdf5_filename` | str | "results.h5" | HDF5 文件名 |
| `save_full_paths` | bool | false | 是否保存全量路径表 `/paths/full` |

### `carrier` — 载波与频率

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `center_frequency_hz` | float (>0) | 3.5e9 | 中心频率 Hz |
| `bandwidth_hz` | float (>0) | 20e6 | 带宽 Hz |
| `num_subcarriers` | int (>=2) | 64 | 子载波数 |

子载波间隔自动推导 = `bandwidth_hz / num_subcarriers`。

### `antenna` — 天线阵列

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `tx_array.num_rows` | int (>=1) | 1 | TX 行数 |
| `tx_array.num_cols` | int (>=1) | 1 | TX 列数 |
| `tx_array.polarization` | str | "V" | TX 极化 (V/H) |
| `rx_array.num_rows` | int (>=1) | 1 | RX 行数 |
| `rx_array.num_cols` | int (>=1) | 1 | RX 列数 |
| `rx_array.polarization` | str | "V" | RX 极化 |

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

### `phy` — 物理层观测

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | true | 是否启用 PHY 观测（false=仅 RT 真值） |
| `snr_db` | float | 30.0 | 信噪比 dB |
| `fft_size` | int (>=2) | 64 | FFT 大小 |
| `cp_length` | int | 0 | 循环前缀长度 |
| `num_ofdm_symbols` | int | 1 | OFDM 符号数 |
| `tx_power_dbm` | float | 0.0 | 发射功率 dBm |

### `impairments` — 损伤模型

每个子项均有 `enabled` 开关。`enabled: false` 时不施加该损伤（参数传 None）。

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `awgn.enabled` | bool | true | AWGN 噪声 |
| `cfo.enabled` | bool | true | CFO 开关 |
| `cfo.cfo_hz` | float\|null | 100.0 | 载波频偏 Hz（disabled 时忽略） |
| `sfo.enabled` | bool | true | SFO 开关 |
| `sfo.sfo_ppm` | float\|null | 5.0 | 采样频偏 ppm |
| `phase_noise.enabled` | bool | true | 相位偏移开关 |
| `phase_noise.phase_offset_rad` | float\|null | 0.5 | 相位偏移 rad |
| `timing_offset.enabled` | bool | true | 定时偏移开关 |
| `timing_offset.timing_offset_samples` | float\|null | 2.0 | 定时偏移（采样点） |
| `agc_adc.enabled` | bool | true | AGC/ADC 开关 |
| `agc_adc.agc_gain_db` | float | 0.0 | AGC 增益 dB |
| `agc_adc.clipping_threshold` | float\|null | 3.0 | ADC 削波阈值（null=不禁用） |
| `impairment_seed` | int | 142 | 损伤随机种子 |

### `receiver` — 接收机

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `estimator_type` | str | "ls" | 估计器类型 |
| `sync_method` | str | "ideal" | 同步方法 |
| `calibration_profile_id` | str | "synthetic_default" | 校准 profile ID |

### `motion` — 运动与多普勒

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | true | 是否启用运动（false=静态单快照） |
| `mobility_mode` | str | "static" | `static` / `doppler_synthetic` |
| `num_time_steps` | int (>=1) | 3 | 时间快照数 |
| `sampling_frequency_hz` | float | 100.0 | 多普勒采样频率 Hz |
| `tx_velocity_mps` | [float×3] | [0,0,0] | TX 速度 m/s |
| `rx_velocity_mps` | [float×3] | [0,0,0] | RX 速度 m/s |

### `calibration` — 校准

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | true | 是否写 `/calibration` group |
| `profile_id` | str | "synthetic_default" | 校准 profile ID |
