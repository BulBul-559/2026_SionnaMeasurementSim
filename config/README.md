# 配置说明

SionnaMeasurementSim 使用 YAML 配置文件控制仿真参数。配置文件必须符合 `MeasurementConfig` pydantic schema，加载时会自动校验，不合规的配置会在 RT/PHY 启动前报错退出。

## 使用方式

```bash
# 使用默认配置（config/defaults/measurement_mvp.yaml）
uv run python -m sionna_measurement_sim.app.cli run-full --config config/defaults/measurement_mvp.yaml

# 使用自定义配置
uv run python -m sionna_measurement_sim.app.cli run-full --config my_experiment.yaml
```

## 配置项说明

### `runtime` — 运行环境

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `seed` | int (>=0) | 42 | 全局随机种子，影响 RT 路径追踪和 PHY 噪声 |
| `device` | str | "cpu" | PyTorch 设备，"cpu" 或 "cuda:0" |
| `require_gpu` | bool | false | 是否要求 GPU，false 时CPU回退 |
| `precision` | str | "single" | 计算精度，"single" 或 "double" |
| `torch_deterministic` | bool | false | 是否使用确定性算法（慢但有复现性） |

### `input` — 输入数据

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `label_file` | str | data/scenes/test/test5.json | TX/RX 位置标签 JSON |
| `scene_file` | str | data/scenes/test/scene.xml | Mitsuba 场景 XML |
| `label_schema` | str | "simplesionna_v1" | 标签格式版本 |
| `coordinate_system` | str | "scene_local_xyz_m" | 坐标系约定 |
| `max_tx` | int (>=1) | 6 | 最大发射机数，从标签文件前 N 个取 |
| `max_rx` | int (>=1) | 100 | 最大接收机数 |

### `output` — 输出控制

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `root_dir` | str | "outputs" | 输出根目录 |
| `hdf5_filename` | str | "results.h5" | HDF5 文件名 |
| `compression` | str | "gzip" | 压缩方式 |
| `save_full_paths` | bool | false | 是否保存全量路径（文件大） |
| `save_sampled_paths` | bool | true | 是否保存采样路径 |
| `save_raw_waveform` | bool | false | 是否保存原始波形（文件很大） |

### `carrier` — 载波与频率

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `center_frequency_hz` | float (>0) | 3.5e9 | 中心频率 Hz |
| `bandwidth_hz` | float (>0) | 20e6 | 带宽 Hz |
| `num_subcarriers` | int (>=2) | 64 | 子载波数 |
| `subcarrier_spacing_hz` | float | 0 (自动推导) | 子载波间隔，0 则自动 = bandwidth/num_subcarriers |

### `antenna` — 天线阵列配置

```yaml
antenna:
  tx_array:
    type: "planar"         # 阵列类型
    num_rows: 1            # 行数
    num_cols: 1            # 列数
    vertical_spacing_lambda: 0.5   # 垂直间距（波长单位）
    horizontal_spacing_lambda: 0.5 # 水平间距
    pattern: "iso"         # 天线方向图
    polarization: "V"      # 极化 (V/H)
  rx_array:
    # 同 tx_array 结构
```

### `rt` — 射线追踪

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | true | 是否启用 RT |
| `engine` | str | "sionna_rt" | RT 引擎 |
| `max_depth` | int (>=0) | 1 | 最大反射/绕射深度 |
| `los` | bool | true | 是否包含直射路径 |
| `specular_reflection` | bool | true | 是否包含镜面反射 |
| `diffuse_reflection` | bool | false | 是否包含漫反射 |
| `refraction` | bool | false | 是否包含折射 |
| `diffraction` | bool | false | 是否包含绕射 |
| `synthetic_array` | bool | false | 是否使用合成阵列 |
| `normalize_cfr` | bool | false | 是否归一化 CFR |
| `normalize_delays` | bool | false | 是否归一化延迟 |

### `phy` — 物理层观测

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | true | 是否启用 PHY 观测 |
| `standard` | str | "custom_ofdm" | 波形标准 |
| `snr_db` | float | 30.0 | 信噪比 dB |
| `fft_size` | int (>=2) | 64 | FFT 大小（即子载波数） |
| `cp_length` | int | 0 | 循环前缀长度 |
| `num_ofdm_symbols` | int | 1 | OFDM 符号数 |
| `pilot_pattern` | str | "all_active_subcarriers" | 导频模式 |
| `channel_estimator` | str | "ls" | 信道估计器 |
| `interpolation` | str | "none" | 插值方法 |
| `tx_power_dbm` | float | 0.0 | 发射功率 dBm |

### `impairments` — 损伤模型

```yaml
impairments:
  awgn:
    enabled: true          # 加性高斯白噪声
  cfo:
    enabled: true
    cfo_hz: 100.0          # 载波频率偏移 Hz（null=禁用）
  sfo:
    enabled: true
    sfo_ppm: 5.0           # 采样频率偏移 ppm
  phase_noise:
    enabled: true
    phase_offset_rad: 0.5  # 相位偏移 rad
  timing_offset:
    enabled: true
    timing_offset_samples: 2.0  # 定时偏移 采样点
  agc_adc:
    enabled: true
    agc_gain_db: 0.0       # AGC 增益 dB
    clipping_threshold: 3.0  # ADC 削波阈值 (null=无削波)
  impairment_seed: 142     # 损伤随机种子
```

### `receiver` — 接收机

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `estimator_type` | str | "ls" | 估计器类型 |
| `sync_method` | str | "ideal" | 同步方法 |
| `interpolation_method` | str | "none" | 插值方法 |
| `packet_detection_threshold` | float | 0.0 | 包检测阈值 |
| `failure_policy` | str | "mark_invalid" | 失败策略 |
| `calibration_profile_id` | str | "synthetic_default" | 校准 profile |

### `motion` — 运动与多普勒

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | true | 是否启用运动 |
| `mobility_mode` | str | "static" | 运动模式 (static/doppler_synthetic) |
| `num_time_steps` | int (>=1) | 3 | 时间快照数 |
| `sampling_frequency_hz` | float | 100.0 | 采样频率 Hz |
| `tx_velocity_mps` | [float,float,float] | [0,0,0] | TX 速度 m/s |
| `rx_velocity_mps` | [float,float,float] | [0,0,0] | RX 速度 m/s |

### `calibration` — 校准

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | true | 是否启用校准 |
| `profile_id` | str | "synthetic_default" | 校准 profile ID |
