# 06. 配置与实验 Schema

本文定义新系统配置结构。配置必须能完整复现实验，并能映射到 [03_data_contract_hdf5.md](03_data_contract_hdf5.md) 的 `/meta/config_snapshot`。

配置实现必须遵守 [00_global_constraints_and_official_references.md](00_global_constraints_and_official_references.md)。Sionna/PyTorch 版本约束以官方安装文档为准：https://nvlabs.github.io/sionna/installation.html

## 1. 基本原则

- 使用 YAML 作为用户配置格式。
- 使用代码 schema 做类型、单位和必填字段校验。
- 禁止只做无校验的 dict merge。
- 所有随机过程必须有 seed。
- 所有物理量必须在字段名或 schema 中明确单位。
- 配置快照必须写入 HDF5 和 manifest。

## 2. 推荐配置分组

```yaml
runtime:
input:
output:
carrier:
antenna:
rt:
phy:
impairments:
receiver:
link:
motion:
analysis:
visualization:
calibration:
```

## 3. runtime

```yaml
runtime:
  seed: 42
  device: "cuda:0"
  require_gpu: true
  precision: "single"
  torch_deterministic: false
```

## 4. input

```yaml
input:
  label_schema: "simplesionna_v1"
  coordinate_system: "scene_local_xyz_m"
  scene_root: "data/scenes"
  label_file: "data/scenes/test/test.json"
```

## 5. output

```yaml
output:
  root_dir: "outputs"
  run_id_format: "{label_stem}_{timestamp}"
  hdf5_filename: "results.h5"
  compression: "gzip"
  save_full_paths: false
  save_sampled_paths: true
  save_raw_waveform: false
```

## 6. carrier

```yaml
carrier:
  center_frequency_hz: 6115000000.0
  bandwidth_hz: 80000000.0
  num_subcarriers: 1024
  subcarrier_spacing_hz: 78125.0
```

注意：对于 NR PUSCH，配置 YAML 中使用 `subcarrier_spacing_khz`（单位 kHz），
HDF5 `/waveform` 中则存储为 `subcarrier_spacing_hz`（单位 Hz）。

## 7. antenna

```yaml
antenna:
  tx_array:
    type: "planar"
    num_rows: 2
    num_cols: 2
    vertical_spacing_lambda: 0.5
    horizontal_spacing_lambda: 0.5
    pattern: "iso"
    polarization: "V"
  rx_array:
    type: "planar"
    num_rows: 2
    num_cols: 2
    vertical_spacing_lambda: 0.5
    horizontal_spacing_lambda: 0.5
    pattern: "iso"
    polarization: "V"
```

## 8. rt

```yaml
rt:
  enabled: true
  engine: "sionna_rt"
  max_depth: 5
  samples_per_src: 1000000
  los: true
  specular_reflection: true
  diffuse_reflection: false
  refraction: true
  diffraction: false
  synthetic_array: true
  out_type: "numpy"
```

## 9. phy

```yaml
phy:
  enabled: true
  standard: "custom_ofdm"
  num_snapshots: 1
  fft_size: 1024
  cp_length: 72
  num_ofdm_symbols: 1
  pilot_pattern: "all_active_subcarriers"
  channel_estimator: "ls"
  interpolation: "none"
```

## 10. impairments

```yaml
impairments:
  awgn:
    enabled: true
    snr_db:
      mode: "uniform"
      min: 10.0
      max: 30.0
  cfo:
    enabled: false
    distribution: "normal"
    std_hz: 200.0
  sfo:
    enabled: false
    std_ppm: 1.0
  phase_noise:
    enabled: false
  iq_imbalance:
    enabled: false
  agc_adc:
    enabled: false
    adc_bits: 12
```

## 11. motion

```yaml
motion:
  enabled: false
  mobility_mode: "static"
  sampling_frequency_hz: 1.0
  num_time_steps: 1
  default_tx_velocity_mps: [0.0, 0.0, 0.0]
  default_rx_velocity_mps: [0.0, 0.0, 0.0]
```

`mobility_mode` 可选：

```text
static
doppler_synthetic
retrace_positions
```

## 12. receiver

```yaml
receiver:
  estimator_type: "ls"
  sync_method: "ideal"
  interpolation_method: "none"
  packet_detection_threshold: 0.0
  failure_policy: "mark_invalid"
```

## 13. link

```yaml
link:
  duplex_mode: "tdd"
  phy_link_direction: "uplink"
  rt_trace_direction: "bs_to_ue"
  reciprocity_mode: "transpose_rt_channel"
  reciprocity_applied: true
```

## 14. calibration

```yaml
calibration:
  enabled: false
  profile_id: "none"
  measurement_dataset: null
```

详见 [11_calibration_and_diagnostics.md](11_calibration_and_diagnostics.md)。

## 15. Schema 校验要求

实现时必须校验：

- 必填字段存在。
- 单位字段范围合理。
- `num_subcarriers * subcarrier_spacing_hz` 与 bandwidth 的关系。
- 天线数量与结果 shape 一致。
- `phy.enabled=true` 时 waveform/receiver 配置完整。
- `motion.enabled=true` 时速度、snapshot、sampling frequency 合法。
- `save_full_paths=false` 时仍必须保存 `/paths/samples`。
- `rt.out_type` 只能取 adapter 已测试通过的输出类型。
- `phy.enabled=true` 时必须启用或显式配置 `/waveform`、`/receiver`、`/observation` 所需字段。
- `impairments.*.enabled=true` 时必须写入对应配置和本次采样值。
- `motion.mobility_mode=doppler_synthetic` 时必须提供 `sampling_frequency_hz` 和 `num_time_steps`。
- 所有 profile 覆盖后仍必须通过完整 schema 校验。

配置校验失败时，程序必须在 RT 或 PHY 开始前停止，不允许跑到中途才失败。

## 16. Profile 策略

推荐配置文件：

```text
config/defaults/rt_only_debug.yaml
config/defaults/measurement_mvp.yaml
config/defaults/wifi6e_like.yaml
config/defaults/doppler_debug.yaml
config/defaults/full_paths_debug.yaml
```

profile 不能覆盖 schema 约束。
