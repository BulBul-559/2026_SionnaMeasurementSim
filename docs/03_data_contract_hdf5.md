# 03. HDF5 数据契约

本文定义新系统 HDF5 数据契约。所有 writer/reader、分析脚本、可视化脚本都必须遵循本文。RT 路径字段来源和提取方式见 [04_sionna_rt_adapter_and_path_data.md](04_sionna_rt_adapter_and_path_data.md)。PHY 观测字段语义见 [05_phy_observation_and_impairments.md](05_phy_observation_and_impairments.md)。

相关全局约束见 [00_global_constraints_and_official_references.md](00_global_constraints_and_official_references.md)。任何 schema 变更都必须同步更新 [08_roadmap_milestones_acceptance.md](08_roadmap_milestones_acceptance.md) 和 [09_testing_and_quality_gates.md](09_testing_and_quality_gates.md) 中的验收/测试要求。

## 1. 设计原则

新系统采用“尽量完整采集，后续按需取子集”的策略。

原则：

- 保存 truth 与 observation。
- 保存路径级几何和物理信息。
- 保存设备状态、速度、朝向、天线、极化。
- 保存动态快照信息。
- 保存观测链诊断。
- 所有数据有单位和维度约定。
- 文件内必须包含 schema version。

## 2. 维度顺序

全局约定：

```text
tx: transmitter index
rx: receiver index
rx_ant: receiver antenna index
tx_ant: transmitter antenna index
path: path index
depth: interaction depth index
snapshot: time/snapshot index
subcarrier: frequency index
xyz: 3D coordinate component
```

核心张量：

```text
H_true:
  [tx, rx, rx_ant, tx_ant, subcarrier]

H_obs / cfr_est:
  [snapshot, tx, rx, rx_ant, tx_ant, subcarrier]

path scalar:
  [tx, rx, rx_ant, tx_ant, path]

path interaction data:
  [tx, rx, rx_ant, tx_ant, path, depth]

path vertices:
  [tx, rx, rx_ant, tx_ant, path, depth, xyz]
```

Sionna RT 原生 Paths 常用 `rx` 在前。adapter 必须转换为本文定义的 TX-first 顺序。

## 3. HDF5 顶层结构

```text
/meta
/input
/topology
/devices
/antenna
/scene
/frequency
/channel/truth
/paths/full
/paths/samples
/motion
/waveform
/observation
/impairments
/receiver
/evaluation
/calibration
/runtime
```

## 4. `/meta`

必选：

```text
schema_version                 string, e.g. "1.0.0"
contract_name                  string, "sionna_measurement_sim_hdf5"
producer                       string
created_at                     string ISO-8601
run_id                         string
git_commit                     string or empty
random_seed                    int64
coordinate_system              string
unit_convention                string
index_order                    string, "tx,rx,rx_ant,tx_ant,..."
truth_branch_enabled           bool
observation_branch_enabled     bool
measurement_realism_level      string
config_snapshot                string YAML/JSON
software_versions              string JSON
```

建议：

```text
notes                          string
source_document_version         string
```

## 5. `/input`

必选：

```text
label_file                     string
scene_file                     string
input_dataset_id               string
input_schema                   string
```

建议：

```text
original_label_json             string JSON
label_conflicts                 string JSON list
```

## 6. `/topology`

必选：

```text
tx_positions_m                 float32 [tx, 3]
rx_positions_m                 float32 [rx, 3]
tx_labels                      string [tx]
rx_labels                      string [rx]
```

建议：

```text
tx_group_id                    string/int [tx]
rx_group_id                    string/int [rx]
link_mask                      bool [tx, rx]
```

## 7. `/devices`

用于保存 TX/RX 自身状态。即使第一版全为静态，也应保存默认值。

必选：

```text
tx_velocity_mps                float32 [snapshot, tx, 3]
rx_velocity_mps                float32 [snapshot, rx, 3]
tx_orientation_rad             float32 [snapshot, tx, 3]
rx_orientation_rad             float32 [snapshot, rx, 3]
```

方向约定：

```text
orientation_rad[..., 0] = yaw
orientation_rad[..., 1] = pitch
orientation_rad[..., 2] = roll
```

建议：

```text
tx_position_m_by_snapshot       float32 [snapshot, tx, 3]
rx_position_m_by_snapshot       float32 [snapshot, rx, 3]
tx_device_type                  string [tx]
rx_device_type                  string [rx]
tx_power_dbm                    float32 [snapshot, tx]
rx_noise_figure_db              float32 [snapshot, rx]
```

## 8. `/antenna`

必选：

```text
tx_array_type                  string
rx_array_type                  string
tx_num_rows                    int32
tx_num_cols                    int32
rx_num_rows                    int32
rx_num_cols                    int32
tx_num_ant                     int32
rx_num_ant                     int32
tx_spacing_lambda              float32 [2]
rx_spacing_lambda              float32 [2]
tx_polarization                string
rx_polarization                string
tx_pattern                     string
rx_pattern                     string
synthetic_array                bool
```

建议：

```text
tx_element_positions_m         float32 [tx_ant, 3]
rx_element_positions_m         float32 [rx_ant, 3]
tx_element_orientation_rad     float32 [tx_ant, 3]
rx_element_orientation_rad     float32 [rx_ant, 3]
tx_polarization_basis          string
rx_polarization_basis          string
```

## 9. `/scene`

必选：

```text
scene_name                     string
scene_file                     string
material_policy                string
```

建议：

```text
object_id                      uint32 [object]
object_name                    string [object]
object_material                string [object]
object_velocity_mps            float32 [snapshot, object, 3]
object_is_dynamic              bool [object]
object_bbox_min_m              float32 [object, 3]
object_bbox_max_m              float32 [object, 3]
```

说明：

- `object_id` 必须能够与 `/paths/full/object_id` 对齐。
- 如果 Sionna/Mitsuba object id 在不同加载方式下不稳定，应额外保存 `object_name` 和材质名称。

## 10. `/frequency`

必选：

```text
center_frequency_hz            float64
bandwidth_hz                   float64
num_subcarriers                int32
subcarrier_spacing_hz          float64
frequencies_hz                 float64 [subcarrier]
```

建议：

```text
active_subcarrier_mask         bool [subcarrier]
dc_subcarrier_index            int32
```

## 11. `/channel/truth`

必选：

```text
cfr                            complex64 [tx, rx, rx_ant, tx_ant, subcarrier]
path_power_db                  float32 [tx, rx]
has_geometric_signal           bool [tx, rx]
geometric_path_count           int32 [tx, rx]
los_exists                     bool [tx, rx]
nlos_exists                     bool [tx, rx]
```

可选：

```text
cfr_snapshots                  complex64 [snapshot, tx, rx, rx_ant, tx_ant, subcarrier]
```

建议：

```text
cir_coefficients               complex64 [snapshot, tx, rx, rx_ant, tx_ant, path]
cir_delays_s                   float32 [tx, rx, rx_ant, tx_ant, path]
delay_doppler_response         complex64 optional
```

命名要求：

- 禁止在新系统中把 truth CFR 命名为 `/channel/cfr` 作为主路径。
- 兼容旧工具时可以创建只读别名或 reader fallback，但新 writer 必须写 `/channel/truth/cfr`。

## 12. `/paths/full`

这是路径级完整数据。第一版如果担心文件过大，可配置是否保存全量，但 schema 必须从一开始定义。

必选或强烈建议：

```text
valid                          bool [tx, rx, rx_ant, tx_ant, path]
a                              complex64 [tx, rx, rx_ant, tx_ant, path]
tau_s                          float32 [tx, rx, rx_ant, tx_ant, path]
doppler_hz                     float32 [tx, rx, rx_ant, tx_ant, path]
theta_t_rad                    float32 [tx, rx, rx_ant, tx_ant, path]
phi_t_rad                      float32 [tx, rx, rx_ant, tx_ant, path]
theta_r_rad                    float32 [tx, rx, rx_ant, tx_ant, path]
phi_r_rad                      float32 [tx, rx, rx_ant, tx_ant, path]
interaction_type               uint32 [tx, rx, rx_ant, tx_ant, path, depth]
object_id                      uint32 [tx, rx, rx_ant, tx_ant, path, depth]
primitive_id                   uint32 [tx, rx, rx_ant, tx_ant, path, depth]
vertices_m                     float32 [tx, rx, rx_ant, tx_ant, path, depth, 3]
path_type                      string [tx, rx, rx_ant, tx_ant, path]
path_depth                     int32 [tx, rx, rx_ant, tx_ant, path]
```

字段语义：

- `vertices_m` 是每条路径在场景中的交互点位置。NLoS 的反射点、折射点、绕射点等都应在这里。
- `interaction_type` 与 `vertices_m` 的 `depth` 维一一对应。
- `object_id` 与 `primitive_id` 表示每个交互点命中的场景对象和 mesh primitive。
- 无效交互使用 Sionna 常量对应的 invalid id，并在 adapter 中同时用 `path_depth` 标出有效深度。

可选：

```text
source_position_m              float32 [tx, tx_ant, 3] or [tx, 3]
target_position_m              float32 [rx, rx_ant, 3] or [rx, 3]
path_length_m                  float32 [tx, rx, rx_ant, tx_ant, path]
reflection_count               int32 [tx, rx, rx_ant, tx_ant, path]
refraction_count               int32 [tx, rx, rx_ant, tx_ant, path]
diffraction_count              int32 [tx, rx, rx_ant, tx_ant, path]
diffuse_count                  int32 [tx, rx, rx_ant, tx_ant, path]
```

## 13. `/paths/samples`

用于可视化和轻量分析。

必选：

```text
sampled_link_indices           int32 [sample, 2]   # [tx_idx, rx_idx]
sampled_rx_ant_indices         int32 [sample]      # rx antenna index per sample
sampled_tx_ant_indices         int32 [sample]      # tx antenna index per sample
sampled_path_indices           int32 [sample, sample_path]
path_count                     int32 [sample]
path_gain_db                   float32 [sample, sample_path]
path_type                      string [sample, sample_path]
vertices_m                     float32 [sample, sample_path, max_vertices, 3]
vertex_count                   int32 [sample, sample_path]
interaction_type               uint32 [sample, sample_path, max_depth]
object_id                      uint32 [sample, sample_path, max_depth]
primitive_id                   uint32 [sample, sample_path, max_depth]
doppler_hz                     float32 [sample, sample_path]
tau_s                          float32 [sample, sample_path]
```

## 14. `/motion`

用于动态快照和 Doppler 相关实验。

必选：

```text
snapshot_id                    int64 [snapshot]
timestamp_s                    float64 [snapshot]
sampling_frequency_hz          float64
num_time_steps                 int32
```

建议：

```text
mobility_mode                  string   # static, doppler_synthetic, retrace_positions
trajectory_id                  string
doppler_bins_hz                float32 [doppler_bin]
delay_bins_s                   float32 [delay_bin]
```

## 15. `/waveform`

必选，如果 observation 分支启用：

```text
standard                       string
sample_rate_hz                 float64
fft_size                       int32
cp_length                      int32
num_ofdm_symbols               int32
pilot_indices                  int32 [...]
data_subcarrier_indices        int32 [...]
pilot_symbols                  complex64 [...]
tx_power_dbm                   float32
```

建议：

```text
resource_grid_description      string JSON
frame_format                   string JSON
```

## 16. `/observation`

必选，如果 observation 分支启用：

```text
cfr_est                        complex64 [snapshot, tx, rx, rx_ant, tx_ant, subcarrier]
valid_mask                     bool [snapshot, tx, rx]
detection_success              bool [snapshot, tx, rx]
estimation_success             bool [snapshot, tx, rx]
snr_db                         float32 [snapshot, tx, rx]
rssi_dbm                       float32 [snapshot, tx, rx]
noise_power_dbm                float32 [snapshot, tx, rx]
cfo_hz                         float32 [snapshot, tx, rx]
sfo_ppm                        float32 [snapshot, tx, rx]
timing_offset_samples          float32 [snapshot, tx, rx]
phase_offset_rad               float32 [snapshot, tx, rx]
agc_gain_db                    float32 [snapshot, rx]
clipping_flag                  bool [snapshot, tx, rx]
```

建议：

```text
rx_signal_samples              complex64 optional chunked
pilot_observations             complex64 optional
estimator_noise_var            float32 [...]
quality_score                  float32 [snapshot, tx, rx]
failure_reason                 string [snapshot, tx, rx]
```

## 17. `/impairments`

保存模型参数和每次运行采样到的具体值。

必选：

```text
model_version                  string
random_seed                    int64
awgn_config                    string JSON/YAML
cfo_sfo_config                 string JSON/YAML
phase_noise_config             string JSON/YAML
iq_imbalance_config            string JSON/YAML
agc_adc_config                 string JSON/YAML
```

建议：

```text
sampled_cfo_hz                 float32 [snapshot, tx, rx]
sampled_sfo_ppm                float32 [snapshot, tx, rx]
sampled_phase_noise_profile    float32 [...]
sampled_iq_gain_imbalance_db   float32 [...]
sampled_iq_phase_imbalance_rad float32 [...]
```

## 18. `/receiver`

必选：

```text
estimator_type                 string
sync_method                    string
interpolation_method           string
packet_detection_threshold     float32
failure_policy                 string
calibration_profile_id         string
```

## 19. `/evaluation`

必选，如果 truth 和 observation 都启用：

```text
nmse_db                        float32 [snapshot, tx, rx]  # vs impaired+noisy (AWGN isolation)
nmse_db_total                  float32 [snapshot, tx, rx]  # vs clean H_true (total distortion)
amplitude_error_db             float32 [snapshot, tx, rx]
phase_error_rad                float32 [snapshot, tx, rx]
correlation                    float32 [snapshot, tx, rx]
detection_rate                 float32
estimation_failure_rate        float32
```

## 20. `/runtime`

必选：

```text
python_version                 string
sionna_version                 string
sionna_rt_version              string
torch_version                  string
mitsuba_version                string
drjit_version                  string
cuda_available                 bool
cuda_device_name               string
command_line                   string
elapsed_seconds                float64
```

## 21. 文件大小策略

默认策略：

- `/channel/truth/cfr` 全量保存。
- `/paths/full` 可配置保存；debug 和标定数据建议保存，全量大场景可关闭或分块。
- `/paths/samples` 总是保存。
- `/observation/cfr_est` 全量保存。
- raw waveform 默认只保存抽样或关闭。

所有大型 dataset 应使用 chunking 和 compression。

## 22. Schema Validator 最低要求

新系统必须实现一个 schema validator，至少用于测试和 readback。validator 必须检查：

### 22.1 必填 group

RT truth 最小文件必须包含：

```text
/meta
/input
/topology
/devices
/antenna
/scene
/frequency
/channel/truth
/paths/samples
/runtime
```

observation 文件还必须包含：

```text
/waveform
/observation
/impairments
/receiver
/evaluation
```

### 22.2 必填 dataset

最小 truth 文件必须包含：

```text
/meta/schema_version
/meta/index_order
/meta/unit_convention
/topology/tx_positions_m
/topology/rx_positions_m
/devices/tx_velocity_mps
/devices/rx_velocity_mps
/devices/tx_orientation_rad
/devices/rx_orientation_rad
/antenna/tx_polarization
/antenna/rx_polarization
/frequency/frequencies_hz
/channel/truth/cfr
/paths/samples/vertices_m
/paths/samples/interaction_type
/paths/samples/object_id
/paths/samples/primitive_id
/paths/samples/doppler_hz
/paths/samples/tau_s
```

最小 observation 文件必须额外包含：

```text
/observation/cfr_est
/observation/valid_mask
/observation/detection_success
/observation/estimation_success
/observation/snr_db
/evaluation/nmse_db
```

### 22.3 Shape 关系

validator 必须检查：

```text
truth_cfr.ndim == 5
truth_cfr.shape == [tx, rx, rx_ant, tx_ant, subcarrier]
len(frequencies_hz) == truth_cfr.shape[-1]

if observation enabled:
  cfr_est.ndim == 6
  cfr_est.shape[1:] == truth_cfr.shape
  valid_mask.shape == [snapshot, tx, rx]
  nmse_db.shape == [snapshot, tx, rx]
```

### 22.4 数值有效性

validator 或统计测试必须检查：

```text
truth_cfr contains at least one finite value
frequencies_hz is finite and strictly increasing
timestamp_s is monotonically non-decreasing when present
path tau_s is finite for valid sampled paths
doppler_hz is finite for valid sampled paths
vertices_m is finite for valid path vertices
```

### 22.5 禁止路径

新 writer 输出的文件中禁止把 truth CFR 主数据写为：

```text
/channel/cfr
```

兼容旧文件的 reader fallback 可以读取 `/channel/cfr`，但写新文件时必须使用：

```text
/channel/truth/cfr
```

