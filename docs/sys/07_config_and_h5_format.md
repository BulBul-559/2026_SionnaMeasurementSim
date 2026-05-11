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

`MeasurementConfig` 包含 12 个分组：

```yaml
runtime:       # 运行环境 (seed, device, precision)
input:         # 输入数据 (label_file, scene_file, max_tx, max_rx)
output:        # 输出控制 (root_dir, hdf5_filename, compression)
carrier:       # 载波频率 (center_frequency_hz, bandwidth_hz, num_subcarriers)
antenna:       # 天线阵列 (tx_array, rx_array)
rt:            # 射线追踪 (max_depth, los, specular_reflection, ...)
link:          # 链路配置 (duplex_mode, reciprocity_*)
phy:           # 物理层 (standard, snr_db, nr_pusch fields)
impairments:   # 损伤模型 (awgn, cfo, sfo, phase_noise, timing, agc_adc)
receiver:      # 接收机 (estimator_type, mimo_detector, failure_policy)
motion:        # 运动/多普勒 (mobility_mode, num_time_steps, velocity)
calibration:   # 校准 (profile_id)
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

#### `input` — 输入数据

| 字段 | 类型 | 默认值 | 约束 | 说明 |
|------|------|--------|------|------|
| `label_file` | str | `"data/scenes/test/test5.json"` | — | TX/RX 位置标签 JSON |
| `scene_file` | str | `"data/scenes/test/scene.xml"` | — | Mitsuba 场景 XML |
| `scene_id` | str | scene 文件名 stem | — | 与地图/平面图系统对齐的稳定场景 ID |
| `map_id` | str | `""` | — | 可选地图版本 ID |
| `label_schema` | str | `"simplesionna_v1"` | — | 标签 schema 版本 |
| `coordinate_system` | str | `"scene_local_xyz_m"` | — | 坐标系统 |
| `max_tx` | int | 6 | ≥1 | TX（BS）数量上限 |
| `max_rx` | int | 100 | ≥1 | RX（UE）数量上限 |

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

每侧天线包含 `tx_array` 和 `rx_array`，均为 `ArraySpec`：

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

天线数 = `num_rows × num_cols`。4x4 MIMO 需两侧均为 2×2。

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
| `rt_trace_direction` | str | `"bs_to_ue"` | RT 追踪方向 |
| `reciprocity_mode` | str | `"transpose_rt_channel"` | 互易性模式 |
| `reciprocity_applied` | bool | true | 是否应用 TDD 互易性 |

#### `phy` — 物理层

**通用字段**（custom OFDM 和 NR PUSCH 共享）：

| 字段 | 类型 | 默认值 | 约束 | 说明 |
|------|------|--------|------|------|
| `enabled` | bool | true | — | 是否启用 PHY 观测 |
| `standard` | str | `"custom_ofdm"` | `"custom_ofdm"`\|`"nr_pusch"` | 波形标准 |
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
| `perfect_csi` | bool | false | 完美 CSI |
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
| `tx_velocity_mps` | [float×3] | [0,0,0] | 必须 3 分量 | TX 速度 m/s |
| `rx_velocity_mps` | [float×3] | [0,0,0] | 必须 3 分量 | RX 速度 m/s |

#### `calibration` — 校准

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | true | 是否写 `/calibration` group |
| `profile_id` | str | `"synthetic_default"` | 校准 profile ID |

### 1.4 Pydantic 校验规则

以下校验在 `model_validator` 中自动执行：

- `carrier.subcarrier_spacing_hz` 与 `bandwidth_hz / num_subcarriers` 一致性（1% 容差）
- `phy.fft_size >= 2`
- `motion.tx_velocity_mps` 和 `motion.rx_velocity_mps` 必须是 3 分量
- `motion.mobility_mode == "doppler_synthetic"` 时 `sampling_frequency_hz > 0`

### 1.5 配置模板

| 文件 | 用途 | 关键差异 |
|------|------|----------|
| `config/defaults/measurement_mvp.yaml` | custom OFDM + 全 impairment + motion | `standard: "custom_ofdm"`, fft_size=64, num_subcarriers=64 |
| `config/defaults/nr_pusch_mvp.yaml` | NR PUSCH 4x4 SU-MIMO TDD uplink | `standard: "nr_pusch"`, 4×4 天线, num_prb=4, num_subcarriers=48 |

### 1.6 输入数据格式

#### Label JSON（`input.label_file`）

```json
{
  "scene_file": "data/test/scene.xml",
  "groups": [
    {
      "name": "默认分组",
      "bs_points": [
        {"x": 2.0, "y": 4.0, "z": 2.4, "label": "BS0"},
        ...
      ],
      "ue_points": [
        {"x": 2.0, "y": 4.0, "z": 1.0, "label": "UE0"},
        ...
      ]
    }
  ]
}
```

- 取第一个 `group` 的 `bs_points[:max_tx]` 和 `ue_points[:max_rx]`
- 坐标单位为米（场景本地坐标系）
- 解析器：`sionna_measurement_sim/io/label_parser.py`

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

CLI 中 `run-full --config <path>` 加载 YAML 后，可用 CLI 参数覆盖部分字段（如 `--snr-db`、`--phy-standard`、`--output-dir`）。

### 1.8 MIMO 配置速查

| 场景 | 关键配置 |
|------|----------|
| 4x4 SU-MIMO perfect CSI | `mimo_mode="su_mimo"`, `num_ant=4`, `num_layers=4`, `perfect_csi=true`, `channel_estimator="perfect"` |
| 4x4 SU-MIMO estimated CSI | `perfect_csi=false`, `num_layers=4`, `num_antenna_ports=4` (必须等秩) |
| MU-MIMO | `mimo_mode="mu_mimo"`, `max_rx >= 2`, `dmrs_port_set` 自动不重叠 |

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
- **大数组压缩**：ndim > 0 且 size > 0 的数组启用 gzip + shuffle filter

### 2.2 Group 层级总览

```
results.h5
├── /meta                          # 元数据
├── /input                         # 输入引用
├── /topology                      # TX/RX 位置与标签
├── /devices                       # 设备状态（速度、朝向）
├── /antenna                       # 天线阵列规格
├── /scene                         # 场景引用
├── /frequency                     # 频率网格
├── /derived                       # 距离、ToA/RTT-like、AoA、LoS/NLoS 派生标签
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
| `schema_version` | string | Schema 版本号 |
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

### 2.4 `/input` — 输入引用

| Dataset | 类型 | 说明 |
|---------|------|------|
| `label_file` | string | 标签文件路径 |
| `scene_file` | string | 场景文件路径 |
| `input_dataset_id` | string | 数据集标识 |
| `input_schema` | string | 标签 schema 版本 |

### 2.5 `/topology` — 拓扑

| Dataset | Shape | Unit | 说明 |
|---------|-------|------|------|
| `tx_positions_m` | [num_tx, 3] | m | TX 三维位置 float32 |
| `rx_positions_m` | [num_rx, 3] | m | RX 三维位置 float32 |
| `tx_labels` | [num_tx] | — | TX 标签 string |
| `rx_labels` | [num_rx] | — | RX 标签 string |

### 2.6 `/devices` — 设备状态

| Dataset | Shape | Unit | 说明 |
|---------|-------|------|------|
| `tx_velocity_mps` | [num_tx, 3] | m/s | TX 速度向量 float32 |
| `rx_velocity_mps` | [num_rx, 3] | m/s | RX 速度向量 float32 |
| `tx_orientation_rad` | [num_tx, 3] | rad | TX 朝向角 float32 |
| `rx_orientation_rad` | [num_rx, 3] | rad | RX 朝向角 float32 |

### 2.7 `/antenna` — 天线

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

### 2.8 `/scene` — 场景

| Dataset | 类型 | 说明 |
|---------|------|------|
| `scene_name` | string | 场景名称 |
| `scene_file` | string | 场景文件路径 |
| `scene_id` | string | 跨仿真/平面图系统对齐 ID |
| `map_id` | string | 可选地图版本 ID |
| `material_policy` | string | 材质策略 |

### 2.9 `/frequency` — 频率网格

| Dataset | 类型 | Unit | 说明 |
|---------|------|------|------|
| `center_frequency_hz` | float64 | — | 中心频率 |
| `bandwidth_hz` | float64 | — | 带宽 |
| `num_subcarriers` | int32 | — | 子载波数 |
| `subcarrier_spacing_hz` | float64 | — | 子载波间隔 |
| `frequencies_hz` | [num_subcarriers] float64 | Hz | 子载波频率数组（严格递增） |

### 2.10 `/channel/truth` — 信道真值

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

### 2.11 `/derived` — 派生物理标签

这些字段始终写入，供定位 baseline 和数据转换器复用统一口径。`/paths/full` 是否落盘仍由 `output.save_full_paths` 控制，但派生标签在 pipeline 内部使用完整路径表计算。

| Dataset | Shape | Unit | 说明 |
|---------|-------|------|------|
| `geometric_distance_m` | [tx, rx] | m | TX/RX 三维欧氏距离 |
| `los_distance_m` | [tx, rx] | m | LoS 路径传播距离；无 LoS 为 NaN |
| `first_path_delay_s` | [tx, rx] | s | 最早有效路径 delay；无路径为 NaN |
| `strongest_path_delay_s` | [tx, rx] | s | 最强路径 delay；无路径为 NaN |
| `rtt_like_m` | [tx, rx] | m | `first_path_delay_s * c`，一程传播 range，不是真实 WiFi RTT |
| `rtt_like_s` | [tx, rx] | s | 与 `first_path_delay_s` 同口径 |
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

### 2.12 `/paths/samples` — 采样路径

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

### 2.13 `/paths/full` — 全量路径表

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

### 2.14 `/link` — 链路配置

| Dataset | 类型 | 说明 |
|---------|------|------|
| `duplex_mode` | string | 双工模式 (tdd/fdd) |
| `phy_link_direction` | string | PHY 方向 (uplink/downlink) |
| `rt_trace_direction` | string | RT 追踪方向 |
| `reciprocity_mode` | string | 互易性模式 |
| `reciprocity_applied` | bool | 是否应用互易性 |

### 2.15 `/waveform` — 波形参数（PHY 启用时）

| Dataset | 类型 | 说明 |
|---------|------|------|
| `standard` | string | `"custom_ofdm"` / `"nr_pusch"` |
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
| `noise_variance` | float32 [snap, ul_tx, ul_rx] | 信道施加时使用的噪声方差 |

不保存 `/waveform/tx_time` 或 `/waveform/rx_time`；custom OFDM 暂不写 fake grid，后续另行适配。

### 2.16 `/array` — 阵列观测与标签（NR PUSCH）

| Dataset | Shape | Unit | 说明 |
|---------|-------|------|------|
| `rx_snapshot_matrix` | [snap, ul_tx, ul_rx, ul_rx_ant, ul_rx_ant] | linear_complex | 由 `rx_grid` 聚合的接收阵列协方差/快照矩阵 |
| `aoa_label_rad` | [snap, ul_tx, ul_rx, 2] | rad | `[zenith, azimuth]` AoA 标签；缺失时为 0 |
| `spatial_spectrum_label` | [snap, ul_tx, ul_rx, 91, 181] | linear | 固定角度网格上的空间谱标签；缺失 AoA 时全 0 |
| `angle_grid_rad` | [91, 181, 2] | rad | zenith `[0, pi]`，azimuth `[-pi, pi]` |

### 2.17 `/observation` — 观测结果（PHY 启用时）

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
| `agc_gain_db` | [snap, tx, rx] | float | dB | AGC 增益 |
| `clipping_flag` | [snap, tx, rx] | bool | — | ADC 削波标志 |

> `cfr_est.shape[-5:] == truth.cfr.shape`，即观测 CFR 的 TX/RX/天线/子载波维度与真值一致，仅在前面多一维 snapshot。

### 2.16 `/impairments` — 损伤配置（PHY 启用时）

| Dataset | 类型 | 说明 |
|---------|------|------|
| `model_version` | string | 损伤模型版本 |
| `random_seed` | int64 | 损伤随机种子 |
| `awgn_config` | string | AWGN 配置摘要 |
| `cfo_sfo_config` | string | CFO/SFO 配置摘要 |
| `phase_noise_config` | string | 相位噪声配置摘要 |
| `iq_imbalance_config` | string | IQ 不平衡配置摘要 |
| `agc_adc_config` | string | AGC/ADC 配置摘要 |

### 2.17 `/receiver` — 接收机配置（PHY 启用时）

| Dataset | 类型 | 说明 |
|---------|------|------|
| `receiver_type` | string | 接收机类型 (`"pusch_receiver"` / `"generic"`) |
| `estimator_type` | string | 估计器类型 |
| `sync_method` | string | 同步方法 |
| `mimo_detector` | string | MIMO 检测器 (`"lmmse"` / `"kbest"`) |
| `input_domain` | string | 输入域 |
| `interpolation_method` | string | 插值方法 |
| `packet_detection_threshold` | float32 | 包检测阈值 |
| `failure_policy` | string | 失败策略 |
| `calibration_profile_id` | string | 校准 profile ID |

> NR PUSCH 场景下 `receiver_type` 必须是 `"pusch_receiver"`，`mimo_detector` 必须是 `"lmmse"` 或 `"kbest"`。

### 2.18 `/evaluation` — 评估指标（PHY 启用时）

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

### 2.19 `/calibration` — 校准

| Dataset | 类型 | 说明 |
|---------|------|------|
| `profile_id` | string | 校准 profile ID |
| `fitted_parameters` | string | 拟合参数 |
| `validation_metrics` | string | 校验指标 |

### 2.20 `/motion` — 运动/多普勒

| Dataset | Shape | Unit | 说明 |
|---------|-------|------|------|
| `snapshot_id` | [num_time_steps] int64 | — | 快照序号 |
| `timestamp_s` | [num_time_steps] float64 | s | 时间戳 |
| `sampling_frequency_hz` | scalar float64 | — | 采样频率 |
| `num_time_steps` | scalar int32 | — | 快照数 |
| `mobility_mode` | scalar string | — | `"static"` / `"doppler_synthetic"` |

### 2.21 `/runtime` — 运行环境

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

### 2.22 核心维度约定

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

### 2.23 Schema 校验规则

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

### 2.24 禁止事项

- 禁止将 truth CFR 写为 `/channel/cfr`（必须在 `/channel/truth/cfr`）
- 禁止在 writer 中 import Sionna
- 禁止 dataset 无 `unit` attribute
- 禁止在 HDF5 中存储 Sionna 原生对象（Paths、Tensor 等）
- 禁止在 `domain/` 和 `io/` 中 import Sionna
- 大文件不入 git
