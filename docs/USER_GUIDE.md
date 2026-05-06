# SionnaMeasurementSim 用户使用说明书

本文档是系统对外的交付接口约定，后续开发以本文为准。

## 1. 概述

SionnaMeasurementSim 是基于 Sionna 2.x RT 的信道测量仿真系统。核心流程：

```
加载场景 → 射线追踪(RT) → 提取真值CFR和路径数据 → 施加PHY损伤 → AWGN+信道估计 → 评估诊断 → 写入HDF5
```

输出包含两类信道数据：

| 数据类型 | HDF5路径 | 维度 | 含义 |
|---------|---------|------|------|
| **H_true** | `/channel/truth/cfr` | `[tx, rx, rx_ant, tx_ant, subcarrier]` | 射线追踪真值CFR |
| **H_obs** | `/observation/cfr_est` | `[snapshot, tx, rx, rx_ant, tx_ant, subcarrier]` | 经损伤+AWGN+LS估计后的观测CFR |

## 2. 快速开始

```bash
# 安装
uv sync

# 全功能仿真（使用默认配置）
uv run python -m sionna_measurement_sim.app.cli --config config/defaults/measurement_mvp.yaml run-full

# 命令行快速测试
uv run python -m sionna_measurement_sim.app.cli run-full \
    --output-dir outputs/my_test --num-subcarriers 64 --snr-db 30 \
    --max-tx 6 --max-rx 100
```

输出在 `outputs/<run_dir>/` 下：
- `results.h5` — HDF5 数据文件
- `manifest.json` — 运行摘要（配置、诊断、耗时）
- `logs/run.log` — 运行日志

## 3. CLI 命令

### 3.1 全局参数

```
sionna-measurement-sim [--config CONFIG] <command> [args...]
```

| 参数 | 说明 |
|------|------|
| `--config CONFIG` | YAML 配置文件路径（加载后自动 schema 校验，失败立即退出） |
| `--version` | 打印版本号 |

### 3.2 命令列表

| 命令 | 功能 |
|------|------|
| `preflight` | 打印环境信息（Python/Sionna/Torch/Mitsuba/Dr.Jit/GPU） |
| `run-rt-truth` | 仅 RT 真值（射线追踪 + 路径提取 + HDF5） |
| `run-motion` | RT 真值 + 多快照运动/Doppler |
| `run-observation` | RT 真值 + PHY 观测（AWGN + LS + 可选损伤） |
| `run-full` | **全功能端到端**：RT + 路径 + 损伤 + 观测 + 运动 + 校准 + 诊断 |
| `run-batch` | 批量实验（多 seed 自动分批） |

### 3.3 run-full 参数

```
run-full [--label-file LF] [--scene-file SF] [--output-dir OD]
         [--num-subcarriers N] [--seed S] [--snr-db SNR]
         [--cfo-hz CFO] [--sfo-ppm SFO] [--phase-offset-rad PHI]
         [--timing-offset-samples T] [--clipping-threshold CLIP]
         [--num-time-steps NTS] [--sampling-frequency-hz FS]
         [--max-tx MTX] [--max-rx MRX]
```

当同时使用 `--config` 和命令行参数时，命令行参数的非默认值覆盖配置文件对应字段。

### 3.4 run-batch 参数

```
run-batch [--label-file LF] [--scene-file SF] [--output-dir OD]
          [--num-subcarriers N] [--seed S] [--snr-db SNR]
          [--batch-count N]
```

每个 batch 使用独立 seed（`base_seed + batch_idx * 1000`），输出到 `<output_dir>/batch_<NNN>/` 子目录。

批量清单写入 `<output_dir>/batch_manifest.json`：
```json
{
  "batching": {
    "enabled": true,
    "total_batches": 2,
    "completed_batches": 2,
    "failed_batches": 0
  },
  "batches": [
    {"batch_index": 0, "batch_id": "batch_000", "status": "completed", "results_h5": "..."},
    {"batch_index": 1, "batch_id": "batch_001", "status": "completed", "results_h5": "..."}
  ],
  "summary": {"total": 2, "succeeded": 2, "failed": 0}
}
```

## 4. 配置文件

### 4.1 使用方式

```bash
uv run python -m sionna_measurement_sim.app.cli --config config/defaults/measurement_mvp.yaml run-full
```

配置模板：`config/defaults/measurement_mvp.yaml`。详细配置项说明见 `config/README.md`。

### 4.2 配置分组

```yaml
runtime:      # 运行环境（seed, device, precision）
input:        # 输入数据（label_file, scene_file, max_tx, max_rx）
output:       # 输出控制（root_dir, hdf5_filename, save_full_paths）
carrier:      # 载波与频率（center_frequency_hz, bandwidth_hz, num_subcarriers）
antenna:      # 天线阵列（tx_array/rx_array: num_rows, num_cols, polarization）
rt:           # 射线追踪（max_depth, los, specular_reflection, diffraction 等）
phy:          # 物理层（enabled, snr_db, fft_size, channel_estimator）
impairments:  # 损伤模型（awgn/cfo/sfo/phase_noise/timing_offset/agc_adc 各有 enabled 开关）
receiver:     # 接收机（estimator_type, sync_method, failure_policy）
motion:       # 运动多普勒（enabled, mobility_mode, num_time_steps, velocity）
calibration:  # 校准（enabled, profile_id）
```

### 4.3 开关语义

| 开关 | false 时行为 |
|------|-------------|
| `phy.enabled` | 不生成 observation/evaluation |
| `motion.enabled` | num_time_steps=1, velocity=0, 不写 /motion |
| `impairments.*.enabled` | 对应损伤参数为 None（不施加） |
| `calibration.enabled` | 不写 /calibration group |

## 5. HDF5 输出结构

### 5.1 Group 总览

```
results.h5
├── /meta              # 运行元信息
├── /input             # 输入文件路径
├── /topology          # TX/RX 位置
├── /devices           # TX/RX 速度、朝向
├── /antenna           # 阵列配置
├── /scene             # 场景信息
├── /frequency         # 频率网格
├── /channel/truth     # 真值信道
├── /paths/samples     # 采样路径
├── /paths/full        # 全量路径（save_full_paths=true 时）
├── /waveform          # OFDM 波形参数
├── /observation       # 观测信道 + 损伤诊断值
├── /impairments       # 损伤配置快照
├── /receiver          # 接收机配置
├── /evaluation        # 评估指标
├── /calibration       # 校准（calibration.enabled=true 时）
├── /motion            # 运动多普勒（motion.enabled=true 时）
├── /runtime           # 软件版本和耗时
```

### 5.2 `/meta`

| Dataset | 类型 | 说明 |
|---------|------|------|
| `schema_version` | string | 契约版本号 |
| `contract_name` | string | 契约名称 `sionna_measurement_sim_hdf5` |
| `index_order` | string | 维度顺序约定 |
| `unit_convention` | string | 单位约定 |
| `config_snapshot` | string | 本次运行完整配置 JSON |
| `run_id` | string | 运行标识 |
| `random_seed` | int64 | 全局随机种子 |
| `created_at` | string | 创建时间 ISO-8601 |
| `truth_branch_enabled` | bool | 是否启用真值分支 |
| `observation_branch_enabled` | bool | 是否启用观测分支 |

### 5.3 `/topology`

| Dataset | 类型 | 维度 | 说明 |
|---------|------|------|------|
| `tx_positions_m` | float32 | `[tx, 3]` | TX 三维坐标 |
| `rx_positions_m` | float32 | `[rx, 3]` | RX 三维坐标 |
| `tx_labels` | string | `[tx]` | TX 标签 |
| `rx_labels` | string | `[rx]` | RX 标签 |

### 5.4 `/devices`

| Dataset | 类型 | 维度 | 说明 |
|---------|------|------|------|
| `tx_velocity_mps` | float32 | `[snapshot, tx, 3]` | TX 速度 m/s |
| `rx_velocity_mps` | float32 | `[snapshot, rx, 3]` | RX 速度 m/s |
| `tx_orientation_rad` | float32 | `[snapshot, tx, 3]` | TX 朝向 rad |
| `rx_orientation_rad` | float32 | `[snapshot, rx, 3]` | RX 朝向 rad |

### 5.5 `/antenna`

| Dataset | 类型 | 说明 |
|---------|------|------|
| `tx_array_type` | string | TX 阵列类型 `planar` |
| `rx_array_type` | string | RX 阵列类型 |
| `tx_num_rows` | int32 | TX 行数 |
| `tx_num_cols` | int32 | TX 列数 |
| `tx_num_ant` | int32 | TX 总天线数 |
| `rx_num_rows` | int32 | RX 行数 |
| `rx_num_cols` | int32 | RX 列数 |
| `rx_num_ant` | int32 | RX 总天线数 |
| `tx_polarization` | string | TX 极化 `V`/`H` |
| `rx_polarization` | string | RX 极化 |
| `synthetic_array` | bool | 是否合成阵列 |

### 5.6 `/frequency`

| Dataset | 类型 | 维度 | 说明 |
|---------|------|------|------|
| `center_frequency_hz` | float64 | scalar | 中心频率 Hz |
| `bandwidth_hz` | float64 | scalar | 带宽 Hz |
| `num_subcarriers` | int32 | scalar | 子载波数 |
| `subcarrier_spacing_hz` | float64 | scalar | 子载波间隔 Hz |
| `frequencies_hz` | float64 | `[subcarrier]` | 子载波频率列表 |

### 5.7 `/channel/truth`

| Dataset | 类型 | 维度 | 说明 |
|---------|------|------|------|
| `cfr` | complex64 | `[tx, rx, rx_ant, tx_ant, subcarrier]` | 真值 CFR，严格5D |
| `path_power_db` | float32 | `[tx, rx]` | 链路功率 dB |
| `has_geometric_signal` | bool | `[tx, rx]` | 是否有几何路径 |
| `geometric_path_count` | int32 | `[tx, rx]` | 每链路有效路径数 |
| `los_exists` | bool | `[tx, rx]` | 存在 LoS 路径 |
| `nlos_exists` | bool | `[tx, rx]` | 存在 NLoS 路径 |
| `cfr_snapshots` | complex64 | `[snapshot, tx, rx, rx_ant, tx_ant, subcarrier]` | 多快照CFR（可选，motion.enabled=true 且 num_time_steps>1 时） |

### 5.8 `/paths/samples`

每个 TX-RX-天线对的路径数据。sample 维度 = `tx * rx * rx_ant * tx_ant`。

| Dataset | 类型 | 维度 | 说明 |
|---------|------|------|------|
| `sampled_link_indices` | int32 | `[sample, 2]` | `[tx_idx, rx_idx]` |
| `sampled_rx_ant_indices` | int32 | `[sample]` | RX 天线索引 |
| `sampled_tx_ant_indices` | int32 | `[sample]` | TX 天线索引 |
| `sampled_path_indices` | int32 | `[sample, sample_path]` | 路径序号，-1 为无效 |
| `path_count` | int32 | `[sample]` | 每样本有效路径数 |
| `path_gain_db` | float32 | `[sample, sample_path]` | 路径增益 dB |
| `path_type` | string | `[sample, sample_path]` | `los`/`reflection`/`diffuse`/`refraction`/`diffraction`/`mixed`/`invalid` |
| `vertices_m` | float32 | `[sample, sample_path, max_vertices, 3]` | 路径顶点坐标（含TX和RX端点） |
| `vertex_count` | int32 | `[sample, sample_path]` | 有效顶点数 |
| `interaction_type` | uint32 | `[sample, sample_path, max_depth]` | 交互类型码 |
| `object_id` | uint32 | `[sample, sample_path, max_depth]` | 命中对象 ID |
| `primitive_id` | uint32 | `[sample, sample_path, max_depth]` | 命中 primitive ID |
| `doppler_hz` | float32 | `[sample, sample_path]` | 多普勒频移 Hz |
| `tau_s` | float32 | `[sample, sample_path]` | 路径延迟 s |

交互类型码（Sionna RT 常量）：

| 值 | 含义 |
|----|------|
| 0 | NONE（无交互） |
| 1 | SPECULAR（镜面反射） |
| 2 | DIFFUSE（漫反射） |
| 4 | REFRACTION（折射） |
| 8 | DIFFRACTION（绕射） |

### 5.9 `/paths/full`

仅在 `save_full_paths=true` 时写入。path 维度的完整路径表（非采样），shape 同 `/paths/samples` 但 path 维度 = 所有路径数。

### 5.10 `/waveform`

| Dataset | 类型 | 说明 |
|---------|------|------|
| `standard` | string | `custom_ofdm` |
| `sample_rate_hz` | float64 | 采样率 |
| `fft_size` | int32 | FFT 大小 |
| `cp_length` | int32 | CP 长度 |
| `num_ofdm_symbols` | int32 | OFDM 符号数 |
| `pilot_indices` | int32 `[n_pilots]` | 导频子载波索引 |
| `pilot_symbols` | complex64 `[n_pilots]` | 导频符号值 |
| `data_subcarrier_indices` | int32 | 数据子载波索引 |
| `tx_power_dbm` | float32 | 发射功率 dBm |

### 5.11 `/observation`

所有诊断量均为 `[snapshot, tx, rx]` 形状（三维）。

| Dataset | 类型 | 说明 |
|---------|------|------|
| `cfr_est` | complex64 `[snapshot, tx, rx, rx_ant, tx_ant, subcarrier]` | 估计 CFR（6D） |
| `valid_mask` | bool `[snapshot, tx, rx]` | 链路有效性（死链路=false） |
| `detection_success` | bool `[snapshot, tx, rx]` | 检测成功 |
| `estimation_success` | bool `[snapshot, tx, rx]` | 估计成功 |
| `snr_db` | float32 `[snapshot, tx, rx]` | 信噪比 dB |
| `rssi_dbm` | float32 `[snapshot, tx, rx]` | 接收信号强度 dBm |
| `noise_power_dbm` | float32 `[snapshot, tx, rx]` | 噪声功率 dBm |
| `cfo_hz` | float32 `[snapshot, tx, rx]` | 载波频偏 Hz |
| `sfo_ppm` | float32 `[snapshot, tx, rx]` | 采样频偏 ppm |
| `timing_offset_samples` | float32 `[snapshot, tx, rx]` | 定时偏移（采样点） |
| `phase_offset_rad` | float32 `[snapshot, tx, rx]` | 相位偏移 rad |
| `agc_gain_db` | float32 `[snapshot, rx]` | AGC 增益 dB（per-RX） |
| `clipping_flag` | bool `[snapshot, tx, rx]` | 削波标记 |

### 5.12 `/impairments`

| Dataset | 类型 | 说明 |
|---------|------|------|
| `model_version` | string | 损伤模型版本 |
| `random_seed` | int64 | 损伤随机种子 |
| `awgn_config` | string (JSON) | AWGN 配置 |
| `cfo_sfo_config` | string (JSON) | CFO/SFO/定时偏移配置 |
| `phase_noise_config` | string (JSON) | 相位噪声配置 |
| `iq_imbalance_config` | string (JSON) | IQ 不平衡配置 |
| `agc_adc_config` | string (JSON) | AGC/ADC 配置 |

### 5.13 `/receiver`

| Dataset | 类型 | 说明 |
|---------|------|------|
| `estimator_type` | string | `ls` |
| `sync_method` | string | `ideal` |
| `packet_detection_threshold` | float32 | 包检测阈值 |
| `failure_policy` | string | `mark_invalid` |
| `calibration_profile_id` | string | 校准 profile ID |

### 5.14 `/evaluation`

| Dataset | 类型 | 维度 | 说明 |
|---------|------|------|------|
| `nmse_db` | float32 | `[snapshot, tx, rx]` | NMSE vs 损伤后+噪声信道（隔离AWGN效应） |
| `nmse_db_total` | float32 | `[snapshot, tx, rx]` | NMSE vs 干净 H_true（含损伤失真总量） |
| `amplitude_error_db` | float32 | `[snapshot, tx, rx]` | 幅度误差 dB |
| `phase_error_rad` | float32 | `[snapshot, tx, rx]` | 相位误差 rad |
| `correlation` | float32 | `[snapshot, tx, rx]` | 相关系数 |
| `detection_rate` | float32 | scalar | 检测成功率 |
| `estimation_failure_rate` | float32 | scalar | 估计失败率 |

### 5.15 `/calibration`

仅当 `calibration.enabled=true` 时写入。

| Dataset | 类型 | 说明 |
|---------|------|------|
| `profile_id` | string | 校准 profile 标识 |
| `fitted_parameters` | string (JSON) | 拟合参数 |
| `validation_metrics` | string (JSON) | 校验指标 |

### 5.16 `/motion`

仅当 `motion.enabled=true` 时写入。

| Dataset | 类型 | 维度 | 说明 |
|---------|------|------|------|
| `snapshot_id` | int64 | `[snapshot]` | 快照 ID |
| `timestamp_s` | float64 | `[snapshot]` | 时间戳（单调递增） s |
| `sampling_frequency_hz` | float64 | scalar | 采样频率 Hz |
| `num_time_steps` | int32 | scalar | 快照总数 |
| `mobility_mode` | string | scalar | `static` / `doppler_synthetic` |

### 5.17 `/runtime`

| Dataset | 类型 | 说明 |
|---------|------|------|
| `python_version` | string | Python 版本 |
| `sionna_version` | string | Sionna 版本 |
| `sionna_rt_version` | string | Sionna RT 版本 |
| `torch_version` | string | PyTorch 版本 |
| `mitsuba_version` | string | Mitsuba 版本 |
| `drjit_version` | string | Dr.Jit 版本 |
| `cuda_available` | bool | CUDA 是否可用 |
| `cuda_device_name` | string | GPU 设备名 |
| `command_line` | string | 运行命令 |
| `elapsed_seconds` | float64 | 运行耗时 s |

## 6. 维度约定

全局维度顺序：

```
tx          发射机索引
rx          接收机索引
rx_ant      接收天线索引
tx_ant      发射天线索引
snapshot    时间快照索引
subcarrier  子载波索引
sample      路径样本索引（= tx * rx * rx_ant * tx_ant）
sample_path 样本内路径索引
depth       交互深度索引
xyz         3D 坐标分量
```

核心张量维度：

```
H_true (CFR truth):      [tx, rx, rx_ant, tx_ant, subcarrier]          # 5D
H_obs  (CFR estimate):   [snapshot, tx, rx, rx_ant, tx_ant, subcarrier] # 6D
CFR snapshots (optional): [snapshot, tx, rx, rx_ant, tx_ant, subcarrier] # 6D
Path scalar:              [tx, rx, rx_ant, tx_ant, path]
Path interaction:         [tx, rx, rx_ant, tx_ant, path, depth]
Path vertices:            [tx, rx, rx_ant, tx_ant, path, depth, 3]
Link diagnostic:          [snapshot, tx, rx]
```

Sionna RT 原生输出为 RX-first 顺序，adapter 自动转换为 TX-first。所有公开接口均为 TX-first。

## 7. 损伤模型

### 7.1 施加顺序

```
H_true → IFFT → CFO(时域) → FFT → SFO → 相位偏移 → 定时偏移 → AGC/ADC → +AWGN → H_obs
```

### 7.2 各损伤效果

| 损伤 | 参数 | 效果 |
|------|------|------|
| CFO | `cfo_hz` (Hz) | 时域相位旋转 `exp(j·2π·cfo·t)`，产生 ICI |
| SFO | `sfo_ppm` | 子载波间相位斜坡 |
| 相位偏移 | `phase_offset_rad` | 常数相位旋转 |
| 定时偏移 | `timing_offset_samples` | 子载波间线性相位 |
| AGC | `agc_gain_db` | 幅度缩放 |
| ADC削波 | `clipping_threshold` | 幅度超过阈值则截断 |
| AWGN | `snr_db` | 加性复高斯噪声 |

### 7.3 评估指标

系统计算两组 NMSE：

- **`nmse_db`** (vs 损伤后+噪声)：`||H_obs - H_impaired||² / ||H_impaired||²` — 仅反映 AWGN 噪声
- **`nmse_db_total`** (vs 干净真值)：`||H_obs - H_true||² / ||H_true||²` — 反映损伤+噪声总失真

## 8. 架构约束

以下约束在代码中强制（validator 校验）：

1. 业务层不直接 import Sionna；所有 Sionna 调用集中在 `adapters/sionna_rt/`
2. HDF5 writer 只消费 domain models，不读取 Sionna 原生对象
3. Domain dataclasses 不保存 Sionna 原生对象
4. 新系统不从 `old/` 导入代码
5. Truth CFR 主路径为 `/channel/truth/cfr`（禁止使用旧路径 `/channel/cfr`）
6. 大文件（`.h5`/`.npy`/`.npz`/`.pt`/`.pth`）不入 git
7. `.venv/` 和 `outputs/` 不入 git

## 9. 开发和测试

```bash
uv sync                    # 安装依赖
uv run ruff check .        # 代码规范检查
uv run pytest              # 运行全部测试（74 项）
uv run pytest tests/unit tests/statistical -k "impairment"  # 按关键词筛选
```
