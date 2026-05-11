# 01. App 层与配置系统

## CLI 入口 (`app/cli.py`)

文件：`sionna_measurement_sim/app/cli.py`

基于 `argparse` 的命令行入口，提供 6 个子命令：

| 命令 | 功能 | 入口函数 |
|------|------|----------|
| `preflight` | 检查本地环境（Python/Sionna/Torch/Mitsuba/GPU） | `collect_basic_environment()` |
| `run-rt-truth` | 仅 RT 真值 | `run_rt_truth_pipeline()` |
| `run-motion` | RT + 多普勒运动 | `run_rt_truth_pipeline()` |
| `run-observation` | RT + PHY 观测 | `run_rt_truth_pipeline()` |
| `run-full` | 全功能端到端 | `run_rt_truth_pipeline()` |
| `run-batch` | 批量实验 | `run_batch_experiment()` |

核心流程：CLI 解析参数 → 构建 `RTTruthRunConfig` → 调用 `run_rt_truth_pipeline()` → 输出 HDF5 路径。

`run-full` 支持两种模式：
- **有 `--config`**：加载 YAML 配置，CLI 参数可覆盖 YAML 中的对应值
- **无 `--config`**：使用 CLI 参数 + 硬编码默认值（功能受限，不支持完整 NR PUSCH MIMO）

关键参数：
```
--config PATH         YAML 配置文件
--max-tx N            TX（BS）数量
--max-rx N            RX（UE）数量
--snr-db N            信噪比
--phy-standard NAME   custom_ofdm | nr_pusch
--output-dir PATH     输出目录
```

## 批处理 (`app/batch_runner.py`)

文件：`sionna_measurement_sim/app/batch_runner.py`

`run_batch_experiment(base_config, batch_config)` 按 seed 循环运行多个实验。每批生成独立的 HDF5 文件，输出 `batch_manifest.json` 记录每批状态。批间自动清理 GPU 显存。

## 配置 Schema (`config/schema.py`)

文件：`sionna_measurement_sim/config/schema.py`

使用 **Pydantic v2** 定义全量配置模型。顶层类 `MeasurementConfig` 包含 12 个分组：

```python
class MeasurementConfig(BaseModel):
    runtime: RuntimeConfig        # seed, device
    input: InputConfig            # label_file, scene_file, max_tx, max_rx
    output: OutputConfig          # root_dir, hdf5_filename, compression
    carrier: CarrierConfig        # center_frequency_hz, bandwidth_hz, num_subcarriers
    antenna: AntennaConfig        # tx_array, rx_array (ArraySpec)
    rt: RTConfig                  # max_depth, los, specular_reflection, ...
    link: LinkConfig              # duplex_mode, reciprocity_*
    phy: PHYConfig                # standard, snr_db, nr_pusch fields
    impairments: ImpairmentsConfig # awgn, cfo, sfo, phase_noise, timing, agc_adc
    receiver: ReceiverConfig      # estimator_type, mimo_detector, failure_policy
    motion: MotionConfig          # mobility_mode, num_time_steps, velocity
    calibration: CalibrationConfig
```

### 关键 PHY 配置字段

```python
class PHYConfig(BaseModel):
    enabled: bool = True
    standard: str = "custom_ofdm"     # "custom_ofdm" | "nr_pusch"
    snr_db: float = 30.0
    fft_size: int = 64                # custom OFDM
    cp_length: int = 0

    # ── NR PUSCH ──
    mimo_mode: str = "su_mimo"        # "su_mimo" | "mu_mimo"
    channel_backend: str = "apply_ofdm"  # "apply_ofdm" | "cir_dataset_ofdm"
    perfect_csi: bool = False
    ebno_db: float | None = None
    num_prb: int = 16
    num_layers: int = 1
    num_antenna_ports: int = 4
    mcs_index: int = 14
    mcs_table: int = 1
    pusch_dmrs_config_type: int = 1
    pusch_dmrs_length: int = 1
    pusch_dmrs_additional_position: int = 1
    pusch_num_cdm_groups_without_data: int = 2
    # ── MIMO / receiver ──
    mimo_detector: str = "lmmse"
    channel_estimator: str = "pusch_ls"
    receiver_failure_policy: str = "fail_fast"
```

Pydantic 的 `@model_validator` 在加载时自动校验：
- `subcarrier_spacing_hz` 与 `bandwidth_hz / num_subcarriers` 一致
- `fft_size >= 2`
- 速度向量必须是 3 分量

## 配置加载 (`config/loader.py`)

```python
def load_config(path) -> MeasurementConfig   # 加载并校验，失败抛异常
def load_config_or_exit(path) -> MeasurementConfig  # 失败则打印错误并 sys.exit(1)
```

支持 `.yaml`、`.yml`、`.json` 格式。加载后返回完整校验过的 `MeasurementConfig` 对象。

## 配置模板

| 文件 | 用途 |
|------|------|
| `config/defaults/measurement_mvp.yaml` | 通用 custom OFDM（默认） |
| `config/defaults/nr_pusch_mvp.yaml` | NR PUSCH 4x4 SU-MIMO |

模板中字段注释标注了推荐值、可选值和约束条件。完整字段说明见 `config/README.md`。
