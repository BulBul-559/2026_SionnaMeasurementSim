# 01. App 层与配置系统

## CLI 入口 (`app/cli.py`)

文件：`sionna_measurement_sim/app/cli.py`

基于 `argparse` 的命令行入口，提供 7 类子命令：

| 命令 | 功能 | 入口函数 |
|------|------|----------|
| `preflight` | 检查本地环境（Python/Sionna/Torch/Mitsuba/GPU） | `collect_basic_environment()` |
| `run-rt-truth` | 仅 RT 真值 | `run_rt_truth_pipeline()` |
| `run-motion` | RT + 多普勒运动 | `run_rt_truth_pipeline()` |
| `run-observation` | RT + PHY 观测 | `run_rt_truth_pipeline()` |
| `run-full` | 全功能端到端 | `run_rt_truth_pipeline()` |
| `run-batch` | 批量实验 | `run_batch_experiment()` |
| `benchmark rt/write/spectrum` | 隔离 RT solve、HDF5 writer/schema validate、Bartlett 空间谱成本 | `benchmark.runner` |

核心流程：CLI 解析参数 → 构建 `RTTruthRunConfig` → 调用 `run_rt_truth_pipeline()` → 输出 HDF5 路径。

`--config` 是全局参数，必须放在子命令前：

```bash
uv run python -m sionna_measurement_sim.app.cli \
  --config config/defaults/nr_srs_indoor_positioning_fr1_100mhz.yaml \
  run-full \
  --output-dir outputs/my_nr_srs_run
```

`run-full` 支持两种模式：
- **有 `--config`**：加载 YAML 配置，CLI 参数可覆盖 YAML 中的对应值
- **无 `--config`**：使用 CLI 参数 + 硬编码默认值（功能受限，不支持完整 NR PUSCH MIMO）

YAML 中的 `output.profile` 会在 CLI 层先应用 preset，再写入输出目录的
`run_config.yaml`：

| profile | CLI 侧效果 |
|---|---|
| `full` | 保持完整配置，按 YAML/CLI 启用 RT、PHY、ranging、array、visualization |
| `rt_lite` | 关闭 PHY/ranging/spectrum/visualization/calibration/full paths，仍写 full HDF5 contract |
| `rt_labels_only` | 同样关闭下游重活，并让 pipeline 跳过 CFR/CIR/path samples，写 compact `/labels/link/*` contract |

关键参数：
```
--config PATH         YAML 配置文件
--max-bs N            BS 数量
--max-ue N            UE 数量
--snr-db N            信噪比
--phy-standard NAME   custom_ofdm | nr_pusch | nr_srs
--output-dir PATH     输出目录
```

`benchmark` 是性能工程入口，不生成正式仿真数据契约，默认只接受显式路径或合成参数，
输出到 ignored `outputs/`。三种第一版模式：

| 命令 | 输出 | 用途 |
|------|------|------|
| `benchmark rt` | `benchmark_summary.json`、`benchmark_rows.csv`、`logs/perf_summary*.json` | 复用 RT solve 能力，测 `rt_solve_s`、path_count、truth CFR shape/bytes 和硬件峰值 |
| `benchmark write` | 合成 HDF5 + JSON/CSV summary | 测 writer wall time、schema validate time、文件大小、raw/storage bytes 和 compression ratio |
| `benchmark spectrum` | JSON/CSV summary | 直接调用 Bartlett 空间谱核心，测 per-source time、输出 shape/bytes、chunk count 和 finite sanity |

示例：

```bash
uv run python -m sionna_measurement_sim.app.cli benchmark write \
  --output-dir outputs/benchmark_write_smoke \
  --tx-count 1 --rx-count 2 --rx-ant 2 --subcarriers 16
```

## 批处理 (`app/batch_runner.py`)

文件：`sionna_measurement_sim/app/batch_runner.py`

`run_batch_experiment(base_config, batch_config)` 按 seed 循环运行多个实验。每批生成独立的 HDF5 文件，输出 `batch_manifest.json` 记录每批状态。批间自动清理 GPU 显存。

## 配置 Schema (`config/schema.py`)

文件：`sionna_measurement_sim/config/schema.py`

使用 **Pydantic v2** 定义全量配置模型。顶层类 `MeasurementConfig` 包含这些分组：

```python
class MeasurementConfig(BaseModel):
    runtime: RuntimeConfig        # seed, device
    input: InputConfig            # label_file, scene_file, max_bs, max_ue
    output: OutputConfig          # root_dir, hdf5_filename, compression
    carrier: CarrierConfig        # center_frequency_hz, bandwidth_hz, num_subcarriers
    antenna: AntennaConfig        # bs_array, ue_array (ArraySpec)
    rt: RTConfig                  # max_depth, los, specular_reflection, ...
    link: LinkConfig              # duplex_mode, phy_link_direction
    phy: PHYConfig                # standard, snr_db, NR-family fields
    array: ArrayConfig            # AoA heatmap / Bartlett spectrum
    ranging: RangingConfig        # waveform-level ToA/range observation
    impairments: ImpairmentsConfig # awgn, cfo, sfo, phase_noise, timing, agc_adc
    receiver: ReceiverConfig      # estimator_type, mimo_detector, failure_policy
    motion: MotionConfig          # mobility_mode, num_time_steps, velocity
    calibration: CalibrationConfig
    visualization: VisualizationConfig # sampled PNG reports
```

### 关键 PHY 配置字段

```python
class PHYConfig(BaseModel):
    enabled: bool = True
    standard: str = "custom_ofdm"     # "custom_ofdm" | "nr_pusch" | "nr_srs"
    snr_db: float = 30.0
    fft_size: int = 64                # custom OFDM
    cp_length: int = 0

    # ── NR-family / NR PUSCH ──
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

`config/schema.py` 是 YAML-facing validation model；`ranging/config.py` 是算法运行时
dataclass。CLI 通过 `config/mappers.py::to_domain_ranging_config()` 做集中转换，避免在
CLI 或 pipeline 中手写字段拷贝，也保持 ranging 算法包不依赖 Pydantic。

## 配置模板

| 文件 | 用途 |
|------|------|
| `config/defaults/measurement_mvp.yaml` | 通用 custom OFDM（默认） |
| `config/defaults/nr_pusch_mvp.yaml` | NR PUSCH 4x4 SU-MIMO |
| `config/defaults/nr_pusch_indoor_positioning_fr1_100mhz.yaml` | 室内 FR1 100 MHz PUSCH-DMRS 定位模板 |
| `config/defaults/nr_srs_indoor_positioning_fr1_100mhz.yaml` | 室内 FR1 100 MHz NR SRS subset 定位模板；生产建议从标准 label `0.1.0` 中选择中等密度 UE 采样、UE shard `20` |

模板中字段注释标注了推荐值、可选值和约束条件。完整字段说明见 `config/README.md`。
生产场景通常先按目标输出目录准备一份 `run_config.yaml`，例如
`outputs/<run_name>/run_config.yaml`，再把 `input.label_file`、`input.scene_file`、
`scene_id`、`output.root_dir` 和 GPU/shard 参数改成目标场景。运行 `run-full` 时，
CLI 会把 YAML 加载和命令行覆盖后的最终配置再写回输出目录根部的 `run_config.yaml`，
让结果目录自包含。tmux 队列、heatmap 后处理和验收脚本的附加日志/汇总也应写回同一目录，
常用布局是 `logs/run.log`、`logs/heatmap.log`、`summary.json`。

## 本地数据路径

`data/` 与 `outputs/` 均为 ignored 本地运行路径，可以是 symlink。标准场景目录应提供
Mitsuba `scene.xml`、标准 label `0.1.0` JSON 和可选 floorplan 资源；label 路径通常位于
场景目录的 `label/` 子目录下，具体采样策略和间隔由文件名表达。默认不要递归扫描
`data/` 或 `outputs/`。

## 可视化入口

通过 `--config <path> run-full` 运行时会读取 `visualization` 配置。若 `visualization.enabled=true`，
pipeline 会输出采样诊断图到 `<run_output_dir>/figures/`，并在 `manifest.json`
写入选中的 BS/UE 和图像列表。大规模生产 SRS 模板默认关闭可视化，避免 path
render 和 Matplotlib 开销拖慢全量仿真。

可视化入口会按图的语义分目录写入：普通采样诊断图在 `figures/standard/`，
multi-UE SRS shared observation 图在 `figures/multiuser/`，RSS radio map 在
`figures/heatmaps/`，根目录保留 `index.json` 作为索引。`multiuser_srs` plot
只在 HDF5 存在 `/multiuser` group 时生成，用于检查资源占用、shared RX grid、
per-UE CFR 拆分、error summary 以及 shared/separated 空间谱。

独立 CLI 入口：

```bash
uv run python -m sionna_measurement_sim.app.cli visualize \
  --hdf5 outputs/run/results.h5 \
  --output-dir outputs/run/figures_manual \
  --mode sample
```

支持模式：

| mode | 说明 |
|------|------|
| `sample` | 使用 `visualization.sample_policy` 或 CLI `--sample-policy` 采样 BS/UE 并生成核心诊断图 |
| `selected` | 使用 `--bs-indices`、`--ue-indices` 和 `--plots` 生成指定图 |
| `full` | 生成全量聚合统计图，不逐 link 爆炸出图 |
| `dataset` | 使用 `--dataset-path` 对任意 HDF5 dataset 做 line/heatmap/hist 预览 |

独立 `visualize` 入口也支持 `--sample-policy spatially_spread_valid_links`，
用于在大规模 UE 场景中选择 XY 位置更分散的采样 UE；同时可用
`--sample-ue-count`、`--max-ue`、`--max-bs` 控制采样规模。

可视化图像保持原始采样网格，不做显示插值。涉及子载波的热力图统一把
subcarrier 放在纵轴；CFR lines 例外，使用 subcarrier 横轴。CFR 的
lines、heatmap、error 都分别输出幅度和相位图。空间谱按来源分开输出
label、truth CFR Bartlett、estimated CFR Bartlett、RX grid Bartlett 四类矩形 PNG，
并额外输出对应 polar PNG；polar 图中每个 link 左侧为上半球，右侧为下半球。
空间谱矩形图和 polar 图使用同一个 UE 内的局部颜色尺度；polar 图不放 colorbar。
`multiuser_srs` 会生成 resource ownership、shared RX grid、resource/allocated
CFR、误差摘要、BS 观测示意和 shared/separated 空间谱，图像写在
`figures/multiuser/`。
