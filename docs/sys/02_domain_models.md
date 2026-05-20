# 02. Domain 模型层

`domain/` 目录定义系统内部数据模型。所有类都是纯 Python dataclass，**零 Sionna 依赖**，只使用 `numpy`。数据通过 `__post_init__` 进行 shape/dtype 校验。

## 依赖关系

```
validation.py  ←──  所有其他 domain 文件
    │
    ├── topology.py      (TX/RX 位置)
    ├── antenna.py       (阵列规格)
    ├── frequency.py     (子载波)
    ├── path.py          (路径表)
    ├── cir.py           (CIR 真值)
    ├── channel.py       (CFR 真值)
    ├── observation.py   (观测 + 评估)
    ├── motion.py        (运动/多普勒)
    ├── link.py          (链路方向/互易性)
    │
    └── results.py       ←── 聚合以上所有
```

## 核心模型

### `Topology` (`topology.py`)

```python
@dataclass(frozen=True)
class Topology:
    tx_positions_m: np.ndarray   # [num_tx, 3] float32
    rx_positions_m: np.ndarray   # [num_rx, 3] float32
    tx_labels: np.ndarray        # [num_tx] string
    rx_labels: np.ndarray        # [num_rx] string
```

### `AntennaSpec` (`antenna.py`)

```python
@dataclass(frozen=True)
class AntennaSpec:
    tx_num_rows: int; tx_num_cols: int
    rx_num_rows: int; rx_num_cols: int
    tx_polarization: str; rx_polarization: str
    tx_spacing_lambda: tuple; rx_spacing_lambda: tuple
    tx_pattern: str; rx_pattern: str
    # Properties: tx_num_ant (= rows*cols), rx_num_ant
```

### `FrequencyGrid` (`frequency.py`)

```python
@dataclass(frozen=True)
class FrequencyGrid:
    center_frequency_hz: float
    bandwidth_hz: float
    frequencies_hz: np.ndarray  # [num_subcarriers] float64
    # Factory: from_center_bandwidth(center, bw, nsub)
```

### `RTTruthResult` (`channel.py`)

CFR 真值容器：

```python
@dataclass(frozen=True)
class RTTruthResult:
    cfr: np.ndarray              # [tx, rx, rx_ant, tx_ant, subcarrier] complex64
    path_power_db: np.ndarray    # [tx, rx] float32
    has_geometric_signal: np.ndarray  # [tx, rx] bool
    los_exists: np.ndarray       # [tx, rx] bool
    nlos_exists: np.ndarray      # [tx, rx] bool
    geometric_path_count: np.ndarray  # [tx, rx] int32
    cfr_snapshots: np.ndarray | None  # [snap, tx, rx, rx_ant, tx_ant, subcarrier]
```

### `CIRTruth` (`cir.py`)

```python
@dataclass(frozen=True)
class CIRTruth:
    coefficients: np.ndarray  # [snap, tx, rx, rx_ant, tx_ant, path] complex64
    delays_s: np.ndarray      # [snap, tx, rx, rx_ant, tx_ant, path] float32
    valid: np.ndarray         # [snap, tx, rx, rx_ant, tx_ant, path] bool
```

### `PathTable` (`path.py`)

全量路径表（`/paths/full`）：

```python
@dataclass(frozen=True)
class PathTable:
    valid: np.ndarray          # [tx, rx, rx_ant, tx_ant, path] bool
    a: np.ndarray              # complex path coefficient
    tau_s: np.ndarray          # delay in seconds
    doppler_hz: np.ndarray     # Doppler shift
    theta_t_rad, phi_t_rad     # AoD (zenith, azimuth)
    theta_r_rad, phi_r_rad     # AoA
    interaction_type: np.ndarray  # [..., depth] uint32
    object_id, primitive_id       # per-interaction
    vertices_m: np.ndarray     # [..., depth, 3] 交互点坐标
    path_type: np.ndarray      # "LoS" / "NLoS"
    path_depth: np.ndarray     # 有效交互数
```

### `PathSamples` (`path.py`)

轻量采样路径（`/paths/samples`），用于可视化和快速分析：

```python
@dataclass(frozen=True)
class PathSamples:
    sampled_link_indices: np.ndarray  # [sample, 2]
    sampled_path_indices: np.ndarray  # [sample, sample_path]
    path_count: np.ndarray            # [sample]
    path_gain_db: np.ndarray          # [sample, sample_path]
    vertices_m, interaction_type, object_id, primitive_id
    doppler_hz, tau_s, path_type
```

### `NLoSPathTruth` (`path.py`)

默认写入的轻量 NLoS 路径真值（`/paths/nlos_truth`），独立于
`output.save_full_paths`：

```python
@dataclass(frozen=True)
class NLoSPathTruth:
    valid: np.ndarray            # [tx, rx, rx_ant, tx_ant, path] bool
    aoa_zenith_rad: np.ndarray   # NLoS AoA zenith; invalid/LoS -> NaN
    aoa_azimuth_rad: np.ndarray  # NLoS AoA azimuth; invalid/LoS -> NaN
    aod_zenith_rad: np.ndarray
    aod_azimuth_rad: np.ndarray
    path_power_db: np.ndarray    # 10*log10(abs(a)^2)
    delay_s: np.ndarray
    path_depth: np.ndarray
    path_type: np.ndarray        # NLoS type or "invalid"
```

### `ArraySpectrumConfig` (`array.py`)

空间谱输出配置。默认关闭 Bartlett 空间谱，角度网格默认覆盖 zenith `[0, pi]`、
azimuth `[-pi, pi]`；开启后可分别从 truth CFR、估计 CFR、NR PUSCH/SRS
`rx_grid` 生成谱。NR SRS 仍接受历史兼容别名 `srs_cfr_est`，但它指向的也是
统一 `/observation/cfr_est`。

### `ObservationResult` (`observation.py`)

```python
@dataclass(frozen=True)
class ObservationResult:
    cfr_est: np.ndarray         # [snap, tx, rx, rx_ant, tx_ant, subcarrier]
    valid_mask: np.ndarray      # [snap, tx, rx]
    detection_success: np.ndarray
    estimation_success: np.ndarray
    snr_db, rssi_dbm, noise_power_dbm
    cfo_hz, sfo_ppm, timing_offset_samples, phase_offset_rad
    agc_gain_db, clipping_flag
```

### `RangingResult` (`ranging/result.py`)

`ranging/` 是独立包，但结果对象挂到 `MeasurementSimulationResult.ranging`。它只吃
numpy/domain 输入，不依赖 Sionna 或 HDF5 writer。

```python
@dataclass(frozen=True)
class RangingResult:
    default_estimator: str
    pdp_peak: PdpPeakResult | None
    phase_slope: PhaseSlopeResult | None
```

两个 estimator 的核心输出 shape 均为 `[snapshot, tx, rx]`。检测失败位置写 NaN，
`detection_success=false`；PDP peak 的 `selected_delay_bin` 失败位置写 `-1`。

### `EvaluationResult` (`observation.py`)

```python
@dataclass(frozen=True)
class EvaluationResult:
    nmse_db: np.ndarray         # [snap, tx, rx]  — 主指标: NMSE(H_obs, H_true)
    nmse_db_total: np.ndarray   # 诊断指标: NMSE vs impaired channel
    amplitude_error_db, phase_error_rad, correlation
    detection_rate, estimation_failure_rate
    ber: float; bler: float     # NR PUSCH link-level metrics
    num_bit_errors: int; num_bits: int
    num_block_errors: int; num_blocks: int  # TB CRC
```

### `ReceiverSpec` (`observation.py`)

```python
@dataclass(frozen=True)
class ReceiverSpec:
    receiver_type: str     # "pusch_receiver" / "generic"
    estimator_type: str    # "ls" / "perfect" / "pusch_ls"
    mimo_detector: str     # "lmmse" / "kbest"
    sync_method: str
    failure_policy: str    # "fail_fast" / "mark_invalid"
```

### `LinkConfig` (`link.py`)

```python
@dataclass(frozen=True)
class LinkConfig:
    duplex_mode: str = "tdd"
    phy_link_direction: str = "uplink"
    tx_role: str = "ue"
    rx_role: str = "bs"
```

`tx_role` 和 `rx_role` 是由 `phy_link_direction` 解析出的 link-view 元数据。
配置/label 层使用 BS/UE；RT、PHY 和 HDF5 张量使用 resolved TX/RX。

### `MotionSpec` (`motion.py`)

```python
@dataclass(frozen=True)
class MotionSpec:
    snapshot_id: np.ndarray   # [num_time_steps] int64
    timestamp_s: np.ndarray   # [num_time_steps] float64
    sampling_frequency_hz: float
    num_time_steps: int
    mobility_mode: str        # "static" / "doppler_synthetic"
```

### `MeasurementSimulationResult` (`results.py`)

顶层聚合容器，HDF5 writer 的直接输入。将所有 domain 对象捆绑在一起，并在 `__post_init__` 中交叉校验维度一致性：

```python
@dataclass(frozen=True)
class MeasurementSimulationResult:
    metadata: Metadata
    input_spec: InputSpec
    topology: Topology
    devices: DeviceState
    motion: MotionSpec | None
    antenna: AntennaSpec
    scene: SceneSpec
    frequency: FrequencyGrid
    truth: RTTruthResult
    path_samples: PathSamples
    cir_truth: CIRTruth | None
    path_table: PathTable | None       # save_full_paths=True 时
    waveform: WaveformSpec | None      # observation enabled 时
    observation: ObservationResult | None
    impairments: ImpairmentSpec | None
    receiver: ReceiverSpec | None
    evaluation: EvaluationResult | None
    ranging: RangingResult | None
    calibration: CalibrationResult | None
    link: LinkConfig | None
    waveform_extras: dict | None       # NR PUSCH/SRS 频域 grid 与专属字段
    diagnostics: DiagnosticsReport | None
    runtime: RuntimeInfo
```
