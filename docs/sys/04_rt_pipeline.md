# 04. RT 真值 Pipeline

文件：`sionna_measurement_sim/rt/truth_pipeline.py`

RT pipeline 是整个系统的编排核心。它接收配置，依次调用 RT adapter、PHY observation、HDF5 writer。

## `RTTruthRunConfig`

用户面配置 dataclass，涵盖 RT、天线、频率、PHY、运动、损伤、链路等全部参数（40+ 字段）：

```python
@dataclass(frozen=True)
class RTTruthRunConfig:
    # ── 输入 ──
    label_file: Path
    scene_file: Path
    output_dir: Path
    scene_id: str = ""
    map_id: str = ""
    max_bs: int = 1
    max_ue: int = 1

    # ── 频率 ──
    center_frequency_hz: float = 3.5e9
    bandwidth_hz: float = 20e6
    num_subcarriers: int = 8

    # ── 天线 ──
    bs_num_rows: int = 1; bs_num_cols: int = 1
    ue_num_rows: int = 1; ue_num_cols: int = 1
    bs_polarization: str = "V"; ue_polarization: str = "V"
    bs_pattern: str = "iso"; ue_pattern: str = "iso"
    bs_spacing_lambda: tuple = (0.5, 0.5)
    ue_spacing_lambda: tuple = (0.5, 0.5)

    # ── RT ──
    seed: int = 1
    device: str = "cpu"
    max_depth: int = 1
    los: bool = True
    specular_reflection: bool = True
    diffuse_reflection: bool = False
    refraction: bool = False
    diffraction: bool = False
    synthetic_array: bool = False
    normalize_cfr: bool = False
    normalize_delays: bool = False
    merge_shapes: bool = False

    # ── PHY ──
    phy_standard: str = "custom_ofdm"    # "custom_ofdm" | "nr_pusch" | "nr_srs"
    observation_snr_db: float | None = None  # None = 仅 RT
    observation_seed: int = 11
    impairment_config: ImpairmentConfig | None = None

    # ── NR PUSCH MIMO ──
    subcarrier_spacing_khz: int = 30
    num_prb: int = 16
    num_layers: int = 1
    num_antenna_ports: int = 4
    mcs_index: int = 14
    mcs_table: int = 1
    perfect_csi: bool = False
    ebno_db: float | None = None
    pusch_dmrs_config_type: int = 1
    pusch_dmrs_length: int = 1
    pusch_dmrs_additional_position: int = 1
    pusch_num_cdm_groups_without_data: int = 2
    mimo_mode: str = "su_mimo"
    channel_backend: str = "apply_ofdm"
    mimo_detector: str = "lmmse"
    channel_estimator: str = "pusch_ls"
    receiver_failure_policy: str = "fail_fast"

    # ── 运动 ──
    num_time_steps: int = 1
    sampling_frequency_hz: float = 0.0
    bs_velocity_mps: tuple = (0, 0, 0)
    ue_velocity_mps: tuple = (0, 0, 0)

    # ── 其他 ──
    hdf5_filename: str = "results.h5"
    hdf5_compression: str = "gzip"
    save_full_paths: bool = False
    debug_config: Any | None = None
    output_sharding_config: Any | None = None
    shard_spec: ShardSpec | None = None
    calibration_enabled: bool = True
    link_config: LinkConfig = LinkConfig()
```

## `run_rt_truth_pipeline()`

```python
def run_rt_truth_pipeline(config: RTTruthRunConfig) -> Path
```

普通模式返回输出 HDF5 文件路径。启用 `output.sharding.enabled=true` 时返回输出目录路径，目录下包含多个 `result_xxx.h5` 和一个 aggregate `manifest.json`。

**内部流程：**

```
1. load_role_topology_from_label() → RoleTopology(BS/UE)
2. resolve_link_roles() + resolve_role_topology() → Topology(TX/RX)
3. FrequencyGrid.from_center_bandwidth() → FrequencyGrid
4. run_sionna_rt_truth()          → SionnaRTTruthAdapterResult
   ├── RTTruthResult (CFR)
   ├── CIRTruth
   ├── PathSamples
   └── PathTable (if save_full_paths)
5. if observation_snr_db is not None:
   └── get_phy_module(phy_standard).run(PHYContext(...))
       → PHYModuleResult → ObservationResult + Evaluation
6. DiagnosticsReport.from_evaluation()
7. write_measurement_result()        → HDF5
8. validate_hdf5_contract()          → schema check
9. write_manifest()                  → manifest.json
```

**Shard 外壳：**

`run_rt_truth_pipeline()` 会先检查 `output_sharding_config`。当 shard 开启且当前不是子 shard 时，会按 UE 范围创建多个 `ShardSpec`，每个 shard 调用同一个单次 pipeline，但写入独立 `result_{shard_index:03d}.h5`。多进程模式下每个 worker 只写自己的 HDF5，根目录 `manifest.json` 记录所有 shard 的全局 UE/BS 覆盖范围、resolved TX/RX 索引和每个 shard 的性能日志。

**链路方向口径：**

配置和 label 层只表达 BS/UE。`link.phy_link_direction` 决定进入 RT/PHY 前的
TX/RX 映射：`uplink` 为 UE→BS，`downlink` 为 BS→UE。HDF5 中
`/channel/truth/cfr`、`/observation/cfr_est` 等张量始终使用 resolved TX/RX
link-view，并在 `/link/tx_role`、`/link/rx_role` 记录 TX/RX 分别对应 `ue` 还是
`bs`。旧的 `rt_trace_direction`、`reciprocity_*` 用户配置口径已移除；低层 legacy
transpose 只作为内部测试/兼容路径存在。

**PHY module 分支** (`phy/modules.py`):
- `custom_ofdm` 适配现有 `run_awgn_ls_observation()`
- `nr_pusch` 适配现有 `run_nr_pusch_observation()`，并保留 waveform grid、array 输出和 batching 统计
- `nr_srs` 调用 `run_nr_srs_observation()`，写 SRS-like full-band sounding 的 `srs_*` grid 和 LS CSI
- pipeline 在 derived labels 可用后统一补齐 `/array` AoA label 和空间谱

## 关键设计点

1. **单配置对象**：`RTTruthRunConfig` 承载了从 RT 到 PHY 到 MIMO 的全部参数，避免多个配置对象同步问题
2. **PHY 可选**：`observation_snr_db=None` 时跳过 PHY 链，仅输出 RT 真值
3. **插件化 PHY**：`phy_standard` 通过 registry 选择 custom OFDM、NR PUSCH 或 NR SRS-like 模块
4. **HDF5 schema 校验内置**：每次运行结束自动调用 `validate_hdf5_contract()`
5. **Debug profiling 可选**：`debug.enabled=true` 时记录阶段耗时、GPU/CPU/RSS 采样和每 shard summary；默认关闭，不影响普通运行。
