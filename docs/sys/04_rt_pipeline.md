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
    max_tx: int = 1
    max_rx: int = 1

    # ── 频率 ──
    center_frequency_hz: float = 3.5e9
    bandwidth_hz: float = 20e6
    num_subcarriers: int = 8

    # ── 天线 ──
    tx_num_rows: int = 1; tx_num_cols: int = 1
    rx_num_rows: int = 1; rx_num_cols: int = 1
    tx_polarization: str = "V"; rx_polarization: str = "V"
    tx_pattern: str = "iso"; rx_pattern: str = "iso"
    tx_spacing_lambda: tuple = (0.5, 0.5)
    rx_spacing_lambda: tuple = (0.5, 0.5)

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
    phy_standard: str = "custom_ofdm"    # "custom_ofdm" | "nr_pusch"
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
    tx_velocity_mps: tuple = (0, 0, 0)
    rx_velocity_mps: tuple = (0, 0, 0)

    # ── 其他 ──
    hdf5_filename: str = "results.h5"
    save_full_paths: bool = False
    calibration_enabled: bool = True
    link_config: LinkConfig = LinkConfig()
```

## `run_rt_truth_pipeline()`

```python
def run_rt_truth_pipeline(config: RTTruthRunConfig) -> Path
```

返回输出 HDF5 文件的路径。

**内部流程：**

```
1. load_topology_from_label()     → Topology
2. FrequencyGrid.from_center_bandwidth() → FrequencyGrid
3. run_sionna_rt_truth()          → SionnaRTTruthAdapterResult
   ├── RTTruthResult (CFR)
   ├── CIRTruth
   ├── PathSamples
   └── PathTable (if save_full_paths)
4. if observation_snr_db is not None:
   ├── if phy_standard == "nr_pusch":
   │     run_nr_pusch_observation()  → ObservationResult + Evaluation
   └── else:
         run_awgn_ls_observation()   → PHYObservationBundle
5. DiagnosticsReport.from_evaluation()
6. write_measurement_result()        → HDF5
7. validate_hdf5_contract()          → schema check
8. write_manifest()                  → manifest.json
```

**NR PUSCH 分支** (`_run_nr_pusch_obs`):
- 从 adapter 结果提取 CIR：`adapter_result.cir_truth.coefficients` 和 `delays_s`
- 调用 `run_nr_pusch_observation(cir_coefficients, cir_delays, link_config, phy_config, carrier_config)`
- 将 `RTTruthRunConfig` 同时作为 `phy_config` 和 `carrier_config` 传入（`RTTruthRunConfig` 包含两者的所有字段）
- 返回 `PHYObservationBundle` + NR PUSCH 补充字段（`waveform_extras`）：num_prb、subcarrier_spacing、slot_number、cyclic_prefix、target_coderate、modulation、num_layers、num_antenna_ports、DMRS 配置

## 关键设计点

1. **单配置对象**：`RTTruthRunConfig` 承载了从 RT 到 PHY 到 MIMO 的全部参数，避免多个配置对象同步问题
2. **PHY 可选**：`observation_snr_db=None` 时跳过 PHY 链，仅输出 RT 真值
3. **双标准**：`phy_standard` 控制走 custom OFDM 还是 NR PUSCH 分支
4. **HDF5 schema 校验内置**：每次运行结束自动调用 `validate_hdf5_contract()`
