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
    output_profile: str = "full"   # "full" | "rt_labels_only" | "iq_link_library"
    output_products: tuple[str, ...] | None = None
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

普通模式返回输出 HDF5 文件路径。CLI 会把最终有效 YAML 写到输出根目录
`run_config.yaml`。启用 `output.sharding.enabled=true` 时返回输出目录路径，默认目录下包含
`run_config.yaml`、`results/result_xxx.h5`、`manifest/manifest.json`、
`manifest/config_snapshot.json` 和可选 `logs/`。实验性
`output.sharding.bundle.enabled=true` 会额外使用 `bundles/bundle_workerxxx_yyy.h5`
作为 appendable shard bundle。外层队列或 heatmap wrapper 的
`run.log`、`heatmap.log`、`summary.json` 也应写在该输出目录内。

**内部流程：**

```
0. build_rt_output_plan(output_profile, output_products)
1. load_role_topology_from_label() → RoleTopology(BS/UE)
2. resolve_link_roles() + resolve_role_topology() → Topology(TX/RX)
3. FrequencyGrid.from_center_bandwidth() → FrequencyGrid
4. run_sionna_rt_truth()          → SionnaRTTruthAdapterResult
   ├── RTTruthResult (CFR)
   ├── CIRTruth
   ├── PathSamples
   └── PathTable (if save_full_paths)
5. if observation_snr_db is not None or output_profile == "iq_link_library":
   └── get_phy_module(phy_standard).run(PHYContext(...))
       → PHYModuleResult → ObservationResult + Evaluation
       → iq_link_library 时只返回 WaveformSpec + rx_grid_clean extras
6. DiagnosticsReport.from_evaluation()
7. write_measurement_result()        → HDF5
8. validate_hdf5_contract()          → schema check
9. write_manifest()                  → manifest.json
```

`output.profile` 与可选 `output.products` 决定 RT adapter、PHY 和 writer 的取舍：

| profile | 行为 |
|---|---|
| `full` | 当前完整 contract，计算并写 CFR、CIR、path samples，可选 PHY/ranging/array/viz |
| `rt_labels_only` | 使用 `sionna_measurement_rt_labels` contract，只从 PathSolver/path table 生成 `/derived` 和 `/labels/link/*`，跳过 `paths.cfr()`、`paths.cir()`、path samples、PHY 和所有下游观测 |
| `iq_link_library` | 使用 `sionna_measurement_iq_link_library` contract，要求 NR SRS；计算 RT CFR 与 `rx_grid_clean=H*x` 后直接写 clean `/iq/link`，跳过 SRS receiver、CFR estimate、impairment/AWGN、ranging、array/viz 和 full-contract 重型组 |
| `full` + `output.products` | 使用 full contract 的 product-aware 变体；`output.products` 选择关键产物并裁剪计算/写盘。例如 `["cfr_truth"]` 只跑 `paths.cfr()` 并只写 `/channel/truth/cfr` 与必要元数据；`["cfr_obs"]` 会运行 PHY observation 但可不写 truth CFR；`["ranging"]` 会内部计算 observation 供 estimator 使用，但 HDF5 可只写 `/ranging` |

`output.products` 支持的产品名包括 `derived`、`link_labels`、`cfr_truth`、`cir_truth`、`path_samples`、
`nlos_path_truth`、`path_full`、`cfr_obs`、`array`、`ranging`、`iq`、`multiuser`、
`calibration`、`motion`、`visualization` 和 `all`。别名 `rtt` 映射到 `ranging`。
产品计划会设置 RT adapter 的 `compute_cfr/compute_cir/compute_path_samples` 标志，
并在未选择对应产品时关闭 PHY、ranging、array spectrum、IQ、noncooperative、
calibration 和 visualization，避免“只少写不少算”。
`array` 产品是 source-aware：`array.spectrum.sources=["truth_cfr"]` 只需要 RT CFR；
包含 `cfr_est` 或 `rx_grid` 时需要 PHY observation，但可只把 `/array` 写入 HDF5。

schema `2.3.0` 起历史 `rt_lite` 和 `custom` profile 已移除；轻量 full-contract 输出统一用
`profile="full"` + `output.products` 表达。
`ranging`/`rtt` 产品会自动开启 ranging estimator，内部使用 observation CFR，
但不要求把 `/observation` 落盘。`iq` 产品会自动开启 per-link IQ capture；
未显式配置 `phy.iq` 时默认写 clean time IQ，且要求 PHY 标准能导出 waveform grids
（当前为 NR SRS/PUSCH）。`multiuser` 产品会自动开启 `phy.srs.multiuser`，当前只支持
NR SRS；它可内部运行普通 SRS observation，但 HDF5 可以只保留 `/multiuser`。
`path_full` 产品会自动启用 `/paths/full` 写盘，不再依赖旧的 `save_full_paths` 开关。
`calibration` 产品会内部运行 PHY 以生成 synthetic calibration payload，但 HDF5 可以只写
`/calibration`；`motion` 产品可独立写 `/motion`，不需要 PHY；`visualization`
产品会主动启用 `visualization.enabled` 并按当前 plot 配置出图。

`rt_labels_only` 的目标是大规模视觉预训练或场景筛选标签；它不是信道数据，不能用于
需要 CFR/CIR/路径顶点可视化的流程。compact table 可通过
`io.hdf5_reader.iter_link_labels()` 读取单文件或 sharded run。

`iq_link_library` 的目标是构建“逐链路 clean IQ 库”，下游按 UE 组合在线相加并统一添加
噪声/同步偏差/前端损伤；它不是已经混合好的非合作帧。clean payload 由
`phy.iq.clean_output` 选择为 time、frequency 或 both。

**Shard 外壳：**

`run_rt_truth_pipeline()` 会先检查 `output_sharding_config`。当 shard 开启且当前不是子 shard 时，会按 UE 范围创建多个 `ShardSpec`。默认路径下，每个 shard 调用同一个单次 pipeline，并写入独立 `results/result_{shard_index:03d}.h5`。多进程模式下每个 worker 只写自己的 HDF5，`manifest/manifest.json` 记录所有 shard 的全局 UE/BS 覆盖范围、resolved TX/RX 索引和每个 shard 的性能日志。若某个 shard 因 CUDA OOM 或 Dr.Jit 2^32 单数组上限失败，fallback 会把该 UE range 递归拆成更小 result 文件，例如 `result_089_00.h5`、`result_089_01.h5`，最终仍由 manifest 对外呈现连续 UE 覆盖。

默认 sharding 使用固定 GPU 分配：按 `output.sharding.gpu_ids` 轮询把计划 shard 绑定到
worker。可选 `output.sharding.gpu_scheduler.enabled=true` 会改为动态提交 shard：父进程
每 `gpu_scheduler.scan_interval_s` 秒用 `nvidia-smi` 扫描候选 GPU 的空闲显存比例，只在
`free_memory_threshold` 达标且该 GPU 当前没有本任务 shard 运行时提交新 shard。若所有
候选 GPU 都处于高峰，父进程等待下一轮扫描；现阶段动态调度只作用于默认 shard HDF5
路径，bundle append 路径仍保持 worker-owned 连续 shard range。

fallback attempt 的隔离策略由 `output.sharding.fallback.isolation_mode` 控制：
`"always"` 保持历史行为，每次 attempt 都用额外子进程隔离；`"on_failure"` 首次直接在
worker 中运行，只有失败后拆分重试才进入隔离子进程；`"never"` 不做额外隔离。
当使用 `"on_failure"` 时，建议同时设置 `output.sharding.recycle_workers=true`，让
`ProcessPoolExecutor` worker 每个 shard 后回收，释放 Sionna RT / Dr.Jit 显存 allocation。
否则长生命周期 worker 可能在 Python 返回后仍占用 GPU memory，动态调度器会持续认为对应
GPU 不空闲。retryable OOM/Dr.Jit fallback 在拆分子 shard 前会清理失败 exception
traceback/cause/context 引用，再运行子 shard，避免外层 worker 因 traceback frame locals
继续持有失败 attempt 的大 GPU allocation。

默认 shard HDF5 还支持实验性写盘解耦：
`output.sharding.postprocess.async_write=true` 时，计算 worker 返回
`PreparedShardOutput`，父进程用 writer pool 物化 HDF5、schema validate、per-shard
manifest 和日志；`postprocess.max_pending_writes` 控制最多排队/运行多少写盘任务。
这条路径主要用于验证写盘重叠收益。Front3D 0p5 CFR-only smoke 显示 prepared CFR
payload 跨进程返回的成本抵消了写盘重叠，当前生产模板因此默认保持同步 per-shard 写盘。

单个 `run-full` 的动态调度只覆盖当前场景的 shard；如果当前场景尾部只剩 2 个 shard，
它不会自动启动下一个场景。多场景大队列可在模板中设置
`output.sharding.gpu_scheduler.cross_scene_pipeline=true`，或给 `run-scene-index` 加
`--pipeline-shards`：队列层会按 index 顺序展开多个场景的默认 HDF5 shard，并用一个
中心 GPU 占用表统一调度。这样 8 张卡都空闲而当前场景只有 2 个 shard 时，其余卡会
继续接后续场景的 shard；每个场景完成后仍写自己的
`runs/<scene>/manifest/manifest.json`。

显式开启 `output.sharding.bundle.enabled=true` 时，pipeline 改走实验 bundle 外壳：每个
shard 仍完整计算 domain result，但先返回 `PreparedShardOutput`，再由
`HDF5ResultBundleWriter` 把多个 fragment append 到 bundle HDF5。parallel workers 会拿到
连续 shard range，每个 worker 只写自己的 `bundle_worker{worker_index}_*.h5`；当该 worker
达到 `bundle.max_planned_shards_per_bundle` 后再开下一个 bundle 文件，因此不同 worker
不会共享同一个 HDF5 写句柄。manifest result 条目记录 `bundle_h5`、
`bundle_fragment_id`、`append_start` 和 `append_count`。该模式保留 fallback 拆分语义：
bundle 容量按计划 shard 计算，fallback 子 shard 会 append 到当前 worker 的当前 bundle，
所以实际 fragment 数可能超过计划 shard 数；默认生产路径不变。

**链路方向口径：**

配置和 label 层只表达 BS/UE。`link.phy_link_direction` 决定进入 RT/PHY 前的
TX/RX 映射：`uplink` 为 UE→BS，`downlink` 为 BS→UE。HDF5 中
`/channel/truth/cfr`、`/observation/cfr_est` 等张量始终使用 resolved TX/RX
link-view，并在 `/link/tx_role`、`/link/rx_role` 记录 TX/RX 分别对应 `ue` 还是
`bs`。旧的 `rt_trace_direction`、`reciprocity_*` 用户配置口径已移除；低层 legacy
transpose 只作为内部测试/兼容路径存在。

**标准 label 口径：**

`input.label_file` 当前按标准 label `0.1.0` 解析：顶层 `bs_points` 和 `ue_points`
是全场景默认点集，`max_bs`、`max_ue`、`ue_start`、显式 index 和 shard 都作用在这两个
顶层列表上。`groups` 只作为房间/区域/生成策略元数据保留，pipeline 暂不提供
`label_group_policy`，也不会默认取第一个 group。点坐标支持 `position: [x, y, z]`
或显式 `x/y/z`，单位均为米。

**PHY module 分支** (`phy/modules.py`):
- `custom_ofdm` 适配现有 `run_awgn_ls_observation()`，作为 legacy 路径保留
- `nr_pusch` 调用 `run_nr_pusch_observation()`，通过 common link 写统一 waveform grid、array 输出和 batching 统计
- `nr_srs` 调用 `run_nr_srs_observation()`，通过 common link 写 NR SRS subset 的统一 waveform grid、resource datasets、resource LS 和插值后的 full-band CSI
- pipeline 在 derived labels 可用后统一补齐 `/array` AoA label 和空间谱

## 关键设计点

1. **单配置对象**：`RTTruthRunConfig` 承载了从 RT 到 PHY 到 MIMO 的全部参数，避免多个配置对象同步问题
2. **PHY 可选**：`observation_snr_db=None` 时跳过 PHY 链，仅输出 RT 真值或 compact labels
3. **插件化 PHY**：`phy_standard` 通过 registry 选择 custom OFDM、NR PUSCH 或 NR SRS 模块
4. **HDF5 schema 校验内置**：每次运行结束自动调用 `validate_hdf5_contract()`
5. **Debug profiling 可选**：`debug.enabled=true` 时记录阶段耗时、GPU/CPU/RSS 采样和每 shard summary；默认关闭，不影响普通运行。
