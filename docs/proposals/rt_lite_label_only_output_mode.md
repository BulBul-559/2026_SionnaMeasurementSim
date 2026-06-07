# RT-lite / Label-only 输出模式设计建议

## 背景

当前系统已经可以通过配置关闭 PHY、ranging、array spectrum、visualization 和
`/paths/full`，得到一个近似 RT-lite 结果。最近的 `front3d_0001 panel0p5`
对比实验说明，配置层压缩已经能把输出从 `1.8G` 降到 `33M`，但它仍不是完整的
label-only 模式：

| 对比项 | Full SRS | 配置版 RT-lite |
|---|---:|---:|
| UE / BS | 144 / 5 | 144 / 5 |
| 并行 | 8 GPU / 8 worker | 1 GPU / 1 worker |
| wall time | 97.13 s | 145.45 s |
| 目录体积 | 1.8G | 33M |
| schema | pass | pass |

配置版 RT-lite 的首个 shard 不再写：

```text
/waveform
/observation
/array
/ranging
/paths/full
```

但仍会写：

```text
/channel/truth/cfr
/channel/truth/cir_coefficients
/channel/truth/cir_delays_s
/paths/samples
/paths/nlos_truth
/derived
```

因此，它适合作为短期低成本实验配置，但不适合作为 3000 场景视觉预训练的最终
label-only 数据格式。

## 目标

建议把 RT-lite 需求拆成两个层级：

1. **一键模式层**：提供面向用户的 `rt_lite` / `rt_labels_only` 模式，避免每次手工改十几个配置项。
2. **模块解耦层**：把“是否计算某类中间结果”和“是否写盘某类 dataset”显式解耦，避免某些内容默认不可配置。

这两个层级不应互相替代。一键模式是常用 preset；细粒度开关是系统长期可维护性的基础。

## 当前耦合点

### 1. PHY 已经可选，但 truth 输出不可选

`phy.enabled=false` 会让 `observation_snr_db=None`，pipeline 跳过 SRS/PUSCH/custom OFDM
observation。这个部分已经比较合理。

问题在于 RT truth 分支仍默认 materialize 并写出：

```text
/channel/truth/cfr
/channel/truth/cir_coefficients
/channel/truth/cir_delays_s
```

这些字段对 full RF 数据集有价值，但对 LoS/path count/path power/delay 这类代理标签不是必需。

### 2. `save_full_paths` 只控制 `/paths/full`

当前 `output.save_full_paths=false` 只会让 `MeasurementSimulationResult.path_table=None`，
从而跳过 `/paths/full`。但 pipeline 内部仍需要 full path table 来构造 derived labels
和 NLoS truth。

这是合理的内部计算策略，但输出控制还不够完整。

### 3. `save_sampled_paths` 配置存在，但当前不是有效写盘开关

schema/config 里有 `output.save_sampled_paths`，但 writer 仍无条件调用 `_write_path_samples()`，
schema validator 也要求 `/paths/samples` 存在。

这使得“只要 compact link labels，不要 path samples”的需求暂时不能通过配置实现。

### 4. Writer 和 schema 以 full measurement contract 为中心

当前 HDF5 writer 按固定顺序写：

```text
meta/input/topology/devices/antenna/scene/frequency/derived
channel/truth
paths/samples
paths/nlos_truth
paths/full
cir_truth
waveform/array/observation/ranging/...
```

schema validator 也把 `/channel/truth/cfr`、`/paths/samples`、`/paths/nlos_truth`
列为 truth contract 的必填项。要支持 label-only，不能简单把这些字段变成全局 optional，
否则会破坏现有 full 数据契约和下游读取假设。

### 5. 计算开关和写盘开关没有分开

只加 `output.exclude_datasets` 只能降低磁盘占用；如果上游仍计算 CFR/CIR/path samples，
时间和显存收益有限。RT-lite 要有价值，需要同时能控制：

```text
compute this?
write this?
validate this?
```

## 建议架构

### 层级一：新增一键输出模式

建议增加模式字段，例如：

```yaml
run:
  mode: "full"          # full | rt_lite | rt_labels_only
```

或者如果不想新增顶层 `run`，也可以放在 output 下：

```yaml
output:
  profile: "full"       # full | rt_lite | rt_labels_only
```

推荐语义：

| profile | 目标 | 默认行为 |
|---|---|---|
| `full` | 当前完整 HDF5 contract | 保持现状，不破坏旧数据 |
| `rt_lite` | 保留 RT truth 的轻量 RF proxy 标签 | 不跑 PHY/ranging/spectrum/viz，不写 waveform/observation/full paths，可保留 NLoS truth |
| `rt_labels_only` | 面向视觉预训练的 compact link-level 标签 | 不写 CFR/CIR/path samples，只写 topology、derived、可选 NLoS summary 和 compact table |

这个模式应只是 preset，不应该在内部复制一套完全独立 pipeline。推荐实现为：

```text
profile -> RTOutputPlan
RTOutputPlan -> compute plan + write plan + schema contract
```

这样以后 WiFi-like、6G-like、RSS-only、AoA-only 等模式也可以复用同一套机制。

### 层级二：显式 compute/write 解耦

建议新增两个配置子树：

```yaml
compute:
  rt_paths: true
  cfr: true
  cir: true
  derived: true
  nlos_truth: true
  path_samples: true
  phy_observation: true
  array_outputs: false
  ranging: false

output:
  datasets:
    topology: true
    derived: true
    channel_truth_cfr: true
    channel_truth_cir: true
    paths_samples: true
    paths_nlos_truth: true
    paths_full: false
    compact_link_table: false
    waveform: true
    observation: true
    array: false
    ranging: false
```

也可以把计算开关放进现有模块：

```yaml
rt:
  outputs:
    cfr: false
    cir: false
    path_samples: false
    nlos_truth: true

output:
  compact_link_table:
    enabled: true
```

关键原则：

- **compute=false 必须阻止上游 materialization**，而不是只在 writer 里丢弃。
- **write=false 只影响落盘**，不应改变其他模块依赖的计算结果。
- 如果某个输出依赖未计算的结果，必须 fail-fast，而不是静默写空数组。

## 推荐数据契约

### Full contract 保持不变

当前 `measurement` HDF5 contract 应继续要求 `/channel/truth/cfr`、`/paths/samples`
等字段。这样现有 SRS/PUSCH、schema 测试和下游脚本不会被打碎。

### RT-lite contract

建议新增 contract name，例如：

```text
contract_name = "sionna_measurement_rt_lite"
```

保留：

```text
/meta
/input
/scene
/topology
/devices
/antenna
/frequency
/link
/derived
/paths/nlos_truth              # 可选，默认开
/channel/truth/path_power_db    # 可选 summary
/channel/truth/geometric_path_count
/channel/truth/los_exists
/channel/truth/nlos_exists
/runtime
/shard
```

可选：

```text
/paths/samples
/channel/truth/cir_delays_s
/channel/truth/cir_coefficients
```

不默认保留：

```text
/channel/truth/cfr
/waveform
/observation
/array
/ranging
/paths/full
```

### RT labels-only contract

建议新增 compact table，HDF5、Parquet 或 NPZ 都可以。HDF5 结构可以是：

```text
/labels/link/tx_index
/labels/link/rx_index
/labels/link/global_tx_index
/labels/link/global_rx_index
/labels/link/tx_xy_m
/labels/link/rx_xy_m
/labels/link/geometric_distance_m
/labels/link/tx_rx_distance_m
/labels/link/tx_rx_bearing_rad
/labels/link/los_flag
/labels/link/nlos_flag
/labels/link/path_count
/labels/link/path_power_db
/labels/link/first_path_delay_s
/labels/link/first_path_range_m
/labels/link/strongest_path_delay_s
/labels/link/first_path_aoa_azimuth_rad
/labels/link/first_path_aoa_zenith_rad
/labels/link/strongest_aoa_azimuth_rad
/labels/link/strongest_aoa_zenith_rad
```

如果未来结合 floorplan/mask 几何代理，还可以扩展：

```text
/labels/link/wall_crossing_count
/labels/link/wall_thickness_integral_m
/labels/link/same_room
/labels/link/geodesic_distance_m
/labels/link/euclidean_geodesic_ratio
/labels/link/proxy_pathloss_db
/labels/link/proxy_rss_dbm
```

这些 floorplan-derived 字段不应硬塞进 Sionna RT adapter；更适合作为独立
geometry proxy module，消费 floorplan/mask/meta 和 topology。

## 需要的重构

### 1. 引入 `RTOutputPlan`

建议新增轻量 dataclass：

```python
@dataclass(frozen=True)
class RTOutputPlan:
    profile: str
    compute_cfr: bool
    compute_cir: bool
    compute_path_samples: bool
    compute_nlos_truth: bool
    compute_compact_link_table: bool
    write_cfr: bool
    write_cir: bool
    write_path_samples: bool
    write_nlos_truth: bool
    write_compact_link_table: bool
```

`MeasurementConfig` 或 `RTTruthRunConfig` 只携带 plan，不在 pipeline 里散落判断。

### 2. 拆分 RT adapter 的 materialization

当前 adapter 结果基本以 full-output 思路组织。建议拆成：

```text
solve_paths()           -> raw Sionna Paths or PathTable
build_derived()         -> DerivedLabels
build_nlos_truth()      -> NLoSPathTruth
build_path_samples()    -> PathSamples
build_cir_truth()       -> CIRTruth
build_cfr_truth()       -> RTTruthResult.cfr
```

其中 `solve_paths()` 是共同底座，其他都是按 plan 可选 materialize。

注意：某些字段如 path power、delay、AoA/AoD 可以直接从 path table 得到，不一定需要
生成全频域 CFR。

### 3. 拆分 writer contract

建议不要把当前 writer 改成到处 `if optional` 的巨型函数。更稳的方式是：

```text
FullMeasurementWriter
RTLiteWriter
RTLabelsWriter
```

它们可以共享 `_write_meta()`、`_write_topology()`、`_write_derived()` 等小函数，
但各自有清晰的 required fields。

### 4. Schema validator profile-aware

`validate_hdf5_contract()` 应根据 `/meta/contract_name` 或显式参数选择 validator：

```python
validate_hdf5_contract(path)
  -> read contract_name
  -> validate_full_measurement()
  -> validate_rt_lite()
  -> validate_rt_labels()
```

这样 current full schema 不需要被放松，label-only schema 也不会被 full schema 的
required CFR/path samples 卡住。

### 5. Manifest 记录 contract/profile

Shard manifest 应记录：

```text
contract_name
output_profile
result_files
global_tx_indices
global_rx_indices
schema_validation
```

下游 reader 可以先看 manifest/profile，再决定按 full HDF5 还是 compact link table 读取。

### 6. Reader / dataloader 后续跟进

RT-lite 和 labels-only 的价值在于服务 3000 场景预训练，因此应尽早提供 reader：

```text
iter_link_labels()
iter_scene_labels()
read_visibility_map()
read_proxy_rss_map()
```

不要让训练脚本自己解析多个 HDF5 group 和 fallback shard。

## 推荐实施顺序

### Phase 0: 配置版临时方案

目的：立即可用，不改代码。

使用配置：

```yaml
phy.enabled: false
ranging.enabled: false
array.spectrum.enabled: false
visualization.enabled: false
calibration.enabled: false
output.save_full_paths: false
carrier.num_subcarriers: 2
antenna.bs_array.num_rows: 1
antenna.bs_array.num_cols: 1
antenna.ue_array.num_rows: 1
antenna.ue_array.num_cols: 1
```

局限：仍写 CFR/CIR/path samples/NLoS truth，不是真正 label-only。

### Phase 1: 一键 `rt_lite` profile

目的：消除手工配置错误。

验收：

- `output.profile: "rt_lite"` 自动关闭 PHY/ranging/spectrum/viz/full paths。
- 保持现有 full schema 不变。
- 同一场景的 `derived` 与 full run 对齐。
- 输出体积不大于配置版 RT-lite。

### Phase 2: 真正 label-only contract

目的：为 3000 场景预训练提供低成本数据。

验收：

- 不写 `/channel/truth/cfr`。
- 不写 CIR 大数组，除非显式开启。
- 不写 `/paths/samples`，除非显式开启。
- 写 compact link table。
- 单场景体积从当前配置版 RT-lite 的 `33M` 进一步降到 `1-5M` 量级。
- 3000 场景预估存储从约 `99G` 降到 `3-15G` 量级。

### Phase 3: Geometry proxy module

目的：把不需要 Sionna 的平面图代理标签接入同一数据生态。

新增模块：

```text
geometry_proxy/
  floorplan_mask.py
  line_of_sight.py
  wall_integral.py
  room_graph.py
  proxy_pathloss.py
```

输出：

```text
wall_crossing_count
wall_thickness_integral
same_room
geodesic_distance
proxy_pathloss
proxy_rss
visibility_map
```

它应独立于 Sionna RT，可单独跑 3000 场景，也可和 20 个 full-sim 场景做校准。

## 预期收益

基于 `front3d_0001 panel0p5` 实验：

| 模式 | 单场景体积 | 3000 场景体积 | 说明 |
|---|---:|---:|---|
| Full SRS | 1.8G | 约 5.4T | 当前完整 RF 产物 |
| 配置版 RT-lite | 33M | 约 99G | 不改代码可达 |
| 真 labels-only | 1-5M | 约 3-15G | 需要新 contract/writer |

时间收益会小于存储收益。只要仍调用 Sionna PathSolver，RT solve 就是基础成本。真正
labels-only 通过跳过 CFR/CIR materialization 和 HDF5 大数组写入，预计能降低写盘和部分
后处理成本；但不会像存储那样直接下降 50 倍。

如果需要更大规模的速度提升，应把 3000 场景主预训练的大部分标签放到 geometry proxy
pipeline 中，只用 Sionna RT-lite/full-sim 做校准和验证。

## 风险与原则

- 不要放松当前 full HDF5 schema 来兼容 label-only；应新增 contract/profile。
- 不要只做 `output.exclude`，否则计算成本仍在。
- 不要让 RT-lite 变成复制粘贴的第二套 pipeline；应共享 RT solve、role mapping、
  sharding、manifest 和 debug tracing。
- 不要把 floorplan 几何代理写进 Sionna adapter；它应该是独立模块。
- 不要静默改变物理配置。例如 `rt_lite` profile 可以关闭输出，但不应偷偷改天线数或
  子载波数；这类压缩参数应由用户或模板显式设置。

## 建议结论

最合适的路线是：

```text
一键 profile 解决易用性
RTOutputPlan 解决模块解耦
profile-aware schema/writer 解决数据契约
compact link table 解决 3000 场景训练成本
geometry proxy module 解决真正大规模低成本预训练
```

当前系统的总体架构已经具备基础：role mapping、shard、manifest、debug tracing、domain
models 和 writer 都有清晰边界。需要重构的不是 PHY/SRS/PUSCH 主链路，而是
RT truth materialization、writer/schema contract 和输出计划这三处。
