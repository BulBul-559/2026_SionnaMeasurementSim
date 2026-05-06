# 04. Sionna RT Adapter 与路径级数据

本文定义 Sionna 2.x RT adapter 的职责和路径级数据提取要求。落盘字段必须遵循 [03_data_contract_hdf5.md](03_data_contract_hdf5.md)。

## 1. 对 NLoS 路径中间反射点的结论

可以获取。

在 Sionna 2.0.1 中，`Paths.vertices` 表示路径与场景交互点的坐标，也就是 NLoS 路径的反射、折射、绕射、散射等中间点。`Paths.interactions` 表示每个 depth 的交互类型，`Paths.objects` 表示每个交互点命中的对象 ID，`Paths.primitives` 表示命中的 mesh primitive。

官方 Sionna 2.0.1 文档说明 `Paths` 包含：

- `a`：路径系数。
- `tau`：路径延迟。
- `theta_t`、`phi_t`：发射角。
- `theta_r`、`phi_r`：接收角。
- `doppler`：每条路径 Doppler shift。
- `interactions`：交互类型。
- `objects`：相交对象 ID。
- `primitives`：相交 primitive ID。
- `vertices`：路径交互点坐标。
- `valid`：有效路径 mask。

参考：

- https://nvlabs.github.io/sionna/rt/api/paths.html
- https://nvlabs.github.io/sionna/rt/developer/dev_understanding_paths.html
- https://nvlabs.github.io/sionna/rt/tutorials/Mobility.html
- https://nvlabs.github.io/sionna/rt/tutorials/Introduction.html

开发约束：

- 每次升级 Sionna 版本后，必须重新运行 adapter shape 测试。
- adapter 中必须记录 Sionna 原始 shape 和转换后的内部 shape。
- adapter 测试必须覆盖 `out_type="numpy"`；若使用 PyTorch 张量，还必须覆盖 `out_type="torch"` 或对应 Sionna 2.x 支持路径。
- 若官方 API 字段不存在或 shape 与本文不一致，禁止在业务层临时绕过；必须先更新本文件和 adapter 测试。

## 2. Adapter 设计目标

Sionna RT adapter 的目标不是把 Sionna 对象到处传，而是把 Sionna 2.x 的 `Scene`、`Paths`、CIR/CFR 结果转换成系统内部稳定数据结构。

核心输出：

```text
RTTruthResult
PathTable
PathSummary
SceneObjectTable
```

禁止：

- 业务层直接依赖 `paths.vertices.numpy()[...]` 这种索引。
- HDF5 writer 直接读 Sionna `Paths`。
- 把 Sionna 原生 rx-first shape 泄漏到系统主数据契约。

## 3. PathAdapter 输入输出

输入：

```text
sionna.rt.Paths
Scene object map
Topology
Antenna spec
RT config
```

输出：

```text
PathTable:
  valid
  a
  tau_s
  doppler_hz
  theta_t_rad
  phi_t_rad
  theta_r_rad
  phi_r_rad
  interaction_type
  object_id
  primitive_id
  vertices_m
  path_type
  path_depth
  source_position_m
  target_position_m
```

所有输出 shape 必须转换为 [03_data_contract_hdf5.md](03_data_contract_hdf5.md) 中的 TX-first 约定。

## 4. Sionna Paths 属性映射

| Sionna `Paths` 属性 | 内部字段 | 说明 |
|---|---|---|
| `paths.a` | `a` | 路径复系数。Sionna 返回实部/虚部 tuple 时 adapter 合成为 complex。 |
| `paths.tau` | `tau_s` | 路径延迟，单位秒。 |
| `paths.doppler` | `doppler_hz` | 每条路径 Doppler shift，单位 Hz。 |
| `paths.theta_t` | `theta_t_rad` | Zenith angle of departure。 |
| `paths.phi_t` | `phi_t_rad` | Azimuth angle of departure。 |
| `paths.theta_r` | `theta_r_rad` | Zenith angle of arrival。 |
| `paths.phi_r` | `phi_r_rad` | Azimuth angle of arrival。 |
| `paths.interactions` | `interaction_type` | 每个 interaction depth 的类型。 |
| `paths.objects` | `object_id` | 每个 interaction depth 命中的场景对象 ID。 |
| `paths.primitives` | `primitive_id` | 每个 interaction depth 命中的 mesh primitive ID。 |
| `paths.vertices` | `vertices_m` | 每个 interaction depth 的 3D 坐标。 |
| `paths.valid` | `valid` | 有效路径 mask。 |
| `paths.sources` | `source_position_m` | 路径源点。synthetic array 时语义需记录。 |
| `paths.targets` | `target_position_m` | 路径目标点。synthetic array 时语义需记录。 |

## 5. InteractionType 约定

Sionna RT 常量：

```text
NONE = 0
SPECULAR = 1
DIFFUSE = 2
REFRACTION = 4
DIFFRACTION = 8
```

内部 `path_type` 推荐规则：

```text
all interactions == NONE -> los
contains SPECULAR -> reflection
contains DIFFUSE -> diffuse
contains REFRACTION -> refraction
contains DIFFRACTION -> diffraction
multiple non-zero kinds -> mixed
```

保留原始 `interaction_type`，不要只保存字符串 `path_type`。

## 6. vertices 与 object_id 的有效性

`vertices_m[..., depth, :]` 与 `interaction_type[..., depth]`、`object_id[..., depth]`、`primitive_id[..., depth]` 一一对应。

无效 depth：

- `interaction_type == NONE`
- `object_id == INVALID_SHAPE`
- `primitive_id == INVALID_PRIMITIVE`
- `vertices` 可能为 `[0, 0, 0]` 或无意义值

adapter 必须计算：

```text
path_depth = count(interaction_type != NONE)
```

后续分析和绘图应使用 `path_depth` 或 valid interaction mask，而不是把 `[0,0,0]` 当真实反射点。

## 7. 对象和 primitive 信息

adapter 应尝试建立：

```text
object_id -> object_name
object_id -> radio_material
object_id -> bbox
object_id -> velocity
```

如果 Sionna 或 Mitsuba object id 在不同运行中不稳定，必须同时保存 object name 和 scene file hash。

primitive 法向量可作为可选字段：

```text
primitive_normal               float32 [tx, rx, rx_ant, tx_ant, path, depth, 3]
```

第一版可不保存全量 primitive normal，但应保留 `primitive_id`，以便后续从 scene object 反查。

## 8. Doppler 与动态

Sionna RT 的 `Paths.doppler` 给出每条路径的 Doppler shift。Doppler 由 TX/RX/交互对象速度和路径方向共同决定。

adapter 和 RT pipeline 必须记录：

```text
tx_velocity_mps
rx_velocity_mps
object_velocity_mps
paths.doppler
sampling_frequency
num_time_steps
```

如果使用 `paths.cfr(..., num_time_steps>1)` 或 `paths.cir(..., num_time_steps>1)` 生成时间演化，必须在 `/motion` 中记录：

```text
mobility_mode = "doppler_synthetic"
sampling_frequency_hz
num_time_steps
timestamp_s
```

如果通过移动场景对象并重新追踪路径生成时间演化，记录：

```text
mobility_mode = "retrace_positions"
```

## 9. Shape 验证

Sionna `Paths` 的 shape 会受 synthetic array、天线阵列、输出类型影响。adapter 必须在 runtime 验证：

- `valid` rank。
- `vertices` rank。
- `interactions` rank。
- `a` 是否为 tuple。
- `num_tx`、`num_rx` 与 topology 一致。
- 路径维、depth 维位置。

任何 shape 假设都必须集中在 `path_adapter.py`，不能散落在业务逻辑中。

## 10. 最小验收

在测试场景 `SionnaMeasurementSim/data/scenes/test/` 上，RT adapter 阶段必须能证明：

- 至少 1 条 LoS 或 NLoS 路径被解析。
- 对 NLoS 路径，能输出中间 `vertices_m`。
- `interaction_type`、`object_id`、`primitive_id` 与 `vertices_m` depth 对齐。
- `doppler_hz` 可读，静态场景允许为 0。
- `H_true` shape 为 `[tx, rx, rx_ant, tx_ant, subcarrier]`。
- 写入 HDF5 后能读回并恢复相同 shape。

验收输出至少包括：

```text
outputs/.../results.h5
outputs/.../manifest.json
outputs/.../logs/run.log
```

`results.h5` 中至少存在：

```text
/channel/truth/cfr
/paths/samples/vertices_m
/paths/samples/interaction_type
/paths/samples/object_id
/paths/samples/primitive_id
/paths/samples/doppler_hz
/paths/samples/tau_s
```
