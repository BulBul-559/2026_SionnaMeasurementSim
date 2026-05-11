# 03. Sionna RT 适配层

`adapters/sionna_rt/` 封装所有 Sionna RT API 调用。业务层不直接 import Sionna，而是通过 adapter 获取 domain 对象。

参考官方文档：
- [Sionna RT Introduction](https://nvlabs.github.io/sionna/rt/tutorials/Introduction.html)
- [Paths API](https://nvlabs.github.io/sionna/rt/api/paths.html)

## `rt_solver.py` — RT 真值求解器

### 配置

```python
@dataclass(frozen=True)
class SionnaRTConfig:
    scene_file: Path
    seed: int = 1
    max_depth: int = 1
    los: bool = True
    specular_reflection: bool = True
    diffuse_reflection: bool = False
    refraction: bool = False
    diffraction: bool = False
    synthetic_array: bool = False
    normalize_cfr: bool = False
    normalize_delays: bool = False
    num_time_steps: int = 1
    sampling_frequency_hz: float = 0.0
    tx_velocity: tuple = (0, 0, 0)
    rx_velocity: tuple = (0, 0, 0)
    merge_shapes: bool = False
```

### 主函数

```python
def run_sionna_rt_truth(
    topology: Topology,
    antenna: AntennaSpec,
    frequency: FrequencyGrid,
    config: SionnaRTConfig,
) -> SionnaRTTruthAdapterResult
```

**内部流程：**
1. `sionna.rt.load_scene(config.scene_file)` 加载场景
2. 为每个 TX 注册发射机（`scene.add_transmitter`），每个 RX 注册接收机
3. 设置天线阵列（`sionna.rt.PlanarArray`）和载波频率
4. 调用 `sionna.rt.PathSolver` 计算路径
5. 通过 `PathAdapter` 将 Sionna `Paths` 转为 domain `PathTable` 和 `PathSamples`
6. 调用 `paths.cir()` 获取 CIR，通过 `cir_to_ofdm_channel` 转为 CFR
7. 返回 `SionnaRTTruthAdapterResult`

**输出数据结构：**

```python
@dataclass(frozen=True)
class SionnaRTTruthAdapterResult:
    truth: RTTruthResult          # CFR 真值
    path_samples: PathSamples     # 采样路径
    path_table: PathTable | None  # 全量路径表
    cir_truth: CIRTruth           # CIR 真值
    raw_cfr_shape: tuple          # Sionna 原生 shape（用于调试）
    internal_cfr_shape: tuple     # 转换后 project shape
    tx_orientation_rad: np.ndarray  # TX 朝向
    rx_orientation_rad: np.ndarray  # RX 朝向
    runtime_versions: dict        # sionna/sionna_rt/torch/mitsuba/drjit 版本
```

## `path_adapter.py` — 路径数据转换

### `paths_to_table()`

将 Sionna `Paths` 对象转为 TX-first domain `PathTable`：

```python
def paths_to_table(
    paths: sionna.rt.Paths,
    topology: Topology,
    antenna: AntennaSpec,
) -> PathTable
```

**关键转换：**
- Sionna Paths 是 rx-first：`[rx, rx_ant, tx, tx_ant, path]`
- Adapter 输出 TX-first：`[tx, rx, rx_ant, tx_ant, path]`
- 路径类型分类：`path_type` 字段根据 `interaction_type` 和 `path_depth` 标记为 `"LoS"` 或 `"NLoS"`
- `vertices_m`：路径交互点坐标，包含 TX/RX 端点 + 所有中间反射/折射点
- `interaction_type` / `object_id` / `primitive_id`：每个交互 depth 的类型和命中对象

### `path_table_to_samples()`

从全量 `PathTable` 抽取轻量 `PathSamples`（用于 HDF5 `/paths/samples`）：

```python
def path_table_to_samples(
    table: PathTable,
    max_samples: int = 100,
    max_paths_per_sample: int = 10,
) -> PathSamples
```

## Shape 转换约定

Sionna 原生 Paths 使用 **rx-first** 维度：

```
Paths.a: [rx, rx_ant, tx, tx_ant, path]
```

Adapter 统一转为 **TX-first**（与项目 HDF5 契约一致）：

```
PathTable.a: [tx, rx, rx_ant, tx_ant, path]
```

转换在 adapter 内完成，业务层和 HDF5 writer 永远只看到 TX-first 数据。

## 版本信息

每次运行记录 Sionna/Mitsuba/Dr.Jit/Torch 版本，写入：
- `/runtime/sionna_version`
- `/runtime/sionna_rt_version`
- `/runtime/mitsuba_version`
- `/runtime/drjit_version`
- `/runtime/torch_version`
