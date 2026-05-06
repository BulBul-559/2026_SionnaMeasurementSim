# 14. NR PUSCH 前置任务：RT 链路硬化计划

本文给下一阶段执行 agent 使用。目标是在进入 [13_tdd_reciprocity_nr_pusch_phy_plan.md](13_tdd_reciprocity_nr_pusch_phy_plan.md) 的 TDD reciprocity + NR PUSCH PHY 之前，先把 Sionna RT adapter 和 RT truth pipeline 中会影响物理真实性、MIMO 可维护性、CIR 接入和实验复现的部分补齐。

本阶段不改 label 解析策略。label 系统后续需要和外部标注系统对齐，暂不在本阶段扩展。

## 1. 为什么要先做 RT 硬化

当前 RT 链路不是 mock，已经真实调用：

```text
sionna.rt.load_scene
sionna.rt.PathSolver
paths.cfr(...)
paths.valid/a/tau/doppler/interactions/objects/primitives/vertices
```

但当前实现仍偏 Phase 2/3 最小闭环：

- TX 朝向写 HDF5 是零，但实际 Sionna scene 里 TX 被 `look_at(receivers[0])`。
- antenna pattern 和 polarization 配置不完整。
- 只稳定产出 CFR，没有形成面向标准 PHY 的 CIR/tap bridge。
- 多天线统计和 samples 管理还不够清晰。
- `merge_shapes=True` 固定写死，不利于对象级追溯。
- config snapshot 没完整记录 RT/antenna/scene adapter 参数。
- 测试主要是 1x1 SISO，不能证明 MIMO shape 管理可靠。

这些问题如果不先修，后续接 NR PUSCH 会把维度、朝向、CIR、天线语义问题带进 PHY 层。

## 2. 决策

本阶段按以下决策执行：

- 天线朝向必须配置化，并有明确默认值。
- 天线 pattern 和 polarization 必须分开配置。`iso` 是 radiation pattern，不是 polarization。
- 默认 antenna pattern 使用 `iso`，表示全向/各向同性方向图。
- 默认 polarization 使用当前 Sionna 可稳定支持的单极化配置，推荐 `"V"`。
- CIR 必须补齐。CFR 继续作为 frequency-grid 上的 truth/GT，但 NR PUSCH backend 应优先通过 CIR/taps 接入 Sionna PHY channel。
- 多天线/MIMO 做架构重整，避免在 path adapter 中用零散 reshape 和隐式 antenna 约定。
- `merge_shapes` 必须成为配置项，默认 `false`。
- config snapshot 必须完整记录 RT、scene、antenna、orientation、CIR/CFR 生成参数。
- 测试必须新增 4x4 MIMO 场景，即 TX 4 antenna、RX 4 antenna。
- label parser 暂不扩展多 group 或外部标注 schema。

## 3. CFR 和 CIR 的职责

### CFR 是什么

当前 `/channel/truth/cfr` 是 ray-traced channel 在项目配置的 frequency grid 上的频域响应：

```text
H_true[f] = CFR over configured subcarriers
shape = [tx, rx, rx_ant, tx_ant, subcarrier]
```

它可以作为：

- truth/GT 标签。
- 简化 `custom_ofdm` observation 的输入。
- ML 任务中的监督目标。
- HDF5 readback 和快速可视化对象。
- 与 `/observation/cfr_est` 计算 NMSE 的 clean truth。

因此 CFR 仍然必须保留，并继续写：

```text
/channel/truth/cfr
```

### CIR 为什么必须补

标准 PHY，尤其 Sionna 官方 NR PUSCH link-level 示例，更自然地通过 CIR/taps 接入 channel model：

```text
paths.cir(...)
or paths.a / paths.tau
  -> CIRDataset / OFDMChannel / ApplyOFDMChannel
  -> PUSCHTransmitter / PUSCHReceiver
```

只保存 CFR 会带来问题：

- 难以准确驱动 Sionna `OFDMChannel` 的时域/路径级模型。
- Doppler、多 snapshot、delay spread、OFDM symbol 级变化不够自然。
- 后续做标准 DMRS channel estimation 时，CIR/tap 语义更接近官方链路。

本阶段必须新增 CIR domain model 和 HDF5 写入。

建议字段：

```text
/channel/truth/cir_coefficients      complex64 [snapshot, tx, rx, rx_ant, tx_ant, path]
/channel/truth/cir_delays_s          float32   [snapshot, tx, rx, rx_ant, tx_ant, path]
/channel/truth/cir_valid             bool      [snapshot, tx, rx, rx_ant, tx_ant, path]
```

如果 Sionna 2.x `paths.cir(...)` 输出 shape 与上面不同，必须在 adapter 层转换，不得把 Sionna 原生 shape 泄漏到业务层或 writer。

## 4. 天线配置与朝向

### 配置 schema

新增或扩展 antenna 配置：

```yaml
antenna:
  tx_array:
    type: "planar"
    num_rows: 2
    num_cols: 2
    vertical_spacing_lambda: 0.5
    horizontal_spacing_lambda: 0.5
    pattern: "iso"
    polarization: "V"
    orientation_mode: "fixed"
    orientation_rad: [0.0, 0.0, 0.0]
  rx_array:
    type: "planar"
    num_rows: 2
    num_cols: 2
    vertical_spacing_lambda: 0.5
    horizontal_spacing_lambda: 0.5
    pattern: "iso"
    polarization: "V"
    orientation_mode: "fixed"
    orientation_rad: [0.0, 0.0, 0.0]
```

允许的 `orientation_mode`：

```text
fixed
look_at_first_peer
look_at_centroid
from_label_future
```

本阶段默认：

```text
orientation_mode = "fixed"
orientation_rad = [0.0, 0.0, 0.0]
```

说明：

- 当前实现中 `tx.look_at(receivers[0])` 和 HDF5 中全零 orientation 不一致，必须移除或配置化。
- 如果使用 `look_at_first_peer`，必须同时把实际 orientation 写入 `/devices/tx_orientation_rad` 和 `/devices/rx_orientation_rad`。
- 如果使用 `fixed`，Sionna scene 中设备 orientation 与 HDF5 必须一致。
- `from_label_future` 只是预留，当前不实现。

### Pattern 与 polarization

必须区分：

```text
pattern: radiation pattern, e.g. "iso", "tr38901"
polarization: polarization model, e.g. "V", "H", "cross"
```

默认：

```text
pattern = "iso"
polarization = "V"
```

执行 agent 必须检查当前 Sionna 2.x `PlanarArray` 支持的 pattern/polarization 值。若某些值无法稳定运行，配置校验必须 fail fast。

## 5. Scene 和 merge_shapes

当前 `load_scene(..., merge_shapes=True)` 是硬编码。下一步改为配置项：

```yaml
rt:
  merge_shapes: false
```

默认：

```text
merge_shapes = false
```

理由：

- `false` 更利于 object/primitive/material 追溯。
- debug 和标注场景优先保证可解释性。
- 如果性能需要，可以在 batch/performance profile 中设为 `true`。

要求：

- `merge_shapes` 必须写入 config snapshot。
- manifest 中记录 `merge_shapes`。
- 如果 `merge_shapes=true` 导致 object_id/object_name 追溯能力下降，manifest 或 diagnostics 中必须记录警告。

## 6. MIMO 架构重整

当前内部 shape 契约保留：

```text
CFR:
  [tx, rx, rx_ant, tx_ant, subcarrier]

Path scalar:
  [tx, rx, rx_ant, tx_ant, path]

CIR:
  [snapshot, tx, rx, rx_ant, tx_ant, path]
```

但代码上应引入集中 shape 管理，避免在 adapter 各处手写 transpose/reshape。

建议新增：

```text
sionna_measurement_sim/adapters/sionna_rt/shape_contracts.py
```

职责：

- 定义 Sionna raw shape 到 project TX-first shape 的转换函数。
- 每个转换函数写清输入/输出 axis 语义。
- 对 CFR、CIR、path scalar、interaction、vertices 分别提供函数。
- 提供 shape assertion helper。

建议函数：

```python
to_project_cfr(raw_cfr, num_time_steps) -> tuple[cfr_5d, cfr_snapshots]
to_project_cir(raw_a, raw_tau, num_time_steps) -> CIRTruth
to_project_path_scalar(value, name) -> np.ndarray
to_project_interaction(value, name) -> np.ndarray
to_project_vertices(value) -> np.ndarray
```

Path adapter 中不再散落复杂 transpose，只调用这些函数。

### Path summary 修正

多天线下 `los_exists` / `nlos_exists` 必须对所有 antenna pair 正确聚合。

当前风险模式：

```text
link_valid aggregates all antenna pairs
path_type sometimes checks only antenna pair 0
```

正确方式：

```text
los_exists[tx, rx] =
  any(valid[tx, rx, rx_ant, tx_ant, path] & path_type == "los" over rx_ant, tx_ant, path)

nlos_exists[tx, rx] =
  any(valid[...] & path_type != "los" over rx_ant, tx_ant, path)
```

## 7. HDF5 与 schema 更新

必须更新 [03_data_contract_hdf5.md](03_data_contract_hdf5.md)，至少新增：

```text
/channel/truth/cir_coefficients
/channel/truth/cir_delays_s
/channel/truth/cir_valid
```

必须明确：

- `/channel/truth/cfr` 是 frequency-domain H_true/GT。
- `/channel/truth/cir_*` 是 path/tap-domain truth，用于标准 PHY channel 接入。
- 两者应来自同一 Sionna RT `Paths` 结果。

必须扩展 `/antenna` 或 `/devices`：

```text
/antenna/tx_pattern
/antenna/rx_pattern
/antenna/tx_polarization
/antenna/rx_polarization
/antenna/tx_orientation_mode
/antenna/rx_orientation_mode
/devices/tx_orientation_rad
/devices/rx_orientation_rad
```

如果已有字段存在，需补语义和 validator。

## 8. Config snapshot 补齐

`/meta/config_snapshot` 和 manifest 的 `config_snapshot` 必须包含：

```text
center_frequency_hz
bandwidth_hz
num_subcarriers
max_depth
los
specular_reflection
diffuse_reflection
refraction
diffraction
synthetic_array
normalize_cfr
normalize_delays
merge_shapes
tx/rx antenna rows/cols
tx/rx antenna spacing
tx/rx pattern
tx/rx polarization
tx/rx orientation_mode
tx/rx orientation_rad
num_time_steps
sampling_frequency_hz
tx/rx velocity
```

禁止只在 Python config object 中存在，而不写入 HDF5/manifest snapshot。

## 9. 测试要求

本阶段测试必须使用 4x4 MIMO 系统：

```text
TX antenna count = 4
RX antenna count = 4
recommended array = 2 x 2 planar at both ends
```

新增测试：

```text
tests/adapter/test_rt_shape_contracts.py
tests/adapter/test_rt_cir_adapter.py
tests/integration/test_rt_mimo_4x4_pipeline.py
tests/schema/test_rt_cir_schema.py
```

最低验收：

- 4x4 MIMO CFR shape 为 `[tx, rx, 4, 4, subcarrier]`。
- 4x4 MIMO CIR shape 为 `[snapshot, tx, rx, 4, 4, path]`。
- `los_exists` / `nlos_exists` 对所有 antenna pair 聚合正确。
- `merge_shapes=false` 为默认值，并写入 config snapshot。
- `pattern="iso"` 默认生效，并写入 HDF5。
- `polarization="V"` 默认生效，并写入 HDF5。
- `orientation_mode="fixed"` 默认生效，Sionna scene 和 HDF5 orientation 一致。
- `look_at_first_peer` 模式下，HDF5 记录实际 look_at 后 orientation。
- `paths.cfr(...)` 和 CIR/tap 数据来自同一次 Sionna RT path solve。
- `uv run ruff check .` 通过。
- `uv run pytest` 通过。

## 10. 不做事项

本阶段暂不做：

- label parser 多 group 或外部标注 schema 变更。
- TDD reciprocity transform。
- NR PUSCH transmitter/receiver。
- FDD UL/DL 双频计算。
- PDSCH downlink。
- 多小区干扰。

这些内容在 RT 硬化通过后，再进入 [13_tdd_reciprocity_nr_pusch_phy_plan.md](13_tdd_reciprocity_nr_pusch_phy_plan.md)。

## 11. 完成标准

完成后，系统应能回答：

```text
在 4x4 MIMO 配置下，Sionna RT adapter 是否能稳定生成
CFR truth、CIR truth、路径级数据、天线/朝向/scene 配置快照，
并且所有 shape、方向、pattern、polarization、merge_shapes 语义可审计？
```

只有答案为肯定，才继续实现 TDD reciprocity 与 NR PUSCH PHY。
