# 10. 旧系统复用与迁移

本文定义旧 `SimpleSionna` 的复用策略。旧系统位于 `SionnaMeasurementSim/old/SimpleSionna/`，但新系统不得直接 import 旧代码。

## 1. 旧系统定位

旧系统是：

```text
Sionna 1.x RT truth CFR 生成器
```

它不是：

```text
Sionna 2.x measurement-oriented PHY observation simulator
```

## 2. 可直接借鉴

- Pipeline phase 设计。
- 输入 label 解析规则。
- `all_positions` 优先，`groups` 只补标签和冲突检查。
- debug subset。
- 输出目录组织。
- manifest 思路。
- HDF5 + stats + plots 的结果组织。
- TX-first CFR 维度。
- 抽样路径可视化。
- 链路覆盖统计。

## 3. 需要抽象后迁移

- `SimulationResult` 拆为：
  - `RTTruthResult`
  - `ObservationResult`
  - `MeasurementSimulationResult`
- `/channel/cfr` 迁移为：
  - `/channel/truth/cfr`
- `has_signal` 迁移为：
  - `has_geometric_signal`
  - `/observation/valid_mask`
- `path_power_db` 明确为 truth channel power，不等于 RSSI。
- 路径抽样逻辑迁移为 `PathSummaryExtractor`。

## 4. 必须重写

- Sionna 1.x `scene_assembler.py`。
- Sionna 1.x `simulator.py` 主体。
- `paths.cfr()` shape 假设。
- `Paths` 属性解析。
- TensorFlow GPU preflight。
- Sionna 1.x 材质补全假设。

## 5. 旧输出兼容

新系统 reader 可以提供旧 HDF5 fallback：

```text
old /channel/cfr -> new logical /channel/truth/cfr
old /channel/has_signal -> new logical /channel/truth/has_geometric_signal
```

但新 writer 不应继续写旧主路径。

## 6. 迁移步骤

1. 确认旧项目位于 `old/SimpleSionna/`。
2. 只人工参考旧代码，不直接 import。
3. 先迁移输入解析测试。
4. 再迁移输出目录和 manifest 思路。
5. 重写 HDF5 schema writer。
6. 重写 Sionna RT adapter。
7. 用同一小场景对比旧 `H_true` 和新 `H_true` 的数量级。

## 7. 对比指标

旧新 RT truth 对比时，不要求完全一致，但应检查：

- shape 一致。
- 频率轴一致。
- 有效链路数量数量级合理。
- path count 数量级合理。
- CFR power 分布数量级合理。

如果 Sionna 2.x RT 结果与 1.x 有差异，应记录在迁移报告中，不强行追求 bitwise 一致。
