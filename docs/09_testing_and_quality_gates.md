# 09. 测试与质量门

本文定义测试体系。里程碑验收见 [08_roadmap_milestones_acceptance.md](08_roadmap_milestones_acceptance.md)。

## 1. 测试分类

```text
unit tests
schema tests
adapter tests
integration tests
statistical tests
visualization smoke tests
regression tests
```

## 2. Unit Tests

必须覆盖：

- config schema validation。
- frequency grid generation。
- index selection。
- shape conversion。
- HDF5 path creation。
- NMSE/error metrics。
- impairment random sampling。

命令：

```bash
uv run pytest tests/unit
```

## 3. Schema Tests

必须验证：

- `/meta/schema_version` 存在。
- 必填 group 存在。
- 数据维度符合 [03_data_contract_hdf5.md](03_data_contract_hdf5.md)。
- 单位字段存在。
- `H_true` 和 `H_obs` 命名不混淆。
- HDF5 readback 后 shape 和 dtype 不变。
- 新 writer 不写 `/channel/cfr` 作为 truth 主路径。
- `/channel/truth/cfr` 与 `/observation/cfr_est` 同时存在时，二者维度关系满足 `cfr_est.shape[1:] == truth_cfr.shape`。
- `/paths/samples/vertices_m`、`interaction_type`、`object_id`、`primitive_id`、`doppler_hz`、`tau_s` 同时存在。

## 4. Adapter Tests

Sionna RT adapter 必须测试：

- import Sionna RT。
- load scene。
- register TX/RX。
- run PathSolver。
- extract `vertices`。
- extract `objects`。
- extract `interactions`。
- extract `primitives`。
- extract `doppler`。
- convert rx-first to tx-first。

如果测试场景没有 NLoS，应额外构造一个能产生反射的极小场景。

## 5. Integration Tests

必须覆盖：

- RT-only small scene。
- RT + HDF5 write/read。
- RT + path samples。
- RT + PHY AWGN-only。
- Full MVP pipeline。

测试数据位置：

```text
data/scenes/test/
```

## 6. Statistical Tests

必须覆盖：

- SNR 提高时 NMSE 下降。
- CFO 增大时相位漂移增大。
- clipping threshold 降低时 clipping rate 增大。
- 静态场景无速度时 Doppler 接近 0。
- 有速度时至少部分路径 Doppler 非 0。

统计测试允许设置容差，不要求每次 bitwise deterministic。

建议默认阈值：

```text
high_snr_nmse_db_max = -20.0
snr_monotonic_trials = 3
static_doppler_abs_hz_max = 1e-3
timestamp_monotonic = true
```

如果实际阈值因场景或算法改变，必须写入测试配置和阶段验收记录。

## 7. Visualization Smoke Tests

至少验证：

- 拓扑图可生成。
- path samples 图可生成。
- CFR magnitude 图可生成。
- NMSE/SNR 诊断图可生成。

Smoke test 只要求文件存在且非空，不要求图像内容严格一致。

## 8. 质量门

每次提交前至少运行：

```bash
uv run ruff check .
uv run pytest
```

涉及 Sionna adapter 的提交必须运行：

```bash
uv run pytest tests/adapter tests/integration
```

涉及 HDF5 schema 的提交必须运行：

```bash
uv run pytest tests/schema
```

涉及统计行为或 impairment 的提交必须运行：

```bash
uv run pytest tests/statistical
```

阶段验收前必须运行：

```bash
uv run ruff check .
uv run pytest
```

## 9. 禁止事项

- 不允许跳过失败测试后提交阶段完成。
- 不允许 schema 改了但文档不改。
- 不允许大输出文件混入 git。
- 不允许业务层绕过 adapter 直接读取 Sionna Paths。
- 不允许测试只检查“文件存在”而不检查关键 dataset、shape、dtype。
- 不允许 statistical test 使用未固定 seed 的随机输入，除非测试显式声明为随机稳健性测试。
