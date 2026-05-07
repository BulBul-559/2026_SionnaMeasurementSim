# 13 与 14 Review 复核

复核日期：2026-05-07

复核范围：

- [13_tdd_reciprocity_nr_pusch_phy_plan.md](13_tdd_reciprocity_nr_pusch_phy_plan.md)
- [14_rt_hardening_before_nr_pusch.md](14_rt_hardening_before_nr_pusch.md)
- 当前代码、HDF5 契约、配置文档和测试中与 13/14 相关的实现。

## 结论

不是“已无问题”。上一版 review 中有一部分已经修复，尤其是 14 的 RT hardening 状态、13 的主链路接入、NR PUSCH 集成测试和统计测试文件。但 13 的核心 PHY 真实性仍未达到文档要求，主要问题集中在：

- NR PUSCH 仍只使用第一个 TX/RX 和第一个 antenna pair，未真正保留 4x4 MIMO 语义。
- `/observation/cfr_est` 不是 receiver 估计出的 observed CSI，而是由 RT CIR 转换得到的 clean/derived CFR。
- `/evaluation/nmse_db` 当前在 NR PUSCH backend 中被固定写为 `-30 dB`，没有按 `H_obs` vs clean `H_true` 实际计算。
- NR PUSCH HDF5 waveform/receiver 契约仍不完整，validator 也没有按 `standard="nr_pusch"` 强制校验完整字段。
- RX orientation 的非 fixed 模式在 domain 层被允许，但 RT 层没有实现，存在配置语义与实际行为不一致。

因此，14 基本可以认为进入收口阶段；13 仍不能标记完成。

## 已修复项

### 文档状态

- 14 已经在文档开头明确写出：必须先完成 14 再进入 13。
- 14 末尾已从过期 review 改为 `Acceptance Status`。
- 13 末尾过期 review 已清理。

### 14 RT hardening

- `tests/adapter/test_rt_shape_contracts.py` 存在。
- `tests/adapter/test_rt_cir_adapter.py` 存在。
- `tests/schema/test_rt_cir_schema.py` 存在。
- `tests/integration/test_rt_mimo_4x4_pipeline.py` 存在。
- `to_project_cir(...)` 已进入 `sionna_measurement_sim/adapters/sionna_rt/shape_contracts.py`。
- `AntennaSpec` 已拆分 `tx_orientation_*` 和 `rx_orientation_*`。
- HDF5 writer 已分别写 `tx_orientation_mode` 与 `rx_orientation_mode`。
- CIR schema validator 已覆盖 all-or-none 数据集规则。

### 13 NR PUSCH 雏形

- CLI 已有 `run-full --phy-standard nr_pusch`。
- `run_rt_truth_pipeline(...)` 已能在 `phy_standard == "nr_pusch"` 时进入 NR PUSCH 分支。
- `nr_pusch_observation.py` 已构建 `PUSCHConfig`、`PUSCHTransmitter`，并调用 `PUSCHReceiver`。
- 主实现中已经不再通过大范围 try/except 把 PUSCHReceiver 失败静默变成 `ber=0.0`。
- `tests/integration/test_nr_pusch_observation.py` 已改为自生成输出，不再依赖已有 `outputs/e2e_nr_pusch_rx/results.h5`。
- `tests/statistical/test_nr_pusch_link_metrics.py` 已存在。
- HDF5 writer 已写入 `receiver/receiver_type`、`receiver/mimo_detector`、`receiver/input_domain`。
- `docs/03_data_contract_hdf5.md` 已把 `nmse_db` 文字语义更新为主指标 `NMSE(H_obs, clean H_true)`。

## 仍未修复的问题

### 1. NR PUSCH 仍未真正使用 MIMO 信道

当前 `nr_pusch_observation.py` 中的 channel application 仍显式选择：

```text
h_tensor[0, 0, 0, 0, :]
```

也就是第一个 TX、RX、RX antenna、TX antenna 的单个 SISO-like channel。代码注释也承认这会丢弃 MIMO 信息。

这意味着：

- 14 虽然能生成 4x4 RT truth。
- 13 的 PUSCH backend 仍没有消费完整 4x4 MIMO channel。
- 当前统计测试使用 1x1 antenna 配置，无法发现这个问题。

13 要求的 MIMO stream management、antenna ports、layers、detector 语义仍未通过验收。

### 2. `/observation/cfr_est` 不是接收机估计结果

NR PUSCH backend 中：

```text
cfr_est = cfr_np[0:1]
```

这里的 `cfr_np` 来自 RT CIR -> CFR 转换，而不是 PUSCH receiver / channel estimator 输出的 observed CSI。

因此当前 `/observation/cfr_est` 更像“由 CIR 计算出的 truth/derived CFR”，不是 `H_obs` 或 receiver estimated channel。这个字段名和 13 的目标语义不一致。

### 3. `/evaluation/nmse_db` 仍是固定值

NR PUSCH backend 当前写：

```text
nmse_db = -30 dB
nmse_db_total = -30 dB
```

没有实际计算：

```text
NMSE(H_obs, clean H_true)
```

所以虽然 03 文档语义已更新，代码实现还没有对齐。当前测试只检查 finite，不能发现固定值问题。

### 4. BER/BLER 仍不等价于完整链路级 BLER

当前 BER 是 `rx_bits` 和 `tx_bits` 的 bit-level 比较，BLER 是：

```text
num_bit_errors > 0 ? 1.0 : 0.0
```

但 `num_block_errors` 和 `num_blocks` 仍保持默认值，`tb_decoder=None`，没有形成完整 transport block / CRC / BLER 语义。13 文档中要求的 LDPC/CRC/rate matching/modulation 级闭环仍未完成。

### 5. perfect CSI 与 estimated CSI 验收缺失

13 要求：

```text
perfect CSI 的 BER/BLER 不差于 estimated CSI
```

当前没有看到真正的 perfect CSI path，也没有对应测试。`phy_config.perfect_csi` 字段存在，但 NR PUSCH backend 未形成可验证分支。

### 6. 统计测试仍可能隐藏 receiver/API 问题

`tests/statistical/test_nr_pusch_link_metrics.py` 中如果 pipeline 抛异常，会：

```text
pytest.skip("NR PUSCH receiver not available on this machine")
```

这对可选环境检查可以接受，但不能作为 13 最终验收。13 验收阶段应该在支持环境中 fail fast，而不是跳过。

### 7. NR PUSCH HDF5 契约仍不完整

`docs/03_data_contract_hdf5.md` 和 writer 已补部分字段，但仍缺少或不一致：

- 13 要求 `subcarrier_spacing_hz`，writer 写 `subcarrier_spacing_khz`。
- 13 要求 `slot_number`，当前未写。
- 13 要求 `cyclic_prefix`，当前未写。
- 13 要求 `target_coderate`，当前未写。
- 13 要求 `modulation`，当前未写。
- 03 文档写 `mcs_table string`，writer 按 `int32` 写。
- schema validator 没有在 `waveform/standard == "nr_pusch"` 时强制校验这些 NR PUSCH 扩展字段。

### 8. `/receiver` 契约仍偏弱

writer 已写：

```text
receiver_type
mimo_detector
input_domain
```

但 schema validator 的 required observation datasets 仍只强制 `receiver/estimator_type`。对 `standard="nr_pusch"` 的最小要求应至少强制：

```text
receiver/receiver_type = "pusch_receiver"
receiver/input_domain = "freq"
receiver/mimo_detector
```

另外 13 文档使用 `channel_estimator`，当前 domain/writer 使用 `estimator_type`。需要统一命名或明确 alias。

### 9. 06 配置文档仍未完整描述 NR PUSCH

`docs/06_config_and_experiment_schema.md` 的 `phy` 示例仍以 `custom_ofdm` 为主，没有完整列出：

```text
standard: "nr_pusch"
num_prb
subcarrier_spacing_khz
num_layers
num_antenna_ports
mcs_index
mcs_table
pusch_dmrs_*
perfect_csi
ebno_db
```

这会让执行 agent 或用户不知道 YAML 中应如何配置 NR PUSCH。

### 10. RX orientation 非 fixed 模式语义不一致

`AntennaSpec` 允许 RX 使用：

```text
look_at_first_peer
look_at_centroid
```

但 RT scene 构建时 RX 始终按 `rx_orientation_rad` 固定写入，没有按 `rx_orientation_mode` 做 look-at。14 文档的 Acceptance Status 写“RX only fixed”，但代码没有 fail fast 拒绝 RX 非 fixed。

应二选一：

- 实现 RX look-at 模式；
- 或在 config/domain 校验中禁止 `rx_orientation_mode != "fixed"`，并在文档中明确。

### 11. pattern/polarization 白名单仍未形成配置级 fail fast

14 要求 Sionna pattern/polarization 不支持时 fail fast。当前 polarization 有轻量映射，pattern 主要直接传给 Sionna。建议在配置/domain 层明确白名单并测试：

```text
pattern: iso, tr38901
polarization: V, H, cross
```

### 12. `merge_shapes=true` traceability warning 仍未实现

14 Acceptance Status 里已明确该项仍是 gap。需要在 manifest 或 diagnostics 中记录 `merge_shapes=true` 会降低 object/primitive 追溯能力。

## 新发现或需要强调的问题

### 1. `phy_config` / `carrier_config` 被 `RTTruthRunConfig` 兼任

`_run_nr_pusch_obs(...)` 中把同一个 `RTTruthRunConfig` 同时作为：

```text
phy_config
carrier_config
```

这能跑通，但架构上容易继续把 RT runner、PHY config、carrier config 混在一起。后续建议拆出明确的 NR PUSCH config/domain object。

### 2. 13 文档仍与当前实现不同步

13 仍写有一些“当前系统没有”的差距项，比如没有 PUSCH transmitter/receiver、没有 BER/BLER 等；这些现在已经部分实现。13 应更新为当前状态，否则执行 agent 会重复做已完成部分。

### 3. README 仍按编号列 13 再列 14

14 文档本身已写 execution order，但 README 目录仍按 13、14 顺序列出。建议在 README 对 13/14 增加一句：

```text
执行顺序：先 14，再 13。
```

## 本次核查命令

```bash
uv run pytest tests/adapter/test_rt_shape_contracts.py tests/adapter/test_rt_cir_adapter.py tests/schema/test_rt_cir_schema.py tests/integration/test_rt_mimo_4x4_pipeline.py tests/unit/test_reciprocity.py tests/unit/test_nr_pusch_config.py tests/integration/test_nr_pusch_observation.py tests/statistical/test_nr_pusch_link_metrics.py
uv run ruff check .
```

结果：

```text
42 passed, 1 warning
ruff: All checks passed
```

## 复核结论

可以标记为已修复的主要是：

- 14 的 RT hardening 基础实现和测试。
- 13 的 NR PUSCH 主链路入口。
- 自生成 integration test。
- 统计测试文件存在。
- PUSCHReceiver 不再被主实现静默 fallback。
- 部分 HDF5 receiver/waveform 字段补齐。

不能标记为已无问题。当前最优先应修：

1. NR PUSCH backend 不能只用第一个 antenna pair，必须保留 MIMO 语义或明确限制为 SISO 并让 13 验收降级。
2. `/observation/cfr_est` 必须来自 receiver/channel estimator 的 observed CSI，或重命名当前 derived CFR。
3. `/evaluation/nmse_db` 必须真实计算，不能固定写 `-30 dB`。
4. 补齐 NR PUSCH HDF5/schema/config 文档和 validator。
5. 对 RX non-fixed orientation、pattern/polarization、merge_shapes warning 做 fail-fast 或明确实现。
