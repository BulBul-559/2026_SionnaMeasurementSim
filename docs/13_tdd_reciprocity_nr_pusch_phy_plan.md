# 13. TDD 互易性与 NR PUSCH PHY 补齐计划

本文给下一阶段执行 agent 使用。目标是在当前 `SionnaMeasurementSim` 基础上，把 PHY 观测链从“自定义 OFDM + 全子载波 pilot + AWGN/LS 最小闭环”推进到“基于 Sionna 官方 5G NR PUSCH 示例的 TDD uplink 建模闭环”。

本阶段只支持 **TDD 同频互易性建模**。FDD 上下行不同频率建模列为明确 TODO/backlog，见 [12. FDD TODO](#12-fdd-todo)。

进入本文任务前，必须先完成 [14_rt_hardening_before_nr_pusch.md](14_rt_hardening_before_nr_pusch.md) 中的 RT 链路硬化任务，尤其是 CIR truth、4x4 MIMO shape、天线朝向、pattern/polarization、`merge_shapes` 配置和完整 config snapshot。

## 1. 关键结论

官方 Sionna RT link-level 示例做的是 5G NR PUSCH uplink。示例中为了 ray tracing 和 UE 采样方便，先把基站配置为 transmitter、UE 配置为 receiver 来生成 BS -> UE 的 RT CIR，然后利用 TDD 信道互易性转换为 UE -> BS 的 uplink 信道。官方也说明：对 ray tracer 本身，uplink/downlink 不改变 simulated paths；方向差异主要体现在 PHY 链路和张量维度语义。

本项目下一步采用同样策略：

```text
RT trace direction:
  BS as TX, UE as RX

PHY link direction:
  uplink, UE as transmitter, BS as receiver

Duplex mode:
  TDD only

Transform:
  use reciprocity transform between RT truth tensors/CIR tensors and PHY PUSCH tensors
```

必须显式记录：

```text
duplex_mode = "tdd"
phy_link_direction = "uplink"
rt_trace_direction = "bs_to_ue"
reciprocity_applied = true
```

## 2. 官方参考入口

执行前必须阅读这些官方材料，并以当前安装的 Sionna 2.x API 为准：

- Sionna RT + PHY link-level 示例：  
  https://nvlabs.github.io/sionna/phy/tutorials/notebooks/Link_Level_Simulations_with_RT.html
- Sionna 5G NR PUSCH Tutorial：  
  https://nvlabs.github.io/sionna/phy/tutorials/notebooks/5G_NR_PUSCH.html
- Sionna PHY documentation：  
  https://nvlabs.github.io/sionna/phy/index.html
- Sionna 5G NR API，重点查 `PUSCHConfig`、`PUSCHTransmitter`、`PUSCHReceiver`、`PUSCHDMRSConfig`、`PUSCHLSChannelEstimator`：  
  https://nvlabs.github.io/sionna/phy/api/nr.html
- Sionna OFDM API，重点查 `ResourceGrid`、`ResourceGridMapper`、`LSChannelEstimator`、interpolator：  
  https://nvlabs.github.io/sionna/phy/api/ofdm.html
- Sionna channel API，重点查 `CIRDataset`、`OFDMChannel`、`ApplyOFDMChannel`：  
  https://nvlabs.github.io/sionna/phy/api/channel.wireless.html
- Sionna installation / PyTorch requirement：  
  https://nvlabs.github.io/sionna/installation.html

不要从旧 TensorFlow Sionna 代码迁移 PHY 主链路。当前项目约束是 Sionna 2.x + PyTorch。

## 3. 当前系统差距

当前 PHY 观测链有这些不足：

- 没有明确 `uplink/downlink` 链路方向。
- 没有 `duplex_mode=tdd/fdd` 配置。
- 没有 TDD reciprocity transform。
- `/waveform` 当前是 `custom_ofdm`，不是 NR PUSCH。
- pilot 当前是所有子载波全 1，不是 5G NR DMRS。
- 没有 PUSCH resource grid、slot、PRB、DMRS config、MCS、transport block。
- 没有 PUSCH transmitter/receiver。
- 没有 LDPC/CRC/rate matching/modulation 的链路级闭环。
- 没有 BER/BLER 结果。
- 没有 MIMO stream management、LMMSE/KBest detector 配置。
- 当前 `/evaluation/nmse_db` 语义必须整理：主字段应表示 `NMSE(H_obs, H_true)`，若保留 AWGN-only 指标，应另命名。

本阶段要补齐的核心不是“更多 HDF5 字段”，而是让 PHY 观测由 Sionna 官方 PUSCH 组件驱动。

## 4. 功能范围

### 必须实现

新增一个 NR PUSCH observation backend：

```text
sionna_measurement_sim/phy/nr_pusch_observation.py
```

它应完成：

1. 接收 RT truth 或 RT CIR。
2. 在 TDD uplink 模式下执行 reciprocity transform。
3. 构建 Sionna `PUSCHConfig`。
4. 构建 `PUSCHTransmitter` 和 `PUSCHReceiver`。
5. 使用 Sionna `OFDMChannel` 或与官方示例一致的 `CIRDataset` 接入 ray-traced CIR。
6. 生成真实 PUSCH resource grid 和 DMRS。
7. 支持 imperfect CSI 和 perfect CSI 两种模式。
8. 至少支持 LMMSE detector；如果 API 和性能允许，再支持 KBest。
9. 输出 observed CFR/CSI 到 `/observation/cfr_est`。
10. 输出 BER/BLER 或至少 bit/block error 统计到 `/evaluation`。
11. 保留当前 HDF5 truth/path data contract，不把 Sionna 原生对象传入 writer。

### 可以延后

- PDSCH downlink。
- HARQ。
- 多小区干扰。
- 真实 scheduler。
- 完整 RF 前端非线性模型。
- 实测标定 profile 自动拟合。

## 5. TDD 互易性设计

新增配置：

```yaml
link:
  duplex_mode: "tdd"
  phy_link_direction: "uplink"
  rt_trace_direction: "bs_to_ue"
  reciprocity_mode: "transpose_rt_channel"
```

约束：

- `duplex_mode` 第一版只能是 `"tdd"`。
- 如果用户配置 `"fdd"`，必须 fail fast，提示尚未支持 FDD。
- `phy_link_direction` 第一版只实现 `"uplink"`。
- `rt_trace_direction` 第一版使用 `"bs_to_ue"`。
- reciprocity transform 必须集中在 adapter/bridge 层，不能散落在 PHY 业务代码里。

推荐新增：

```text
sionna_measurement_sim/phy/reciprocity.py
```

职责：

- 把当前内部 TX-first truth/CIR 从 BS->UE 转为 UE->BS。
- 记录转换前后 shape。
- 明确 antenna 维度转换规则。
- 单元测试覆盖单天线和多天线。

注意：

- RT truth 主路径 `/channel/truth/cfr` 仍保持项目既有契约 `[tx, rx, rx_ant, tx_ant, subcarrier]`，其中 `tx/rx` 是 RT trace 方向。
- PUSCH backend 内部可使用 uplink 视角张量，但写回 HDF5 时必须清楚记录其语义。
- 如果新增 `/channel/uplink_truth/cfr` 或类似字段，必须先更新 [03_data_contract_hdf5.md](03_data_contract_hdf5.md) 和 schema tests。

## 6. 数据契约变更

需要更新 [03_data_contract_hdf5.md](03_data_contract_hdf5.md)，新增或明确以下字段。

建议新增 `/link` group：

```text
/link/duplex_mode                  string, "tdd"
/link/phy_link_direction           string, "uplink"
/link/rt_trace_direction           string, "bs_to_ue"
/link/reciprocity_mode             string, "transpose_rt_channel"
/link/reciprocity_applied          bool
```

扩展 `/waveform`：

```text
standard                           string, "nr_pusch"
subcarrier_spacing_hz              float64
num_prb                            int32
num_ofdm_symbols                   int32
slot_number                        int32
cyclic_prefix                      string
dmrs_config_type                   int32
dmrs_length                        int32
dmrs_additional_position           int32
num_cdm_groups_without_data        int32
num_layers                         int32
num_antenna_ports                  int32
mcs_index                          int32
mcs_table                          int32
target_coderate                    float32
modulation                         string
```

扩展 `/receiver`：

```text
receiver_type                      string, "pusch_receiver"
channel_estimator                  string, "perfect" or "pusch_ls"
mimo_detector                      string, "lmmse" or "kbest"
input_domain                       string, "freq"
```

扩展 `/evaluation`：

```text
ber                               float32
bler                              float32
num_bit_errors                    int64
num_bits                          int64
num_block_errors                  int64
num_blocks                        int64
nmse_db                           float32 [snapshot, tx, rx]  # must be H_obs vs clean H_true
nmse_awgn_db                      float32 [snapshot, tx, rx]  # optional, AWGN-only diagnostic
```

如果继续保存 `nmse_db_total`，必须重命名或文档化。推荐：

- `/evaluation/nmse_db`：主指标，始终是 `NMSE(H_obs, H_true)`。
- `/evaluation/nmse_awgn_db`：可选诊断，表示相对 impaired channel 的 AWGN-only 误差。

## 7. 配置变更

更新 [06_config_and_experiment_schema.md](06_config_and_experiment_schema.md)，新增：

```yaml
link:
  duplex_mode: "tdd"
  phy_link_direction: "uplink"
  rt_trace_direction: "bs_to_ue"
  reciprocity_mode: "transpose_rt_channel"

phy:
  standard: "nr_pusch"
  perfect_csi: false
  ebno_db: 10.0
  num_prb: 16
  num_ofdm_symbols: 14
  subcarrier_spacing_hz: 30000.0
  num_layers: 1
  num_antenna_ports: 4
  mcs_index: 14
  mcs_table: 1
  dmrs:
    config_type: 1
    length: 1
    additional_position: 1
    num_cdm_groups_without_data: 2
  receiver:
    channel_estimator: "pusch_ls"
    mimo_detector: "lmmse"
```

规则：

- `phy.standard="custom_ofdm"` 保留为 legacy/minimal debug backend。
- `phy.standard="nr_pusch"` 走新 backend。
- `link.duplex_mode!="tdd"` 必须 fail fast。
- `link.phy_link_direction!="uplink"` 必须 fail fast，直到 PDSCH/downlink 被实现。
- `phy.num_antenna_ports` 必须与 UE/PUSCH transmitter 配置一致。
- `antenna` 配置必须与 PUSCH backend 预期的 BS/UE antenna 数一致。

## 8. 推荐实现顺序

1. 新增 link/domain config model，不接入运行。
2. 新增 reciprocity transform 单元测试，用小型 synthetic tensors 验证维度转换。
3. 新增 NR PUSCH backend skeleton，只构建 `PUSCHConfig` 并写 `/waveform` 配置。
4. 接入 `PUSCHTransmitter`、`PUSCHReceiver`，先用官方 stochastic/simple channel smoke test。
5. 按官方 link-level RT 示例方式，把 RT CIR 接入 `CIRDataset`/`OFDMChannel`。
6. 把 BER/BLER 和 observed CSI 写入 domain model。
7. 扩展 HDF5 writer/validator。
8. 增加 CLI：`run-nr-pusch` 或 `run-full --phy-standard nr_pusch`。
9. 加集成测试和统计测试。
10. 更新 `docs/phase_progress.md` 和 acceptance report。

## 9. 测试要求

必须新增测试：

```text
tests/unit/test_reciprocity.py
tests/unit/test_nr_pusch_config.py
tests/integration/test_nr_pusch_observation.py
tests/statistical/test_nr_pusch_link_metrics.py
```

最低验收：

- TDD reciprocity transform 后 shape 与 Sionna PUSCH channel model 期望一致。
- `link.duplex_mode=fdd` 明确报错。
- `phy.standard=nr_pusch` 写出 `/waveform/standard = "nr_pusch"`。
- `/receiver/receiver_type = "pusch_receiver"`。
- `/evaluation/ber`、`/evaluation/bler` 存在且有限。
- 高 Eb/N0 的 BER/BLER 不高于低 Eb/N0。
- perfect CSI 的 BER/BLER 不差于 estimated CSI。
- `/evaluation/nmse_db` 表示 `H_obs` vs clean `H_true`。
- `uv run ruff check .` 通过。
- `uv run pytest` 通过。

## 10. 不要做的事

- 不要把 Sionna 原生 `Paths` 或 PHY 对象传给 HDF5 writer。
- 不要在业务层直接索引 Sionna `Paths`。
- 不要引入 TensorFlow 到核心链路。
- 不要把 FDD 伪装成 TDD 互易。
- 不要把 all-one/full-subcarrier pilot 继续称为 3GPP DMRS。
- 不要把 `/evaluation/nmse_db` 改成非 `H_obs` vs `H_true` 的语义。
- 不要一次性实现 PUSCH/PDSCH/FDD/HARQ/多小区，先把 TDD uplink PUSCH 闭环做稳。

## 11. 完成后的判断标准

完成后，项目应能回答：

```text
在 TDD 同频互易假设下，给定 BS->UE ray-traced scene truth，
系统能否生成符合 5G NR PUSCH resource grid 和 DMRS 的 uplink observation，
并输出 observed CSI、BER/BLER、NMSE、诊断字段和可复现 HDF5？
```

如果答案是肯定的，当前系统才从“自定义 OFDM 测量模拟器”推进到了“基于官方 Sionna 5G NR PUSCH 的测量模拟器雏形”。

## 12. FDD TODO

FDD 不在本阶段实现，但必须作为后续正式任务保留。不要用 TDD 互易性结果冒充 FDD。

后续 FDD 支持至少需要：

- 新增配置：

```yaml
link:
  duplex_mode: "fdd"
  phy_link_direction: "uplink"   # or "downlink"
  reciprocity_mode: "none"

carrier:
  ul_center_frequency_hz: ...
  dl_center_frequency_hz: ...
  ul_bandwidth_hz: ...
  dl_bandwidth_hz: ...
```

- 分别按 UL/DL 频率计算或采样 channel，不允许直接复用同一个 CFR。
- 明确记录 UL/DL carrier、bandwidth、subcarrier grid 和 frequency-dependent material/antenna response。
- 对同一几何路径，可复用路径拓扑作为优化，但路径相位、频响、损耗、材料响应必须按对应频率重新计算或重新评估。
- HDF5 schema 需要区分：

```text
/channel/uplink/truth/cfr
/channel/downlink/truth/cfr
/observation/uplink/cfr_est
/observation/downlink/cfr_est
```

或采用等价但明确的方向字段设计。禁止把 FDD truth 写入现有 `/channel/truth/cfr` 后不说明方向。

- 测试必须覆盖：

```text
duplex_mode=fdd refuses reciprocity_mode=transpose_rt_channel
ul_center_frequency_hz != dl_center_frequency_hz
UL/DL CFR shapes both valid
UL/DL frequency grids differ as configured
```

完成 TDD uplink PUSCH 后，再开启 FDD 设计与实现。

## review

Review date: 2026-05-07

结论：13 号任务当前没有完成到可验收状态。代码中已经出现了 `LinkConfig`、TDD reciprocity helper、`nr_pusch_observation.py`、`nr_pusch_mvp.yaml` 和一组 NR PUSCH 配置单元测试，但整体仍停在 skeleton/metadata 层，尚未形成本文要求的“基于官方 Sionna 5G NR PUSCH 的 TDD uplink 建模闭环”。

已完成或部分完成：

- `sionna_measurement_sim/domain/link.py` 已有 `duplex_mode="tdd"`、`phy_link_direction="uplink"`、`rt_trace_direction="bs_to_ue"`、`reciprocity_mode="transpose_rt_channel"` 等字段，并且非 TDD、非 uplink 会 fail fast。
- `sionna_measurement_sim/phy/reciprocity.py` 已有 CFR/CIR 的 TX/RX 维度互换函数，`tests/unit/test_reciprocity.py` 覆盖了基础 shape。
- `sionna_measurement_sim/phy/nr_pusch_observation.py` 能构建 Sionna `PUSCHConfig` 和 `PUSCHTransmitter`，并能用 `cir_to_ofdm_channel` 将项目 CIR 转成频域 channel。
- `tests/unit/test_nr_pusch_config.py` 覆盖了 PUSCHConfig 构造、配置快照、CIR 输入下的 observation 函数 smoke test。
- `config/defaults/nr_pusch_mvp.yaml` 已提供 NR PUSCH 默认配置草案。

阻塞项：

1. NR PUSCH backend 没有接入主运行链路。`run_nr_pusch_observation()` 只在单元测试中被调用，`run-full`/`run-observation` 仍然只通过 `run_awgn_ls_observation()` 生成 observation。因此即使配置 `phy.standard: "nr_pusch"`，实际 HDF5 仍不会由 NR PUSCH backend 驱动。

2. 没有实现本文要求的 PUSCH receiver 闭环。`nr_pusch_observation.py` 只实例化了 `PUSCHTransmitter`，没有构建或运行 `PUSCHReceiver`、`PUSCHLSChannelEstimator`、LMMSE/KBest detector，也没有真实解码 transport block。BER/BLER 当前是固定占位值 `0.0`，不是链路仿真结果。

3. 当前 observed CFR 仍来自旧 AWGN+LS 管线。`nr_pusch_observation.py` 将 CIR 转成 CFR 后调用 `run_awgn_ls_observation()`，没有通过官方示例中的 `CIRDataset`/`OFDMChannel`/`ApplyOFDMChannel` 和 PUSCH receiver 路径产生 observed CSI。因此它不能证明“符合 5G NR PUSCH resource grid 和 DMRS 的 uplink observation”已经完成。

4. HDF5 writer/schema 没有写入本文要求的 NR PUSCH waveform/receiver 字段。当前 `_write_waveform()` 仍只写 `standard/sample_rate_hz/fft_size/pilot_indices/pilot_symbols` 等 custom OFDM 字段，没有写 `num_prb`、slot、DMRS、MCS、num_layers、num_antenna_ports 等字段；`_write_receiver()` 也没有写 `/receiver/receiver_type="pusch_receiver"`、`channel_estimator`、`mimo_detector`、`input_domain`。

5. `/evaluation/nmse_db` 语义仍和本文要求冲突。`EvaluationResult` 注释中仍写 `nmse_db` 是相对 impaired+noisy channel、`nmse_db_total` 是相对 clean H_true；而本文要求 `/evaluation/nmse_db` 必须表示 `H_obs` vs clean `H_true`，AWGN-only 诊断应另命名为 `nmse_awgn_db`。

6. 验收测试缺失。本文要求的 `tests/integration/test_nr_pusch_observation.py` 和 `tests/statistical/test_nr_pusch_link_metrics.py` 当前不存在；也没有覆盖高/低 Eb/N0 BER/BLER 单调性、perfect CSI 不差于 estimated CSI、HDF5 中 `/waveform/standard="nr_pusch"` 和 `/receiver/receiver_type="pusch_receiver"` 的端到端写入。

7. `_cir_to_cfr()` 失败时会 fallback 到全 1 CFR，这会掩盖 Sionna CIR 接入或 shape 错误。对于 13 的目标，这类失败应该 fail fast 或至少让验收测试失败，而不是继续生成看似有效的 observation。

8. CLI 未提供本文要求的 `run-nr-pusch` 或 `run-full --phy-standard nr_pusch` 路径。配置文件中存在 `phy.standard="nr_pusch"`，但 CLI 没有根据该字段选择 NR PUSCH backend。

本次检查运行过：

```bash
uv run pytest tests/unit/test_reciprocity.py tests/unit/test_nr_pusch_config.py
uv run ruff check .
```

结果：上述单元测试和 ruff 均通过。但这些测试只能证明 NR PUSCH 配置 skeleton 可构建，不能证明 13 的端到端验收通过。

建议下一步优先级：

1. 在 `RTTruthRunConfig` 或更高层 config 中真正传入完整 `phy`/`carrier` 配置，并在 `phy.standard=="nr_pusch"` 时调用 `run_nr_pusch_observation()`。
2. 将 NR PUSCH observation 的结果接入 `MeasurementSimulationResult`，扩展 domain model、writer、validator 和 `03_data_contract_hdf5.md`。
3. 用官方 Sionna PUSCH receiver/channel 路径替换 AWGN+LS fallback，至少实现 perfect CSI 和 estimated CSI 两种 smoke path。
4. 真实计算 BER/BLER 和 bit/block 计数，禁止固定占位值通过验收。
5. 新增本文要求的 integration/statistical tests，再用 `uv run pytest` 和 `uv run ruff check .` 作为最终质量门。
