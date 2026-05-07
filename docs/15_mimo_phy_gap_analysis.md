# 15. NR PUSCH MIMO 问题分析与修复方案

日期：2026-05-07

本文只关注当前 review 中与 MIMO 直接相关的问题。目标是给后续执行 agent 一个可落地的修复说明：当前系统为什么没达标，哪些关键实现位置有问题，以及应该按什么方法修改。

## 结论

当前系统的 RT truth 侧已经能表达 4x4 MIMO：`/channel/truth/cfr` 使用 `[tx, rx, rx_ant, tx_ant, subcarrier]`，CIR 使用 `[snapshot, tx, rx, rx_ant, tx_ant, path]`。问题出在 NR PUSCH PHY observation 侧：代码只取第一个 TX/RX/antenna pair 做 SISO-like 链路，然后把单链路 LS 估计 broadcast 成完整 MIMO shape。

因此，当前输出的 MIMO shape 是对的，但 PHY 内容不是完整 MIMO 接收机处理结果。它不能标记为“4x4 MIMO NR PUSCH 闭环完成”。

## 参考基线

官方 Sionna RT link-level 示例的 MIMO PUSCH 流程是：

1. 用 RT 生成 CIR。
2. 将 CIR 包装为 `sionna.phy.channel.CIRDataset`。
3. 用 `OFDMChannel(channel_model, pusch_transmitter.resource_grid, return_channel=True)` 施加多用户 MIMO OFDM channel。
4. 为每个 UE 构建 `PUSCHConfig`，并为不同 UE 配置不同 DMRS port set。
5. 用 `StreamManagement(rx_tx_association, num_layers)` 建立 RX/TX stream 关系。
6. 用 `LinearDetector` 或 `KBestDetector` 作为 MIMO detector。
7. `perfect_csi=True` 时把 `OFDMChannel` 返回的 `h` 传给 `PUSCHReceiver`；否则让 `PUSCHReceiver` 内部使用 PUSCH channel estimator。

官方资源：

- Sionna RT link-level tutorial: https://nvlabs.github.io/sionna/phy/tutorials/notebooks/Link_Level_Simulations_with_RT.html
- Sionna 5G NR PUSCH tutorial: https://nvlabs.github.io/sionna/phy/tutorials/notebooks/5G_NR_PUSCH.html
- 重点 API：`CIRDataset`、`OFDMChannel`、`PUSCHConfig`、`PUSCHTransmitter`、`PUSCHReceiver`、`PUSCHLSChannelEstimator`、`StreamManagement`、`LinearDetector`、`KBestDetector`。

## 当前系统为什么没达标

### 1. NR PUSCH backend 显式丢弃 MIMO 维度

位置：

- `sionna_measurement_sim/phy/nr_pusch_observation.py`
- `run_nr_pusch_observation(...)`

当前关键代码：

```text
h_tensor = torch.as_tensor(cfr_np[0], dtype=torch.complex64)
h_ch = h_tensor[0, 0, 0, 0, :]
tx_freq = tx_signal[0, 0, 0, :, :]
y_freq = tx_freq * h_ch.unsqueeze(0)
y_rx = y_noisy.unsqueeze(0).unsqueeze(0).unsqueeze(0)
```

问题：

- `cfr_np[0]` 只取第一个 snapshot。
- `[0, 0, 0, 0, :]` 只取第一个 TX、第一个 RX、第一个 RX antenna、第一个 TX antenna。
- `tx_signal[0, 0, 0, :, :]` 只取第一个 transmitter 和第一个 stream。
- `y_rx` 被构造成 `[1, 1, 1, symbols, subcarriers]`，接收机看到的是 SISO，而不是 `[batch, num_rx, num_rx_ant, symbols, fft_size]` 的真实 MIMO 输入。

这意味着即使 RT truth 是 4x4 MIMO，NR PUSCH backend 实际也只跑了一个 SISO 子链路。

### 2. `cfr_est` 是单链路 LS 后广播，不是 MIMO CSI

位置：

- `sionna_measurement_sim/phy/nr_pusch_observation.py`
- `_ls_estimate_cfr(...)`
- `run_nr_pusch_observation(...)`

当前关键代码：

```text
cfr_est = _ls_estimate_cfr(y_noisy.cpu().numpy(), pusch_cfg, num_subcarriers)
if cfr_est.shape != cfr_clean_ref.shape:
    cfr_est = np.broadcast_to(cfr_est, cfr_clean_ref.shape).copy()
```

问题：

- `_ls_estimate_cfr(...)` 的输入是单个 `y_noisy`，shape 为 `[num_ofdm_symbols, num_subcarriers]`。
- 返回值固定是 `[1, 1, 1, 1, 1, subcarrier]`。
- 后续为了满足 HDF5 shape contract，把这个单链路估计复制到所有 TX/RX/antenna 组合。

影响：

- `/observation/cfr_est` 的 shape 可以通过 validator，但每个 MIMO link 的内容并不是对应天线对的估计。
- `evaluation/nmse_db` 会把同一个单链路估计和所有 MIMO truth link 比较，不能代表真实 MIMO estimator 性能。

### 3. PUSCHConfig 没有按多 UE / 多 stream 建模

位置：

- `sionna_measurement_sim/phy/nr_pusch_observation.py`
- `build_nr_pusch_config(...)`

当前实现只返回单个 `PUSCHConfig`，然后：

```text
tx = PUSCHTransmitter(pusch_cfg, output_domain="freq", return_bits=True)
```

问题：

- 多用户 uplink 中，项目维度 `tx` 应该对应多个 UE transmitter。
- 官方方式是为多个 transmitter 提供 `list[PUSCHConfig]`，并为每个 UE 分配不同 DMRS port set。
- 当前没有根据 `num_tx` 生成多个 `PUSCHConfig`。
- 当前没有显式校验 `num_layers * num_tx <= num_antenna_ports` 或 DMRS port 资源是否足够。

### 4. 没有 StreamManagement，MIMO detector 语义没有生效

位置：

- `sionna_measurement_sim/phy/nr_pusch_observation.py`
- `sionna_measurement_sim/domain/observation.py`
- `sionna_measurement_sim/io/hdf5_writer.py`

当前 receiver 构造：

```text
rx = PUSCHReceiver(
    pusch_transmitter=tx,
    tb_decoder=None,
    return_tb_crc_status=False,
    input_domain="freq",
)
```

问题：

- 没有构造 `StreamManagement(rx_tx_association, num_layers)`。
- 没有根据配置创建 `LinearDetector` 或 `KBestDetector`。
- HDF5 中可以写 `/receiver/mimo_detector`，但这个字段没有真正驱动 backend。

影响：

- `num_layers`、`num_antenna_ports`、`mimo_detector` 只是配置/metadata 层面的字段，不是实际 MIMO 接收处理链路。

### 5. 测试没有覆盖 4x4 NR PUSCH MIMO observation

位置：

- `tests/integration/test_nr_pusch_observation.py`
- `tests/statistical/test_nr_pusch_link_metrics.py`
- `tests/integration/test_rt_mimo_4x4_pipeline.py`

当前状态：

- NR PUSCH integration/statistical tests 使用 `tx_num_rows=1, tx_num_cols=1, rx_num_rows=1, rx_num_cols=1, num_antenna_ports=1`。
- 4x4 MIMO tests 主要验证已有输出文件的 RT truth/CIR/path shape，且文件不存在时 skip。
- 没有测试断言 NR PUSCH backend 在 4x4 truth 下实际消费了所有 antenna links。
- 没有测试防止 `cfr_est` 由单链路广播产生。

因此当前测试可以证明“NR PUSCH SISO smoke 能跑”，不能证明“NR PUSCH 4x4 MIMO 闭环正确”。

## 建议修复架构

### 新增 MIMO channel bridge

新增文件：

```text
sionna_measurement_sim/phy/nr_mimo_channel.py
```

职责：

- 输入项目 CIR：

```text
coefficients: [snapshot, tx, rx, rx_ant, tx_ant, path]
delays_s:     [snapshot, tx, rx, rx_ant, tx_ant, path]
```

- 根据 `link_config` 做 TDD reciprocity：

```text
[snapshot, tx, rx, rx_ant, tx_ant, path]
-> [snapshot, ul_tx, ul_rx, ul_rx_ant, ul_tx_ant, path]
```

- 转换成 Sionna `CIRDataset` / channel model 需要的 shape。
- 保留维度映射 metadata，供写回 `/observation/cfr_est`、`/evaluation/nmse_db` 和调试使用。

建议接口：

```python
@dataclass(frozen=True)
class NRMIMOChannelBundle:
    channel_model: Any
    num_tx: int
    num_rx: int
    num_tx_ant: int
    num_rx_ant: int
    num_paths: int
    num_time_steps: int
    reciprocity_applied: bool
    project_to_ul_permutation: tuple[int, ...]


def build_cir_dataset_from_project_cir(
    cir_coefficients: np.ndarray,
    cir_delays_s: np.ndarray,
    *,
    link_config,
    batch_size: int,
    device: str | None = None,
) -> NRMIMOChannelBundle:
    ...
```

实现方法：

- 参考官方 tutorial 的 `CIRGenerator`。
- 不要在 business/domain/HDF5 writer 读取 Sionna `Paths`；这里的输入必须是项目 domain CIR。
- 如果 `snapshot > 1`，先支持 `batch_size == snapshot` 或按 snapshot 迭代；不要静默只取 `snapshot=0`。
- shape 转换必须写单元测试，使用非对称 shape，例如 `snapshot=2, tx=3, rx=1, rx_ant=4, tx_ant=2, path=5`，避免 transpose 写反但测试仍通过。

### 重构 NR PUSCH backend

修改文件：

```text
sionna_measurement_sim/phy/nr_pusch_observation.py
```

把当前 `run_nr_pusch_observation(...)` 拆成以下层次：

```text
build_multiuser_pusch_configs(...)
build_mimo_detector(...)
run_sionna_pusch_mimo(...)
extract_project_cfr_est(...)
build_observation_result(...)
```

#### 1. `build_multiuser_pusch_configs(...)`

输入：

```text
num_tx
num_tx_ant
num_layers
phy_config
carrier_config
```

输出：

```text
list[PUSCHConfig]
```

规则：

- 每个 project TX 视为一个 uplink UE。
- `num_antenna_ports` 默认应等于 UE TX antenna 数，即 uplink 视角的 `num_tx_ant`。
- 对第 `i` 个 UE，设置：

```text
pc.dmrs.dmrs_port_set = list(range(i*num_layers, (i+1)*num_layers))
```

- 校验：

```text
num_layers >= 1
num_tx >= 1
num_antenna_ports == num_tx_ant  # 首轮建议强约束
num_tx * num_layers <= 可用 DMRS ports
```

- 当 `num_layers < num_antenna_ports` 时使用 codebook precoding，并显式设置 `tpmi`。首轮可以默认 `tpmi=1`，但要进入 config snapshot。

#### 2. `build_mimo_detector(...)`

输入：

```text
resource_grid
stream_management
mimo_detector: "lmmse" | "kbest"
num_bits_per_symbol
```

输出：

```text
LinearDetector 或 KBestDetector
```

方法：

- `mimo_detector="lmmse"` 时使用 `sionna.phy.ofdm.LinearDetector(equalizer="lmmse", output="bit", demapping_method="maxlog", ...)`。
- `mimo_detector="kbest"` 时使用 `sionna.phy.ofdm.KBestDetector(output="bit", num_streams=num_tx*num_layers, k=64, ...)`。
- `StreamManagement` 使用：

```python
rx_tx_association = np.ones([num_rx, num_tx], dtype=bool)
stream_management = StreamManagement(rx_tx_association, num_layers)
```

首轮若只支持单 BS RX，可强制 `num_rx == 1`，但必须 fail fast，不要自动取第一个 RX。

#### 3. `run_sionna_pusch_mimo(...)`

替换当前手写：

```text
y_freq = tx_freq * h_ch + noise
```

改为官方方式：

```python
tx = PUSCHTransmitter(pusch_configs, output_domain="freq", return_bits=True)
channel = OFDMChannel(channel_model, tx.resource_grid, normalize_channel=False, return_channel=True)
y, h = channel(x, no)
if perfect_csi:
    rx_bits = rx(y, no, h)
else:
    rx_bits = rx(y, no)
```

注意：

- `no` 应优先由 `ebnodb2no(ebno_db, num_bits_per_symbol, target_coderate, resource_grid)` 计算，而不是用 `signal_power / snr_linear` 手写。
- `return_channel=True` 返回的 `h` 是 perfect CSI 路径和 truth 对齐的关键，不应该丢掉。
- receiver 失败应默认 fail fast。若保留 fallback，必须由配置 `receiver.failure_policy` 控制，且验收默认不允许吞异常。

#### 4. `extract_project_cfr_est(...)`

目标：

- 写入 `/observation/cfr_est` 的必须是 `[snapshot, tx, rx, rx_ant, tx_ant, subcarrier]`。
- 如果 link_config 使用 TDD reciprocity，写回时必须反变换到 HDF5 契约定义的 project orientation。

建议两阶段实现：

第一阶段：

- 对 `perfect_csi=True`，直接从 `OFDMChannel(..., return_channel=True)` 返回的 `h` 抽取 full MIMO CFR，转换回 project shape，作为 `/observation/cfr_est`。
- 这能先验证 MIMO 维度链路和 reciprocity 写回是否正确。

第二阶段：

- 对 `perfect_csi=False`，显式实例化 `PUSCHLSChannelEstimator`：

```python
estimator = PUSCHLSChannelEstimator(
    tx.resource_grid,
    dmrs_length=phy_config.pusch_dmrs_length,
    dmrs_additional_position=phy_config.pusch_dmrs_additional_position,
    num_cdm_groups_without_data=phy_config.pusch_num_cdm_groups_without_data,
    interpolation_type=receiver_config.interpolation_method,
)
h_hat, err_var = estimator(y, no)
```

- 将 `h_hat` 转为项目 shape 后写 `/observation/cfr_est`。
- 同时写 `/observation/estimator_error_variance` 或复用现有 `estimator_noise_var` 契约字段。如果 03 契约没有最终字段名，需要先更新契约和 validator。

### 配置与 domain 调整

修改文件：

```text
sionna_measurement_sim/config/schema.py
sionna_measurement_sim/rt/truth_pipeline.py
sionna_measurement_sim/domain/observation.py
sionna_measurement_sim/io/hdf5_writer.py
sionna_measurement_sim/io/schema_validator.py
docs/03_data_contract_hdf5.md
docs/06_config_and_experiment_schema.md
```

具体修改：

1. `ReceiverConfig` 增加：

```python
mimo_detector: str = "lmmse"
channel_estimator: str = "pusch_ls"  # 或统一到 estimator_type
```

2. `RTTruthRunConfig` 增加并透传：

```python
mimo_detector: str = "lmmse"
receiver_failure_policy: str = "fail_fast"
```

3. `truth_pipeline._run_nr_pusch_obs(...)` 不要只把 `config` 当作 `phy_config`，应显式传入 `receiver_config` 或把 receiver 字段纳入 run config。

4. `ReceiverSpec` 中的 `/receiver/mimo_detector` 必须来自配置并真实驱动 detector，而不是默认 metadata。

5. `schema_validator` 在 `/waveform/standard == "nr_pusch"` 时强制校验：

```text
/waveform/num_layers
/waveform/num_antenna_ports
/receiver/mimo_detector
/receiver/estimator_type 或 /receiver/channel_estimator
/observation/cfr_est.shape[1:] == /channel/truth/cfr.shape
```

6. 增加校验：当文件声明 `nr_pusch` 且 `tx_ant/rx_ant > 1` 时，不能出现所有 MIMO antenna pair 的 `cfr_est` 完全相同。这个校验更适合测试，不建议放入通用 schema validator。

### 测试修复计划

新增或修改测试：

```text
tests/unit/test_nr_mimo_channel_bridge.py
tests/unit/test_nr_pusch_mimo_config.py
tests/integration/test_nr_pusch_mimo_observation.py
tests/statistical/test_nr_pusch_mimo_metrics.py
tests/schema/test_nr_pusch_schema.py
```

#### 1. shape bridge 单元测试

目标：

- 验证 project CIR -> Sionna CIRDataset 输入 shape -> project CFR 写回 shape。
- 使用非对称维度，确保 transpose 不会误通过。
- 验证 reciprocity 前后 TX/RX/antenna 维度互换正确。

#### 2. PUSCH 多用户配置测试

目标：

- `num_tx=4, num_layers=1, num_antenna_ports=4` 时生成 4 个 `PUSCHConfig`。
- 每个 config 的 DMRS port set 不冲突。
- `num_layers < num_antenna_ports` 时 `precoding="codebook"`。
- 非法组合 fail fast。

#### 3. 4x4 MIMO integration test

目标配置：

```text
tx_num_rows=2
tx_num_cols=2
rx_num_rows=2
rx_num_cols=2
max_tx=1 或 4
max_rx=1
phy_standard="nr_pusch"
num_prb=4
num_layers=1
num_antenna_ports=4
mimo_detector="lmmse"
perfect_csi=True
```

首轮建议先用 `max_tx=1, max_rx=1, 4x4 antenna` 打通 SU-MIMO，再扩展 `max_tx=4, max_rx=1` 的 MU-MIMO。

必须断言：

```text
/channel/truth/cfr.shape[2] == 4
/channel/truth/cfr.shape[3] == 4
/observation/cfr_est.shape[1:] == /channel/truth/cfr.shape
/receiver/mimo_detector == "lmmse"
/observation/estimation_success 全 true
```

并增加防广播断言：

```python
cfr_est = h5["observation/cfr_est"][()]
assert not np.allclose(cfr_est[..., 0, 0, :], cfr_est[..., 1, 1, :])
```

实际断言应根据有效 path/信道强度选取非零 antenna pair，避免极端静态场景误判。

#### 4. estimated CSI 统计测试

目标：

- 跑 `perfect_csi=True` 和 `perfect_csi=False`。
- 验证 perfect CSI 的 BER/BLER 不差于 estimated CSI。
- 验证高 Eb/N0 比低 Eb/N0 更好。
- 如果 Sionna receiver 因环境问题失败，只允许 `ImportError` skip；其它异常必须 fail。

### 分阶段落地顺序

建议不要一次性把 SU-MIMO、MU-MIMO、estimated CSI、BLER 全部混在一起改。

1. `NRMIMOChannelBundle`：只做 CIR shape bridge 和 reciprocity 单测。
2. `perfect_csi=True` SU-MIMO：使用 `OFDMChannel(return_channel=True)` 和 `PUSCHReceiver(channel_estimator="perfect")` 打通 4x4。
3. 写回 full MIMO `/observation/cfr_est`，移除 broadcast。
4. 加入 `StreamManagement` 和 `LinearDetector`，让 `/receiver/mimo_detector` 真实生效。
5. 扩展到 estimated CSI：使用 `PUSCHLSChannelEstimator` 或 PUSCHReceiver 默认 estimator。
6. 扩展到 MU-MIMO：多个 project TX -> 多个 `PUSCHConfig`，不同 DMRS port set。
7. 最后补 TB/CRC BLER 语义。

## 验收标准

MIMO 相关问题修复后，至少需要满足：

1. 代码中不再出现用于正式 NR PUSCH backend 的 `h_tensor[0, 0, 0, 0, :]`。
2. NR PUSCH 4x4 integration test 不 skip，能自生成 HDF5。
3. `/observation/cfr_est` 来自 full MIMO `h` 或 full MIMO estimator，不允许由单链路 broadcast。
4. `/receiver/mimo_detector` 与实际 detector 一致。
5. `num_layers`、`num_antenna_ports`、DMRS port set、StreamManagement 有单元测试。
6. `perfect_csi=True` 至少能跑通 SU-MIMO 4x4。
7. `perfect_csi=False` 至少能跑通 SU-MIMO 4x4 LS 估计，且 NMSE/BER 随 Eb/N0 改善。
8. HDF5 schema 对 `nr_pusch` 扩展字段有强校验。

## 本轮未修改代码

本文是分析与修复方案文档，没有修改实现代码。当前 review 中 MIMO 相关的未修复问题仍以 [review.md](review.md) 为准。
