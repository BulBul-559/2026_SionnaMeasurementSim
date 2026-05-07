# Review：15 号 MIMO 修复复核

复核日期：2026-05-07

复核范围：

- [15_mimo_phy_gap_analysis.md](15_mimo_phy_gap_analysis.md)
- `sionna_measurement_sim/phy/nr_pusch_observation.py`
- `sionna_measurement_sim/phy/nr_mimo_channel.py`
- NR PUSCH MIMO 相关配置、schema validator 和新增测试。

## 结论

15 号文档中最核心的旧问题已经明显修复：正式 NR PUSCH backend 中不再出现 `h_tensor[0, 0, 0, 0, :]`，也不再把单个 SISO LS estimate 直接 `broadcast_to` 成完整 MIMO shape。新增的 4x4 SU-MIMO integration/schema/unit 测试可以自生成 HDF5 并通过。

但还不能标记为“15 号 MIMO 方案全部完成”。当前实现达到了 **4x4 SU-MIMO perfect-CSI 主链路可运行**，但距离 15 号文档设定的官方式完整 MIMO PHY 闭环仍有差距。

## 仍未修复的问题

### 1. 仍未按 15 号文档实现 `CIRDataset + OFDMChannel` 官方式 channel bridge (仍 unresolved)

15 号文档建议新增 MIMO bridge，将项目 CIR 包装为 Sionna `CIRDataset` / channel model，并通过：

```text
OFDMChannel(channel_model, tx.resource_grid, return_channel=True)
```

驱动 PUSCH 链路。

当前实现实际使用的是：

```text
build_mimo_cfr_from_cir(...)
cir_to_ofdm_channel(...)
ApplyOFDMChannel()
```

也就是先把项目 CIR 转成 CFR，再逐个 `(snapshot, ul_tx, ul_rx)` 构造 perfect `h`，最后用 `ApplyOFDMChannel` 施加信道。

影响：

- 当前 4x4 SU-MIMO 能跑通，但没有形成官方示例中 `CIRDataset -> OFDMChannel(return_channel=True)` 的统一 channel model。
- 后续要扩展多 snapshot、time-varying channel、官方 channel model 复用、批量仿真时，当前逐 link ApplyOFDMChannel 路线维护成本更高。
- 如果项目接受这个实现路线，需要更新 15 号文档，把 `CIRDataset + OFDMChannel` 从强要求改为可选优化；否则这项仍未完成。

### 2. MU-MIMO 仍只是配置 helper，不是实际并发多 UE PUSCH 链路

当前 `build_multiuser_pusch_configs(...)` 可以通过手工设置 `num_pusch_tx` 生成多个 `PUSCHConfig`，DMRS port set 也有单元测试。

但主链路 `run_nr_pusch_observation(...)` 没有把 `channel.num_ul_tx` 传给 `build_multiuser_pusch_configs(...)`。默认仍是：

```text
num_tx = 1
```

随后代码按 `(snapshot, ul_tx, ul_rx)` 循环逐 link 处理，每个 link 复用单 UE transmitter/receiver。

影响：

- 这不是官方意义上的多 UE 同时上行 MU-MIMO。
- 多个 project RX/UE 时，当前更像“多个单 UE 链路逐个跑”，而不是同一 resource grid 上的多 UE PUSCH。
- `StreamManagement(num_rx=1, num_tx=_num_pusch_tx, ...)` 在主链路中不会自动覆盖实际 `num_ul_tx > 1` 的场景。

验收缺口：

- 需要让 `num_pusch_tx = channel.num_ul_tx` 成为真实主链路配置，或显式 fail fast：当前只支持 SU-MIMO。
- 需要新增真正的 MU-MIMO integration test，而不是只测 config helper。

### 3. BLER 仍不是完整 TB/CRC 语义 (仍 unresolved)

这不是 15 号文档的首要 MIMO 修复点，但它仍影响“官方级 PUSCH link-level”结论。

当前 PUSCHReceiver 仍使用：

```text
tb_decoder=None
return_tb_crc_status=False
```

BLER 仍是根据 bit errors 简化得到，不是完整 transport block / CRC 结果。

影响：

- MIMO channel 和 detector 可以先验收，但不能宣称完整 5G NR PUSCH BLER 仿真已完成。

## 已解决的旧问题

以下旧 review 问题本次复核认为已经解决，不再保留为阻塞项：

- 正式 NR PUSCH backend 中的 `h_tensor[0, 0, 0, 0, :]` SISO 取法已移除。
- `/observation/cfr_est` 不再由单个 SISO LS estimate 直接 broadcast 成完整 shape。
- 已新增 `nr_mimo_channel.py`，并有 shape / reciprocity 单元测试。
- 已新增 `StreamManagement` 和 `LinearDetector` / `KBestDetector` builder。
- `/receiver/mimo_detector` 已从配置透传到 `ReceiverSpec`，并有测试检查。
- 4x4 SU-MIMO perfect CSI integration test 已能自生成 HDF5。
- NR PUSCH schema validator 已开始强制校验 NR PUSCH 关键字段。

## 第二轮修复 (2026-05-07) 已解决

- **SU-MIMO 边界已明确**：`mimo_mode="su_mimo"` 字段写入 `PHYConfig` / `ReceiverConfig` / `RTTruthRunConfig`；非 `su_mimo` 模式直接 `NotImplementedError`。
- **MU-MIMO helper 已标记为 future**：`build_multiuser_pusch_configs` 保留但主链路默认 `num_pusch_tx=1`；不暗示 MU-MIMO 主链路已完成。
- **estimated CSI 零填充已移除**：`num_layers < num_antenna_ports` 时 `perfect_csi=False` 代码现在抛出 `NotImplementedError`，不再用零填充伪装成完整 physical antenna-pair CFR。测试覆盖了错误路径和合法路径（`num_layers=4, num_antenna_ports=4`）。
- **4x4 MIMO perfect vs estimated CSI 统计测试已补齐**：新增 `tests/statistical/test_nr_pusch_mimo_metrics.py`（7 个测试），验证 NMSE/BER 单调性、perfect CSI 不劣于 estimated CSI、CFR shape 一致性和 `mimo_detector` metadata。
- **`ebno_db` 字段已加入 `RTTruthRunConfig`**。

## 本次核查命令

```bash
uv run pytest tests/unit/test_nr_mimo_channel_bridge.py tests/unit/test_nr_pusch_mimo_config.py tests/integration/test_nr_pusch_mimo_observation.py tests/schema/test_nr_pusch_schema.py tests/statistical/test_nr_pusch_mimo_metrics.py
uv run pytest
uv run ruff check .
```

结果：

```text
48 passed (MIMO-specific), 166 passed (full suite), 2 warnings
ruff: All checks passed
```

## 下一步优先级

1. 实现官方式 `CIRDataset + OFDMChannel` channel backend（Step 4-5 from 执行指南）。
2. 实现 MU-MIMO 主链路（Step 6）。
3. 实现完整 TB/CRC BLER（Step 7）。
4. 以上三项应独立提交，不可混在一个 PR 中。

## 补丁还是重构

建议采用 **分阶段定向重构**，不是推倒重写，也不是只打零散补丁。

原因：

- 当前 4x4 SU-MIMO perfect CSI 已经能跑通，说明现有 `nr_pusch_observation.py + nr_mimo_channel.py` 的基本方向可用，不值得整体推翻。
- 剩余问题分成两类：一类是明确 bug 或边界缺失，适合补丁；另一类是 channel bridge 与官方示例架构不一致，属于架构选择，继续打补丁会让主函数越来越复杂。
- 最危险的不是 `ApplyOFDMChannel` 本身，而是当前代码没有把“当前只稳定支持 SU-MIMO perfect CSI”这个边界写死，导致 estimated CSI、MU-MIMO 和 full physical CSI 容易被误认为已经完成。

建议执行路线：

1. **先补丁收口**：用最小改动修正错误语义，明确当前支持范围，补统计测试。
2. **再小范围重构**：把 channel bridge 抽成 backend 接口，保留当前 `ApplyOFDMChannel` backend，同时新增或替换为官方式 `CIRDataset + OFDMChannel` backend。
3. **最后扩展 MU-MIMO 和完整 BLER**：不要和前两步混在一个 PR / commit 中。

不建议现在直接大重构：

- 当前已有 40 个 MIMO 相关测试通过，大重构会扩大回归面。
- 官方 `CIRDataset + OFDMChannel` 的 shape、batch、time step、stream management 需要单独验证；把它和 estimated CSI、MU-MIMO、BLER 同时改，定位失败会很痛苦。

不建议只打补丁长期维持：

- `run_nr_pusch_observation(...)` 已经包含 config、channel、detector、receiver、估计、指标、写回 shape 多种职责。
- 如果继续往里面堆 MU-MIMO、estimated physical CSI、CIRDataset、BLER，函数会变成不可维护的调度脚本。

## 详细执行指南

### Step 0：建立当前基线

先运行当前 MIMO 质量门，确认不是在坏基线上继续改：

```bash
uv run pytest tests/unit/test_nr_mimo_channel_bridge.py tests/unit/test_nr_pusch_mimo_config.py tests/integration/test_nr_pusch_mimo_observation.py tests/schema/test_nr_pusch_schema.py
uv run ruff check .
git status --short
```

预期基线：

```text
40 passed, 2 warnings
ruff: All checks passed
```

### Step 1：补丁收口 SU-MIMO 支持边界

目标：先诚实声明并强制当前主链路只验收 SU-MIMO，避免 MU-MIMO helper 被误认为主链路已经支持多 UE 并发。

修改文件：

```text
sionna_measurement_sim/config/schema.py
sionna_measurement_sim/rt/truth_pipeline.py
sionna_measurement_sim/phy/nr_pusch_observation.py
docs/06_config_and_experiment_schema.md
docs/15_mimo_phy_gap_analysis.md
```

建议修改：

1. 在 `PHYConfig` 和 `RTTruthRunConfig` 增加显式字段：

```python
mimo_mode: str = "su_mimo"  # allowed: "su_mimo"; future: "mu_mimo"
channel_backend: str = "apply_ofdm"  # future: "cir_dataset_ofdm"
```

2. 在 `run_nr_pusch_observation(...)` 中，在构建 `channel` 后立即 fail fast：

```python
if getattr(phy_config, "mimo_mode", "su_mimo") != "su_mimo":
    raise NotImplementedError("Only su_mimo is currently supported")

if channel.num_ul_tx != 1 or channel.num_ul_rx != 1:
    raise NotImplementedError(
        "NR PUSCH currently supports one UE and one BS receiver in SU-MIMO mode"
    )
```

3. 保留 `build_multiuser_pusch_configs(...)` 单元测试，但在文档中标记为 future helper，不作为当前验收完成项。

4. 新增测试：

```text
tests/unit/test_nr_pusch_mimo_config.py
```

增加断言：

- `mimo_mode="su_mimo"` 且 `max_tx=max_rx=1` 可以跑；
- `mimo_mode="mu_mimo"` 抛 `NotImplementedError`；
- `channel.num_ul_tx > 1` 或 `channel.num_ul_rx > 1` 抛 `NotImplementedError`。

### Step 2：修复 estimated CSI 的零填充问题

目标：禁止把 effective stream estimate 补零伪装成完整 physical antenna-pair CFR。

修改文件：

```text
sionna_measurement_sim/phy/nr_pusch_observation.py
tests/integration/test_nr_pusch_mimo_observation.py
tests/schema/test_nr_pusch_schema.py
docs/03_data_contract_hdf5.md
docs/15_mimo_phy_gap_analysis.md
```

当前风险代码：

```text
cfr_est_slice = np.zeros(...)
cfr_est_slice[:, :h_hat_dim4, :] = cfr_slice_eff
```

建议短期补丁：

```python
if not perfect_csi and h_hat.shape[4] != channel.cfr.shape[4]:
    raise NotImplementedError(
        "Estimated CSI physical antenna-pair CFR requires "
        "num_layers == num_antenna_ports; effective-channel export is not yet supported"
    )
```

同时把当前 estimated CSI smoke test 改成两类：

1. `num_layers=4, num_antenna_ports=4, perfect_csi=False` 应继续通过。
2. `num_layers=1, num_antenna_ports=4, perfect_csi=False` 应明确抛 `NotImplementedError`，除非本轮决定修改 HDF5 契约支持 effective channel。

如果想支持 effective channel，不要写入 `/observation/cfr_est` 冒充 physical CFR。应新增契约，例如：

```text
/observation/effective_cfr_est
```

并在 `/receiver` 或 `/observation` metadata 中写：

```text
csi_domain = "effective_stream_channel"
```

这条路线需要先更新 `docs/03_data_contract_hdf5.md` 和 schema validator。

### Step 3：补 4x4 MIMO statistical 测试

目标：补齐 15 号文档要求的 perfect CSI vs estimated CSI 统计验收。

新增文件：

```text
tests/statistical/test_nr_pusch_mimo_metrics.py
```

建议测试配置：

```text
max_tx=1
max_rx=1
tx_num_rows=2
tx_num_cols=2
rx_num_rows=2
rx_num_cols=2
num_layers=4
num_antenna_ports=4
num_prb=4
mimo_detector="lmmse"
receiver_failure_policy="fail_fast"
```

建议测试项：

1. `perfect_csi=True, ebno_db=30` 的 BER/BLER/NMSE 不差于 `perfect_csi=False, ebno_db=30`。
2. `perfect_csi=False, ebno_db=30` 的 BER/BLER/NMSE 不差于 `perfect_csi=False, ebno_db=10`。
3. 所有文件满足：

```text
/observation/cfr_est.shape[1:] == /channel/truth/cfr.shape
/observation/estimation_success 全 true
/receiver/mimo_detector == "lmmse"
```

注意：

- 如果 Sionna API 缺失可 `pytest.skip`。
- 如果 receiver runtime 失败，不要 skip，应 fail。

### Step 4：把 channel bridge 做成可插拔 backend

目标：小范围重构，不直接推翻当前通过测试的 `ApplyOFDMChannel` 路线。

新增或重构文件：

```text
sionna_measurement_sim/phy/nr_mimo_channel.py
sionna_measurement_sim/phy/nr_channel_backend.py
```

建议接口：

```python
class NRChannelBackend(Protocol):
    def build(self, cir_coefficients, cir_delays, link_config, phy_config) -> NRChannelBundle:
        ...

    def apply(self, x, no, snap_idx: int, ul_tx_idx: int, ul_rx_idx: int):
        ...

    def perfect_h(self, snap_idx: int, ul_tx_idx: int, ul_rx_idx: int):
        ...
```

先实现两个 backend：

1. `ApplyOFDMChannelBackend`：包装当前 `build_mimo_cfr_from_cir(...) + ApplyOFDMChannel` 逻辑，保持现有测试通过。
2. `CIRDatasetOFDMChannelBackend`：按官方示例实现 `CIRDataset + OFDMChannel(return_channel=True)`。

这样做的好处：

- 当前 SU-MIMO 不被重构破坏。
- 是否采用官方 channel backend 可以由配置切换：

```yaml
phy:
  channel_backend: "apply_ofdm"          # current stable
  # channel_backend: "cir_dataset_ofdm"  # official-style backend
```

### Step 5：实现官方式 `CIRDataset + OFDMChannel` backend

目标：满足 15 号文档中最强的官方对齐要求。

实现要点：

1. 写一个 project CIR generator，输出 Sionna CIRDataset 需要的：

```text
a:   [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
tau: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
```

2. 使用：

```python
from sionna.phy.channel import CIRDataset, OFDMChannel

dataset = CIRDataset(
    cir_generator=generator,
    batch_size=batch_size,
    num_rx=num_rx,
    num_rx_ant=num_rx_ant,
    num_tx=num_tx,
    num_tx_ant=num_tx_ant,
    num_paths=num_paths,
    num_time_steps=num_time_steps,
)

channel = OFDMChannel(
    dataset,
    tx.resource_grid,
    normalize_channel=False,
    return_channel=True,
)

y, h = channel(x, no)
```

3. `h` 必须通过统一函数转换回：

```text
[snapshot, tx, rx, rx_ant, tx_ant, subcarrier]
```

4. 新增非对称 shape 单测：

```text
snapshot=2
tx=1
rx=3
rx_ant=4
tx_ant=2
path=5
```

5. 新增 backend 对比测试：

- 同一份静态 CIR 下，`ApplyOFDMChannelBackend.perfect_h` 与 `CIRDatasetOFDMChannelBackend` 返回的 `h` 在 shape 和数值上应接近。

### Step 6：再做 MU-MIMO

只有在 SU-MIMO + estimated CSI + backend 抽象稳定后再做。

修改点：

```text
sionna_measurement_sim/phy/nr_pusch_observation.py
sionna_measurement_sim/phy/nr_channel_backend.py
tests/integration/test_nr_pusch_mu_mimo_observation.py
```

实现要求：

- `num_pusch_tx = channel.num_ul_tx`。
- `PUSCHTransmitter` 接收 `list[PUSCHConfig]`。
- 每个 UE 的 `dmrs_port_set` 不重叠。
- `StreamManagement(rx_tx_association=np.ones([num_ul_rx, num_ul_tx]), num_layers)`。
- 不再逐 UE 单独调用 receiver，而是在同一个 resource grid 中并发发射和检测。

验收：

- `max_rx > 1` 的 project UE 场景能生成多 UE uplink PUSCH。
- BER/BLER/NMSE 以 per UE 或 per link 形式写入，HDF5 契约先更新再实现。

### Step 7：最后补完整 TB/CRC BLER

这一步可以独立于 MIMO channel bridge。

目标：

- 使用 Sionna 的 TB decoder / CRC status。
- `return_tb_crc_status=True`。
- `/evaluation/bler`、`num_block_errors`、`num_blocks` 具有真实 transport block 语义。

不要在 Step 1-5 中混入这一步。

## 建议的提交切分

建议拆成 4 个提交：

1. `review: document remaining mimo execution plan`
2. `fix: enforce su-mimo boundaries and remove estimated-csi padding`
3. `test: add nr pusch mimo statistical acceptance`
4. `refactor: add pluggable nr channel backend`

如果继续实现官方 backend，再单独提交：

```text
refactor: add cir-dataset ofdm channel backend
```

每个提交前至少运行：

```bash
uv run pytest tests/unit/test_nr_mimo_channel_bridge.py tests/unit/test_nr_pusch_mimo_config.py tests/integration/test_nr_pusch_mimo_observation.py tests/schema/test_nr_pusch_schema.py
uv run ruff check .
```
