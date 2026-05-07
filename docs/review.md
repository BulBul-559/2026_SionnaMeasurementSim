# Review：NR PUSCH MIMO 当前复核

复核日期：2026-05-07

复核范围：

- [15_mimo_phy_gap_analysis.md](15_mimo_phy_gap_analysis.md)
- `sionna_measurement_sim/phy/nr_pusch_observation.py`
- `sionna_measurement_sim/phy/nr_mimo_channel.py`
- `sionna_measurement_sim/phy/nr_channel_backend.py`
- NR PUSCH MIMO 相关配置、schema validator 和测试。

## 结论

当前系统已经不再停留在旧的 SISO 化 NR PUSCH 状态。以下核心问题已经解决：

- 正式 NR PUSCH backend 中不再使用 `h_tensor[0, 0, 0, 0, :]`。
- `/observation/cfr_est` 不再由单个 SISO LS estimate 直接 broadcast 到完整 MIMO shape。
- 4x4 SU-MIMO perfect CSI 和 estimated CSI 路径已有自生成测试。
- `StreamManagement`、`LinearDetector` / `KBestDetector` builder 已存在并被使用。
- `mimo_mode="mu_mimo"` 主链路、MU-MIMO integration test 已存在。
- `channel_backend` 已抽象为可插拔 backend，并提供 `apply_ofdm` 与 `cir_dataset_ofdm`。
- PUSCHReceiver 已设置 `return_tb_crc_status=True`，并写出 `num_block_errors` / `num_blocks`。
- 相关目标测试、全量测试和 ruff 均通过。

但仍不能标记为“15 号 MIMO 方案已完全无问题”。当前剩余问题主要是 **官方 backend 语义没有完全闭合、MU-MIMO backend 选择被绕过、BLER 契约没有被 schema/test 锁死**。

## 当前未完成问题

### 1. `cir_dataset_ofdm` backend 没有把 `OFDMChannel(return_channel=True)` 返回的 `h` 用作 perfect CSI / `cfr_est`

当前 `CIRDatasetOFDMChannelBackend.apply(...)` 中确实创建了：

```text
CIRDataset(...)
OFDMChannel(..., return_channel=True)
```

但 `OFDMChannel` 返回的 `_h` 被直接丢弃：

```text
y, _h = ofdm_ch(x, no)
return y
```

同时 `CIRDatasetOFDMChannelBackend.perfect_h(...)` 仍从预计算 CFR 构造：

```text
cfr_to_pusch_perfect_h(...)
```

影响：

- `channel_backend="cir_dataset_ofdm"` 时，接收信号 `y` 走了官方 `OFDMChannel`。
- 但 perfect CSI / `/observation/cfr_est` 仍来自预计算 CFR，不是来自官方 `OFDMChannel(return_channel=True)` 返回的 channel tensor。
- 这会让“官方 backend 已完整闭合”的说法不够准确。

修复要求：

1. 修改 backend API，使 `apply(...)` 能返回 `y` 和 `h`，或新增 `apply_with_h(...)`。
2. 在 `run_nr_pusch_observation(...)` / per-link processing 中，当 backend 是 `cir_dataset_ofdm` 且 `perfect_csi=True` 时，应把 `OFDMChannel` 返回的 `h` 传入 `PUSCHReceiver`，并由同一个 `h` 写回 `/observation/cfr_est`。
3. 增加测试，断言 `cir_dataset_ofdm` backend 的 `/observation/cfr_est` 来自 backend returned `h`，而不是预计算 CFR 快捷路径。

### 2. `CIRDatasetOFDMChannelBackend` 只取第一个天线对 delay

当前 generator 中：

```text
tau_slice = self._tau_ul[snap_idx, ul_tx_idx, ul_rx_idx, 0, 0, :]
```

也就是只取 `(rx_ant=0, tx_ant=0)` 的 path delay。

影响：

- 当前测试里的 delay 基本是 broadcast，所以这个问题不会暴露。
- 如果真实 RT CIR 中不同 antenna pair 的 delay 不完全相同，`OFDMChannel` 施加的信道会与项目 `/channel/truth/cfr` 或预计算 perfect CFR 不一致。

修复要求：

1. 先确认 Sionna `CIRDataset` 对 `tau` 的官方 shape 语义。若官方只支持 `[num_rx, num_tx, path]`，则必须在文档中明确：`cir_dataset_ofdm` backend 使用 link-level shared delay，而非 per-antenna delay。
2. 如果项目 HDF5 契约坚持 CIR delay 为 `[snapshot, tx, rx, rx_ant, tx_ant, path]`，需要定义从 per-antenna delay 到 Sionna shared delay 的降维策略，例如：

```text
tau_shared = tau[..., reference_rx_ant, reference_tx_ant, :]
```

或：

```text
tau_shared = median/mean over antenna pairs
```

3. 增加非 broadcast delay 的测试，验证 `cir_dataset_ofdm` backend 的行为被明确锁定。

### 3. MU-MIMO 分支会绕过配置选择的 channel backend

在 `_process_mu_mimo(...)` 中，代码先调用：

```text
y = backend.apply(...)
```

随后又无条件执行：

```text
apply_ch = ApplyOFDMChannel()
y = apply_ch(tx_signal, h_full, no)
```

影响：

- `mimo_mode="mu_mimo"` 时，即使配置 `channel_backend="cir_dataset_ofdm"`，最终 `y` 也会被 `ApplyOFDMChannel` 覆盖。
- 因此 MU-MIMO 路径中的 `channel_backend` 配置不是严格生效的。

修复要求：

1. 删除无条件覆盖逻辑。
2. 为 backend 增加 full-MIMO apply 接口，例如：

```python
apply_full(x, no, snap_idx, num_ofdm_symbols, resource_grid)
```

3. `ApplyOFDMChannelBackend.apply_full(...)` 使用 `cfr_to_full_mimo_h(...) + ApplyOFDMChannel`。
4. `CIRDatasetOFDMChannelBackend.apply_full(...)` 使用包含所有 `num_ul_tx / num_ul_rx` 的 `CIRDataset + OFDMChannel`，或在不支持时明确 `NotImplementedError`。
5. 新增测试：`mimo_mode="mu_mimo", channel_backend="cir_dataset_ofdm"` 时，不能被悄悄切回 `ApplyOFDMChannel`。

### 4. TB/CRC BLER 已初步实现，但 HDF5 schema 和测试还没有锁死语义

当前代码已经使用：

```text
return_tb_crc_status=True
```

并写入：

```text
/evaluation/num_block_errors
/evaluation/num_blocks
```

但 schema validator 的 observation required fields 只要求了：

```text
evaluation/num_blocks
```

没有同时要求：

```text
evaluation/num_block_errors
```

测试也主要检查 BLER finite，尚未强制：

```text
num_blocks > 0
0 <= num_block_errors <= num_blocks
bler == num_block_errors / num_blocks
```

修复要求：

1. 在 `schema_validator.py` 中把 `/evaluation/num_block_errors` 加入 observation required fields。
2. 增加 schema / integration test：

```text
/evaluation/num_blocks > 0
/evaluation/num_block_errors >= 0
/evaluation/num_block_errors <= /evaluation/num_blocks
/evaluation/bler == /evaluation/num_block_errors / /evaluation/num_blocks
```

3. 如果 SU-MIMO 和 MU-MIMO 的 aggregate BLER 语义不同，必须在 `docs/03_data_contract_hdf5.md` 中说明。

### 5. 文档仍有历史状态残留

`docs/13_tdd_reciprocity_nr_pusch_phy_plan.md` 和 `docs/15_mimo_phy_gap_analysis.md` 中仍有部分旧状态描述，例如：

- MIMO limited to first antenna pair。
- BER/BLER transport-block CRC pending。
- MU-MIMO future/pending。

这些描述已经和当前代码不完全一致。

修复要求：

1. 更新 13/15 文档的 Acceptance Status。
2. 明确当前真实状态：

```text
SU-MIMO: implemented and tested
estimated CSI: implemented for num_layers == num_antenna_ports; num_layers < num_antenna_ports rejected for physical cfr_est
MU-MIMO: implemented/tested for current 1 BS / 2 UE fixture, but backend selection still needs apply_full cleanup
cir_dataset_ofdm: implemented for per-link apply, but returned h is not yet used for cfr_est
TB/CRC BLER: implemented in code, schema/test contract needs stronger validation
```

## 本次核查命令

```bash
uv run pytest tests/unit/test_nr_mimo_channel_bridge.py tests/unit/test_nr_pusch_mimo_config.py tests/unit/test_nr_pusch_config.py tests/integration/test_nr_pusch_mimo_observation.py tests/integration/test_nr_pusch_mu_mimo_observation.py tests/schema/test_nr_pusch_schema.py tests/statistical/test_nr_pusch_mimo_metrics.py
uv run pytest
uv run ruff check .
git status --short --branch
```

结果：

```text
74 passed, 2 warnings
176 passed, 2 warnings
ruff: All checks passed
git status: clean
```

## 下一步优先级

1. 先修 `cir_dataset_ofdm` returned `h` 没有进入 perfect CSI / `cfr_est` 的问题。
2. 明确并测试 `CIRDataset` delay 语义，处理非 broadcast per-antenna delay。
3. 清理 MU-MIMO 分支绕过 backend 的逻辑，新增 `apply_full(...)` 或明确 `cir_dataset_ofdm` 不支持 MU-MIMO。
4. 加强 TB/CRC BLER schema 与测试契约。
5. 更新 13/15 文档，删除历史状态残留。

## 当前总体判断

当前实现已经具备可运行的 NR PUSCH MIMO 雏形，并且质量门是绿的。但要达到“与官方 Sionna link-level RT 示例语义接近”的要求，还需要把 `CIRDataset + OFDMChannel` backend 的 channel tensor 闭环、MU-MIMO backend 选择、BLER 契约和文档状态全部收齐。

## 执行指南

本轮建议继续采用小步补齐，不要推倒重写。优先顺序是：

```text
官方 backend h 闭环
→ CIRDataset delay 语义
→ MU-MIMO backend 选择
→ TB/CRC BLER 契约
→ 文档同步
```

每一步都应单独测试。不要把 FDD、PDSCH、HARQ、标注系统或更复杂 impairments 混入本轮。

### Step 0：建立基线

开始前运行：

```bash
git status --short --branch
uv run pytest tests/unit/test_nr_mimo_channel_bridge.py tests/unit/test_nr_pusch_mimo_config.py tests/unit/test_nr_pusch_config.py tests/integration/test_nr_pusch_mimo_observation.py tests/integration/test_nr_pusch_mu_mimo_observation.py tests/schema/test_nr_pusch_schema.py tests/statistical/test_nr_pusch_mimo_metrics.py
uv run ruff check .
```

预期当前基线：

```text
74 passed, 2 warnings
ruff: All checks passed
```

如果基线不通过，先修基线，不要继续扩展。

### Step 1：让 `cir_dataset_ofdm` returned `h` 进入 perfect CSI / `cfr_est` 闭环

目标：

- `channel_backend="cir_dataset_ofdm"` 时，`OFDMChannel(return_channel=True)` 返回的 `h` 必须被用于：
  - PUSCHReceiver perfect CSI 输入；
  - `/observation/cfr_est` 写回；
  - NMSE 计算。

修改文件：

```text
sionna_measurement_sim/phy/nr_channel_backend.py
sionna_measurement_sim/phy/nr_pusch_observation.py
tests/unit/test_nr_mimo_channel_bridge.py
tests/integration/test_nr_pusch_mimo_observation.py
```

建议实现：

1. 在 `nr_channel_backend.py` 增加返回 `h` 的接口：

```python
@dataclass(frozen=True)
class ChannelApplyResult:
    y: torch.Tensor
    h: torch.Tensor

def apply_with_h(..., resource_grid: Any = None) -> ChannelApplyResult:
    ...
```

2. `ApplyOFDMChannelBackend.apply_with_h(...)`：

```python
h = self.perfect_h(...)
y = self._apply(x, h, no)
return ChannelApplyResult(y=y, h=h)
```

3. `CIRDatasetOFDMChannelBackend.apply_with_h(...)`：

```python
y, h = ofdm_ch(x, no)
return ChannelApplyResult(y=y, h=h)
```

4. 保留原 `apply(...)` 作为兼容 wrapper：

```python
return self.apply_with_h(...).y
```

5. 在 `_process_one_pusch_link(...)` 中替换：

```python
h_perfect = backend.perfect_h(...)
y = backend.apply(...)
```

为：

```python
channel_result = backend.apply_with_h(...)
y = channel_result.y
h_perfect = channel_result.h
```

6. perfect CSI 分支继续：

```python
rx_bits, tb_crc_status = rx(y, no, h_perfect)
cfr_est_slice = pusch_h_to_cfr_est(h_perfect)
```

测试要求：

- 新增/修改 unit test，确认 `CIRDatasetOFDMChannelBackend.apply_with_h(...)` 返回的 `h` shape 正确。
- 新增 integration test：

```text
channel_backend="cir_dataset_ofdm"
perfect_csi=True
4x4 SU-MIMO
```

断言：

```text
/observation/cfr_est.shape[1:] == /channel/truth/cfr.shape
/observation/estimation_success 全 true
median(/evaluation/nmse_db) finite
```

如果数值因为官方 `OFDMChannel` 与预计算 CFR 的 delay 语义存在差异，不能硬写 `allclose`，应先完成 Step 2。

### Step 2：明确并测试 `CIRDataset` delay 语义

目标：

- 解决 `tau_slice = tau[..., 0, 0, :]` 的隐含假设。
- 明确 `CIRDatasetOFDMChannelBackend` 是否支持 per-antenna delay，或只支持 link-level shared delay。

修改文件：

```text
sionna_measurement_sim/phy/nr_channel_backend.py
tests/unit/test_nr_mimo_channel_bridge.py
docs/15_mimo_phy_gap_analysis.md
docs/03_data_contract_hdf5.md
```

建议先做保守实现：

1. 在 backend build 阶段检测 per-antenna delay 是否一致：

```python
tau_ref = tau_ul[..., 0, 0, :]
if not np.allclose(tau_ul, tau_ref[..., None, None, :], rtol=..., atol=...):
    raise NotImplementedError(
        "CIRDatasetOFDMChannelBackend currently requires antenna-pair shared delays"
    )
```

注意实际广播 shape 要写对，不要用易错隐式广播。

2. 如果 delay 一致，继续使用 reference pair 作为 shared delay。

3. 在文档中明确：

```text
cir_dataset_ofdm backend currently requires per-link shared path delays.
Project HDF5 may store per-antenna delays, but this backend rejects non-shared delays until a documented reduction policy is chosen.
```

测试要求：

- `test_cir_dataset_backend_accepts_shared_delays`
- `test_cir_dataset_backend_rejects_non_shared_antenna_delays`

非 shared delay fixture 要显式构造：

```python
delays[..., 1, 0, :] += 1e-9
```

验收：

- 不允许继续静默取 `[0,0]`。
- 非 broadcast delay 必须 fail fast，除非已经实现并文档化降维策略。

### Step 3：修复 MU-MIMO 绕过 backend 的问题

目标：

- `mimo_mode="mu_mimo"` 时，`channel_backend` 配置必须真实生效。
- 不允许先调用 `backend.apply(...)` 再无条件用 `ApplyOFDMChannel` 覆盖结果。

修改文件：

```text
sionna_measurement_sim/phy/nr_channel_backend.py
sionna_measurement_sim/phy/nr_pusch_observation.py
tests/integration/test_nr_pusch_mu_mimo_observation.py
```

建议实现：

1. 给 backend 增加 full MIMO 接口：

```python
def perfect_h_full(snap_idx: int, num_ofdm_symbols: int) -> torch.Tensor:
    ...

def apply_full_with_h(
    x: torch.Tensor,
    no: torch.Tensor,
    *,
    snap_idx: int,
    num_ofdm_symbols: int,
    resource_grid: Any = None,
) -> ChannelApplyResult:
    ...
```

2. `ApplyOFDMChannelBackend.apply_full_with_h(...)`：

```python
h = cfr_to_full_mimo_h(...)
y = ApplyOFDMChannel()(x, h, no)
return ChannelApplyResult(y=y, h=h)
```

3. `CIRDatasetOFDMChannelBackend.apply_full_with_h(...)` 二选一：

优先实现：

```text
构造包含全部 num_ul_rx / num_ul_tx 的 CIRDataset + OFDMChannel
```

如果本轮无法保证 shape 正确，则明确：

```python
raise NotImplementedError(
    "cir_dataset_ofdm backend does not yet support mu_mimo apply_full"
)
```

4. `_process_mu_mimo(...)` 中删除：

```python
apply_ch = ApplyOFDMChannel()
y = apply_ch(tx_signal, h_full, no)
```

改为：

```python
channel_result = backend.apply_full_with_h(...)
y = channel_result.y
h_full = channel_result.h
```

测试要求：

- `mimo_mode="mu_mimo", channel_backend="apply_ofdm"` 继续通过。
- `mimo_mode="mu_mimo", channel_backend="cir_dataset_ofdm"`：
  - 如果已实现 full backend，则必须自生成 HDF5 并通过；
  - 如果未实现，则必须抛 `NotImplementedError`，测试不能允许悄悄回退到 `ApplyOFDMChannel`。

验收：

- MU-MIMO 分支中不再出现无条件新建 `ApplyOFDMChannel()` 覆盖 backend 输出。
- backend 选择行为被测试锁定。

### Step 4：加强 TB/CRC BLER 契约

目标：

- 让 HDF5 schema 和测试明确 `bler` 与 TB CRC counters 的关系。

修改文件：

```text
sionna_measurement_sim/io/schema_validator.py
tests/schema/test_nr_pusch_schema.py
tests/integration/test_nr_pusch_mimo_observation.py
tests/integration/test_nr_pusch_mu_mimo_observation.py
docs/03_data_contract_hdf5.md
```

建议实现：

1. 在 observation required fields 中加入：

```text
evaluation/num_block_errors
```

2. 在 validator 中增加：

```python
num_blocks = int(h5["evaluation/num_blocks"][()])
num_block_errors = int(h5["evaluation/num_block_errors"][()])
bler = float(h5["evaluation/bler"][()])

if num_blocks <= 0:
    raise SchemaValidationError(...)
if num_block_errors < 0 or num_block_errors > num_blocks:
    raise SchemaValidationError(...)
if not np.isclose(bler, num_block_errors / num_blocks, atol=...):
    raise SchemaValidationError(...)
```

3. 更新 tests：

```text
test_bler_matches_tb_crc_counters
test_num_block_errors_required
test_invalid_block_error_count_fails
```

注意：

- 如果当前 MU-MIMO aggregate BLER 是 joint BLER 或 averaged per-link BLER，必须先统一代码和契约。
- 当前最直接的契约建议是：

```text
/evaluation/bler = /evaluation/num_block_errors / /evaluation/num_blocks
```

如果不采用这个公式，必须在 `docs/03_data_contract_hdf5.md` 明确说明 aggregate 规则。

### Step 5：同步 13/15 文档

目标：

- 删除历史状态残留，避免后续 agent 依据旧结论重复工作。

修改文件：

```text
docs/13_tdd_reciprocity_nr_pusch_phy_plan.md
docs/15_mimo_phy_gap_analysis.md
docs/phase_progress.md
```

必须更新的状态：

```text
SU-MIMO: implemented and tested
estimated CSI: implemented for num_layers == num_antenna_ports
estimated CSI with num_layers < num_antenna_ports: rejected for physical cfr_est
MU-MIMO: implemented/tested for current 1 BS / 2 UE fixture
cir_dataset_ofdm: per-link backend implemented; returned h / full-MIMO apply semantics still being closed
TB/CRC BLER: code path implemented; schema/test contract being strengthened
```

不要删除 FDD TODO；FDD 仍然是 backlog。

### Step 6：质量门

完成上述修改后至少运行：

```bash
uv run pytest tests/unit/test_nr_mimo_channel_bridge.py tests/unit/test_nr_pusch_mimo_config.py tests/unit/test_nr_pusch_config.py
uv run pytest tests/integration/test_nr_pusch_mimo_observation.py tests/integration/test_nr_pusch_mu_mimo_observation.py
uv run pytest tests/schema/test_nr_pusch_schema.py tests/statistical/test_nr_pusch_mimo_metrics.py
uv run pytest
uv run ruff check .
git status --short --branch
```

验收：

```text
所有目标测试通过
全量 pytest 通过
ruff 通过
review.md 中只保留真实未完成问题
phase_progress.md 追加本轮记录
```

### 建议提交切分

建议拆成 4 个提交：

1. `fix: use cir-dataset returned channel for nr pusch csi`
2. `fix: enforce cir-dataset delay semantics`
3. `fix: respect channel backend in mu-mimo path`
4. `test: lock nr pusch tb-crc bler contract`

文档同步可随对应提交一起做，或单独提交：

```text
docs: update nr pusch mimo acceptance status
```
