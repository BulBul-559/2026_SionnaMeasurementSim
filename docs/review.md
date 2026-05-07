# 13 与 14 整体 Review

Review 日期：2026-05-07

Review 范围：

- [13_tdd_reciprocity_nr_pusch_phy_plan.md](13_tdd_reciprocity_nr_pusch_phy_plan.md)
- [14_rt_hardening_before_nr_pusch.md](14_rt_hardening_before_nr_pusch.md)
- 当前代码和测试中与 13/14 直接相关的实现状态。

## 总体结论

13 和 14 的方向是对的：先把 Sionna RT truth、CIR、shape contract、天线配置、4x4 MIMO 和 config snapshot 做硬，再在这个基础上做 TDD 互易性和 NR PUSCH PHY。

但当前文档还不能作为干净的执行入口。主要问题是：

- 14 是 13 的前置条件，但编号在 13 后面，容易让执行 agent 误以为先做 13。
- 13 和 14 末尾都追加过历史 review，其中不少判断已经过期。
- 14 的实现状态已经明显推进，但文档仍像“待实现计划”，没有更新为“已完成 / 待完成 / 验收命令 / 已知风险”的状态文档。
- 13 的代码也已经从 skeleton 前进到主链路可选 `nr_pusch`，并尝试 `PUSCHReceiver`，但 PHY 物理闭环仍有明显简化，不能直接宣称达到官方示例级别。
- 03 HDF5 契约和 06 配置文档仍没有完全同步 13 要求，尤其是 NR PUSCH waveform、receiver、NMSE 语义和 config 单位命名。

## 14 的 Review

当前判断：14 大部分核心任务已经落地，但 14 文档本身需要刷新。

已经看到的积极进展：

- `tests/adapter/test_rt_shape_contracts.py` 存在。
- `tests/adapter/test_rt_cir_adapter.py` 存在。
- `tests/schema/test_rt_cir_schema.py` 存在。
- `tests/integration/test_rt_mimo_4x4_pipeline.py` 存在。
- `to_project_cir(...)` 已经进入 `sionna_measurement_sim/adapters/sionna_rt/shape_contracts.py`。
- `AntennaSpec` 已经拆出 `tx_orientation_mode/rad` 和 `rx_orientation_mode/rad`。
- HDF5 writer 已经分别写 `tx_orientation_mode` 和 `rx_orientation_mode`。
- 当前 4x4 MIMO、CIR、shape contract 相关测试通过。

14 文档里的过期内容：

- 14 末尾历史 review 仍说 RX 朝向没有拆分、4x4/CIR/shape 测试缺失、`to_project_cir` 缺失。这些已经不符合当前代码状态。
- 14 仍以“下一步改造计划”的口吻写成，但许多要求已经实现。继续保留这种写法会误导后续 agent 重复实现或错误修复。

14 仍需明确的点：

- RX 是否支持 `look_at_first_peer` / `look_at_centroid` 需要写清楚。当前 RT 代码看起来 TX 支持 look-at 模式，RX 仍主要按配置 orientation 写入，不应让文档暗示 TX/RX look-at 能力完全对称。
- Sionna `PlanarArray` 的 `pattern` / `polarization` 支持范围需要写成明确白名单，并要求配置校验 fail fast。
- `merge_shapes=true` 时 object/primitive 可追溯性下降的 warning 是否已落入 manifest 或 diagnostics，需要文档和测试明确。
- 14 应该增加一张当前验收表，列出每个要求的状态和对应测试。

建议对 14 的处理：

1. 删除或改名 14 末尾旧 `## review`，避免过期结论继续传播。
2. 把 14 改成“RT hardening acceptance status”，而不是纯计划。
3. 明确记录通过命令：

```bash
uv run pytest tests/adapter/test_rt_shape_contracts.py tests/adapter/test_rt_cir_adapter.py tests/schema/test_rt_cir_schema.py tests/integration/test_rt_mimo_4x4_pipeline.py
uv run ruff check .
```

## 13 的 Review

当前判断：13 已经有可运行雏形，但没有达到完整验收。

已经看到的积极进展：

- `LinkConfig` 已有 TDD/uplink/bs_to_ue/reciprocity 配置，并对非 TDD、非 uplink 做限制。
- `apply_tdd_reciprocity(...)` 和 `apply_tdd_reciprocity_cir(...)` 已有基础单测。
- CLI 已有 `run-full --phy-standard nr_pusch`。
- `run_rt_truth_pipeline(...)` 会在 `phy_standard == "nr_pusch"` 时进入 NR PUSCH 分支。
- `nr_pusch_observation.py` 会构建 Sionna `PUSCHConfig`、`PUSCHTransmitter`，并尝试构建 `PUSCHReceiver` / `PUSCHLSChannelEstimator` / `LinearDetector`。
- HDF5 writer 已经出现 `receiver_type` 写入。
- `tests/integration/test_nr_pusch_observation.py` 已存在。

13 的主要阻塞：

1. NR PUSCH 的 MIMO 信道使用仍过度简化。当前实现会把 CFR antenna 维度平均，然后取第一个 TX/RX pair 走一个近似频域链路。这不能证明 4x4 MIMO PUSCH receiver 语义正确，也不能替代官方示例中的完整 channel/receiver 路径。

2. PUSCH receiver 失败会被吞掉。当前 receiver try/except 失败后回退到 `ber=0.0`、`bler=0.0`、bit count 为 0。这是验收风险：真实 receiver 坏了也可能显示完美链路。13 验收时必须 fail fast，或者至少把 detection/estimation 标为失败并让测试失败。

3. BER/BLER 仍不够可信。虽然字段已存在，但如果 fallback 发生，BER/BLER 仍是占位式结果。13 要求的是通过真实接收链路得到 bit/block error 统计。

4. perfect CSI 和 imperfect CSI 两条路径没有形成强验收。13 要求 perfect CSI BER/BLER 不差于 estimated CSI，但当前没有看到对应 statistical test。

5. 高/低 Eb/N0 的统计验收不足。`tests/statistical/test_nr_pusch_link_metrics.py` 当前未发现，现有 integration test 主要检查 HDF5 字段存在和有限值。

6. `tests/integration/test_nr_pusch_observation.py` 依赖已有 `outputs/e2e_nr_pusch_rx/results.h5`，如果文件不存在会 skip。这种测试不能作为强制端到端质量门。验收测试应该在测试内生成输出或使用 fixture 生成输出。

7. `/evaluation/nmse_db` 语义仍冲突。13 要求 `/evaluation/nmse_db` 表示 `H_obs` vs clean `H_true`，但 `03_data_contract_hdf5.md` 仍写 `nmse_db` 是 AWGN isolation，`nmse_db_total` 是 clean truth total distortion。这需要统一，否则后续诊断和训练标签会混乱。

8. `/waveform` 和 `/receiver` 契约仍不完整。13 要求记录 PRB、slot、DMRS、layers、antenna ports、MCS、coderate、modulation，以及 receiver_type/channel_estimator/mimo_detector/input_domain。当前 03 文档仍偏 custom OFDM。

9. 配置单位命名不一致。13 文档写 `subcarrier_spacing_hz`，当前代码和配置使用 `subcarrier_spacing_khz`。建议配置层保留 `subcarrier_spacing_khz`，HDF5 派生写 `subcarrier_spacing_hz`，并在 06/03 中明确。

## 13 与 14 的交叉问题

### 执行顺序

必须写成：

```text
先验收 14 RT hardening
再进入 13 TDD NR PUSCH
```

不能因为文件编号是 13、14 就按数字顺序执行。

### RT truth 与 PHY uplink 方向

13 说 `/channel/truth/cfr` 保持 RT trace direction，即 BS -> UE。PUSCH backend 内部使用 TDD reciprocity 得到 UE -> BS uplink。这里必须在 HDF5 中明确：

- `/channel/truth/cfr` 是 RT 方向。
- `/channel/truth/cir_*` 是 RT 方向。
- `/observation/cfr_est` 对 `standard="nr_pusch"` 到底是 uplink 方向还是和 truth 同方向，需要明确写入 link metadata。

否则后续训练或评估会把 TX/RX 角色弄反。

### Reciprocity 的物理假设

13 需要补一句硬约束：

```text
当前 TDD reciprocity 只在同频、窄时间间隔、RF chain 已校准、antenna phase reference 一致的假设下成立。
```

并且要明确当前 transform 是 transpose、conjugate transpose，还是带 calibration matrix 的映射。不要只写“互易性转换”。

### MIMO 维度桥接

14 定义 RT 维度：

```text
[tx, rx, rx_ant, tx_ant, subcarrier]
```

13 定义 PUSCH 维度：

```text
num_antenna_ports
num_layers
num_streams
receiver antenna
```

两者之间缺少一张桥接表。必须明确：

```text
RT tx_ant/rx_ant
  -> UE/BS physical antennas
  -> PUSCH antenna ports
  -> PUSCH layers/streams
  -> detector input shape
```

否则 4x4 RT channel 很容易被实现成“测试看起来是 4x4，PHY 实际只用第一个 link 或平均后的 SISO”。

## 建议下一步

优先级 1：清理文档状态。

- 删除或重命名 13/14 末尾的历史 `## review`。
- 13/14 不要继续把旧 review 当正文。统一使用本文件作为当前 review。
- README 或 phase_progress 中明确“14 先于 13”。

优先级 2：把 14 收口成已验收状态。

- 更新 14 中过期描述。
- 明确 RX look-at 能力边界。
- 明确 pattern/polarization 支持白名单。
- 记录 4x4 MIMO、CIR、shape contract 的验收命令和结果。

优先级 3：修 13 的 contract。

- 更新 `docs/03_data_contract_hdf5.md`。
- 更新 `docs/06_config_and_experiment_schema.md`。
- 更新 schema validator 和 schema tests。
- 统一 `nmse_db` / `nmse_awgn_db` / `nmse_db_total` 命名。
- 写入 NR PUSCH waveform 和 receiver 必填字段。

优先级 4：修 13 的 PHY 真实性。

- 去掉 PUSCHReceiver 静默 fallback。
- 不允许平均 antenna 维度后只取第一个 TX/RX pair 作为最终验收路径。
- 使用能保留 MIMO 语义的 Sionna channel / receiver 路径。
- BER/BLER 必须来自真实 bit/block 对比。
- 增加 high/low Eb/N0、perfect/imperfect CSI 的统计测试。
- 集成测试必须自己生成输出，不能依赖已有 outputs 后 skip。

## 本次运行命令

```bash
uv run pytest tests/adapter/test_rt_shape_contracts.py tests/adapter/test_rt_cir_adapter.py tests/schema/test_rt_cir_schema.py tests/integration/test_rt_mimo_4x4_pipeline.py tests/unit/test_reciprocity.py tests/unit/test_nr_pusch_config.py tests/integration/test_nr_pusch_observation.py
uv run ruff check .
```

结果：

```text
43 passed, 1 warning
ruff: All checks passed
```

说明：当前 RT hardening 和 NR PUSCH 雏形测试是绿的，这是好信号；但 13 仍缺少强制端到端生成、统计验收和完整 MIMO/PUSCH receiver 语义，所以不能仅凭这些测试宣称 13 完成。
