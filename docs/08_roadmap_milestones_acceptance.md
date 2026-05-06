# 08. 路线图、里程碑与验收标准

本文定义开发阶段、里程碑和每阶段验收标准。测试细节见 [09_testing_and_quality_gates.md](09_testing_and_quality_gates.md)，工程提交规范见 [07_project_layout_uv_git_workflow.md](07_project_layout_uv_git_workflow.md)，全局约束见 [00_global_constraints_and_official_references.md](00_global_constraints_and_official_references.md)。

## 0. 阶段通用验收规则

每个 Phase 结束时必须完成：

```bash
uv run ruff check .
uv run pytest
git status --short
```

通用通过标准：

- 所有本阶段要求的测试通过。
- `git status --short` 中没有意外大文件、输出文件或临时文件。
- 文档与实现一致。
- 若新增或修改 HDF5 字段，已更新 [03_data_contract_hdf5.md](03_data_contract_hdf5.md)。
- 若新增或修改 Sionna RT API 适配，已更新 [04_sionna_rt_adapter_and_path_data.md](04_sionna_rt_adapter_and_path_data.md)。
- 通过验收后必须提交 git。

推荐每阶段生成一个简短验收记录：

```text
artifacts/phase_reports/phase_N_acceptance.md
```

该记录不应包含大型数据，只记录命令、结果、输出路径和关键指标。

## Phase 0: 项目骨架

目标：

- 建立新仓库结构。
- 使用 `uv` 初始化环境。
- 建立基础 package、CLI、配置 loader、测试框架。
- 确认本文档群位于 `SionnaMeasurementSim/docs/`。

必须产物：

```text
pyproject.toml
uv.lock
.python-version
.gitignore
sionna_measurement_sim/
tests/
docs/
data/scenes/test/
```

验收命令：

```bash
uv sync
uv run pytest
uv run python -m sionna_measurement_sim.app.cli --help
git status --short
```

通过标准：

- `uv sync` 成功并生成 `.venv/`。
- `uv.lock` 存在且已纳入 git。
- `.venv/` 被 `.gitignore` 忽略。
- CLI help 返回退出码 0。
- `data/scenes/test/` 存在，若尚无真实场景，必须有 README 占位说明。
- 首次 git commit 完成。

## Phase 1: HDF5 Schema 与 Domain Model

目标：

- 实现 domain dataclasses。
- 实现 HDF5 writer/reader skeleton。
- 写入无 Sionna 依赖的最小合法 HDF5。

必须产物：

```text
sionna_measurement_sim/domain/
sionna_measurement_sim/io/hdf5_writer.py
sionna_measurement_sim/io/hdf5_reader.py
tests/schema/
outputs/.../results.h5
```

验收命令：

```bash
uv run pytest tests/unit tests/schema
```

HDF5 必须包含：

```text
/meta/schema_version
/meta/contract_name
/meta/index_order
/meta/unit_convention
/meta/config_snapshot
/topology/tx_positions_m
/topology/rx_positions_m
/antenna/tx_polarization
/antenna/rx_polarization
/frequency/frequencies_hz
```

通过标准：

- readback 能验证 dtype、shape、单位、index order。
- 不 import Sionna 也能通过全部 Phase 1 测试。
- schema test 明确拒绝缺失 `/meta/schema_version` 的文件。
- git commit。

## Phase 2: Sionna 2.x RT Truth 最小闭环

目标：

- 接入 Sionna 2.x RT。
- 加载 `data/scenes/test/` 场景。
- 注册 TX/RX 和天线。
- 运行 PathSolver。
- 生成 `/channel/truth/cfr`。

官方参考：

- https://nvlabs.github.io/sionna/installation.html
- https://nvlabs.github.io/sionna/rt/tutorials/Introduction.html
- https://nvlabs.github.io/sionna/rt/api/paths.html

验收命令：

```bash
uv run pytest tests/adapter tests/integration -k "rt_truth"
```

HDF5 必须包含：

```text
/runtime/sionna_version
/runtime/sionna_rt_version
/runtime/mitsuba_version
/runtime/drjit_version
/runtime/torch_version
/channel/truth/cfr
/channel/truth/path_power_db
/channel/truth/has_geometric_signal
```

通过标准：

- `H_true` shape 精确为 `[tx, rx, rx_ant, tx_ant, subcarrier]`。
- `frequencies_hz.shape[-1] == H_true.shape[-1]`。
- `H_true` dtype 为 complex64 或文档明确允许的 complex dtype。
- 至少一个链路有有限数值，不能全 NaN。
- manifest 记录 scene file、config snapshot、software versions。
- HDF5 readback 后 `H_true` shape 和 dtype 不变。
- git commit。

## Phase 3: Path Adapter 与路径级数据

目标：

- 提取 `valid`、`a`、`tau`、`doppler`、AoA/AoD。
- 提取 `vertices`、`interactions`、`objects`、`primitives`。
- 写入 `/paths/samples`。
- debug 配置下写入 `/paths/full`。

官方参考：

- https://nvlabs.github.io/sionna/rt/api/paths.html
- https://nvlabs.github.io/sionna/rt/developer/dev_understanding_paths.html
- https://nvlabs.github.io/sionna/rt/tutorials/Mobility.html

验收命令：

```bash
uv run pytest tests/adapter tests/schema tests/integration -k "path"
```

`/paths/samples` 必须包含：

```text
sampled_link_indices
sampled_path_indices
vertices_m
interaction_type
object_id
primitive_id
doppler_hz
tau_s
path_gain_db
path_type
```

通过标准：

- 至少一条 valid path 被解析。
- 对每条 sample path，`vertex_count >= interaction_count + 2`，其中 2 是 TX/RX 端点。
- `interaction_type`、`object_id`、`primitive_id` 与 interaction depth 对齐。
- 对 NLoS path，至少一个中间 `vertices_m` 为有限坐标。
- 静态场景允许 `doppler_hz == 0`，但字段必须存在且为有限值。
- 如果测试场景没有 NLoS，必须额外提供一个反射最小场景 fixture。
- path visualization smoke test 生成非空图片。
- git commit。

## Phase 4: 最小 PHY Observation

目标：

- 实现 custom OFDM pilot。
- 实现 AWGN。
- 实现 LS channel estimator。
- 输出 `/observation/cfr_est`。
- 输出 `/evaluation/nmse_db`。

验收命令：

```bash
uv run pytest tests/unit tests/integration tests/statistical -k "awgn or observation or nmse"
```

HDF5 必须包含：

```text
/waveform/standard
/waveform/fft_size
/waveform/pilot_indices
/waveform/pilot_symbols
/receiver/estimator_type
/observation/cfr_est
/observation/valid_mask
/observation/detection_success
/observation/estimation_success
/observation/snr_db
/evaluation/nmse_db
```

通过标准：

- `cfr_est` shape 精确为 `[snapshot, tx, rx, rx_ant, tx_ant, subcarrier]`。
- `cfr_est.shape[1:] == H_true.shape`。
- AWGN-only 下，至少比较两个 SNR 点；高 SNR 的 median NMSE 必须低于低 SNR。
- 高 SNR debug case 中，median NMSE 应小于配置阈值，默认阈值建议 `-20 dB`。
- `valid_mask`、`detection_success`、`estimation_success` shape 为 `[snapshot, tx, rx]`。
- git commit。

## Phase 5: 基础 Impairments

目标：

- CFO。
- SFO。
- phase offset。
- timing offset。
- AGC/ADC 简化模型。

验收命令：

```bash
uv run pytest tests/unit tests/statistical -k "cfo or sfo or clipping or impairment"
```

HDF5 必须包含：

```text
/impairments/model_version
/impairments/random_seed
/observation/cfo_hz
/observation/sfo_ppm
/observation/timing_offset_samples
/observation/phase_offset_rad
/observation/agc_gain_db
/observation/clipping_flag
```

通过标准：

- 固定 seed 下，impairment 采样可复现。
- CFO 增大时，相位漂移指标增大。
- 降低 clipping threshold 时，`clipping_flag` 比例不下降。
- 构造失败场景时，至少一个样本 `valid_mask=false`，并记录 failure reason 或 failure code。
- impairment 配置和采样值都写入 HDF5。
- git commit。

## Phase 6: Motion 与 Doppler

目标：

- 保存 TX/RX velocity 和 orientation。
- 支持 `doppler_synthetic` time evolution。
- 支持多 snapshot。

官方参考：

- https://nvlabs.github.io/sionna/rt/tutorials/Mobility.html
- https://nvlabs.github.io/sionna/rt/api/paths.html

验收命令：

```bash
uv run pytest tests/adapter tests/integration tests/statistical -k "doppler or motion"
```

HDF5 必须包含：

```text
/devices/tx_velocity_mps
/devices/rx_velocity_mps
/devices/tx_orientation_rad
/devices/rx_orientation_rad
/motion/snapshot_id
/motion/timestamp_s
/paths/samples/doppler_hz
```

通过标准：

- 静态配置下，所有 path-level Doppler 接近 0，容差由测试配置定义。
- 非零速度配置下，至少部分 valid path 的 Doppler 非 0。
- 多 time-step CFR shape 与 `num_time_steps` 一致。
- `timestamp_s` 单调递增。
- Delay-Doppler 基础统计或图可生成。
- git commit。

## Phase 7: Calibration 与 Diagnostics

目标：

- 定义 calibration profile。
- 接入小规模实测统计或占位接口。
- 输出诊断报告。

验收命令：

```bash
uv run pytest tests/unit tests/integration -k "calibration or diagnostics"
```

HDF5 必须包含：

```text
/calibration/profile_id
/calibration/fitted_parameters
/calibration/validation_metrics
/evaluation/nmse_db
/evaluation/detection_rate
/evaluation/estimation_failure_rate
```

通过标准：

- calibration profile 可被配置引用。
- 诊断报告至少包含 NMSE、SNR、phase drift、failure rate。
- 若无实测数据，必须明确 `profile_id="none"` 或 `profile_id="synthetic_default"`。
- profile 和诊断字段 readback 通过。
- git commit。

## Phase 8: 批处理与性能

目标：

- TX/RX 分批。
- HDF5 chunked append。
- 显存清理。
- 失败 batch 记录。

验收命令：

```bash
uv run pytest tests/integration -k "batch"
```

manifest 必须记录：

```text
batching.enabled
batching.total_batches
batching.completed_batches
batching.failed_batches
```

通过标准：

- 大于 debug 规模的数据可分批运行。
- 每批写入的 HDF5 slice 位置正确。
- 中途失败可在 manifest 中定位 batch。
- 失败 batch 不会伪装成全局成功。
- `outputs/` 不进入 git。
- git commit。

## 阶段推进规则

- 不通过当前阶段验收，不进入下一阶段。
- schema 变更必须同步更新 [03_data_contract_hdf5.md](03_data_contract_hdf5.md)。
- Sionna API 适配变更必须同步更新 [04_sionna_rt_adapter_and_path_data.md](04_sionna_rt_adapter_and_path_data.md)。
- 每阶段结束必须测试并提交。
