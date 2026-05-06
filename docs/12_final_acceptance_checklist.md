# 12. 最终验收清单

本文是 `SionnaMeasurementSim` 开发完成后的最终总验收清单。它不是替代 [08_roadmap_milestones_acceptance.md](08_roadmap_milestones_acceptance.md)，而是在所有 Phase 完成后用于逐项确认系统是否真正达到交付标准。

验收时应逐项勾选。任何必选项未通过，都不能认为系统完成。

## 1. 项目与环境

- [ ] git 仓库根目录是 `SionnaMeasurementSim/`。
- [ ] Python 包目录是 `SionnaMeasurementSim/sionna_measurement_sim/`。
- [ ] `pyproject.toml` 存在。
- [ ] `uv.lock` 存在并已纳入 git。
- [ ] `.python-version` 存在。
- [ ] `.venv/` 被 `.gitignore` 忽略。
- [ ] `outputs/` 被 `.gitignore` 忽略。
- [ ] 大型 HDF5、NumPy、PyTorch 权重、实测数据默认不进入 git。
- [ ] `uv sync` 成功。
- [ ] `uv run python -m sionna_measurement_sim.app.cli --help` 成功。
- [ ] `uv run ruff check .` 成功。
- [ ] `uv run pytest` 成功。

## 2. 文档

- [ ] `docs/00_global_constraints_and_official_references.md` 存在。
- [ ] `docs/03_data_contract_hdf5.md` 存在。
- [ ] `docs/08_roadmap_milestones_acceptance.md` 存在。
- [ ] `docs/09_testing_and_quality_gates.md` 存在。
- [ ] 当前实现与 HDF5 数据契约一致。
- [ ] 当前实现与 Phase 验收文档一致。
- [ ] 若 Sionna API 适配有变更，`docs/04_sionna_rt_adapter_and_path_data.md` 已同步更新。
- [ ] 若 schema 有变更，`docs/03_data_contract_hdf5.md` 和 schema tests 已同步更新。

## 3. 目录结构

- [ ] `sionna_measurement_sim/app/` 存在。
- [ ] `sionna_measurement_sim/domain/` 存在。
- [ ] `sionna_measurement_sim/io/` 存在。
- [ ] `sionna_measurement_sim/adapters/sionna_rt/` 存在。
- [ ] `sionna_measurement_sim/rt/` 存在。
- [ ] `sionna_measurement_sim/phy/` 存在。
- [ ] `sionna_measurement_sim/impairments/` 存在。
- [ ] `sionna_measurement_sim/analysis/` 存在。
- [ ] `sionna_measurement_sim/preflight/` 存在。
- [ ] `tests/unit/` 存在。
- [ ] `tests/schema/` 存在。
- [ ] `tests/adapter/` 存在。
- [ ] `tests/integration/` 存在。
- [ ] `tests/statistical/` 存在。
- [ ] `data/scenes/test/` 存在。
- [ ] `old/SimpleSionna/` 仅作为参考存在，或明确记录为何暂缺。

## 4. 架构边界

- [ ] 业务层不直接 import Sionna。
- [ ] HDF5 writer 不直接读取 Sionna `Paths`。
- [ ] Sionna RT 相关 import 集中在 `adapters/sionna_rt/`。
- [ ] Path adapter 输出内部稳定数据结构。
- [ ] Domain dataclasses 不保存 Sionna 原生对象作为主数据。
- [ ] 新系统没有从 `old/SimpleSionna/` import 代码。

## 5. 配置与 Preflight

- [ ] 配置 schema 会校验必填字段。
- [ ] 配置 schema 会校验单位和基本范围。
- [ ] 配置 schema 会校验 `phy.enabled=true` 时 observation 所需字段。
- [ ] 配置 schema 会校验 `motion` 和 `impairments` 相关字段。
- [ ] 配置校验失败会在 RT/PHY 开始前停止。
- [ ] Preflight 会检查 GPU/backend。
- [ ] Preflight 会记录 Sionna、Sionna RT、Torch、Mitsuba、Dr.Jit 版本。
- [ ] 运行配置快照写入 HDF5 `/meta/config_snapshot`。
- [ ] 运行配置快照写入 manifest。

## 6. HDF5 基础契约

生成一个完整测试输出：

```bash
uv run python -m sionna_measurement_sim.app.cli --config config/defaults/measurement_mvp.yaml --label data/scenes/test/test.json
```

然后检查 `outputs/.../results.h5`：

- [ ] `/meta/schema_version` 存在。
- [ ] `/meta/contract_name` 存在。
- [ ] `/meta/index_order` 存在。
- [ ] `/meta/unit_convention` 存在。
- [ ] `/meta/config_snapshot` 存在。
- [ ] `/input/label_file` 存在。
- [ ] `/topology/tx_positions_m` 存在。
- [ ] `/topology/rx_positions_m` 存在。
- [ ] `/devices/tx_velocity_mps` 存在。
- [ ] `/devices/rx_velocity_mps` 存在。
- [ ] `/devices/tx_orientation_rad` 存在。
- [ ] `/devices/rx_orientation_rad` 存在。
- [ ] `/antenna/tx_polarization` 存在。
- [ ] `/antenna/rx_polarization` 存在。
- [ ] `/frequency/frequencies_hz` 存在。
- [ ] `/runtime/sionna_version` 存在。
- [ ] `/runtime/torch_version` 存在。
- [ ] `/runtime/mitsuba_version` 存在。
- [ ] `/runtime/drjit_version` 存在。

## 7. Truth Channel

- [ ] `/channel/truth/cfr` 存在。
- [ ] 新 writer 没有把 truth 主数据写为 `/channel/cfr`。
- [ ] `truth_cfr.ndim == 5`。
- [ ] `truth_cfr.shape == [tx, rx, rx_ant, tx_ant, subcarrier]`。
- [ ] `len(/frequency/frequencies_hz) == truth_cfr.shape[-1]`。
- [ ] `truth_cfr` 至少包含一个 finite value。
- [ ] `/channel/truth/path_power_db` 存在。
- [ ] `/channel/truth/has_geometric_signal` 存在。
- [ ] `/channel/truth/geometric_path_count` 存在。
- [ ] `/channel/truth/los_exists` 存在。
- [ ] `/channel/truth/nlos_exists` 存在。

## 8. Path Data

- [ ] `/paths/samples/sampled_link_indices` 存在。
- [ ] `/paths/samples/vertices_m` 存在。
- [ ] `/paths/samples/interaction_type` 存在。
- [ ] `/paths/samples/object_id` 存在。
- [ ] `/paths/samples/primitive_id` 存在。
- [ ] `/paths/samples/doppler_hz` 存在。
- [ ] `/paths/samples/tau_s` 存在。
- [ ] `/paths/samples/path_gain_db` 存在。
- [ ] `/paths/samples/path_type` 存在。
- [ ] 至少一条 valid path 被解析。
- [ ] 对每条 sampled path，`interaction_type`、`object_id`、`primitive_id` 的 depth 维对齐。
- [ ] 对 NLoS sampled path，至少一个中间 `vertices_m` 是 finite 3D 坐标。
- [ ] 静态场景允许 `doppler_hz == 0`，但字段必须存在且为 finite。

## 9. Observation Channel

- [ ] `/waveform/standard` 存在。
- [ ] `/waveform/fft_size` 存在。
- [ ] `/waveform/pilot_indices` 存在。
- [ ] `/waveform/pilot_symbols` 存在。
- [ ] `/receiver/estimator_type` 存在。
- [ ] `/observation/cfr_est` 存在。
- [ ] `/observation/valid_mask` 存在。
- [ ] `/observation/detection_success` 存在。
- [ ] `/observation/estimation_success` 存在。
- [ ] `/observation/snr_db` 存在。
- [ ] `/evaluation/nmse_db` 存在。
- [ ] `cfr_est.ndim == 6`。
- [ ] `cfr_est.shape[1:] == truth_cfr.shape`。
- [ ] `valid_mask.shape == [snapshot, tx, rx]`。
- [ ] `nmse_db.shape == [snapshot, tx, rx]`。

## 10. Impairments

- [ ] `/impairments/model_version` 存在。
- [ ] `/impairments/random_seed` 存在。
- [ ] `/observation/cfo_hz` 存在。
- [ ] `/observation/sfo_ppm` 存在。
- [ ] `/observation/timing_offset_samples` 存在。
- [ ] `/observation/phase_offset_rad` 存在。
- [ ] `/observation/agc_gain_db` 存在。
- [ ] `/observation/clipping_flag` 存在。
- [ ] 固定 seed 下 impairment 采样可复现。
- [ ] CFO 增大时，相位漂移指标增大。
- [ ] clipping threshold 降低时，`clipping_flag` 比例不下降。

## 11. Motion 与 Doppler

- [ ] `/motion/snapshot_id` 存在。
- [ ] `/motion/timestamp_s` 存在。
- [ ] `timestamp_s` 单调非递减。
- [ ] `/devices/tx_velocity_mps` 存在。
- [ ] `/devices/rx_velocity_mps` 存在。
- [ ] `/devices/tx_orientation_rad` 存在。
- [ ] `/devices/rx_orientation_rad` 存在。
- [ ] 静态配置下 path-level Doppler 接近 0。
- [ ] 非零速度配置下至少部分 valid path 的 Doppler 非 0。
- [ ] 多 time-step CFR shape 与 `num_time_steps` 一致。

## 12. Calibration 与 Diagnostics

- [ ] `/calibration/profile_id` 存在。
- [ ] `/calibration/fitted_parameters` 存在，或明确为空 profile。
- [ ] `/calibration/validation_metrics` 存在，或明确为空 profile。
- [ ] `/evaluation/detection_rate` 存在。
- [ ] `/evaluation/estimation_failure_rate` 存在。
- [ ] 可生成 NMSE/SNR/phase drift/failure rate 诊断报告。

## 13. 批处理与性能

- [ ] 支持 TX/RX 分批配置。
- [ ] batch run 能写入正确 HDF5 slice。
- [ ] manifest 记录 `batching.enabled`。
- [ ] manifest 记录 `batching.total_batches`。
- [ ] manifest 记录 `batching.completed_batches`。
- [ ] manifest 记录 `batching.failed_batches`。
- [ ] 中途失败可定位到具体 batch。
- [ ] 失败 batch 不会伪装成全局成功。

## 14. 统计有效性

必须通过统计测试：

- [ ] AWGN-only 下，高 SNR 的 median NMSE 低于低 SNR。
- [ ] 高 SNR debug case 中，median NMSE 小于配置阈值，默认建议 `-20 dB`。
- [ ] CFO 增大时，相位漂移指标增大。
- [ ] clipping threshold 降低时，clipping rate 不下降。
- [ ] 静态无速度时 Doppler 接近 0。
- [ ] 有速度时至少部分 path Doppler 非 0。

## 15. 可视化与报告

- [ ] 拓扑图可生成且非空。
- [ ] path samples 图可生成且非空。
- [ ] CFR magnitude 图可生成且非空。
- [ ] NMSE/SNR 诊断图可生成且非空。
- [ ] phase drift 或 Doppler 诊断图可生成，若对应功能启用。
- [ ] 每个 run 生成 manifest。
- [ ] 每个 run 生成 logs。

## 16. 最终通过标准

系统只有在以下条件全部满足时，才算最终验收通过：

- [ ] Phase 0 到 Phase 8 全部完成。
- [ ] 所有 Phase 的验收记录存在。
- [ ] 本清单所有必选项已勾选。
- [ ] `uv run ruff check .` 成功。
- [ ] `uv run pytest` 成功。
- [ ] 最新 commit 包含最终验收状态说明。

