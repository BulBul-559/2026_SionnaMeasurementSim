# Performance Docs Index

`docs/performance/` 保存历史性能实验、规模 sweep、生产化验收和性能设计记录。这里的数字、
参数和命令反映当时实验事实，不代表当前默认配置。当前系统事实以
`docs/agent_handoff.md`、`docs/sys/` 和 `config/README.md` 为准；active 性能 TODO
以 `docs/todo/performance.md` 为准。

## 当前参考入口

| 目标 | 当前入口 |
|---|---|
| 当前 SRS 100 MHz baseline 与推荐 shard | `docs/sys/indoor_fr1_100mhz_validation.md` |
| 当前配置字段和模板说明 | `config/README.md` |
| 当前 HDF5/schema 契约 | `docs/sys/07_config_and_h5_format.md` |
| 当前 benchmark harness | `docs/performance/benchmark_harness.md` |
| active 性能优化事项 | `docs/todo/performance.md` |
| active 结构优化事项 | `docs/todo/structure.md` |

## Legacy Audit 2026-05-22

本次审查结论：大多数 performance 文档仍作为历史实验记录保留在本目录；只迁移已被结果文档
取代、且本身不是实验结果的执行计划。

| 文档 | 状态 | 说明 |
|---|---|---|
| `nr_pusch_5x5000_profile.md` | 保留，历史记录 | 已有 status update；英文版单 GPU profiling 记录。 |
| `nr_pusch_5x5000_profile_zh.md` | 保留，历史记录 | 已有 status update；中文版 profiling 记录。 |
| `nr_pusch_optimization_iteration_1.md` | 保留，历史记录 | 已有 status update；记录 PUSCH batch/shard 优化迭代。 |
| `nr_pusch_sharded_productionization.md` | 保留，生产化历史验收 | 当前仍是 PUSCH shard/batch 性能的重要历史验收。 |
| `nr_srs_7x500_sharded_confirmation.md` | 保留，历史确认测试 | 已补状态说明；`shard_size=25` 是历史结果，不是当前推荐。 |
| `nr_srs_direct_uplink_scale_sweep_round1.md` | 保留，历史 sweep | 已补状态说明；用于解释 early direct-uplink scale 边界。 |
| `nr_srs_direct_uplink_scale_sweep_round2_bs_results.md` | 保留，历史 sweep | 已补状态说明；对应 BS sweep 结果。 |
| `nr_srs_direct_uplink_scale_sweep_round3_ue_block_results.md` | 保留，历史 sweep | 已补状态说明；对应 UE block 边界。 |
| `nr_srs_rt_variant_sweep_6x5.md` | 保留，历史 RT variant sweep | 已补状态说明；其中 `srs_cfr_est` 是历史 alias。 |
| `performance_optimization_design_guide.md` | 保留，历史设计指南 | 已补状态说明并修正过期引用；active TODO 仍在 `docs/todo/`。 |
| `nr_srs_direct_uplink_scale_sweep_round2_bs_plan.md` | 迁移到 legacy | 这是已被 round2 results 取代的执行计划，不是实验结果。 |

迁移位置：

```text
docs/legacy/performance/nr_srs_direct_uplink_scale_sweep_round2_bs_plan.md
```

## 阅读规则

- 不要从 performance 文档反推出当前默认配置；先读当前参考入口。
- `shard_size=25`、`array.spectrum.sources: ["srs_cfr_est"]`、
  `/array/spatial_spectrum_label`、`/array/spatial_spectrum_srs` 等口径只表示历史实验事实。
- schema `1.5.0` 后，SRS/array 当前口径见 `docs/sys/07_config_and_h5_format.md`。
- 如果历史文档只有“计划”而没有实验结果，并且已被结果文档取代，应迁到 `docs/legacy/`。
