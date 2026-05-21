# TODO History

已完成 TODO 的简洁归档。每个分类按完成顺序往后追加，只保留完成日期和一句话描述。

## Feature

| 完成时间 | 描述 |
|---|---|
| 2026-05-19 | 实现 waveform-level ranging v1，从 `/observation/cfr_est` 输出 PDP peak 和 phase-slope ToA/range observation。 |
| 2026-05-21 | 完成 NR SRS standards-shaped v2 subset：comb/BWP、hopping、sequence hopping、cyclic-shift multiplexing、port map、power scaling 和 resource receiver。 |

## Structure

| 完成时间 | 描述 |
|---|---|
| 2026-05-18 | PUSCH/SRS 接入通用 clean channel → impairment/AWGN → receiver 链路，统一 waveform 字段和 impairment metadata。 |

## Performance

| 完成时间 | 描述 |
|---|---|
| 2026-05-14 | 修复 CLI `--max-bs` / `--max-ue` override 并补测试。 |
| 2026-05-14 | 增加配置驱动 debug profiling，输出阶段耗时和硬件采样。 |
| 2026-05-14 | 生产化 `run-full` UE shard 输出，使用多 `result_xxx.h5` 与 manifest 汇总。 |
| 2026-05-14 | 完成 4 GPU UE shard 运行和 `6x8884` 全输出验收。 |
| 2026-05-14 | 增加 NR PUSCH SU-MIMO batch 配置并完成 batch64 验证。 |
| 2026-05-14 | 增加空间谱 link chunk 生成，降低大规模空间谱内存压力。 |
| 2026-05-14 | 打通 HDF5 compression 配置传递。 |
| 2026-05-21 | `path_samples` 可视化默认限制为当前采样选择中的一条 UE-BS link，消除多链路路径长尾和混叠。 |

## Bug

| 完成时间 | 描述 |
|---|---|
| 2026-05-21 | 修正 visualization 空间谱和 AoA 绘图的 uplink/downlink role-aware 索引。 |
| 2026-05-21 | 修正 Bartlett steering 与 Sionna PlanarArray 元素顺序/方向不一致的问题，并统一到 scene/global angle frame。 |
