# NR PUSCH 性能优化迭代 1 报告

日期：2026-05-14

来源：性能优化实验分支。本文记录实验结论和建议实现方式，内容应能独立带回 `main`；不要求读者保留或检出实验分支代码。

状态更新：本文是优化迭代 1 的历史报告。文中关于 CLI override、正式 batch 配置、run-full 原生 UE shard、多 GPU shard 和空间谱 chunk 的“后续建议”，已经在后续生产化工作中完成；当前生产状态以 `docs/performance/nr_pusch_sharded_productionization.md` 为准，剩余 TODO 以 `docs/todo/performance.md` 为准。当前配置命名已改为 BS/UE（`max_bs/max_ue`），下文旧 `max_tx/max_rx` 只表示历史实验记录。

## 结论

本轮验证了第一批优化方向：NR PUSCH SU-MIMO link batching、Bartlett 空间谱向量化、采样可视化读数复用、`result_xxx.h5` 多文件输出语义，以及通用性能模块设计指南。

最有效的优化是 PUSCH link batching。3x3000 端到端 full-output 从旧基线 448.8 s 降到 153.5 s，主要来自 PUSCH 阶段从 286.3 s 降到 8.1 s。当前 3x3000 的新瓶颈已经转移到 HDF5 写入、RT 和可视化。

在本次 RTX 4090、4x4 SU-MIMO、3x3000/5x5000/6x8884 实验环境下，batch size 64 是表现最好的经验值。batch 128 没有 OOM，但比 64 慢，说明继续增大 batch 会遇到 Sionna receiver 内部开销或 GPU kernel/内存访问效率平台期。正式落地时应把 batch size 作为配置项或启动时探测结果，而不是写死为全局默认。

## 可迁移结论

| 结论 | 建议落地方式 |
|---|---|
| SU-MIMO link batching 是最有效的 NR PUSCH 优化 | 在 NR PUSCH 链路中把多个 `(snapshot, UE, BS)` link 合成 batch 输入，一次调用 transmitter、channel apply、LS estimator 和 receiver |
| `PUSCHLSChannelEstimator` 等重复对象应外提或缓存 | 按 resource grid、DMRS 和 receiver 配置建立可复用对象，避免逐 link 构建 |
| `result_xxx.h5` 多文件输出语义可行 | 生产实现应在 `run-full` 中按 UE/BS shard 直接写多个结果文件，而不是先写完整 `results.h5` 再拆 |
| 空间谱向量化对中等规模有效 | 大规模生产应采用 UE chunk 向量化，控制内存峰值并避免逐 link Python 循环 |
| 可视化应只读取采样 link 所需数据 | pipeline 可视化保持示意采样，不重复读取完整大数组 |

## PUSCH-only Batch Sweep

输入：3 BS x 3000 UE，共 9000 条 SU-MIMO link，复用同一个 `rt_cache.h5`。

| Batch size | PUSCH-only 总耗时 | `process_links` | 平均 link | p95 | Receiver failures | OOM fallback |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 276.2 s | 272.6 s | 30.26 ms | 30.97 ms | 0 | 0 |
| 2 | 142.4 s | 138.5 s | 15.33 ms | 15.75 ms | 0 | 0 |
| 4 | 73.5 s | 69.8 s | 7.71 ms | 7.91 ms | 0 | 0 |
| 8 | 39.2 s | 35.3 s | 3.89 ms | 4.02 ms | 0 | 0 |
| 16 | 22.5 s | 18.6 s | 2.04 ms | 2.53 ms | 0 | 0 |
| 32 | 13.1 s | 9.8 s | 1.07 ms | 1.32 ms | 0 | 0 |
| 64 | 9.2 s | 5.7 s | 0.61 ms | 0.71 ms | 0 | 0 |
| 128 | 10.3 s | 7.0 s | 0.75 ms | 0.80 ms | 0 | 0 |

旧 PUSCH-only 基线是 297.3 s。batch=1 已降到 276.2 s，主要来自 `PUSCHLSChannelEstimator` 从逐 link 构建改为复用；batch=64 则进一步降到 9.2 s。

瓶颈原因判断：

- 旧实现是 9000 次小粒度 `PUSCHReceiver` 调用，GPU 常驻显存但利用率低。
- batching 后每 64 条 link 一次 `tx(batch)`、一次 channel apply、一次 LS estimator、一次 `PUSCHReceiver`，显著减少 Python 调度和小 kernel 开销。
- batch 128 GPU util 更高但耗时反弹，说明不是越大越好；当前 64 是本机 4090、3x3000、4x4 SU-MIMO 下的经验甜点。

## Write-only 输出侧结果

输入：旧 3x3000 full-output `results.h5`。

| Benchmark | 旧基线 | 本轮结果 | 变化 |
|---|---:|---:|---:|
| Write-only 总耗时 | 53.1 s | 35.5 s | -33.1% |
| Array outputs | 18.0 s | 5.1 s | -71.7% |
| Visualization | 29.2 s | 25.0 s | -14.4% |
| HDF5 clone | 3.2 s | 3.2 s | 基本不变 |
| Schema validate | 1.9 s | 1.9 s | 基本不变 |

有效优化：

- Bartlett 空间谱新增中等规模向量化路径，避免 9000 个 link 的 Python 循环。
- 可视化复用选中 link 的 CFR/spectrum slice，减少重复 HDF5 读取。

失败实验：

- 曾把 `angle_chunk_size=2048` 作为默认，导致 steering matrix 在每个 link 上重复生成，`array_outputs` 超过 120 s 仍未结束。已改回默认快速预计算路径，chunk 仅作为显式可选手段。

## Multi-file `result_xxx.h5` 实验

结果：

| 指标 | 数值 |
|---|---:|
| 文件数 | 6 |
| 命名 | `result_000.h5` 到 `result_005.h5` |
| schema status | 全部 valid |
| `hdf5_shard_write` | 115.0 s |
| `array_outputs` | 5.1 s |
| 总耗时 | 120.4 s |

判断：

- 多文件语义是正确的，manifest 记录 `ue_range`、`bs_range`、`scene_id`、shape 和 schema status。
- 当前实验不是写盘加速方案，因为它仍是“从完整 HDF5 再拆 shard”；即使采用 HDF5 slice 读取，也要重复创建多个结果文件并分别校验 schema。
- 它的价值是验证 `result_xxx.h5` 命名、manifest 语义和“不并发写同一个 HDF5”的方案，为后续多 GPU UE shard 打底。
- 下一步真正提速需要在 `run-full` 中按 UE shard 直接生成 `result_xxx.h5`，而不是先生成单个完整 `results.h5` 再拆。

## 3x3000 端到端验证

验证口径：3 BS x 3000 UE，NR PUSCH 4x4 SU-MIMO，开启 full-output pressure，包括三类空间谱和采样可视化；batch size 固定为 64。

| 阶段 | 旧 3x3000 | 本轮 3x3000 batch64 |
|---|---:|---:|
| 总耗时 | 448.8 s | 153.5 s |
| RT solve | 42.3 s | 42.0 s |
| NR PUSCH observation | 286.3 s | 8.1 s |
| `nr_pusch.process_links` | 约 294.4 s（PUSCH-only） | 5.7 s |
| Array outputs | 16.3 s | 3.4 s |
| HDF5 write | 71.8 s | 72.1 s |
| Visualization | 29.2 s | 25.2 s |
| Schema validate | 1.9 s | 1.8 s |

验收：

- HDF5 schema 通过。
- `/waveform/tx_time` 和 `/waveform/rx_time` 不存在。
- 生成 21 张 PNG，`figures/index.json` 存在且非空。
- 关键 shape 保持：
  - `/channel/truth/cfr`: `(3, 3000, 4, 4, 48)`
  - `/waveform/tx_grid`: `(1, 3000, 3, 4, 14, 48)`
  - `/waveform/rx_grid`: `(1, 3000, 3, 4, 14, 48)`
  - 三类空间谱: `(1, 3000, 3, 91, 181)`
- `nr_pusch_batching`: requested/effective batch size 均为 64，141 个 batch，0 OOM fallback，0 per-link fallback。

## 标准与全量测试

5x5000 标准 full-output 已通过：

| 阶段 | 旧 5x5000 | 本轮 5x5000 batch64 |
|---|---:|---:|
| 总耗时 | 1275.5 s | 430.6 s |
| RT solve | 114.8 s | 113.9 s |
| NR PUSCH observation | 839.3 s | 17.9 s |
| `nr_pusch.process_links` | 约 839 s 主体 | 15.2 s |
| Array outputs | 52.8 s | 45.0 s |
| HDF5 write | 206.5 s | 203.9 s |
| Visualization | 55.9 s | 43.7 s |
| Schema validate | 4.5 s | 4.5 s |

5x5000 输出检查：

- HDF5 schema 通过。
- `results.h5` 约 3.9 GB。
- `/waveform/tx_time` 和 `/waveform/rx_time` 不存在。
- 21 张 PNG 正常生成。
- `nr_pusch_batching`: 391 个 batch，0 OOM fallback，0 per-link fallback。

6x8884 全量 full-output 已通过：

| 阶段 | 本轮 6x8884 batch64 |
|---|---:|
| 总耗时 | 857.1 s |
| RT solve | 231.8 s |
| NR PUSCH observation | 35.7 s |
| `nr_pusch.process_links` | 32.5 s |
| Array outputs | 95.3 s |
| HDF5 write | 431.4 s |
| Visualization | 51.0 s |
| Schema validate | 8.6 s |

6x8884 输出检查：

- HDF5 schema 通过。
- `results.h5` 约 8.0 GB。
- `/channel/truth/cfr`: `(6, 8884, 4, 4, 48)`。
- `/waveform/tx_grid` 和 `/waveform/rx_grid`: `(1, 8884, 6, 4, 14, 48)`。
- 三类空间谱: `(1, 8884, 6, 91, 181)`。
- `/waveform/tx_time` 和 `/waveform/rx_time` 不存在。
- 21 张 PNG 正常生成。
- `nr_pusch_batching`: 833 个 batch，0 OOM fallback，0 per-link fallback。

注意：第一次尝试 6x8884 时，CLI `--max-tx 6` 被 argparse 默认值判断吞掉，实际只跑了 `max_tx=3`。已中断并删除该错误输出；最终验收使用临时实验 YAML 显式设置 `input.max_tx=6`、`input.max_rx=8884`。

## 后续建议

1. 当前新瓶颈是 HDF5 write、RT solve、array spectrum 和 visualization；下一轮应优先做 run-full 原生 UE shard，直接输出 `result_xxx.h5`，不要先写完整单文件再拆。
2. 回主分支时建议把 batch size 做成正式配置项，而不是只靠环境变量；默认值仍可保守为 1 或按设备自动探测。
3. CLI 对 `--max-tx 6` 的 config override 规则需要修正，否则默认值刚好等于目标值时无法覆盖 YAML。正式修复应让 CLI 能区分“用户未传参”和“用户显式传入默认值相同的参数”。
4. 5x5000/6x8884 的空间谱输出超过当前向量化内存阈值，走了保守路径；后续可以做按 UE chunk 的 vectorized spectrum，兼顾内存和速度。
5. `aoa_heatmap_label` / `spatial_spectrum_label` 冗余仍未处理，按计划留到输出字段精简阶段。
