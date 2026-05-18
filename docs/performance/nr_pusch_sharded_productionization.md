# NR PUSCH Sharded Productionization Report

日期：2026-05-14

本文记录 `run-full` 原生 UE shard、NR PUSCH SU-MIMO batching、空间谱 chunk、HDF5 compression 配置和 debug profiling 的生产化验收结果。

## 实现摘要

| 项目 | 状态 | 说明 |
|---|---|---|
| CLI override | DONE | `run-full --config` 下显式传入的 CLI 参数会覆盖 YAML |
| Debug profiling | DONE | `debug.enabled=true` 输出 `perf_events*.jsonl`、`hardware_samples*.csv`、`perf_summary*.json` |
| UE shard | DONE | 直接输出 `result_000.h5`、`result_001.h5`；不并发写同一个 HDF5 |
| Shard manifest | DONE | 根目录 `manifest.json` 汇总 shard 文件、全局 UE 覆盖、resolved TX/RX 索引、batching stats、perf summary |
| `/shard` HDF5 元数据 | DONE | 每个 shard 写入局部到全局索引映射 |
| SU-MIMO batching | DONE | `phy.su_mimo_link_batch_size` 控制独立 link batch；失败时可降级 |
| 空间谱 chunk | DONE | `array.spectrum.link_chunk_size` 控制 Bartlett link chunk |
| HDF5 compression | DONE | `output.compression` 支持 `gzip`、`lzf`、`none` |

## 3x3000 验收

命令使用 `config/perf/nr_pusch_3x3000_sharded.yaml`，单 GPU、3 个 shard、batch size 64、空间谱三源和 visualization 开启。

| 指标 | 结果 |
|---|---|
| 规模 | 3 BS x 3000 UE x 4x4 NR PUSCH |
| 端到端耗时 | 178.01 s |
| 输出文件 | 3 个 `result_xxx.h5` |
| 输出体积 | 1.6 GB |
| UE 覆盖 | `0..2999`，无缺口 |
| Schema | 3/3 shard 通过 |
| 空间谱 | `truth`、`cfr_est`、`observation` 均存在 |
| 时域 waveform | 未写入 `tx_time` / `rx_time` |
| Batching | requested=64，effective=64，无 fallback |

阶段耗时聚合：

| 阶段 | 耗时 |
|---|---:|
| `hdf5_write` | 67.99 s |
| `rt_solve` | 57.26 s |
| `visualization` | 24.67 s |
| `nr_pusch_observation` | 13.88 s |
| `array_outputs` | 10.92 s |
| `schema_validate` | 2.23 s |

## 6x8884 验收

命令基于 `config/perf/nr_pusch_6x8884_sharded.yaml`，实际使用空闲 GPU `[0, 2, 3, 4]`，9 个 UE shard、batch size 64、空间谱三源和首 shard visualization 开启。

| 指标 | 结果 |
|---|---|
| 规模 | 6 BS x 8884 UE x 4x4 NR PUSCH |
| 端到端耗时 | 279.45 s |
| GPU | 4 GPU shard: `[0, 2, 3, 4]` |
| 输出文件 | 9 个 `result_xxx.h5` |
| 输出体积 | 8.5 GB |
| shard 大小 | 8 x 1000 UE + 1 x 884 UE |
| UE 覆盖 | `0..8883`，无缺口 |
| Schema | 9/9 shard 通过 |
| 空间谱 | `truth`、`cfr_est`、`observation` 均存在 |
| 时域 waveform | 未写入 `tx_time` / `rx_time` |
| Batching | requested=64，effective=64，无 fallback |

阶段耗时聚合：

| 阶段 | 耗时 |
|---|---:|
| `hdf5_write` | 439.58 s |
| `rt_solve` | 287.17 s |
| `nr_pusch_observation` | 76.24 s |
| `array_outputs` | 75.07 s |
| `visualization` | 43.72 s |
| `schema_validate` | 11.76 s |

注意：阶段耗时是所有 shard 的求和；端到端耗时因 4 GPU 并行而显著小于阶段求和。

## 当前瓶颈

| 优先级 | 瓶颈 | 证据 | 后续方向 |
|---|---|---|---|
| P0 | HDF5 写盘/压缩 | 6x8884 聚合 `hdf5_write` 439.58 s | 做 chunk shape、`lzf`/`none`、压缩等级、分 dataset 写入策略实验 |
| P1 | RT solve | 6x8884 聚合 `rt_solve` 287.17 s | 建立 RT-only 参数矩阵，区分性能收益和传播语义变化 |
| P1 | 空间谱输出 | 6x8884 聚合 `array_outputs` 75.07 s | 继续优化 Bartlett chunk 内存布局，评估按 source 可选关闭 |
| P2 | Visualization | 只对首 shard 绘图仍 43.72 s | 减少重复 HDF5 打开和 Matplotlib 初始化 |

## 结论

`run-full` 已具备可生产使用的多文件 shard 输出路径。对当前 6x8884 规模，4 GPU shard + batch64 将全输出端到端耗时压到约 4.66 分钟；后续主要优化空间已经从 NR PUSCH receiver 转移到 HDF5 写盘、RT solve 和重型派生输出。
