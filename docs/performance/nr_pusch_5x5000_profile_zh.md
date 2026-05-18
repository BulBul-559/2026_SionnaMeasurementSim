# NR PUSCH 5x5000 性能分析报告

日期：2026-05-12

分析分支：`perf/nr-pusch-5x5000-profile`

输出目录：`outputs/perf_nrp5x5000_full_output`

状态更新：本文是 5x5000 单 GPU 基线分析的历史报告，用于解释当时的瓶颈判断。后续已经完成 CLI override 修复、NR PUSCH batching、run-full 原生 UE shard、4 GPU shard 和 6x8884 全量验收；当前生产能力和剩余 TODO 以 `docs/performance/nr_pusch_sharded_productionization.md`、`docs/performance/nr_pusch_performance_optimization_todo.md` 为准。当前配置命名已改为 BS/UE（`max_bs/max_ue`），下文旧 `max_tx/max_rx` 只表示历史实验记录。

## 运行配置

本次使用单张 RTX 4090 隔离运行，`CUDA_VISIBLE_DEVICES=0`。测试目标是“全输出压力”，也就是尽量保留当前大规模数据准备中可能使用的重型输出。

- 5 BS x 5000 UE，共 25000 条 SU-MIMO link
- NR PUSCH 4x4 SU-MIMO
- CSI 估计：`pusch_ls`
- MIMO detector：`lmmse`
- `array.spectrum.enabled=true`
- 空间谱来源：`truth_cfr`、`cfr_est`、`rx_grid`
- 可视化开启
- `save_full_paths=false`

当前 CLI 的全局 `--config` 参数需要放在 `run-full` 前面：

```bash
CUDA_VISIBLE_DEVICES=0 \
SIONNA_PERF_TRACE=1 \
SIONNA_PERF_HARDWARE_INTERVAL_S=1 \
SIONNA_PERF_LINK_LOG_INTERVAL=250 \
uv run python -m sionna_measurement_sim.app.cli \
  --config config/perf/nr_pusch_5x5000_full_output.yaml \
  run-full \
  --max-tx 5 \
  --max-rx 5000 \
  --output-dir outputs/perf_nrp5x5000_full_output
```

性能分析分支额外生成了：

- `logs/perf_events.jsonl`
- `logs/hardware_samples.csv`
- `logs/link_chunks.csv`

这些临时打点代码没有合并进 `main`，`main` 只保留本报告。

## 总体结果

总耗时约 1275.5 秒，也就是约 21 分 16 秒。

| 阶段 | 耗时 | 占比 | 主要信号 |
|---|---:|---:|---|
| NR PUSCH observation | 839.3 s | 65.8% | GPU 利用率低，逐 link 串行 receiver 调用 |
| HDF5 写入 | 206.5 s | 16.2% | I/O 和压缩长尾明显 |
| Sionna RT 求解 | 114.8 s | 9.0% | GPU compute-bound |
| 可视化 | 55.9 s | 4.4% | CPU 读 HDF5 + Matplotlib |
| Array 输出和空间谱 | 52.8 s | 4.1% | CPU 多核向量化计算 |
| Schema 校验 | 4.5 s | 0.4% | 小头 |

NR PUSCH 共处理 25000 条 link，没有 receiver failure。

| PUSCH 子阶段 | 总耗时 | 单 link 平均 | 判断 |
|---|---:|---:|---|
| `PUSCHReceiver` | 625.3 s | 25.0 ms | PUSCH 最大瓶颈 |
| LS estimator | 114.1 s | 4.6 ms | 第二大成本，当前逐 link 构建/调用 |
| TX generation | 45.5 s | 1.8 ms | 单次不大，但重复 25000 次 |
| Channel apply total | 19.5 s | 0.8 ms | 不是主要瓶颈 |
| metrics / CPU conversion / slice | 15.7 s | 小于 1 ms | 单项较小 |

每 250 条 link 的统计非常稳定：

- 平均 link 时间：33.35 ms
- 最快 chunk：32.92 ms/link
- 最慢 chunk：34.07 ms/link
- 第一个 chunk 有一次冷启动尖峰，最大约 282 ms

硬件观测：

| 阶段 | GPU 利用率 平均/峰值 | GPU 显存峰值 | CPU 平均 | RSS 峰值 |
|---|---:|---:|---:|---:|
| RT 求解 | 82% / 100% | 6.5 GB | 103% | 3.3 GB |
| PUSCH link loop | 10.9% / 11% | 11.6 GB | 101% | 5.6 GB |
| Array 空间谱 | 0% | 11.6 GB | 765% | 11.8 GB |
| HDF5 写入 | 0% | 11.6 GB | 101% | 12.0 GB |
| 可视化 | 0% | 11.6 GB | 100% | 12.4 GB |

输出检查：

- `results.h5`：4.08 GB
- `/array/spatial_spectrum_truth`：`(1, 5000, 5, 91, 181)`
- `/array/spatial_spectrum_cfr_est`：`(1, 5000, 5, 91, 181)`
- `/array/spatial_spectrum_observation`：`(1, 5000, 5, 91, 181)`
- `/waveform/tx_grid`：`(1, 5000, 5, 4, 14, 48)`
- `/waveform/rx_grid`：`(1, 5000, 5, 4, 14, 48)`
- `/waveform/tx_time` 和 `/waveform/rx_time`：不存在，符合当前约定
- `figures/index.json` 存在，PNG 非空
- HDF5 schema 校验通过

## 当前分块策略

当前项目实际上还没有面向大规模 UE/BS 的正式计算分块或多 GPU shard 策略。

现有行为可以分成几类：

| 类型 | 当前状态 | 说明 |
|---|---|---|
| 输入数量限制 | 有 | `max_tx` / `max_rx` 只是在读取 label 时取 `bs_points[:max_tx]` 和 `ue_points[:max_rx]`，属于截断，不是分块 |
| SU-MIMO PUSCH 计算分块 | 基本没有 | 当前按 `snap -> UE(ul_tx) -> BS(ul_rx)` 三层循环逐 link 处理，每次调用一次 transmitter/channel/receiver |
| PUSCH batch 维度 | 未用于跨 link 合批 | Sionna 调用里 batch 基本是 `1`，没有把多个 UE/BS link 合成一个大 batch |
| 多 GPU 分片 | 没有 | 目前一次 run 只绑定一个 `runtime.device` / `CUDA_VISIBLE_DEVICES`，没有按 UE 或 BS 自动分配到多进程多 GPU |
| HDF5 输出分块 | 没有生产级 shard | 当前是单个 `results.h5`，一次性写完整数组；没有 per-shard HDF5 和后处理合并 |
| `run-batch` | 不是 UE/BS 分块 | 现有 batch runner 主要是多 seed / 多实验批次，输出多个实验目录，不是把一个 5x5000 场景拆成 UE shard |
| 可视化采样 | 有，但只影响画图 | 可视化最多采样少量 UE/BS，不减少仿真和 HDF5 主数据量 |
| profiling `link_chunks` | 仅日志分块 | 性能分析里每 250 link 写一行统计，只是观察窗口，不改变计算方式 |

所以这次 5x5000 的真实执行方式是：

1. RT 阶段一次性对 5 BS x 5000 UE 生成路径/CIR/CFR 所需数据。
2. NR PUSCH 阶段对 25000 条 link 串行循环。
3. 每条 link 独立执行一次 PUSCH 发送、信道应用、LS 估计、PUSCHReceiver、指标计算。
4. 所有结果数组在内存中累计。
5. 最后一次性生成 array 输出、写一个 HDF5、做 schema 校验和采样可视化。

这就是为什么会出现“显存不低但 GPU 利用率低”的情况：数据和对象已经占在显存里，但每次喂给 GPU 的工作太小，且由 Python 逐 link 调度，GPU 没有被大 batch 持续填满。

## 瓶颈判断

第一瓶颈是 SU-MIMO 逐 link NR PUSCH loop。

PUSCH 阶段 GPU 显存常驻约 11.6 GB，但 GPU 利用率只有约 11%。这说明它不是显存容量瓶颈，也不是 GPU 算力饱和瓶颈，而是“大量小任务串行调度 + receiver 调用颗粒度太细”。其中 `PUSCHReceiver` 单独占了 625.3 秒，是最值得优先处理的热点。

第二瓶颈是 HDF5 写入。

全输出压力下 HDF5 写入耗时 206.5 秒，最终文件约 4.08 GB。这个阶段 GPU 空闲，CPU 接近单核工作。也就是说，即使后续把 PUSCH 加速很多，重型输出仍会留下明显长尾。

第三个独立瓶颈是 RT 求解。

RT 阶段 GPU 平均利用率 82%，峰值 100%，功耗峰值约 451 W。这和 PUSCH 阶段完全不同：RT 更像真正的 GPU compute-bound。所以“GPU 利用率很高但显存占用不大”的情况，主要对应 RT 阶段。

Array 空间谱是 CPU 多核阶段。

三个 Bartlett 空间谱合计约 50 秒，CPU 平均利用率较高，RSS 接近 12 GB。它不是总耗时第一名，但会显著提高内存压力，并推高后续 HDF5 写入体积。

## 优化建议

1. 优先做 SU-MIMO link batching。

   当前 25000 条 link 是 25000 次小 receiver 调用。最优先的改造是把多个 `(UE, BS)` link 合到一个 batch 里处理，先测试 batch size 8、16、32、64，观察 GPU 利用率、显存和结果一致性。

2. 缓存或外提可复用对象。

   当前 `PUSCHLSChannelEstimator` 在每条 link 内构建。即使构建成本不是全部 LS 成本，这种 25000 次重复也不合理，应该按 resource grid / DMRS config 缓存或提前构造。

3. 先做 UE shard，再做多 GPU。

   多 GPU 不建议直接在现有单 HDF5 写入链路上硬并发。更稳的方式是先支持 `rx_start/rx_count` 或显式 `rx_indices`，按 UE 维度切 shard，每个进程绑定一张 GPU，输出独立 shard HDF5 和 manifest。

4. 重型输出改成 shard-aware。

   waveform grid 和三类空间谱让 HDF5 写入成为第二大瓶颈。大规模生产建议先写 per-shard 文件，后续用索引或离线合并，而不是所有数据一次性汇总到一个 HDF5。

5. 避免重复存储大数组。

   `aoa_heatmap_label` 和 `spatial_spectrum_label` 语义上是 alias。5x5000、91x181 分辨率下，如果两者都是物理 dataset，会增加文件体积和写入时间。可以考虑 HDF5 hard link 或只保留一个物理 dataset。

6. 把 RT、PUSCH、HDF5 分开 benchmark。

   RT 是 GPU compute-bound，PUSCH 是小任务调度/receiver-bound，HDF5 是 I/O-bound。后续每次优化都应该用单独 benchmark 判断它到底改善了哪一段。

## 下一步建议实验

1. 关闭 `array.spectrum` 和 visualization，跑 5x5000，隔离 RT + PUSCH。
2. 从缓存 CIR/CFR 直接跑 PUSCH，去掉 RT 影响。
3. 原型实现 SU-MIMO batch size 8、16、32、64。
4. 做 2 GPU UE shard 原型，每张 GPU 一个进程、一个 HDF5。
5. 对比空间谱关闭、空间谱开启、label alias 去重三种 HDF5 写入耗时。

## 结论

当前系统的大规模性能问题不是单一瓶颈。

最主要的运行时间花在 NR PUSCH 逐 link receiver 调用上，这导致显存占用不低但 GPU 利用率很低。RT 阶段则确实能吃满 GPU，属于另一类 compute-bound 问题。全输出压力下，HDF5 写入和空间谱/可视化也会形成明显长尾。

因此下一阶段最值得先做的是：UE/BS link batching + UE shard + per-shard HDF5。多 GPU 应该建立在 shard 机制之上，而不是直接把当前单进程单 HDF5 流程并发化。
