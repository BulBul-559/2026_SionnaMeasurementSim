# Single-Scene Full Simulation Performance Optimization 2026-06-16

本记录对应分支 `codex/single-scene-performance-optimization` 上的单场景性能优化实验。性能数字是
本机当时的实验事实，不代表所有机器或所有场景的默认速度。

## Scenario

- 数据：`front3d_20/front3d_0002/label/label_panel_0p5.json`
- 输出模式：`output.profile: "full"`
- 主要产品：RT truth、CFR/CIR、path samples、full path、NR SRS observation、array spectrum、
  ranging、calibration、motion、visualization
- SRS：64 PRB，768 subcarriers，23 dBm，`absolute_thermal`
- Sharding：`shard_size=5`，共 57 shards

## Results

| Run | Workers / GPUs | Compression | Wall time | Size | Notes |
|---|---:|---|---:|---:|---|
| gzip baseline current | 6 / `[0,2,3,4,6,7]` | `gzip` | 222.9 s | 8.6 GiB | 当前代码优化前基线 |
| mixed stable | 4 / `[2,3,5,6]` | `mixed` | 290.1 s | 9.3 GiB | 稳定完成，worker 数较少 |
| mixed + covariance reuse | 4 / `[2,3,5,6]` | `mixed` | 293.9 s | 9.3 GiB | 空间谱复用 `rx_snapshot_matrix` covariance |
| mixed + covariance reuse + gzip level 1 | 4 / `[2,3,5,6]` | `mixed`, `gzip_level=1` | 235.3 s | 9.3 GiB | 当前推荐；manifest elapsed，`/usr/bin/time real=244.3 s` |
| mixed + matmul projection | 4 / `[2,3,5,6]` | `mixed` | 326.6 s | 9.3 GiB | 已放弃；BLAS 多线程竞争导致正式运行变慢 |

## Stage Totals

| Stage | gzip baseline | mixed stable | mixed + covariance reuse | mixed + covariance reuse + gzip level 1 |
|---|---:|---:|---:|---:|
| `hdf5_write` | 364.7 s | 173.1 s | 171.1 s | 113.6 s |
| `rt_solve` | 305.9 s | 321.5 s | 329.2 s | 282.6 s |
| `array_outputs` | 215.8 s | 229.4 s | 224.5 s | 171.1 s |
| `nr_srs_observation` | 70.5 s | 79.0 s | 79.3 s | 56.3 s |
| `visualization` | 64.1 s | 65.4 s | 69.7 s | 62.5 s |
| `schema_validate` | 30.2 s | 25.5 s | 25.5 s | 17.6 s |
| `ranging` | 23.5 s | 23.6 s | 24.7 s | 23.0 s |

## HDF5 Compression Finding

`/waveform/rx_grid` 是原 gzip run 最大写入热点：

| Dataset | gzip write time | mixed write time | gzip storage/raw | mixed storage/raw |
|---|---:|---:|---:|---:|
| `/waveform/rx_grid` | 159.5 s | 8.7-11.2 s | 0.894 | 1.000 |

结论：noisy/high-entropy complex observation grid 用 gzip 只节省少量空间，却消耗大量 CPU。
新增 `output.compression: "mixed"` 后，路径表和稀疏/结构化数组仍使用 gzip + shuffle，高熵复数观测
数组不压缩。该策略不改变数值和 HDF5 schema，但本场景文件体积约增加 0.7 GiB。

在 `mixed` 之上继续把 gzip dataset 的等级从默认 4 降到 1，`hdf5_write` 进一步降到约
113.6 s。真实 shard 抽样里，level 1 与 level 4 的文件体积差异小于 6%，但写入时间约少
35%。因此正式 64 PRB full 任务当前推荐：

```yaml
output:
  compression: "mixed"
  gzip_level: 1
```

## Array Spectrum Finding

已保留的优化：

- `build_bartlett_spectrum_from_covariance()`：新增从 covariance 直接生成 Bartlett spectrum 的入口。
- `build_array_outputs_from_waveform()` 对 `rx_grid` observation spectrum 复用已写出的
  `rx_snapshot_matrix`，避免重复计算一次 RX covariance。

收益较小：本场景 `array_outputs` 从约 229.4 s 降到 224.5 s。说明主要成本仍在 Bartlett
投影本身。

已放弃的优化：

- 将 `a^H R a` 从 `einsum` 改成 batched `matmul`。单进程微基准更快，但正式多 worker 运行中
  CPU/BLAS 线程竞争明显，`array_outputs` 反而恶化到 267.8 s，墙钟增加到 326.6 s。
- 对 full run 统一设置 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`。
  spectrum-only 微基准更快，但完整运行中 `nr_srs_observation` 与 `hdf5_write` 恶化，总墙钟没有收益。

## Stability Notes

一次 6-worker mixed run 在当时 GPU 0/1/4 已被其他进程高占用时出现 Dr.Jit/OptiX OOM stderr，
最终未形成完整 57-shard manifest。该结果不用于性能结论。正式对照使用 4 workers 稳定完成，
fallback split count 为 0。

## Buffered Writer Probe

用户提出的下一层优化是把 “compute chunk” 和 “write batch” 解耦：GPU 仍按 5/10/20 UE
小块计算，但多个小块先进入内存 buffer，再统一写入较少的 HDF5 文件或 appendable dataset。
这条方向架构上可行，但不能直接替代当前 shard writer，需要先做受控设计。

本次用现有 full run 产物做了两个轻量探针：

| Probe | Result | Interpretation |
|---|---:|---|
| 真实 shard 抽取 5 个大 dataset，8 份 many-files | 3.99 s |
| 同样数据写入一个 H5 的 8 个 group | 4.09 s | 单纯少开文件收益很小 |
| 同样数据写入 5 个 extendable dataset | 4.83 s | 大 chunk append/resize 反而略慢 |
| `benchmark write`，10 个 5-UE synthetic H5 | 3.57 s |
| `benchmark write`，1 个 50-UE synthetic H5 | 3.55 s | 合成大 dataset 没有明显 writer 收益，CFR gzip 还可能更慢 |

同时，57-shard full run 中 `hdf5_write=113.6 s`，但 9405 个 tracked dataset create/write
事件合计只有约 38.9 s。剩余约 74.7 s 主要来自 scalar 写入、attrs、group 创建、Python writer
遍历和 HDF5 metadata flush。也就是说，buffered writer 的潜在收益不在 “把大数组写得更快”，
而在减少重复写 root-level metadata、重复建 group/dataset、重复写 attrs 和重复 schema validate。

建议落地路线：

1. 先实现 `output.sharding.write_batch_size` 的实验性模式，但不要改默认值。
2. 不让多个 GPU worker 同时写同一个 H5。更稳的第一版是 “每个 GPU worker 顺序处理多个
   compute chunks，并写自己的 bundle 文件”，例如 `result_bundle_gpu2_000.h5`。
3. bundle 内部不要简单放多个完整 root group；需要新 `sionna_measurement_shard_bundle`
   contract 或 appendable writer，否则 reader/schema 会变复杂且收益有限。
4. 对固定 shape 的 tx-major dataset 做 extendable append；对 `/paths/samples`、path full、
   link labels 这类 link/path 维度数据，需要单独定义 append 轴和 global index。
5. 每次 flush 后写 bundle manifest/checkpoint；buffer 应有 `buffer_max_ue` 与
   `buffer_max_bytes`，避免 full 模式下 CFR、rx_grid、空间谱或 IQ 把内存吃爆。
6. 验收必须比较：同一场景同一配置下的数值一致性、manifest 可读性、失败恢复粒度、RSS 峰值、
   wall time、`hdf5_write`、schema validate 成本和下游 reader 改动量。

当前结论：append/buffer writer 是合理的二阶段性能工程，但复杂度明显高于 gzip/mixed 调参。
在没有新 bundle contract 和 reader 适配前，不建议作为本轮默认优化。

## Current Recommendation

- 正式 64 PRB full 仿真优先使用 `output.compression: "mixed"` 与 `output.gzip_level: 1`。
- 若机器 GPU 内存被外部任务占用，降低 `parallel_workers` 或避开高占用 GPU；不要把 OOM fallback
  运行的墙钟作为性能基线。
- 若用户不需要 `/array/spatial_spectrum_observation`，从配置层减少
  `array.spectrum.sources` 比继续微调 Bartlett 核心更有效。
- 当前主要剩余瓶颈是 `rt_solve`、`array_outputs` 和必要的 HDF5 写入。继续提升通常需要减少输出产品、
  改变 RT fidelity，或做更重的 spectrum 算法重构；这些会影响功能口径，不能作为低风险默认优化。
