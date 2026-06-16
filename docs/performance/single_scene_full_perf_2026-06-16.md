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
| mixed + matmul projection | 4 / `[2,3,5,6]` | `mixed` | 326.6 s | 9.3 GiB | 已放弃；BLAS 多线程竞争导致正式运行变慢 |

## Stage Totals

| Stage | gzip baseline | mixed stable | mixed + covariance reuse |
|---|---:|---:|---:|
| `hdf5_write` | 364.7 s | 173.1 s | 171.1 s |
| `rt_solve` | 305.9 s | 321.5 s | 329.2 s |
| `array_outputs` | 215.8 s | 229.4 s | 224.5 s |
| `nr_srs_observation` | 70.5 s | 79.0 s | 79.3 s |
| `visualization` | 64.1 s | 65.4 s | 69.7 s |
| `schema_validate` | 30.2 s | 25.5 s | 25.5 s |
| `ranging` | 23.5 s | 23.6 s | 24.7 s |

## HDF5 Compression Finding

`/waveform/rx_grid` 是原 gzip run 最大写入热点：

| Dataset | gzip write time | mixed write time | gzip storage/raw | mixed storage/raw |
|---|---:|---:|---:|---:|
| `/waveform/rx_grid` | 159.5 s | 8.7-11.2 s | 0.894 | 1.000 |

结论：noisy/high-entropy complex observation grid 用 gzip 只节省少量空间，却消耗大量 CPU。
新增 `output.compression: "mixed"` 后，路径表和稀疏/结构化数组仍使用 gzip + shuffle，高熵复数观测
数组不压缩。该策略不改变数值和 HDF5 schema，但本场景文件体积约增加 0.7 GiB。

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

## Stability Notes

一次 6-worker mixed run 在当时 GPU 0/1/4 已被其他进程高占用时出现 Dr.Jit/OptiX OOM stderr，
最终未形成完整 57-shard manifest。该结果不用于性能结论。正式对照使用 4 workers 稳定完成，
fallback split count 为 0。

## Current Recommendation

- 正式 64 PRB full 仿真优先使用 `output.compression: "mixed"`。
- 若机器 GPU 内存被外部任务占用，降低 `parallel_workers` 或避开高占用 GPU；不要把 OOM fallback
  运行的墙钟作为性能基线。
- 若用户不需要 `/array/spatial_spectrum_observation`，从配置层减少
  `array.spectrum.sources` 比继续微调 Bartlett 核心更有效。
