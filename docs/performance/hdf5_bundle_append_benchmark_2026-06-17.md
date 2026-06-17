# HDF5 Bundle Append Benchmark 2026-06-17

本文记录实验性 `output.sharding.bundle.enabled` 落地后的 write-only 对照，以及同日
lightweight fragment recorder 优化后的复测。真实 sharded pipeline follow-up 见
`docs/performance/hdf5_bundle_real_sharding_benchmark_2026-06-17.md`。
数据是 synthetic `MeasurementSimulationResult`，不代表真实 RT/PHY 全流程耗时；用途是隔离
HDF5 writer、bundle append、schema validate 和文件大小。

## 代码状态

- 分支：`codex/bundle-append-writer`
- 功能提交：`e4c3a77 Add experimental appendable HDF5 bundles`
- 本轮 benchmark 扩展：`benchmark write --bundle-shards`
- 后续优化：bundle writer 不再把 fragment 写成 h5py core-driver 内存 HDF5，而是复用现有
  writer 写入轻量 recorder；shared dataset 比对增加内存 cache，避免每个 fragment 反复从
  bundle HDF5 读小 metadata。

## Benchmark 命令

小 CFR-only 对照（复测输出目录为 `outputs/benchmark_bundle_compare_recorder_cache_cfr`）：

```bash
uv run python -m sionna_measurement_sim.app.cli benchmark write \
  --output-dir outputs/benchmark_bundle_compare_recorder_cache_cfr \
  --tx-count 1 --rx-count 1 --rx-ant 1 --tx-ant 1 \
  --subcarriers 8 \
  --bundle-shards 3 --bundle-max-planned-shards 2 \
  --compression mixed --gzip-level 1 \
  --warmup 1 --repeat 3 \
  --no-write-hardware-samples
```

稍重 waveform 对照（复测输出目录为 `outputs/benchmark_bundle_compare_recorder_cache_waveform`）：

```bash
uv run python -m sionna_measurement_sim.app.cli benchmark write \
  --output-dir outputs/benchmark_bundle_compare_recorder_cache_waveform \
  --tx-count 2 --rx-count 2 --rx-ant 2 --tx-ant 1 \
  --subcarriers 16 --snapshots 1 --include-waveform \
  --bundle-shards 4 --bundle-max-planned-shards 2 \
  --compression mixed --gzip-level 1 \
  --warmup 1 --repeat 2 \
  --no-write-hardware-samples
```

`--warmup 1` 很重要：第一次进入 perf span 可能包含 Python/Torch 探测开销，warmup 后的
aggregate 更适合比较 writer 行为。

## 结果

### Initial Bundle V1 Baseline

第一版 bundle 复用现有 writer 的方式是：先把每个 result 写成内存 HDF5 fragment，再从
fragment append 到 bundle 文件。该版本已减少文件数、文件大小和 schema validate 时间，
但 writer 本体更慢。

#### CFR-Only 3 Shards

| Mode | writer_s mean | schema_validate_s mean | wall_time_s mean | file_count | file_size_bytes |
|---|---:|---:|---:|---:|---:|
| `shard_files` | 0.1276 | 0.0703 | 0.1981 | 3 | 652440 |
| `bundle_append` | 0.3971 | 0.0105 | 0.4078 | 2 | 423488 |

#### Waveform 4 Shards

| Mode | writer_s mean | schema_validate_s mean | wall_time_s mean | file_count | file_size_bytes |
|---|---:|---:|---:|---:|---:|
| `shard_files` | 0.2190 | 0.1205 | 0.3397 | 4 | 1224192 |
| `bundle_append` | 0.7179 | 0.0108 | 0.7289 | 2 | 503792 |

### Recorder + Shared Cache Follow-Up

优化后，bundle fragment 不再经过 HDF5 二次序列化。writer helper 仍生成同样的 dataset
path、array 和 attrs，但目标是 bundle 私有的 in-memory recorder；最终只在 append 到 bundle
文件时触发真实 HDF5 写入。

#### CFR-Only 3 Shards

| Mode | writer_s mean | schema_validate_s mean | wall_time_s mean | file_count | file_size_bytes |
|---|---:|---:|---:|---:|---:|
| `shard_files` | 0.1267 | 0.0714 | 0.1983 | 3 | 652440 |
| `bundle_append` | 0.1865 | 0.0102 | 0.1969 | 2 | 421280 |

#### Waveform 4 Shards

| Mode | writer_s mean | schema_validate_s mean | wall_time_s mean | file_count | file_size_bytes |
|---|---:|---:|---:|---:|---:|
| `shard_files` | 0.2692 | 0.1216 | 0.3910 | 4 | 1224192 |
| `bundle_append` | 0.2445 | 0.0110 | 0.2556 | 2 | 500176 |

## Interpretation

append bundle 的稳定收益出现在文件数量、文件大小和 schema validate 时间上：

- bundle 文件数按 `bundle_max_planned_shards` 合并，本轮从 3/4 个 shard 文件降到 2 个 bundle。
- 因为 metadata、frequency、BS topology 等不再每 shard 重复写，bundle 文件大小明显下降。
- schema validate 从每个 shard 文件校验变为每个 bundle 文件校验，小样本下降约 6-11 倍。

lightweight recorder 去掉了 v1 的内存 HDF5 二次写入后，writer 本体明显改善：
CFR-only 小样本仍略慢于独立 shard writer，但总 wall time 已基本持平；稍重 waveform
样本中 bundle writer 本体和总 wall time 都快于 shard files。该结论仍只来自 synthetic
write-only benchmark，不能直接外推到真实 RT/PHY pipeline 或训练 loader 吞吐。

## Current Recommendation

- 默认生产路径继续保持一个 shard 一个 `results/result_xxx.h5`。
- `output.sharding.bundle.enabled=true` 继续保留为实验模式，用于训练读取和
  metadata/validate 成本探索；它还不替代默认生产路径。
- 真实 `cfr_truth` sharded pipeline 已通过 `benchmark sharding` 做第一轮对照：bundle
  降低文件数、文件大小、dataset write event 数和 schema validate 时间，但小 payload 下
  bundle writer 固定成本仍明显，且同进程 RT warm cache 会污染 end-to-end wall time。
- 下一步应在更大真实 shard 输出和训练 loader 上比较读取吞吐，并继续测试 chunk shape、
  append batch、flush 策略、mode order 隔离和 schema validate 开关。
