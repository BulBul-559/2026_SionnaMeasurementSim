# HDF5 Bundle Append Benchmark 2026-06-17

本文记录实验性 `output.sharding.bundle.enabled` 落地后的第一轮 write-only 对照。
数据是 synthetic `MeasurementSimulationResult`，不代表真实 RT/PHY 全流程耗时；用途是隔离
HDF5 writer、bundle append、schema validate 和文件大小。

## 代码状态

- 分支：`codex/bundle-append-writer`
- 功能提交：`e4c3a77 Add experimental appendable HDF5 bundles`
- 本轮 benchmark 扩展：`benchmark write --bundle-shards`

## Benchmark 命令

小 CFR-only 对照：

```bash
uv run python -m sionna_measurement_sim.app.cli benchmark write \
  --output-dir outputs/benchmark_bundle_compare_repeat \
  --tx-count 1 --rx-count 1 --rx-ant 1 --tx-ant 1 \
  --subcarriers 8 \
  --bundle-shards 3 --bundle-max-planned-shards 2 \
  --compression mixed --gzip-level 1 \
  --warmup 1 --repeat 3 \
  --no-write-hardware-samples
```

稍重 waveform 对照：

```bash
uv run python -m sionna_measurement_sim.app.cli benchmark write \
  --output-dir outputs/benchmark_bundle_compare_waveform \
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

### CFR-Only 3 Shards

| Mode | writer_s mean | schema_validate_s mean | wall_time_s mean | file_count | file_size_bytes |
|---|---:|---:|---:|---:|---:|
| `shard_files` | 0.1276 | 0.0703 | 0.1981 | 3 | 652440 |
| `bundle_append` | 0.3971 | 0.0105 | 0.4078 | 2 | 423488 |

### Waveform 4 Shards

| Mode | writer_s mean | schema_validate_s mean | wall_time_s mean | file_count | file_size_bytes |
|---|---:|---:|---:|---:|---:|
| `shard_files` | 0.2190 | 0.1205 | 0.3397 | 4 | 1224192 |
| `bundle_append` | 0.7179 | 0.0108 | 0.7289 | 2 | 503792 |

## Interpretation

第一版 append bundle 的收益已经出现在文件数量、文件大小和 schema validate 时间上：

- bundle 文件数按 `bundle_max_planned_shards` 合并，本轮从 3/4 个 shard 文件降到 2 个 bundle。
- 因为 metadata、frequency、BS topology 等不再每 shard 重复写，bundle 文件大小明显下降。
- schema validate 从每个 shard 文件校验变为每个 bundle 文件校验，小样本下降约 6-11 倍。

但 writer 本体目前更慢。主要原因是 bundle v1 为了复用所有现有 writer contract，会先把每个
domain result 写成内存 HDF5 fragment，再从 fragment append 到 bundle 文件。这个二次序列化
开销在小到中等 synthetic 样本里超过了减少 metadata/group/dataset 创建的收益。

## Current Recommendation

- 默认生产路径继续保持一个 shard 一个 `results/result_xxx.h5`。
- `output.sharding.bundle.enabled=true` 保留为实验模式，用于训练读取和 metadata/validate 成本探索。
- 后续优化应优先减少内存 fragment 二次写入：
  - 让 shared writer group 函数能直接向 bundle writer 提供 dataset spec 或 array stream。
  - 为 bundle append path 定制 chunk shape 和 append batch，减少 resize 次数。
  - 在真实 shard 输出上比较 end-to-end `hdf5_bundle_append`、`hdf5_write`、`schema_validate` 和训练 loader 读取吞吐。
