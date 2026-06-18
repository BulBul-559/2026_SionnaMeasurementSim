# Front3D CFR Truth Pipeline Perf Optimization 2026-06-18

本记录固化 2026-06-18 对 Front3D 0p5 CFR truth-only 队列的调度、fallback 和写盘优化 smoke
结果。实验输出位于 ignored `outputs/perfopt_*`，可在清理本地产物后用本文追溯结论。

## 背景

目标模板：

```text
config/tasks/nr_srs_64prb_cfr_truth_only.yaml
```

目标 index：

```text
data/front3d_full/indices/small_normal_3000_panel0p5_seed2026.jsonl
```

本轮代码路径新增或验证：

- `output.sharding.fallback.isolation_mode`
- `output.sharding.recycle_workers`
- `output.sharding.postprocess.async_write`
- `output.sharding.postprocess.max_pending_writes`
- `output.sharding.gpu_scheduler.cross_scene_pipeline`

所有 smoke 均为 `run-scene-index` 跨场景 shard pipeline，候选 GPU 为 `[0..7]`，
`free_memory_threshold=0.6`，`scan_interval_s=0.2`。运行期间部分 GPU 有其他任务占用，
因此数字主要用于同机同时间窗口的相对比较。

## 结果

| 输出目录 | 配置重点 | 场景 | H5 数 | 输出大小 | 总 wall time | 场景耗时 |
|---|---|---:|---:|---:|---:|---:|
| `outputs/perfopt_async_smoke_run` | `shard_size=20`, `async_write=true`, `recycle_workers=true` | 2 | 37 | 977,446,871 B | 142.57 s | `front3d_0001`: 70.58 s; `front3d_0003`: 100.63 s |
| `outputs/perfopt_syncwrite_smoke_run` | `shard_size=20`, `async_write=false`, `recycle_workers=true` | 1 | 14 | 287,534,622 B | 71.06 s | `front3d_0001`: 68.18 s |
| `outputs/perfopt_shard30_smoke_run` | `shard_size=30`, `async_write=false`, `recycle_workers=true` | 1 | 10 | 287,110,956 B | 109.42 s | `front3d_0001`: 106.68 s |

`front3d_0001` 的直接对照：

| 方案 | 计划/完成 shard | 场景耗时 | 结论 |
|---|---:|---:|---|
| `shard_size=20`, async write | 14 / 14 | 70.58 s | 可完成，但 prepared CFR payload 回传成本抵消写盘重叠收益 |
| `shard_size=20`, sync write | 14 / 14 | 68.18 s | 本轮最优，作为生产模板默认 |
| `shard_size=30`, sync write | 9 / 9 | 106.68 s | shard 变大导致并行度和尾部调度变差，不采用 |

## 结论

- CFR-only 生产模板保留 `shard_size=20`。
- `postprocess.async_write` 保持实验能力，但 `nr_srs_64prb_cfr_truth_only.yaml` 默认关闭。
- `fallback.isolation_mode="on_failure"` 可以减少成功 shard 的额外隔离层；必须配合
  `recycle_workers=true`，否则 Sionna RT / Dr.Jit GPU allocations 可能留在长生命周期
  worker 内，动态调度器会持续认为 GPU 不空闲。
- 跨场景 shard pipeline 可用，适合 3000 场景队列；当当前场景尾部 shard 不足时，会继续
  调度后续场景 shard。

最终推荐模板组合：

```yaml
output:
  sharding:
    shard_size: 20
    parallel_workers: 8
    recycle_workers: true
    gpu_scheduler:
      enabled: true
      free_memory_threshold: 0.6
      scan_interval_s: 0.2
      cross_scene_pipeline: true
    fallback:
      enabled: true
      isolation_mode: "on_failure"
    postprocess:
      async_write: false
      max_pending_writes: 16
```
