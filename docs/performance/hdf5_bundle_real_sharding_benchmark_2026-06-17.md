# HDF5 Bundle Real Sharding Benchmark 2026-06-17

本文记录 `benchmark sharding` 在真实轻量 `cfr_truth` sharded pipeline 上的第一轮对照。
它补充 write-only synthetic 结果，用于观察默认 `results/result_xxx.h5` 与实验 append
bundle 在真实 manifest、schema validate 和 pipeline perf summary 下的行为。

## Code State

- 分支：`codex/bundle-append-writer`
- 对照入口：`benchmark sharding`
- 结果目录：`outputs/benchmark_sharding_real_cfr_2026_06_17`
- 运行提交：`5264efd Add real sharding bundle benchmark`

## Command

```bash
uv run python -m sionna_measurement_sim.app.cli benchmark sharding \
  --output-dir outputs/benchmark_sharding_real_cfr_2026_06_17 \
  --label-file tests/fixtures/scenes/test/test5.json \
  --scene-file tests/fixtures/scenes/test/scene.xml \
  --max-bs 1 --max-ue 3 --num-subcarriers 8 --max-depth 1 \
  --shard-size 1 --bundle-max-planned-shards 2 \
  --compression mixed --gzip-level 1 \
  --warmup 1 --repeat 3 \
  --no-write-hardware-samples
```

每个 iteration 先跑 `shard_files`，再跑 `bundle_append`。这会使第二个 mode 受 Sionna
RT/scene 级 warm cache 影响，所以本页不把 `wall_time_s` 或 `rt_solve_s` 当成 bundle
端到端提速证据；主要看 HDF5 写盘、schema validate、文件数、文件大小和 dataset write
事件。

## Results

正式 repeat 的 mean：

| Mode | wall time mean | RT solve mean | HDF5 write span | bundle append span | bundle write span | schema validate mean | files | bytes | dataset writes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `shard_files` | 20.3884 s | 7.2895 s | 0.0610 s | 0.0000 s | 0.0000 s | 0.0359 s | 3 | 288288 | 51 |
| `bundle_append` | 0.3462 s | 0.1189 s | 0.0000 s | 0.0695 s | 0.3095 s | 0.0113 s | 2 | 181992 | 15 |

Derived deltas on directly comparable write artifacts:

| Metric | Bundle vs shard files |
|---|---:|
| HDF5 file count | 2 vs 3 files, 33.3% fewer |
| File size | 181992 vs 288288 bytes, 36.9% smaller |
| Dataset write events | 15 vs 51 events, 70.6% fewer |
| Schema validate span | 0.0113 s vs 0.0359 s, 68.5% lower |
| Bundle writer span | 0.3095 s total bundle write, including 0.0695 s append |

## Interpretation

The first real pipeline benchmark confirms the expected structural benefits: append bundles reduce
metadata duplication, file count, schema validation work and dataset write event count. On this very
small CFR-only fixture, the bundle writer span is still larger than the default shard HDF5 writer
span because the fixed bundle setup/metadata cost dominates tiny payloads. That is acceptable for
the experimental mode, but it is not yet enough to replace the production default.

`wall_time_s` and `rt_solve_s` are intentionally treated as non-comparable in this run. The benchmark
runs both modes inside one Python process and always executes `shard_files` first, so `bundle_append`
benefits from warmed RT/scene caches. Future larger or isolated-process comparisons should either
alternate mode order or run modes in separate processes when judging end-to-end time.

## Current Recommendation

- Keep the production default as one computed shard per `results/result_xxx.h5`.
- Use `output.sharding.bundle.enabled=true` only for controlled experiments that benefit from fewer
  files and lower repeated metadata/schema validation overhead.
- Continue benchmark work on larger real scenes, training-loader read throughput, chunk shape/flush
  policy, and optional schema validate frequency before promoting bundle append beyond experimental.
