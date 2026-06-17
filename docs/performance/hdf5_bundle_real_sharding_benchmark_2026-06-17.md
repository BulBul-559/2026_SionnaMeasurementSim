# HDF5 Bundle Real Sharding Benchmark 2026-06-17

本文记录 `benchmark sharding` 在真实轻量 `cfr_truth` sharded pipeline 上的第一轮对照。
它补充 write-only synthetic 结果，用于观察默认 `results/result_xxx.h5` 与实验 append
bundle 在真实 manifest、schema validate 和 pipeline perf summary 下的行为。

## Code State

- 分支：`codex/bundle-append-writer`
- 对照入口：`benchmark sharding`
- 结果目录：`outputs/benchmark_sharding_batch_readback_cfr_2026_06_17`
- 运行状态：`benchmark sharding` 已包含 manifest-aware batch CFR readback probe

## Command

```bash
uv run python -m sionna_measurement_sim.app.cli benchmark sharding \
  --output-dir outputs/benchmark_sharding_batch_readback_cfr_2026_06_17 \
  --label-file tests/fixtures/scenes/test/test5.json \
  --scene-file tests/fixtures/scenes/test/scene.xml \
  --max-bs 1 --max-ue 3 --num-subcarriers 8 --max-depth 1 \
  --shard-size 1 --bundle-max-planned-shards 2 \
  --readback-dataset channel/truth/cfr \
  --readback-batch-fragments 16 \
  --compression mixed --gzip-level 1 \
  --warmup 1 --repeat 3 \
  --no-write-hardware-samples
```

每个 iteration 先跑 `shard_files`，再跑 `bundle_append`。这会使第二个 mode 受 Sionna
RT/scene 级 warm cache 影响，所以本页不把 `wall_time_s` 或 `rt_solve_s` 当成 bundle
端到端提速证据；主要看 HDF5 写盘、schema validate、文件数、文件大小和 dataset write
事件。每次 pipeline 输出后，benchmark 还会通过 `iter_manifest_dataset_batches()` 读回
`/channel/truth/cfr`，记录 manifest-aware batch readback 指标。

## Results

正式 repeat 的 mean：

| Mode | wall time mean | pipeline span mean | RT solve mean | HDF5 write span | bundle append span | bundle write span | schema validate mean | readback mean | readback batches | files | bytes | dataset writes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `shard_files` | 21.4941 s | 21.4484 s | 7.5900 s | 0.0659 s | 0.0000 s | 0.0000 s | 0.0386 s | 0.0451 s | 1 | 3 | 288288 | 51 |
| `bundle_append` | 0.3927 s | 0.3684 s | 0.1088 s | 0.0000 s | 0.0729 s | 0.3271 s | 0.0140 s | 0.0238 s | 1 | 2 | 181992 | 15 |

Derived deltas on directly comparable write artifacts:

| Metric | Bundle vs shard files |
|---|---:|
| HDF5 file count | 2 vs 3 files, 33.3% fewer |
| File size | 181992 vs 288288 bytes, 36.9% smaller |
| Dataset write events | 15 vs 51 events, 70.6% fewer |
| Schema validate span | 0.0140 s vs 0.0386 s, 63.6% lower |
| Manifest batch readback span | 0.0238 s vs 0.0451 s, 47.2% lower |
| Bundle writer span | 0.3271 s total bundle write, including 0.0729 s append |

## Interpretation

The first real pipeline benchmark confirms the expected structural benefits: append bundles reduce
metadata duplication, file count, schema validation work and dataset write event count. The
manifest-aware batch readback probe also shows lower small-payload readback time for bundle outputs
because fewer HDF5 files and less repeated metadata are touched; both modes read the three fragments
as one logical batch. On this very small CFR-only
fixture, the bundle writer span is still larger than the default shard HDF5 writer span because the
fixed bundle setup/metadata cost dominates tiny payloads. That is acceptable for the experimental
mode, but it is not yet enough to replace the production default.

Readback numbers are still a smoke signal, not a training-loader throughput result: the payload is
only 192 bytes across three fragments, `readback_batch_count=1`, and the reader performs contract
validation. Larger tensors and real loader iteration are still needed before making a production
recommendation.

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
