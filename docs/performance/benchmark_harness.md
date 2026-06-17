# Benchmark Harness

本文档说明当前稳定 benchmark 入口。它们用于隔离性能瓶颈，不生成正式仿真数据集；
输出默认写到 ignored `outputs/`，不要提交生成结果。

## 命令

```bash
uv run python -m sionna_measurement_sim.app.cli benchmark rt ...
uv run python -m sionna_measurement_sim.app.cli benchmark write ...
uv run python -m sionna_measurement_sim.app.cli benchmark sharding ...
uv run python -m sionna_measurement_sim.app.cli benchmark spectrum ...
```

公共参数：

| 参数 | 说明 |
|---|---|
| `--output-dir` | benchmark 输出目录，默认在 `outputs/benchmark_*` |
| `--seed` | deterministic synthetic/RT seed |
| `--repeat` | 正式测量次数 |
| `--warmup` | warmup 次数；写入 rows，但 aggregate 不统计 |
| `--device` | 记录设备意图；RT/YAML 配置可进一步决定执行设备 |
| `--debug-hardware-interval-s` | hardware sampler 间隔 |
| `--write-hardware-samples` / `--no-write-hardware-samples` | 是否写 `hardware_samples*.csv` |
| `--summary-name` | JSON summary 文件名 stem，默认 `benchmark_summary` |

## 输出格式

每次运行写：

| 文件 | 内容 |
|---|---|
| `benchmark_summary.json` | 总入口，含参数、环境、iterations、aggregate、perf summary 和 artifact 路径 |
| `benchmark_rows.csv` | 每次 iteration 一行，便于后续画图或聚合 |
| `config_snapshot.json` | benchmark 参数快照 |
| `logs/perf_events*.jsonl` | debug span/event 日志 |
| `logs/perf_summary*.json` | status、stage totals、hardware summary、dataset write summary |
| `logs/hardware_samples*.csv` | 可选硬件采样 |

`benchmark_summary.json` 的顶层字段稳定为：

```text
benchmark_type
status
parameters
environment
iterations
aggregate
aggregate_by_write_mode
perf_summary
artifacts
```

## RT-Only

RT-only 复用现有 RT solve 能力，但不跑 PHY、ranging、visualization 或正式 HDF5 输出。
它适合做 max_depth、LoS/NLoS、reflection/refraction/diffraction、subcarrier 数量和场景规模 sweep。

示例：

```bash
uv run python -m sionna_measurement_sim.app.cli benchmark rt \
  --output-dir outputs/benchmark_rt_smoke \
  --label-file tests/fixtures/scenes/test/test5.json \
  --scene-file tests/fixtures/scenes/test/scene.xml \
  --max-bs 1 --max-ue 2 --num-subcarriers 8 --max-depth 1 \
  --repeat 3 --warmup 1
```

主要指标：

| 指标 | 说明 |
|---|---|
| `rt_solve_s` | RT solve span 耗时 |
| `path_count` | 采样路径数量汇总 |
| `los_rate` / `nlos_rate` | link-level LoS/NLoS 比例 |
| `truth_cfr_shape` / `truth_cfr_bytes` | truth CFR 形状和内存量级 |

`benchmark rt --config <yaml>` 会读取 YAML 的 input/carrier/rt/antenna/link 字段；
显式 CLI 参数才覆盖 YAML。

## Write-Only

Write-only 构造 synthetic `MeasurementSimulationResult`，直接测 HDF5 writer、compression
和 schema validate。它是后续 HDF5 写入深度优化的基础。`--compression` 支持
`gzip`、`lzf`、`none`、`mixed`；其中 `mixed` 用于评估正式 full 仿真中“路径表仍压缩、
高熵观测网格不压缩”的折中策略。`--gzip-level` 控制 gzip dataset 的压缩等级，
用于快速比较 level 1..9 的 CPU/体积折中。

示例：

```bash
uv run python -m sionna_measurement_sim.app.cli benchmark write \
  --output-dir outputs/benchmark_write_smoke \
  --tx-count 1 --rx-count 2 --rx-ant 2 --tx-ant 1 \
  --subcarriers 16 --snapshots 1 \
  --include-waveform --include-array --include-ranging \
  --compression mixed --gzip-level 1 --validate-schema
```

主要指标：

| 指标 | 说明 |
|---|---|
| `writer_s` | `write_measurement_result()` 耗时 |
| `schema_validate_s` | schema validation 耗时 |
| `file_size_bytes` | 生成 HDF5 文件大小 |
| `perf_summary.dataset_write_summary` | dataset raw/storage bytes、压缩比、最慢/最大 dataset |

### Bundle Append 对照

`benchmark write` 可选开启 append bundle 对照：

```bash
uv run python -m sionna_measurement_sim.app.cli benchmark write \
  --output-dir outputs/benchmark_bundle_compare \
  --tx-count 2 --rx-count 2 --rx-ant 2 --tx-ant 1 \
  --subcarriers 16 --snapshots 1 --include-waveform \
  --bundle-shards 4 --bundle-max-planned-shards 2 \
  --compression mixed --gzip-level 1 \
  --warmup 1 --repeat 3 \
  --no-write-hardware-samples
```

开启 `--bundle-shards N` 后，每个正式 iteration 会写两行：

| `write_mode` | 说明 |
|---|---|
| `shard_files` | N 个 synthetic shard 分别写为 `result_xxx.h5` 并逐个 validate |
| `bundle_append` | 同样 N 个 fragment 按 `--bundle-max-planned-shards` append 到 bundle HDF5 |

`benchmark_summary.json` 会额外写 `aggregate_by_write_mode`，便于比较 `writer_s`、
`schema_validate_s`、`file_count` 和 `file_size_bytes`。2026-06-17 结果见
`docs/performance/hdf5_bundle_append_benchmark_2026-06-17.md`：初版 bundle 已降低文件数、
文件大小和 validate 时间；后续 lightweight fragment recorder 优化去掉内存 HDF5 二次序列化后，
synthetic waveform 对照中的 bundle writer 本体和总 wall time 已快于 shard files。

## Real Sharding 对照

`benchmark sharding` 会跑真实的轻量 `run_rt_truth_pipeline()` 两遍，固定
`output_products=("cfr_truth",)` 并开启 UE shard，用同一组场景/label/RT 参数比较两种
写盘路径：

| `write_mode` | 说明 |
|---|---|
| `shard_files` | 默认生产路径：每个计算 shard 写一个自包含 `results/result_xxx.h5` |
| `bundle_append` | 实验路径：多个 shard fragment append 到 `bundles/bundle_workerxxx_yyy.h5` |

示例：

```bash
uv run python -m sionna_measurement_sim.app.cli benchmark sharding \
  --output-dir outputs/benchmark_sharding_smoke \
  --label-file tests/fixtures/scenes/test/test5.json \
  --scene-file tests/fixtures/scenes/test/scene.xml \
  --max-bs 1 --max-ue 3 --num-subcarriers 8 --max-depth 1 \
  --shard-size 1 --bundle-max-planned-shards 2 \
  --readback-dataset channel/truth/cfr \
  --compression mixed --gzip-level 1 \
  --no-write-hardware-samples
```

每个正式 iteration 会生成 `sharding_iter_000_shard_files/` 和
`sharding_iter_000_bundle_append/` 两个真实 pipeline 输出目录，并从各自
`manifest/manifest.json` 汇总：

| 指标 | 说明 |
|---|---|
| `rt_solve_s` | RT solve 总耗时 |
| `hdf5_write_s` | 默认 shard HDF5 写入 span |
| `hdf5_bundle_write_s` / `hdf5_bundle_append_s` | bundle 写入和 append span |
| `schema_validate_s` | schema validation 总耗时 |
| `planned_shard_count` / `fragment_count` | 计划 shard 数和实际 fragment 数 |
| `file_count` / `file_size_bytes` | HDF5 artifact 数量和总大小 |
| `dataset_write_count` | tracer 记录的 HDF5 dataset 写入次数 |
| `readback_s` / `readback_bytes` | 通过 `iter_manifest_dataset()` 读回指定 dataset 的耗时和字节数 |

2026-06-17 的第一轮真实 `cfr_truth` 对照见
`docs/performance/hdf5_bundle_real_sharding_benchmark_2026-06-17.md`。该结果显示 bundle
降低文件数、文件大小、dataset write event 数、schema validate 时间和 manifest readback
时间；小 payload 下 bundle writer 固定成本仍明显，且同进程 mode 顺序会让 RT warm cache
影响端到端 wall time。

## Spectrum-Only

Spectrum-only 直接调用 Bartlett 空间谱核心，不跑 RT/PHY。输入为 deterministic synthetic
complex samples，角度网格和 array orientation 口径与正式 pipeline 一致。

示例：

```bash
uv run python -m sionna_measurement_sim.app.cli benchmark spectrum \
  --output-dir outputs/benchmark_spectrum_smoke \
  --links 4 --rx-ant 4 --subcarriers 32 --ofdm-symbols 2 \
  --zenith-bins 9 --azimuth-bins 13 \
  --sources truth_cfr,cfr_est,rx_grid \
  --link-chunk-size 512
```

主要指标：

| 指标 | 说明 |
|---|---|
| `<source>_time_s` | 每个 source 的 Bartlett 计算耗时 |
| `<source>_shape` | 输出 spectrum shape |
| `output_bytes` | 所有 source 输出数组总字节数 |
| `chunk_count` | link chunk 数 |
| `finite_rate_min` | 输出 finite sanity |

## Debug Summary

`PerfTracer` 现在在成功和失败运行中都尽量写 `perf_summary*.json`。summary 包括：

- `status` / `exception`
- `stage_totals_s`
- `hardware_summary`
- `dataset_write_summary`
- `logs`

普通 pipeline 的 `link_log_interval` 当前不主动生成 link chunk；第一版 benchmark 和未来
PHY link batching 可通过 `record_link_chunk()` 写 `link_chunks*.csv`。
