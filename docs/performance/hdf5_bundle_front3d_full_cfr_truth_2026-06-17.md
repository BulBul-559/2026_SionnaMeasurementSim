# HDF5 Bundle Front3D Full CFR Truth 2026-06-17

本文记录 `front3d_0002` panel 0.5、64 PRB SRS、`full + products=["cfr_truth"]`
真实场景下，默认一 shard 一个 HDF5 与实验性 append bundle 的对照。记录目的主要是保留
`outputs/` 清理前的可追溯性能数据；本文是历史实验记录，不代表当前默认生产配置。

## Code State

- 分支：`codex/bundle-append-writer`
- HEAD：`1516a06`
- 当前默认生产路径：`results/result_xxx.h5`
- 实验路径：`output.sharding.bundle.enabled=true`
- baseline 输出目录：
  `outputs/front3d_0002_panel0p5_cfr_truth_only_srs64prb_6w`
- bundle 1 worker 输出目录：
  `outputs/front3d_0002_panel0p5_cfr_truth_only_srs64prb_bundle_1w`
- bundle 6 worker 输出目录：
  `outputs/front3d_0002_panel0p5_cfr_truth_only_srs64prb_bundle_6w`
- 临时 bundle 配置目录：
  `outputs/front3d_0002_panel0p5_bundle_compare_2026_06_17/configs`

## Scenario

配置源是 baseline 的 `run_config.yaml`。bundle 对照只修改输出目录、worker/GPU 数和
`output.sharding.bundle`；其余场景、PHY、RT 和输出产品沿用 baseline。

关键参数：

| Field | Value |
|---|---|
| label | `data/front3d_20/front3d_0002/label/label_panel_0p5.json` |
| scene | `data/front3d_20/front3d_0002/scene.xml` |
| `input.max_bs` / `input.max_ue` | `64` / `20000` |
| `output.profile` / `output.products` | `full` / `["cfr_truth"]` |
| compression | `mixed`, `gzip_level=1` |
| sharding | `axis=ue`, `shard_size=20`, fallback enabled |
| baseline workers / GPUs | `6`, `[0, 1, 2, 4, 5, 6]` |
| bundle 1w workers / GPUs | `1`, `[0]` |
| bundle 6w workers / GPUs | `6`, `[0, 1, 2, 4, 5, 6]` |
| bundle split | `max_planned_shards_per_bundle=10` |
| PHY | `nr_srs`, `num_prb=64`, `num_antenna_ports=2`, 768 subcarriers |
| antenna | BS `4x4`, UE `1x2` |
| RT | `max_depth=4`, los/specular/refraction/diffraction on, synthetic array off |

三组运行都规划了 15 个 shard。shard `010` 在三组运行中均 fallback split 为
`010_00` 和 `010_01`，最终都是 16 个结果 fragment，UE 覆盖一致。

## Commands

Bundle 1 worker：

```bash
uv run python -m sionna_measurement_sim.app.cli \
  --config outputs/front3d_0002_panel0p5_bundle_compare_2026_06_17/configs/bundle_1w.yaml \
  run-full \
  --output-dir outputs/front3d_0002_panel0p5_cfr_truth_only_srs64prb_bundle_1w
```

Bundle 6 worker：

```bash
uv run python -m sionna_measurement_sim.app.cli \
  --config outputs/front3d_0002_panel0p5_bundle_compare_2026_06_17/configs/bundle_6w.yaml \
  run-full \
  --output-dir outputs/front3d_0002_panel0p5_cfr_truth_only_srs64prb_bundle_6w
```

## Overall Results

| Run | Wall time | Workers | Planned shards | Result fragments | H5 files | H5 size | Output files | Output size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline shard 6w | `68.70 s` | 6 | 15 | 16 | 16 | `280.90 MiB` | 103 | `281.66 MiB` |
| bundle 1w | `197.71 s` | 1 | 15 | 16 | 2 | `266.83 MiB` | 9 | `267.05 MiB` |
| bundle 6w | `57.51 s` | 6 | 15 | 16 | 6 | `271.31 MiB` | 28 | `271.61 MiB` |

相对 baseline shard 6w，本次 bundle 6w：

| Metric | Bundle 6w vs baseline 6w |
|---|---:|
| Wall time | `57.51 s` vs `68.70 s`, about `16.3%` lower |
| H5 file count | `6` vs `16`, `62.5%` fewer |
| H5 artifact size | `271.31 MiB` vs `280.90 MiB`, about `3.4%` smaller |
| Output directory files | `28` vs `103`, `72.8%` fewer |

该 wall time 结果是一次真实场景运行结果。由于 RT solve、GPU 调度和缓存状态会影响端到端
耗时，不能仅凭本次结果断言 bundle 模式天然更快。

## Stage Totals

以下来自 `manifest/manifest.json` 的 `performance.stage_totals_s`。并行运行中这些 span
是跨 shard 或 worker 的累计时间，可能大于 wall time。

| Stage | baseline shard 6w | bundle 1w | bundle 6w |
|---|---:|---:|---:|
| `topology_load` | `0.21 s` | `0.03 s` | `0.21 s` |
| `rt_solve` | `159.88 s` | `175.31 s` | `174.64 s` |
| `derived_nlos` | `0.60 s` | `0.60 s` | `0.59 s` |
| `hdf5_write` | `9.35 s` | n/a | n/a |
| `hdf5_bundle_append` | n/a | `17.31 s` | `17.82 s` |
| `hdf5_bundle_write` | n/a | `194.41 s` | `194.34 s` |
| `schema_validate` | `3.32 s` | `2.97 s` | `2.99 s` |
| `manifest_write` | `0.012 s` | n/a | n/a |

Metric interpretation:

- `hdf5_write` 是 baseline 每个 shard 独立写 `result_xxx.h5` 的累计写盘 span。
- `hdf5_bundle_append` 是 bundle 中每个已计算 fragment 调用 `writer.append_result(...)`
  追加到 bundle HDF5 的累计 span，更适合与 `hdf5_write` 对比。
- `hdf5_bundle_write` 是 bundle 文件生命周期外层 span，包含打开 bundle、逐 shard
  RT solve、append、关闭文件等，不是纯 HDF5 写盘时间。它不应与 `hdf5_write` 直接比较。

## Fair 10-Shard Write Comparison

为了避免 fallback 小 shard 和最后一个 2-UE shard 拉偏均值，这里只比较相同 shard id
`000` 到 `009` 的写盘 span：

| Run | Span | Count | Sum | Mean per shard |
|---|---|---:|---:|---:|
| baseline shard 6w | `hdf5_write` | 10 | `6.63 s` | `0.663 s` |
| bundle 1w | `hdf5_bundle_append` | 10 | `13.43 s` | `1.343 s` |
| bundle 6w | `hdf5_bundle_append` | 10 | `12.09 s` | `1.209 s` |

按同一批 10 个完整 shard 的累计写入 span 估算，本次 bundle append 写入成本约为 baseline
独立 shard writer 的 `1.8x` 到 `2.0x`。

## Dataset Write Hotspots

最慢 dataset 都是 `/channel/truth/cfr`：

| Run | Dataset | Events | Raw bytes | Tracked storage bytes | Write duration |
|---|---|---:|---:|---:|---:|
| baseline shard 6w | `/channel/truth/cfr` | 16 | `609878016` | `291886241` | `7.86 s` |
| bundle 6w | `/channel/truth/cfr` | 16 | `609878016` | `536989639` | `16.42 s` |

这说明 bundle append 慢主要来自 CFR 主数组的 appendable dataset 写入，而不是 metadata。
baseline 对每个 shard 使用一次性 `create_dataset(data=array)`；bundle 对同一逻辑 dataset
执行 `resize` 后写 slice，并且使用 chunked layout、gzip 和 shuffle。当前通用 chunk 规则
没有针对 CFR payload 调优，因此 append 写入比独立 shard 文件慢。

## H5 Artifact File List

### Baseline shard 6w

| File | Bytes | MiB |
|---|---:|---:|
| `results/result_000.h5` | 16876682 | 16.09 |
| `results/result_001.h5` | 16963462 | 16.18 |
| `results/result_002.h5` | 21898612 | 20.88 |
| `results/result_003.h5` | 21795039 | 20.79 |
| `results/result_004.h5` | 21984304 | 20.97 |
| `results/result_005.h5` | 22966761 | 21.90 |
| `results/result_006.h5` | 25061578 | 23.90 |
| `results/result_007.h5` | 26181808 | 24.97 |
| `results/result_008.h5` | 20086683 | 19.16 |
| `results/result_009.h5` | 15807344 | 15.08 |
| `results/result_010_00.h5` | 13513800 | 12.89 |
| `results/result_010_01.h5` | 10359841 | 9.88 |
| `results/result_011.h5` | 18936674 | 18.06 |
| `results/result_012.h5` | 25544304 | 24.36 |
| `results/result_013.h5` | 14977509 | 14.28 |
| `results/result_014.h5` | 1590084 | 1.52 |

### Bundle 1w

| File | Bytes | MiB |
|---|---:|---:|
| `bundles/bundle_worker000_000.h5` | 199531666 | 190.29 |
| `bundles/bundle_worker000_001.h5` | 80259444 | 76.54 |

### Bundle 6w

| File | Bytes | MiB |
|---|---:|---:|
| `bundles/bundle_worker000_000.h5` | 54381552 | 51.86 |
| `bundles/bundle_worker001_000.h5` | 64887116 | 61.88 |
| `bundles/bundle_worker002_000.h5` | 67759553 | 64.62 |
| `bundles/bundle_worker003_000.h5` | 38995637 | 37.19 |
| `bundles/bundle_worker004_000.h5` | 42408646 | 40.44 |
| `bundles/bundle_worker005_000.h5` | 16052188 | 15.31 |

## Caveats

- `hdf5_bundle_write` 命名容易误导。它是 bundle 外层生命周期 span，不是纯写盘 span。
- `performance.dataset_write_summary.total_storage_bytes` 在 bundle 模式下可能因为
  appendable dataset 的 `get_storage_size()` 重复采样而大于真实文件大小。本文的文件大小
  以 filesystem `stat` 的 H5 artifact size 为准。
- bundle 1w 运行过程中出现一次 Dr.Jit allocation cache flush warning，会拖慢该轮
  单 worker 结果。
- 当前 bundle 切分规则是 worker-owned bundle：先按 `parallel_workers` 连续分配 planned
  shards，再按 `max_planned_shards_per_bundle` 在 worker 内切 bundle 文件。fallback 子 shard
  会 append 到当前 bundle，不按最终 fragment 数重新 rotate。

## Current Interpretation

本次 full Front3D 结果支持以下判断：

- append bundle 已能跑完整真实 `cfr_truth` 场景，且与 baseline 产生一致的 shard/UE 覆盖。
- bundle 明显减少 H5 文件数和输出目录文件数，并略微减少 H5 artifact 总体积。
- bundle 6w 本次 wall time 没有拖慢，反而低于 baseline 6w，但端到端速度仍需多轮隔离运行
  才能下结论。
- 当前 appendable CFR dataset 写入吞吐弱于独立 shard writer。后续优化应优先关注 CFR
  chunk shape、压缩策略、预分配、批量 append 和更清晰的 perf span 命名。
- 在这些写盘和 reader 优化完成前，生产默认仍应保持一个 shard 一个 `results/result_xxx.h5`；
  bundle 继续作为训练读取和 metadata/文件数优化的实验模式。
