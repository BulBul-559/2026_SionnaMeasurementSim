# CFR Scene Dataset Offline Postprocess Benchmark 2026-06-29

## 背景

本轮新增离线 `postprocess-cfr-scenes` 基础设施，用于把已完成 CFR truth-only shard run
整理成一个新的 scene-level 训练数据集。它只读原始生产 shard，不修改原始 run：

```text
<output_root>/
  scenes/<scene_id>.h5
  assets/<scene_id>/
  manifest.json
  link_index.pt
  link_index.parquet
  room_index.pt
  room_index.parquet
```

单 scene H5 使用独立 contract `sionna_cfr_scene_dataset` / schema `0.2.0`。当前只支持
CFR truth-only；`/cfr` shape 为 `[link, rx_ant, tx_ant, subcarrier]`，并保留
BS/UE/TX/RX id、位置、valid/power/finite/source metadata、结构化 room metadata、
floorplan 和 label 资产。
默认 `/cfr` 不压缩，目的是把训练读取从 CPU gzip 解压中解放出来；轻量 metadata 仍可 gzip。

## 单场景 Smoke

输入：

```text
outputs/front3d_small_normal_3000_panel0p5_cfr_truth_only/runs/000507_front3d_1115_panel0p5_cfr_truth_srs64prb
```

命令：

```bash
uv run python -m sionna_measurement_sim.app.cli postprocess-cfr-scenes \
  --run-dir outputs/front3d_small_normal_3000_panel0p5_cfr_truth_only/runs/000507_front3d_1115_panel0p5_cfr_truth_srs64prb \
  --output-root outputs/postprocess_cfr_scene_smoke_front3d1115_rooms \
  --workers 1 \
  --overwrite \
  --cfr-compression none
```

结果：

| 指标 | 值 |
|---|---:|
| scene | `front3d_1115` |
| source shard 数 | 15 |
| link 数 | 3,113 |
| `/cfr` shape | `[3113, 16, 2, 768]` |
| valid link | 2,363 |
| room 数 | 9 |
| source results size | 233 MiB |
| scene dataset size | 585 MiB |
| command elapsed | 4.98 s |
| scene write elapsed | 3.62 s |
| `/cfr` compression | none |
| global indexes | `link_index.pt`, `link_index.parquet`, `room_index.pt`, `room_index.parquet` |
| assets | floorplan、label JSON/PNG、label report、meta copied and embedded |

## Worker 压力探测

输入 workload 固定为 3000 场景任务中的前 64 个 completed scene：

```text
outputs/front3d_small_normal_3000_panel0p5_cfr_truth_only
```

固定 workload 汇总：

| 指标 | 值 |
|---|---:|
| scene 数 | 64 |
| link 数 | 181,488 |
| valid link | 156,994 |
| source bytes | 17.23 GiB |
| output H5 bytes | 33.27 GiB |
| output du | 34 GiB / run |

注意：Linux page cache 会让 app wall time 明显低估真实落盘完成时间。第一轮 cold 8-worker
app-only 耗时 251.9 s；随后同一批 source 已进入 page cache，16-worker app-only 只需
22-24 s 量级，但这时还有大量 dirty pages 未刷盘。因此本表采用更保守的
`app elapsed + command 后立刻 sync` 口径。这里的 `sync` 仍是进程外全局刷盘近似，不是每个
HDF5 文件内部 fsync。

| workers | app elapsed | sync elapsed | app+sync | scenes/hour | links/s | output MiB/s | 结论 |
|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | 33.69 s | 135.13 s | 168.82 s | 1,365 | 1,075 | 202 | 可用，但 app 已把主要成本推给刷盘尾巴 |
| 16 | 22.51 s | 107.15 s | 129.66 s | 1,777 | 1,400 | 263 | 本轮最佳稳定点 |
| 24 | 20.10 s | 131.25 s | 151.35 s | 1,522 | 1,199 | 225 | app 更快，但总落盘变慢 |
| 32 | 18.28 s | 167.11 s | 185.39 s | 1,243 | 979 | 184 | 已过压，dirty page 刷盘尾巴显著放大 |

非 commit-aware 的 app-only 结果仍可用于观察 Python/HDF5 前端调度：

| workers | app elapsed |
|---:|---:|
| 8 | 33.69 s |
| 16 | 22.51 s |
| 24 | 20.10 s |
| 32 | 18.28 s |

## 结论

本机、本 workload、`/cfr` 不压缩场景下，推荐先用 `workers=16` 作为离线合并稳定值。
32 worker 虽然 app 阶段更快，但只是更快堆积 dirty pages，commit-aware 总耗时反而更差。

后续若要正式批量转换 2988 个 scene，建议：

- 保持原始 CFR truth-only shard run 只读，输出到新的 dataset root。
- 以 `workers=16` 起步，并同时监控 `/data` 空间和系统 dirty/writeback。
- 记录 app elapsed 与刷盘完成时间，避免只看 Python 进程退出时间。
- 若训练侧确认未压缩 scene H5 读取瓶颈解除，再考虑是否为存储成本引入 `lzf` 或分块压缩对照。
