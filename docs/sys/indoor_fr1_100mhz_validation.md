# Indoor FR1 100 MHz Validation Notes

本页记录 `bistro_0000` 室内 FR1 100 MHz 模板的真实场景验证结果和全量成本估算。
这些结论来自 2026-05-17 在 RTX 4090 上的 probe run。

2026-05-18 之后，SRS-like 模板默认 `rt.synthetic_array=false`。这会显著增加
Sionna RT `PathSolver` 的底层显存需求；新的 RT 参数 sweep 见
`docs/performance/nr_srs_rt_variant_sweep_6x5.md`。

## 场景规模

`data/bistro_0000/label.json` 当前包含：

| 项 | 数量 |
|---|---:|
| BS / TRP points | 7 |
| UE sample positions | 2583 |

当前 indoor 模板默认使用 `max_tx: 6`，因此完整 `bistro_0000` 运行口径是
`6 BS x 2583 UE`。模板中的 `max_rx: 1000` 是阶段性目标规模，不是本轮已完成验收规模。

## 已跑通的真实场景 probe

两条链路都使用：

- `max_tx=6`
- `max_rx=5`
- `output.sharding.shard_size=5`
- `array.spectrum.enabled=true`
- `visualization.enabled=true`
- `output.save_full_paths=false`
- `runtime.device="cuda"`

| 链路 | 输出目录 | aggregate elapsed | shell real time | HDF5 | 目录总量 | schema | 备注 |
|---|---|---:|---:|---:|---:|---|---|
| NR PUSCH-DMRS CSI proxy | `outputs/bistro_0000_pusch_shard5_probe` | 526.9 s | 546.1 s | 248.7 MB | 255 MB | pass | batch size 16, 30 links |
| NR SRS-like full-band sounding | `outputs/bistro_0000_srs_shard5_probe` | 503.6 s | 522.3 s | 132.0 MB | 143 MB | pass | 30 links |

关键 shape：

| Dataset | PUSCH shape | SRS-like shape |
|---|---|---|
| `/channel/truth/cfr` | `[6, 5, 2, 16, 3276]` | `[6, 5, 2, 16, 3276]` |
| `/observation/cfr_est` | `[1, 6, 5, 2, 16, 3276]` | `[1, 6, 5, 2, 16, 3276]` |
| `/observation/srs_cfr_est` | n/a | `[1, 6, 5, 2, 16, 3276]` |
| `/waveform/rx_grid` | `[1, 5, 6, 16, 14, 3276]` | n/a |
| `/waveform/srs_rx_grid` | n/a | `[1, 5, 6, 16, 2, 3276]` |
| `/array/spatial_spectrum_truth` | `[1, 5, 6, 91, 181]` | `[1, 5, 6, 91, 181]` |
| `/array/spatial_spectrum_cfr_est` | `[1, 5, 6, 91, 181]` | n/a |
| `/array/spatial_spectrum_srs` | n/a | `[1, 5, 6, 91, 181]` |

两条链路均未写 `/waveform/tx_time` 或 `/waveform/rx_time`。

同一组 probe 使用 `scripts/compare_phy_csi_outputs.py` 对比：

| 链路 | links | valid rate | median NMSE | p95 NMSE |
|---|---:|---:|---:|---:|
| PUSCH-DMRS CSI proxy | 30 | 1.0000 | 19.48 dB | 36.46 dB |
| SRS-like full-band sounding | 30 | 1.0000 | -29.99 dB | -29.97 dB |

这说明当前 100 MHz 室内定位模板中，SRS-like full-band sounding 更适合作为 CSI
监督/定位基线；PUSCH-DMRS proxy 仍可用于链路功能验证和 proxy 对照，但若要写论文中的
定位 CSI 质量，需要进一步调 DMRS、MIMO layers、receiver 或改用标准 SRS。

## 显存与 shard 粒度

100 MHz 模板使用 `273 PRB x 12 = 3276` 个 active subcarrier。对 6 个 BS 同时生成
CIR/CFR 时，中间频域张量很大。

已观察到：

| 配置 | 结果 |
|---|---|
| 原始一次性 CIR to CFR，`6x25` | OOM，单次分配约 30.7 GiB |
| 原始一次性 CIR to CFR，`6x5` | OOM，单次分配约 13.2 GiB |
| chunked CIR to CFR，`6x5` | 通过，运行中采样 GPU memory 约 11.5 GiB |

因此当前模板将 `output.sharding.shard_size` 设为 5，并且
`nr_mimo_channel.py` 默认按 independent link chunk 调用 `cir_to_ofdm_channel`。
可用环境变量 `SIONNA_CIR_TO_CFR_LINK_CHUNK_SIZE` 调整 link chunk 大小。更大的 shard
是否可行需要单独做 batch/shard sweep，不能直接假设 25 或 1000 UE shard 能放进单张 4090。

## 全量成本估算

按 `6x5` probe 线性外推，`max_rx=1000` 需要 200 个 shard，完整 `bistro_0000`
的 2583 UE 需要 517 个 shard。实际耗时会受 schema validation、HDF5 压缩、可视化和
GPU/CPU/IO 负载影响，以下只作为规划估算。

| 规模 | 链路 | 单 GPU 顺序估算 | 4 GPU 理想估算 | HDF5 估算 | 目录总量估算 |
|---|---|---:|---:|---:|---:|
| `6x1000` | PUSCH | 29.3 h | 7.3 h | 49.7 GB | 约 51 GB |
| `6x1000` | SRS-like | 28.0 h | 7.0 h | 26.4 GB | 约 28.6 GB |
| `6x2583` | PUSCH | 75.7 h | 18.9 h | 128.6 GB | 约 132 GB |
| `6x2583` | SRS-like | 72.3 h | 18.1 h | 68.2 GB | 约 72 GB |

## 当前建议

- 本轮不要把 `1000 UE` 作为提交前必跑验收；成本已经接近单 GPU 一整天。
- 若 `rt.synthetic_array=false`，不要直接假设 `6 BS x 5 UE` 或 `6 BS x 1 UE`
  RX shard 能通过；当前 bistro 场景会在 RT 阶段触发 Dr.Jit OOM。短期可用
  `1 BS x 1 UE` micro-sweep 做参数对比，长期应实现二维 TX/RX shard。
- 后续如果要跑 `1000 UE` 或完整 `2583 UE`，建议先开启 `debug.enabled=true`，
  记录阶段耗时和硬件峰值。
- 若要把 100 MHz 模板作为论文生产数据生成路径，优先优化：
  - 减少每 shard 的 schema validation 重复读全文件成本。
  - 对 HDF5 压缩策略做 `gzip/lzf/none` 对比。
  - 对空间谱输出做可选降分辨率或只保留需要的 source。
  - 使用多 GPU shard 并行，但每个进程仍写独立 `result_xxx.h5`。
