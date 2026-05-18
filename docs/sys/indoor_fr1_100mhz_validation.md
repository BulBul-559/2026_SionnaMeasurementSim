# Indoor FR1 100 MHz Validation Notes

本页记录 `bistro_0000` 室内 FR1 100 MHz 模板的真实场景验证结果和全量成本估算。
这些结论来自 2026-05-17 在 RTX 4090 上的 probe run。

2026-05-18 之后，SRS-like 模板默认 direct uplink、`rt.synthetic_array=false`。
这会显著增加 Sionna RT `PathSolver` 的底层显存需求；新的 RT 参数 sweep 见
`docs/performance/nr_srs_rt_variant_sweep_6x5.md`。后续 uplink scale sweep 已验证
`7 BS x 30 UE` 单 shard 可运行、`7 BS x 35 UE` 会在 PathSolver 阶段 OOM，
后续 `medium_0000` 的 `label0p2` 全量 baseline 又验证了 `shard_size=25` 会在
`paths.cfr()` 阶段触发 Dr.Jit 单数组 entry 数超过 `2^32` 的限制。因此默认
SRS-like 生产模板使用 `output.sharding.shard_size=20`。

## 场景规模

`data/bistro_0000/label.json` 当前包含：

| 项 | 数量 |
|---|---:|
| BS / TRP points | 7 |
| UE sample positions | 2583 |

当前 SRS-like indoor 模板默认使用 `max_bs: 7`、`max_ue: 2500`，因此完整
`bistro_0000` 生产口径约为 `7 BS x 2500 UE`。模板中的 `max_ue: 2500` 是目标数据规模，
不是提交前必跑验收规模。

## 历史已跑通的真实场景 probe

下表记录的是 2026-05-17 的真实场景 probe。SRS-like 这一行发生在模板默认
`synthetic_array=false` 之前，因此只能作为 SRS-like 链路功能和输出字段验证，
不能证明当前 `synthetic_array=false` 口径可以直接跑普通 `6 BS x 5 UE` UE shard。
当前 SRS RT 参数 sweep 和 OOM 结论见
`docs/performance/nr_srs_rt_variant_sweep_6x5.md`。

两条链路都使用：

- `max_bs=6`
- `max_ue=5`
- `output.sharding.shard_size=5`
- `array.spectrum.enabled=true`
- `visualization.enabled=true`
- `output.save_full_paths=false`
- `runtime.device="cuda"`

| 链路 | 输出目录 | aggregate elapsed | shell real time | HDF5 | 目录总量 | schema | 备注 |
|---|---|---:|---:|---:|---:|---|---|
| NR PUSCH-DMRS CSI proxy | `outputs/bistro_0000_pusch_shard5_probe` | 526.9 s | 546.1 s | 248.7 MB | 255 MB | pass | batch size 16, 30 links |
| NR SRS-like full-band sounding | `outputs/bistro_0000_srs_shard5_probe` | 503.6 s | 522.3 s | 132.0 MB | 143 MB | pass | 30 links，历史 synthetic-array 口径 |

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

direct uplink 语义完成后，`UE` 是 source，`BS` 是 target。当前 Bistro RT 配置下，
实测 `7x30` 可运行、`7x35` 和 `7x40` 在 PathSolver/Dr.Jit 阶段 OOM。因此当前
SRS-like 生产模板将 `output.sharding.shard_size` 设为 20，并且默认关闭空间谱与
可视化，避免生产模板直接触发重型派生输出。`medium_0000 label0p2` 的
`7 BS x 2583 UE` baseline 中，`shard_size=25` 在后段 shard 的 `paths.cfr()` 触发
Dr.Jit `2^32` entry 上限；改为 `shard_size=20` 后完成 130 个 shard。

`nr_mimo_channel.py` 默认按 independent link chunk 调用 `cir_to_ofdm_channel`。
可用环境变量 `SIONNA_CIR_TO_CFR_LINK_CHUNK_SIZE` 调整 link chunk 大小。若后续场景
路径更复杂、BS 数更多、或打开空间谱/可视化，应重新做 shard sweep，不要直接假设
`shard_size=20` 一定仍然安全。

## 全量成本估算

按当前 scale sweep 估算，在关闭空间谱/可视化、保存完整 truth CFR 与 SRS 输出时，
`7x2500` 的总 HDF5 约 46 GiB。`shard_size=20` 时约 125 个 shard；单 GPU 串行约
59 分钟，4/5 GPU 理想调度约 15/12 分钟。实际耗时会受 schema validation、HDF5
压缩、磁盘并发写入、GPU 负载和最后一个 shard 不满块影响。确认测试见
`config/perf/nr_srs_7x500_sharded.yaml`。

## 当前建议

- 本轮不要把 `2500 UE` 作为提交前必跑验收；使用 `7x500` shard confirmation
  确认多 GPU shard 语义和估算。
- 若 `rt.synthetic_array=false`，不要只按 link 数估算 RT 风险；direct uplink 下
  source 数量即 `UE_block` 更关键。当前保守生产值为 `UE_block=20`。
- 当前主线已经使用 BS/UE role-view 到 TX/RX link-view 的 direct mapping。
  若 `phy_link_direction="uplink"`，RT source 是 UE，receiver 是 BS；如果后续仍要
  高保真非合成阵列生产数据，优先做二维 BS/UE shard 和 RT 参数 sweep。
- 后续如果要跑 `1000 UE` 或完整 `2583 UE`，建议先开启 `debug.enabled=true`，
  记录阶段耗时和硬件峰值。
- 若要把 100 MHz 模板作为论文生产数据生成路径，优先优化：
  - 减少每 shard 的 schema validation 重复读全文件成本。
  - 对 HDF5 压缩策略做 `gzip/lzf/none` 对比。
  - 对空间谱输出做可选降分辨率或只保留需要的 source。
  - 使用多 GPU shard 并行，但每个进程仍写独立 `result_xxx.h5`。
