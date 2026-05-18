# NR SRS Direct Uplink Scale Sweep Round 1

日期：2026-05-18

分支：`codex/uplink-scale-sweep`

本文记录 `bistro_0000` 室内 FR1 100 MHz SRS-like 链路在 **真实 UE→BS uplink**
语义下的第一轮规模上限测试。该轮测试基于 BS/UE role-view 到 TX/RX link-view
的语义重构：`phy_link_direction="uplink"` 时，HDF5 中 `/link/tx_role="ue"`、
`/link/rx_role="bs"`，`/channel/truth/cfr` 的 shape 为
`[ue, bs, bs_ant, ue_ant, subcarrier]`。

## 测试口径

固定条件：

| 项 | 值 |
|---|---|
| 场景 | `data/bistro_0000` |
| PHY | `phy.standard="nr_srs"` |
| 链路方向 | `phy_link_direction="uplink"`，TX=UE，RX=BS |
| BS 数量 | 固定 6 |
| 频域 | 100 MHz 模板，`3276` active subcarriers |
| 天线 | BS `4x4=16`，UE `1x2=2` |
| RT 参数 | 模板默认：`max_depth=4`，refraction/diffraction enabled |
| shard | disabled；每个 job 是一个完整 batch |
| visualization | disabled |
| array spectrum | disabled |
| save full paths | false |
| debug profiling | enabled |
| GPU | GPU 3-7 并行执行 |

输出目录：

```text
outputs/uplink_scale_sweep_round1/
```

汇总文件：

```text
outputs/uplink_scale_sweep_round1/summary.csv
outputs/uplink_scale_sweep_round1/summary.json
```

## 结果表

| Job | BS | UE | Links | synthetic_array | Status | Wall time | RT solve | SRS obs | HDF5 write | HDF5 | 失败原因 |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---|
| A01 | 6 | 1 | 6 | false | PASS | 6.54 s | 2.53 s | 0.07 s | 0.62 s | 16.77 MB | |
| A02 | 6 | 2 | 12 | false | PASS | 8.53 s | 3.63 s | 0.14 s | 1.11 s | 33.44 MB | |
| A03 | 6 | 5 | 30 | false | PASS | 10.41 s | 3.46 s | 0.33 s | 2.85 s | 82.36 MB | |
| A04 | 6 | 10 | 60 | false | PASS | 15.64 s | 5.07 s | 0.69 s | 5.58 s | 163.98 MB | |
| A05 | 6 | 20 | 120 | false | PASS | 26.69 s | 7.20 s | 1.41 s | 12.98 s | 325.88 MB | |
| A06 | 6 | 50 | 300 | false | FAIL | 5.27 s | n/a | n/a | n/a | n/a | Dr.Jit CUDA OOM，单次分配约 2.0 GiB |
| A07 | 6 | 100 | 600 | false | FAIL | 5.38 s | n/a | n/a | n/a | n/a | Dr.Jit CUDA OOM，单次分配约 1.0 GiB |
| A08 | 6 | 1 | 6 | true | PASS | 7.44 s | 3.48 s | 0.06 s | 0.74 s | 23.32 MB | |
| A09 | 6 | 5 | 30 | true | PASS | 13.73 s | 5.90 s | 0.33 s | 3.70 s | 118.81 MB | |
| A10 | 6 | 20 | 120 | true | FAIL | 5.64 s | n/a | n/a | n/a | n/a | Dr.Jit tensor entries 超过 `2^32`：`6742794240 > 4294967296` |
| A11 | 6 | 100 | 600 | true | FAIL | 4.92 s | n/a | n/a | n/a | n/a | Dr.Jit CUDA OOM，单次分配约 2.0 GiB |
| A12 | 6 | 500 | 3000 | true | FAIL | 5.04 s | n/a | n/a | n/a | n/a | Dr.Jit CUDA OOM，单次分配约 2.0 GiB |
| A13 | 6 | 1000 | 6000 | true | FAIL | 12.48 s | n/a | n/a | n/a | n/a | Dr.Jit CUDA OOM，单次分配约 4.0 GiB |

所有 PASS job 均通过 HDF5 schema 校验，且均满足：

```text
/link/tx_role = "ue"
/link/rx_role = "bs"
/link/phy_link_direction = "uplink"
/waveform/tx_time 不存在
/waveform/rx_time 不存在
```

典型 shape：

| Job | `/channel/truth/cfr` | `/observation/cfr_est` |
|---|---|---|
| A01 | `[1, 6, 16, 2, 3276]` | `[1, 1, 6, 16, 2, 3276]` |
| A05 | `[20, 6, 16, 2, 3276]` | `[1, 20, 6, 16, 2, 3276]` |
| A09 | `[5, 6, 16, 2, 3276]` | `[1, 5, 6, 16, 2, 3276]` |

## 关键结论

1. **真实 uplink 已突破旧口径的 `6 BS x 1 UE` OOM。**

   历史 `synthetic_array=false` 的旧 downlink/trace+transpose 口径中，`6x1`
   和 `6x5` 都会在 Sionna RT PathSolver 阶段 OOM。本轮 direct uplink 中，
   `synthetic_array=false` 的 `6x1`、`6x2`、`6x5`、`6x10`、`6x20`
   均成功。

2. **当前 `synthetic_array=false` 的单 batch 安全上限位于 `UE=20` 与 `UE=50`
   之间。**

   `6x20` 成功，`6x50` 和 `6x100` 在 PathSolver 阶段 OOM。下一轮应对
   `UE=25/30/40` 做二分搜索，找到更精确的上限。

3. **`synthetic_array=true` 在本轮 direct uplink 下并不更稳。**

   `6x1` 和 `6x5` 成功，但 `6x20` 已经在 `paths.cfr()` 阶段触发 Dr.Jit
   单数组 entry 数超过 `2^32` 的限制；`6x100/500/1000` 在 PathSolver 阶段
   OOM。这说明 synthetic array 在当前 100 MHz、3276 subcarrier、Bistro 几何和
   RT 参数下，并不能简单视为“大规模更安全”的开关。

4. **HDF5 写入已经是成功样本里的主要长尾之一。**

   `synthetic_array=false` 下从 `6x1` 到 `6x20`，HDF5 从 `16.77 MB` 线性增长到
   `325.88 MB`，`hdf5_write` 从 `0.62 s` 增长到 `12.98 s`。后续如果只做
   RT 上限探索，可考虑继续保留当前完整输出；如果要做更细的 RT-only 边界搜索，
   应增加只测 RT/CFR 而不写完整 HDF5 的 benchmark。

## 规模估算

在当前模板和关闭空间谱/可视化的条件下，`synthetic_array=false` 成功样本的 HDF5
体积近似线性：

```text
约 16.3 MB / UE   (BS=6, 3276 subcarrier, BS 16 ant, UE 2 ant)
```

`6x20` 的端到端 wall time 为 `26.69 s`，其中：

```text
rt_solve      7.20 s
nr_srs_obs    1.41 s
hdf5_write   12.98 s
schema        1.26 s
```

因此若直接用单 batch 扩到更大 UE，瓶颈会同时来自 Dr.Jit PathSolver/CFR 张量上限和
HDF5 写入体积。当前不能按 `6x20` 线性外推到 `6x100`，因为 `6x50` 已经失败。

## 下一步建议

1. **对 `synthetic_array=false` 做二分搜索。**

   推荐下一轮只跑：

   ```text
   6x25 false
   6x30 false
   6x40 false
   ```

   如果 `6x40` 失败而 `6x30` 成功，再补 `6x35`。

2. **对 `synthetic_array=true` 补 `6x10` 和 `6x15`。**

   当前 true 的边界在 `5 < UE < 20`，且失败原因不是单纯显存，而是
   `paths.cfr()` 张量 entry 上限。需要确认 true 的实用上限到底是 10 还是 15。

3. **开始设计二维 BS/UE block shard。**

   仅 UE shard 不能解决单 batch 中 BS 侧和 UE 侧共同放大的 RT 张量。对
   `synthetic_array=false`，当前建议以 `BS=6, UE<=20` 作为保守单 batch 上限；
   生产全量数据应优先考虑二维 block，例如：

   ```text
   bs_block_size = 6
   ue_block_size = 20
   ```

   或在 BS 数继续增加时使用：

   ```text
   bs_block_size = 1/2/3
   ue_block_size = 20
   ```

4. **把 RT-only benchmark 独立出来。**

   当前 run 会写完整 HDF5 并做 schema；这对生产链路是好事，但对 RT 上限搜索会让
   写盘成本参与结果。下一轮可以同时保留完整 run 和 RT-only run，用来区分
   PathSolver/CFR 上限和输出写盘成本。
