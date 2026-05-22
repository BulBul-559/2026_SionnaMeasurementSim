# NR SRS Direct Uplink Scale Sweep Round 2 BS Results

日期：2026-05-18

分支：`codex/uplink-scale-sweep`

状态更新：本文是 direct-uplink BS sweep 的历史结果记录，保留用于解释 BS 维度扩大时
RT、HDF5 写入和 `synthetic_array` 的历史行为。当前生产默认不从本文推导；SRS baseline
和推荐 shard 参数见 `docs/sys/indoor_fr1_100mhz_validation.md`，active 优化项见
`docs/todo/performance.md` 和 `docs/todo/structure.md`。

本文记录第二轮 **固定 UE=20，扫描 BS 数量** 的真实 UE->BS uplink 规模测试。测试目标是回答：

1. 在 direct uplink 语义下，BS 数量是否比 UE 数量更宽松。
2. `synthetic_array=false` 和 `synthetic_array=true` 在固定 UE=20 时的可运行 BS 上限。
3. BS 数增加时，RT、SRS、HDF5 写入和文件大小的增长趋势。

原始输出：

```text
outputs/uplink_scale_sweep_round2_bs_sweep/
outputs/uplink_scale_sweep_round2_bs_sweep/summary.csv
outputs/uplink_scale_sweep_round2_bs_sweep/summary.json
```

## 测试口径

| 项 | 值 |
|---|---|
| 场景 | `data/bistro_0000` |
| PHY | `phy.standard="nr_srs"` |
| 链路方向 | `phy_link_direction="uplink"`，TX=UE，RX=BS |
| UE 数量 | 固定 `20` |
| BS 数量 | `1,2,4,6,7,8,12,16,24,32` |
| 频域 | 100 MHz 模板，`3276` active subcarriers |
| 天线 | BS `4x4=16`，UE `1x2=2` |
| RT 参数 | 模板默认：`max_depth=4`，refraction/diffraction enabled |
| shard | disabled；每个 job 是一个完整 batch |
| visualization | disabled |
| array spectrum | disabled |
| save full paths | false |
| debug profiling | enabled |
| GPU | GPU 3-7 并行执行 |

BS 来源：

- `BS=1,2,4,6,7` 使用 `data/bistro_0000/label.json` 中真实 BS。
- `BS=8,12,16,24,32` 保留前 7 个真实 BS，再按固定 seed `20260518` 在 UE 区域内补充 generated BS。

所有 PASS job 均通过 HDF5 schema 校验，并满足：

```text
/link/tx_role = "ue"
/link/rx_role = "bs"
/link/phy_link_direction = "uplink"
/waveform/tx_time 不存在
/waveform/rx_time 不存在
```

## 结果表

| Job | BS | UE | Links | synthetic_array | Status | Wall time | RT solve | SRS obs | HDF5 write | HDF5 | 失败原因 |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---|
| B01 | 1 | 20 | 20 | false | PASS | 9.96 s | 3.70 s | 0.23 s | 2.23 s | 66.89 MB | |
| B02 | 2 | 20 | 40 | false | PASS | 14.01 s | 5.26 s | 0.44 s | 4.09 s | 122.57 MB | |
| B03 | 4 | 20 | 80 | false | PASS | 20.25 s | 6.17 s | 0.93 s | 8.77 s | 226.38 MB | |
| B04 | 6 | 20 | 120 | false | PASS | 25.37 s | 6.29 s | 1.47 s | 12.65 s | 325.89 MB | |
| B05 | 7 | 20 | 140 | false | PASS | 30.34 s | 7.98 s | 1.68 s | 14.97 s | 380.84 MB | |
| B06 | 8 | 20 | 160 | false | PASS | 32.32 s | 8.29 s | 1.87 s | 16.77 s | 435.44 MB | |
| B07 | 12 | 20 | 240 | false | PASS | 42.62 s | 9.89 s | 2.85 s | 23.62 s | 591.47 MB | |
| B08 | 16 | 20 | 320 | false | PASS | 53.29 s | 11.52 s | 3.74 s | 31.15 s | 789.68 MB | |
| B09 | 24 | 20 | 480 | false | PASS | 77.04 s | 15.31 s | 5.68 s | 46.55 s | 1094.13 MB | |
| B10 | 32 | 20 | 640 | false | PASS | 101.43 s | 18.71 s | 7.61 s | 62.84 s | 1412.37 MB | |
| B11 | 1 | 20 | 20 | true | PASS | 15.40 s | 8.36 s | 0.23 s | 3.09 s | 102.42 MB | |
| B12 | 2 | 20 | 40 | true | PASS | 19.44 s | 9.02 s | 0.43 s | 5.60 s | 188.33 MB | |
| B13 | 4 | 20 | 80 | true | FAIL | 6.59 s | n/a | n/a | n/a | n/a | Dr.Jit tensor entries 超过 `2^32`：`5367398400 > 4294967296` |
| B14 | 6 | 20 | 120 | true | FAIL | 5.33 s | n/a | n/a | n/a | n/a | Dr.Jit tensor entries 超过 `2^32`：`6717634560 > 4294967296` |
| B15 | 7 | 20 | 140 | true | FAIL | 6.25 s | n/a | n/a | n/a | n/a | Dr.Jit tensor entries 超过 `2^32`：`8306887680 > 4294967296` |
| B16 | 8 | 20 | 160 | true | FAIL | 6.39 s | n/a | n/a | n/a | n/a | Dr.Jit tensor entries 超过 `2^32`：`9443266560 > 4294967296` |
| B17 | 12 | 20 | 240 | true | FAIL | 6.67 s | n/a | n/a | n/a | n/a | Dr.Jit tensor entries 超过 `2^32`：`12152125440 > 4294967296` |
| B18 | 16 | 20 | 320 | true | FAIL | 5.56 s | n/a | n/a | n/a | n/a | Dr.Jit tensor entries 超过 `2^32`：`13552680960 > 4294967296` |
| B19 | 24 | 20 | 480 | true | FAIL | 6.57 s | n/a | n/a | n/a | n/a | Dr.Jit tensor entries 超过 `2^32`：`16303472640 > 4294967296` |
| B20 | 32 | 20 | 640 | true | FAIL | 5.99 s | n/a | n/a | n/a | n/a | Dr.Jit tensor entries 超过 `2^32`：`15498362880 > 4294967296` |

典型 shape：

| Job | `/channel/truth/cfr` | `/observation/cfr_est` |
|---|---|---|
| B01 | `[20, 1, 16, 2, 3276]` | `[1, 20, 1, 16, 2, 3276]` |
| B10 | `[20, 32, 16, 2, 3276]` | `[1, 20, 32, 16, 2, 3276]` |
| B12 | `[20, 2, 16, 2, 3276]` | `[1, 20, 2, 16, 2, 3276]` |

## 关键结论

1. **在 `synthetic_array=false` 下，BS 数量确实比 UE 数量宽松。**

   第一轮固定 `BS=6` 时，`UE=20` 成功、`UE=50` 失败。本轮固定 `UE=20` 时，
   `BS=32` 仍成功。因此在当前 direct uplink 口径里，更危险的维度是 UE 数量，
   不是 BS 数量。

2. **`synthetic_array=false` 的测试上限至少达到 `32 BS x 20 UE`。**

   本轮没有测到 false 的 BS 失败点。`B10=32 BS x 20 UE` 成功，HDF5 schema 通过，
   wall time `101.43 s`，HDF5 体积 `1412.37 MB`。

3. **`synthetic_array=true` 在 `UE=20` 时只通过到 `BS=2`。**

   `B11=1 BS` 和 `B12=2 BS` 成功；`B13=4 BS` 开始已经在 `paths.cfr()` 阶段触发
   Dr.Jit 单数组 entry 数超过 `2^32` 的限制。这个失败不是普通显存 OOM，而是底层
   张量规模上限。

4. **BS 数增加后，HDF5 写入是最明显的时间占比。**

   `synthetic_array=false` 下，`B10` 的阶段耗时为：

   ```text
   total          95.91 s  (perf summary)
   rt_solve       18.71 s
   nr_srs_obs      7.61 s
   hdf5_write     62.84 s
   schema          6.01 s
   ```

   也就是说，在关闭可视化和空间谱后，SRS 本体不是主要瓶颈；完整保存
   truth CFR、SRS grids 和估计 CFR 的写盘成本会随着 `BS x UE` 线性放大。

## 增长趋势

对 `synthetic_array=false` 的 PASS 样本做近似线性拟合，固定 `UE=20` 时：

```text
wall time     约 +2.89 s / BS
rt_solve      约 +0.46 s / BS
nr_srs_obs    约 +0.24 s / BS
hdf5_write    约 +1.93 s / BS
HDF5 size     约 +43.16 MB / BS
```

按 link 粗略平均：

```text
wall time     约 0.24 s / link
nr_srs_obs    约 0.012 s / link
hdf5_write    约 0.10 s / link
HDF5 size     约 2.68 MB / link
```

这些数字只适用于当前输出契约：关闭空间谱/可视化、保存 truth CFR 和 SRS 估计结果、
100 MHz、3276 active subcarriers、BS 16 天线、UE 2 天线。

## 配置建议

1. **大规模 direct uplink SRS 上限探索优先使用 `synthetic_array=false`。**

   在当前 Bistro 场景和 100 MHz 模板下，`synthetic_array=false` 的 `UE=20` 可以承受
   至少 `BS=32`；而 `synthetic_array=true` 在 `BS=4, UE=20` 已经触发 Dr.Jit entry 上限。

2. **生产 shard 的第一版建议先按 UE 控制块大小。**

   第一轮和第二轮合起来说明：`BS<=32, UE=20` 比较稳，但 `BS=6, UE=50` 已失败。
   因此当前更稳妥的初始 block 是：

   ```text
   ue_block_size = 20
   bs_block_size = 6..32
   ```

   如果希望每个 HDF5 文件更小、更快写完，建议先用：

   ```text
   ue_block_size = 20
   bs_block_size = 8..16
   ```

   对应本轮文件大小约 `435 MB .. 790 MB`，wall time 约 `32 s .. 53 s`。

3. **如果保留 32 BS block，需要接受单文件约 1.4 GB。**

   `32 BS x 20 UE` 在计算上通过，但当前完整输出会生成约 `1.4 GB` HDF5，写盘约
   `63 s`。如果后续加入空间谱或可视化，这个 block 会明显变重。

4. **后续仍需要二维 shard，而不是只做 UE shard。**

   虽然本轮 BS 维度更宽松，但输出体积和写盘成本仍按 `BS x UE` 增长。生产全场数据时，
   建议同时支持：

   ```text
   UE block shard
   BS block shard
   ```

   这样既能避开 Dr.Jit/CFR 张量上限，也能把单个 HDF5 文件控制在可管理大小。

## 下一步建议

1. **继续对 `synthetic_array=false` 做 UE 二分。**

   当前真正的失败边界在 UE 侧。建议补：

   ```text
   BS=6, UE=25/30/40, synthetic_array=false
   ```

2. **如果要找 BS 上限，再测更大的 false BS。**

   本轮 `BS=32, UE=20` 成功。如果需要硬上限，可以继续测：

   ```text
   BS=48/64, UE=20, synthetic_array=false
   ```

   但这会优先放大写盘成本，建议先明确是否需要保存完整 truth CFR。

3. **把 write-only benchmark 和输出裁剪作为正式优化项。**

   成功样本中 HDF5 写入已经超过 RT 和 SRS 本体。若实验主要使用 path 信息和
   `cfr_est`，应提供配置开关裁剪 truth CFR、grids、空间谱和可视化，避免把生产成本花在
   暂时不用的字段上。
