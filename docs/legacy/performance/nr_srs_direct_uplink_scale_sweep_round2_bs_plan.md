# NR SRS Direct Uplink Scale Sweep Round 2 BS Plan

日期：2026-05-18

分支：`codex/uplink-scale-sweep`

Legacy note：本文是已执行完毕的历史计划，已被
`docs/performance/nr_srs_direct_uplink_scale_sweep_round2_bs_results.md` 取代。保留在
`docs/legacy/performance/` 仅供人工复核当时的实验矩阵设计。

本文定义第二轮 **固定 UE 数量、扫描 BS 数量** 的测试矩阵。第一轮已经证明在
真实 UE→BS uplink 语义下，`synthetic_array=false` 的 `6 BS x 20 UE` 可以通过，
`6 BS x 50 UE` 会在 Sionna RT/Dr.Jit 阶段失败。因此本轮暂不做 UE 二分，先固定：

```text
UE_ref = 20
```

然后扫描 BS 数量，观察 BS 侧规模对 PathSolver/CFR 张量、HDF5 体积和耗时的影响。

## 固定口径

| 项 | 值 |
|---|---|
| 场景 | `data/bistro_0000` |
| PHY | `phy.standard="nr_srs"` |
| 链路方向 | `phy_link_direction="uplink"`，TX=UE，RX=BS |
| UE 数量 | 固定 `20` |
| 频域 | 100 MHz 模板，`3276` active subcarriers |
| 天线 | BS `4x4=16`，UE `1x2=2` |
| RT 参数 | 模板默认：`max_depth=4`，refraction/diffraction enabled |
| shard | disabled；每个 job 是一个完整 batch |
| visualization | disabled |
| array spectrum | disabled |
| save full paths | false |
| debug profiling | enabled |
| GPU | 仍优先使用 GPU 3-7 并行执行 |

## BS 点生成规则

真实 label 中最多有 7 个 BS。为了测试更大 BS 数，本轮规则为：

1. `BS=1,2,4,6,7`：直接使用 `data/bistro_0000/label.json` 中前 N 个真实 BS。
2. `BS=8,12,16,24,32`：先保留前 7 个真实 BS，再用确定性随机生成的 BS 点补齐。
3. 生成 BS 点的 XY 范围来自同一 label 的 UE 采样点包围盒。
4. 生成 BS 的 `z` 使用模板中真实 BS 的中位高度；如果高度缺失，则回退到 `z=2.4 m`。
5. 生成点要求 BS-BS 最小水平间距不小于 `1.0 m`。
6. 固定随机种子：

```text
seed = 20260518
```

生成出的临时 label 只放在 `outputs/uplink_scale_sweep_round2_bs_sweep/labels/`，
不提交到 git。

## 测试矩阵

`B04` 与第一轮 `A05` 口径相同，`B14` 与第一轮 `A10` 口径相同。为了同一轮日志和
BS 生成逻辑一致，正式执行时可以重跑；如果只想节省时间，也可以在汇总时复用 A 轮结果并
标注 `reused_from_round1`。

| Job | BS | UE | Links | synthetic_array | BS 来源 | 预期/目的 |
|---|---:|---:|---:|---|---|---|
| B01 | 1 | 20 | 20 | false | real | false 口径下的单 BS 基线 |
| B02 | 2 | 20 | 40 | false | real | 观察 BS 从 1 到 2 的 RT/HDF5 增量 |
| B03 | 4 | 20 | 80 | false | real | 中等真实 BS 数 |
| B04 | 6 | 20 | 120 | false | real | A05 已通过；本轮锚点 |
| B05 | 7 | 20 | 140 | false | real | 最大真实 BS 数 |
| B06 | 8 | 20 | 160 | false | real + generated | 首个生成 BS 参与点 |
| B07 | 12 | 20 | 240 | false | real + generated | false 口径 BS 上限探索 |
| B08 | 16 | 20 | 320 | false | real + generated | false 口径 BS 上限探索 |
| B09 | 24 | 20 | 480 | false | real + generated | 预期高风险 |
| B10 | 32 | 20 | 640 | false | real + generated | 预期高风险 |
| B11 | 1 | 20 | 20 | true | real | true 口径单 BS 基线，验证 A10 失败是否主要由 BS 数放大触发 |
| B12 | 2 | 20 | 40 | true | real | true 口径小 BS 数 |
| B13 | 4 | 20 | 80 | true | real | true 口径中等真实 BS 数 |
| B14 | 6 | 20 | 120 | true | real | A10 已失败；本轮锚点 |
| B15 | 7 | 20 | 140 | true | real | 最大真实 BS 数 |
| B16 | 8 | 20 | 160 | true | real + generated | 首个生成 BS 参与点 |
| B17 | 12 | 20 | 240 | true | real + generated | true 口径 BS 上限探索 |
| B18 | 16 | 20 | 320 | true | real + generated | true 口径 BS 上限探索 |
| B19 | 24 | 20 | 480 | true | real + generated | 预期高风险 |
| B20 | 32 | 20 | 640 | true | real + generated | 预期高风险 |

## 执行策略

先跑低风险和关键边界，不把 20 个 job 一次性全部扔给 GPU：

1. 第一批：

```text
B01, B02, B03, B04, B11, B12, B13, B14
```

2. 若第一批中 `B04=false` 仍通过，继续跑真实最大 BS 和首个 generated BS：

```text
B05, B06, B15, B16
```

3. 若 `B06=false` 通过，再继续大 BS：

```text
B07, B08, B09, B10
```

4. `synthetic_array=true` 若在小 BS 已失败，则停止 true 的更大 BS 任务，避免重复 OOM。

失败后不立即继续增大 BS；例如 `B07=12` 通过而 `B08=16` 失败，则补测：

```text
BS = 14
```

## 汇总指标

每个 job 需要记录：

| 指标 | 说明 |
|---|---|
| status | pass/fail/timeout |
| failure_summary | Dr.Jit OOM、`2^32` entry limit、其他异常 |
| wall time | console `/usr/bin/time` |
| `rt_solve` | debug `perf_summary.json` |
| `nr_srs_observation` | debug `perf_summary.json` |
| `hdf5_write` | debug `perf_summary.json` |
| HDF5 size | `results.h5` 体积 |
| peak RSS | `/usr/bin/time -v` |
| HDF5 schema | pass/fail |
| link roles | `/link/tx_role="ue"`、`/link/rx_role="bs"` |
| shape | `/channel/truth/cfr = [20, BS, 16, 2, 3276]` |

## 预期结论形式

本轮结束后需要给出：

1. `UE=20` 时，`synthetic_array=false` 的最大可运行 BS 数。
2. `UE=20` 时，`synthetic_array=true` 的最大可运行 BS 数。
3. BS 数增加对 `rt_solve`、HDF5 size、`hdf5_write` 的近似斜率。
4. 是否需要二维 BS/UE block shard，以及推荐初始值，例如：

```text
bs_block_size = ?
ue_block_size = 20
```
