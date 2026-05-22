# NR SRS Direct Uplink Scale Sweep Round 3 UE Block Results

日期：2026-05-18

分支：`codex/uplink-scale-sweep`

状态更新：本文是 direct-uplink UE block 边界的历史 sweep 结果。文中的 `UE_block`
边界用于解释当时 Bistro 场景和 100 MHz 模板下的风险，不等同于当前生产模板默认值；
当前推荐 `shard_size=20` 和已完成 baseline 见
`docs/sys/indoor_fr1_100mhz_validation.md`。

本文记录第三轮 **生产 shard 粒度精细实验**。当前目标场景约为：

```text
BS_all x UE_all ~= 7 x 2500
```

本轮不再寻找理论最大 link 数，而是验证单个 shard 中 `UE_block` 的安全边界，
并估计完整 `7 x 2500` 仿真的 shard 数、总文件大小和端到端成本。

原始输出：

```text
outputs/uplink_scale_sweep_round3_ue_block_sweep/
outputs/uplink_scale_sweep_round3_ue_block_sweep/summary.csv
outputs/uplink_scale_sweep_round3_ue_block_sweep/summary.json
```

## 测试口径

| 项 | 值 |
|---|---|
| 场景 | `data/bistro_0000` |
| PHY | `phy.standard="nr_srs"` |
| 链路方向 | `phy_link_direction="uplink"`，TX=UE，RX=BS |
| synthetic_array | `false` |
| 频域 | 100 MHz 模板，`3276` active subcarriers |
| 天线 | BS `4x4=16`，UE `1x2=2` |
| RT 参数 | 模板默认：`max_depth=4`，refraction/diffraction enabled |
| shard | disabled；每个 job 是一个候选 shard |
| visualization | disabled |
| array spectrum | disabled |
| save full paths | false |
| debug profiling | enabled |
| GPU | GPU 3-7 并行执行 |

E06/E07 用 10 个 BS 模拟未来更多 BS 的情况：7 个真实 BS + 3 个固定 seed 生成 BS。

## 结果表

| Job | BS_block | UE_block | Links | Status | Wall time | RT solve | SRS obs | HDF5 write | HDF5 | 失败原因 |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| E01 | 7 | 20 | 140 | PASS | 28.52 s | 6.84 s | 1.67 s | 14.86 s | 381.14 MB | |
| E02 | 7 | 25 | 175 | PASS | 35.51 s | 8.11 s | 2.06 s | 19.73 s | 474.34 MB | |
| E03 | 7 | 30 | 210 | PASS | 41.13 s | 9.24 s | 2.54 s | 23.36 s | 569.62 MB | |
| E04 | 7 | 35 | 245 | FAIL | 4.82 s | n/a | n/a | n/a | n/a | Dr.Jit CUDA OOM，PathSolver 阶段分配 2.0 GiB 失败 |
| E05 | 7 | 40 | 280 | FAIL | 4.60 s | n/a | n/a | n/a | n/a | Dr.Jit CUDA OOM，PathSolver 阶段分配 2.0 GiB 失败 |
| E06 | 10 | 20 | 200 | PASS | 37.30 s | 9.36 s | 2.33 s | 19.70 s | 495.57 MB | |
| E07 | 10 | 25 | 250 | PASS | 44.85 s | 9.71 s | 2.83 s | 25.96 s | 619.78 MB | |

所有 PASS job 均通过 HDF5 schema 校验，并满足：

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
| E03 | `[30, 7, 16, 2, 3276]` | `[1, 30, 7, 16, 2, 3276]` |
| E07 | `[25, 10, 16, 2, 3276]` | `[1, 25, 10, 16, 2, 3276]` |

## 关键结论

1. **当前 `7 BS` 生产口径下，`UE_block=30` 可跑，`UE_block=35` 已失败。**

   因此在当前 Bistro 场景、RT 参数和 SRS 100 MHz 输出契约下，真实 uplink 的单 shard
   安全边界落在：

   ```text
   30 < UE_block < 35   (BS_block=7, synthetic_array=false)
   ```

2. **`links` 仍不能作为唯一尺度。**

   本轮出现：

   ```text
   7 BS x 35 UE = 245 links 失败
   10 BS x 25 UE = 250 links 通过
   ```

   这说明是否失败不只取决于 `BS_block * UE_block`。在 direct uplink 下，
   UE 是 source，PathSolver 候选路径和 Dr.Jit buffer 对 `UE_block` 更敏感；同时，
   BS 位置变化也会改变路径候选数量。因此 shard 公式必须区分：

   ```text
   N_src = UE_block
   N_tgt = BS_block
   N_link = UE_block * BS_block
   ```

3. **输出大小和写盘成本基本按 link 线性增长。**

   PASS 样本的经验成本约为：

   ```text
   HDF5 size   ~= 2.5 .. 2.7 MiB / link
   HDF5 write  ~= 0.10 .. 0.11 s / link
   SRS obs     ~= 0.011 .. 0.012 s / link
   ```

   这解释了为什么总文件大小不会因为 shard 变多而显著变化；shard 主要影响的是
   单次失败重跑成本、多 GPU 调度颗粒度、单文件大小、schema validate 时间和并发写盘压力。

## `7 x 2500` 全量估算

按 `BS_all=7`、`UE_all=2500`、关闭空间谱/可视化、保存完整 truth CFR 与 SRS 输出估算：

| UE_block | shard 数 | 单 shard HDF5 | 单 shard wall | 总 HDF5 | 单 GPU 串行 | 4 GPU 理想 | 5 GPU 理想 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 125 | 381 MB | 28.52 s | 46.5 GiB | 59.4 min | 14.9 min | 11.9 min |
| 25 | 100 | 474 MB | 35.51 s | 46.3 GiB | 59.2 min | 14.8 min | 11.8 min |
| 30 | 84 | 570 MB | 41.13 s | 46.7 GiB | 57.6 min | 14.4 min | 11.5 min |

这些是理想调度估算。真实多 GPU 运行会受到磁盘并发写入、HDF5 gzip 压缩、GPU 负载波动、
最后一个 shard 不满块等因素影响。

## 推荐 shard 策略

当前建议分成两档：

### 保守生产档

```text
synthetic_array = false
bs_block_size = 7
ue_block_size = 25
```

原因：

- `7x25` 已通过，距离 `7x35` 失败边界有一定余量。
- 单 shard 约 `474 MB`，失败重跑成本可控。
- `2500 UE` 正好约 `100` 个 shard，调度比较整齐。

### 激进生产档

```text
synthetic_array = false
bs_block_size = 7
ue_block_size = 30
```

原因：

- `7x30` 已通过，是目前测试到的最高可用 UE block。
- shard 数从 100 降到 84，单 GPU 串行估计略低。
- 风险是距离 `7x35` OOM 边界较近；如果后续场景路径更多、RT 参数更重或打开额外输出，
  可能需要回退到 25。

## 后续建议

1. **补测 `7x32` 或 `7x33`。**

   如果希望把边界更精确定位在 30 到 35 之间，可补：

   ```text
   7 BS x 32 UE
   7 BS x 33 UE
   ```

   但生产上未必需要压得这么极限，`UE_block=25/30` 已足够指导切分。

2. **补测 `10x30`。**

   `10x25` 已通过，但还不知道 10 BS 下 `UE_block=30` 是否稳定。若未来 BS 数可能到 10，
   建议补一个：

   ```text
   10 BS x 30 UE
   ```

3. **正式实现二维 shard planner。**

   生产 planner 不应只按 link 数切分，而应使用：

   ```text
   primary constraint: UE_block <= measured_source_limit
   secondary constraint: BS_block * UE_block * MiB_per_link <= target_shard_size
   ```

   当前可用经验公式：

   ```text
   target_shard_size_MiB ~= 2.7 * BS_block * UE_block
   target_write_s        ~= 0.11 * BS_block * UE_block
   ```

   RT 安全边界暂时不能只靠公式，需要按场景和 RT 参数保留测量值：

   ```text
   Bistro/current RT: UE_block_safe = 25
   Bistro/current RT: UE_block_max_verified = 30
   Bistro/current RT: UE_block_fail = 35
   ```

4. **继续做输出裁剪消融。**

   如果实验主要使用 path 信息和 `cfr_est`，应加入配置开关关闭 truth CFR 或重型 grids。
   这不会改变总 link 数，但会显著影响 `MiB/link` 和 `write_s/link`。
