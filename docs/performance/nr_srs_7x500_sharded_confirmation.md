# NR SRS Direct Uplink 7x500 Sharded Confirmation

日期：2026-05-18

分支：`codex/srs-shard-confirmation`

状态更新：本文是 2026-05-18 的历史确认测试，用于记录当时 `7 BS x 500 UE`、
`shard_size=25`、SRS-like direct uplink 的 shard 可行性。当前 SRS 生产建议已改为
`shard_size=20`，schema 已升级到 `1.5.0`，旧 array/SRS 兼容别名不再写入；当前默认口径见
`docs/sys/indoor_fr1_100mhz_validation.md`、`docs/agent_handoff.md` 和
`config/README.md`。

本文记录 `nr_srs` direct uplink 的 UE shard 生产口径确认测试。目标是确认：

1. shard runner 按新 BS/UE 语义切 UE，而不是旧 TX/RX 语义。
2. 每个 shard 直接生成独立 `result_xxx.h5`，aggregate `manifest.json` 记录全局索引。
3. 多 GPU 并行 shard 支持 SRS-like full-band sounding。

## 配置

运行配置：

```text
config/perf/nr_srs_7x500_sharded.yaml
```

核心口径：

| 项 | 值 |
|---|---|
| 场景 | `data/bistro_0000` |
| PHY | `phy.standard="nr_srs"` |
| 链路方向 | `phy_link_direction="uplink"`，TX=UE，RX=BS |
| 规模 | `7 BS x 500 UE` |
| synthetic_array | `false` |
| 频域 | 100 MHz，`3276` active subcarriers |
| 天线 | BS `4x4=16`，UE `1x2=2` |
| shard | `axis="ue"`, `shard_size=25`, `shard_count=20` |
| GPU | `gpu_ids=[3,4,5,6,7]`, `parallel_workers=5` |
| visualization | disabled |
| array spectrum | disabled |
| debug profiling | enabled |

RT 配置固定为：

```yaml
rt:
  max_depth: 4
  los: true
  specular_reflection: true
  diffuse_reflection: false
  refraction: true
  diffraction: true
  synthetic_array: false
  normalize_cfr: false
  normalize_delays: false
  merge_shapes: false
```

命令：

```bash
/usr/bin/time -v uv run python -m sionna_measurement_sim.app.cli \
  --config config/perf/nr_srs_7x500_sharded.yaml \
  run-full
```

输出目录：

```text
outputs/nr_srs_7x500_sharded_confirmation/
```

## 结果

| 指标 | 结果 |
|---|---:|
| shell wall time | `2:18.59` |
| aggregate manifest elapsed | `136.08 s` |
| exit status | `0` |
| result files | `20/20` |
| HDF5 schema | `20/20` pass |
| total HDF5 size | `9.68 GiB` |
| average shard HDF5 size | `495.45 MiB` |
| max shard duration | `35.90 s` |
| sum shard duration | `665.95 s` |

Aggregate stage totals across all shards:

| Stage | Sum duration |
|---|---:|
| `rt_solve` | `167.42 s` |
| `nr_srs_observation` | `41.95 s` |
| `hdf5_write` | `408.84 s` |
| `schema_validate` | `38.56 s` |
| `derived_nlos` | `1.13 s` |
| `array_outputs` | `1.44 s` |

注意：stage totals 是所有 shard 的求和；端到端 wall time 因 5 GPU 并行而远小于 stage sum。

## 语义检查

通过 aggregate `manifest.json` 和每个 `result_xxx.h5` 验证：

| 检查 | 结果 |
|---|---|
| UE 全局覆盖 | `0..499`，无缺失、无重复 |
| BS 全局索引 | 每个 shard 均为 `[0,1,2,3,4,5,6]` |
| `/link/tx_role` | `"ue"` |
| `/link/rx_role` | `"bs"` |
| `/link/phy_link_direction` | `"uplink"` |
| `/waveform/tx_time` | 不存在 |
| `/waveform/rx_time` | 不存在 |

每个 shard 的典型 shape：

```text
/channel/truth/cfr      [25, 7, 16, 2, 3276]
/observation/cfr_est    [1, 25, 7, 16, 2, 3276]
```

其中 HDF5 的 TX/RX link-view 在 uplink 下解释为：

```text
[ue, bs, bs_ant, ue_ant, subcarrier]
```

## 结论

三个待确认点均通过：

1. **shard runner 已按 BS/UE 语义切 UE。**

   `output.sharding.axis="ue"` 生成 20 个 shard，每个 shard 的 `global_tx_indices`
   对应当前 uplink TX=UE 的全局 UE 范围，覆盖 `0..499`。

2. **每个 shard 直接写独立 `result_xxx.h5`。**

   输出根目录包含：

   ```text
   result_000.h5
   ...
   result_019.h5
   manifest.json
   logs/perf_summary_shard_*.json
   ```

   不并发写同一个 HDF5，也不需要物理合并。

3. **多 GPU 并行支持 `nr_srs`。**

   `parallel_workers=5` 和 `gpu_ids=[3,4,5,6,7]` 正常完成。每个 worker 独立输出 shard
   HDF5 和 debug logs。

## 全量估算

以当前 `7x500` 结果线性外推到 `7x2500`：

| 规模 | shard 数 | 端到端 wall | HDF5 总量 |
|---|---:|---:|---:|
| `7x500` | 20 | `2.31 min` | `9.68 GiB` |
| `7x2500` | 100 | `11.5 min` | `48.4 GiB` |

该估算假设：

- 仍使用 `shard_size=25`、5 GPU 并行。
- 仍关闭空间谱和可视化。
- 仍保存当前核心 truth CFR、SRS grid、`cfr_est` 和 path/derived 输出。
- 磁盘并发写入能力没有明显下降。

如果改用 4 GPU，按同一 shard 耗时粗估约为 `14-15 min`。如果打开空间谱、可视化或额外保存字段，
需要重新测试，不能直接套用本估算。

## 历史生产建议（已 superseded）

下面是本次 2026-05-18 确认测试后形成的历史建议，后来已被
`median_0000 label0p2` 全量 baseline 修正：当前推荐 `shard_size=20`，见
`docs/sys/indoor_fr1_100mhz_validation.md`。

当时 Bistro SRS-like direct uplink 推荐模板：

```yaml
input:
  max_bs: 7
  max_ue: 2500
output:
  sharding:
    enabled: true
    axis: "ue"
    shard_size: 25
rt:
  synthetic_array: false
link:
  phy_link_direction: "uplink"
array:
  spectrum:
    enabled: false
visualization:
  enabled: false
```

`shard_size=25` 当时被认为是保守生产值；后续 `median_0000 label0p2` 全量 baseline
发现后段 shard 会在 `paths.cfr()` 触发 Dr.Jit `2^32` entry 上限，因此当前模板改用
`shard_size=20`。如果后续场景复杂度或 RT 参数变化，应重新做 UE block sweep。
