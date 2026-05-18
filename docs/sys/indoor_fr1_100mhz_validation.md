# Indoor FR1 100 MHz Validation Notes

本页记录室内 FR1 100 MHz PUSCH/SRS-like 模板的当前生产口径、已验证 baseline、
历史 probe 和主要成本结论。更短的交接摘要见 `docs/agent_handoff.md`。

## 当前数据口径

真实场景数据位于 ignored `data/` 路径下，当前按密度分为：

```text
data/dense/
data/medium/
data/sparse/
```

每个场景目录使用对应前缀，例如：

```text
data/medium/medium_0000/
```

每个场景保留三种 UE 采样 label：

| 文件 | UE 采样间隔 | UE 数 |
|---|---:|---:|
| `label0p1.json` | 0.1 m | 10360 |
| `label0p2.json` | 0.2 m | 2583 |
| `label0p4.json` | 0.4 m | 654 |

当前推荐 baseline 使用 `label0p2.json`。`label0p1.json` 计算和存储成本约为
`label0p2.json` 的 4 倍，不建议作为默认全量测试。

## 当前推荐 SRS-like 生产配置

默认模板：

```text
config/defaults/nr_srs_indoor_positioning_fr1_100mhz.yaml
```

当前生产口径：

- `link.phy_link_direction: "uplink"`，即 `TX=UE`、`RX=BS`
- `phy.standard: "nr_srs"`
- `rt.synthetic_array: false`
- `rt.max_depth: 4`
- `rt.los/specular_reflection/refraction/diffraction: true`
- `rt.diffuse_reflection: false`
- `array.spectrum.enabled: false`
- `visualization.enabled: false`
- `output.sharding.enabled: true`
- `output.sharding.axis: "ue"`
- `output.sharding.shard_size: 20`

使用真实场景时，建议复制模板到 `outputs/local_configs/`，再修改
`input.label_file`、`input.scene_file`、`input.scene_id`、`output.root_dir` 和 GPU 配置。

## 已完成 baseline

### `medium_0000 label0p2` SRS-like direct uplink

输出目录：

```text
outputs/nr_srs_medium_0000_label0p2_full_baseline_shard20
```

实际路径通常为：

```text
/data/sunmeiyuan/projects/sionna/outputs/nr_srs_medium_0000_label0p2_full_baseline_shard20
```

| 项 | 值 |
|---|---:|
| 场景 | `medium_0000` |
| label | `label0p2.json` |
| BS / UE | 7 / 2583 |
| shard size | 20 |
| shard count | 130 |
| GPU | `[5, 6, 7]` |
| wall time | 1274.72 s，约 21.2 min |
| output size | 约 52 GB |

产物检查：

- `result_000.h5` 到 `result_129.h5` 连续存在。
- UE 覆盖 `0..2582`，无缺失、无重复。
- BS 覆盖 `[0,1,2,3,4,5,6]`。
- `/link/tx_role = "ue"`。
- `/link/rx_role = "bs"`。
- 没有 `/waveform/tx_time` 或 `/waveform/rx_time`。
- `manifest.json` 已生成。

普通 shard 的关键 shape：

| Dataset | Shape |
|---|---|
| `/channel/truth/cfr` | `[20, 7, 16, 2, 3276]` |
| `/observation/cfr_est` | `[1, 20, 7, 16, 2, 3276]` |
| `/waveform/srs_rx_grid` | `[1, 20, 7, 16, 2, 3276]` |
| `/waveform/srs_tx_grid` | `[1, 20, 7, 2, 2, 3276]` |

最后一个 shard 只有 3 个 UE，因此 TX 维为 3。

## 为什么默认 `shard_size=20`

direct uplink 语义完成后，`UE` 是 RT source，`BS` 是 RT receiver。实际风险不只取决于
`BS × UE` link 数，还和 source 数、路径复杂度、CFR 子载波数、阵列维度共同相关。

已观察到：

| 配置 | 结果 |
|---|---|
| `7 BS × 30 UE` 单 shard | 可运行 |
| `7 BS × 35 UE` / `7 BS × 40 UE` 单 shard | `PathSolver` / Dr.Jit 阶段失败 |
| `medium_0000 label0p2`，`shard_size=25` | 后段 shard 在 `paths.cfr()` 触发 Dr.Jit `2^32` entry 上限 |
| `medium_0000 label0p2`，`shard_size=20` | 完成 130 个 shard |

`shard_size=25` 失败不是普通显存 OOM，而是 Dr.Jit 单数组 entry 数超过
`2^32 == 4294967296` 的底层限制。后续如果换更复杂场景、更多 BS、打开空间谱或改变
RT 参数，需要重新做 shard sweep；不要假设 `20` 永远安全。

## 历史 probe

下表是 2026-05-17 早期真实场景 probe，发生在后续 BS/UE 与 TX/RX 语义彻底解耦、
SRS-like direct uplink 和 `synthetic_array=false` 生产口径稳定之前。它们只能作为链路功能
和输出字段的历史参考，不代表当前生产性能。

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

早期 probe 的 PUSCH-DMRS CSI proxy 与 SRS-like full-band sounding 对比：

| 链路 | links | valid rate | median NMSE | p95 NMSE |
|---|---:|---:|---:|---:|
| PUSCH-DMRS CSI proxy | 30 | 1.0000 | 19.48 dB | 36.46 dB |
| SRS-like full-band sounding | 30 | 1.0000 | -29.99 dB | -29.97 dB |

这说明 SRS-like full-band sounding 更适合作为当前 CSI 监督/定位基线；PUSCH-DMRS proxy
仍可用于链路功能验证和 proxy 对照。若要称为标准 NR SRS，还需要补齐
`docs/sys/nr_srs_standard_todo.md` 中的 3GPP SRS 细节。

## 当前建议

- 默认使用 `label0p2.json` 做 baseline。
- 默认使用 `shard_size=20`。
- 生产默认关闭空间谱和可视化；二者适合小样本诊断，不适合默认全量生成。
- 下游训练优先按 `manifest.json` 读取多个 `result_xxx.h5`，不要假设单文件。
- 后续性能优化优先考虑：
  - 减少每 shard 的 schema validation 重复读全文件成本。
  - 对 HDF5 压缩策略做 `gzip/lzf/none` 对比。
  - 支持只保存训练需要的字段，例如 path 相关数据和 `cfr_est`。
  - 继续探索二维 BS/UE shard，但不要让多个进程写同一个 HDF5。
