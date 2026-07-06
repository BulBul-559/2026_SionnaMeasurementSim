# CFR Truth ToA / First-Path RTT GT Pilot 2026-07-02

## 目的

本轮小规模 pilot 验证：

- GT：RT/path truth 写出的 first-path delay/range、LoS/NLoS 和 path power。
- 估计：只从 `/channel/truth/cfr` 做后处理 ToA / RTT-like range 估计。
- 禁止：ToA 估计阶段不使用 UE/BS 坐标、`/derived` label 或其它 GT；GT 只在最后算误差。

## 仿真配置

模板：

```text
config/tasks/nr_srs_64prb_cfr_truth_rtt_gt_pilot.yaml
```

该模板从 `config/tasks/nr_srs_64prb_cfr_truth_only.yaml` 派生，保持 3000 场景
CFR truth-only 任务的 64 PRB carrier、Front3D 0p5 label、RT、antenna、sharding、
dynamic GPU scheduler 和 fallback 口径，只把输出产品扩展为：

```yaml
output:
  products: ["derived", "link_labels", "cfr_truth"]
phy:
  enabled: false
ranging:
  enabled: false
```

因此本轮没有使用内置 `/ranging`。内置 ranging 当前从 `/observation/cfr_est` 估计，
不符合本轮“只从 CFR truth 后处理估计”的约束。

场景 index：

```text
outputs/pilot_cfr_truth_toa_rtt_gt_20260702/pilot_front3d_5scene_panel0p5_seed2026.jsonl
```

选取 5 个 0p5 panel 场景：

| scene | class | planned raw links |
|---|---|---:|
| `front3d_0007` | normal_room | 1,316 |
| `front3d_0019` | normal_room | 786 |
| `front3d_0022` | normal_room | 1,204 |
| `front3d_0015` | small_room | 2,448 |
| `front3d_0021` | small_room | 775 |

仿真命令：

```bash
.venv/bin/python -m sionna_measurement_sim.app.cli \
  --config config/tasks/nr_srs_64prb_cfr_truth_rtt_gt_pilot.yaml \
  run-scene-index \
  --index-file outputs/pilot_cfr_truth_toa_rtt_gt_20260702/pilot_front3d_5scene_panel0p5_seed2026.jsonl \
  --output-root outputs/pilot_cfr_truth_toa_rtt_gt_20260702/runs
```

仿真结果：

| 指标 | 值 |
|---|---:|
| completed scenes | 5 |
| failed scenes | 0 |
| shard H5 files | 48 |
| scheduler elapsed | 66.48 s |
| valid GT links used for evaluation | 6,122 |

## 后处理

脚本：

```text
scripts/evaluate_cfr_truth_toa_pilot.py
```

命令：

```bash
.venv/bin/python scripts/evaluate_cfr_truth_toa_pilot.py \
  --input-root outputs/pilot_cfr_truth_toa_rtt_gt_20260702/runs \
  --output-dir outputs/pilot_cfr_truth_toa_rtt_gt_20260702/analysis
```

输出：

```text
outputs/pilot_cfr_truth_toa_rtt_gt_20260702/analysis/summary.json
outputs/pilot_cfr_truth_toa_rtt_gt_20260702/analysis/summary.md
outputs/pilot_cfr_truth_toa_rtt_gt_20260702/analysis/toa_link_details.csv
outputs/pilot_cfr_truth_toa_rtt_gt_20260702/analysis/toa_link_details.jsonl
```

估计器：

- `pdp`：Hann window + 8x zero-padded IFFT，取相对最强峰 `-12 dB` 内最早 peak cluster。
- `phase`：逐 RX/TX antenna pair 对 CFR phase-slope 做 weighted fit，再按 power 做 weighted median。

两者均只读 CFR 和 frequency grid。`first_path_propagation_range_m`、`los_flag`、
`path_power_db` 只在估计完成后用于误差统计。

## 结果

整体 absolute one-way range error：

| estimator | success rate | mean | median | P90 | P95 | signed mean |
|---|---:|---:|---:|---:|---:|---:|
| PDP peak | 100.00% | 1.039 m | 0.322 m | 2.959 m | 4.786 m | +1.017 m |
| phase slope | 98.79% | 1.041 m | 0.324 m | 2.928 m | 5.025 m | +1.026 m |

LoS vs NLoS：

| estimator | group | valid links | success rate | median | P90 | P95 |
|---|---|---:|---:|---:|---:|---:|
| PDP peak | LoS | 3,358 | 100.00% | 0.225 m | 0.608 m | 0.771 m |
| PDP peak | NLoS | 2,764 | 100.00% | 1.106 m | 5.101 m | 6.904 m |
| phase slope | LoS | 3,358 | 100.00% | 0.234 m | 0.630 m | 0.800 m |
| phase slope | NLoS | 2,764 | 97.32% | 1.011 m | 5.378 m | 7.159 m |

按 `path_power_db` 分桶，PDP peak：

| path_power_db bin | valid links | success rate | median | P90 | P95 |
|---|---:|---:|---:|---:|---:|
| `[-inf,-117)` | 124 | 100.00% | 0.070 m | 1.138 m | 1.628 m |
| `[-117,-100)` | 357 | 100.00% | 0.081 m | 3.594 m | 5.523 m |
| `[-100,-80)` | 1,128 | 100.00% | 1.216 m | 5.517 m | 7.381 m |
| `[-80,-60)` | 1,443 | 100.00% | 1.002 m | 4.725 m | 6.427 m |
| `[-60,inf)` | 3,070 | 100.00% | 0.224 m | 0.612 m | 0.770 m |

按 `path_power_db` 分桶，phase slope：

| path_power_db bin | valid links | success rate | median | P90 | P95 |
|---|---:|---:|---:|---:|---:|---:|
| `[-inf,-117)` | 124 | 40.32% | 0.039 m | 1.368 m | 2.066 m |
| `[-117,-100)` | 357 | 100.00% | 0.059 m | 3.715 m | 5.861 m |
| `[-100,-80)` | 1,128 | 100.00% | 1.021 m | 5.760 m | 7.549 m |
| `[-80,-60)` | 1,443 | 100.00% | 0.919 m | 4.852 m | 6.690 m |
| `[-60,inf)` | 3,070 | 100.00% | 0.237 m | 0.635 m | 0.803 m |

本轮 `csi_power_db` 分桶统计与 `path_power_db` 分桶一致；明细 CSV/JSONL 中保留了两列，
后续若切到 observation CFR 或不同归一化口径，应重新检查二者是否仍一致。

## 结论

这个 RTT-like range 估计可以作为 signal-processing weak reference，但还不足以单独作为
可靠的 floorplan+ranging baseline：

- 在 LoS 链路上，noise-free CFR truth 的 PDP/phase 都给出较稳定的亚米级结果，
  median 约 0.22-0.23 m，P95 约 0.8 m。
- 在 NLoS 链路上，首径可能较弱，PDP/phase 更容易被强反射或混叠结构拉偏；
  median 约 1 m，P95 约 7 m。
- 本轮使用的是 CFR truth，没有 AWGN、CFO、timing、SRS 资源稀疏估计误差和硬件损伤；
  真实 `/observation/cfr_est` 或论文式带噪 CFR image 下误差只会更差。
- 64 PRB / 23.04 MHz 的带宽给出的 PDP 时间分辨率有限，8x oversampling 是插值而不是
  增加真实物理带宽；强 LoS 可以稳定，复杂 NLoS 不可靠。

建议定位实验中把它作为一个辅助输入或诊断上界，而不是主 baseline。若要作为真正的
floorplan+ranging baseline，需要至少补充 LoS gating、NLoS robustness、观测 CFR 噪声/损伤、
多 SNR 分档和更大场景覆盖验证。
