# NR SRS RT Variant Sweep 6x5

状态更新：本文是 2026-05-18 的历史 RT 参数 sweep。文中的 `array.spectrum.sources:
["truth_cfr", "srs_cfr_est"]` 使用的是历史 source alias；schema `1.5.0` 后不再接受
`srs_cfr_est`，当前应使用 `cfr_est`，并且不再写 `/array/spatial_spectrum_srs`。
本文仍保留在 performance 中，因为它记录了 refraction/diffuse/max_depth 对 RT 成本和
CFR 差异的实验事实；当前默认配置见 `docs/sys/indoor_fr1_100mhz_validation.md`。

本页记录 `bistro_0000` 上 SRS-like 100 MHz、`synthetic_array=false` 的四组 RT
参数对比。测试日期：2026-05-18；分支：`codex/phy-module-nr-srs`。

## 测试口径

四组配置均使用：

- `phy.standard: "nr_srs"`
- `runtime.device: "cuda"`
- `input.max_bs: 6`
- `input.max_ue: 5`
- `rt.synthetic_array: false`
- `array.spectrum.enabled: true`
- `array.spectrum.sources: ["truth_cfr", "srs_cfr_est"]`
- `visualization.enabled: false`
- `output.save_full_paths: false`

四组 RT 变量：

| 变体 | refraction | diffuse_reflection | max_depth |
|---|---:|---:|---:|
| `refraction_on` | true | false | 4 |
| `refraction_off` | false | false | 4 |
| `refraction_off_diffuse_on` | false | true | 4 |
| `refraction_on_depth6` | true | false | 6 |

## 直接 Shard 结果

直接运行 `6 BS x 5 UE`，以及把 UE shard 改到 `6 BS x 1 UE`，都会在 Sionna RT
`PathSolver` 阶段失败：

| 粒度 | 结果 | 失败点 |
|---|---|---|
| `6x5` | 失败 | `PathSolver -> paths.add_paths` 中 Dr.Jit device memory OOM |
| `6x1` | 失败 | 同上 |

典型错误是：

```text
jit_malloc(): out of memory! Could not allocate 2147483648 bytes of device memory.
dr.while_loop(): encountered an exception.
```

这说明 `synthetic_array=false` 后，RT 路径候选/阵列展开会显著放大底层 Dr.Jit
连续显存需求。它发生在 RT 求解阶段，早于 CIR/CFR、SRS LS、空间谱和 HDF5 写盘。

## Micro-Sweep 结果

为保持同一批 `6 BS x 5 UE` 覆盖，同时规避 RT OOM，本次使用
`scripts/run_srs_rt_variant_micro_sweep.py` 将每个变体拆成 30 个 `1 BS x 1 UE`
单链路作业，并用 4 张 GPU 并行执行。四组共 120 个单链路作业全部成功。

命令示例：

```bash
uv run python scripts/run_srs_rt_variant_micro_sweep.py \
  --variant refraction_on=config/perf/nr_srs_6x5_rt_refraction_on.yaml \
  --variant refraction_off=config/perf/nr_srs_6x5_rt_refraction_off.yaml \
  --variant refraction_off_diffuse_on=config/perf/nr_srs_6x5_rt_refraction_off_diffuse_on.yaml \
  --variant refraction_on_depth6=config/perf/nr_srs_6x5_rt_refraction_on_depth6.yaml \
  --output-root outputs/rt_sweep_srs_6x5_micro \
  --gpu-ids 3,4,5,6 \
  --max-bs 6 \
  --max-ue 5 \
  --workers 4
```

总墙钟时间为 192.3 s。每个变体的 30 个单链路作业统计如下：

| 变体 | 作业数 | 单链路均值 | 单链路 p95 | 单链路耗时总和 | HDF5 总量 | 目录总量 |
|---|---:|---:|---:|---:|---:|---:|
| `refraction_on` | 30 | 6.38 s | 7.00 s | 191.4 s | 192.8 MB | 194.2 MB |
| `refraction_off` | 30 | 6.27 s | 7.17 s | 188.2 s | 210.1 MB | 211.5 MB |
| `refraction_off_diffuse_on` | 30 | 6.28 s | 6.79 s | 188.3 s | 210.1 MB | 211.5 MB |
| `refraction_on_depth6` | 30 | 6.47 s | 8.53 s | 194.2 s | 209.4 MB | 210.8 MB |

所有 120 个 HDF5 均通过 schema 校验。

## 2583 UE 估算

完整 `bistro_0000` 口径为 `6 BS x 2583 UE`，相当于当前 `6x5` probe 的
`2583/5 = 516.6` 倍。按单链路 micro-sweep 线性外推：

| 变体 | 单 GPU 顺序估算 | 4 GPU 估算 | HDF5 估算 | 目录总量估算 |
|---|---:|---:|---:|---:|
| `refraction_on` | 27.5 h | 6.9 h | 99.6 GB | 100.3 GB |
| `refraction_off` | 27.0 h | 6.8 h | 108.6 GB | 109.3 GB |
| `refraction_off_diffuse_on` | 27.0 h | 6.8 h | 108.5 GB | 109.2 GB |
| `refraction_on_depth6` | 27.9 h | 7.0 h | 108.2 GB | 108.9 GB |

这是 micro-link 粒度的保守估算。它比一次性 `6x5` 更稳定，但进程启动和重复加载场景开销较大；
生产化更理想的方向是实现原生二维 shard：按较小的 TX/RX block 跑 RT，同时仍把多个 link
合并写入较少的 `result_xxx.h5`。

## Synthetic Array True/False 对照

不能直接拿历史一次性 `6x5` synthetic-array probe 证明 `synthetic_array=true` 与
`false` 对数据影响很小，因为历史 probe 和本次 micro-sweep 的执行粒度不同。补跑同一
micro-sweep 口径后，观察到：

| 对比 | truth CFR 复相关 | truth CFR NMSE | 幅度 MAE | 相位圆周 MAE |
|---|---:|---:|---:|---:|
| 历史 true shard vs micro true | 0.8836 | -1.29 dB | 3.74 dB | 0.481 rad |
| micro true vs micro false | 0.3341 | 6.35 dB | 5.27 dB | 0.723 rad |

同一 micro 口径下，`/observation/cfr_est` 的 true/false 差异与 truth CFR 基本一致：
复相关 0.3344、NMSE 6.35 dB、幅度 MAE 5.28 dB、相位圆周 MAE 0.726 rad。
这说明当前 bistro 场景和 100 MHz 宽带配置下，`synthetic_array=true` 与 `false`
不能视为“数据影响微乎其微”。`true` 更像可扩展的阵列响应近似，`false` 更像
element-level 几何 probe，但当前普通 UE shard 会在 6 个 BS 同时参与 RT 时 OOM。

补跑的 synthetic-array micro true 性能：

| 口径 | 作业数 | 单链路均值 | 单链路 p95 | 单链路耗时总和 | HDF5 总量 | 2583 UE 4 GPU 估算 | 2583 UE HDF5 估算 |
|---|---:|---:|---:|---:|---:|---:|---:|
| micro true | 30 | 6.48 s | 9.26 s | 194.3 s | 160.6 MB | 7.0 h | 83.0 GB |
| micro false refraction_on | 30 | 6.38 s | 7.00 s | 191.4 s | 192.8 MB | 6.9 h | 99.6 GB |

这两组 micro 结果的耗时接近；主要差异体现在 HDF5 体积和 CFR 数值。由于 micro-sweep
每个作业都重复加载场景并启动 pipeline，这个耗时不能直接等价为未来原生二维 shard 的最佳性能。

## CFR 相似性

使用 `scripts/compare_srs_rt_variants.py` 比较四组 `/observation/cfr_est`。指标覆盖
30 条公共全局链路，共 `3,144,960` 个复数样本。参考组为 `refraction_on`。

| 变体 | NMSE | 复相关 | 幅度 MAE | 相位圆周 MAE |
|---|---:|---:|---:|---:|
| `refraction_on` | -300.00 dB | 1.0000 | 0.00 dB | 0.000 rad |
| `refraction_off` | -17.22 dB | 0.9935 | 2.20 dB | 0.254 rad |
| `refraction_off_diffuse_on` | -17.22 dB | 0.9934 | 2.21 dB | 0.256 rad |
| `refraction_on_depth6` | -13.06 dB | 0.9918 | 3.00 dB | 0.291 rad |

额外观察：

- `refraction_off` 与 `refraction_off_diffuse_on` 很接近：二者 NMSE 为 -36.43 dB，
  复相关 0.9999，幅度 MAE 0.17 dB，相位 MAE 0.018 rad。
- `refraction_on_depth6` 相对 `refraction_on` 的变化更明显，说明提高深度引入了更多
  高阶路径/候选，对宽带 CFR 的相干叠加影响大于本次 diffuse toggle。

同链路可视化命令：

```bash
uv run python scripts/compare_srs_rt_variants.py \
  outputs/rt_sweep_srs_6x5_micro/refraction_on \
  outputs/rt_sweep_srs_6x5_micro/refraction_off \
  outputs/rt_sweep_srs_6x5_micro/refraction_off_diffuse_on \
  outputs/rt_sweep_srs_6x5_micro/refraction_on_depth6 \
  --labels refraction_on,refraction_off,refraction_off_diffuse_on,refraction_on_depth6 \
  --reference refraction_on \
  --bs-index 0 \
  --ue-index 0 \
  --output-dir outputs/rt_sweep_srs_6x5_micro/comparison
```

输出：

- `outputs/rt_sweep_srs_6x5_micro/comparison/metrics.json`
- `outputs/rt_sweep_srs_6x5_micro/comparison/metrics.csv`
- `outputs/rt_sweep_srs_6x5_micro/comparison/cfr_magnitude_variants.png`
- `outputs/rt_sweep_srs_6x5_micro/comparison/cfr_phase_variants.png`

## 建议

1. 如果继续保持 `synthetic_array=false`，不要使用单纯 UE shard 直接跑 `6x5` 或 `6x1`；
   当前场景会在 RT 阶段 OOM。
2. 下一步生产化应支持二维 shard，例如 `bs_block_size=1`、`ue_block_size=N`，
   并让 manifest 汇总全局 BS/UE 覆盖。
3. 当前主线已经用 `phy_link_direction` 直接解析 direct uplink/downlink。若
   `synthetic_array=false` 仍 OOM，应优先做二维 BS/UE shard 和 RT 参数 sweep，
   而不是回到旧的 trace+transpose 口径。
4. 论文生产数据如果只需要 path 与 `cfr_est`，应继续评估关闭 `truth_cfr` 和部分空间谱 source
   后的体积下降；本次估算包含当前模板默认的 truth CFR、SRS CFR、空间谱和相关派生字段。
