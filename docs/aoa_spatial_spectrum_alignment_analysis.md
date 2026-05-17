# AoA 与空间谱对齐问题分析

本文记录一次对 `/array/aoa_label_rad`、`/array/spatial_spectrum_label`、`/array/spatial_spectrum_truth`、`/array/spatial_spectrum_cfr_est`、`/array/spatial_spectrum_observation` 不一致现象的简要分析。

> 状态更新：reverse/uplink 端点问题已经修复。当前当
> `reciprocity_applied=true` 且 `phy_link_direction="uplink"` 时，
> `/derived/*aoa*` 和 `/array/aoa_label_rad` 使用原 RT 的 AoD，表示 BS 作为
> uplink receiver 看到的到达方向。本文其余关于 first-path label、多径 Bartlett
> 谱、观测谱质量和 steering 坐标校准的判断仍然成立。

分析样本来自：

```text
outputs/nr_pusch_5bs_5rx_cfr_est_spectrum_check/results.h5
```

## 现象

在 5 BS x 5 UE 的 NR PUSCH 结果中，空间谱 label、truth、observation 的峰值位置差异很大。直接从 HDF5 读取峰值角度后，典型统计如下：

```text
label vs truth:
  median zenith diff ~= 58 deg
  median azimuth diff ~= 66 deg

truth vs observation:
  median zenith diff ~= 48 deg
  median azimuth diff ~= 120 deg
```

因此，这不是绘图插值、色标或极坐标显示导致的问题，而是原始数据中的角度口径和空间谱生成口径本身存在不一致。

## 主要原因

### 1. Label 与空间谱的接收端点不一致

历史问题中，`spatial_spectrum_label` 来自：

```text
/array/aoa_label_rad
```

其构造来源是：

```text
derived.first_path_aoa_zenith_rad
derived.first_path_aoa_azimuth_rad
```

当时 `derived.first_path_aoa_*` 取自 RT path table 的 receiver-side AoA：

```text
theta_r_rad / phi_r_rad
```

NR PUSCH 配置是上行链路：

```text
rt_trace_direction: bs_to_ue
phy_link_direction: uplink
reciprocity_applied: true
```

也就是说，RT trace 是 BS -> UE 的下行几何路径，但 PUSCH 空间谱是在 UE -> BS 的上行接收阵列上生成。旧 label 更接近“到达 UE 侧的 AoA”，而 truth/observation 空间谱是在“BS 接收阵列”上扫描方向。两者物理端点不同，峰值自然会明显偏离。当前代码已经将 reverse/uplink 的 derived AoA 改为原 RT AoD。

### 2. Label 是 first-path one-hot，不等价于多径空间谱

当前 `spatial_spectrum_label` 是从单个 first-path AoA 画出的 one-hot heatmap。它表达的是一个监督标签点：

```text
first path AoA -> nearest grid bin
```

而 `spatial_spectrum_truth` 是从完整 CFR 做 Bartlett 扫描：

```text
R = X X^H / N
P(theta, phi) = real(a(theta, phi)^H R a(theta, phi))
```

它表达的是接收阵列看到的多径空间能量分布。多径场景下，最大谱峰可能来自 LoS、strongest NLoS、多条 path 的叠加，或者阵列歧义峰，不一定等于 first path。

因此，即使修正 AoA 的端点，`first_path` label 和 Bartlett truth spectrum peak 也不保证完全重合。

### 3. Observation 谱受 PUSCH 接收质量影响较大

该样本中 observation 质量较差：

```text
SNR = 30 dB
median NMSE ~= 25.3 dB
BER ~= 0.485
BLER = 1.0
```

这说明当前 PUSCH LS 估计和链路解调结果不可靠。基于 `cfr_est` 或 `rx_grid` 生成的空间谱不应直接作为高置信度 AoA 观测。

另外，`spatial_spectrum_observation` 当前来自 `/waveform/rx_grid`。`rx_grid` 是接收信号网格，不是纯信道：

```text
rx_grid = channel * PUSCH resource grid + noise + receiver effects
```

它混合了数据符号、DMRS、噪声和接收处理影响，因此与 `truth_cfr` 的 Bartlett 谱存在差异是正常的。若要构造更稳定的观测空间谱，应优先使用可靠的 pilot/DMRS 资源或校验后的 `cfr_est`。

### 4. Steering vector 坐标约定仍需校准

当前 Bartlett steering 使用简化 UPA 模型：

```text
direction_y = sin(zenith) * sin(azimuth)
direction_z = cos(zenith)
phase = 2*pi*(element_y*direction_y + element_z*direction_z)
```

这假设接收阵列位于局部 y-z 平面，并且未完整处理 Sionna `PlanarArray` 的本地坐标、阵列朝向、极化和阵元顺序约定。即使修正 AoA 端点，如果 steering 坐标和 Sionna 的阵列坐标不完全一致，空间谱峰值仍可能偏移。

## 初步结论

当前 AoA label 和空间谱 truth/observation 差异大的原因，优先级判断如下：

1. **AoA label 端点错误或不适配上行 PUSCH**：当前 label 使用 RT 下行 receiver 侧 AoA，而空间谱在上行 BS 接收阵列上生成。
2. **Label 表达过于单一**：first-path one-hot label 不等价于完整多径 Bartlett 空间谱。
3. **Observation 谱质量受限**：当前 `cfr_est` NMSE、BER、BLER 指标显示 PUSCH 观测链路质量较差。
4. **Steering 坐标可能未完全匹配 Sionna**：需要用可控场景校准阵列方向和元素顺序。

## 建议验证步骤

下一步建议先做小规模 sanity check，而不是直接在大规模数据上判断：

1. 已完成：BS 侧上行 AoA label 从 RT 的 departure-side angle，即 `theta_t_rad / phi_t_rad`，派生 reciprocal uplink receive AoA。
2. 保留现有 UE 侧 AoA label，明确区分：
   - `dl_rx_aoa_label_rad`
   - `ul_rx_aoa_label_rad`
3. 用 LoS-only、`perfect_csi=true`、小规模 1 BS x 1 UE 或 1 BS x 5 UE 场景验证：
   - `ul_rx_aoa_label_rad`
   - `spatial_spectrum_truth`
   - `spatial_spectrum_cfr_est`
4. 若 LoS-only 下仍明显偏离，再优先检查 Bartlett steering 的坐标系、阵列平面和阵元顺序。
5. 对 observation 谱，优先比较 `truth_cfr` 与 `perfect_csi cfr_est`；待 CFR 估计质量达标后，再比较普通 LS `cfr_est` 和 `rx_grid` observation。

## 数据解释口径

建议后续文档和 HDF5 schema 中明确区分：

```text
aoa_heatmap_label / spatial_spectrum_label:
  从真值 AoA 生成的监督标签，不是观测谱。

spatial_spectrum_truth:
  从 truth CFR 生成的理想 Bartlett 空间谱。

spatial_spectrum_cfr_est:
  从估计 CFR 生成的观测信道空间谱。

spatial_spectrum_observation:
  从 rx_grid 生成的接收信号空间谱，混合数据符号、导频、噪声和接收处理影响。
```
