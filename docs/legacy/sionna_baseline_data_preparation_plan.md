# Sionna 仿真数据准备文档：面向第一阶段定位模型库

## 0. 当前结论

第一阶段先移除 **AAResCNN、DLoc、GenMetaLoc**。

移除原因：

| 模型 | 第一阶段移除原因 |
|---|---|
| AAResCNN | 定位版可以用 CSI/CFR 适配，但如果严格复现 tracking 或 IMU-aided 版本，需要历史轨迹/IMU 等额外传感器数据；第一阶段不优先。 |
| DLoc | 原方法依赖 MapFind / location-tagged map / 视觉或运动过程中的地图构建信息，当前 Sionna 仿真系统不能直接产生。 |
| GenMetaLoc | 需要 first Fresnel zone 3D point cloud / geometry-conditioned fingerprint generation，工程量较大，暂不考虑。 |

第一阶段保留的模型库对象：

| 类别 | 模型 |
|---|---|
| 传统指纹 | KNN, WKNN |
| CFR/CNN | MPRI |
| CSI/CFR Transformer / SSL / Foundation | SigMap w/o map, SWiT, LWLM |
| Map/Floor-plan 多模态 | SigMap w/ map, Floor-Plan-Aided 简化版 |
| AoA/信号处理 | MoD-DNN 简化版 / AoA 类 baseline |
| 跨场景少样本 | MetaLoc |

---

## 1. 系统边界划分

当前系统应该按“原料生产”和“训练组织”分层，不要把所有任务都塞进仿真系统。

| 模块 | 职责 | 是否属于仿真系统 |
|---|---|---|
| Sionna 仿真系统 | 生成 RT/PHY 原始数据、真值、观测值、路径信息、派生物理标签 | 是 |
| 平面图生成/地图系统 | 生成 floor plan、世界坐标到像素坐标的对齐关系、可选 3D mesh 解析结果 | 部分独立，和仿真系统通过 scene_id 对齐 |
| 数据转换器 | 把 HDF5 原料转换成模型输入格式，例如 CFR image、token、幅相交织、flatten fingerprint | 否，单独做 |
| Dataloader | train/val/test、support/query、few-shot、scene split、batch 组织 | 否，单独做 |
| 模型库 | KNN/WKNN/MPRI/SigMap/SWiT/LWLM/MetaLoc 等模型实现 | 否 |

关键判断：

- **A：地图与坐标对齐**应在平面图生成系统中完善。
- **B：训练组织和 split**应放在数据转换器/Dataloader，不应污染仿真原料。
- **C：RT/PHY 派生物理量**应在仿真系统中计算或至少标准化保存。
- **D：raw waveform / array observation**可以由 Sionna PHY 产生，但需要在仿真系统里显式保存；部分 AoA 空间谱标签需要我们自己派生。
- **E：Fresnel point cloud / GenMetaLoc 相关几何条件**第一阶段不考虑。

---

## 2. 第一阶段模型输入与监督需求总表

| 模型 | 输入信息 | 监督信息 | 当前仿真系统是否已有 | 需要补充的系统 |
|---|---|---|---|---|
| KNN | CSI/CFR fingerprint、RSSI fingerprint、幅度/相位/实虚部展平 | RX/UE 坐标 | 已有 CFR、观测 CFR、RSSI、RX 坐标 | 数据转换器：特征展平与归一化 |
| WKNN | 同 KNN，额外需要 top-k 距离或相似度 | RX/UE 坐标 | 已有 | 数据转换器：距离度量、加权策略 |
| MPRI | CFR image / CFR-derived image | RX/UE 坐标 | 已有 CFR 与坐标 | 数据转换器：CFR → image，例如幅度/相位/实虚部/多通道图 |
| SigMap w/o map | CSI/CFR token，单 BS 或多 BS channel | RX/UE 坐标；预训练阶段可无标签 | 已有 CFR、TX/RX 坐标 | 数据转换器：token 化、mask 策略、预训练样本组织 |
| SWiT | 无标签 CSI/CFR channel estimates；fine-tune 时用少量标签 | fine-tune 阶段需要 RX/UE 坐标 | 已有 CFR/CFR_est 与坐标 | Dataloader：无标签池、label ratio、support/query |
| LWLM | 空间-频率 channel，多 BS/单 BS channel；可能还要 ToA/AoA 任务标签 | 坐标、ToA、AoA、单 BS/多 BS任务标签 | CFR 和坐标已有；ToA/AoA 需要从路径派生 | 仿真系统：保存 full paths，派生 ToA/AoA；Dataloader：任务组织 |
| SigMap w/ map | CSI/CFR、3D map 或 2D floor plan、BS positions | RX/UE 坐标 | CFR、TX/RX 坐标已有；地图对齐关系未进统一契约 | 平面图系统：floor plan + world_to_pixel；仿真系统/地图系统：mesh 引用或图结构 |
| Floor-Plan-Aided 简化版 | RTT-like/range、RSSI、AP/RX 图结构、floor-plan crop、AP 坐标 | RX 坐标；距离修正分支需要 AP-RX 真值距离 | RSSI、坐标已有；RTT-like 与 floor-plan crop 未标准化 | 仿真系统：range/ToA/RTT-like；平面图系统：crop/rotate 所需映射 |
| MoD-DNN / AoA 类 | 阵列接收观测、AoA 标签、可选 spatial spectrum | AoA 或 spatial spectrum | full paths 可派生 AoA；raw waveform 默认未存 | 仿真系统：raw rx signal / array snapshot、AoA label、spatial spectrum label |
| MetaLoc | 多场景 CSI/CFR fingerprint | RX/UE 坐标；support/query 组织 | 多场景 HDF5 + 坐标即可 | Dataloader：scene task、support/query、few-shot split |

---

## 3. 当前仿真系统已经能支持什么

当前 HDF5 原料已经比较完整，能直接支撑多数 CSI/CFR-only 定位模型。

### 3.1 已有核心监督与输入

| 数据 | 用途 | 状态 |
|---|---|---|
| `tx_positions_m` | BS/TX 坐标，服务多 BS、SigMap、Floor-Plan-Aided、LWLM | 已有 |
| `rx_positions_m` | 定位监督标签 | 已有 |
| `tx_labels`, `rx_labels` | 样本索引与可读标签 | 已有 |
| `cfr` | 理想 CFR 真值，服务 KNN/WKNN/MPRI/SigMap/SWiT/LWLM | 已有 |
| `cfr_snapshots` | 多时间快照 CFR，服务动态/多快照模型 | 条件已有 |
| `cfr_est` | PHY 观测/估计 CFR，服务更接近真实观测的 CSI baseline | 已有 |
| `cir_coefficients`, `cir_delays_s`, `cir_valid` | CIR、ToA、路径延迟派生 | 已有 |
| `rssi_dbm`, `snr_db` | RSS/RSS-like、观测质量、Floor-Plan-Aided 简化版 | 已有 |
| `los_exists`, `nlos_exists`, `geometric_path_count` | LoS/NLoS 分层评估、路径复杂度分析 | 已有 |
| `/paths/samples` | 可视化与快速路径分析 | 已有 |
| `/paths/full` | AoA/AoD/ToA/完整路径派生 | 条件已有，需要打开保存 |

### 3.2 现阶段可以直接跑的模型

| 模型 | 说明 |
|---|---|
| KNN/WKNN | dataloader 展平 CFR/RSSI 即可。 |
| MPRI | dataloader 把 CFR 转成图像输入即可。 |
| SigMap w/o map | dataloader 把 CFR token 化即可。 |
| SWiT | 需要构造无标签预训练样本和少标签 fine-tune 样本。 |
| MetaLoc | 如果已经有多个场景或多个 TX layout，可以由 Dataloader 组织成 meta-learning task。 |

---

## 4. 需要补充的字段：按系统归属划分

## 4.1 平面图/地图系统需要补充

这部分不要强行写进 Sionna 仿真逻辑，但必须和仿真 HDF5 通过 `scene_id` / `map_id` 对齐。

| 字段 | 类型/形状建议 | 用途 | 服务模型 |
|---|---|---|---|
| `scene_id` | string | 关联仿真场景、平面图、3D mesh | 所有跨场景模型 |
| `map_id` | string | 地图版本 ID | SigMap w/ map、Floor-Plan-Aided |
| `floor_plan_path` | string | 平面图文件路径 | Floor-plan/map 模型 |
| `floor_plan_image` | optional image tensor | 直接存图像，或只存路径 | Floor-Plan-Aided、你的多模态模型 |
| `world_to_pixel` | 3×3 或 2×3 float matrix | 世界坐标到像素坐标 | floor-plan crop、heatmap 标签 |
| `pixel_to_world` | 3×3 或 2×3 float matrix | 像素坐标还原世界坐标 | heatmap 输出恢复 |
| `pixels_per_meter` | float | 尺度统一 | 地图输入、裁剪窗口 |
| `origin_xy_m` | [2] float | 平面图原点对应世界坐标 | 坐标对齐 |
| `floor_z_m` | float | 平面图对应高度层 | 多楼层扩展 |
| `mesh_file` | string | 3D 模型文件路径 | SigMap w/ map、几何 prompt |
| `mesh_coordinate_system` | string | mesh 坐标系说明 | 防止坐标错位 |
| `mesh_transform_to_world` | 4×4 float matrix | mesh 到仿真世界坐标变换 | 3D map prompt |

最低必须补：`scene_id`、`floor_plan_path`、`world_to_pixel`、`pixel_to_world`、`pixels_per_meter`、`origin_xy_m`。

---

## 4.2 数据转换器 / Dataloader 负责补充

这些字段本质上不是物理仿真结果，而是训练组织信息。

| 字段 | 说明 | 服务模型 |
|---|---|---|
| `split` | train / val / test | 所有模型 |
| `domain_split` | seen_scene / unseen_scene | 跨场景实验 |
| `support_query_role` | support / query | MetaLoc、few-shot |
| `label_ratio` | 少样本比例，例如 1%, 5%, 10% | SWiT、LWLM、MetaLoc |
| `scene_task_id` | meta-learning task ID | MetaLoc |
| `tx_layout_id` | 区分不同 TX/BS 布局 | SigMap、LWLM、跨 BS 泛化 |
| `rx_grid_id` | 区分不同 RX 采样网格 | 数据复现与泛化 |
| `sample_id` | 单样本唯一 ID | 所有模型 |
| `batch_view` | single-BS / multi-BS / all-TX-stacked | LWLM、SigMap、你的模型 |
| `feature_profile_id` | 当前输入特征构造方式 | 模型复现实验管理 |

建议：这些不要写入仿真原始 HDF5 的核心 group，可以写成单独的 `manifest.jsonl` 或由 dataset converter 生成。

---

## 4.3 仿真系统需要新增或标准化的派生物理量

这部分属于 C。它们来自 RT/PHY 真值，建议在仿真系统中统一计算并写入 HDF5，避免每个 dataloader 重复实现、口径不一致。

建议新增 group：`/derived`。

### 4.3.1 距离与 RTT-like

| 字段 | Shape | 来源 | 说明 | 服务模型 |
|---|---|---|---|---|
| `/derived/geometric_distance_m` | [tx, rx] | TX/RX 坐标 | 欧氏距离 | Floor-Plan-Aided、range baseline |
| `/derived/los_distance_m` | [tx, rx] | LoS path 或几何距离 | LoS 存在时的直达距离；无 LoS 可置 NaN | ToA/range baseline |
| `/derived/first_path_delay_s` | [tx, rx] | full paths 或 CIR delays | 最早到达路径 delay | ToA、LWLM |
| `/derived/strongest_path_delay_s` | [tx, rx] | path gain + delay | 最强路径 delay | NLoS 分析、ToA 变体 |
| `/derived/rtt_like_m` | [tx, rx] | delay × c，可选加偏置/噪声 | 只代表传播时延或 synthetic ranging，不是真实 802.11mc RTT | Floor-Plan-Aided 简化版 |
| `/derived/rtt_like_s` | [tx, rx] | delay | 合成 RTT-like 时间量 | Floor-Plan-Aided 简化版 |

注意：`rtt_like` 不能直接写成真实 WiFi RTT。真实 RTT 含协议、硬件时钟、MAC delay、芯片偏置等。第一阶段可命名为 `rtt_like` 或 `propagation_range`。

### 4.3.2 AoA / AoD 标签

| 字段 | Shape | 来源 | 说明 | 服务模型 |
|---|---|---|---|---|
| `/derived/los_aoa_azimuth_rad` | [tx, rx] | LoS path | LoS 到达方位角 | MoD-DNN/AoA baseline |
| `/derived/los_aoa_zenith_rad` | [tx, rx] | LoS path | LoS 到达天顶角 | MoD-DNN/AoA baseline |
| `/derived/strongest_aoa_azimuth_rad` | [tx, rx] | 最强 path | NLoS 场景可用 | AoA baseline |
| `/derived/strongest_aoa_zenith_rad` | [tx, rx] | 最强 path | NLoS 场景可用 | AoA baseline |
| `/derived/first_path_aoa_azimuth_rad` | [tx, rx] | 最早 path | ToA/AoA 联合定位 | LWLM、MoD-DNN |
| `/derived/path_selection_policy` | string | 配置 | `los` / `first` / `strongest` / `all_paths` | 实验可复现 |

### 4.3.3 LoS/NLoS 和难度分析

| 字段 | Shape | 来源 | 说明 | 服务模型 |
|---|---|---|---|---|
| `/derived/los_flag` | [tx, rx] | `los_exists` | 是否 LoS | 分层评估 |
| `/derived/nlos_flag` | [tx, rx] | `nlos_exists` | 是否存在 NLoS | 分层评估 |
| `/derived/path_count` | [tx, rx] | path count | 有效路径数 | 难度分析 |
| `/derived/path_power_db` | [tx, rx] | path power | 总路径功率 | RSS/range 分析 |
| `/derived/link_valid_mask` | [tx, rx] | path + observation | 是否有效链路 | 所有模型 |

### 4.3.4 地图辅助裁剪所需派生量

这部分由仿真系统和平面图系统共同支持。

| 字段 | Shape | 来源 | 说明 | 服务模型 |
|---|---|---|---|---|
| `/derived/tx_rx_midpoint_m` | [tx, rx, 2] | TX/RX 坐标 | AP-RX 裁剪中心候选 | Floor-Plan-Aided |
| `/derived/tx_rx_bearing_rad` | [tx, rx] | TX/RX 坐标 | AP→RX 方向角 | floor-plan crop/rotate |
| `/derived/tx_rx_distance_m` | [tx, rx] | TX/RX 坐标 | 裁剪宽度参考 | Floor-Plan-Aided |

---

## 4.4 D：raw waveform / array observation 能否由 Sionna 得到？

结论：**可以得到一部分，而且是官方 PHY 机制支持的；但 spatial spectrum label、硬件相位误差这类不是 Sionna 自动输出，需要我们自己派生或注入。**

### 4.4.1 Sionna 能直接得到的内容

| 数据 | 是否能由 Sionna 得到 | 说明 |
|---|---|---|
| TX frequency-domain OFDM grid | 可以 | PUSCHTransmitter 在 `output_domain="freq"` 时输出频域 resource grid。 |
| TX time-domain waveform | 可以 | PUSCHTransmitter 在 `output_domain="time"` 时输出时域信号。 |
| RX frequency-domain observation `y_freq` | 可以 | ApplyOFDMChannel 输入 `x` 和 `h_freq`，输出接收端频域观测 `y`。 |
| RX time-domain observation `y_time` | 可以 | ApplyTimeChannel 输入时域信号和时域信道响应，输出接收时域信号。 |
| LS channel estimate `h_hat` | 可以 | LSChannelEstimator / PUSCHLSChannelEstimator 可以从接收 resource grid 和噪声方差估计 channel。 |
| channel estimation error variance | 可以 | LSChannelEstimator / PUSCHLSChannelEstimator 输出 err_var。 |
| decoded bits / CRC | 可以 | PUSCHReceiver 输出 decoded bits，可选 TB CRC status。 |

### 4.4.2 Sionna 不会自动给，但可由我们派生

| 数据 | 处理方式 | 服务模型 |
|---|---|---|
| array snapshot matrix | 从 `y_freq` 或 `y_time` 按 RX antenna × snapshot/OFDM symbol 组织 | MoD-DNN、MUSIC |
| spatial spectrum label | 用 AoA 标签 + 阵列 steering vector 生成，或用 MUSIC/beam scan 计算 | MoD-DNN |
| coarray spatial spectrum | 需要按 MoD-DNN 的阵列/协阵列定义自己实现 | MoD-DNN |
| hardware phase error profile | Sionna 不会真实模拟 gNB 硬件相位误差；需要自定义 impairment | MoD-DNN |
| IQ imbalance | 当前 impairment 契约里有 config 摘要，但如果要真实作用到波形，需要实现具体模型 | PHY realism |

### 4.4.3 建议新增 HDF5 字段

建议新增或扩展 `/waveform` 与 `/array`。

#### `/waveform` 建议补充

| 字段 | Shape | 说明 |
|---|---|---|
| `/waveform/tx_grid` | [snap, tx, tx_ant, ofdm_symbol, subcarrier] | 发送端频域 resource grid |
| `/waveform/rx_grid` | [snap, rx, rx_ant, ofdm_symbol, subcarrier] | 接收端频域观测 y_freq |
| `/waveform/tx_time` | [snap, tx, tx_ant, time_sample] | 发送端时域波形，可选保存 |
| `/waveform/rx_time` | [snap, rx, rx_ant, time_sample] | 接收端时域波形，可选保存 |
| `/waveform/noise_variance` | [snap, tx, rx] 或 broadcastable | 噪声方差 |
| `/waveform/pilot_grid` | same as tx_grid or sparse index | 导频结构 |

#### `/array` 建议新增

| 字段 | Shape | 说明 |
|---|---|---|
| `/array/rx_snapshot_matrix` | [snap, tx, rx, rx_ant, observation] | AoA/MUSIC/MoD-DNN 输入矩阵 |
| `/array/aoa_label_rad` | [tx, rx, 2] | 方位角/天顶角标签 |
| `/array/spatial_spectrum_label` | [tx, rx, angle_bin] 或 [tx, rx, az_bin, el_bin] | MoD-DNN 的监督标签 |
| `/array/angle_grid_rad` | [angle_bin] 或 [az_bin, el_bin, 2] | 空间谱角度网格 |
| `/array/phase_error_profile` | [rx_ant] 或 [rx, rx_ant] | 合成硬件相位误差，可选 |

---

## 5. 各系统需要改动的清单

## 5.1 Sionna 仿真系统改动

优先级从高到低：

| 优先级 | 改动 | 目的 | 服务模型 |
|---|---|---|---|
| P0 | 打开并标准化 `save_full_paths=true` 的输出路径 | ToA/AoA/路径级监督 | LWLM、MoD-DNN |
| P0 | 新增 `/derived` group | 统一距离、ToA、AoA、LoS/NLoS 口径 | LWLM、Floor-Plan-Aided、MoD-DNN |
| P0 | 新增 `rtt_like_m/s`、`geometric_distance_m` | 支持 range/RTT-like baseline | Floor-Plan-Aided 简化版 |
| P0 | 新增 first/strongest/LoS path 派生标签 | 支持 ToA/AoA 任务 | LWLM、MoD-DNN |
| P1 | 保存 `tx_grid`、`rx_grid` | 支持 waveform/array snapshot | MoD-DNN、PHY realism |
| P1 | 保存 `tx_time`、`rx_time` 可选 | 支持 raw IQ 与时域处理 | MoD-DNN、底层 PHY |
| P1 | 新增 `/array` group | 标准化 AoA/MUSIC/MoD-DNN 输入与标签 | MoD-DNN |
| P2 | 硬件相位误差 / IQ imbalance 等自定义 impairment | 增强真实感 | MoD-DNN、真实 PHY |

## 5.2 平面图生成/地图系统改动

| 优先级 | 改动 | 目的 | 服务模型 |
|---|---|---|---|
| P0 | 输出 `floor_plan_path` | 地图输入索引 | SigMap w/ map、Floor-Plan-Aided |
| P0 | 输出 `world_to_pixel`、`pixel_to_world` | 坐标对齐 | 所有 floor-plan 模型 |
| P0 | 输出 `pixels_per_meter`、`origin_xy_m` | 尺度统一 | crop/heatmap |
| P1 | 输出 `mesh_file`、`mesh_transform_to_world` | 3D map prompt | SigMap w/ map |
| P1 | 可选输出 wall mask / obstacle mask | 简化 floor plan 输入 | 你的模型、Floor-Plan-Aided |

## 5.3 数据转换器改动

| 优先级 | 改动 | 目的 | 服务模型 |
|---|---|---|---|
| P0 | CFR feature builder | 幅度/相位、实/虚、幅相交织、flatten | KNN/WKNN/MPRI/SigMap/SWiT |
| P0 | CFR image builder | 生成 MPRI 输入 | MPRI |
| P0 | token builder | 生成 Transformer 输入 | SigMap、SWiT、LWLM |
| P0 | floor-plan crop/rotate builder | 生成 AP-RX 图像 patch | Floor-Plan-Aided |
| P1 | multi-BS sample builder | 单 BS、多 BS、all-TX stack | LWLM、SigMap |
| P1 | AoA/ToA task builder | 生成 ToA/AoA 任务样本 | LWLM、MoD-DNN |

## 5.4 Dataloader 改动

| 优先级 | 改动 | 目的 | 服务模型 |
|---|---|---|---|
| P0 | scene-level split | 跨场景评估 | 所有模型 |
| P0 | random-point split | 常规 seen-scene 评估 | 所有模型 |
| P0 | support/query split | few-shot/meta-learning | MetaLoc、SWiT、LWLM |
| P1 | tx_layout split | 跨 TX/BS 布局泛化 | SigMap、LWLM |
| P1 | label ratio sampler | 少样本实验 | MetaLoc、SWiT、LWLM |
| P1 | valid-link filtering | 去掉无有效路径样本 | 所有模型 |

---

## 6. 第一阶段最低可行数据版本

如果要最快支撑第一阶段模型库，建议先完成以下字段。

### 6.1 仿真 HDF5 必须有

| 字段 | 必要性 |
|---|---|
| `/topology/tx_positions_m` | 必须 |
| `/topology/rx_positions_m` | 必须 |
| `/frequency/frequencies_hz` | 必须 |
| `/channel/truth/cfr` | 必须 |
| `/observation/cfr_est` | 建议 |
| `/observation/rssi_dbm` | 建议 |
| `/channel/truth/cir_coefficients` | 建议 |
| `/channel/truth/cir_delays_s` | 建议 |
| `/paths/full/*` | 对 LWLM/MoD-DNN 必须 |
| `/derived/geometric_distance_m` | 必须 |
| `/derived/first_path_delay_s` | 建议 |
| `/derived/strongest_path_delay_s` | 建议 |
| `/derived/rtt_like_m` | Floor-Plan-Aided 简化版必须 |
| `/derived/los_aoa_*`、`strongest_aoa_*` | MoD-DNN/LWLM 建议 |

### 6.2 地图系统必须有

| 字段 | 必要性 |
|---|---|
| `scene_id` | 必须 |
| `floor_plan_path` | 必须 |
| `world_to_pixel` | 必须 |
| `pixel_to_world` | 必须 |
| `pixels_per_meter` | 必须 |
| `origin_xy_m` | 必须 |
| `mesh_file` | SigMap w/ map 建议 |
| `mesh_transform_to_world` | SigMap w/ map 建议 |

### 6.3 转换器/Dataloader 必须支持

| 功能 | 服务模型 |
|---|---|
| CFR flatten | KNN/WKNN |
| CFR image | MPRI |
| CFR token | SigMap/SWiT/LWLM |
| single-BS / multi-BS 切换 | LWLM/SigMap |
| map crop/rotate | Floor-Plan-Aided |
| train/val/test split | 所有模型 |
| support/query split | MetaLoc |
| scene-level split | 跨场景实验 |

---

## 7. 推荐执行顺序

### 第一步：先补仿真派生字段 C

1. 打开并保存 full paths。
2. 新增 `/derived/geometric_distance_m`。
3. 新增 `/derived/first_path_delay_s`、`strongest_path_delay_s`。
4. 新增 `/derived/rtt_like_m`。
5. 新增 AoA 标签：LoS、first path、strongest path 三套。
6. 新增 `/derived/link_valid_mask`。

### 第二步：补地图对齐 A

1. 平面图系统输出 `scene_id`、`floor_plan_path`。
2. 输出 `world_to_pixel`、`pixel_to_world`。
3. 输出 `pixels_per_meter`、`origin_xy_m`。
4. 可选输出 `mesh_file`、`mesh_transform_to_world`。

### 第三步：做数据转换器和 Dataloader B

1. CFR feature builder。
2. MPRI image builder。
3. Transformer token builder。
4. Floor-plan crop builder。
5. scene split / support-query split。

### 第四步：补 D 的可选波形字段

1. 保存 `tx_grid`、`rx_grid`。
2. 需要时保存 `tx_time`、`rx_time`。
3. 构造 `rx_snapshot_matrix`。
4. 根据 AoA 标签生成 `spatial_spectrum_label`。

---

## 8. 最终判断

第一阶段的数据准备核心不是“让仿真系统直接输出每个模型的最终输入 shape”，而是让仿真系统输出稳定、可追溯、物理口径一致的原料。

优先补三件事：

1. **路径派生标签**：distance / ToA / AoA / LoS-NLoS / RTT-like。
2. **地图坐标对齐**：floor plan 与 Sionna 世界坐标的转换矩阵。
3. **训练组织解耦**：scene split、support/query、label ratio 全部交给转换器和 Dataloader。

完成这些后，KNN、WKNN、MPRI、SigMap w/o map、SWiT、LWLM、SigMap w/ map、Floor-Plan-Aided 简化版、MoD-DNN 简化版、MetaLoc 都可以在同一套数据原料上逐步接入。
