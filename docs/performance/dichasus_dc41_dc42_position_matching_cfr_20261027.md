# DICHASUS dc41/dc42 官方位置匹配 CFR 数据集

日期：2026-10-27。状态：两个数据集仿真、后处理和下游读取检查完成。

## 目标与位置来源

用户要求按官方 `dc41`、`dc42` 采样位置，在同一 DICHASUS DC4x 空大厅中分别生成
新仿真数据集。位置直接读取已 materialize 的官方 HDF5
`DICHASUS_dcxx_CSI_769/scenes/dichasus_dcxx_dc{41,42}.h5` 的 `target_xyz`，
没有重新解析 1.7 GB TFRecords，也没有用网格近似原位置。

每项官方 measurement 对应一个 UE；其 `measurement_id` 顺序直接作为仿真 `ue_id`。
少量完全重复的测量位置被保留，以维持原 measurement 数、时间记录和后续身份配对：

| Run | measurement/UE | BS | 仿真链路 | 唯一XYZ | 重复额外记录 | Z范围m |
|---|---:|---:|---:|---:|---:|---|
| dc41 | 3,282 | 2 | 6,564 | 3,258 | 24 | 0.924899–0.935598 |
| dc42 | 3,308 | 2 | 6,616 | 3,278 | 30 | 0.924200–0.935894 |

位置范围为 dc41 X 8.347746–12.467476、Y 23.337559–29.540590；dc42
X 8.357538–12.430869、Y 23.402643–29.481701 m。原 transmitter 高度按
`target_xyz` 保留，不改成上一轮人工网格的 1.6 m。

## 仿真配置

两套 run 共用上一轮经过自遮挡诊断的 `scene_for_rt/scene.xml`：建筑网格与原始
DC4x 空大厅一致，仅排除会被重复建模成混凝土的四个天线 CAD 对象。BS 位置、
4×8 阵元间距和官方朝向沿用上一轮：

| BS | 官方水平角 | 官方下倾角 |
|---|---:|---:|
| BS0 北阵列 | 108.018080° | 0.881570° |
| BS1 南阵列 | -76.597832° | 0.194769° |

官方 direction 不唯一确定 roll，继续采用 roll=0。UE 为单天线、零朝向。
其余载频/网格/RT 参数与冻结 DICHASUS-like v1 相同：3.438 GHz、768点/30 kHz、
depth4、LoS/镜面反射/折射/绕射开启、漫反射关闭、无 PHY/噪声/损伤。
因此输出是 768-bin 理想 RT CFR；官方对照 H5 保留 769-bin 物理窗，不直接声称频点逐 bin 相同。

部署使用6张空闲GPU、6 workers、每个UE shard 20。各3个真实位置先做 smoke，
两阵列均为 `[3,2,32,1,768]`、6/6非零，所有阵元收到信号。

| Run | shards | RT耗时s | scene H5 bytes |
|---|---:|---:|---:|
| dc41 | 165 | 157.204 | 1,291,945,931 |
| dc42 | 166 | 157.427 | 1,302,178,527 |

两组总耗时约315s，包含实际 RT/HDF5写盘。scene postprocess使用不压缩 complex64 CFR。

## 输出

根目录：

```text
/data/sunmeiyuan/projects/sionna/outputs/dichasus_dc41_dc42_matching_positions_v1/
```

每个run在独立 `dataset/dc41`、`dataset/dc42` 下，均有：

```text
scenes/<scene_id>.h5
assets/<scene_id>/label_panel_official.json
assets/<scene_id>/floorplan_1p60.png
assets/<scene_id>/meta.json
assets/<scene_id>/class_mask.npy
assets/<scene_id>/geometry.npz
manifest.json
link_index.pt / .parquet
room_index.pt / .parquet
paired_official_links.csv
splits/official_positions_all_links_v1/
  scene_split.jsonl
  test_scenes.txt
  test_links.pt
  bs0_test_links.pt
  bs1_test_links.pt
visionradioloc_data.yaml
```

`test_links.pt` 不做功率筛选；单BS plan各保留3,282/3,308条官方位置。
`paired_official_links.csv`记录仿真local row、measurement ID、BS、官方源row、
XYZ、仿真功率/valid、官方valid与769-bin power，供后续逐样本sim2real匹配。
官方原始H5 SHA256记录在 `preparation.json` 和 `summary.json`。

## 验收

- 两个run共329/329分片完成；所有raw HDF5通过full schema。
- 每个run的UE覆盖恰好一次，两个BS均完整；全量13,180个UE–BS身份对唯一。
- 每个仿真`ue_pos`与官方`target_xyz`按ID逐点一致，误差≤1e-6m；
  BS位置和南北朝向逐shard核对。
- 全部CFR有限；dc41 6,564/6,564、dc42 6,616/6,616链路有效，无全零链路。
- 北阵列存在部分链路全零阵元：dc41最少22/32、dc42最少17/32；南阵列均为32/32。
  结果保持原值，没有插值或补值。
- 单BS平均线性CFR功率→dB min/median/max：dc41北
  -96.611/-72.127/-54.147，dc41南 -89.644/-66.531/-57.282；dc42北
  -96.874/-73.212/-54.034，dc42南 -89.165/-66.295/-56.599。
  这些是原始RT CFR功率，不是应用发射功率后的RSSI dBm。
- VisionRadioLoc实际loader读取两套双BS全集和四个单BS计划；样本图像/CSI有限，
  `energy_norm_complex`及collate输出`[5,2,32,768]`；`ue`和`grid`候选策略均通过。
- 平面图来自DICHASUS `stack.npz`；派生`geometry.npz`的`sdf_valid_mask`
  使用其floor occupancy，90687个有效像素。
- 关键输出56项SHA256复核通过；朝向/离线CFR后处理单元测试 **19 passed**；本轮输出脚本ruff通过。
  未改生产代码、schema或通用模板；README/config/sys和TODO无需改变。
  未运行模型训练或定位性能结论。
