# DICHASUS dcxx 官方位置理想 CFR 全集

日期：2026-10-27。状态：官方可用10个run、118925个位置和237850条双阵列链路已完成仿真、
后处理及VisionRadioLoc读取验收。

## 范围和组织

用户要求仿真 `dc30` 到 `dc44`。本地官方 materialized HDF5 实际包含
`dc30–dc34` 与 `dc40–dc44` 共10个run；`dc35–dc39` 没有对应文件，因此没有被跳过或遗漏。

统一输出根目录：

```text
/data/sunmeiyuan/projects/sionna/outputs/dichasus_dcxx_official_positions_cfr_v1/
```

目录按职责拆分为 `sources/`、`labels/`、`configs/`、`queues/`、`raw_runs/`、
`datasets/`、`logs/`和`scripts/`。`raw_runs/dcXX`与`datasets/dcXX`是稳定入口，
queue生成的内部目录保留原始运行记录。`dc41`、`dc42`通过相对软链接复用之前的
`dichasus_dc41_dc42_matching_positions_v1`结果；新集合中的路径组织不复制两份大文件。

## 数据与场景

位置直接读取官方materialized HDF5的`target_xyz`；每项measurement对应一个UE，
`ue_id=measurement_id`，南/北阵列各生成一个BS链路。重复的官方位置保留。
UE高度保持官方约0.920–0.939m，不使用人工网格的1.6m。

| Run | 场景 | UE | 链路 | 有效 | 全零 |
|---|---|---:|---:|---:|---:|
| dc30 | dc3x northwest blocker | 28,372 | 56,744 | 55,962 | 782 |
| dc31 | dc3x northwest blocker | 3,857 | 7,714 | 7,714 | 0 |
| dc32 | dc3x northwest blocker | 4,495 | 8,990 | 8,990 | 0 |
| dc33 | dc3x northwest blocker | 17,359 | 34,718 | 34,457 | 261 |
| dc34 | dc3x northwest blocker | 9,870 | 19,740 | 19,708 | 32 |
| dc40 | dc4x empty lobby | 19,216 | 38,432 | 38,310 | 122 |
| dc41 | dc4x empty lobby | 3,282 | 6,564 | 6,564 | 0 |
| dc42 | dc4x empty lobby | 3,308 | 6,616 | 6,616 | 0 |
| dc43 | dc4x empty lobby | 15,480 | 30,960 | 30,918 | 42 |
| dc44 | dc4x empty lobby | 13,686 | 27,372 | 27,331 | 41 |
| 合计 | — | 118,925 | 237,850 | 236,570 | 1,280 |

全零链路表示RT在官方坐标上没有找到有效传播路径，保留原measurement身份和位置，
在scene H5中写为`valid_mask=false`、功率约-300dB；没有按结果重新筛选样本。

dc3x场景从`blend_0001(dc3x)`派生，只排除4个antenna-array外壳/贴片对象；
源网格1506个三角形、保留建筑及blocker714个三角形，坐标多重集与原建筑部分一致。
dc4x复用已经验收的排除天线CAD空大厅场景。南/北BS位置、官方direction和阵元间距
在两套场景中相同。

共同数值口径为3.438GHz、768点/30kHz、4×8 BS/1×1 UE、0.04364m间距，
depth4、LoS/镜面反射/折射/绕射开启、漫反射关闭、synthetic_array和CFR归一化关闭。
只输出理想RT CFR；官方769-bin和仿真768-bin的差异未隐藏。

## 运行

两个复用run已有165/166 shards。其余8个run先以dc30的3个真实位置做smoke：
南/北均3/3链路非零、32/32阵元收到信号。8场景queue dry-run为8 planned、0 failed。

正式队列使用`run-scene-index`跨场景调度、8张GPU、8 workers、20 UE/shard：

```text
completed scenes: 8
completed shards: 5619
failed shards/scenes: 0
wall time: 4411.77s
```

各场景运行耗时：dc30 1128.01s、dc31 149.63s、dc32 174.33s、dc33 674.50s、
dc34 386.43s、dc40 762.72s、dc43 621.05s、dc44 548.91s。
scene后处理使用complex64、`/cfr`不压缩。新增8个scene H5合计约41GB；
`dc41/42`复用不重复计算时，raw_runs约31GB、datasets约42GB。

## 输出和验收

每个`datasets/dcXX/`提供：

```text
scenes/dcXX.h5                         /cfr [link,32,1,768]
splits/official_positions_all_links_v1/
  scene_split.jsonl
  test_scenes.txt
  test_links.pt
  bs0_test_links.pt
  bs1_test_links.pt
paired_official_links.csv
visionradioloc_data.yaml
manifest.json
link_index.pt / link_index.parquet
room_index.pt / room_index.parquet
assets/dcXX/{label,floorplan,meta,class_mask,geometry}
```

根目录的`datasets.yaml`、`catalog.json`、`summary.json`和
`model_input_check.json`汇总全集合。label的长期点属性保留`measurement_id`、
官方time和`gt_interp_age_tachy`；训练侧资产只增加`ue_positions/bs_positions`
兼容镜像，不改变点表或CFR。

- 8个新run共5619/5619 shards完成；10个run全部通过full/scene HDF5 schema。
- 每个run的UE覆盖恰好一次、双BS完整；237850个`(ue_id,bs_id)`身份对唯一。
- scene中的UE XYZ按ID与官方`target_xyz`逐点匹配，误差≤1e-6m；BS位置和实际朝向
  逐shard核对。重复位置保留，`ue_id=measurement_id`。
- 全部CFR finite。有效链路236570条；1280条全零链路按官方位置保留并标记。
- 每个run均生成双BS全集和南/北单BS plan；`paired_official_links.csv`保留源row映射。
- VisionRadioLoc实际loader通过10个run的双BS全集和20个单BS计划；样本图像/CSI finite，
  energy_norm_complex和collate输出正确；dc30、dc40的grid候选geometry也通过。
- 项目ruff通过；相关朝向/后处理单元测试19 passed；未训练模型，未生成定位性能结论。

本轮没有修改生产代码、schema或通用模板，因此README/config/sys契约和TODO分类不变。
建议下一步直接从根目录`datasets.yaml`选择run，并报告全零链路是否从评测中排除。
