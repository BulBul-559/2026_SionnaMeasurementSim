# DICHASUS DC4x 空大厅：官方阵列方向CFR仿真

日期：2026-10-26。状态：完整880 UE×2 BS仿真、后处理和训练侧读取验收完成。

## 本轮配置

在[已确认的大厅0.2m采样](dichasus_dc4x_empty_lobby_label_preview_20261026.md)上，
用户要求改用原始DICHASUS南北阵列的实际方向并开始仿真。所有880个UE及2个BS的
位置、编号、高度和大厅ROI保持不变；UE仍为1.6m、零朝向。

方向与[官方数据说明](https://dichasus.inue.uni-stuttgart.de/datasets/data/dichasus-dcxx/)
及本地DICHASUSFactory的DCXX_ARRAYS常量核对一致：

| BS | 官方direction | yaw（度） | pitch（度） |
|---|---|---:|---:|
| 0 北 | [-0.30928047,0.9508464,-0.01538569] | 108.0180795 | 0.8815699 |
| 1 南 | [0.23178338,-0.97276149,-0.00339935] | -76.5978316 | 0.1947688 |

Sionna使用 `Rz(yaw) Ry(pitch) Rx(roll)`，面板法向为局部+X：
`yaw=atan2(dy,dx)`，`pitch=-atan2(dz,hypot(dx,dy))`。
保留向量中实际下倾；direction不唯一确定roll，本轮roll=0。源模型只做平移，方向无需平移。
这次不再使用此前Front3D方案的整图中心朝向，通用冻结模板保持不变。

其余使用冻结DICHASUS-like v1参数：3.438GHz、每BS 4×8 RX、单TX、0.04364m阵元间距、
768点/30kHz/23.04MHz原NR网格，depth4、LoS/镜面反射/折射/绕射开启、漫反射关闭，
synthetic_array/normalize_cfr/normalize_delays关闭。只计算理想RT CFR，不运行PHY/损伤。
部署使用GPU2/3、2个worker、20 UE/shard，共44个shard。

## 天线CAD自遮挡诊断和场景处理

初次真实smoke使用原导出scene.xml，发现南阵列全零，北阵列仅8/32阵元非零。
该XML引用的合并OBJ把天线外壳、贴片也全部赋为混凝土；接收点位于这些CAD物体中。
派生场景只排除以下四个对象，让RX阵列由Sionna的天线模型表示：

```text
antenna-array-north
antenna-array-north-patches
antenna-array-south
antenna-array-south-patches
```

原scene.obj和原RT资产的1,482个三角形先做坐标多重集一致性检查；排除天线CAD的792个
三角形后，保留690个建筑三角形，坐标多重集与原建筑部分完全一致。墙体、地面、天花、
柱子等未改变，既有统一混凝土材质未改变，源文件未修改。
派生XML/OBJ位于新输出目录的 `scene_for_rt/`，有完整hash及对象列表。

同3个UE、2个BS的真实RT对照：

| 场景 | 非零链路 | 北阵列每链路非零阵元 | 南阵列每链路非零阵元 |
|---|---:|---|---|
| 原合并网格，包含天线CAD | 3/6 | 8/32 | 0/32 |
| 派生建筑网格，排除天线CAD | 6/6 | 32/32 | 32/32 |

正式运行采用后者。两个smoke结果和诊断报告均保留，没有把全零控制结果当成正式数据。

## 输出与读取

输出根目录：

```text
/data/sunmeiyuan/projects/sionna/outputs/dichasus_dc4x_empty_lobby_actual_orientation_v1/
```

`dataset/scenes/dichasus_dc4x_empty_lobby.h5` 使用既有scene contract 0.2.0，
`/cfr [1760,32,1,768]`、complex64、不压缩，346,577,251 bytes。原始44个分片在
`raw_run/`，每个CFR为 `[20,2,32,1,768]`。不合并两BS的天线轴，每条link是一个UE–BS对。

`dataset/splits/lobby_grid_all_links_v1/` 保存单场景test记录、全部1,760条链路索引，
以及 `bs0_test_links.pt`、`bs1_test_links.pt`（各880条）。这是新生成的0.2m测试网格，
未做m117或其他功率过滤，也没有沿用原339场景的链路计划。

`visionradioloc_data.yaml` 提供对应数据入口。训练侧label资产补充
`ue_positions`/`bs_positions` 坐标镜像并修改split记录指向该资产，以兼容VisionRadioLoc旧字段；
顶层point列表及原RT输入/H5保持不变。记录见 `model_asset_compatibility.json`。

复现流程脚本保存在本轮输出目录：prepare.py → 原网格smoke → prepare_rt_scene.py →
排除天线CAD的smoke → activate.py → run-full → postprocess-cfr-scenes → finalize.py →
prepare_model_assets.py → check_model_input.py。默认不覆盖已完成的正式结果。
原预览目录及其hash不改写，正式结果写到独立目录。

## 验收

- 44/44原始H5通过schema；880个UE恰好覆盖一次，2个BS均保留。
- 全部H5的实际设备角度与label一致，重建法向与官方direction一致；UE保持零朝向。
- 1,760个UE–BS身份对完整且唯一；原始分片与scene文件抽取频点逐值一致；位置正确。
- CFR全部有限，1,760/1,760链路有效，无全零链路。每个BS各880条有效链路。
- 北阵列部分链路有全零阵元，非零阵元最少20/32；南阵列全部32/32。保留原值，未填补。
- 平均CFR功率dB（min/median/max）：北 -96.526/-69.729/-49.684；南 -92.533/-69.422/-54.387。
  此为原始CFR的相对功率，不是施加发射功率后的RSSI dBm。
- VisionRadioLoc实际loader加载全部1,760条和两个独立880条计划；6条样本的图像与CSI有限，
  energy_norm_complex预处理及collate输出 `[6,2,32,768]`。候选策略检查使用ue。
- 朝向/离线CFR后处理单元测试19 passed；项目与本轮输出脚本ruff通过。
- 不修改生产代码、schema和通用模板；README/config/sys接口及TODO分类无需改变。
  本轮没有训练模型或运行定位性能评估。
