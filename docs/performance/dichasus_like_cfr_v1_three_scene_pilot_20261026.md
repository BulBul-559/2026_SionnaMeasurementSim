# DICHASUS-like CFR v1：大中小三场景试跑

日期：2026-10-26。状态：三个完整场景已仿真、后处理和验收；339场景全量未启动。

使用[冻结模板和176项审查](../proposals/dichasus_like_cfr_v1.md)。本轮只覆盖部署参数：
`gpu_ids=[2,3]`、`parallel_workers=2`，物理配置不变。未修改原2988数据集或VisionRadioLoc代码。

## 场景选择

仅在原 `m117_seed2026_scene_level` 的339个test场景中选取。面积为label中唯一房间的
`floor_area_m2` 之和；按 `(total_floor_area_m2, scene_key)` 排序，选择
`round((339-1)*q)` 对应的场景，`q=0.1,0.5,0.9`。大中小指整套场景总面积，
不使用原目录的 `small_room/normal_room` 分类，也没有切出单个子房间。

| 大小 | 场景 | 总面积㎡ | 子房间数 | BS | UE | 原始链路 | 原m117测试链路 |
|---|---|---:|---:|---:|---:|---:|---:|
| 小 | front3d_4148 | 42.555 | 5 | 5 | 133 | 665 | 665 |
| 中 | front3d_2508 | 73.344 | 9 | 9 | 222 | 1,998 | 1,810 |
| 大 | front3d_1380 | 119.207 | 15 | 11 | 358 | 3,938 | 2,787 |
| 合计 | 3场景 | — | 29 | 25 | 713 | 6,601 | 5,262 |

源场景、材质、平面图、BS/UE位置/编号/数量/groups均保留。BS面板竖直朝整张地图中心，
UE固定零朝向。局部输入副本位于试跑目录 `inputs/<scene>/label/label_panel_0p5.json`；
内容与上一轮prepared派生label逐字节一致，使用原文件名以适配既有训练侧资产读取。

## 数据和测试入口

实际输出根目录：

```text
/data/sunmeiyuan/projects/sionna/outputs/front3d_339_dichasus_like_cfr_v1/pilot_three/
  selection.json                 # 面积排序规则、源339名单、输入hash、三个场景明细
  pilot_config.yaml              # 冻结模板，仅改GPU部署选项
  pilot_scene_index.jsonl        # 只有三个场景的运行名单，顺序为小/中/大
  inputs/<scene>/                # 派生label、原floorplan/meta及预览副本
  raw_runs/                      # 37个原始H5分片、配置、日志、manifest
  dataset/
    scenes/front3d_4148.h5       # /cfr [665,32,1,768]
    scenes/front3d_2508.h5       # /cfr [1998,32,1,768]
    scenes/front3d_1380.h5       # /cfr [3938,32,1,768]
    assets/<scene>/             # label、floorplan/meta、class_mask、geometry
    manifest.json
    link_index.pt / .parquet
    room_index.pt / .parquet
    splits/m117_seed2026_scene_level/
      scene_split.jsonl         # 仅三个test场景，顺序小/中/大
      test_scenes.txt
      test_links.pt             # 原m117身份集合，映射到新的H5行
  visionradioloc_data.yaml       # 可合并到原评估配置的数据选项
  paired_links.csv               # 原/新global/local索引、UE/BS、功率和当前valid标记
  summary.json                   # 全部数值检查摘要
  model_input_check.json         # 实际VisionRadioLoc loader/预处理检查
  checksums.sha256
  logs/
  prepare.py / finalize.py / check_model_input.py
```

scene H5共约1.30GB（十进制），使用既有 `sionna_cfr_scene_dataset` / `0.2.0` contract，
complex64、`/cfr`不压缩，便于训练侧读取。现有CSI预处理展平后仍为 `[B,2,32,768]`。
模型、checkpoint、候选采样和预处理参数继续使用原评估设置；这里只换数据入口。

`test_links.pt` 保留原m117身份集合，即**原CFR**的 `valid_mask=true` 且
`csi_power_db>=-117`，没有对新CFR重新按功率筛选。它按 `(scene_id,ue_id,bs_id)`
重新映射H5行；`record_index=0/1/2` 对应小/中/大场景。保存新功率、原功率、当前valid、
原global/local link id；`paired_links.csv` 另保留原339名单中的record序号。
原数据集的 `test_links.pt` 不能直接用于新三场景目录。

**5,262条配对测试链路中有81条新CFR全零**：中场景37、大场景44、小场景0。
它们在新H5中为 `valid_mask=false`、功率约-300dB，仍保留在配对名单中，避免更换测试人群。
这是RT结果，本轮未将原因归于某一配置变化。做评估时应同时报告这些无效/全零样本的处理方式。

## 执行和验收

已执行：

```bash
uv run python outputs/front3d_339_dichasus_like_cfr_v1/pilot_three/prepare.py

uv run python -m sionna_measurement_sim.app.cli \
  --config outputs/front3d_339_dichasus_like_cfr_v1/pilot_three/pilot_config.yaml \
  run-scene-index \
  --index-file outputs/front3d_339_dichasus_like_cfr_v1/pilot_three/pilot_scene_index.jsonl \
  --output-root outputs/front3d_339_dichasus_like_cfr_v1/pilot_three/raw_runs \
  --limit 3 --stop-on-failure

uv run python -m sionna_measurement_sim.app.cli postprocess-cfr-scenes \
  --input-root outputs/front3d_339_dichasus_like_cfr_v1/pilot_three/raw_runs \
  --output-root outputs/front3d_339_dichasus_like_cfr_v1/pilot_three/dataset \
  --workers 3 --cfr-compression none

uv run python outputs/front3d_339_dichasus_like_cfr_v1/pilot_three/finalize.py

/home/sunmeiyuan/projects/VisionRadioLoc/.venv/bin/python \
  /data/sunmeiyuan/projects/sionna/outputs/front3d_339_dichasus_like_cfr_v1/pilot_three/check_model_input.py
```

准备工具不覆盖既有结果；`finalize.py --refresh-metadata` 仅用于重建本轮派生索引/检查摘要，
不修改原始RT或scene H5。上述命令是本轮记录，不应为重复测试而重新仿真。

- RT队列：3 completed、0 failed；37/37分片，运行184.18s；后处理3 completed、0 failed。
- 原始分片均通过full HDF5 schema、UE覆盖恰好一次、全部BS保留、实际设备朝向检查。
- 三个scene H5均通过独立scene contract；全部复数CFR有限，shape/频率网格正确。
- 整套6,601条链路的身份和位置与原H5逐项对齐；原5,262条m117身份均映射成功。
- 平面图/meta逐字节一致；class_mask/geometry复制并核对hash；源label/XML等输入hash不变。
- 实际VisionRadioLoc `CfrLinkDataset` 成功加载3个records、5,262条索引；每个场景抽一条链路，
  图像与CSI均有限；`cfr_link_collate_fn` 输出CSI `[3,2,32,768]`。
- 项目ruff及三个输出辅助脚本ruff通过；朝向/离线CFR后处理相关单元测试 **19 passed**。
- 未运行定位模型、未产生新模型性能结论。等待这三个场景的测试反馈后再决定全量任务。

当前任务仅生成输出和实验记录，不改生产代码或HDF5 schema；README/config系统接口保持原义。
