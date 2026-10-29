# DICHASUS-like CFR v1：339 场景全量运行

日期：2026-10-28/29。状态：**已完成**：339 场景、934,128 条链路全部仿真、后处理、验收通过。

按用户要求把冻结的 [DICHASUS-like CFR v1](../proposals/dichasus_like_cfr_v1.md) 从三场景试跑
推到 339 个 Front3D 测试场景全量。物理配置完全沿用冻结模板，只改部署参数。

## 部署参数

配置文件 `outputs/front3d_339_dichasus_like_cfr_v1/full_config.yaml`，相对
`pilot_three/pilot_config.yaml` 只有 6 个键变化（已逐键 diff 核对）：

| 键 | 试跑 | 全量 |
|---|---|---|
| `output.sharding.gpu_ids` | `[2,3]` | `[0,1,2,3,4,5,6,7]` |
| `output.sharding.parallel_workers` | 2 | 8 |
| `output.sharding.gpu_scheduler.free_memory_threshold` | 0.6 | **0.85** |
| `output.sharding.gpu_scheduler.scan_interval_s` | 0.2 | 5.0 |
| `output.sharding.postprocess.async_write` | false | true |
| `output.sharding.postprocess.max_pending_writes` | 16 | 32 |

0.85 是用户要求的排队语义：分片只在某张卡**显存 ≥85% 空闲**时才会被调度，因此本任务
不与同时在跑的 VisionRadioLoc 训练/评估抢卡；有几张卡空出来就用几张，卡全空时 8 张并用。

## 规模

| 项 | 值 |
|---|---:|
| 场景 | 339（`m117_seed2026_scene_level` test 名单） |
| BS / UE | 3,283 / 85,124 |
| 原始链路 | 934,128 |
| UE 分片（shard_size=20） | 4,414 |
| 预期 scene H5（complex64、不压缩） | 约 172 GiB |
| 预期原始分片（mixed） | 约 83 GiB |

## 执行

单进程链式脚本 `outputs/front3d_339_dichasus_like_cfr_v1/run_all.sh`，后台 `setsid` 常驻：

```text
stage1  run-scene-index        full_config.yaml + prepared/test_scene_index.jsonl -> raw_runs/
stage2  postprocess-cfr-scenes raw_runs/ -> dataset/（--workers 8 --cfr-compression none）
stage3  finalize_full.py       dataset/ 验收 + splits/m117_seed2026_scene_level/test_links.pt
```

- 断点续跑：`run-scene-index` 默认 `skip_existing`，按
  `raw_runs/runs/<scene>/manifest/manifest.json` 判断场景完成；重跑同一脚本从断点继续。
- 失败策略：不加 `--stop-on-failure`，单场景失败不终止队列，事后按缺失场景重跑并复核。
- 进度：`outputs/front3d_339_dichasus_like_cfr_v1/monitor_progress.py`（场景/链路/shard 计数
  与逐卡显存），`run_all.sh` 每 15 分钟写入 `console/progress.log`。

## 试跑基线（用于估算）

`pilot_three/` 三场景 6,601 条链路、37 分片、双卡 **184 s**（约 35.9 链路/s）。
按此推算全量 8 卡约 2 小时计算时间，另加 stage2 的约 250 GiB 读写；实际受磁盘竞争影响。

## 运行与暂停（2026-10-28 22:51–22:56）

排队约 21 分钟后 GPU 放行，stage1 于 22:51:26 开始出分片；**22:56 按用户要求暂停后台任务**。

| 项 | 值 |
|---|---:|
| 已完成场景 | 5 / 339（front3d_0015、0055、0080、0095、0119） |
| 已完成分片 | 52（14/8/10/11/9） |
| 已完成链路 | 7,503 |
| 单场景耗时 | 21.9–32.3 s，合计 143.5 s（约 52 链路/s 并行） |
| 有效输出 | `raw_runs/` 910 MB |

暂停处理：

- 先停 `watchdog.sh`（避免它把链条拉起），再停 `run_all.sh` 与 `run-scene-index` 进程。
- 第 6 个场景 `front3d_0149` 的 3 个已落盘分片（HDF5 均可读但场景未 finalize）已整目录删除，
  避免续跑时复用未完成分片；其余 5 个场景的 `manifest/manifest.json` 完整保留。
- `console/STOP_WATCHDOG` 已清除；`console/DONE`/`FAILED` 均未产生。
- 当前无任何 339 任务进程，GPU 全部归还给训练/评估。

恢复方式：重新执行 `outputs/front3d_339_dichasus_like_cfr_v1/run_all.sh`（同样 `setsid` 常驻），
`skip_existing` 会跳过上述 5 个场景，从 front3d_0149 继续；需要自动重启守护时再一并启动 `watchdog.sh`。

## 最终结果（2026-10-29 12:24）

| 项 | 值 |
|---|---:|
| 场景 | 339 / 339 |
| BS / UE | 3,283 / 85,124 |
| 原始链路 | 934,128 |
| 分片文件 | 4,418（计划 4,414；front3d_1913、front3d_5425 各触发一次 OOM fallback 二分） |
| 有效链路 | 795,587 |
| 全零链路 | 138,541（14.83%，按官方身份保留并标 `valid_mask=false`） |
| m117 配对链路 | 783,687（与原 test 名单一致） |
| 其中新 CFR 全零 | 23,404（2.99%），涉及 298 个场景 |
| `dataset/` | 172 GiB，`/cfr` complex64 不压缩，contract `sionna_cfr_scene_dataset_manifest` |
| 功率范围 | −300 dB（全零）… −7.38 dB |

耗时：

| 阶段 | 区间 | 耗时 |
|---|---|---:|
| stage1 仿真 | 09:29:02–11:20:56（+ 13 s 补跑 1 场景） | 1 h 52 min |
| stage2 后处理 | 11:30:22–11:46:58 | 16.6 min |
| stage3 验收 + 索引 | 11:49:52–12:24 | 34 min |
| 模型侧读取校验 | 12:26 | 通过 |

GPU 排队：22:30 入队，22:51 首次放卡；01:00 暂停时已完成 5 场景；09:29 全量 8 卡恢复后 1 h 52 min 跑完。

## 运行中修掉的问题

1. **断点续跑的完成判定**：`finalize_full.py` 只把 `completed` 当完成，续跑时 338 个场景是
   `skipped`，被误判为未完成并中止。改为 `completed`/`skipped` 均视为完成。
2. **`test_links.pt` 元数据**：自建 metadata 丢了 `kind: scene_dataset_link_index_plan`，
   VisionRadioLoc `CfrLinkDataset` 直接拒绝加载。改为继承原计划 metadata 再覆盖场景数/链接数等字段。
3. **label 文件名**：资产里只有 `label_panel_0p5_bs_center_yaw.json`，而读取侧按
   `assets/<scene>/label_panel_0p5.json` 取。现在 finalize 会复制一份规范名（内容一致，已校验 hash）。
4. **偶发 HDF5 文件锁**：`front3d_5806` 6 个分片算完后写 manifest 报 `errno 11 unable to lock`，
   删该未完成场景目录重跑 13 s 通过；4,362 次调度中仅 1 次。

## 验收证据

- 每个场景：原始分片 full-HDF5 schema、UE 覆盖恰好一次、BS 全集、实际逐设备朝向、
  `[links,32,1,768]`、`is_finite`/`nan_count`、scene contract、频率网格
  （3.438 GHz / 30 kHz / 768）、无 waveform/observation/iq/ranging 组。
- 与原始 2988 数据集逐场景对齐：`(ue_id,bs_id)` 集合一致、`ue_pos`/`bs_pos`/`target_xy`/
  `target_room_index` 逐行一致；floorplan/meta hash 一致；geometry/class_mask 复制后 hash 一致。
- m117 身份映射：783,687 行全部落到新 H5 行，保留原 `global_link_id`、原功率、新功率与新 valid 标记。
- 产物：`dataset/splits/m117_seed2026_scene_level/{scene_split.jsonl,test_scenes.txt,test_links.pt}`、
  `paired_links.csv`（73.8 MB）、`summary_full.json`（逐场景 339 条）、`checksums_full.sha256`、
  `model_input_check_full.json`。
- 模型侧：VisionRadioLoc `CfrLinkDataset`（`energy_norm_complex`）成功加载 339 场景、783,687 链路，
  抽检每条 `csi_input` 为 `[2,32,768]` 且有限，collate 输出 `[339,2,32,768]`。

## 评估使用提示

- `scene_split.jsonl` 的 `record_index` 是本目录顺序（= 原 test 名单顺序），
  `source_test_record_index` 保留原 2988 名单位置，两者当前相等。
- 23,404 条 m117 链路的新 CFR 全零（`valid_mask=false`，CFR 全 0、功率 −300 dB）。
  评估时需明确处理方式（剔除、或按无效样本统计），不要默认它们与实采零值等价。

## 收尾确认（2026-10-29 13:03）

- 一致性复核：`stage3 exit=0`、`ALL DONE`，落 `console/DONE`；watchdog 见到 DONE 自行退出，无残留进程。
- `checksums_full.sha256` 四项（`link_index.pt`、`manifest.json`、`scene_split.jsonl`、`test_links.pt`）全部 OK。
- `test_links.pt`：`kind=scene_dataset_link_index_plan`、783,687 行、760,283 条 `valid_mask=true`。
- `model_input_check_full.json` 复核通过：339 场景 / 783,687 链路，`csi_input [2,32,768]`，collate `[339,2,32,768]`。
- 磁盘占用：`dataset/` 172 GiB、`raw_runs/` 82 GiB。
