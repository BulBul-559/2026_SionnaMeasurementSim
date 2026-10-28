# DICHASUS-like CFR v1：339 场景全量运行

日期：2026-10-28。状态：**已入队，等待 GPU 空闲**（结果与验收待补）。

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

## 待补

运行完成后补充：stage1 实际耗时/失败数、stage2 输出大小、`finalize_full.py` 的
`summary_full.json`（有效链路、m117 配对链路、全零链路数）、scene H5 抽样与模型读取复核。
