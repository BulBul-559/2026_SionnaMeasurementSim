# Front3D CFR Truth Fallback Stall Incidents 2026-06-20

本记录用于复盘 3000 场景 Front3D 0p5 CFR truth-only 生产队列中反复出现的
fallback inner worker 卡住问题。本文只记录已在运行过程中观察到的场景编号和症状，
不作为最终根因结论。

## 背景

运行模板：

```text
config/tasks/nr_srs_64prb_cfr_truth_only.yaml
```

目标 index：

```text
data/front3d_full/indices/small_normal_3000_panel0p5_seed2026.jsonl
```

输出根目录：

```text
outputs/front3d_small_normal_3000_panel0p5_cfr_truth_only
```

关键配置：

- `output.products: ["cfr_truth"]`
- `output.sharding.shard_size: 20`
- `output.sharding.gpu_scheduler.enabled: true`
- `output.sharding.gpu_scheduler.cross_scene_pipeline: true`
- `output.sharding.fallback.isolation_mode: "on_failure"`
- `output.sharding.recycle_workers: true`

## 症状

典型表现：

- GPU 显存长期被某个 fallback outer worker 持有。
- `nvidia-smi` 显示对应 GPU utilization 接近或等于 0%。
- outer worker 下有一个 inner 子进程，`CUDA_VISIBLE_DEVICES=<gpu>`，CPU 约 100%。
- 目标 scene 的大多数 shard 已经写出，只剩某个父 shard 或 fallback 子 shard 长时间不收口。
- 对 inner 子进程发送 `kill -TERM <inner_pid>` 后，outer worker 通常会继续触发更细粒度 fallback，
  例如从 `result_007.h5` 改为写出 `result_007_00_00.h5`、`result_007_00_01.h5` 等，
  scene 随后可以完成。

复盘时要区分两件事：

- 普通 fallback：某个父 shard 被拆分，但进程仍在正常跑 GPU kernel。
- fallback stall：inner 子进程长期 CPU 100%、GPU 0%，且没有继续产出 H5/manifest。

## 已确认场景

下表记录的是运行期间人工观察到过 fallback stall 症状、或在 TERM 后完成收口的场景。
`run_index` 是 3000 场景 JSONL index 中的 `index`；`scene_key` 同时对应 Front3D
本地目录名。

| run_index | scene_key | class | UE | BS | 受影响父 shard | 观察到的 GPU/PID | 结果 |
|---:|---|---|---:|---:|---|---|---|
| 341 | `front3d_0745` | normal_room | 243 | 10 | `007` | GPU0, inner `4017450`, outer `4016856` | TERM inner 后写出 `result_007_00_00` 至 `result_007_01_01`，scene completed |
| 393 | `front3d_0883` | normal_room | 221 | 12 | `003` | GPU7, inner `4106747`, outer `4105779` | TERM inner 后写出 `result_003_00_00` 至 `result_003_01_01`，scene completed |
| 1908 | `front3d_4285` | normal_room | 322 | 13 | `000` | GPU1, inner `2838906`, outer `2837870` | TERM inner 后继续 fallback，scene completed |
| 2044 | `front3d_4587` | normal_room | 390 | 13 | `002` | 与 2026-06-20 09:31 左右的 GPU0/1/2/7 批量 stall 同窗口；inner PID 批量处理为 `1789992`, `1845593`, `1846088`, `3161889` | 后续写出 `result_002_00_00` 至 `result_002_01_01`，scene completed |
| 2549 | `front3d_5728` | normal_room | 399 | 16 | `007` | 与 2026-06-20 09:31 左右的 GPU0/1/2/7 批量 stall 同窗口；inner PID 批量处理为 `1789992`, `1845593`, `1846088`, `3161889` | 后续写出 `result_007_00_00_00`、`result_007_00_00_01`、`result_007_00_01`、`result_007_01_00`、`result_007_01_01`，scene completed |
| 2567 | `front3d_5773` | normal_room | 226 | 7 | `002`, `004` | GPU0 曾观察到 inner `2114800` CPU 100%、GPU0 util 0；同 scene 早先也在批量 stall 窗口内 | 后续写出 `result_002_*` 和 `result_004_*` fallback 子 shard，scene completed |
| 2904 | `front3d_6586` | normal_room | 319 | 11 | `002` | GPU0, inner `2643004`, outer `2641338`；观察时 CPU 100%、GPU0 util 0 | 后续写出 `result_002_*` fallback 子 shard，scene completed |

截至 2026-06-20T12:28:55+08:00，以上场景均已在
`scene_index_run_manifest.jsonl` 中登记为 `completed`。

## 需继续跟踪的疑似项

这些项在观察窗口中出现过相近症状或未收口状态，但当时尚未有足够信息把具体 PID
稳定映射到单个 scene。后续复盘应结合最终 manifest、run log 和硬件采样确认。

| run_index | scene_key | class | UE | BS | 观察状态 | 需补证点 |
|---:|---|---|---:|---:|---|---|
| 2909 | `front3d_6604` | normal_room | 390 | 15 | 2026-06-20T12:28 左右尚无 scene manifest；已有 `result_012_*`、`result_013_*` fallback 子 shard，父 shard `003` 尚未收口 | 对齐最终 completed/failed 状态；确认当时 GPU1/GPU2/GPU7 inner PID 是否对应本 scene |
| 2910 | `front3d_6609` | normal_room | 275 | 11 | 2026-06-20T12:28 左右尚无 scene manifest；已有父 shard `001` 的 fallback 子 shard | 判断是否只是普通 fallback 尾部等待，还是同类 inner stall |
| 2946 | `front3d_6706` | normal_room | 283 | 12 | 2026-06-20T13:17 左右是全队列最后一个未登记场景；缺父 shard `005`；GPU5 上 outer `2784843`、inner `2785839`，inner 已运行约 1h35m、CPU 100%、GPU5 上 Sionna 显存约 16.4 GiB | 当前 open incident；TERM inner 后确认是否会继续 fallback 并完成 summary |

## 现场处理口径

如果再次遇到同类现象，优先只处理 inner 子进程：

```bash
kill -TERM <inner_pid>
```

不要优先 kill outer worker 或主 `run-scene-index` driver。outer worker 通常负责捕获
inner 退出并继续拆分 fallback；杀 outer 更容易让整 scene 失败，或留下更难追踪的
manifest 缺口。

现场判断建议：

1. 用 `nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu` 看显存和 util。
2. 用 `nvidia-smi pmon -c 1` 找占卡 PID。
3. 用 `ps -o pid,ppid,stat,etime,pcpu,cmd -p <pid>` 找 outer/inner 关系和运行时长。
4. 用 `/proc/<pid>/environ` 确认 `CUDA_VISIBLE_DEVICES`。
5. 看目标 run 的 `results/result_*.h5` 和 `manifest/manifest_*.json` 是否仍在增长。

## 初步假设

目前更像是少数重 shard 在 Dr.Jit / CUDA 内存压力或 fallback 隔离路径中进入了
inner worker 长时间无 GPU kernel 的状态。已观察到的共同点是：

- 场景通常是 normal_room，UE/BS 数较高或 path 数较重。
- 受影响 shard 在 TERM inner 后能通过更细 fallback 子 shard 完成，说明输入 scene
  本身不一定不可仿真。
- 频繁人工 TERM 会影响总 wall time 和调度利用率，需要后续做自动 stall 检测与
  inner worker timeout。

后续复盘建议：

- 在 fallback attempt 级别记录 scene index、parent shard、fallback level、GPU、outer PID、
  inner PID、开始/结束时间和退出原因。
- 为 isolated inner worker 增加 wall-time timeout；超时后由 outer 自动 TERM 并继续二分。
- 区分 CUDA OOM、Dr.Jit unrecoverable error、OBJ parse failure、inner stall 四类失败原因。
- 对 `on_failure + recycle_workers` 下的 fallback 路径补充长跑 watchdog 测试。
