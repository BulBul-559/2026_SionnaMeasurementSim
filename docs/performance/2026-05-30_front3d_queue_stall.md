# Front3D 队列卡住记录，2026-05-30

> 历史运行记录。本文记录 Front3D 生产队列中的两次卡住现象、当时的处置方式、
> 以及后续应对策略。它不改变当前默认配置；当前系统事实仍以
> `docs/agent_handoff.md`、`docs/sys/` 和 `config/README.md` 为准。
>
> 2026-06-01 补充：后续本地队列脚本的运行级 artifact 已统一改为 output-local
> 布局，即 `outputs/<run_name>/logs/run.log`、`logs/heatmap.log` 和 `summary.json`。
> 不再建议在 `outputs/` 根目录写 `<run_name>.run.log` 或 `<run_name>_summary.json`
> sidecar。

## 范围

- 队列脚本：
  `outputs/local_runs/front3d_remaining_panel_density_queue/run_queue.sh`
- tmux 会话名：`front3d_remaining_density_queue`
- 工作负载：Front3D SRS 64PRB direct-array，`shard_size=10`，每个任务使用 8 张 GPU。
- 目标任务：多个 Front3D 场景的 `0p1`、`0p2`、`0p5` label 仿真，并为每个完成任务生成
  RSS radio map heatmap。

## 事故一：`front3d_0002 0p2`

### 现象

在 `2026-05-30 01:12 CST` 左右，第 7/14 个任务长时间没有推进：

- 任务：`front3d_0002`，density `0p2`。
- 输出目录：
  `outputs/front3d_20_front3d_0002_panel0p2_srs_64prb_direct_array_shard10`
- `results/` 中已有 172 个 `result*.h5`。
- `manifest/` 中已有 172 个 per-shard manifest。
- 最新 `result_*.h5` 和 `manifest_*.json` 的时间大约停在
  `2026-05-30 00:02 CST`。
- 没有写出 aggregate `manifest/manifest.json`。
- 没有生成 heatmap。
- 没有生成该任务的 summary JSON。
- runner log 没有进入 `simulation_done` 阶段。

该任务计划 UE 数是 1709，按 `shard_size=10` 估算，172 个 shard 文件已经接近应有数量。
因此它不像是正常的全任务推进，而更像是最后某个 shard 或 worker 没有正常退出。

### 证据

进程检查显示原始 runner 进程组 `1831896` 仍存在。多数 worker 已经空闲，但仍有一个
spawn worker 在跑：

```text
PID 2340919, shard_016 worker, 100% CPU, GPU SM 活动很低，没有 result_016*
```

该 worker 打开了以下文件：

```text
logs/perf_events_shard_016.jsonl
logs/hardware_samples_shard_016.csv
```

但没有对应的：

```text
results/result_016*.h5
manifest/manifest_016*.json
```

GPU 检查还显示，另一个用户的 DDP 训练任务占用了 8 张 GPU：

```text
/home/zhengyurui/miniconda3/envs/signal/bin/python3.13 -u main_train_ddp.py
```

当时该训练任务每张卡大约占用 `7.9 GB` 显存，并消耗约 `66%~68%` GPU SM。
卡住的 Sionna worker 在 GPU 上仍有 CUDA context，但几乎没有实际 SM 计算。
`shard_016` 的 hardware sample 中，GPU0 一度接近满显存：

```text
23896 MB / 24564 MB
```

这与日志中的 Dr.Jit memory flush warning 和 CUDA OOM fallback 压力一致。

主机 RAM 和 IO 没有明显瓶颈：

- 可用内存约 `448 GiB`。
- swap 使用约 `503 MiB`。
- CPU load 相对整机规模不高。
- `vmstat` 没有显示明显 I/O wait。

### 判断

并发 DDP 训练几乎肯定增加了运行时波动，也提高了 CUDA OOM 和 fallback 概率，因为每张
GPU 已经被占用了一部分显存和算力。但这次具体卡住不只是“GPU 忙导致变慢”：

- 大多数 shard 输出已经写出。
- 大约 70 分钟没有新 shard 文件出现。
- 卡住 worker 更像 CPU 侧拖尾，GPU 侧几乎没有实际计算。
- 缺失集中在 `shard_016` 附近。

更可能的原因是：在 GPU 资源竞争和显存压力下，某个尾部 shard 进入了 fallback、Dr.Jit
或 CUDA 清理/同步相关的异常拖尾状态，导致父队列一直等待它退出。

### 当时处置

为了不阻塞后续场景：

1. 保留不完整的 `front3d_0002 0p2` 输出目录。
2. 对原始队列进程组 `1831896` 发送 `SIGTERM`。
3. 确认该进程组已退出。
4. 如旧 tmux 会话仍存在，则关闭旧 tmux 会话。
5. 新建 resume 脚本：
   `outputs/local_runs/front3d_remaining_panel_density_queue/run_queue_resume_from_008.sh`
6. 用同名 tmux 会话 `front3d_remaining_density_queue` 从第 8/14 个任务继续。

resume log：

```text
outputs/local_runs/front3d_remaining_panel_density_queue/runner_resume_from_008.log
```

该 log 明确记录跳过任务：

```text
skipped task=7 scene=front3d_0002 density=0p2 reason=stalled shard_016 under GPU contention
```

没有删除数据；不完整的 `front3d_0002 0p2` 目录保留给后续恢复或检查。

## 事故二：`front3d_0002 0p5`

### 现象

从第 8/14 个任务恢复后，同类卡住现象再次出现：

- 任务：`front3d_0002`，density `0p5`。
- 输出目录：
  `outputs/front3d_20_front3d_0002_panel0p5_srs_64prb_direct_array_shard10`
- 在 `2026-05-30 10:46 CST` 检查时，目录中已有 29 个 `result*.h5` 和 29 个
  per-shard manifest。
- 该任务计划 UE 数是 282，按 `shard_size=10` 估算，29 个 shard 基本已经覆盖预期规模。
- 最新 result 时间约为 `2026-05-30 01:29 CST`。
- 没有 aggregate manifest。
- 没有 heatmap。
- 没有 summary JSON。
- worker 进程仍存活了约 9 小时，但几乎不消耗 CPU，也没有新输出。

### 判断

这次复现说明问题不只是 `0p2` 中某个孤立 shard 的偶发慢，而是 queue-level tail stall：
大部分 shard 已完成，但某个 worker、父进程或 multiprocessing cleanup 没有正常结束，导致
整个顺序队列无法进入后处理阶段。

两次卡住都发生在 `front3d_0002` 的 density 任务上，而且都处于 GPU 资源竞争和 Dr.Jit
memory flush warning 背景下。当前更倾向于以下组合原因：

1. GPU 被其他训练任务占用，导致显存余量不足、OOM/fallback 概率增加。
2. Sionna/Dr.Jit 在内存压力下触发 cache flush 或 fallback，尾部 shard 进入异常慢路径。
3. 队列脚本是严格顺序执行，单个任务未退出就不会进入 heatmap、schema validation 和下一个任务。
4. 当前脚本没有 watchdog，不会根据“长时间无新 result/manifest”自动跳过或恢复。

### 当时处置

为了继续推进后续场景：

1. 保留不完整的 `front3d_0002 0p5` 输出目录。
2. 停止对应 runner 进程组。
3. 新建第二个 resume 脚本：
   `outputs/local_runs/front3d_remaining_panel_density_queue/run_queue_resume_from_009.sh`
4. 从第 9/14 个任务 `front3d_0003 0p2` 继续。

这两个 `front3d_0002` density 结果都必须视为不完整，不能作为正式结果使用，除非后续完成
aggregate manifest、schema validation 和 heatmap 生成，或通过干净重跑替代。

## 当前停止状态

用户要求停止所有任务后，已停止当前 Front3D 队列：

- 停止时间检查点：`2026-05-30 11:02 CST`。
- tmux 会话 `front3d_remaining_density_queue` 已不存在。
- 未发现仍在运行的本项目 Front3D 队列/仿真/heatmap/validator 进程。
- 未停止其他用户的 DDP 训练进程。

停止时的任务状态：

| # | 场景 | density | 状态 |
|---:|---|---|---|
| 1 | `front3d_0002` | `0p1` | 完成，summary/heatmap 已生成 |
| 2 | `front3d_0003` | `0p1` | 完成，summary/heatmap 已生成 |
| 3 | `front3d_0004` | `0p1` | 完成，summary/heatmap 已生成 |
| 4 | `front3d_0005` | `0p1` | 完成，summary/heatmap 已生成 |
| 5 | `front3d_0001` | `0p2` | 完成，summary/heatmap 已生成 |
| 6 | `front3d_0001` | `0p5` | 完成，summary/heatmap 已生成 |
| 7 | `front3d_0002` | `0p2` | 不完整，172 result/172 manifest，无 aggregate summary/heatmap |
| 8 | `front3d_0002` | `0p5` | 不完整，29 result/29 manifest，无 aggregate summary/heatmap |
| 9 | `front3d_0003` | `0p2` | 被主动停止，132 result/131 manifest，无 aggregate summary/heatmap |
| 10-14 | 其余任务 | - | 未开始 |

## 应对策略

长队列运行时建议采用以下策略：

1. 如果某个任务 30 到 60 分钟没有产生新的 result/manifest，并且没有进入 aggregate
   manifest、heatmap 或 summary 阶段，就视为疑似 tail stall。
2. 不让单个卡住任务阻塞整个多场景队列。
3. 只停止本项目对应的 runner 进程组，不杀其他用户或其他项目的 GPU 任务。
4. 保留部分输出和日志，不直接删除。
5. 用明确的 resume 脚本或队列状态文件从下一个任务继续。
6. 在 resume log 和最终 inventory 里显式标记被跳过任务。
7. 当前队列结束后，再单独恢复被跳过任务。恢复时优先考虑：
   - 缩小 `shard_size`，必要时对疑似 shard 范围降到 1；
   - 降低 `parallel_workers`；
   - 只选择显存压力较小的 GPU；
   - 等待较安静的 GPU 时间窗口；
   - 或直接干净重跑受影响 density。

对于 `front3d_0002 0p2`，优先检查和恢复 `shard_016` 附近的 UE 范围。
对于 `front3d_0002 0p5`，虽然部分 shard 文件看起来接近完整，也必须通过 aggregate
manifest 重建、schema validation 和 heatmap 生成后才可视为可用；否则建议干净重跑。
对于被主动停止的 `front3d_0003 0p2`，应视为 partial run，后续也需要补齐或重跑。

## 工程改进建议

- 给本地队列脚本增加 watchdog：记录每个任务的最新 result/manifest mtime，超过阈值后自动
  标记为 stalled，并选择跳过或停止等待人工处理。
- 增加正式的 `resume_from_task_index` 参数，避免手工复制 resume 脚本。
- 增加 partial run 恢复工具：从已有 HDF5 和 shard global indices 识别缺失 UE 范围，并生成
  小规模补跑配置。
- 对共享 GPU 场景增加保守运行模板：即使用户指定全部 GPU，也可以选择更小
  `parallel_workers` 或更小 shard。
- 在 debug/perf summary 中记录“最后一次 result/manifest 写入时间”和“当前最慢 shard”，便于
  判断是正常慢还是 tail stall。
