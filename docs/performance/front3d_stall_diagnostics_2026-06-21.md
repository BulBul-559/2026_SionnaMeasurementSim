# Front3D CFR Truth Fallback Stall 复现诊断记录

日期：2026-06-21

分支：`codex/front3d-stall-diagnostics`

主分支落地备注：诊断 runner 仅保留在探索分支；`main` 只保留本记录和 R1 修复本体。

## 目标

针对 3000 场景生产队列中观察到的 fallback inner-worker 高 CPU / GPU 空闲或低利用率卡住问题做定向复现与定位。该分支只用于探索诊断工具和可能的修复路径，后续不直接合并回 `main`；最终只把解决文档或说明文档保留回 `main`。

复现约束：

- 只跑已记录的异常/疑似异常场景，不扩大到全量 3000 队列。
- 只使用固定 GPU `5,6`，不参与 0-4/7 的生产调度。
- 使用 `config/tasks/nr_srs_64prb_cfr_truth_only.yaml` 作为模板，运行时只 patch GPU 和 debug/diagnostic 配置。
- 在没有修改代码或设置的前提下，最多连续重跑 10 次。
- 如果某次复现到问题，或修改了诊断/运行设置，则重跑计数重置。
- 每次重跑前清理上一次复现 attempt 目录中的 HDF5 result 产物，只保留目录结构、日志、marker、summary 和诊断 JSONL。

结束判定：

- 连续 10 次未复现：若之前曾复现并做过修改，可暂判“问题可能已被规避/解决”；若从未复现，则判定“未定位到问题”。
- 若有修改，最终落盘解决文档；若无修改，最终落盘说明文档。

## 场景范围

数据源 index：

`data/front3d_full/indices/small_normal_3000_panel0p5_seed2026.jsonl`

| run index | scene key | class | split | BS | UE | links | 已知现象 |
|---:|---|---|---|---:|---:|---:|---|
| 341 | `front3d_0745` | normal_room | train | 10 | 243 | 2430 | shard 007 卡住 |
| 393 | `front3d_0883` | normal_room | train | 12 | 221 | 2652 | shard 003 卡住 |
| 1908 | `front3d_4285` | normal_room | train | 13 | 322 | 4186 | shard 000 卡住 |
| 2044 | `front3d_4587` | normal_room | train | 13 | 390 | 5070 | shard 002 卡住 |
| 2549 | `front3d_5728` | normal_room | train | 16 | 399 | 6384 | shard 007 卡住 |
| 2567 | `front3d_5773` | normal_room | train | 7 | 226 | 1582 | shard 002/004 卡住 |
| 2904 | `front3d_6586` | normal_room | train | 11 | 319 | 3509 | shard 002 卡住 |
| 2909 | `front3d_6604` | normal_room | train | 15 | 390 | 5850 | 疑似卡住 |
| 2910 | `front3d_6609` | normal_room | test | 11 | 275 | 3025 | 疑似卡住 |
| 2946 | `front3d_6706` | normal_room | train | 12 | 283 | 3396 | 最终失败，`dr.if_stmt()` exception |

历史事件详见：

`docs/performance/front3d_cfr_truth_fallback_stalls_2026-06-20.md`

## 诊断工具

新增脚本：

`scripts/diagnose_front3d_stalls.py`

脚本行为：

- 从上面的 3000 场景 index 中抽取异常场景，生成本次复现专用 `incident_scenes.jsonl`。
- 复制 CFR truth-only 模板到输出目录，并 patch：
  - `output.sharding.gpu_ids: [5, 6]`
  - `output.sharding.parallel_workers: 2`
  - `output.sharding.gpu_scheduler.enabled: true`
  - `output.sharding.gpu_scheduler.cross_scene_pipeline: true`
  - `output.sharding.gpu_scheduler.free_memory_threshold: 0.6`
  - `output.sharding.recycle_workers: true`
  - `output.sharding.fallback.isolation_mode: "on_failure"`
  - `debug.enabled: true`
  - `debug.write_hardware_samples: true`
- 通过 `run-scene-index --pipeline-shards` 启动复现。
- 每 10 秒采集：
  - GPU 5/6 的显存、GPU 利用率、free ratio。
  - `nvidia-smi --query-compute-apps` 中的 compute pid 与显存。
  - `nvidia-smi pmon -c 1` 的 SM/MEM 采样。
  - 相关 Python 进程的 `pid/ppid/stat/etimes/pcpu/pmem/wchan/cmd`。
  - 子进程的 `CUDA_VISIBLE_DEVICES`。
  - 当前 attempt 下最近修改文件和 HDF5 数量。
- 检测条件默认是：外层 compute pid GPU SM 低于 5%，存在 fallback 子进程 CPU 超过 80%，子进程 elapsed 超过 600 秒且非 D state。`CUDA_VISIBLE_DEVICES` 只作为 best-effort 记录；如果能读到且与外层 compute pid 所在 GPU 不一致才过滤，如果读不到则不作为过滤条件，避免漏掉进程启动后 `os.environ` 动态设置 GPU 的 worker。
- 检测到卡住后写 `stall_events.jsonl`，默认 `SIGTERM` inner child，然后终止本次 run process group。

## 安全清理策略

每个复现 run root 写：

- `.front3d_stall_diagnostic_run.json`

每个 attempt root 写：

- `.front3d_stall_diagnostic_attempt.json`

清理前必须同时满足：

- attempt 目录存在 marker。
- marker 的 `created_by` 为 `scripts/diagnose_front3d_stalls.py`。
- marker 中的 `attempt_dir` 与要清理的绝对路径完全一致。
- attempt 目录位于指定的 `--output-root` 下。
- 同一次脚本内自动清理时，marker 中的 `diagnostic_run_root` 必须等于当前 run root。

实际删除范围：

- 仅删除该 attempt 目录下递归匹配的 `*.h5` 文件。
- 不删除目录本身。
- 不删除 driver log、monitor log、hardware samples、scene summary、manifest JSON、marker 或配置。

清理事件写入：

`diagnostics/cleanup_events.jsonl`

如果后续修改设置后重新启动脚本，可显式传入：

```bash
--cleanup-previous-attempt-dir outputs/front3d_stall_diagnostics/<old_run>/attempt_XX
```

这会在新一轮第一次重跑前清理指定旧 attempt 的 HDF5 result 产物；若 marker 校验失败，脚本直接拒绝清理并退出。

## 当前验证记录

已完成：

- `uv run python -m py_compile scripts/diagnose_front3d_stalls.py`
- `uv run ruff check scripts/diagnose_front3d_stalls.py`
- P0 无显存 pressure 对照：
  - run root: `outputs/front3d_stall_diagnostics/repro_20260621T070731Z`
  - attempt 1：10/10 scenes completed，未复现。
  - attempt 2：10/10 scenes completed，未复现。
  - attempt 3：运行到后段时因切换显存压力设置而手动停止；随后由新一轮显式清理该 attempt 的 HDF5 产物。

当前结论：

- clean-GPU 条件下至少两轮完整 10-scene attempt 未复现，说明问题不是“必现的固定 scene/shard bug”。
- 由于生产中异常出现频率较高，后续应重点检查高显存水位、显存碎片化、OOM 后 fallback 路径和 Dr.Jit/Sionna RT 状态残留。

## 显存压力复现设计

2026-06-21 追加假设：卡住可能不强依赖某个具体场景，而是显存水位、显存碎片化或 Dr.Jit/Sionna RT 内部分配状态达到某个阈值后进入异常路径。因此在原始 clean-GPU 复现外，新增可控显存压力模式。

脚本新增参数：

```bash
--memory-pressure-gb-per-gpu 4
--memory-pressure-warmup-s 8
--memory-pressure-chunk-mib 256
```

行为：

- 在每个目标 GPU 上启动一个独立 PyTorch ballast 进程。
- 每个进程只看到自己的 `CUDA_VISIBLE_DEVICES=<physical gpu id>`，在 `cuda:0` 上按 chunk 分配 uint8 tensor，并保持引用不释放。
- pressure 进程的 pid、目标占用和 log 写到：
  - `diagnostics/memory_pressure_processes.json`
  - `diagnostics/memory_pressure_gpu_<id>.log`
  - `diagnostics/memory_pressure_events.jsonl`
- shard 调度仍只使用 `[5,6]`；pressure 进程会真实降低 `nvidia-smi` 的 free memory，因此 shard 会在更高显存水位下运行。
- stall detector 不会把 pressure 进程当成 shard：pressure pid 不在 `sionna_measurement_sim.app.cli` 或 multiprocessing shard worker 进程树内，因此只作为 compute-app 背景占用被记录。

建议分层：

| 档位 | 参数 | 目的 | 预期 |
|---|---|---|---|
| P0 | 无 pressure | 对照组 | 已完成 2 次完整 10-scene attempt，未复现 |
| P1 | `4 GiB/GPU` | 把 19 GiB 级重 shard 推到 23 GiB 左右 | 更容易触发 OOM/fallback/碎片化相关路径，但仍保留调度空间 |
| P2 | `3 GiB/GPU` | 如果 P1 过强导致大量直接 OOM/失败，降低压力 | 保持高水位但减少硬 OOM |
| P3 | `6 GiB/GPU` | 如果 P1 仍完全稳定，进一步强压 | 可能引入较多 OOM，用于逼出 fallback 异常而非吞吐评估 |

注意：

- 当前 `free_memory_threshold=0.6`。4090 约 24 GiB，总预占超过约 9 GiB 后，空闲比例会低于 0.6，调度器可能不再提交 shard；若要测试更高 ballast，需要同步降低 threshold，这属于新的设置变更并重置 10 次计数。
- 显存压力实验会改变运行条件，因此从启用 pressure 起重新计数。
- 切换 pressure 档位也视为设置变更，计数重新开始。

## 显存压力复现结果

命令：

```bash
uv run python scripts/diagnose_front3d_stalls.py \
  --gpu-ids 5,6 \
  --max-runs 10 \
  --output-root outputs/front3d_stall_diagnostics \
  --memory-pressure-gb-per-gpu 4 \
  --cleanup-previous-attempt-dir outputs/front3d_stall_diagnostics/repro_20260621T070731Z/attempt_03
```

run root：

`outputs/front3d_stall_diagnostics/repro_20260621T081241Z`

显存压力：

- GPU 5 ballast pid: `1102791`
- GPU 6 ballast pid: `1102792`
- 每张卡目标占用：4 GiB；`nvidia-smi` 中每个 pressure 进程约占用 4542 MiB。
- pressure 进程结束记录：`diagnostics/memory_pressure_events.jsonl`

清理记录：

- 显式清理上一轮 `attempt_03`：156 个 H5，3,745,634,225 bytes。
- P1 attempt 1 完成后自动清理：159 个 H5，3,799,951,303 bytes。
- P1 attempt 2 复现并完成证据采集后手动清理：140 个 H5，3,353,244,346 bytes；剩余 H5 数为 0。

P1 结果：

| attempt | 结果 | 耗时 | 说明 |
|---:|---|---:|---|
| 1 | completed | 1269.69 s | 10/10 scenes completed，未复现 |
| 2 | reproduced | 1454.07 s | detector 命中后 `SIGTERM` inner child，并终止本次 run process group |

stall event：

```json
{"action":"term-inner","attempt":2,"detected_at":"2026-06-21T08:58:04.987354Z","gpu_id":5,"inner_cuda_visible_devices":"5","inner_elapsed_s":610,"inner_pcpu":99.7,"inner_pid":1522476,"inner_stat":"Rl","inner_wchan":"-","outer_pid":1521400,"outer_sm_pct":null,"outer_stat":"Sl","outer_used_memory_mib":18430,"outer_wchan":"futex_wait_queue"}
```

定位到的 scene/shard：

- scene: `front3d_6586`
- run index: `2904`
- class: `normal_room`
- failing shard: `012`
- fallback shard: `012_00`

证据链：

- `shard_012` 在 `rt_solve` 结束后立即失败，异常为 `jit_malloc(): out of memory! Could not allocate 134217728 bytes of device memory.`
- `perf_summary_shard_012.json` 记录 peak GPU memory 为 24027 MiB，已经贴近 4090 可用显存上限。
- 失败后 fallback 创建 `shard_012_00`，日志只有 `run_start`、`topology_load.start/end`、`rt_solve.start`，没有 `rt_solve.end`。
- detector 命中时，outer pid `1521400` 在 GPU 5 上仍持有 18430 MiB，inner pid `1522476` 已运行 610 s，CPU 约 99.7%，GPU 侧没有有效推进信号。
- 同一 scene 中 `result_000` 到 `result_011`、`result_013` 到 `result_015` 均存在，唯独 `result_012.h5` 和 `manifest_012.json` 缺失，说明不是整个 run 或输出目录写盘完全停住，而是该 OOM 后 fallback 分支单独悬住。
- driver log 出现大量 `jit_flush_malloc_cache(): Dr.Jit exhausted the available memory...` 警告，说明运行期间 Dr.Jit 反复进入显存回收路径。

初步判断：

- 这次复现支持“高显存水位/显存碎片化/Dr.Jit 分配状态触发 fallback 卡住”的假设。
- 不能完全排除 scene 复杂度影响；`front3d_6586` 的该 shard 本身是高显存 shard。但 P0 两轮未复现、P1 第二轮复现，说明触发条件更像是 scene 高显存需求叠加全局显存压力。
- 纯 IO 拥塞不是当前最强解释：卡住进程状态不是 D state，`wchan` 不指向 IO wait；同一时间其它 shard 仍能继续写出 H5 和 manifest；卡住点停在 `rt_solve.start` 后而非 HDF5 write stage。

后续修复方向：

- OOM 后不要在仍持有大量 GPU allocation 的 outer worker 内继续做 fallback；应先彻底回收该 worker，再用新进程执行 split 后的小 shard。
- fallback inner worker 增加独立 watchdog：超过阈值没有 `rt_solve.end` 或输出推进时，杀掉当前 fallback 分支并重新拆分/重试。
- 对接近显存上限的 shard 提前分裂，避免先让大 shard OOM 再 fallback。
- 在调度器中加入更保守的“估算 peak + 当前 free memory”门槛，避免把 19 GiB 级 shard 放到只剩约 20 GiB 的卡上运行。
- 在生产模板中保留 `recycle_workers=true` 的必要性仍然成立；但 OOM/fallback 分支需要更强的进程级隔离，避免外层进程持有 Dr.Jit/CUDA 状态时再派生子分支。

## 修复尝试 R1

提交点：

- 分支：`codex/front3d-stall-diagnostics`
- 修复思路：retryable fallback 不能在父 attempt 的 `except Exception as exc:` 块内部递归运行子 shard。

原因：

- Python exception 的 traceback 会保留失败调用栈上的 frame locals。
- 失败的 `run_rt_truth_pipeline_single` 栈上很可能有 Sionna/Dr.Jit/CUDA 大对象。
- 如果在 `except` 块内部启动 fallback 子 shard，外层 worker 会在子 shard 运行期间继续持有 `exc.__traceback__`，从而保留失败 attempt 的 GPU allocation。这与 P1 复现中 outer pid 持有 18430 MiB、inner pid 高 CPU 卡住的现象一致。

代码改动：

- `_run_shard_spec_attempt()` 在捕获 retryable error 后只记录 `retry_error`、`error`、children 列表和 attempt metadata。
- 对 retryable exception 执行 `_clear_retry_exception_references()`，best-effort 清理 traceback/cause/context frame 引用。
- 离开 `except` 后再执行 `_clear_accelerator_caches()` 和 fallback children。
- `shard_attempts.jsonl` 的 split attempt 增加 `exception_traceback_cleared: true`，便于后续审计。

验证：

- `uv run pytest tests/unit/test_sharding.py -q`
- `uv run ruff check sionna_measurement_sim/rt/truth_pipeline.py tests/unit/test_sharding.py`
- `uv run python -m py_compile sionna_measurement_sim/rt/truth_pipeline.py scripts/diagnose_front3d_stalls.py`

新增单测：

- `test_fallback_children_run_outside_retry_exception_context`：模拟首次 shard OOM，确认 fallback 子 shard 运行时 `sys.exc_info()[0] is None`，也就是已经离开父 attempt 的 active exception context。

回归计划：

- 由于已经修改代码，按规则重置计数。
- 使用同一批异常/疑似异常 scene，只固定 GPU 5/6。
- 使用 P1 pressure：`--memory-pressure-gb-per-gpu 4`。
- 启动最多 10 次 rerun；每次 rerun 前清理上一 attempt 的 H5 result 产物，只保留目录结构与日志。
- 第一轮启动前显式指定上一轮已复现实验目录 `outputs/front3d_stall_diagnostics/repro_20260621T081241Z/attempt_02` 作为 cleanup target，确认不会误删其它目录。

## R1 压力回归结果

命令：

```bash
uv run python scripts/diagnose_front3d_stalls.py \
  --gpu-ids 5,6 \
  --max-runs 10 \
  --output-root outputs/front3d_stall_diagnostics \
  --memory-pressure-gb-per-gpu 4 \
  --cleanup-previous-attempt-dir outputs/front3d_stall_diagnostics/repro_20260621T081241Z/attempt_02
```

run root：

`outputs/front3d_stall_diagnostics/repro_20260621T091332Z`

完成审计：

- `diagnostics/diagnostic_summary.json`：`attempts_run=10`、`max_runs=10`、`stall_found=false`。
- `diagnostics/attempts.jsonl`：10/10 attempt 均 `returncode=0`、10/10 scenes completed、0 failed、`scheduled_count=159`、`stall_detected=false`。
- `diagnostics/stall_events.jsonl`：未生成有效事件。
- `pgrep -af 'diagnose_front3d_stalls|sionna_measurement_sim.app.cli|memory_pressure'`：诊断进程、scene-index 子进程和 pressure 进程均已退出。
- `tmux ls | rg 'front3d_stall|repro'`：复现 tmux session 已退出。

| attempt | 结果 | 耗时 | 场景 | shard | stall |
|---:|---|---:|---:|---:|---|
| 1 | completed | 1193.89 s | 10/10 | 159 | no |
| 2 | completed | 1295.43 s | 10/10 | 159 | no |
| 3 | completed | 1214.37 s | 10/10 | 159 | no |
| 4 | completed | 1214.85 s | 10/10 | 159 | no |
| 5 | completed | 1240.98 s | 10/10 | 159 | no |
| 6 | completed | 1276.57 s | 10/10 | 159 | no |
| 7 | completed | 1342.34 s | 10/10 | 159 | no |
| 8 | completed | 1313.17 s | 10/10 | 159 | no |
| 9 | completed | 1207.11 s | 10/10 | 159 | no |
| 10 | completed | 1065.60 s | 10/10 | 159 | no |

耗时汇总：

- 10 次总耗时：约 206.07 min。
- 单次最短：1065.60 s。
- 单次最长：1342.34 s。
- 单次平均：约 1236.43 s。

清理审计：

- 第一轮启动前显式清理旧复现实验 `repro_20260621T081241Z/attempt_02`，该目录此前已清理，删除 0 个 H5。
- `attempt_01` 到 `attempt_09` 均在下一轮开始前由脚本自动清理。
- `attempt_10` 在完成证据采集后手动清理，清理前校验 `.front3d_stall_diagnostic_attempt.json`：
  - `created_by == "scripts/diagnose_front3d_stalls.py"`
  - `attempt_dir` 等于目标目录绝对路径
  - `attempt == 10`
  - 目标目录位于 `outputs/front3d_stall_diagnostics` 下
- 当前回归 run 合计清理 1590 个 H5，37,995,355,315 bytes。
- 完成后 `attempt_01` 到 `attempt_10` 的 H5 数量均为 0；目录结构、日志、marker、manifest 和诊断 JSONL 保留。

结论：

- R1 修改后，在已复现过 stall 的同一批 10 个异常/疑似异常 Front3D 场景、固定 GPU 5/6、每卡 4 GiB 显存压力下，连续 10 次未再复现 fallback inner-worker 卡住。
- 这满足本次诊断任务的结束条件：曾在 P1 pressure 下复现问题，随后修改 fallback exception handling，并在同一 pressure 设置下完成 10 次连续无复现回归。
- 当前最可信的根因仍是：retryable fallback 在父 attempt 的 active exception context 内递归执行子 shard，导致 traceback frame locals 持有失败 attempt 的 Sionna/Dr.Jit/CUDA 大对象；高显存水位下 fallback 子进程更容易进入异常长耗时路径。
- R1 不是数学证明：它说明该问题在当前复现条件下已经被规避/大幅降低，但未来如果提高 ballast、降低调度阈值、增大 shard 或遇到更大 scene，仍应继续保留 stall detector 和 cleanup marker 机制。

后续建议：

- 保留 `_clear_retry_exception_references()` 和“离开 `except` 后再运行 fallback children”的结构，不要回退到在 `except Exception as exc:` 内递归执行子 shard。
- 生产任务继续使用 `recycle_workers=true` 与 `fallback.isolation_mode="on_failure"`。
- 后续若再次出现同类卡住，优先检查是否存在新的 active exception/context 保留大对象路径，再考虑更强的 fallback child watchdog 或基于 peak 估算的提前拆分。
