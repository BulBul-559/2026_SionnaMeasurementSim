# Front3D Queue Stall Incident, 2026-05-30

> Historical run note. This document records one production queue incident and
> the operational response. It does not redefine current defaults; current
> system truth remains in `docs/agent_handoff.md`, `docs/sys/`, and
> `config/README.md`.

## Scope

- Queue: `outputs/local_runs/front3d_remaining_panel_density_queue/run_queue.sh`
- tmux session before intervention: `front3d_remaining_density_queue`
- Workload: Front3D SRS 64PRB direct-array shard10 queue, 8 GPUs per task.
- Stalled task: task 7/14, `front3d_0002`, density `0p2`.
- Output directory:
  `outputs/front3d_20_front3d_0002_panel0p2_srs_64prb_direct_array_shard10`

## Symptoms

At approximately `2026-05-30 01:12 CST`, task 7 had not advanced for about
70 minutes:

- `results/` contained 172 `result*.h5` files.
- `manifest/` contained 172 per-shard manifests.
- Latest `result_*.h5` and `manifest_*.json` timestamps were around
  `2026-05-30 00:02 CST`.
- No aggregate `manifest/manifest.json`, heatmap output, or summary JSON had
  been written for this task.
- The runner log had not reached `simulation_done`.

The task's planned UE count was 1709, so 172 shard files were close to the
expected shard10 count. The run was therefore likely blocked by one trailing
failed or extremely slow shard rather than normal full-task progress.

## Evidence

Process inspection showed the original runner process group `1831896` still
alive. Most worker processes were idle, while one spawned worker was still
running:

```text
PID 2340919, shard_016 worker, 100% CPU, low GPU SM activity, no result_016*
```

The worker had open file descriptors for:

```text
logs/perf_events_shard_016.jsonl
logs/hardware_samples_shard_016.csv
```

No corresponding `results/result_016*.h5` or `manifest/manifest_016*.json`
files existed.

GPU inspection also showed another user's DDP training job occupying all eight
GPUs:

```text
/home/zhengyurui/miniconda3/envs/signal/bin/python3.13 -u main_train_ddp.py
```

At the time of inspection, that job used about `7.9 GB` per GPU and roughly
`66%~68%` SM per GPU. The stalled Sionna worker had CUDA contexts on the GPUs,
but negligible SM activity. A hardware sample for `shard_016` recorded GPU0
near the memory limit (`23896 MB / 24564 MB`), consistent with the observed
Dr.Jit memory flush warnings and CUDA OOM fallback pressure.

System RAM and host I/O did not appear saturated:

- Memory available: about `448 GiB`.
- Swap usage: about `503 MiB`.
- CPU load was modest relative to machine capacity.
- `vmstat` did not show meaningful I/O wait.

## Interpretation

The concurrent DDP job almost certainly increased runtime variance and raised
the probability of CUDA OOM/fallback because all GPUs were already partially
occupied. However, the specific failure mode was not just "GPU busy":

- Most planned shard outputs had already been written.
- No new shard output had appeared for about 70 minutes.
- The stuck worker was CPU-bound with little GPU activity.
- The missing output was concentrated around `shard_016`.

The most likely diagnosis is a trailing shard stuck after GPU-memory pressure,
fallback, or Dr.Jit/CUDA cleanup behavior under contention.

## Immediate Response

To avoid blocking the remaining queue:

1. Preserved the partial `front3d_0002 0p2` output directory.
2. Sent `SIGTERM` to the original queue process group `1831896`.
3. Verified the process group was gone.
4. Killed the old tmux session if still present.
5. Created a resume script:
   `outputs/local_runs/front3d_remaining_panel_density_queue/run_queue_resume_from_008.sh`
6. Started a new tmux session with the same user-facing name,
   `front3d_remaining_density_queue`, beginning from task 8/14.

The resume log is:

```text
outputs/local_runs/front3d_remaining_panel_density_queue/runner_resume_from_008.log
```

The skipped task is explicitly recorded in that log:

```text
skipped task=7 scene=front3d_0002 density=0p2 reason=stalled shard_016 under GPU contention
```

No data was deleted. The incomplete `front3d_0002 0p2` directory remains for
later recovery or inspection.

## Recurrence After Resume

The same queue-stall pattern recurred after resuming from task 8/14:

- Task: `front3d_0002`, density `0p5`.
- Output directory:
  `outputs/front3d_20_front3d_0002_panel0p5_srs_64prb_direct_array_shard10`
- At `2026-05-30 10:46 CST`, the directory contained 29 `result*.h5` files
  and 29 per-shard manifests, matching the expected shard10 scale for 282 UEs.
- Latest result timestamps were around `2026-05-30 01:29 CST`.
- No aggregate manifest, heatmap, or summary had been written.
- The task's worker processes were still alive after about 9 hours, but nearly
  idle, with no new output during the inspection window.

This strengthens the interpretation that the failure mode is a queue-level
tail stall after most shard outputs have completed, not normal slow GPU
progress. The immediate operational response was repeated:

1. Preserve the incomplete `front3d_0002 0p5` output.
2. Stop the affected runner process group.
3. Start a second resume script from task 9/14:
   `outputs/local_runs/front3d_remaining_panel_density_queue/run_queue_resume_from_009.sh`
4. Continue with `front3d_0003 0p2`.

Both skipped `front3d_0002` density runs (`0p2` and `0p5`) should be treated
as incomplete until explicitly recovered.

## Recommended Response Policy

For long multi-scene queues:

1. Treat a task as stalled when it has produced no new result/manifest files for
   30 to 60 minutes while one worker remains CPU-bound and no aggregate manifest
   is being written.
2. Do not let one stalled shard block the entire scene queue.
3. Stop only the affected runner process group; do not kill unrelated GPU jobs.
4. Preserve partial outputs and logs.
5. Resume the queue from the next task in a separate resume script or explicit
   queue state file.
6. Mark the skipped task clearly in the resume log and final run inventory.
7. Recover the skipped scene after the queue finishes, preferably with:
   - smaller shard size, down to shard size 1 for the suspected shard range;
   - fewer parallel workers;
   - explicit GPU selection on less-contended GPUs;
   - or a quiet GPU window if available.

For `front3d_0002 0p2`, first recovery target should be the missing
`shard_016` range. For `front3d_0002 0p5`, the partial shard outputs appear
near-complete but should still be recovered through aggregate manifest
reconstruction or a clean rerun. If recovery tooling cannot cleanly patch the
aggregate manifest from partial outputs, rerun the affected `front3d_0002`
density tasks with a more conservative configuration after the current queue
completes.

## Engineering Follow-ups

- Add a queue-level watchdog that records last result/manifest mtime per task
  and can automatically skip to the next task after a configurable stale timeout.
- Add a first-class "resume from task index" option to local queue scripts
  instead of copying ad hoc resume scripts.
- Add a recovery helper that can identify missing global UE indices from partial
  shard outputs and generate a small targeted rerun config.
- Consider using lower default `parallel_workers` when all GPUs are shared with
  another heavy training job, even if all GPU IDs are requested intentionally.
