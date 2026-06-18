"""Scene index creation and ordered scene-index runner."""

from __future__ import annotations

import copy
import json
import random
import re
import subprocess
import sys
import time
import traceback
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from sionna_measurement_sim.config.loader import load_config
from sionna_measurement_sim.config.mappers import to_domain_ranging_config
from sionna_measurement_sim.config.schema import MeasurementConfig
from sionna_measurement_sim.domain.array import ArraySpectrumConfig
from sionna_measurement_sim.domain.link import LinkConfig as DomainLinkConfig
from sionna_measurement_sim.phy.impairments import ImpairmentConfig
from sionna_measurement_sim.rt import truth_pipeline
from sionna_measurement_sim.rt.truth_pipeline import RTTruthRunConfig
from sionna_measurement_sim.visualization.config import VisualizationRunConfig

_OUTPUT_RUN_CONFIG_NAME = "run_config.yaml"


@dataclass(frozen=True)
class SceneIndexBuildResult:
    """Paths and counts produced by scene index creation."""

    index_path: Path
    summary_path: Path
    total_count: int
    selected_counts: dict[str, int]


@dataclass(frozen=True)
class SceneIndexRunResult:
    """Paths and counts produced by ordered scene-index execution."""

    manifest_path: Path
    summary_path: Path
    total: int
    completed: int
    failed: int
    skipped: int
    planned: int


@dataclass
class _SceneRunPlan:
    entry: dict[str, Any]
    base_entry: dict[str, Any]
    scene_config: dict[str, Any]
    generated_config_path: Path
    output_dir: Path
    run_log_path: Path
    config: RTTruthRunConfig | None = None
    shard_specs: list[Any] | None = None
    shard_results: list[dict[str, object]] | None = None
    shard_attempts: list[dict[str, object]] | None = None
    planned_shard_count: int = 0
    scheduled_shard_count: int = 0
    completed_shard_count: int = 0
    active_shard_count: int = 0
    failed: bool = False
    finalized: bool = False
    error: str = ""
    traceback_text: str = ""
    started_perf: float | None = None
    queued_perf: float = 0.0


@dataclass(frozen=True)
class _PipelineShardTask:
    scene: _SceneRunPlan
    spec: Any


def build_scene_index(
    *,
    source_root: Path,
    output_path: Path,
    classes: tuple[str, ...] = ("small_room", "normal_room"),
    total_count: int = 3000,
    label_name: str = "label_panel_0p5.json",
    seed: int = 2026,
    order: str = "interleaved",
) -> SceneIndexBuildResult:
    """Build a deterministic stratified scene index for production simulation."""

    if total_count <= 0:
        raise ValueError("total_count must be positive")
    if not classes:
        raise ValueError("classes must not be empty")
    source_root = Path(source_root)
    output_path = Path(output_path)
    records_by_class = _load_scene_records_by_class(source_root, classes, label_name)
    available_counts = {name: len(records) for name, records in records_by_class.items()}
    if any(count == 0 for count in available_counts.values()):
        empty = [name for name, count in available_counts.items() if count == 0]
        raise ValueError(f"No scenes available for classes: {', '.join(empty)}")
    if total_count > sum(available_counts.values()):
        raise ValueError(
            f"Requested {total_count} scenes but only {sum(available_counts.values())} "
            "are available"
        )

    selected_counts = _allocate_counts(total_count, available_counts)
    rng = random.Random(seed)
    selected_by_class: dict[str, list[dict[str, Any]]] = {}
    for class_name in classes:
        records = list(records_by_class[class_name])
        count = selected_counts[class_name]
        sampled = rng.sample(records, count)
        selected_by_class[class_name] = sorted(
            sampled,
            key=lambda item: (int(item.get("source_scene_index", 0)), item["scene_key"]),
        )

    ordered = _order_selected_records(selected_by_class, selected_counts, order)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for index, item in enumerate(ordered):
            entry = dict(item)
            entry["index"] = index
            entry["index_id"] = f"{index:06d}"
            handle.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")

    summary = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "source_root": source_root.as_posix(),
        "index_path": output_path.as_posix(),
        "classes": list(classes),
        "available_counts": available_counts,
        "requested_total_count": total_count,
        "selected_counts": selected_counts,
        "label_name": label_name,
        "seed": seed,
        "order": order,
        "total_count": len(ordered),
        "estimated_links": sum(int(item.get("link_count", 0)) for item in ordered),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return SceneIndexBuildResult(
        index_path=output_path,
        summary_path=summary_path,
        total_count=len(ordered),
        selected_counts=selected_counts,
    )


def run_scene_index(
    *,
    index_path: Path,
    config_path: Path,
    output_root: Path,
    start_index: int = 0,
    limit: int | None = None,
    dry_run: bool = False,
    skip_existing: bool = True,
    stop_on_failure: bool = False,
    pipeline_shards: bool = False,
    python_executable: str | None = None,
) -> SceneIndexRunResult:
    """Run scene simulations in the exact order recorded by a scene index."""

    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided")
    entries = [
        entry
        for entry in read_scene_index(index_path)
        if int(entry.get("index", 0)) >= start_index
    ]
    entries.sort(key=lambda item: int(item.get("index", 0)))
    if limit is not None:
        entries = entries[:limit]

    output_root = Path(output_root)
    configs_dir = output_root / "configs"
    logs_dir = output_root / "logs"
    output_root.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    template = _load_yaml_mapping(config_path)
    pipeline_shards = pipeline_shards or _template_cross_scene_pipeline_enabled(template)

    manifest_path = output_root / "scene_index_run_manifest.jsonl"
    summary_path = output_root / "scene_index_run_summary.json"
    if pipeline_shards and not dry_run:
        return _run_scene_index_pipelined_shards(
            entries=entries,
            template=template,
            index_path=Path(index_path),
            config_path=Path(config_path),
            output_root=output_root,
            configs_dir=configs_dir,
            logs_dir=logs_dir,
            manifest_path=manifest_path,
            summary_path=summary_path,
            start_index=start_index,
            limit=limit,
            skip_existing=skip_existing,
            stop_on_failure=stop_on_failure,
        )

    counts = {"completed": 0, "failed": 0, "skipped": 0, "planned": 0}
    executable = python_executable or sys.executable
    with manifest_path.open("a", encoding="utf-8") as manifest:
        for entry in entries:
            run_entry = _run_one_scene_index_entry(
                entry=entry,
                template=template,
                config_path=Path(config_path),
                configs_dir=configs_dir,
                output_root=output_root,
                logs_dir=logs_dir,
                dry_run=dry_run,
                skip_existing=skip_existing,
                python_executable=executable,
            )
            status = str(run_entry["status"])
            counts[status] = counts.get(status, 0) + 1
            manifest.write(json.dumps(run_entry, sort_keys=True, ensure_ascii=False) + "\n")
            manifest.flush()
            if status == "failed" and stop_on_failure:
                break

    summary = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "index_path": Path(index_path).as_posix(),
        "config_path": Path(config_path).as_posix(),
        "output_root": output_root.as_posix(),
        "start_index": start_index,
        "limit": limit,
        "dry_run": dry_run,
        "skip_existing": skip_existing,
        "stop_on_failure": stop_on_failure,
        "pipeline_shards": pipeline_shards,
        "total": sum(counts.values()),
        **counts,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return SceneIndexRunResult(
        manifest_path=manifest_path,
        summary_path=summary_path,
        total=sum(counts.values()),
        completed=counts.get("completed", 0),
        failed=counts.get("failed", 0),
        skipped=counts.get("skipped", 0),
        planned=counts.get("planned", 0),
    )


def read_scene_index(index_path: Path) -> list[dict[str, Any]]:
    """Read a JSONL scene index."""

    entries: list[dict[str, Any]] = []
    with Path(index_path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {index_path}:{line_no}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Scene index entry must be an object at {index_path}:{line_no}")
            entries.append(item)
    return entries


def _run_one_scene_index_entry(
    *,
    entry: dict[str, Any],
    template: dict[str, Any],
    config_path: Path,
    configs_dir: Path,
    output_root: Path,
    logs_dir: Path,
    dry_run: bool,
    skip_existing: bool,
    python_executable: str,
) -> dict[str, Any]:
    plan = _prepare_scene_run_plan(
        entry=entry,
        template=template,
        config_path=config_path,
        configs_dir=configs_dir,
        output_root=output_root,
        logs_dir=logs_dir,
    )
    run_manifest = plan.output_dir / "manifest" / "manifest.json"
    if skip_existing and run_manifest.exists():
        return {
            **plan.base_entry,
            "status": "skipped",
            "finished_at": _utc_now(),
            "returncode": 0,
        }

    _write_scene_config(plan.scene_config, plan.generated_config_path)
    command = [
        python_executable,
        "-m",
        "sionna_measurement_sim.app.cli",
        "--config",
        plan.generated_config_path.as_posix(),
        "run-full",
        "--output-dir",
        plan.output_dir.as_posix(),
    ]
    if dry_run:
        return {
            **plan.base_entry,
            "status": "planned",
            "finished_at": _utc_now(),
            "returncode": None,
            "command": command,
        }

    start = time.perf_counter()
    plan.run_log_path.parent.mkdir(parents=True, exist_ok=True)
    with plan.run_log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    duration_s = time.perf_counter() - start
    status = "completed" if process.returncode == 0 else "failed"
    return {
        **plan.base_entry,
        "status": status,
        "finished_at": _utc_now(),
        "returncode": int(process.returncode),
        "duration_s": duration_s,
        "command": command,
    }


def _prepare_scene_run_plan(
    *,
    entry: dict[str, Any],
    template: dict[str, Any],
    config_path: Path,
    configs_dir: Path,
    output_root: Path,
    logs_dir: Path,
) -> _SceneRunPlan:
    scene_key = str(entry["scene_key"])
    index = int(entry["index"])
    label_tag = str(entry.get("label_tag") or _label_tag(str(entry.get("label_name", ""))))
    run_name = _safe_name(f"{index:06d}_{scene_key}_{label_tag}_cfr_truth_srs64prb")
    scene_output_dir = output_root / "runs" / run_name
    scene_config_path = configs_dir / f"{run_name}.yaml"
    run_log_path = logs_dir / f"{run_name}.log"
    base_entry = {
        "index": index,
        "scene_key": scene_key,
        "scene_class": entry.get("scene_class"),
        "split": entry.get("split", ""),
        "label_file": entry.get("label_file"),
        "scene_file": entry.get("scene_file"),
        "config_template": config_path.as_posix(),
        "generated_config": scene_config_path.as_posix(),
        "output_dir": scene_output_dir.as_posix(),
        "run_log": run_log_path.as_posix(),
        "started_at": _utc_now(),
    }
    scene_config = _scene_config(template, entry, scene_output_dir)
    return _SceneRunPlan(
        entry=entry,
        base_entry=base_entry,
        scene_config=scene_config,
        generated_config_path=scene_config_path,
        output_dir=scene_output_dir,
        run_log_path=run_log_path,
    )


def _write_scene_config(scene_config: dict[str, Any], path: Path) -> None:
    path.write_text(
        yaml.safe_dump(scene_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _run_scene_index_pipelined_shards(
    *,
    entries: list[dict[str, Any]],
    template: dict[str, Any],
    index_path: Path,
    config_path: Path,
    output_root: Path,
    configs_dir: Path,
    logs_dir: Path,
    manifest_path: Path,
    summary_path: Path,
    start_index: int,
    limit: int | None,
    skip_existing: bool,
    stop_on_failure: bool,
) -> SceneIndexRunResult:
    counts = {"completed": 0, "failed": 0, "skipped": 0, "planned": 0}
    pending: deque[_PipelineShardTask] = deque()
    scenes: list[_SceneRunPlan] = []
    global_start = time.perf_counter()
    scheduler_summary: dict[str, object] = {}

    with manifest_path.open("a", encoding="utf-8") as manifest:
        for entry in entries:
            plan = _prepare_scene_run_plan(
                entry=entry,
                template=template,
                config_path=config_path,
                configs_dir=configs_dir,
                output_root=output_root,
                logs_dir=logs_dir,
            )
            run_manifest = plan.output_dir / "manifest" / "manifest.json"
            if skip_existing and run_manifest.exists():
                _write_scene_index_manifest_entry(
                    manifest,
                    {
                        **plan.base_entry,
                        "status": "skipped",
                        "finished_at": _utc_now(),
                        "returncode": 0,
                        "pipeline_shards": True,
                    },
                    counts,
                )
                continue

            _write_scene_config(plan.scene_config, plan.generated_config_path)
            measurement_config = load_config(plan.generated_config_path)
            plan.config = _rt_run_config_from_measurement_config(measurement_config)
            plan.config = truth_pipeline._normalize_output_profile_config(plan.config)
            _validate_pipeline_shard_config(plan.config)
            _write_output_run_config(measurement_config, plan.output_dir)
            plan.shard_specs = truth_pipeline._build_shard_specs(plan.config)
            plan.shard_results = []
            plan.shard_attempts = []
            plan.planned_shard_count = len(plan.shard_specs)
            plan.queued_perf = time.perf_counter()
            scenes.append(plan)
            for spec in plan.shard_specs:
                pending.append(_PipelineShardTask(scene=plan, spec=spec))

        if pending:
            scheduler_summary = _run_pipeline_shard_scheduler(
                pending,
                stop_on_failure=stop_on_failure,
                manifest=manifest,
                counts=counts,
            )

        for plan in scenes:
            if not plan.finalized:
                _finalize_pipeline_scene(plan, manifest, counts, scheduler_summary)

    summary = {
        "schema_version": 1,
        "created_at": _utc_now(),
        "index_path": index_path.as_posix(),
        "config_path": config_path.as_posix(),
        "output_root": output_root.as_posix(),
        "start_index": start_index,
        "limit": limit,
        "dry_run": False,
        "skip_existing": skip_existing,
        "stop_on_failure": stop_on_failure,
        "pipeline_shards": True,
        "total": sum(counts.values()),
        "elapsed_seconds": time.perf_counter() - global_start,
        "scheduler": scheduler_summary,
        **counts,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    return SceneIndexRunResult(
        manifest_path=manifest_path,
        summary_path=summary_path,
        total=sum(counts.values()),
        completed=counts.get("completed", 0),
        failed=counts.get("failed", 0),
        skipped=counts.get("skipped", 0),
        planned=counts.get("planned", 0),
    )


def _run_pipeline_shard_scheduler(
    pending: deque[_PipelineShardTask],
    *,
    stop_on_failure: bool,
    manifest: Any,
    counts: dict[str, int],
) -> dict[str, object]:
    first_config = pending[0].scene.config
    if first_config is None:
        raise ValueError("pipeline shard scheduler requires prepared scene configs")
    sharding = first_config.output_sharding_config
    scheduler = getattr(sharding, "gpu_scheduler", None)
    threshold = float(getattr(scheduler, "free_memory_threshold", 0.6))
    scan_interval_s = float(getattr(scheduler, "scan_interval_s", 1.0))
    configured_gpu_ids = [int(gpu_id) for gpu_id in getattr(sharding, "gpu_ids", [])]
    candidate_gpu_ids = (
        configured_gpu_ids
        if configured_gpu_ids
        else truth_pipeline._discover_cuda_gpu_ids()
    )
    if not candidate_gpu_ids:
        raise RuntimeError(
            "cross-scene shard pipeline requires configured or discoverable GPUs"
        )
    parallel_workers = int(getattr(sharding, "parallel_workers", 1))
    max_active_workers = min(max(parallel_workers, 1), len(candidate_gpu_ids), len(pending))
    active: dict[Any, tuple[_PipelineShardTask, int]] = {}
    scheduled_count = 0
    wait_count = 0
    scan_count = 0
    failed_scene_count = 0

    with ProcessPoolExecutor(max_workers=max_active_workers) as executor:
        while pending or active:
            failed_scene_count += _collect_completed_pipeline_shards(
                active,
                pending,
                manifest,
                counts,
                stop_on_failure=stop_on_failure,
                scheduler_summary={
                    "enabled": True,
                    "cross_scene_pipeline": True,
                    "candidate_gpu_ids": candidate_gpu_ids,
                    "free_memory_threshold": threshold,
                    "scan_interval_s": scan_interval_s,
                    "max_active_workers": max_active_workers,
                    "scheduled_count": scheduled_count,
                    "wait_count": wait_count,
                    "scan_count": scan_count,
                },
            )
            scheduled_this_scan = False
            while pending and len(active) < max_active_workers:
                scan_count += 1
                free_ratios = truth_pipeline._query_gpu_free_memory_ratios(
                    candidate_gpu_ids
                )
                busy_gpu_ids = {gpu_id for _, gpu_id in active.values()}
                available_gpu_ids = [
                    gpu_id
                    for gpu_id in candidate_gpu_ids
                    if gpu_id not in busy_gpu_ids
                    and free_ratios.get(gpu_id, 0.0) >= threshold
                ]
                if not available_gpu_ids:
                    break
                gpu_id = min(
                    available_gpu_ids,
                    key=lambda item: (-free_ratios.get(item, 0.0), item),
                )
                task = pending.popleft()
                scene = task.scene
                if scene.failed:
                    continue
                if scene.started_perf is None:
                    scene.started_perf = time.perf_counter()
                    scene.base_entry["started_at"] = _utc_now()
                scene.scheduled_shard_count += 1
                scene.active_shard_count += 1
                if scene.config is None:
                    raise ValueError("pipeline shard task is missing scene config")
                future = executor.submit(
                    truth_pipeline._run_shard_spec_with_fallback,
                    scene.config,
                    task.spec,
                    gpu_id,
                )
                active[future] = (task, gpu_id)
                scheduled_count += 1
                scheduled_this_scan = True

            if active:
                done, _ = wait(
                    list(active),
                    timeout=scan_interval_s,
                    return_when=FIRST_COMPLETED,
                )
                if done:
                    failed_scene_count += _collect_completed_pipeline_shards(
                        active,
                        pending,
                        manifest,
                        counts,
                        completed=done,
                        stop_on_failure=stop_on_failure,
                        scheduler_summary={
                            "enabled": True,
                            "cross_scene_pipeline": True,
                            "candidate_gpu_ids": candidate_gpu_ids,
                            "free_memory_threshold": threshold,
                            "scan_interval_s": scan_interval_s,
                            "max_active_workers": max_active_workers,
                            "scheduled_count": scheduled_count,
                            "wait_count": wait_count,
                            "scan_count": scan_count,
                        },
                    )
                elif pending:
                    wait_count += 1
                continue

            if pending and not scheduled_this_scan:
                wait_count += 1
                time.sleep(scan_interval_s)

    return {
        "enabled": True,
        "cross_scene_pipeline": True,
        "candidate_gpu_ids": candidate_gpu_ids,
        "free_memory_threshold": threshold,
        "scan_interval_s": scan_interval_s,
        "max_active_workers": max_active_workers,
        "scheduled_count": scheduled_count,
        "wait_count": wait_count,
        "scan_count": scan_count,
        "failed_scene_count": failed_scene_count,
    }


def _collect_completed_pipeline_shards(
    active: dict[Any, tuple[_PipelineShardTask, int]],
    pending: deque[_PipelineShardTask],
    manifest: Any,
    counts: dict[str, int],
    *,
    completed: set[Any] | None = None,
    stop_on_failure: bool,
    scheduler_summary: dict[str, object],
) -> int:
    futures = completed if completed is not None else {
        future for future in active if future.done()
    }
    failed_scene_count = 0
    for future in list(futures):
        if future not in active:
            continue
        task, _gpu_id = active.pop(future)
        scene = task.scene
        scene.active_shard_count = max(scene.active_shard_count - 1, 0)
        try:
            outcome = future.result()
        except Exception as exc:  # noqa: BLE001 - record and continue other scenes
            if not scene.failed:
                failed_scene_count += 1
            scene.failed = True
            scene.error = str(exc)
            scene.traceback_text = traceback.format_exc()
            _remove_pending_scene_tasks(pending, scene)
            if stop_on_failure:
                pending.clear()
        else:
            if scene.shard_results is None or scene.shard_attempts is None:
                raise ValueError("pipeline scene is missing shard result buffers")
            scene.shard_results.extend(outcome["results"])
            scene.shard_attempts.extend(outcome["attempts"])
            scene.completed_shard_count += 1
        if _pipeline_scene_ready_to_finalize(scene):
            _finalize_pipeline_scene(scene, manifest, counts, scheduler_summary)
    return failed_scene_count


def _pipeline_scene_ready_to_finalize(scene: _SceneRunPlan) -> bool:
    if scene.finalized:
        return False
    if scene.failed:
        return scene.active_shard_count == 0
    return scene.completed_shard_count >= scene.planned_shard_count


def _remove_pending_scene_tasks(
    pending: deque[_PipelineShardTask],
    scene: _SceneRunPlan,
) -> None:
    kept = [task for task in pending if task.scene is not scene]
    pending.clear()
    pending.extend(kept)


def _finalize_pipeline_scene(
    scene: _SceneRunPlan,
    manifest: Any,
    counts: dict[str, int],
    scheduler_summary: dict[str, object],
) -> None:
    if scene.finalized:
        return
    started = scene.started_perf if scene.started_perf is not None else scene.queued_perf
    duration_s = time.perf_counter() - started if started else 0.0
    if scene.failed:
        _write_pipeline_scene_log(scene, status="failed")
        _write_scene_index_manifest_entry(
            manifest,
            {
                **scene.base_entry,
                "status": "failed",
                "finished_at": _utc_now(),
                "returncode": 1,
                "duration_s": duration_s,
                "pipeline_shards": True,
                "planned_shard_count": scene.planned_shard_count,
                "scheduled_shard_count": scene.scheduled_shard_count,
                "completed_shard_count": scene.completed_shard_count,
                "error": scene.error,
            },
            counts,
        )
        scene.finalized = True
        return

    if scene.config is None or scene.shard_specs is None:
        raise ValueError("pipeline scene cannot finalize without config and shard specs")
    shard_results = scene.shard_results or []
    shard_attempts = scene.shard_attempts or []
    scene_runtime = {
        **scheduler_summary,
        "scheduled_count": scene.scheduled_shard_count,
        "global_scheduled_count": scheduler_summary.get("scheduled_count", 0),
        "global_wait_count": scheduler_summary.get("wait_count", 0),
        "global_scan_count": scheduler_summary.get("scan_count", 0),
    }
    truth_pipeline.finalize_sharded_rt_truth_run(
        scene.config,
        scene.shard_specs,
        shard_results,
        shard_attempts,
        start_time=started or time.perf_counter(),
        scheduler_runtime=scene_runtime,
        phase="sharded_run_full_scene_index_pipeline",
    )
    _write_pipeline_scene_log(scene, status="completed")
    _write_scene_index_manifest_entry(
        manifest,
        {
            **scene.base_entry,
            "status": "completed",
            "finished_at": _utc_now(),
            "returncode": 0,
            "duration_s": duration_s,
            "pipeline_shards": True,
            "planned_shard_count": scene.planned_shard_count,
            "scheduled_shard_count": scene.scheduled_shard_count,
            "completed_shard_count": scene.completed_shard_count,
        },
        counts,
    )
    scene.finalized = True


def _write_scene_index_manifest_entry(
    manifest: Any,
    entry: dict[str, Any],
    counts: dict[str, int],
) -> None:
    status = str(entry["status"])
    counts[status] = counts.get(status, 0) + 1
    manifest.write(json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n")
    manifest.flush()


def _write_pipeline_scene_log(scene: _SceneRunPlan, *, status: str) -> None:
    scene.run_log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"scene_key: {scene.entry.get('scene_key')}",
        f"status: {status}",
        f"planned_shard_count: {scene.planned_shard_count}",
        f"scheduled_shard_count: {scene.scheduled_shard_count}",
        f"completed_shard_count: {scene.completed_shard_count}",
    ]
    if scene.error:
        lines.append(f"error: {scene.error}")
    if scene.traceback_text:
        lines.extend(["traceback:", scene.traceback_text])
    scene.run_log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_pipeline_shard_config(config: RTTruthRunConfig) -> None:
    sharding = config.output_sharding_config
    if sharding is None or not bool(getattr(sharding, "enabled", False)):
        raise ValueError("cross-scene shard pipeline requires output.sharding.enabled")
    if bool(getattr(getattr(sharding, "bundle", None), "enabled", False)):
        raise ValueError(
            "cross-scene shard pipeline only supports default shard HDF5 mode"
        )
    if not truth_pipeline._dynamic_gpu_scheduler_enabled(config, sharding):
        raise ValueError(
            "cross-scene shard pipeline requires CUDA device and "
            "output.sharding.gpu_scheduler.enabled=true"
        )


def _write_output_run_config(config: MeasurementConfig, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / _OUTPUT_RUN_CONFIG_NAME
    path.write_text(
        yaml.safe_dump(
            config.model_dump(mode="json"),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


def _rt_run_config_from_measurement_config(cfg: MeasurementConfig) -> RTTruthRunConfig:
    if cfg.output.profile == "rt_labels_only":
        cfg.phy.enabled = False
        cfg.ranging.enabled = False
        cfg.array.spectrum.enabled = False
        cfg.visualization.enabled = False
        cfg.calibration.enabled = False
        cfg.phy.iq.enabled = False
        cfg.noncooperative.enabled = False
        cfg.output.save_full_paths = False

    phy_enabled = cfg.phy.enabled
    motion_enabled = cfg.motion.enabled
    imp = cfg.impairments
    impairment = ImpairmentConfig(
        random_seed=imp.impairment_seed,
        cfo_hz=imp.cfo.cfo_hz if imp.cfo.enabled else None,
        sfo_ppm=imp.sfo.sfo_ppm if imp.sfo.enabled else None,
        phase_offset_rad=(
            imp.phase_noise.phase_offset_rad if imp.phase_noise.enabled else None
        ),
        timing_offset_samples=(
            imp.timing_offset.timing_offset_samples
            if imp.timing_offset.enabled
            else None
        ),
        agc_gain_db=imp.agc_adc.agc_gain_db if imp.agc_adc.enabled else 0.0,
        clipping_threshold=(
            imp.agc_adc.clipping_threshold if imp.agc_adc.enabled else None
        ),
    )
    return RTTruthRunConfig(
        label_file=Path(cfg.input.label_file),
        scene_file=Path(cfg.input.scene_file),
        output_dir=Path(cfg.output.root_dir),
        scene_id=cfg.input.scene_id,
        map_id=cfg.input.map_id,
        center_frequency_hz=cfg.carrier.center_frequency_hz,
        bandwidth_hz=cfg.carrier.bandwidth_hz,
        num_subcarriers=cfg.carrier.num_subcarriers,
        seed=cfg.runtime.seed,
        device=cfg.runtime.device,
        max_depth=cfg.rt.max_depth,
        los=cfg.rt.los,
        specular_reflection=cfg.rt.specular_reflection,
        diffuse_reflection=cfg.rt.diffuse_reflection,
        refraction=cfg.rt.refraction,
        diffraction=cfg.rt.diffraction,
        synthetic_array=cfg.rt.synthetic_array,
        merge_shapes=cfg.rt.merge_shapes,
        normalize_cfr=cfg.rt.normalize_cfr,
        normalize_delays=cfg.rt.normalize_delays,
        observation_snr_db=cfg.phy.snr_db if phy_enabled else None,
        impairment_config=impairment,
        phy_standard=cfg.phy.standard,
        subcarrier_spacing_khz=cfg.phy.subcarrier_spacing_khz,
        num_prb=cfg.phy.num_prb,
        num_layers=cfg.phy.num_layers,
        num_antenna_ports=cfg.phy.num_antenna_ports,
        mcs_index=cfg.phy.mcs_index,
        mcs_table=cfg.phy.mcs_table,
        perfect_csi=cfg.phy.perfect_csi,
        ebno_db=cfg.phy.ebno_db,
        pusch_dmrs_config_type=cfg.phy.pusch_dmrs_config_type,
        pusch_dmrs_length=cfg.phy.pusch_dmrs_length,
        pusch_dmrs_additional_position=cfg.phy.pusch_dmrs_additional_position,
        pusch_num_cdm_groups_without_data=cfg.phy.pusch_num_cdm_groups_without_data,
        tx_power_dbm=cfg.phy.tx_power_dbm,
        power_config=cfg.phy.power,
        iq_config=cfg.phy.iq,
        su_mimo_link_batch_size=cfg.phy.su_mimo_link_batch_size,
        num_ofdm_symbols=cfg.phy.num_ofdm_symbols,
        cp_length=cfg.phy.cp_length,
        num_time_steps=cfg.motion.num_time_steps if motion_enabled else 1,
        sampling_frequency_hz=(
            cfg.motion.sampling_frequency_hz if motion_enabled else 0.0
        ),
        max_bs=cfg.input.max_bs,
        max_ue=cfg.input.max_ue,
        bs_num_rows=cfg.antenna.bs_array.num_rows,
        bs_num_cols=cfg.antenna.bs_array.num_cols,
        ue_num_rows=cfg.antenna.ue_array.num_rows,
        ue_num_cols=cfg.antenna.ue_array.num_cols,
        bs_polarization=cfg.antenna.bs_array.polarization,
        ue_polarization=cfg.antenna.ue_array.polarization,
        bs_pattern=cfg.antenna.bs_array.pattern,
        ue_pattern=cfg.antenna.ue_array.pattern,
        bs_orientation_mode=cfg.antenna.bs_array.orientation_mode,
        bs_orientation_rad=tuple(cfg.antenna.bs_array.orientation_rad),
        ue_orientation_mode=cfg.antenna.ue_array.orientation_mode,
        ue_orientation_rad=tuple(cfg.antenna.ue_array.orientation_rad),
        bs_spacing_lambda=(
            cfg.antenna.bs_array.vertical_spacing_lambda,
            cfg.antenna.bs_array.horizontal_spacing_lambda,
        ),
        ue_spacing_lambda=(
            cfg.antenna.ue_array.vertical_spacing_lambda,
            cfg.antenna.ue_array.horizontal_spacing_lambda,
        ),
        hdf5_filename=cfg.output.hdf5_filename,
        hdf5_compression=cfg.output.compression,
        hdf5_gzip_level=cfg.output.gzip_level,
        output_profile=cfg.output.profile,
        output_products=tuple(cfg.output.products) if cfg.output.products else None,
        save_full_paths=cfg.output.save_full_paths,
        calibration_enabled=cfg.calibration.enabled,
        link_config=DomainLinkConfig(
            duplex_mode=cfg.link.duplex_mode,
            phy_link_direction=cfg.link.phy_link_direction,
        ),
        debug_config=cfg.debug,
        output_sharding_config=cfg.output.sharding,
        visualization_config=VisualizationRunConfig(
            enabled=cfg.visualization.enabled,
            output_dir=cfg.visualization.output_dir,
            sample_policy=cfg.visualization.sample_policy,
            random_seed=cfg.visualization.random_seed,
            max_bs=cfg.visualization.max_bs,
            sample_ue_count=cfg.visualization.sample_ue_count,
            max_ue=cfg.visualization.max_ue,
            dpi=cfg.visualization.dpi,
            format=cfg.visualization.format,
            plots=tuple(cfg.visualization.plots),
            radio_map_mode=cfg.visualization.radio_map_mode,
            radio_map_grid_resolution_m=cfg.visualization.radio_map_grid_resolution_m,
            radio_map_show_samples=cfg.visualization.radio_map_show_samples,
        ),
        spectrum_config=ArraySpectrumConfig(
            enabled=cfg.array.spectrum.enabled,
            sources=tuple(cfg.array.spectrum.sources),
            method=cfg.array.spectrum.method,
            zenith_bins=cfg.array.spectrum.zenith_bins,
            azimuth_bins=cfg.array.spectrum.azimuth_bins,
            zenith_min_rad=cfg.array.spectrum.zenith_min_rad,
            zenith_max_rad=cfg.array.spectrum.zenith_max_rad,
            azimuth_min_rad=cfg.array.spectrum.azimuth_min_rad,
            azimuth_max_rad=cfg.array.spectrum.azimuth_max_rad,
            normalize=cfg.array.spectrum.normalize,
            aggregate_subcarriers=cfg.array.spectrum.aggregate_subcarriers,
            aggregate_symbols=cfg.array.spectrum.aggregate_symbols,
            link_chunk_size=cfg.array.spectrum.link_chunk_size,
        ),
        bs_velocity_mps=(
            cfg.motion.bs_velocity_mps[0],
            cfg.motion.bs_velocity_mps[1],
            cfg.motion.bs_velocity_mps[2],
        ) if motion_enabled else (0.0, 0.0, 0.0),
        ue_velocity_mps=(
            cfg.motion.ue_velocity_mps[0],
            cfg.motion.ue_velocity_mps[1],
            cfg.motion.ue_velocity_mps[2],
        ) if motion_enabled else (0.0, 0.0, 0.0),
        mimo_mode=cfg.phy.mimo_mode,
        channel_backend=cfg.phy.channel_backend,
        mimo_detector=cfg.phy.mimo_detector,
        channel_estimator=cfg.phy.channel_estimator,
        receiver_failure_policy=cfg.phy.receiver_failure_policy,
        srs_config=cfg.phy.srs,
        noncooperative_config=cfg.noncooperative,
        ranging_config=to_domain_ranging_config(cfg.ranging),
    )


def _scene_config(
    template: dict[str, Any],
    entry: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    config = copy.deepcopy(template)
    input_cfg = config.setdefault("input", {})
    input_cfg["label_file"] = str(entry["label_file"])
    input_cfg["scene_file"] = str(entry["scene_file"])
    input_cfg["scene_id"] = str(entry.get("scene_key") or entry.get("scene_dir"))
    output_cfg = config.setdefault("output", {})
    output_cfg["root_dir"] = output_dir.as_posix()
    return config


def _load_scene_records_by_class(
    source_root: Path,
    classes: tuple[str, ...],
    label_name: str,
) -> dict[str, list[dict[str, Any]]]:
    manifest_path = source_root / "split_manifest.jsonl"
    records: dict[str, list[dict[str, Any]]] = {name: [] for name in classes}
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                raw = json.loads(line)
                class_name = str(raw.get("split_class") or raw.get("class") or "")
                if class_name not in records:
                    continue
                scene_dir = str(raw["scene_dir"])
                records[class_name].append(
                    _scene_record_from_dir(
                        source_root=source_root,
                        class_name=class_name,
                        scene_dir=scene_dir,
                        label_name=label_name,
                        raw=raw,
                    )
                )
        return records

    for class_name in classes:
        for scene_path in sorted((source_root / class_name).glob("front3d_*")):
            if scene_path.is_dir() or scene_path.is_symlink():
                records[class_name].append(
                    _scene_record_from_dir(
                        source_root=source_root,
                        class_name=class_name,
                        scene_dir=scene_path.name,
                        label_name=label_name,
                        raw={},
                    )
                )
    return records


def _scene_record_from_dir(
    *,
    source_root: Path,
    class_name: str,
    scene_dir: str,
    label_name: str,
    raw: dict[str, Any],
) -> dict[str, Any]:
    scene_root = source_root / class_name / scene_dir
    label_file = scene_root / "label" / label_name
    scene_file = scene_root / "scene.xml"
    if not label_file.exists():
        raise FileNotFoundError(f"Missing label file: {label_file}")
    if not scene_file.exists():
        raise FileNotFoundError(f"Missing scene file: {scene_file}")
    counts = _label_counts(label_file)
    source_index = int(raw.get("scene_index", _scene_index_from_key(scene_dir)))
    return {
        "source_root": source_root.as_posix(),
        "scene_root": scene_root.as_posix(),
        "scene_key": scene_dir,
        "scene_dir": scene_dir,
        "scene_class": class_name,
        "split": raw.get("split", ""),
        "front3d_scene_id": raw.get("front3d_scene_id", ""),
        "source_scene_index": source_index,
        "label_name": label_name,
        "label_tag": _label_tag(label_name),
        "label_file": label_file.as_posix(),
        "scene_file": scene_file.as_posix(),
        "bs_count": counts["bs_count"],
        "ue_count": counts["ue_count"],
        "link_count": counts["bs_count"] * counts["ue_count"],
    }


def _label_counts(path: Path) -> dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "bs_count": len(data.get("bs_points", [])),
        "ue_count": len(data.get("ue_points", [])),
    }


def _allocate_counts(total_count: int, available_counts: dict[str, int]) -> dict[str, int]:
    available_total = sum(available_counts.values())
    raw = {
        class_name: total_count * count / available_total
        for class_name, count in available_counts.items()
    }
    selected = {class_name: int(value) for class_name, value in raw.items()}
    remaining = total_count - sum(selected.values())
    remainders = sorted(
        raw,
        key=lambda class_name: (raw[class_name] - selected[class_name], class_name),
        reverse=True,
    )
    for class_name in remainders[:remaining]:
        selected[class_name] += 1
    for class_name, count in selected.items():
        if count > available_counts[class_name]:
            available = available_counts[class_name]
            raise ValueError(
                f"Allocated {count} scenes for {class_name}, only available {available}"
            )
    return selected


def _order_selected_records(
    selected_by_class: dict[str, list[dict[str, Any]]],
    selected_counts: dict[str, int],
    order: str,
) -> list[dict[str, Any]]:
    if order == "source":
        return sorted(
            [item for items in selected_by_class.values() for item in items],
            key=lambda item: (int(item["source_scene_index"]), item["scene_key"]),
        )
    if order != "interleaved":
        raise ValueError("order must be 'interleaved' or 'source'")
    emitted = {class_name: 0 for class_name in selected_by_class}
    total = sum(selected_counts.values())
    ordered: list[dict[str, Any]] = []
    for position in range(total):
        best_class = max(
            (
                class_name
                for class_name, count in selected_counts.items()
                if emitted[class_name] < count
            ),
            key=lambda class_name: (
                selected_counts[class_name] * (position + 1) / total - emitted[class_name],
                class_name,
            ),
        )
        ordered.append(selected_by_class[best_class][emitted[best_class]])
        emitted[best_class] += 1
    return ordered


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return payload


def _template_cross_scene_pipeline_enabled(template: dict[str, Any]) -> bool:
    output = template.get("output", {})
    if not isinstance(output, dict):
        return False
    sharding = output.get("sharding", {})
    if not isinstance(sharding, dict):
        return False
    scheduler = sharding.get("gpu_scheduler", {})
    if not isinstance(scheduler, dict):
        return False
    return bool(scheduler.get("cross_scene_pipeline", False))


def _label_tag(label_name: str) -> str:
    stem = Path(label_name).stem
    return stem.removeprefix("label_").replace("_", "")


def _scene_index_from_key(scene_key: str) -> int:
    match = re.search(r"(\d+)$", scene_key)
    return int(match.group(1)) if match else 0


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
