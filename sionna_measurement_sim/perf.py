"""Opt-in performance tracing helpers."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class PerfTracer:
    """Small JSONL/CSV tracer used when debug profiling is enabled."""

    def __init__(
        self,
        output_dir: str | Path,
        config: Any | None = None,
        *,
        worker_id: str = "main",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.config = config
        self.enabled = bool(getattr(config, "enabled", False))
        self.hardware_interval_s = float(getattr(config, "hardware_interval_s", 1.0))
        self.link_log_interval = int(getattr(config, "link_log_interval", 250))
        self.torch_synchronize = bool(getattr(config, "torch_synchronize", True))
        self.write_hardware_samples = bool(
            getattr(config, "write_hardware_samples", True)
        )
        self.worker_id = worker_id
        self.logs_dir = self.output_dir / "logs"
        suffix = "" if worker_id == "main" else f"_{worker_id}"
        self.events_path = self.logs_dir / f"perf_events{suffix}.jsonl"
        self.hardware_path = self.logs_dir / f"hardware_samples{suffix}.csv"
        self.link_chunks_path = self.logs_dir / f"link_chunks{suffix}.csv"
        self.summary_path = self.logs_dir / f"perf_summary{suffix}.json"
        self._events_file = None
        self._hardware_file = None
        self._hardware_writer = None
        self._stop_event = threading.Event()
        self._hardware_thread: threading.Thread | None = None
        self._stages: list[dict[str, Any]] = []
        self._dataset_writes: list[dict[str, Any]] = []
        self._hardware_samples: list[dict[str, str]] = []
        self._run_start = 0.0
        self._finished = False

    def start(self) -> None:
        if not self.enabled:
            return
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._run_start = time.perf_counter()
        self._events_file = self.events_path.open("w", encoding="utf-8")
        self.record_event("run_start", worker_id=self.worker_id)
        if self.write_hardware_samples:
            self._hardware_file = self.hardware_path.open("w", newline="", encoding="utf-8")
            fieldnames = (
                "timestamp_s",
                "worker_id",
                "pid",
                "rss_mb",
                "thread_count",
                "cpu_percent",
                "gpu_index",
                "gpu_util_percent",
                "gpu_mem_used_mb",
                "gpu_mem_total_mb",
                "gpu_power_w",
                "gpu_temp_c",
            )
            self._hardware_writer = csv.DictWriter(self._hardware_file, fieldnames=fieldnames)
            self._hardware_writer.writeheader()
            self._hardware_thread = threading.Thread(
                target=self._sample_hardware_loop,
                name="sionna-perf-hardware-sampler",
                daemon=True,
            )
            self._hardware_thread.start()

    @contextmanager
    def span(self, name: str, **metadata: Any) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        self._sync_torch()
        start = time.perf_counter()
        self.record_event(f"{name}.start", **metadata)
        try:
            yield
        finally:
            self._sync_torch()
            end = time.perf_counter()
            duration_s = end - start
            stage = {
                "name": name,
                "duration_s": duration_s,
                "start_offset_s": start - self._run_start,
                "end_offset_s": end - self._run_start,
                **metadata,
            }
            self._stages.append(stage)
            self.record_event(f"{name}.end", duration_s=duration_s, **metadata)

    def record_event(self, event: str, **payload: Any) -> None:
        if not self.enabled or self._events_file is None:
            return
        if event == "hdf5.dataset_write":
            self._dataset_writes.append(dict(payload))
        data = {
            "timestamp_s": time.time(),
            "offset_s": time.perf_counter() - self._run_start if self._run_start else 0.0,
            "event": event,
            "worker_id": self.worker_id,
            **payload,
        }
        self._events_file.write(json.dumps(data, sort_keys=True) + "\n")
        self._events_file.flush()

    def record_link_chunk(self, **payload: Any) -> None:
        if not self.enabled:
            return
        write_header = not self.link_chunks_path.exists()
        self.link_chunks_path.parent.mkdir(parents=True, exist_ok=True)
        with self.link_chunks_path.open("a", newline="", encoding="utf-8") as handle:
            fieldnames = tuple(payload.keys())
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(payload)

    def finish(
        self,
        extra: dict[str, Any] | None = None,
        *,
        status: str = "success",
        exception: BaseException | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {}
        if self._finished:
            return {}
        self._finished = True
        total_s = time.perf_counter() - self._run_start
        if exception is not None:
            self.record_event(
                "run_failed",
                total_duration_s=total_s,
                exception_type=type(exception).__name__,
                exception_message=str(exception),
            )
        self.record_event("run_end", total_duration_s=total_s, status=status)
        self._stop_event.set()
        if self._hardware_thread is not None:
            self._hardware_thread.join(timeout=max(self.hardware_interval_s * 2.0, 1.0))
        if self._hardware_file is not None:
            self._hardware_file.close()
        if self._events_file is not None:
            self._events_file.close()

        summary = {
            "enabled": True,
            "worker_id": self.worker_id,
            "status": status,
            "exception": _exception_summary(exception),
            "total_duration_s": total_s,
            "stages": self._stages,
            "stage_totals_s": _stage_totals(self._stages),
            "hardware_summary": _hardware_summary(self._hardware_samples),
            "dataset_write_summary": _dataset_write_summary(self._dataset_writes),
            "logs": {
                "events": self.events_path.as_posix(),
                "hardware_samples": self.hardware_path.as_posix()
                if self.write_hardware_samples
                else "",
                "link_chunks": self.link_chunks_path.as_posix()
                if self.link_chunks_path.exists()
                else "",
            },
            **(extra or {}),
        }
        self.summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return summary

    def _sample_hardware_loop(self) -> None:
        while not self._stop_event.is_set():
            self._write_hardware_sample()
            self._stop_event.wait(self.hardware_interval_s)

    def _write_hardware_sample(self) -> None:
        if self._hardware_writer is None or self._hardware_file is None:
            return
        base = {
            "timestamp_s": f"{time.time():.6f}",
            "worker_id": self.worker_id,
            "pid": os.getpid(),
            **_process_sample(),
        }
        gpu_rows = _gpu_samples()
        if not gpu_rows:
            row = {
                **base,
                "gpu_index": "",
                "gpu_util_percent": "",
                "gpu_mem_used_mb": "",
                "gpu_mem_total_mb": "",
                "gpu_power_w": "",
                "gpu_temp_c": "",
            }
            self._hardware_samples.append(row)
            self._hardware_writer.writerow(row)
        else:
            for row in gpu_rows:
                sample = {**base, **row}
                self._hardware_samples.append(sample)
                self._hardware_writer.writerow(sample)
        self._hardware_file.flush()

    def _sync_torch(self) -> None:
        if not self.torch_synchronize:
            return
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            return


def _stage_totals(stages: list[dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for stage in stages:
        name = str(stage["name"])
        totals[name] = totals.get(name, 0.0) + float(stage["duration_s"])
    return totals


def _exception_summary(exception: BaseException | None) -> dict[str, str] | None:
    if exception is None:
        return None
    return {
        "type": type(exception).__name__,
        "message": str(exception),
    }


def _hardware_summary(samples: list[dict[str, str]]) -> dict[str, Any]:
    rss_values = [_parse_float(row.get("rss_mb", "")) for row in samples]
    cpu_values = [_parse_float(row.get("cpu_percent", "")) for row in samples]
    thread_values = [_parse_float(row.get("thread_count", "")) for row in samples]
    gpu_mem_values = [_parse_float(row.get("gpu_mem_used_mb", "")) for row in samples]
    gpu_util_values = [_parse_float(row.get("gpu_util_percent", "")) for row in samples]
    gpu_rows = [row for row in samples if str(row.get("gpu_index", "")).strip()]
    return {
        "sample_count": len(samples),
        "gpu_sample_count": len(gpu_rows),
        "peak_rss_mb": _max_or_none(rss_values),
        "max_thread_count": _max_or_none(thread_values),
        "max_cpu_percent": _max_or_none(cpu_values),
        "peak_gpu_mem_used_mb": _max_or_none(gpu_mem_values),
        "max_gpu_util_percent": _max_or_none(gpu_util_values),
        "mean_gpu_util_percent": _mean_or_none(gpu_util_values),
    }


def _dataset_write_summary(events: list[dict[str, Any]], *, top_n: int = 10) -> dict[str, Any]:
    total_raw = sum(int(event.get("raw_bytes", 0) or 0) for event in events)
    storage_values = [int(event.get("storage_bytes", -1) or -1) for event in events]
    positive_storage = [value for value in storage_values if value >= 0]
    total_storage = sum(positive_storage)
    top_by_duration = sorted(
        events,
        key=lambda event: float(event.get("duration_s", 0.0) or 0.0),
        reverse=True,
    )[:top_n]
    top_by_raw = sorted(
        events,
        key=lambda event: int(event.get("raw_bytes", 0) or 0),
        reverse=True,
    )[:top_n]
    return {
        "dataset_count": len(events),
        "total_raw_bytes": total_raw,
        "total_storage_bytes": total_storage if positive_storage else None,
        "storage_to_raw_ratio": (
            float(total_storage) / float(total_raw)
            if total_raw > 0 and positive_storage
            else None
        ),
        "raw_to_storage_ratio": (
            float(total_raw) / float(total_storage)
            if total_storage > 0 and positive_storage
            else None
        ),
        "top_by_duration": [_dataset_write_item(event) for event in top_by_duration],
        "top_by_raw_bytes": [_dataset_write_item(event) for event in top_by_raw],
    }


def _dataset_write_item(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(event.get("path", "")),
        "shape": list(event.get("shape", ())),
        "dtype": str(event.get("dtype", "")),
        "raw_bytes": int(event.get("raw_bytes", 0) or 0),
        "storage_bytes": int(event.get("storage_bytes", -1) or -1),
        "compression": str(event.get("compression", "")),
        "compression_opts": event.get("compression_opts"),
        "duration_s": float(event.get("duration_s", 0.0) or 0.0),
    }


def _parse_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _max_or_none(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return max(finite) if finite else None


def _mean_or_none(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None]
    return sum(finite) / len(finite) if finite else None


def _process_sample() -> dict[str, str]:
    try:
        import psutil

        proc = psutil.Process()
        return {
            "rss_mb": f"{proc.memory_info().rss / (1024.0 * 1024.0):.3f}",
            "thread_count": str(proc.num_threads()),
            "cpu_percent": f"{proc.cpu_percent(interval=None):.3f}",
        }
    except Exception:
        return _process_sample_procfs()


def _process_sample_procfs() -> dict[str, str]:
    rss_mb = ""
    thread_count = ""
    try:
        status = Path(f"/proc/{os.getpid()}/status").read_text(encoding="utf-8")
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                rss_mb = f"{float(parts[1]) / 1024.0:.3f}"
            elif line.startswith("Threads:"):
                thread_count = line.split()[1]
    except Exception:
        pass
    return {"rss_mb": rss_mb, "thread_count": thread_count, "cpu_percent": ""}


def _gpu_samples() -> list[dict[str, str]]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
    except Exception:
        return []
    rows: list[dict[str, str]] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        rows.append(
            {
                "gpu_index": parts[0],
                "gpu_util_percent": parts[1],
                "gpu_mem_used_mb": parts[2],
                "gpu_mem_total_mb": parts[3],
                "gpu_power_w": parts[4],
                "gpu_temp_c": parts[5],
            }
        )
    return rows
