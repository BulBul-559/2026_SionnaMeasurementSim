"""Run SRS-like RT variants as per-link 1 BS x 1 UE jobs.

This script is intentionally an experiment helper for synthetic_array=false
cases where Sionna RT can OOM even for 6 BS x 1 UE. It preserves the same
global BS/UE coverage by creating temporary single-link label files and
encoding global indices in output directory names.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Job:
    label: str
    config_path: Path
    gpu_id: int
    bs_index: int
    ue_index: int
    output_dir: Path
    temp_config: Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="Variant spec as label=config.yaml. Repeat for multiple variants.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu-ids", default="3,4,5,6")
    parser.add_argument("--max-bs", type=int, default=6)
    parser.add_argument("--max-ue", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    variants = [_parse_variant(item) for item in args.variant]
    gpu_ids = [int(item.strip()) for item in args.gpu_ids.split(",") if item.strip()]
    if not gpu_ids:
        msg = "--gpu-ids must not be empty"
        raise ValueError(msg)

    args.output_root.mkdir(parents=True, exist_ok=True)
    jobs = _prepare_jobs(
        variants,
        output_root=args.output_root,
        gpu_ids=gpu_ids,
        max_bs=args.max_bs,
        max_ue=args.max_ue,
    )
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_job = {executor.submit(_run_job, job): job for job in jobs}
        for future in concurrent.futures.as_completed(future_to_job):
            result = future.result()
            results.append(result)
            status = "ok" if result["returncode"] == 0 else "fail"
            print(
                f"[{status}] {result['label']} bs={result['bs_index']} "
                f"ue={result['ue_index']} gpu={result['gpu_id']} "
                f"elapsed={result['elapsed_s']:.2f}s"
            )
    elapsed = time.perf_counter() - started
    summary = {
        "elapsed_s": elapsed,
        "workers": args.workers,
        "gpu_ids": gpu_ids,
        "max_bs": args.max_bs,
        "max_ue": args.max_ue,
        "jobs": sorted(
            results,
            key=lambda item: (item["label"], item["bs_index"], item["ue_index"]),
        ),
    }
    (args.output_root / "micro_sweep_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    failed = [result for result in results if result["returncode"] != 0]
    if failed:
        print(f"{len(failed)} jobs failed; see micro_sweep_manifest.json")
        return 1
    return 0


def _parse_variant(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        msg = f"Variant must be label=config.yaml, got {raw!r}"
        raise ValueError(msg)
    label, path = raw.split("=", 1)
    return label.strip(), Path(path.strip())


def _prepare_jobs(
    variants: list[tuple[str, Path]],
    *,
    output_root: Path,
    gpu_ids: list[int],
    max_bs: int,
    max_ue: int,
) -> list[Job]:
    jobs: list[Job] = []
    job_index = 0
    for label, config_path in variants:
        base_config = _read_yaml(config_path)
        label_data = _read_label(Path(base_config["input"]["label_file"]))
        variant_root = output_root / label
        config_root = output_root / "_configs" / label
        label_root = output_root / "_labels" / label
        config_root.mkdir(parents=True, exist_ok=True)
        label_root.mkdir(parents=True, exist_ok=True)
        for bs_index in range(max_bs):
            for ue_index in range(max_ue):
                link_name = f"bs{bs_index:03d}_ue{ue_index:04d}"
                link_label = _single_link_label(label_data, bs_index, ue_index)
                label_path = label_root / f"{link_name}.json"
                label_path.write_text(
                    json.dumps(link_label, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                link_config = _single_link_config(
                    base_config,
                    label_path=label_path,
                    output_dir=variant_root / link_name,
                )
                temp_config = config_root / f"{link_name}.yaml"
                temp_config.write_text(
                    yaml.safe_dump(link_config, sort_keys=False),
                    encoding="utf-8",
                )
                jobs.append(
                    Job(
                        label=label,
                        config_path=config_path,
                        gpu_id=gpu_ids[job_index % len(gpu_ids)],
                        bs_index=bs_index,
                        ue_index=ue_index,
                        output_dir=variant_root / link_name,
                        temp_config=temp_config,
                    )
                )
                job_index += 1
    return jobs


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _read_label(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _single_link_label(data: dict[str, Any], bs_index: int, ue_index: int) -> dict[str, Any]:
    copied = json.loads(json.dumps(data))
    groups = copied.get("groups")
    if not isinstance(groups, list) or not groups:
        msg = "Label JSON must contain a non-empty groups list"
        raise ValueError(msg)
    group = groups[0]
    bs_points = group.get("bs_points", [])
    ue_points = group.get("ue_points", [])
    group["bs_points"] = [bs_points[bs_index]]
    group["ue_points"] = [ue_points[ue_index]]
    return copied


def _single_link_config(
    base_config: dict[str, Any],
    *,
    label_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    config = json.loads(json.dumps(base_config))
    config["input"]["label_file"] = label_path.as_posix()
    config["input"]["max_tx"] = 1
    config["input"]["max_rx"] = 1
    config["output"]["root_dir"] = output_dir.as_posix()
    config["output"]["hdf5_filename"] = "results.h5"
    config["output"]["sharding"]["enabled"] = False
    config["output"]["sharding"]["parallel_workers"] = 1
    config["output"]["sharding"]["gpu_ids"] = []
    config["visualization"]["enabled"] = False
    config["debug"]["enabled"] = True
    return config


def _run_job(job: Job) -> dict[str, Any]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(job.gpu_id)
    job.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = job.output_dir / "run.log"
    command = [
        sys.executable,
        "-m",
        "sionna_measurement_sim.app.cli",
        "--config",
        job.temp_config.as_posix(),
        "run-full",
        "--output-dir",
        job.output_dir.as_posix(),
    ]
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        completed = subprocess.run(
            command,
            cwd=Path.cwd(),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.perf_counter() - started
    result_h5 = job.output_dir / "results.h5"
    return {
        "label": job.label,
        "bs_index": job.bs_index,
        "ue_index": job.ue_index,
        "gpu_id": job.gpu_id,
        "elapsed_s": elapsed,
        "returncode": completed.returncode,
        "output_dir": job.output_dir.as_posix(),
        "result_h5": result_h5.as_posix(),
        "result_h5_size_bytes": result_h5.stat().st_size if result_h5.exists() else 0,
        "log_path": log_path.as_posix(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
