#!/usr/bin/env python
"""Summarize NR SRS resource observation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from sionna_measurement_sim.io.schema_validator import validate_hdf5_contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="HDF5 file or output directory")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    h5_paths = _discover_h5(args.input)
    if not h5_paths:
        raise SystemExit(f"No HDF5 files found under {args.input}")
    output_dir = args.output_dir or _default_output_dir(args.input)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = [_summarize_one(path) for path in h5_paths]
    aggregate = _aggregate(records)
    summary = {
        "input": str(args.input),
        "file_count": len(records),
        "files": records,
        "aggregate": aggregate,
    }
    (output_dir / "srs_resource_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_csv(output_dir / "srs_resource_summary.csv", records)
    print(output_dir / "srs_resource_summary.json")
    return 0


def _discover_h5(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    candidates = []
    if (path / "results.h5").exists():
        candidates.append(path / "results.h5")
    results_dir = path / "results"
    if results_dir.exists():
        candidates.extend(sorted(results_dir.glob("*.h5")))
    return candidates


def _default_output_dir(path: Path) -> Path:
    return path.parent / "srs_resource_summary" if path.is_file() else path / "srs_resource_summary"


def _summarize_one(path: Path) -> dict[str, object]:
    validate_hdf5_contract(path)
    with h5py.File(path, "r") as h5:
        truth = h5["channel/truth/cfr"][()]
        cfr_resource = h5["observation/cfr_est_resource"][()]
        re_indices = h5["waveform/srs_re_subcarrier_indices"][()]
        truth_resource = truth[np.newaxis, ..., re_indices]
        resource_error = cfr_resource - truth_resource
        resource_nmse = _nmse_db(truth_resource, cfr_resource)
        full_nmse = h5["evaluation/nmse_db"][()]
        eval_resource_nmse = h5["evaluation/srs_resource_nmse_db"][()]
        eval_interp_nmse = h5["evaluation/srs_interpolation_nmse_db"][()]
        mask = h5["waveform/srs_resource_mask"][()]
        return {
            "path": str(path),
            "schema_pass": True,
            "resource_re_count": int(re_indices.size),
            "resource_mask_count": int(np.count_nonzero(mask)),
            "tx_grid_shape": list(h5["waveform/tx_grid"].shape),
            "rx_grid_shape": list(h5["waveform/rx_grid"].shape),
            "resource_nmse_db_mean": _finite_mean(resource_nmse),
            "resource_nmse_db_median": _finite_median(resource_nmse),
            "resource_abs_error_mean": _finite_mean(np.abs(resource_error)),
            "full_nmse_db_mean": _finite_mean(full_nmse),
            "full_nmse_db_median": _finite_median(full_nmse),
            "eval_resource_nmse_db_mean": _finite_mean(eval_resource_nmse),
            "eval_interpolation_nmse_db_mean": _finite_mean(eval_interp_nmse),
        }


def _aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    keys = (
        "resource_nmse_db_mean",
        "resource_abs_error_mean",
        "full_nmse_db_mean",
        "eval_resource_nmse_db_mean",
        "eval_interpolation_nmse_db_mean",
    )
    aggregate: dict[str, object] = {
        "schema_pass": all(bool(record["schema_pass"]) for record in records),
        "total_resource_re_count": int(sum(int(record["resource_re_count"]) for record in records)),
    }
    for key in keys:
        aggregate[key] = _finite_mean(np.asarray([record[key] for record in records]))
    return aggregate


def _write_csv(path: Path, records: list[dict[str, object]]) -> None:
    columns = (
        "path",
        "schema_pass",
        "resource_re_count",
        "resource_mask_count",
        "resource_nmse_db_mean",
        "resource_nmse_db_median",
        "resource_abs_error_mean",
        "full_nmse_db_mean",
        "full_nmse_db_median",
        "eval_resource_nmse_db_mean",
        "eval_interpolation_nmse_db_mean",
    )
    lines = [",".join(columns)]
    for record in records:
        lines.append(",".join(str(record[column]) for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _nmse_db(truth: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    signal = np.sum(np.abs(truth) ** 2, axis=(3, 4, 5))
    error = np.sum(np.abs(estimate - truth) ** 2, axis=(3, 4, 5))
    return 10.0 * np.log10(np.maximum(error / np.maximum(signal, 1e-30), 1e-30))


def _finite_mean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _finite_median(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return float(np.median(finite)) if finite.size else float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
