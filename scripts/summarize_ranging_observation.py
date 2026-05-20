#!/usr/bin/env python3
"""Summarize `/ranging` outputs in a SionnaMeasurementSim HDF5 file."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", required=True, help="Input results.h5")
    parser.add_argument("--output-dir", required=True, help="Directory for summary files")
    args = parser.parse_args()

    hdf5_path = Path(args.hdf5)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_hdf5(hdf5_path)
    write_summary(summary, output_dir)
    plot_summary(summary, output_dir)
    return 0


def summarize_hdf5(path: Path) -> dict[str, object]:
    with h5py.File(path, "r") as h5:
        if "ranging" not in h5:
            raise ValueError(f"{path} does not contain /ranging")
        truth = h5["derived/first_path_propagation_range_m"][()]
        truth_by_snapshot = np.broadcast_to(
            truth[np.newaxis, ...],
            h5["observation/cfr_est"].shape[:3],
        )
        summary: dict[str, object] = {
            "hdf5": path.as_posix(),
            "default_estimator": _read_str(h5["ranging/default_estimator"][()]),
            "estimators": {},
        }
        for estimator in ("pdp_peak", "phase_slope"):
            if f"ranging/{estimator}" not in h5:
                continue
            group = h5[f"ranging/{estimator}"]
            estimate = group["one_way_range_est_m"][()]
            error = group["range_error_m"][()]
            success = group["detection_success"][()].astype(bool)
            finite = success & np.isfinite(estimate) & np.isfinite(error)
            abs_error = np.abs(error[finite])
            summary["estimators"][estimator] = {
                "finite_rate": float(np.mean(finite)) if finite.size else 0.0,
                "success_count": int(np.sum(finite)),
                "total_count": int(finite.size),
                "mean_abs_range_error_m": _stat(abs_error, "mean"),
                "median_abs_range_error_m": _stat(abs_error, "median"),
                "p80_abs_range_error_m": _stat(abs_error, "p80"),
                "p95_abs_range_error_m": _stat(abs_error, "p95"),
                "truth_range_m": truth_by_snapshot[finite].astype(float).tolist(),
                "one_way_range_est_m": estimate[finite].astype(float).tolist(),
                "range_error_m": error[finite].astype(float).tolist(),
            }
    return summary


def write_summary(summary: dict[str, object], output_dir: Path) -> None:
    json_path = output_dir / "ranging_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    rows = []
    for estimator, metrics in summary["estimators"].items():
        rows.append(
            {
                "estimator": estimator,
                "finite_rate": metrics["finite_rate"],
                "success_count": metrics["success_count"],
                "total_count": metrics["total_count"],
                "mean_abs_range_error_m": metrics["mean_abs_range_error_m"],
                "median_abs_range_error_m": metrics["median_abs_range_error_m"],
                "p80_abs_range_error_m": metrics["p80_abs_range_error_m"],
                "p95_abs_range_error_m": metrics["p95_abs_range_error_m"],
            }
        )
    csv_path = output_dir / "ranging_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["estimator"])
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary: dict[str, object], output_dir: Path) -> None:
    estimators = summary["estimators"]
    if not estimators:
        return
    fig, axes = plt.subplots(1, len(estimators), figsize=(5 * len(estimators), 4), squeeze=False)
    for ax, (estimator, metrics) in zip(axes[0], estimators.items(), strict=False):
        truth = np.asarray(metrics["truth_range_m"], dtype=float)
        estimate = np.asarray(metrics["one_way_range_est_m"], dtype=float)
        ax.scatter(truth, estimate, s=18, alpha=0.75)
        if truth.size and estimate.size:
            low = float(min(np.min(truth), np.min(estimate)))
            high = float(max(np.max(truth), np.max(estimate)))
            ax.plot([low, high], [low, high], color="black", linewidth=1.0)
        ax.set_title(estimator)
        ax.set_xlabel("truth first-path range [m]")
        ax.set_ylabel("estimated one-way range [m]")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "ranging_truth_vs_estimate.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(estimators), figsize=(5 * len(estimators), 4), squeeze=False)
    for ax, (estimator, metrics) in zip(axes[0], estimators.items(), strict=False):
        error = np.asarray(metrics["range_error_m"], dtype=float)
        ax.hist(error, bins=min(20, max(5, error.size // 2)) if error.size else 5)
        ax.set_title(estimator)
        ax.set_xlabel("range error [m]")
        ax.set_ylabel("count")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "ranging_range_error_hist.png", dpi=150)
    plt.close(fig)


def _read_str(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _stat(values: np.ndarray, name: str) -> float:
    if values.size == 0:
        return float("nan")
    if name == "mean":
        return float(np.mean(values))
    if name == "median":
        return float(np.median(values))
    if name == "p80":
        return float(np.percentile(values, 80))
    if name == "p95":
        return float(np.percentile(values, 95))
    raise ValueError(name)


if __name__ == "__main__":
    raise SystemExit(main())
