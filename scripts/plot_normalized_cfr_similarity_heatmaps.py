"""Normalize existing CFR similarity CSVs and plot floorplan heatmaps.

This is a post-processing helper for outputs created by
plot_cfr_similarity_floorplan_heatmaps.py. It excludes zero-valued similarity
samples per metric, applies min-max normalization, and writes plots to a new
output tree without touching the original comparison directories.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import plot_cfr_similarity_floorplan_heatmaps as base  # noqa: E402

METRICS = ("magnitude", "phase", "i", "q")


@dataclass(frozen=True)
class Comparison:
    label: str
    csv_path: Path
    floorplan_image: Path
    floorplan_meta: Path


DEFAULT_COMPARISONS = (
    Comparison(
        label="medium_0000_vs_medium_0001",
        csv_path=REPO_ROOT
        / "outputs/cfr_similarity_medium_0000_vs_medium_0001/cfr_similarity_by_ue.csv",
        floorplan_image=REPO_ROOT / "data/medium/medium_0000/floorplan/000_z_1.60.png",
        floorplan_meta=REPO_ROOT / "data/medium/medium_0000/floorplan/meta.json",
    ),
    Comparison(
        label="dense_0000_vs_sparse_0000",
        csv_path=REPO_ROOT
        / "outputs/cfr_similarity_dense_0000_vs_sparse_0000/cfr_similarity_by_ue.csv",
        floorplan_image=REPO_ROOT / "data/dense/dense_0000/floorplan/000_z_1.60.png",
        floorplan_meta=REPO_ROOT / "data/dense/dense_0000/floorplan/meta.json",
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/cfr_similarity_normalized_nonzero_heatmaps",
        help="New output directory for normalized plots.",
    )
    parser.add_argument(
        "--normalization",
        choices=("per_metric_global", "all_metrics_global", "per_comparison_metric"),
        default="per_metric_global",
        help=(
            "per_metric_global normalizes each metric using both comparisons; "
            "all_metrics_global uses one min/max for every metric and comparison; "
            "per_comparison_metric normalizes each metric inside each comparison."
        ),
    )
    parser.add_argument("--image-origin", choices=("upper", "lower"), default="upper")
    parser.add_argument("--heatmap-alpha", type=float, default=0.68)
    parser.add_argument("--point-size", type=float, default=5.0)
    parser.add_argument("--show-samples", action="store_true")
    args = parser.parse_args()

    comparisons = list(DEFAULT_COMPARISONS)
    raw_tables = {
        comparison.label: _read_similarity_csv(comparison.csv_path)
        for comparison in comparisons
    }
    clean_tables, zero_counts = _drop_metric_zeros(raw_tables)
    ranges = _normalization_ranges(clean_tables, args.normalization)
    normalized_tables = {
        label: _normalize_table(table, ranges[label]) for label, table in clean_tables.items()
    }

    args.output_dir.mkdir(parents=True, exist_ok=False)
    summary: dict[str, Any] = {
        "normalization": args.normalization,
        "zero_policy": "Similarity values <= 0 are excluded per metric before normalization.",
        "comparisons": {},
    }

    for comparison in comparisons:
        label = comparison.label
        comparison_dir = args.output_dir / label
        comparison_dir.mkdir(parents=True, exist_ok=False)
        table = normalized_tables[label]
        plot_args = argparse.Namespace(
            floorplan_image=comparison.floorplan_image,
            floorplan_meta=comparison.floorplan_meta,
            image_origin=args.image_origin,
            heatmap_alpha=args.heatmap_alpha,
            point_size=args.point_size,
            show_samples=args.show_samples,
        )

        normalized_csv = comparison_dir / "cfr_similarity_normalized_nonzero_by_ue.csv"
        normalized_stats = comparison_dir / "cfr_similarity_normalized_nonzero_stats.csv"
        original_stats = comparison_dir / "cfr_similarity_original_nonzero_stats.csv"
        base._write_csv(normalized_csv, table)
        base._write_stats_csv(normalized_stats, table)
        base._write_stats_csv(original_stats, clean_tables[label])
        heatmaps = base._plot_heatmaps(plot_args, table, output_dir=comparison_dir)

        summary["comparisons"][label] = {
            "source_csv": comparison.csv_path.as_posix(),
            "floorplan_image": comparison.floorplan_image.as_posix(),
            "floorplan_meta": comparison.floorplan_meta.as_posix(),
            "ue_count": int(table["ue_index"].size),
            "zero_counts": zero_counts[label],
            "normalization_ranges": ranges[label],
            "outputs": {
                "normalized_csv": normalized_csv.as_posix(),
                "normalized_stats": normalized_stats.as_posix(),
                "original_nonzero_stats": original_stats.as_posix(),
                **{key: value.as_posix() for key, value in heatmaps.items()},
            },
        }

    ranges_csv = args.output_dir / "normalization_ranges.csv"
    _write_ranges_csv(ranges_csv, ranges)
    summary["normalization_ranges_csv"] = ranges_csv.as_posix()
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {"output_dir": args.output_dir.as_posix(), "summary": summary_path.as_posix()}
        )
    )
    return 0


def _read_similarity_csv(path: Path) -> dict[str, np.ndarray]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows.extend(reader)
    if not rows:
        msg = f"No rows found in {path}"
        raise ValueError(msg)

    positions = np.asarray(
        [[float(row["x_m"]), float(row["y_m"]), float(row["z_m"])] for row in rows],
        dtype=np.float32,
    )
    table = {
        "ue_index": np.asarray([int(row["ue_index"]) for row in rows], dtype=np.int64),
        "position_m": positions,
    }
    for metric in METRICS:
        table[metric] = np.asarray(
            [float(row[f"{metric}_similarity"]) for row in rows], dtype=np.float32
        )
    return table


def _drop_metric_zeros(
    tables: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, int]]]:
    clean_tables: dict[str, dict[str, np.ndarray]] = {}
    zero_counts: dict[str, dict[str, int]] = {}
    for label, table in tables.items():
        clean = {
            "ue_index": table["ue_index"],
            "position_m": table["position_m"],
        }
        zero_counts[label] = {}
        for metric in METRICS:
            values = np.asarray(table[metric], dtype=np.float32).copy()
            zeros = np.isfinite(values) & (values <= 0.0)
            zero_counts[label][metric] = int(np.count_nonzero(zeros))
            values[zeros] = np.nan
            clean[metric] = values
        clean_tables[label] = clean
    return clean_tables, zero_counts


def _normalization_ranges(
    tables: dict[str, dict[str, np.ndarray]],
    mode: str,
) -> dict[str, dict[str, dict[str, float]]]:
    if mode == "per_metric_global":
        metric_ranges = {
            metric: _range_for_values([table[metric] for table in tables.values()])
            for metric in METRICS
        }
        return {
            label: {metric: dict(metric_ranges[metric]) for metric in METRICS}
            for label in tables
        }
    if mode == "all_metrics_global":
        common = _range_for_values(
            [table[metric] for table in tables.values() for metric in METRICS]
        )
        return {label: {metric: dict(common) for metric in METRICS} for label in tables}
    if mode == "per_comparison_metric":
        return {
            label: {metric: _range_for_values([table[metric]]) for metric in METRICS}
            for label, table in tables.items()
        }
    msg = f"Unknown normalization mode: {mode}"
    raise ValueError(msg)


def _range_for_values(values_list: list[np.ndarray]) -> dict[str, float]:
    finite = np.concatenate([values[np.isfinite(values)] for values in values_list])
    if finite.size == 0:
        return {"min": float("nan"), "max": float("nan")}
    return {"min": float(np.min(finite)), "max": float(np.max(finite))}


def _normalize_table(
    table: dict[str, np.ndarray],
    ranges: dict[str, dict[str, float]],
) -> dict[str, np.ndarray]:
    normalized = {
        "ue_index": table["ue_index"],
        "position_m": table["position_m"],
    }
    for metric in METRICS:
        values = np.asarray(table[metric], dtype=np.float32)
        min_value = ranges[metric]["min"]
        max_value = ranges[metric]["max"]
        if not np.isfinite(min_value) or not np.isfinite(max_value):
            normalized[metric] = np.full(values.shape, np.nan, dtype=np.float32)
            continue
        scale = max_value - min_value
        if scale <= 0.0:
            out = np.where(np.isfinite(values), 1.0, np.nan).astype(np.float32)
        else:
            out = ((values - min_value) / scale).astype(np.float32)
            out = np.where(np.isfinite(values), np.clip(out, 0.0, 1.0), np.nan)
        normalized[metric] = out
    return normalized


def _write_ranges_csv(
    path: Path,
    ranges: dict[str, dict[str, dict[str, float]]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["comparison", "metric", "min", "max"])
        writer.writeheader()
        for comparison, metric_ranges in ranges.items():
            for metric, values in metric_ranges.items():
                writer.writerow(
                    {
                        "comparison": comparison,
                        "metric": metric,
                        "min": values["min"],
                        "max": values["max"],
                    }
                )


if __name__ == "__main__":
    raise SystemExit(main())
