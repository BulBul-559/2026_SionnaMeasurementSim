"""Plot CFR similarity heatmaps after excluding a rectangular region.

The default inputs use the already computed medium-vs-medium and
dense-vs-sparse similarity CSVs. Points inside the exclusion rectangle are not
used for statistics or heatmap samples. Zero similarity values are treated as
missing by default, which removes the known missing-shard zeros from sparse.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import plot_cfr_similarity_floorplan_heatmaps as base  # noqa: E402

METRICS = ("magnitude", "phase", "i", "q")
METRIC_TITLES = {
    "magnitude": "Magnitude Similarity",
    "phase": "Phase Similarity",
    "i": "I Similarity",
    "q": "Q Similarity",
}


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
    parser.add_argument("--xmin", type=float, default=4.0)
    parser.add_argument("--xmax", type=float, default=11.0)
    parser.add_argument("--ymin", type=float, default=5.0)
    parser.add_argument("--ymax", type=float, default=9.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/cfr_similarity_outer_ring_exclude_x4_11_y5_9",
    )
    parser.add_argument(
        "--keep-zero",
        action="store_true",
        help="Keep zero similarity values instead of treating them as missing.",
    )
    parser.add_argument("--image-origin", choices=("upper", "lower"), default="upper")
    parser.add_argument("--heatmap-alpha", type=float, default=0.68)
    parser.add_argument("--show-samples", action="store_true")
    parser.add_argument("--point-size", type=float, default=5.0)
    args = parser.parse_args()

    exclude_rect = (args.xmin, args.xmax, args.ymin, args.ymax)
    comparisons = list(DEFAULT_COMPARISONS)
    raw_tables = {item.label: _read_similarity_csv(item.csv_path) for item in comparisons}
    filtered_tables, counts = _filter_tables(
        raw_tables,
        exclude_rect=exclude_rect,
        drop_zero=not args.keep_zero,
    )
    normalized_tables, ranges = _normalize_per_metric_global(filtered_tables)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    raw_dir = args.output_dir / "raw"
    normalized_dir = args.output_dir / "normalized_per_metric_global"
    raw_dir.mkdir()
    normalized_dir.mkdir()

    raw_stats = args.output_dir / "outer_ring_similarity_stats_raw.csv"
    normalized_stats = args.output_dir / "outer_ring_similarity_stats_normalized.csv"
    _write_comparison_stats(raw_stats, filtered_tables)
    _write_comparison_stats(normalized_stats, normalized_tables)

    summary: dict[str, Any] = {
        "exclude_rect": {
            "xmin": args.xmin,
            "xmax": args.xmax,
            "ymin": args.ymin,
            "ymax": args.ymax,
            "inside_policy": "Points inside this rectangle are excluded from stats and plots.",
        },
        "zero_policy": "kept" if args.keep_zero else "values <= 0 are treated as missing",
        "raw_stats": raw_stats.as_posix(),
        "normalized_stats": normalized_stats.as_posix(),
        "normalization": {
            "mode": "per_metric_global",
            "ranges": ranges,
        },
        "comparisons": {},
    }

    for comparison in comparisons:
        raw_outputs = _write_plots(
            raw_dir / comparison.label,
            comparison,
            filtered_tables[comparison.label],
            exclude_rect=exclude_rect,
            image_origin=args.image_origin,
            heatmap_alpha=args.heatmap_alpha,
            point_size=args.point_size,
            show_samples=args.show_samples,
        )
        normalized_outputs = _write_plots(
            normalized_dir / comparison.label,
            comparison,
            normalized_tables[comparison.label],
            exclude_rect=exclude_rect,
            image_origin=args.image_origin,
            heatmap_alpha=args.heatmap_alpha,
            point_size=args.point_size,
            show_samples=args.show_samples,
        )
        summary["comparisons"][comparison.label] = {
            "source_csv": comparison.csv_path.as_posix(),
            "source_ue_count": counts[comparison.label]["source_ue_count"],
            "outside_ue_count": counts[comparison.label]["outside_ue_count"],
            "inside_excluded_ue_count": counts[comparison.label]["inside_excluded_ue_count"],
            "zero_counts_outside": counts[comparison.label]["zero_counts_outside"],
            "raw_outputs": {key: value.as_posix() for key, value in raw_outputs.items()},
            "normalized_outputs": {
                key: value.as_posix() for key, value in normalized_outputs.items()
            },
        }

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
        rows.extend(csv.DictReader(file))
    if not rows:
        msg = f"No rows found in {path}"
        raise ValueError(msg)

    table = {
        "ue_index": np.asarray([int(row["ue_index"]) for row in rows], dtype=np.int64),
        "position_m": np.asarray(
            [[float(row["x_m"]), float(row["y_m"]), float(row["z_m"])] for row in rows],
            dtype=np.float32,
        ),
    }
    for metric in METRICS:
        table[metric] = np.asarray(
            [float(row[f"{metric}_similarity"]) for row in rows], dtype=np.float32
        )
    return table


def _filter_tables(
    tables: dict[str, dict[str, np.ndarray]],
    *,
    exclude_rect: tuple[float, float, float, float],
    drop_zero: bool,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    filtered: dict[str, dict[str, np.ndarray]] = {}
    counts: dict[str, Any] = {}
    xmin, xmax, ymin, ymax = exclude_rect
    for label, table in tables.items():
        positions = table["position_m"]
        inside = (
            (positions[:, 0] >= xmin)
            & (positions[:, 0] <= xmax)
            & (positions[:, 1] >= ymin)
            & (positions[:, 1] <= ymax)
        )
        outside = ~inside
        out_table = {
            "ue_index": table["ue_index"][outside],
            "position_m": positions[outside],
        }
        zero_counts: dict[str, int] = {}
        for metric in METRICS:
            values = table[metric][outside].astype(np.float32, copy=True)
            zeros = np.isfinite(values) & (values <= 0.0)
            zero_counts[metric] = int(np.count_nonzero(zeros))
            if drop_zero:
                values[zeros] = np.nan
            out_table[metric] = values
        filtered[label] = out_table
        counts[label] = {
            "source_ue_count": int(table["ue_index"].size),
            "inside_excluded_ue_count": int(np.count_nonzero(inside)),
            "outside_ue_count": int(np.count_nonzero(outside)),
            "zero_counts_outside": zero_counts,
        }
    return filtered, counts


def _normalize_per_metric_global(
    tables: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, float]]]:
    ranges = {
        metric: _range_for_values([table[metric] for table in tables.values()])
        for metric in METRICS
    }
    normalized: dict[str, dict[str, np.ndarray]] = {}
    for label, table in tables.items():
        out = {
            "ue_index": table["ue_index"],
            "position_m": table["position_m"],
        }
        for metric in METRICS:
            values = table[metric]
            min_value = ranges[metric]["min"]
            max_value = ranges[metric]["max"]
            scale = max_value - min_value
            if not np.isfinite(scale) or scale <= 0.0:
                out[metric] = np.where(np.isfinite(values), 1.0, np.nan).astype(np.float32)
            else:
                normalized_values = (values - min_value) / scale
                out[metric] = np.where(
                    np.isfinite(values),
                    np.clip(normalized_values, 0.0, 1.0),
                    np.nan,
                ).astype(np.float32)
        normalized[label] = out
    return normalized, ranges


def _range_for_values(values_list: list[np.ndarray]) -> dict[str, float]:
    finite = np.concatenate([values[np.isfinite(values)] for values in values_list])
    if finite.size == 0:
        return {"min": float("nan"), "max": float("nan")}
    return {"min": float(np.min(finite)), "max": float(np.max(finite))}


def _write_comparison_stats(path: Path, tables: dict[str, dict[str, np.ndarray]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["comparison", "metric", "count", "min", "max", "mean"],
        )
        writer.writeheader()
        for comparison, table in tables.items():
            for metric in METRICS:
                stats = _metric_summary(table[metric])
                writer.writerow(
                    {
                        "comparison": comparison,
                        "metric": metric,
                        **stats,
                    }
                )


def _metric_summary(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "count": 0,
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
        }
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def _write_plots(
    output_dir: Path,
    comparison: Comparison,
    table: dict[str, np.ndarray],
    *,
    exclude_rect: tuple[float, float, float, float],
    image_origin: str,
    heatmap_alpha: float,
    point_size: float,
    show_samples: bool,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_filtered_csv(output_dir / "cfr_similarity_outer_ring_by_ue.csv", table)
    floorplan = np.asarray(Image.open(comparison.floorplan_image).convert("RGB"))
    meta = json.loads(comparison.floorplan_meta.read_text(encoding="utf-8"))
    extent = base._floorplan_extent(meta, floorplan)

    paths: dict[str, Path] = {}
    for metric in METRICS:
        path = output_dir / f"cfr_similarity_{metric}_outer_ring_floorplan.png"
        _plot_single_heatmap(
            floorplan,
            extent,
            table,
            metric,
            output_path=path,
            exclude_rect=exclude_rect,
            image_origin=image_origin,
            heatmap_alpha=heatmap_alpha,
            point_size=point_size,
            show_samples=show_samples,
        )
        paths[metric] = path

    combined = output_dir / "cfr_similarity_four_panel_outer_ring_floorplan.png"
    _plot_combined_heatmap(
        floorplan,
        extent,
        table,
        output_path=combined,
        exclude_rect=exclude_rect,
        image_origin=image_origin,
        heatmap_alpha=heatmap_alpha,
        point_size=point_size,
        show_samples=show_samples,
    )
    paths["combined"] = combined
    paths["csv"] = output_dir / "cfr_similarity_outer_ring_by_ue.csv"
    return paths


def _write_filtered_csv(path: Path, table: dict[str, np.ndarray]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "ue_index",
                "x_m",
                "y_m",
                "z_m",
                "magnitude_similarity",
                "phase_similarity",
                "i_similarity",
                "q_similarity",
            ],
        )
        writer.writeheader()
        for row_index, position in enumerate(table["position_m"]):
            writer.writerow(
                {
                    "ue_index": int(table["ue_index"][row_index]),
                    "x_m": float(position[0]),
                    "y_m": float(position[1]),
                    "z_m": float(position[2]),
                    "magnitude_similarity": float(table["magnitude"][row_index]),
                    "phase_similarity": float(table["phase"][row_index]),
                    "i_similarity": float(table["i"][row_index]),
                    "q_similarity": float(table["q"][row_index]),
                }
            )


def _plot_single_heatmap(
    floorplan: np.ndarray,
    extent: tuple[float, float, float, float],
    table: dict[str, np.ndarray],
    metric: str,
    *,
    output_path: Path,
    exclude_rect: tuple[float, float, float, float],
    image_origin: str,
    heatmap_alpha: float,
    point_size: float,
    show_samples: bool,
) -> None:
    fig, axis = plt.subplots(figsize=(8.0, 8.5))
    image = _draw_heatmap(
        axis,
        floorplan,
        extent,
        table,
        metric,
        exclude_rect=exclude_rect,
        image_origin=image_origin,
        heatmap_alpha=heatmap_alpha,
        point_size=point_size,
        show_samples=show_samples,
    )
    cbar = fig.colorbar(image, ax=axis, shrink=0.82)
    cbar.set_label("similarity")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_combined_heatmap(
    floorplan: np.ndarray,
    extent: tuple[float, float, float, float],
    table: dict[str, np.ndarray],
    *,
    output_path: Path,
    exclude_rect: tuple[float, float, float, float],
    image_origin: str,
    heatmap_alpha: float,
    point_size: float,
    show_samples: bool,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 13.5), squeeze=False)
    image = None
    for axis, metric in zip(axes.ravel(), METRICS, strict=True):
        image = _draw_heatmap(
            axis,
            floorplan,
            extent,
            table,
            metric,
            exclude_rect=exclude_rect,
            image_origin=image_origin,
            heatmap_alpha=heatmap_alpha,
            point_size=point_size,
            show_samples=show_samples,
        )
    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.84)
        cbar.set_label("similarity")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _draw_heatmap(
    axis: plt.Axes,
    floorplan: np.ndarray,
    extent: tuple[float, float, float, float],
    table: dict[str, np.ndarray],
    metric: str,
    *,
    exclude_rect: tuple[float, float, float, float],
    image_origin: str,
    heatmap_alpha: float,
    point_size: float,
    show_samples: bool,
) -> Any:
    positions = table["position_m"]
    x_m = positions[:, 0]
    y_m = positions[:, 1]
    values = table[metric]
    axis.imshow(floorplan, extent=extent, origin=image_origin)
    grid, grid_extent = _rasterize_outer_ring_grid(x_m, y_m, values, exclude_rect)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    image = axis.imshow(
        np.ma.masked_invalid(grid),
        extent=grid_extent,
        origin="lower",
        cmap=cmap,
        interpolation="bilinear",
        alpha=heatmap_alpha,
        vmin=0.0,
        vmax=1.0,
    )
    if show_samples:
        finite = np.isfinite(values)
        axis.scatter(
            x_m[finite],
            y_m[finite],
            c=values[finite],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            s=point_size,
            linewidths=0.0,
            alpha=0.9,
        )
    xmin, xmax, ymin, ymax = exclude_rect
    axis.add_patch(
        Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            edgecolor="#ff4d4f",
            linewidth=2.0,
            linestyle="--",
        )
    )
    axis.set_title(METRIC_TITLES[metric])
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_xlim(extent[0], extent[1])
    axis.set_ylim(extent[2], extent[3])
    axis.set_aspect("equal", adjustable="box")
    return image


def _rasterize_outer_ring_grid(
    x_m: np.ndarray,
    y_m: np.ndarray,
    values: np.ndarray,
    exclude_rect: tuple[float, float, float, float],
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    finite_xy = np.isfinite(x_m) & np.isfinite(y_m)
    finite_values = finite_xy & np.isfinite(values)
    if np.count_nonzero(finite_xy) < 3:
        msg = "Need at least three finite UE positions to render a heatmap."
        raise ValueError(msg)

    x_all = x_m[finite_xy]
    y_all = y_m[finite_xy]
    x_min = float(np.min(x_all))
    x_max = float(np.max(x_all))
    y_min = float(np.min(y_all))
    y_max = float(np.max(y_all))
    dx = base._infer_grid_spacing(x_all)
    dy = base._infer_grid_spacing(y_all)
    x_grid = base._regular_axis(x_min, x_max, dx)
    y_grid = base._regular_axis(y_min, y_max, dy)
    nx = x_grid.size
    ny = y_grid.size

    grid_sum = np.zeros((ny, nx), dtype=np.float64)
    grid_count = np.zeros((ny, nx), dtype=np.int64)
    x = x_m[finite_values]
    y = y_m[finite_values]
    z = np.clip(values[finite_values], 0.0, 1.0)
    ix = np.rint((x - x_min) / dx).astype(np.int64)
    iy = np.rint((y - y_min) / dy).astype(np.int64)
    inside_grid = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    np.add.at(grid_sum, (iy[inside_grid], ix[inside_grid]), z[inside_grid])
    np.add.at(grid_count, (iy[inside_grid], ix[inside_grid]), 1)

    grid = np.zeros((ny, nx), dtype=np.float32)
    sampled = grid_count > 0
    grid[sampled] = (grid_sum[sampled] / grid_count[sampled]).astype(np.float32)

    xmin, xmax, ymin, ymax = exclude_rect
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    excluded = (x_mesh >= xmin) & (x_mesh <= xmax) & (y_mesh >= ymin) & (y_mesh <= ymax)
    grid[excluded] = np.nan
    return grid, (float(x_grid[0]), float(x_grid[-1]), float(y_grid[0]), float(y_grid[-1]))


if __name__ == "__main__":
    raise SystemExit(main())
