"""Compare CFR similarity by matching UE positions instead of shard indices.

This is intended for cross-density runs where one side is a subset of the
other. Only positions present on both sides are compared; unmatched positions
are excluded from statistics and rendered as transparent heatmap cells.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--dataset", default=base.DEFAULT_DATASET)
    parser.add_argument("--snapshot-index", type=int, default=0)
    parser.add_argument("--floorplan-image", type=Path, required=True)
    parser.add_argument("--floorplan-meta", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--position-decimals", type=int, default=3)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--image-origin", choices=("upper", "lower"), default="upper")
    parser.add_argument("--heatmap-alpha", type=float, default=0.68)
    parser.add_argument("--show-samples", action="store_true")
    parser.add_argument("--point-size", type=float, default=5.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    left_files = _resolve_h5_files(args.left)
    right_files = _resolve_h5_files(args.right)
    left_count, left_bs = _count_positions_and_bs(left_files)
    right_index, right_count, right_bs = _build_position_index(
        right_files, decimals=args.position_decimals
    )
    table = _compute_all_file_matches(
        left_files,
        right_index,
        left_bs,
        right_bs,
        dataset=args.dataset,
        snapshot_index=args.snapshot_index,
        decimals=args.position_decimals,
        workers=max(1, args.workers),
    )

    csv_path = args.output_dir / "cfr_similarity_by_position.csv"
    stats_path = args.output_dir / "cfr_similarity_stats.csv"
    summary_path = args.output_dir / "summary.json"
    _write_csv(csv_path, table)
    _write_stats_csv(stats_path, table)
    figures = _plot_heatmaps(args, table)
    histograms = _plot_histograms(args.output_dir, table)
    summary = {
        "left": args.left.as_posix(),
        "right": args.right.as_posix(),
        "dataset": args.dataset,
        "snapshot_index": args.snapshot_index,
        "position_decimals": args.position_decimals,
        "left_position_count": int(left_count),
        "right_position_count": int(right_count),
        "matched_position_count": int(table["ue_index"].shape[0]),
        "unmatched_left_count": int(left_count - table["ue_index"].shape[0]),
        "unmatched_right_count": int(right_count - table["ue_index"].shape[0]),
        "left_file_count": len(left_files),
        "right_file_count": len(right_files),
        "bs_indices": [int(value) for value in left_bs],
        "rendering": {
            "heatmap_grid": "rectangular matched-position x/y grid",
            "missing_grid_cell_value": "transparent NaN",
            "interpolation": "bilinear over finite matched cells only",
        },
        "metrics": {metric: _metric_summary(table[metric]) for metric in METRICS},
        "outputs": {
            "csv": csv_path.as_posix(),
            "stats": stats_path.as_posix(),
            **{key: value.as_posix() for key, value in {**figures, **histograms}.items()},
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {"output_dir": args.output_dir.as_posix(), "summary": summary_path.as_posix()}
        )
    )
    return 0


def _count_positions_and_bs(files: list[Path]) -> tuple[int, np.ndarray]:
    count = 0
    bs_indices: np.ndarray | None = None
    for file in files:
        metadata = base._load_ue_metadata(file)
        current_bs = np.asarray(metadata["bs_indices"], dtype=np.int64)
        if bs_indices is None:
            bs_indices = current_bs
        elif not np.array_equal(current_bs, bs_indices):
            msg = f"BS index order differs in {file}"
            raise ValueError(msg)
        count += int(np.asarray(metadata["positions_m"]).shape[0])
    if bs_indices is None:
        msg = "No metadata files found."
        raise ValueError(msg)
    return count, bs_indices


def _build_position_index(
    files: list[Path],
    *,
    decimals: int,
) -> tuple[dict[tuple[float, float, float], tuple[str, int]], int, np.ndarray]:
    index: dict[tuple[float, float, float], tuple[str, int]] = {}
    count = 0
    bs_indices: np.ndarray | None = None
    for file in files:
        metadata = base._load_ue_metadata(file)
        current_bs = np.asarray(metadata["bs_indices"], dtype=np.int64)
        if bs_indices is None:
            bs_indices = current_bs
        elif not np.array_equal(current_bs, bs_indices):
            msg = f"BS index order differs in {file}"
            raise ValueError(msg)
        positions = np.asarray(metadata["positions_m"], dtype=np.float32)
        for local_index, key in enumerate(_position_keys(positions, decimals)):
            index.setdefault(key, (file.as_posix(), local_index))
        count += int(positions.shape[0])
    if bs_indices is None:
        msg = "No metadata files found."
        raise ValueError(msg)
    return index, count, bs_indices


def _compute_all_file_matches(
    left_files: list[Path],
    right_index: dict[tuple[float, float, float], tuple[str, int]],
    left_bs: np.ndarray,
    right_bs: np.ndarray,
    *,
    dataset: str,
    snapshot_index: int,
    decimals: int,
    workers: int,
) -> dict[str, np.ndarray]:
    if workers == 1:
        parts = [
            _compute_file_matches(
                file,
                right_index,
                left_bs,
                right_bs,
                dataset,
                snapshot_index,
                decimals,
            )
            for file in left_files
        ]
    else:
        parts = []
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _compute_file_matches,
                    file,
                    right_index,
                    left_bs,
                    right_bs,
                    dataset,
                    snapshot_index,
                    decimals,
                )
                for file in left_files
            ]
            for future in as_completed(futures):
                parts.append(future.result())
    return _merge_match_tables(parts)


def _compute_file_matches(
    left_file: Path,
    right_index: dict[tuple[float, float, float], tuple[str, int]],
    left_bs: np.ndarray,
    right_bs: np.ndarray,
    dataset: str,
    snapshot_index: int,
    decimals: int,
) -> dict[str, np.ndarray]:
    left = base._load_ue_major_cfr(left_file, dataset, snapshot_index)
    positions = np.asarray(left["positions_m"], dtype=np.float32)
    grouped: dict[str, list[tuple[int, int]]] = {}
    for left_row, key in enumerate(_position_keys(positions, decimals)):
        match = right_index.get(key)
        if match is None:
            continue
        right_file, right_row = match
        grouped.setdefault(right_file, []).append((left_row, right_row))

    if not grouped:
        return _empty_match_table()

    order = _right_bs_order(left_bs, right_bs)
    rows = []
    for right_file, pairs in grouped.items():
        right = base._load_ue_major_cfr(Path(right_file), dataset, snapshot_index)
        left_rows = np.asarray([pair[0] for pair in pairs], dtype=np.int64)
        right_rows = np.asarray([pair[1] for pair in pairs], dtype=np.int64)
        right_cfr = np.asarray(right["cfr"], dtype=np.complex64)[:, order]
        right_valid = np.asarray(right["valid"], dtype=np.bool_)[:, order]
        rows.append(
            _compute_similarity_table(
                {
                    "ue_index": np.asarray(left["ue_indices"], dtype=np.int64)[left_rows],
                    "position_m": positions[left_rows],
                    "left_cfr": np.asarray(left["cfr"], dtype=np.complex64)[left_rows],
                    "right_cfr": right_cfr[right_rows],
                    "left_valid": np.asarray(left["valid"], dtype=np.bool_)[left_rows],
                    "right_valid": right_valid[right_rows],
                }
            )
        )
    return _merge_match_tables(rows)


def _right_bs_order(left_bs: np.ndarray, right_bs: np.ndarray) -> np.ndarray:
    left_bs = np.asarray(left_bs, dtype=np.int64)
    right_bs = np.asarray(right_bs, dtype=np.int64)
    if np.array_equal(left_bs, right_bs):
        return np.arange(right_bs.shape[0], dtype=np.int64)
    if set(left_bs.tolist()) != set(right_bs.tolist()):
        msg = f"BS sets differ: left={left_bs.tolist()} right={right_bs.tolist()}"
        raise ValueError(msg)
    return np.asarray([int(np.flatnonzero(right_bs == bs)[0]) for bs in left_bs], dtype=np.int64)


def _empty_match_table() -> dict[str, np.ndarray]:
    return {
        "ue_index": np.zeros((0,), dtype=np.int64),
        "position_m": np.zeros((0, 3), dtype=np.float32),
        "magnitude": np.zeros((0,), dtype=np.float32),
        "phase": np.zeros((0,), dtype=np.float32),
        "i": np.zeros((0,), dtype=np.float32),
        "q": np.zeros((0,), dtype=np.float32),
    }


def _merge_match_tables(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    nonempty = [part for part in parts if part["ue_index"].size > 0]
    if not nonempty:
        msg = "No common UE positions found between left and right runs."
        raise ValueError(msg)
    return {
        "ue_index": np.concatenate([part["ue_index"] for part in nonempty]),
        "position_m": np.concatenate([part["position_m"] for part in nonempty], axis=0),
        "magnitude": np.concatenate([part["magnitude"] for part in nonempty]),
        "phase": np.concatenate([part["phase"] for part in nonempty]),
        "i": np.concatenate([part["i"] for part in nonempty]),
        "q": np.concatenate([part["q"] for part in nonempty]),
    }


def _resolve_h5_files(path: Path) -> list[Path]:
    path = path.expanduser()
    if path.is_file():
        return [path]

    manifest = path / "manifest.json"
    nested_manifest = path / "manifest" / "manifest.json"
    for candidate in (manifest, nested_manifest):
        if candidate.exists():
            data = json.loads(candidate.read_text(encoding="utf-8"))
            results = data.get("results") or []
            files = []
            for item in results:
                raw = Path(item["result_h5"])
                files.append(raw if raw.is_absolute() else (path / raw))
            if files:
                return sorted(files, key=_h5_sort_key)
            result_h5 = data.get("results_h5")
            if result_h5:
                raw = Path(result_h5)
                return [raw if raw.is_absolute() else (path / raw)]

    search_root = path / "results" if (path / "results").is_dir() else path
    files = sorted(search_root.glob("result*.h5"), key=_h5_sort_key)
    if not files:
        files = sorted(search_root.glob("results.h5"))
    if not files:
        msg = f"No HDF5 result files found under {path}"
        raise FileNotFoundError(msg)
    return files


def _h5_sort_key(path: Path) -> tuple[int, ...]:
    numbers = tuple(int(value) for value in re.findall(r"\d+", path.name))
    return numbers or (0,)


def _load_position_table(path: Path, dataset: str, snapshot_index: int) -> dict[str, Any]:
    files = _resolve_h5_files(path)
    chunks = []
    for file in files:
        item = base._load_ue_major_cfr(file, dataset, snapshot_index)
        chunks.append(item)
    bs_indices = np.asarray(chunks[0]["bs_indices"], dtype=np.int64)
    for file, item in zip(files, chunks, strict=True):
        if not np.array_equal(np.asarray(item["bs_indices"], dtype=np.int64), bs_indices):
            msg = f"BS index order differs in {file}"
            raise ValueError(msg)
    return {
        "files": files,
        "ue_index": np.concatenate(
            [np.asarray(item["ue_indices"], dtype=np.int64) for item in chunks]
        ),
        "position_m": np.concatenate(
            [np.asarray(item["positions_m"], dtype=np.float32) for item in chunks]
        ),
        "cfr": np.concatenate(
            [np.asarray(item["cfr"], dtype=np.complex64) for item in chunks], axis=0
        ),
        "valid": np.concatenate(
            [np.asarray(item["valid"], dtype=np.bool_) for item in chunks], axis=0
        ),
        "bs_indices": bs_indices,
    }


def _align_bs_order(
    left: dict[str, Any],
    right: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    left_bs = np.asarray(left["bs_indices"], dtype=np.int64)
    right_bs = np.asarray(right["bs_indices"], dtype=np.int64)
    if np.array_equal(left_bs, right_bs):
        return left, right
    if set(left_bs.tolist()) != set(right_bs.tolist()):
        msg = f"BS sets differ: left={left_bs.tolist()} right={right_bs.tolist()}"
        raise ValueError(msg)
    order = np.asarray([int(np.flatnonzero(right_bs == bs)[0]) for bs in left_bs], dtype=np.int64)
    right = dict(right)
    right["cfr"] = right["cfr"][:, order]
    right["valid"] = right["valid"][:, order]
    right["bs_indices"] = left_bs
    return left, right


def _match_by_position(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    decimals: int,
) -> dict[str, np.ndarray]:
    right_keys = _position_keys(right["position_m"], decimals)
    right_by_key: dict[tuple[float, float, float], int] = {}
    for index, key in enumerate(right_keys):
        right_by_key.setdefault(key, index)

    left_indices: list[int] = []
    right_indices: list[int] = []
    for index, key in enumerate(_position_keys(left["position_m"], decimals)):
        right_index = right_by_key.get(key)
        if right_index is None:
            continue
        left_indices.append(index)
        right_indices.append(right_index)

    if not left_indices:
        msg = "No common UE positions found between left and right runs."
        raise ValueError(msg)
    left_idx = np.asarray(left_indices, dtype=np.int64)
    right_idx = np.asarray(right_indices, dtype=np.int64)
    return {
        "ue_index": left["ue_index"][left_idx],
        "position_m": left["position_m"][left_idx],
        "left_cfr": left["cfr"][left_idx],
        "right_cfr": right["cfr"][right_idx],
        "left_valid": left["valid"][left_idx],
        "right_valid": right["valid"][right_idx],
    }


def _position_keys(
    positions: np.ndarray,
    decimals: int,
) -> list[tuple[float, float, float]]:
    rounded = np.round(np.asarray(positions, dtype=np.float64), decimals=decimals)
    return [tuple(float(value) for value in row) for row in rounded]


def _compute_similarity_table(matched: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    left_cfr = matched["left_cfr"]
    right_cfr = matched["right_cfr"]
    valid = matched["left_valid"] & matched["right_valid"]
    sample_mask = base._finite_complex_mask(left_cfr, right_cfr, valid)
    mag_left = np.abs(left_cfr).astype(np.float32, copy=False)
    mag_right = np.abs(right_cfr).astype(np.float32, copy=False)
    return {
        "ue_index": matched["ue_index"],
        "position_m": matched["position_m"],
        "magnitude": base._normalized_l2_similarity(mag_left, mag_right, sample_mask),
        "phase": base._weighted_phase_similarity(
            left_cfr, right_cfr, mag_left, mag_right, sample_mask
        ),
        "i": base._normalized_l2_similarity(left_cfr.real, right_cfr.real, sample_mask),
        "q": base._normalized_l2_similarity(left_cfr.imag, right_cfr.imag, sample_mask),
    }


def _write_csv(path: Path, table: dict[str, np.ndarray]) -> None:
    positions = table["position_m"]
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
        for row, position in enumerate(positions):
            writer.writerow(
                {
                    "ue_index": int(table["ue_index"][row]),
                    "x_m": float(position[0]),
                    "y_m": float(position[1]),
                    "z_m": float(position[2]),
                    "magnitude_similarity": float(table["magnitude"][row]),
                    "phase_similarity": float(table["phase"][row]),
                    "i_similarity": float(table["i"][row]),
                    "q_similarity": float(table["q"][row]),
                }
            )


def _write_stats_csv(path: Path, table: dict[str, np.ndarray]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["metric", "count", "min", "max", "mean"])
        writer.writeheader()
        for metric in METRICS:
            writer.writerow({"metric": metric, **_metric_summary(table[metric])})


def _metric_summary(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0, "min": float("nan"), "max": float("nan"), "mean": float("nan")}
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def _plot_heatmaps(args: argparse.Namespace, table: dict[str, np.ndarray]) -> dict[str, Path]:
    floorplan = np.asarray(Image.open(args.floorplan_image).convert("RGB"))
    meta = json.loads(args.floorplan_meta.read_text(encoding="utf-8"))
    extent = base._floorplan_extent(meta, floorplan)
    paths: dict[str, Path] = {}
    for metric in METRICS:
        output_path = args.output_dir / f"cfr_similarity_{metric}_floorplan.png"
        _plot_single_heatmap(
            floorplan,
            extent,
            table,
            metric,
            output_path=output_path,
            image_origin=args.image_origin,
            heatmap_alpha=args.heatmap_alpha,
            point_size=args.point_size,
            show_samples=args.show_samples,
        )
        paths[metric] = output_path
    combined = args.output_dir / "cfr_similarity_four_panel_floorplan.png"
    _plot_combined_heatmap(
        floorplan,
        extent,
        table,
        output_path=combined,
        image_origin=args.image_origin,
        heatmap_alpha=args.heatmap_alpha,
        point_size=args.point_size,
        show_samples=args.show_samples,
    )
    paths["combined"] = combined
    return paths


def _plot_single_heatmap(
    floorplan: np.ndarray,
    extent: tuple[float, float, float, float],
    table: dict[str, np.ndarray],
    metric: str,
    *,
    output_path: Path,
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
    grid, grid_extent = _rasterize_nan_grid(x_m, y_m, values)
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
    axis.set_title(METRIC_TITLES[metric])
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_xlim(extent[0], extent[1])
    axis.set_ylim(extent[2], extent[3])
    axis.set_aspect("equal", adjustable="box")
    return image


def _rasterize_nan_grid(
    x_m: np.ndarray,
    y_m: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    finite_xy = np.isfinite(x_m) & np.isfinite(y_m)
    finite_values = finite_xy & np.isfinite(values)
    if np.count_nonzero(finite_values) < 3:
        msg = "Need at least three matched finite samples to render a heatmap."
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
    grid_sum = np.zeros((y_grid.size, x_grid.size), dtype=np.float64)
    grid_count = np.zeros((y_grid.size, x_grid.size), dtype=np.int64)

    x = x_m[finite_values]
    y = y_m[finite_values]
    z = np.clip(values[finite_values], 0.0, 1.0)
    ix = np.rint((x - x_min) / dx).astype(np.int64)
    iy = np.rint((y - y_min) / dy).astype(np.int64)
    inside = (ix >= 0) & (ix < x_grid.size) & (iy >= 0) & (iy < y_grid.size)
    np.add.at(grid_sum, (iy[inside], ix[inside]), z[inside])
    np.add.at(grid_count, (iy[inside], ix[inside]), 1)
    grid = np.full((y_grid.size, x_grid.size), np.nan, dtype=np.float32)
    sampled = grid_count > 0
    grid[sampled] = (grid_sum[sampled] / grid_count[sampled]).astype(np.float32)
    return grid, (float(x_grid[0]), float(x_grid[-1]), float(y_grid[0]), float(y_grid[-1]))


def _plot_histograms(output_dir: Path, table: dict[str, np.ndarray]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for metric in METRICS:
        output_path = output_dir / f"cfr_similarity_{metric}_histogram.png"
        _plot_single_histogram(table[metric], title=METRIC_TITLES[metric], output_path=output_path)
        paths[f"{metric}_histogram"] = output_path
    combined = output_dir / "cfr_similarity_histograms.png"
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), squeeze=False)
    for axis, metric in zip(axes.ravel(), METRICS, strict=True):
        _draw_histogram(axis, table[metric], title=METRIC_TITLES[metric])
    fig.savefig(combined, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths["histograms"] = combined
    return paths


def _plot_single_histogram(values: np.ndarray, *, title: str, output_path: Path) -> None:
    fig, axis = plt.subplots(figsize=(7.0, 4.5))
    _draw_histogram(axis, values, title=title)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _draw_histogram(axis: plt.Axes, values: np.ndarray, *, title: str) -> None:
    finite = values[np.isfinite(values)]
    axis.hist(finite, bins=np.linspace(0.0, 1.0, 51), color="#3b82a0", edgecolor="white")
    axis.set_title(title)
    axis.set_xlabel("similarity")
    axis.set_ylabel("matched UE count")
    axis.set_xlim(0.0, 1.0)
    axis.grid(axis="y", alpha=0.25)


if __name__ == "__main__":
    raise SystemExit(main())
