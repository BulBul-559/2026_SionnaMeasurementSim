"""Plot UE-centric radio maps from a small set of BS observations.

This helper fixes one or more UE transmitters and visualizes the per-BS
observed value, usually ``/observation/rssi_dbm``, over the floorplan.  It is
the UE-centric counterpart of ``plot_radio_map_heatmaps.py``: the samples are
BS locations instead of UE locations, so the interpolation is only an
illustrative fill from a sparse set of receivers.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sionna_measurement_sim.visualization.radio_map import (  # noqa: E402
    DEFAULT_RSSI_DATASET,
    RadioMapRenderConfig,
    _idw_interpolate,
    _infer_grid_spacing,
    _resolve_floorplan_paths,
    floorplan_extent,
    load_radio_map_table,
    resolve_result_files,
)


@dataclass(frozen=True)
class UERenderItem:
    ue_index: int
    ue_position_m: np.ndarray
    bs_indices: np.ndarray
    bs_positions_m: np.ndarray
    values: np.ndarray
    valid_mask: np.ndarray


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_or_hdf5", type=Path, help="Run directory, manifest, or HDF5 file.")
    parser.add_argument(
        "--ue-indices",
        required=True,
        help="Comma-separated global UE indices, e.g. '0,7,19'.",
    )
    parser.add_argument(
        "--bs-indices",
        default=None,
        help="Optional comma-separated global BS indices to keep, e.g. '0,1,2,3,4,5,6'.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <run>/figures/ue_bs_radio_maps.",
    )
    parser.add_argument("--floorplan-image", type=Path, default=None)
    parser.add_argument("--floorplan-meta", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=["interpolated", "samples", "both"],
        default="both",
    )
    parser.add_argument("--dataset", default=DEFAULT_RSSI_DATASET)
    parser.add_argument("--snapshot-index", type=int, default=0)
    parser.add_argument("--grid-resolution-m", type=float, default=None)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--idw-power", type=float, default=2.0)
    parser.add_argument("--heatmap-alpha", type=float, default=0.68)
    parser.add_argument("--point-size", type=float, default=72.0)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--colormap", default="viridis")
    parser.add_argument("--vmin", type=float, default=None)
    parser.add_argument("--vmax", type=float, default=None)
    parser.add_argument("--show-samples", action="store_true")
    args = parser.parse_args()

    ue_indices = _parse_indices(args.ue_indices)
    summary = generate_ue_bs_radio_maps(
        args.run_or_hdf5,
        ue_indices=ue_indices,
        bs_indices=_parse_indices(args.bs_indices) if args.bs_indices else None,
        output_dir=args.output_dir,
        floorplan_image=args.floorplan_image,
        floorplan_meta=args.floorplan_meta,
        config=RadioMapRenderConfig(
            render_mode=args.mode,
            snapshot_index=args.snapshot_index,
            value_dataset=args.dataset,
            grid_resolution_m=args.grid_resolution_m,
            interpolation_neighbors=args.neighbors,
            interpolation_power=args.idw_power,
            heatmap_alpha=args.heatmap_alpha,
            point_size=args.point_size,
            dpi=args.dpi,
            colormap=args.colormap,
            show_samples_on_interpolated=args.show_samples,
        ),
        vmin=args.vmin,
        vmax=args.vmax,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def generate_ue_bs_radio_maps(
    run_or_hdf5: str | Path,
    *,
    ue_indices: list[int],
    bs_indices: list[int] | None = None,
    output_dir: str | Path | None = None,
    floorplan_image: str | Path | None = None,
    floorplan_meta: str | Path | None = None,
    config: RadioMapRenderConfig | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> dict[str, Any]:
    cfg = config or RadioMapRenderConfig(render_mode="both")
    root = Path(run_or_hdf5)
    files, manifest = resolve_result_files(root)
    if not files:
        raise FileNotFoundError(f"No HDF5 result files found for {root}")

    output_path = Path(output_dir) if output_dir is not None else _default_output_dir(root)
    output_path.mkdir(parents=True, exist_ok=True)
    floorplan_paths = _resolve_floorplan_paths(
        root,
        manifest,
        files=files,
        floorplan_image=Path(floorplan_image) if floorplan_image is not None else None,
        floorplan_meta=Path(floorplan_meta) if floorplan_meta is not None else None,
    )
    floorplan = np.asarray(mpimg.imread(floorplan_paths["image"]))
    meta = json.loads(floorplan_paths["meta"].read_text(encoding="utf-8"))
    extent = floorplan_extent(meta, floorplan)
    table = load_radio_map_table(files, cfg)
    bs_mask = _bs_mask(table.bs_indices, bs_indices)
    items = [_select_ue(table, index, bs_mask=bs_mask) for index in ue_indices]
    finite_values = np.concatenate(
        [item.values[item.valid_mask & np.isfinite(item.values)] for item in items]
    )
    if finite_values.size == 0:
        raise ValueError("No finite UE-to-BS values found for the requested UE indices.")
    plot_vmin = float(np.nanmin(finite_values) if vmin is None else vmin)
    plot_vmax = float(np.nanmax(finite_values) if vmax is None else vmax)
    if np.isclose(plot_vmin, plot_vmax):
        plot_vmax = plot_vmin + 1.0

    generated: list[Path] = []
    modes = ("interpolated", "samples") if cfg.render_mode == "both" else (cfg.render_mode,)
    for item in items:
        for mode in modes:
            output = output_path / f"ue_bs_radio_map_ue_{item.ue_index:05d}_{mode}.png"
            _plot_one_ue_map(
                floorplan,
                extent,
                item,
                output_path=output,
                mode=mode,
                vmin=plot_vmin,
                vmax=plot_vmax,
                config=cfg,
            )
            generated.append(output)

    csv_path = output_path / "ue_bs_radio_map_values.csv"
    _write_csv(csv_path, items)
    summary_path = output_path / "ue_bs_radio_map_summary.json"
    summary = {
        "input": root.as_posix(),
        "output_dir": output_path.as_posix(),
        "floorplan_image": floorplan_paths["image"].as_posix(),
        "floorplan_meta": floorplan_paths["meta"].as_posix(),
        "value_dataset": cfg.value_dataset,
        "snapshot_index": cfg.snapshot_index,
        "render_mode": cfg.render_mode,
        "ue_indices": [int(index) for index in ue_indices],
        "bs_indices": [int(index) for index in table.bs_indices[bs_mask]],
        "bs_count": int(np.count_nonzero(bs_mask)),
        "vmin": plot_vmin,
        "vmax": plot_vmax,
        "generated_files": [path.as_posix() for path in generated],
        "csv": csv_path.as_posix(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["summary"] = summary_path.as_posix()
    return summary


def _select_ue(table: Any, ue_index: int, *, bs_mask: np.ndarray) -> UERenderItem:
    matches = np.flatnonzero(table.ue_indices == int(ue_index))
    if matches.size == 0:
        available = ", ".join(str(int(v)) for v in table.ue_indices[:20])
        raise ValueError(f"UE index {ue_index} not found. First available UE indices: {available}")
    row = int(matches[0])
    return UERenderItem(
        ue_index=int(ue_index),
        ue_position_m=table.positions_m[row],
        bs_indices=table.bs_indices[bs_mask],
        bs_positions_m=table.bs_positions_m[bs_mask],
        values=table.rss_dbm[row, bs_mask],
        valid_mask=table.valid_mask[row, bs_mask] & np.isfinite(table.rss_dbm[row, bs_mask]),
    )


def _bs_mask(all_indices: np.ndarray, selected: list[int] | None) -> np.ndarray:
    if selected is None:
        return np.ones_like(all_indices, dtype=np.bool_)
    selected_set = {int(index) for index in selected}
    mask = np.asarray([int(index) in selected_set for index in all_indices], dtype=np.bool_)
    missing = sorted(selected_set - {int(index) for index in all_indices.tolist()})
    if missing:
        raise ValueError(f"Requested BS indices not found: {missing}")
    if not np.any(mask):
        raise ValueError("--bs-indices removed all BS observations")
    return mask


def _plot_one_ue_map(
    floorplan: np.ndarray,
    extent_m: tuple[float, float, float, float],
    item: UERenderItem,
    *,
    output_path: Path,
    mode: str,
    vmin: float,
    vmax: float,
    config: RadioMapRenderConfig,
) -> None:
    valid = item.valid_mask & np.all(np.isfinite(item.bs_positions_m[:, :2]), axis=1)
    if not np.any(valid):
        raise ValueError(f"UE {item.ue_index} has no finite BS observations.")
    bs_x = item.bs_positions_m[valid, 0].astype(np.float64, copy=False)
    bs_y = item.bs_positions_m[valid, 1].astype(np.float64, copy=False)
    values = item.values[valid].astype(np.float64, copy=False)

    fig, axis = plt.subplots(figsize=(8.0, 8.5))
    axis.imshow(floorplan, extent=extent_m, origin=config.image_origin)
    if mode == "interpolated":
        grid = _interpolate_over_extent(
            bs_x,
            bs_y,
            values,
            extent_m=extent_m,
            resolution_m=config.grid_resolution_m,
            neighbors=config.interpolation_neighbors,
            power=config.interpolation_power,
        )
        mappable = axis.imshow(
            grid,
            extent=extent_m,
            origin="lower",
            cmap=config.colormap,
            interpolation="bilinear",
            alpha=config.heatmap_alpha,
            vmin=vmin,
            vmax=vmax,
        )
        if config.show_samples_on_interpolated:
            axis.scatter(
                bs_x,
                bs_y,
                c=values,
                cmap=config.colormap,
                vmin=vmin,
                vmax=vmax,
                s=max(config.point_size * 0.55, 10.0),
                edgecolors="black",
                linewidths=0.35,
                alpha=0.9,
                zorder=5,
            )
    elif mode == "samples":
        mappable = axis.scatter(
            bs_x,
            bs_y,
            c=values,
            cmap=config.colormap,
            vmin=vmin,
            vmax=vmax,
            s=config.point_size,
            edgecolors="black",
            linewidths=0.45,
            alpha=0.95,
            zorder=5,
        )
    else:
        raise ValueError(f"Unsupported render mode: {mode!r}")

    _draw_bs_labels(axis, item.bs_indices[valid], bs_x, bs_y)
    _draw_ue_marker(axis, item)
    cbar = fig.colorbar(mappable, ax=axis, shrink=0.82)
    cbar.set_label(_value_label(config.value_dataset))
    axis.set_title(f"UE {item.ue_index} Observation Map")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_xlim(extent_m[0], extent_m[1])
    axis.set_ylim(extent_m[2], extent_m[3])
    axis.set_aspect("equal", adjustable="box")
    fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)


def _interpolate_over_extent(
    x_m: np.ndarray,
    y_m: np.ndarray,
    values: np.ndarray,
    *,
    extent_m: tuple[float, float, float, float],
    resolution_m: float | None,
    neighbors: int,
    power: float,
) -> np.ndarray:
    if x_m.size < 1:
        raise ValueError("Need at least one BS observation.")
    if resolution_m is None:
        spacing = min(_infer_grid_spacing(x_m), _infer_grid_spacing(y_m))
        if not np.isfinite(spacing) or spacing <= 0.0:
            width = max(extent_m[1] - extent_m[0], 1.0)
            height = max(extent_m[3] - extent_m[2], 1.0)
            spacing = max(min(width, height) / 160.0, 0.05)
        spacing = min(max(float(spacing), 0.05), 0.25)
    else:
        spacing = float(resolution_m)
    x_axis = np.arange(extent_m[0], extent_m[1] + spacing * 0.5, spacing, dtype=np.float64)
    y_axis = np.arange(extent_m[2], extent_m[3] + spacing * 0.5, spacing, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(x_axis, y_axis)
    points = np.column_stack([x_m, y_m]).astype(np.float64, copy=False)
    targets = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    grid = _idw_interpolate(
        points,
        values.astype(np.float64, copy=False),
        targets,
        neighbors=min(int(neighbors), points.shape[0]),
        power=float(power),
    )
    return grid.reshape(grid_y.shape).astype(np.float32, copy=False)


def _draw_bs_labels(axis: Any, indices: np.ndarray, x_m: np.ndarray, y_m: np.ndarray) -> None:
    for index, x, y in zip(indices, x_m, y_m, strict=True):
        axis.annotate(
            f"BS {int(index)}",
            xy=(float(x), float(y)),
            xytext=(5, 5),
            textcoords="offset points",
            color="white",
            fontsize=7,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "black",
                "edgecolor": "none",
                "alpha": 0.62,
            },
            zorder=7,
        )


def _draw_ue_marker(axis: Any, item: UERenderItem) -> None:
    if not np.all(np.isfinite(item.ue_position_m[:2])):
        return
    x = float(item.ue_position_m[0])
    y = float(item.ue_position_m[1])
    axis.scatter(
        [x],
        [y],
        marker="*",
        s=260.0,
        c="#ff3b30",
        edgecolors="black",
        linewidths=1.0,
        zorder=8,
    )
    axis.annotate(
        f"UE {item.ue_index}",
        xy=(x, y),
        xytext=(8, 8),
        textcoords="offset points",
        color="white",
        fontsize=9,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "black",
            "edgecolor": "none",
            "alpha": 0.7,
        },
        zorder=9,
    )


def _write_csv(path: Path, items: list[UERenderItem]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = [
            "ue_index",
            "ue_x_m",
            "ue_y_m",
            "ue_z_m",
            "bs_index",
            "bs_x_m",
            "bs_y_m",
            "bs_z_m",
            "value",
            "valid",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            for col, bs_index in enumerate(item.bs_indices):
                writer.writerow(
                    {
                        "ue_index": int(item.ue_index),
                        "ue_x_m": float(item.ue_position_m[0]),
                        "ue_y_m": float(item.ue_position_m[1]),
                        "ue_z_m": float(item.ue_position_m[2]),
                        "bs_index": int(bs_index),
                        "bs_x_m": float(item.bs_positions_m[col, 0]),
                        "bs_y_m": float(item.bs_positions_m[col, 1]),
                        "bs_z_m": float(item.bs_positions_m[col, 2]),
                        "value": float(item.values[col]) if item.valid_mask[col] else float("nan"),
                        "valid": bool(item.valid_mask[col]),
                    }
                )


def _default_output_dir(root: Path) -> Path:
    if root.is_file() and root.suffix == ".h5":
        return root.parent / "figures" / "ue_bs_radio_maps"
    return root / "figures" / "ue_bs_radio_maps"


def _parse_indices(value: str) -> list[int]:
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not indices:
        raise argparse.ArgumentTypeError("--ue-indices must contain at least one integer")
    return indices


def _value_label(dataset: str) -> str:
    normalized = dataset.strip("/")
    if normalized.endswith("rssi_dbm"):
        return "RSSI [dBm]"
    if normalized.endswith("noise_power_dbm"):
        return "Noise power [dBm]"
    if normalized.endswith("snr_db"):
        return "SNR [dB]"
    return normalized


if __name__ == "__main__":
    raise SystemExit(main())
