"""Floorplan radio-map heatmaps from per-link RSS observations."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

RADIO_MAP_RENDER_MODES = ("interpolated", "samples", "both")
DEFAULT_RSSI_DATASET = "/observation/rssi_dbm"


@dataclass(frozen=True)
class RadioMapRenderConfig:
    """Rendering options for BS-wise RSS radio maps."""

    render_mode: str = "interpolated"
    snapshot_index: int = 0
    value_dataset: str = DEFAULT_RSSI_DATASET
    grid_resolution_m: float | None = None
    interpolation_neighbors: int = 8
    interpolation_power: float = 2.0
    image_origin: str = "upper"
    heatmap_alpha: float = 0.68
    point_size: float = 16.0
    dpi: int = 180
    colormap: str = "viridis"
    show_samples_on_interpolated: bool = False

    def __post_init__(self) -> None:
        if self.render_mode not in RADIO_MAP_RENDER_MODES:
            raise ValueError(
                "radio_map render_mode must be one of "
                f"{RADIO_MAP_RENDER_MODES}, got {self.render_mode!r}"
            )
        if self.snapshot_index < 0:
            raise ValueError("snapshot_index must be non-negative")
        if self.grid_resolution_m is not None and self.grid_resolution_m <= 0.0:
            raise ValueError("grid_resolution_m must be positive when set")
        if self.interpolation_neighbors < 1:
            raise ValueError("interpolation_neighbors must be >= 1")
        if self.interpolation_power <= 0.0:
            raise ValueError("interpolation_power must be positive")
        if self.image_origin not in ("upper", "lower"):
            raise ValueError("image_origin must be upper/lower")
        if self.dpi < 50:
            raise ValueError("dpi must be >= 50")


@dataclass(frozen=True)
class RadioMapTable:
    """UE-position RSS table with one column per BS."""

    ue_indices: np.ndarray
    bs_indices: np.ndarray
    positions_m: np.ndarray
    bs_positions_m: np.ndarray
    rss_dbm: np.ndarray
    valid_mask: np.ndarray


def generate_radio_map_heatmaps(
    run_or_hdf5: str | Path,
    output_dir: str | Path | None = None,
    *,
    floorplan_image: str | Path | None = None,
    floorplan_meta: str | Path | None = None,
    config: RadioMapRenderConfig | None = None,
) -> dict[str, Any]:
    """Generate one RSS radio-map image per BS.

    ``run_or_hdf5`` may be a single HDF5 file, a sharded run directory with
    ``manifest/manifest.json``, a manifest file, or a directory containing
    ``result_*.h5`` files.
    """

    cfg = config or RadioMapRenderConfig()
    root = Path(run_or_hdf5)
    files, manifest = resolve_result_files(root)
    if not files:
        msg = f"No HDF5 result files found for {root}"
        raise FileNotFoundError(msg)
    output_path = (
        Path(output_dir)
        if output_dir is not None
        else _default_output_dir(root, manifest)
    )
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
    generated = _plot_radio_maps(
        table,
        floorplan,
        extent,
        output_path,
        cfg,
    )
    csv_path = output_path / "radio_map_values.csv"
    summary_path = output_path / "radio_map_summary.json"
    _write_radio_map_csv(csv_path, table)
    summary = {
        "input": Path(run_or_hdf5).as_posix(),
        "output_dir": output_path.as_posix(),
        "floorplan_image": floorplan_paths["image"].as_posix(),
        "floorplan_meta": floorplan_paths["meta"].as_posix(),
        "value_dataset": cfg.value_dataset,
        "snapshot_index": cfg.snapshot_index,
        "render_mode": cfg.render_mode,
        "ue_count": int(table.ue_indices.size),
        "bs_count": int(table.bs_indices.size),
        "bs_indices": [int(value) for value in table.bs_indices],
        "rss_dbm": _summary_stats(table.rss_dbm[table.valid_mask]),
        "generated_files": [path.as_posix() for path in generated],
        "csv": csv_path.as_posix(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    summary["summary"] = summary_path.as_posix()
    return summary


def resolve_result_files(root: Path) -> tuple[list[Path], dict[str, Any]]:
    """Resolve HDF5 result files and optional aggregate manifest."""

    path = root.expanduser()
    if path.is_file() and path.suffix == ".h5":
        return [path], {}
    manifest_path = path if path.is_file() and path.name == "manifest.json" else None
    if manifest_path is None:
        direct_manifest = path / "manifest.json"
        nested_manifest = path / "manifest" / "manifest.json"
        if direct_manifest.exists():
            manifest_path = direct_manifest
        elif nested_manifest.exists():
            manifest_path = nested_manifest
    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_dir = (
            manifest_path.parent.parent
            if manifest_path.parent.name == "manifest"
            else manifest_path.parent
        )
        files = []
        for item in manifest.get("results", []):
            files.append(_resolve_manifest_path(run_dir, Path(item["result_h5"])))
        result_h5 = manifest.get("results_h5")
        if result_h5:
            files.append(_resolve_manifest_path(run_dir, Path(result_h5)))
        return sorted(files, key=_h5_sort_key), manifest

    search_root = path / "results" if (path / "results").is_dir() else path
    files = sorted(search_root.glob("result*.h5"), key=_h5_sort_key)
    if not files:
        files = sorted(search_root.glob("results.h5"), key=_h5_sort_key)
    return files, {}


def load_radio_map_table(files: list[Path], config: RadioMapRenderConfig) -> RadioMapTable:
    """Load all UE positions and per-BS RSS values from HDF5 shards."""

    chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    bs_reference: np.ndarray | None = None
    bs_positions_reference: np.ndarray | None = None
    for file in files:
        with h5py.File(file, "r") as h5:
            tx_role, rx_role = _link_roles(h5)
            bs_indices = _role_global_indices(h5, "bs", tx_role=tx_role, rx_role=rx_role)
            bs_positions = _role_positions(h5, "bs", tx_role=tx_role, rx_role=rx_role)
            ue_indices = _role_global_indices(h5, "ue", tx_role=tx_role, rx_role=rx_role)
            ue_positions = _role_positions(h5, "ue", tx_role=tx_role, rx_role=rx_role)
            rss = _read_rss_by_ue_bs(h5, config.value_dataset, config.snapshot_index)
            valid = _read_valid_by_ue_bs(h5, rss.shape, tx_role=tx_role, rx_role=rx_role)
            if tx_role == "bs" and rx_role == "ue":
                rss = np.transpose(rss, (1, 0))
                valid = np.transpose(valid, (1, 0))
            if bs_reference is None:
                bs_reference = bs_indices
                bs_positions_reference = bs_positions
            elif not np.array_equal(bs_reference, bs_indices):
                msg = f"BS index order differs in {file}: {bs_indices} vs {bs_reference}"
                raise ValueError(msg)
            chunks.append((ue_indices, bs_indices, ue_positions, rss, valid))
    if not chunks or bs_reference is None or bs_positions_reference is None:
        msg = "No radio-map chunks loaded."
        raise ValueError(msg)
    ue = np.concatenate([item[0] for item in chunks]).astype(np.int64, copy=False)
    positions = np.concatenate([item[2] for item in chunks], axis=0).astype(
        np.float32,
        copy=False,
    )
    rss = np.concatenate([item[3] for item in chunks], axis=0).astype(np.float32, copy=False)
    valid = np.concatenate([item[4] for item in chunks], axis=0).astype(np.bool_, copy=False)
    order = np.argsort(ue, kind="stable")
    return RadioMapTable(
        ue_indices=ue[order],
        bs_indices=bs_reference.astype(np.int64, copy=False),
        positions_m=positions[order],
        bs_positions_m=bs_positions_reference.astype(np.float32, copy=False),
        rss_dbm=rss[order],
        valid_mask=valid[order] & np.isfinite(rss[order]),
    )


def floorplan_extent(
    meta: dict[str, Any],
    floorplan: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return matplotlib extent ``(xmin, xmax, ymin, ymax)`` in meters."""

    origin_x, origin_y = [float(value) for value in meta.get("origin_xy_m", [0.0, 0.0])]
    if "extent_xy_m" in meta:
        width_m, height_m = [float(value) for value in meta["extent_xy_m"]]
    else:
        resolution = float(meta["resolution_m_per_pixel"])
        height_px, width_px = floorplan.shape[:2]
        width_m = width_px * resolution
        height_m = height_px * resolution
    return origin_x, origin_x + width_m, origin_y, origin_y + height_m


def _plot_radio_maps(
    table: RadioMapTable,
    floorplan: np.ndarray,
    floorplan_extent_m: tuple[float, float, float, float],
    output_dir: Path,
    config: RadioMapRenderConfig,
) -> list[Path]:
    finite_values = table.rss_dbm[table.valid_mask]
    if finite_values.size == 0:
        msg = "No finite RSS values found for radio-map rendering."
        raise ValueError(msg)
    vmin = float(np.nanmin(finite_values))
    vmax = float(np.nanmax(finite_values))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    modes = ("interpolated", "samples") if config.render_mode == "both" else (config.render_mode,)
    paths: list[Path] = []
    for bs_col, bs_index in enumerate(table.bs_indices):
        mask = table.valid_mask[:, bs_col] & np.all(np.isfinite(table.positions_m[:, :2]), axis=1)
        if not np.any(mask):
            continue
        x = table.positions_m[mask, 0].astype(np.float64, copy=False)
        y = table.positions_m[mask, 1].astype(np.float64, copy=False)
        values = table.rss_dbm[mask, bs_col].astype(np.float64, copy=False)
        bs_position = table.bs_positions_m[bs_col]
        for mode in modes:
            filename = f"radio_map_bs_{int(bs_index):03d}_{mode}.png"
            output_path = output_dir / filename
            _plot_one_radio_map(
                floorplan,
                floorplan_extent_m,
                x,
                y,
                values,
                title=f"BS {int(bs_index)} RSS Radio Map",
                output_path=output_path,
                mode=mode,
                bs_index=int(bs_index),
                bs_position_m=bs_position,
                vmin=vmin,
                vmax=vmax,
                config=config,
            )
            paths.append(output_path)
    return paths


def _plot_one_radio_map(
    floorplan: np.ndarray,
    floorplan_extent_m: tuple[float, float, float, float],
    x_m: np.ndarray,
    y_m: np.ndarray,
    values_dbm: np.ndarray,
    *,
    title: str,
    output_path: Path,
    mode: str,
    bs_index: int,
    bs_position_m: np.ndarray,
    vmin: float,
    vmax: float,
    config: RadioMapRenderConfig,
) -> None:
    fig, axis = plt.subplots(figsize=(8.0, 8.5))
    axis.imshow(floorplan, extent=floorplan_extent_m, origin=config.image_origin)
    mappable: Any
    if mode == "interpolated":
        grid, grid_extent = _interpolate_idw_grid(
            x_m,
            y_m,
            values_dbm,
            resolution_m=config.grid_resolution_m,
            neighbors=config.interpolation_neighbors,
            power=config.interpolation_power,
        )
        mappable = axis.imshow(
            grid,
            extent=grid_extent,
            origin="lower",
            cmap=config.colormap,
            interpolation="bilinear",
            alpha=config.heatmap_alpha,
            vmin=vmin,
            vmax=vmax,
        )
        if config.show_samples_on_interpolated:
            axis.scatter(
                x_m,
                y_m,
                c=values_dbm,
                cmap=config.colormap,
                vmin=vmin,
                vmax=vmax,
                s=max(config.point_size * 0.55, 3.0),
                linewidths=0.0,
                alpha=0.85,
            )
    elif mode == "samples":
        mappable = axis.scatter(
            x_m,
            y_m,
            c=values_dbm,
            cmap=config.colormap,
            vmin=vmin,
            vmax=vmax,
            s=config.point_size,
            linewidths=0.0,
            alpha=0.9,
        )
    else:
        raise ValueError(f"Unsupported radio-map plot mode: {mode!r}")
    if np.all(np.isfinite(bs_position_m[:2])):
        bs_x = float(bs_position_m[0])
        bs_y = float(bs_position_m[1])
        axis.scatter(
            [bs_x],
            [bs_y],
            marker="*",
            s=220.0,
            c="#ff3b30",
            edgecolors="black",
            linewidths=1.0,
            zorder=5,
        )
        axis.annotate(
            f"BS {bs_index}",
            xy=(bs_x, bs_y),
            xytext=(7, 7),
            textcoords="offset points",
            color="white",
            fontsize=9,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "black",
                "edgecolor": "none",
                "alpha": 0.68,
            },
            zorder=6,
        )
    cbar = fig.colorbar(mappable, ax=axis, shrink=0.82)
    cbar.set_label("RSS [dBm]")
    axis.set_title(title)
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_xlim(floorplan_extent_m[0], floorplan_extent_m[1])
    axis.set_ylim(floorplan_extent_m[2], floorplan_extent_m[3])
    axis.set_aspect("equal", adjustable="box")
    fig.savefig(output_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)


def _interpolate_idw_grid(
    x_m: np.ndarray,
    y_m: np.ndarray,
    values: np.ndarray,
    *,
    resolution_m: float | None,
    neighbors: int,
    power: float,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    finite = np.isfinite(x_m) & np.isfinite(y_m) & np.isfinite(values)
    if np.count_nonzero(finite) < 1:
        msg = "Need at least one finite sample to render a radio map."
        raise ValueError(msg)
    x = x_m[finite]
    y = y_m[finite]
    z = values[finite]
    spacing = (
        float(resolution_m)
        if resolution_m is not None
        else min(_infer_grid_spacing(x), _infer_grid_spacing(y))
    )
    x_axis = _regular_axis(float(np.min(x)), float(np.max(x)), spacing)
    y_axis = _regular_axis(float(np.min(y)), float(np.max(y)), spacing)
    points = np.column_stack([x, y]).astype(np.float64, copy=False)
    grid_x, grid_y = np.meshgrid(x_axis, y_axis)
    targets = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    grid_values = _idw_interpolate(
        points,
        z.astype(np.float64, copy=False),
        targets,
        neighbors=min(int(neighbors), points.shape[0]),
        power=float(power),
    )
    grid = grid_values.reshape(grid_y.shape).astype(np.float32, copy=False)
    return grid, (float(x_axis[0]), float(x_axis[-1]), float(y_axis[0]), float(y_axis[-1]))


def _idw_interpolate(
    points: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
    *,
    neighbors: int,
    power: float,
) -> np.ndarray:
    out = np.empty((targets.shape[0],), dtype=np.float64)
    chunk_size = 8192
    for start in range(0, targets.shape[0], chunk_size):
        stop = min(start + chunk_size, targets.shape[0])
        delta = targets[start:stop, np.newaxis, :] - points[np.newaxis, :, :]
        distance = np.linalg.norm(delta, axis=2)
        nearest = np.argpartition(distance, kth=neighbors - 1, axis=1)[:, :neighbors]
        nearest_distance = np.take_along_axis(distance, nearest, axis=1)
        nearest_values = values[nearest]
        exact = nearest_distance[:, 0] <= 1e-9
        weights = 1.0 / np.maximum(nearest_distance, 1e-9) ** power
        interpolated = np.sum(weights * nearest_values, axis=1) / np.sum(weights, axis=1)
        if np.any(exact):
            interpolated[exact] = nearest_values[exact, 0]
        out[start:stop] = interpolated
    return out


def _write_radio_map_csv(path: Path, table: RadioMapTable) -> None:
    fieldnames = ["ue_index", "x_m", "y_m", "z_m"]
    fieldnames.extend(f"bs_{int(bs):03d}_rss_dbm" for bs in table.bs_indices)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row, position in enumerate(table.positions_m):
            item: dict[str, float | int] = {
                "ue_index": int(table.ue_indices[row]),
                "x_m": float(position[0]),
                "y_m": float(position[1]),
                "z_m": float(position[2]),
            }
            for col, bs_index in enumerate(table.bs_indices):
                key = f"bs_{int(bs_index):03d}_rss_dbm"
                item[key] = (
                    float(table.rss_dbm[row, col])
                    if table.valid_mask[row, col]
                    else float("nan")
                )
            writer.writerow(item)


def _resolve_floorplan_paths(
    root: Path,
    manifest: dict[str, Any],
    *,
    files: list[Path],
    floorplan_image: Path | None,
    floorplan_meta: Path | None,
) -> dict[str, Path]:
    if floorplan_image is not None and floorplan_meta is not None:
        return {"image": floorplan_image, "meta": floorplan_meta}
    scene_root = _infer_scene_root(root, manifest, files)
    floorplan_dir = scene_root / "floorplan"
    image = floorplan_image or _find_floorplan_image(floorplan_dir)
    meta = floorplan_meta or (floorplan_dir / "meta.json")
    if not image.exists():
        msg = f"Floorplan image not found: {image}"
        raise FileNotFoundError(msg)
    if not meta.exists():
        msg = f"Floorplan meta not found: {meta}"
        raise FileNotFoundError(msg)
    return {"image": image, "meta": meta}


def _infer_scene_root(root: Path, manifest: dict[str, Any], files: list[Path]) -> Path:
    label_file = manifest.get("label_file")
    if not label_file:
        config_snapshot = manifest.get("config_snapshot", {})
        if isinstance(config_snapshot, dict):
            label_file = config_snapshot.get("label_file")
    if not label_file:
        config_path = manifest.get("config_snapshot_path")
        if config_path:
            resolved = _resolve_manifest_path(_run_dir_from_path(root), Path(config_path))
            if resolved.exists():
                data = json.loads(resolved.read_text(encoding="utf-8"))
                label_file = data.get("label_file")
    if not label_file:
        label_file = _label_file_from_hdf5(files)
    if not label_file:
        msg = "Cannot infer floorplan path; pass --floorplan-image and --floorplan-meta."
        raise FileNotFoundError(msg)
    label_path = Path(label_file)
    if label_path.parent.name == "label":
        return label_path.parent.parent
    return label_path.parent


def _find_floorplan_image(floorplan_dir: Path) -> Path:
    preferred = (
        "floorplan_1p60.png",
        "000_z_1.60.png",
        "floorplan.png",
    )
    for name in preferred:
        candidate = floorplan_dir / name
        if candidate.exists():
            return candidate
    candidates = [
        path
        for path in sorted(floorplan_dir.glob("*.png"))
        if path.name not in {"geometry_raw.png", "preview.png", "side_view.png"}
    ]
    if candidates:
        return candidates[0]
    return floorplan_dir / preferred[0]


def _label_file_from_hdf5(files: list[Path]) -> str:
    for file in files:
        if not file.exists():
            continue
        with h5py.File(file, "r") as h5:
            for path in ("input/label_file", "meta/config_snapshot"):
                if path not in h5:
                    continue
                value = h5[path][()]
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                if path == "input/label_file":
                    return str(value)
                try:
                    snapshot = json.loads(str(value))
                except json.JSONDecodeError:
                    continue
                label_file = snapshot.get("label_file")
                if label_file:
                    return str(label_file)
    return ""


def _read_rss_by_ue_bs(
    h5: h5py.File,
    dataset_path: str,
    snapshot_index: int,
) -> np.ndarray:
    normalized = dataset_path[1:] if dataset_path.startswith("/") else dataset_path
    if normalized not in h5:
        msg = f"Missing RSS dataset /{normalized}"
        raise KeyError(msg)
    data = np.asarray(h5[normalized][()])
    if data.ndim == 3:
        if snapshot_index >= data.shape[0]:
            raise IndexError(f"snapshot_index {snapshot_index} outside {data.shape}")
        data = data[snapshot_index]
    if data.ndim != 2:
        msg = f"/{normalized} must have shape [snapshot,tx,rx] or [tx,rx], got {data.shape}"
        raise ValueError(msg)
    return data.astype(np.float32, copy=False)


def _read_valid_by_ue_bs(
    h5: h5py.File,
    shape: tuple[int, int],
    *,
    tx_role: str,
    rx_role: str,
) -> np.ndarray:
    if "observation/valid_mask" in h5:
        valid = np.asarray(h5["observation/valid_mask"][()])
        if valid.ndim == 3:
            valid = valid[0]
    elif "derived/link_valid_mask" in h5:
        valid = np.asarray(h5["derived/link_valid_mask"][()])
    else:
        valid = np.ones(shape, dtype=np.bool_)
    if valid.shape != shape:
        if tx_role == "bs" and rx_role == "ue" and valid.T.shape == shape:
            valid = valid.T
        else:
            valid = np.ones(shape, dtype=np.bool_)
    return valid.astype(np.bool_, copy=False)


def _role_positions(
    h5: h5py.File,
    role: str,
    *,
    tx_role: str,
    rx_role: str,
) -> np.ndarray:
    dataset = _role_dataset_path(role, "positions_m", tx_role=tx_role, rx_role=rx_role)
    return np.asarray(h5[dataset][()], dtype=np.float32)


def _role_global_indices(
    h5: h5py.File,
    role: str,
    *,
    tx_role: str,
    rx_role: str,
) -> np.ndarray:
    axis = _role_axis(role, tx_role=tx_role, rx_role=rx_role)
    shard_path = f"shard/global_{axis}_indices"
    if shard_path in h5:
        return np.asarray(h5[shard_path][()], dtype=np.int64)
    positions_path = f"topology/{axis}_positions_m"
    return np.arange(h5[positions_path].shape[0], dtype=np.int64)


def _role_dataset_path(
    role: str,
    name: str,
    *,
    tx_role: str,
    rx_role: str,
) -> str:
    axis = _role_axis(role, tx_role=tx_role, rx_role=rx_role)
    return f"topology/{axis}_{name}"


def _role_axis(role: str, *, tx_role: str, rx_role: str) -> str:
    if role == tx_role:
        return "tx"
    if role == rx_role:
        return "rx"
    msg = f"Role {role!r} is not present in link roles tx={tx_role!r}, rx={rx_role!r}"
    raise ValueError(msg)


def _link_roles(h5: h5py.File) -> tuple[str, str]:
    tx_role = _read_string(h5, "link/tx_role", "bs")
    rx_role = _read_string(h5, "link/rx_role", "ue")
    if {tx_role, rx_role} != {"bs", "ue"}:
        return "bs", "ue"
    return tx_role, rx_role


def _read_string(h5: h5py.File, path: str, default: str) -> str:
    if path not in h5:
        return default
    value = h5[path][()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _resolve_manifest_path(run_dir: Path, result_path: Path) -> Path:
    if result_path.is_absolute():
        return result_path
    if result_path.exists():
        return result_path
    candidate = run_dir / result_path
    if candidate.exists():
        return candidate
    return result_path


def _run_dir_from_path(path: Path) -> Path:
    if path.is_file() and path.parent.name == "manifest":
        return path.parent.parent
    if path.name == "manifest":
        return path.parent
    return path


def _default_output_dir(root: Path, manifest: dict[str, Any]) -> Path:
    run_dir = _run_dir_from_path(root)
    if not manifest and root.is_file() and root.suffix == ".h5":
        run_dir = root.parent
    return run_dir / "figures" / "heatmaps"


def _h5_sort_key(path: Path) -> tuple[int, ...]:
    numbers = tuple(int(value) for value in re.findall(r"\d+", path.name))
    return numbers or (0,)


def _regular_axis(min_value: float, max_value: float, spacing: float) -> np.ndarray:
    if np.isclose(min_value, max_value):
        return np.asarray([min_value], dtype=np.float64)
    count = int(round((max_value - min_value) / spacing)) + 1
    return min_value + np.arange(max(count, 1), dtype=np.float64) * spacing


def _infer_grid_spacing(values: np.ndarray) -> float:
    unique = np.unique(np.round(values.astype(np.float64), decimals=6))
    deltas = np.diff(unique)
    positive = deltas[deltas > 1e-6]
    if positive.size == 0:
        return 1.0
    return float(np.median(positive))


def _summary_stats(values: np.ndarray) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0, "min": float("nan"), "max": float("nan"), "mean": float("nan")}
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }
