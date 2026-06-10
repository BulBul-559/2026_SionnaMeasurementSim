"""Configuration-driven visualization reports for simulation HDF5 files."""

from __future__ import annotations

import json
import warnings
from csv import DictWriter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np

from sionna_measurement_sim.domain.array import ArraySpectrumConfig
from sionna_measurement_sim.phy.spatial_spectrum import build_bartlett_spectrum
from sionna_measurement_sim.visualization.config import VisualizationRunConfig
from sionna_measurement_sim.visualization.radio_map import (
    RadioMapRenderConfig,
    generate_radio_map_heatmaps,
)

_EPS = 1e-30


def generate_visualization_report(
    hdf5_path: str | Path,
    output_dir: str | Path,
    config: VisualizationRunConfig | None = None,
    *,
    mode: str = "sample",
    bs_indices: list[int] | None = None,
    ue_indices: list[int] | None = None,
    plots: list[str] | tuple[str, ...] | None = None,
    dataset_path: str | None = None,
    plot_type: str = "auto",
) -> dict[str, Any]:
    """Generate PNG visualizations plus an index JSON file from an HDF5 result."""

    hdf5_path = Path(hdf5_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = config or VisualizationRunConfig(enabled=True)
    selected_plots = tuple(plots) if plots is not None else cfg.plots
    if mode == "full":
        selected_plots = ("full_summary",)
    elif mode == "dataset":
        selected_plots = ("dataset_preview",)

    with h5py.File(hdf5_path, "r") as h5:
        selection = select_visualization_links(
            h5,
            cfg,
            bs_indices=bs_indices,
            ue_indices=ue_indices,
            mode=mode,
        )
        index: dict[str, Any] = {
            "hdf5_path": hdf5_path.as_posix(),
            "output_dir": output_dir.as_posix(),
            "mode": mode,
            "config": asdict(cfg),
            "selected_bs_indices": selection["bs_indices"],
            "selected_ue_indices": selection["ue_indices"],
            "generated_files": [],
            "skipped_plots": [],
        }

        for plot_name in selected_plots:
            try:
                plot_output_dir = _plot_output_dir(output_dir, plot_name)
                path = _dispatch_plot(
                    h5,
                    plot_output_dir,
                    plot_name,
                    selection,
                    cfg,
                    dataset_path=dataset_path,
                    plot_type=plot_type,
                )
            except _SkipPlot as exc:
                index["skipped_plots"].append({"plot": plot_name, "reason": str(exc)})
                continue
            for generated_path in _as_path_list(path):
                index["generated_files"].append(
                    {
                        "plot": plot_name,
                        "path": generated_path.as_posix(),
                        "bytes": generated_path.stat().st_size,
                    }
                )

    index_path = output_dir / "index.json"
    index["index_path"] = index_path.as_posix()
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    return index


def select_visualization_links(
    h5: h5py.File,
    config: VisualizationRunConfig,
    *,
    bs_indices: list[int] | None = None,
    ue_indices: list[int] | None = None,
    mode: str = "sample",
) -> dict[str, Any]:
    """Select BS/UE indices using BS/UE semantics regardless of HDF5 group ordering."""

    tx_role, rx_role = _link_roles(h5)
    num_bs = _role_count(h5, "bs", tx_role=tx_role, rx_role=rx_role)
    num_ue = _role_count(h5, "ue", tx_role=tx_role, rx_role=rx_role)
    if bs_indices is None:
        bs = list(range(min(num_bs, config.max_bs)))
    else:
        bs = _clip_unique(bs_indices, num_bs)[: config.max_bs]
    if not bs and num_bs:
        bs = [0]

    if ue_indices is not None:
        ue = _clip_unique(ue_indices, num_ue)[: config.max_ue]
    elif mode == "full":
        ue = list(range(num_ue))
    elif config.sample_policy == "first":
        ue = list(range(min(num_ue, min(config.sample_ue_count, config.max_ue))))
    else:
        ue = _sample_ue_indices(
            h5,
            config,
            bs,
            num_ue,
            tx_role=tx_role,
            rx_role=rx_role,
        )
    if not ue and num_ue:
        ue = [0]

    return {
        "bs_indices": [int(value) for value in bs],
        "ue_indices": [int(value) for value in ue],
        "tx_role": tx_role,
        "rx_role": rx_role,
    }


def _sample_ue_indices(
    h5: h5py.File,
    config: VisualizationRunConfig,
    bs: list[int],
    num_ue: int,
    *,
    tx_role: str,
    rx_role: str,
) -> list[int]:
    rng = np.random.default_rng(config.random_seed)
    target = min(config.sample_ue_count, config.max_ue, num_ue)
    if target <= 0:
        return []

    candidates: np.ndarray
    if (
        config.sample_policy in ("valid_links_first", "spatially_spread_valid_links")
        and "derived/link_valid_mask" in h5
    ):
        valid = np.asarray(h5["derived/link_valid_mask"][()])
        candidates = _valid_ue_candidates(valid, bs, tx_role=tx_role, rx_role=rx_role)
    else:
        candidates = np.arange(num_ue)

    selected: list[int] = []
    if candidates.size:
        count = min(target, candidates.size)
        if (
            config.sample_policy == "spatially_spread_valid_links"
            and _role_positions_dataset(h5, "ue", tx_role=tx_role, rx_role=rx_role) is not None
        ):
            selected.extend(
                _select_spatially_spread_ues(
                    h5,
                    candidates,
                    count,
                    tx_role=tx_role,
                    rx_role=rx_role,
                )
            )
        else:
            selected.extend(rng.choice(candidates, size=count, replace=False).tolist())
    if len(selected) < target:
        remaining = np.setdiff1d(np.arange(num_ue), np.asarray(selected, dtype=np.int64))
        count = min(target - len(selected), remaining.size)
        if count:
            selected.extend(rng.choice(remaining, size=count, replace=False).tolist())
    return sorted(int(value) for value in selected)


def _select_spatially_spread_ues(
    h5: h5py.File,
    candidates: np.ndarray,
    count: int,
    *,
    tx_role: str,
    rx_role: str,
) -> list[int]:
    positions_dataset = _role_positions_dataset(h5, "ue", tx_role=tx_role, rx_role=rx_role)
    if positions_dataset is None:
        return candidates[:count].astype(int).tolist()
    positions = np.asarray(positions_dataset[candidates, :2], dtype=np.float64)
    finite = np.all(np.isfinite(positions), axis=1)
    if not np.any(finite):
        return candidates[:count].astype(int).tolist()
    candidate_values = candidates[finite].astype(int)
    xy = positions[finite]
    if candidate_values.size <= count:
        return candidate_values.tolist()

    centroid = np.mean(xy, axis=0)
    first = int(np.argmax(np.linalg.norm(xy - centroid, axis=1)))
    selected_positions = [xy[first]]
    selected_offsets = [first]
    remaining = np.ones(candidate_values.shape[0], dtype=bool)
    remaining[first] = False
    while len(selected_offsets) < count and np.any(remaining):
        remaining_indices = np.flatnonzero(remaining)
        distances = np.stack(
            [np.linalg.norm(xy[remaining_indices] - point, axis=1) for point in selected_positions],
            axis=1,
        )
        next_index = int(remaining_indices[int(np.argmax(np.min(distances, axis=1)))])
        selected_offsets.append(next_index)
        selected_positions.append(xy[next_index])
        remaining[next_index] = False
    return candidate_values[np.asarray(selected_offsets, dtype=np.int64)].tolist()


def _link_roles(h5: h5py.File) -> tuple[str, str]:
    """Return resolved TX/RX roles, defaulting to legacy BS->UE files."""

    tx_role = _read_h5_string(h5, "link/tx_role", default="bs")
    rx_role = _read_h5_string(h5, "link/rx_role", default="ue")
    if {tx_role, rx_role} != {"bs", "ue"}:
        return "bs", "ue"
    return tx_role, rx_role


def _read_h5_string(h5: h5py.File, path: str, *, default: str) -> str:
    if path not in h5:
        return default
    value = h5[path][()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _role_count(h5: h5py.File, role: str, *, tx_role: str, rx_role: str) -> int:
    dataset = _role_positions_dataset(h5, role, tx_role=tx_role, rx_role=rx_role)
    if dataset is None:
        return 0
    return int(dataset.shape[0])


def _role_positions_dataset(
    h5: h5py.File,
    role: str,
    *,
    tx_role: str,
    rx_role: str,
) -> h5py.Dataset | None:
    if role == tx_role and "topology/tx_positions_m" in h5:
        return h5["topology/tx_positions_m"]
    if role == rx_role and "topology/rx_positions_m" in h5:
        return h5["topology/rx_positions_m"]
    return None


def _role_positions(h5: h5py.File, role: str, selection: dict[str, Any]) -> np.ndarray:
    dataset = _role_positions_dataset(
        h5,
        role,
        tx_role=str(selection.get("tx_role", "bs")),
        rx_role=str(selection.get("rx_role", "ue")),
    )
    if dataset is None:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(dataset[()], dtype=np.float32)


def _link_index_pair(selection: dict[str, Any], bs_idx: int, ue_idx: int) -> tuple[int, int]:
    tx_role = str(selection.get("tx_role", "bs"))
    if tx_role == "bs":
        return int(bs_idx), int(ue_idx)
    return int(ue_idx), int(bs_idx)


def _selected_link_pairs(selection: dict[str, Any]) -> list[tuple[int, int]]:
    return [
        _link_index_pair(selection, int(bs_idx), int(ue_idx))
        for bs_idx in selection["bs_indices"]
        for ue_idx in selection["ue_indices"]
    ]


def _first_selected_link_pair(selection: dict[str, Any]) -> tuple[int, int, int, int]:
    """Return the first selected link as BS/UE indices plus resolved TX/RX indices."""

    if not selection["bs_indices"] or not selection["ue_indices"]:
        raise _SkipPlot("no selected BS/UE link for path samples")
    bs_idx = int(selection["bs_indices"][0])
    ue_idx = int(selection["ue_indices"][0])
    tx_idx, rx_idx = _link_index_pair(selection, bs_idx, ue_idx)
    return bs_idx, ue_idx, tx_idx, rx_idx


def _link_matrix_for_bs_ue(values: np.ndarray, selection: dict[str, Any]) -> np.ndarray:
    array = np.asarray(values)
    out = np.empty(
        (len(selection["bs_indices"]), len(selection["ue_indices"])),
        dtype=array.dtype,
    )
    for row, bs_idx in enumerate(selection["bs_indices"]):
        for col, ue_idx in enumerate(selection["ue_indices"]):
            tx_idx, rx_idx = _link_index_pair(selection, int(bs_idx), int(ue_idx))
            out[row, col] = array[tx_idx, rx_idx]
    return out


def _valid_ue_candidates(
    valid: np.ndarray,
    bs: list[int],
    *,
    tx_role: str,
    rx_role: str,
) -> np.ndarray:
    _ = rx_role
    if not bs:
        return np.array([], dtype=np.int64)
    bs_array = np.asarray(bs, dtype=np.int64)
    if tx_role == "bs":
        return np.flatnonzero(np.any(valid[bs_array, :], axis=0))
    return np.flatnonzero(np.any(valid[:, bs_array], axis=1))


def _dispatch_plot(
    h5: h5py.File,
    output_dir: Path,
    plot_name: str,
    selection: dict[str, Any],
    config: VisualizationRunConfig,
    *,
    dataset_path: str | None,
    plot_type: str,
) -> Path | list[Path]:
    dispatch = {
        "topology": _plot_topology,
        "link_overview": _plot_link_overview,
        "cfr_lines": _plot_cfr_lines,
        "cfr_heatmap": _plot_cfr_heatmap,
        "cfr_error": _plot_cfr_error,
        "waveform_grid": _plot_waveform_grid,
        "aoa": _plot_aoa,
        "nlos_paths": _plot_nlos_paths,
        "spatial_spectrum": _plot_spatial_spectrum,
        "nmse_snr": _plot_nmse_snr,
        "path_samples": _plot_path_samples,
        "multiuser_srs": _plot_multiuser_srs,
        "radio_map": _plot_radio_map,
        "full_summary": _plot_full_summary,
        "dataset_preview": _plot_dataset_preview,
    }
    if plot_name not in dispatch:
        raise _SkipPlot(f"unknown plot {plot_name!r}")
    return dispatch[plot_name](
        h5,
        output_dir,
        selection,
        config,
        dataset_path=dataset_path,
        plot_type=plot_type,
    )


def _plot_output_dir(output_dir: Path, plot_name: str) -> Path:
    if plot_name == "radio_map":
        return output_dir
    if plot_name == "multiuser_srs":
        return output_dir / "multiuser"
    return output_dir / "standard"


def _plot_radio_map(
    h5: h5py.File,
    output_dir: Path,
    selection: dict[str, Any],
    config: VisualizationRunConfig,
    *,
    dataset_path: str | None,
    plot_type: str,
) -> list[Path]:
    _ = selection, plot_type
    summary = generate_radio_map_heatmaps(
        h5.filename,
        output_dir / "heatmaps",
        config=RadioMapRenderConfig(
            render_mode=config.radio_map_mode,
            value_dataset=dataset_path or "/observation/rssi_dbm",
            grid_resolution_m=config.radio_map_grid_resolution_m,
            dpi=config.dpi,
            show_samples_on_interpolated=config.radio_map_show_samples,
        ),
    )
    return [Path(path) for path in summary["generated_files"]]


def _plot_multiuser_srs(
    h5: h5py.File,
    output_dir: Path,
    selection: dict[str, Any],
    config: VisualizationRunConfig,
    **_: Any,
) -> list[Path]:
    _require(
        h5,
        (
            "multiuser/rx_grid_shared",
            "multiuser/active_tx_indices",
            "multiuser/active_tx_mask",
            "multiuser/re_symbol_indices",
            "multiuser/re_subcarrier_indices",
            "multiuser/re_mask",
            "multiuser/allocated_subcarrier_indices",
            "multiuser/allocated_subcarrier_mask",
            "multiuser/resource_occupancy_count",
            "multiuser/resource_collision_mask",
            "multiuser/cfr_est_resource",
            "multiuser/cfr_est_allocated",
        ),
    )
    entries = _multiuser_selected_entries(h5, selection)
    if not entries:
        raise _SkipPlot("no selected active UE appears in /multiuser frames")
    rx_indices = _multiuser_selected_rx_indices(h5, selection)
    if not rx_indices:
        raise _SkipPlot("no selected BS/RX for /multiuser visualization")

    generated = [
        _plot_multiuser_resource_grid(h5, output_dir, entries, config),
        _plot_multiuser_resource_vs_allocated(h5, output_dir, entries, config),
        _plot_multiuser_shared_rx_grid(h5, output_dir, entries, rx_indices, config),
    ]
    generated.extend(
        _plot_multiuser_cfr_lines(
            h5,
            output_dir,
            entries,
            rx_indices,
            config,
            dataset_kind="resource",
        )
    )
    generated.extend(
        _plot_multiuser_cfr_lines(
            h5,
            output_dir,
            entries,
            rx_indices,
            config,
            dataset_kind="allocated",
        )
    )
    rows = _multiuser_error_rows(h5, entries, rx_indices)
    generated.extend(_write_multiuser_summary_outputs(output_dir, rows, config))
    generated.append(
        _plot_multiuser_bs_observation_map(h5, output_dir, entries, rx_indices, config)
    )
    generated.extend(_plot_multiuser_spatial_spectra(h5, output_dir, entries, rx_indices, config))
    return generated


def _multiuser_selected_entries(
    h5: h5py.File,
    selection: dict[str, Any],
) -> list[dict[str, int]]:
    if str(selection.get("tx_role", "bs")) != "ue" or str(selection.get("rx_role", "ue")) != "bs":
        raise _SkipPlot("multiuser SRS visualization requires uplink role mapping tx=UE, rx=BS")
    active_tx = np.asarray(h5["multiuser/active_tx_indices"][()], dtype=np.int64)
    active_mask = np.asarray(h5["multiuser/active_tx_mask"][()], dtype=np.bool_)
    selected_tx = set(int(value) for value in selection["ue_indices"])
    entries: list[dict[str, int]] = []
    for frame in range(active_tx.shape[0]):
        for slot in range(active_tx.shape[1]):
            if not active_mask[frame, slot]:
                continue
            tx_idx = int(active_tx[frame, slot])
            if tx_idx not in selected_tx:
                continue
            entries.append({"frame": int(frame), "slot": int(slot), "tx": tx_idx})
    return entries


def _multiuser_selected_rx_indices(
    h5: h5py.File,
    selection: dict[str, Any],
) -> list[int]:
    rx_count = int(h5["multiuser/rx_grid_shared"].shape[2])
    if str(selection.get("rx_role", "ue")) != "bs":
        return []
    return [idx for idx in (int(value) for value in selection["bs_indices"]) if 0 <= idx < rx_count]


def _selected_multiuser_frames(entries: list[dict[str, int]]) -> list[int]:
    return sorted({int(entry["frame"]) for entry in entries})


def _plot_multiuser_resource_grid(
    h5: h5py.File,
    output_dir: Path,
    entries: list[dict[str, int]],
    config: VisualizationRunConfig,
) -> Path:
    frames = _selected_multiuser_frames(entries)
    occupancy = np.asarray(h5["multiuser/resource_occupancy_count"][()])
    collision = np.asarray(h5["multiuser/resource_collision_mask"][()])
    active_tx = np.asarray(h5["multiuser/active_tx_indices"][()])
    active_mask = np.asarray(h5["multiuser/active_tx_mask"][()])
    re_symbols = np.asarray(h5["multiuser/re_symbol_indices"][()])
    re_subcarriers = np.asarray(h5["multiuser/re_subcarrier_indices"][()])
    re_mask = np.asarray(h5["multiuser/re_mask"][()])

    figure, axes = plt.subplots(
        len(frames),
        1,
        figsize=(8.5, 3.1 * len(frames)),
        squeeze=False,
    )
    for row, frame in enumerate(frames):
        axis = axes[row, 0]
        assignment = np.full(occupancy.shape[1:], np.nan, dtype=np.float32)
        for slot in range(active_tx.shape[1]):
            if not active_mask[frame, slot]:
                continue
            mask = re_mask[frame, slot]
            symbols = re_symbols[frame, slot, mask]
            subcarriers = re_subcarriers[frame, slot, mask]
            assignment[symbols, subcarriers] = float(active_tx[frame, slot])
        cmap = plt.get_cmap("tab20").copy()
        cmap.set_bad(color="white")
        image = axis.imshow(
            assignment.T,
            aspect="auto",
            origin="lower",
            interpolation="none",
            cmap=cmap,
        )
        collision_y, collision_x = np.where(collision[frame].T)
        if collision_x.size:
            axis.scatter(
                collision_x,
                collision_y,
                marker="x",
                s=14,
                color="black",
                label="collision",
            )
            axis.legend(loc="upper right", fontsize=7)
        axis.set_xlabel("OFDM symbol")
        axis.set_ylabel("subcarrier")
        axis.set_title(f"Frame {frame} SRS resource ownership")
        figure.colorbar(image, ax=axis, fraction=0.026, pad=0.02, label="local UE/TX index")
    figure.suptitle("Multi-UE SRS Resource Grid")
    return _save_figure(figure, output_dir / f"multiuser_resource_grid.{config.format}", config)


def _plot_multiuser_resource_vs_allocated(
    h5: h5py.File,
    output_dir: Path,
    entries: list[dict[str, int]],
    config: VisualizationRunConfig,
) -> Path:
    subcarrier_count = int(h5["multiuser/resource_occupancy_count"].shape[-1])
    allocated = np.asarray(h5["multiuser/allocated_subcarrier_indices"][()])
    allocated_mask = np.asarray(h5["multiuser/allocated_subcarrier_mask"][()])
    re_subcarriers = np.asarray(h5["multiuser/re_subcarrier_indices"][()])
    re_mask = np.asarray(h5["multiuser/re_mask"][()])
    grid = np.zeros((len(entries), subcarrier_count), dtype=np.float32)
    labels: list[str] = []
    for row, entry in enumerate(entries):
        frame = entry["frame"]
        slot = entry["slot"]
        alloc_idx = allocated[frame, slot, allocated_mask[frame, slot]]
        res_idx = re_subcarriers[frame, slot, re_mask[frame, slot]]
        grid[row, alloc_idx] = 0.45
        grid[row, res_idx] = 1.0
        labels.append(f"F{frame}/UE{entry['tx']}")
    figure, axis = plt.subplots(figsize=(9.0, max(2.6, 0.45 * len(entries) + 1.8)))
    image = axis.imshow(
        grid,
        aspect="auto",
        origin="lower",
        interpolation="none",
        vmin=0.0,
        vmax=1.0,
    )
    axis.set_xlabel("subcarrier")
    axis.set_ylabel("active UE")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_title("Allocated band (dim) vs actual SRS RE (bright)")
    figure.colorbar(image, ax=axis, fraction=0.026, pad=0.02)
    return _save_figure(
        figure,
        output_dir / f"multiuser_resource_vs_allocated.{config.format}",
        config,
    )


def _plot_multiuser_shared_rx_grid(
    h5: h5py.File,
    output_dir: Path,
    entries: list[dict[str, int]],
    rx_indices: list[int],
    config: VisualizationRunConfig,
) -> Path:
    frames = _selected_multiuser_frames(entries)
    rx_grid = h5["multiuser/rx_grid_shared"]
    figure, axes = plt.subplots(
        len(frames),
        len(rx_indices),
        figsize=(4.2 * len(rx_indices), 3.0 * len(frames)),
        squeeze=False,
    )
    for row, frame in enumerate(frames):
        for col, rx_idx in enumerate(rx_indices):
            axis = axes[row, col]
            data = np.asarray(rx_grid[0, frame, rx_idx])
            power = 10.0 * np.log10(np.mean(np.abs(data) ** 2, axis=0) + _EPS)
            image = axis.imshow(power.T, aspect="auto", origin="lower", interpolation="none")
            axis.set_xlabel("OFDM symbol")
            axis.set_ylabel("subcarrier")
            axis.set_title(f"Frame {frame} - BS/RX {rx_idx}")
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="power [dB]")
    figure.suptitle("Multi-UE Shared RX Grid")
    return _save_figure(figure, output_dir / f"multiuser_shared_rx_grid.{config.format}", config)


def _plot_multiuser_cfr_lines(
    h5: h5py.File,
    output_dir: Path,
    entries: list[dict[str, int]],
    rx_indices: list[int],
    config: VisualizationRunConfig,
    *,
    dataset_kind: str,
) -> list[Path]:
    generated = []
    for value_kind in ("magnitude", "phase"):
        figure, axes = plt.subplots(
            len(entries),
            len(rx_indices),
            figsize=(4.4 * len(rx_indices), 2.7 * len(entries)),
            squeeze=False,
        )
        for row, entry in enumerate(entries):
            for col, rx_idx in enumerate(rx_indices):
                axis = axes[row, col]
                subcarriers, values = _multiuser_cfr_values(
                    h5,
                    entry,
                    rx_idx,
                    dataset_kind=dataset_kind,
                )
                _draw_multiuser_cfr_lines(axis, subcarriers, values, value_kind=value_kind)
                axis.set_title(f"F{entry['frame']} UE {entry['tx']} - BS {rx_idx}", fontsize=8)
        figure.suptitle(f"Multi-UE {dataset_kind} CFR {value_kind}")
        generated.append(
            _save_figure(
                figure,
                output_dir / f"multiuser_cfr_{dataset_kind}_{value_kind}.{config.format}",
                config,
            )
        )
    return generated


def _multiuser_cfr_values(
    h5: h5py.File,
    entry: dict[str, int],
    rx_idx: int,
    *,
    dataset_kind: str,
) -> tuple[np.ndarray, np.ndarray]:
    frame = entry["frame"]
    slot = entry["slot"]
    if dataset_kind == "resource":
        mask = np.asarray(h5["multiuser/re_mask"][frame, slot], dtype=np.bool_)
        subcarriers = np.asarray(h5["multiuser/re_subcarrier_indices"][frame, slot])[mask]
        values_full = np.asarray(h5["multiuser/cfr_est_resource"][0, frame, slot, rx_idx])
        values = np.take(values_full, np.flatnonzero(mask), axis=-1)
        return subcarriers, values
    if dataset_kind == "allocated":
        mask = np.asarray(h5["multiuser/allocated_subcarrier_mask"][frame, slot], dtype=np.bool_)
        subcarriers = np.asarray(h5["multiuser/allocated_subcarrier_indices"][frame, slot])[mask]
        values_full = np.asarray(h5["multiuser/cfr_est_allocated"][0, frame, slot, rx_idx])
        values = np.take(values_full, np.flatnonzero(mask), axis=-1)
        return subcarriers, values
    raise ValueError(f"unsupported multiuser CFR dataset kind: {dataset_kind}")


def _draw_multiuser_cfr_lines(
    axis: Any,
    subcarriers: np.ndarray,
    values: np.ndarray,
    *,
    value_kind: str,
) -> None:
    flat = np.asarray(values).reshape(-1, values.shape[-1])
    if value_kind == "magnitude":
        plot_values = 20.0 * np.log10(np.abs(flat) + _EPS)
        axis.set_ylabel("|H| [dB]")
    elif value_kind == "phase":
        plot_values = np.angle(flat)
        axis.set_ylabel("phase [rad]")
    else:
        raise ValueError(f"unsupported value_kind: {value_kind}")
    for row in plot_values:
        axis.plot(subcarriers, row, alpha=0.65, linewidth=0.8)
    axis.set_xlabel("subcarrier")
    axis.grid(True, alpha=0.25)


def _multiuser_error_rows(
    h5: h5py.File,
    entries: list[dict[str, int]],
    rx_indices: list[int],
) -> list[dict[str, float | int]]:
    truth = h5["channel/truth/cfr"] if "channel/truth/cfr" in h5 else None
    active_tx = np.asarray(h5["multiuser/active_tx_indices"][()])
    comb_offset = np.asarray(h5["multiuser/comb_offset"][()])
    prb_start = np.asarray(h5["multiuser/prb_start"][()])
    prb_count = np.asarray(h5["multiuser/prb_count"][()])
    collision = np.asarray(h5["multiuser/resource_collision_mask"][()])
    rows: list[dict[str, float | int]] = []
    for entry in entries:
        frame = entry["frame"]
        slot = entry["slot"]
        tx_idx = int(active_tx[frame, slot])
        resource_re = int(np.count_nonzero(h5["multiuser/re_mask"][frame, slot]))
        allocated_sc = int(np.count_nonzero(h5["multiuser/allocated_subcarrier_mask"][frame, slot]))
        collision_count = int(np.count_nonzero(collision[frame]))
        for rx_idx in rx_indices:
            resource_nmse = np.nan
            allocated_nmse = np.nan
            if truth is not None:
                resource_nmse = _multiuser_resource_nmse_db(h5, truth, entry, rx_idx)
                allocated_nmse = _multiuser_allocated_nmse_db(h5, truth, entry, rx_idx)
            rows.append(
                {
                    "frame": frame,
                    "active_slot": slot,
                    "tx_index": tx_idx,
                    "rx_index": int(rx_idx),
                    "comb_offset": int(comb_offset[frame, slot]),
                    "prb_start": int(prb_start[frame, slot, 0]),
                    "prb_count": int(prb_count[frame, slot, 0]),
                    "resource_re": resource_re,
                    "allocated_subcarriers": allocated_sc,
                    "frame_collision_count": collision_count,
                    "resource_nmse_db": float(resource_nmse),
                    "allocated_nmse_db": float(allocated_nmse),
                }
            )
    return rows


def _multiuser_resource_nmse_db(
    h5: h5py.File,
    truth: h5py.Dataset,
    entry: dict[str, int],
    rx_idx: int,
) -> float:
    frame = entry["frame"]
    slot = entry["slot"]
    tx_idx = entry["tx"]
    mask = np.asarray(h5["multiuser/re_mask"][frame, slot], dtype=np.bool_)
    subcarriers = np.asarray(h5["multiuser/re_subcarrier_indices"][frame, slot])[mask]
    est_full = np.asarray(h5["multiuser/cfr_est_resource"][0, frame, slot, rx_idx])
    est = np.take(est_full, np.flatnonzero(mask), axis=-1)
    truth_slice = np.asarray(truth[tx_idx, rx_idx])
    pair_count = min(est.shape[1], truth_slice.shape[1])
    truth_values = np.take(truth_slice[:, :pair_count, :], subcarriers, axis=-1)
    return _nmse_db(est[:, :pair_count, :], truth_values)


def _multiuser_allocated_nmse_db(
    h5: h5py.File,
    truth: h5py.Dataset,
    entry: dict[str, int],
    rx_idx: int,
) -> float:
    frame = entry["frame"]
    slot = entry["slot"]
    tx_idx = entry["tx"]
    mask = np.asarray(h5["multiuser/allocated_subcarrier_mask"][frame, slot], dtype=np.bool_)
    subcarriers = np.asarray(h5["multiuser/allocated_subcarrier_indices"][frame, slot])[mask]
    est_full = np.asarray(h5["multiuser/cfr_est_allocated"][0, frame, slot, rx_idx])
    est = np.take(est_full, np.flatnonzero(mask), axis=-1)
    truth_slice = np.asarray(truth[tx_idx, rx_idx])
    pair_count = min(est.shape[1], truth_slice.shape[1])
    truth_values = np.take(truth_slice[:, :pair_count, :], subcarriers, axis=-1)
    return _nmse_db(est[:, :pair_count, :], truth_values)


def _nmse_db(estimate: np.ndarray, truth: np.ndarray) -> float:
    denom = float(np.sum(np.abs(truth) ** 2))
    if denom <= 0.0:
        return float("nan")
    value = float(np.sum(np.abs(estimate - truth) ** 2) / denom)
    return 10.0 * np.log10(max(value, _EPS))


def _write_multiuser_summary_outputs(
    output_dir: Path,
    rows: list[dict[str, float | int]],
    config: VisualizationRunConfig,
) -> list[Path]:
    csv_path = output_dir / "multiuser_frame_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")

    labels = [f"F{row['frame']}/UE{row['tx_index']}/BS{row['rx_index']}" for row in rows]
    resource = np.array([float(row["resource_nmse_db"]) for row in rows], dtype=np.float32)
    allocated = np.array([float(row["allocated_nmse_db"]) for row in rows], dtype=np.float32)
    x = np.arange(len(rows))
    figure, axis = plt.subplots(figsize=(max(7.0, 0.55 * len(rows)), 4.5))
    axis.bar(x - 0.18, resource, width=0.36, label="resource")
    axis.bar(x + 0.18, allocated, width=0.36, label="allocated")
    axis.set_xticks(x, labels, rotation=45, ha="right")
    axis.set_ylabel("NMSE [dB]")
    axis.set_title("Multi-UE CFR Error Summary")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend(loc="best")
    png_path = _save_figure(
        figure,
        output_dir / f"multiuser_cfr_error_summary.{config.format}",
        config,
    )

    table_rows = rows[: min(len(rows), 16)]
    figure, axis = plt.subplots(figsize=(12.0, max(2.6, 0.42 * len(table_rows) + 1.3)))
    axis.axis("off")
    if table_rows:
        columns = (
            "frame",
            "active_slot",
            "tx_index",
            "rx_index",
            "comb_offset",
            "prb_start",
            "prb_count",
            "resource_re",
            "allocated_subcarriers",
            "frame_collision_count",
        )
        cell_text = [[str(row[column]) for column in columns] for row in table_rows]
        table = axis.table(cellText=cell_text, colLabels=columns, loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.3)
    axis.set_title("Multi-UE Frame Summary")
    table_path = _save_figure(
        figure,
        output_dir / f"multiuser_frame_summary.{config.format}",
        config,
    )
    return [csv_path, png_path, table_path]


def _plot_multiuser_bs_observation_map(
    h5: h5py.File,
    output_dir: Path,
    entries: list[dict[str, int]],
    rx_indices: list[int],
    config: VisualizationRunConfig,
) -> Path:
    bs_positions = _role_positions(
        h5,
        "bs",
        {"tx_role": "ue", "rx_role": "bs", "bs_indices": rx_indices, "ue_indices": []},
    )
    if bs_positions.size == 0:
        raise _SkipPlot("missing BS positions for multiuser BS observation map")
    rows = len(entries)
    figure, axes = plt.subplots(rows, 1, figsize=(7.0, max(3.0, 2.7 * rows)), squeeze=False)
    for row, entry in enumerate(entries):
        axis = axes[row, 0]
        values = []
        for rx_idx in rx_indices:
            _, cfr = _multiuser_cfr_values(h5, entry, rx_idx, dataset_kind="resource")
            values.append(10.0 * np.log10(float(np.mean(np.abs(cfr) ** 2)) + _EPS))
        values_array = np.asarray(values, dtype=np.float32)
        xy = bs_positions[np.asarray(rx_indices, dtype=np.int64), :2]
        image = axis.scatter(xy[:, 0], xy[:, 1], c=values_array, s=90, cmap="viridis")
        for idx, rx_idx in enumerate(rx_indices):
            axis.text(xy[idx, 0], xy[idx, 1], f"BS {rx_idx}", fontsize=8, color="black")
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_title(f"UE {entry['tx']} separated CFR power proxy at BSs")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="mean |H|^2 [dB]")
    figure.suptitle("Multi-UE UE-to-BS Observation Map")
    return _save_figure(
        figure,
        output_dir / f"multiuser_bs_observation_map.{config.format}",
        config,
    )


def _plot_multiuser_spatial_spectra(
    h5: h5py.File,
    output_dir: Path,
    entries: list[dict[str, int]],
    rx_indices: list[int],
    config: VisualizationRunConfig,
) -> list[Path]:
    if "array/angle_grid_rad" not in h5:
        return []
    try:
        spectrum_config = _array_spectrum_config_from_h5(h5)
        shared = _multiuser_shared_spectrum(h5, entries, rx_indices, spectrum_config)
        separated = _multiuser_separated_spectrum(h5, entries, rx_indices, spectrum_config)
    except ValueError:
        return []
    angle_grid = np.asarray(h5["array/angle_grid_rad"][()])
    return [
        _plot_multiuser_spectrum_grid(
            shared,
            angle_grid,
            output_dir / f"multiuser_spatial_spectrum_shared.{config.format}",
            [f"F{frame}" for frame in _selected_multiuser_frames(entries)],
            rx_indices,
            "Shared RX-grid Bartlett spectrum",
            config,
        ),
        _plot_multiuser_spectrum_grid(
            separated,
            angle_grid,
            output_dir / f"multiuser_spatial_spectrum_separated.{config.format}",
            [f"F{entry['frame']}/UE{entry['tx']}" for entry in entries],
            rx_indices,
            "Separated resource-CFR Bartlett spectrum",
            config,
        ),
    ]


def _array_spectrum_config_from_h5(h5: h5py.File) -> ArraySpectrumConfig:
    angle_grid = np.asarray(h5["array/angle_grid_rad"][()])
    return ArraySpectrumConfig(
        enabled=True,
        zenith_bins=int(angle_grid.shape[0]),
        azimuth_bins=int(angle_grid.shape[1]),
        zenith_min_rad=float(angle_grid[0, 0, 0]),
        zenith_max_rad=float(angle_grid[-1, 0, 0]),
        azimuth_min_rad=float(angle_grid[0, 0, 1]),
        azimuth_max_rad=float(angle_grid[0, -1, 1]),
        link_chunk_size=64,
    )


def _rx_orientation_for_indices(h5: h5py.File, rx_indices: list[int]) -> np.ndarray | None:
    if "devices/rx_orientation_rad" not in h5:
        return None
    orientation = np.asarray(h5["devices/rx_orientation_rad"][()])
    if orientation.ndim == 2:
        orientation = orientation[np.newaxis, ...]
    return orientation[:, np.asarray(rx_indices, dtype=np.int64), :]


def _multiuser_shared_spectrum(
    h5: h5py.File,
    entries: list[dict[str, int]],
    rx_indices: list[int],
    spectrum_config: ArraySpectrumConfig,
) -> np.ndarray:
    frames = _selected_multiuser_frames(entries)
    samples = np.asarray(h5["multiuser/rx_grid_shared"][:, frames, :, :, :, :])
    samples = samples[:, :, np.asarray(rx_indices, dtype=np.int64), :, :, :]
    return build_bartlett_spectrum(
        samples,
        rx_num_rows=int(h5["antenna/rx_num_rows"][()]),
        rx_num_cols=int(h5["antenna/rx_num_cols"][()]),
        rx_spacing_lambda=tuple(float(v) for v in h5["antenna/rx_spacing_lambda"][()]),
        rx_orientation_rad=_rx_orientation_for_indices(h5, rx_indices),
        config=spectrum_config,
    )


def _multiuser_separated_spectrum(
    h5: h5py.File,
    entries: list[dict[str, int]],
    rx_indices: list[int],
    spectrum_config: ArraySpectrumConfig,
) -> np.ndarray:
    resource = h5["multiuser/cfr_est_resource"]
    snap = int(resource.shape[0])
    rx_ant = int(resource.shape[4])
    port = int(resource.shape[5])
    max_re = int(resource.shape[6])
    samples = np.zeros(
        (snap, len(entries), len(rx_indices), rx_ant, port, max_re),
        dtype=np.complex64,
    )
    for entry_idx, entry in enumerate(entries):
        for col, rx_idx in enumerate(rx_indices):
            samples[:, entry_idx, col] = resource[:, entry["frame"], entry["slot"], rx_idx]
    return build_bartlett_spectrum(
        samples,
        rx_num_rows=int(h5["antenna/rx_num_rows"][()]),
        rx_num_cols=int(h5["antenna/rx_num_cols"][()]),
        rx_spacing_lambda=tuple(float(v) for v in h5["antenna/rx_spacing_lambda"][()]),
        rx_orientation_rad=_rx_orientation_for_indices(h5, rx_indices),
        config=spectrum_config,
    )


def _plot_multiuser_spectrum_grid(
    spectrum: np.ndarray,
    angle_grid: np.ndarray,
    output_path: Path,
    row_labels: list[str],
    rx_indices: list[int],
    title: str,
    config: VisualizationRunConfig,
) -> Path:
    _ = angle_grid
    rows = max(len(row_labels), 1)
    cols = max(len(rx_indices), 1)
    figure, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.0 * rows), squeeze=False)
    for row, row_label in enumerate(row_labels):
        row_values = spectrum[0, row]
        finite = row_values[np.isfinite(row_values)]
        if finite.size:
            vmin, vmax = _stable_color_limits(float(np.min(finite)), float(np.max(finite)))
        else:
            vmin, vmax = (0.0, 1.0)
        for col, rx_idx in enumerate(rx_indices):
            axis = axes[row, col]
            image = axis.imshow(
                spectrum[0, row, col],
                aspect="auto",
                origin="lower",
                interpolation="none",
                vmin=vmin,
                vmax=vmax,
            )
            axis.set_xlabel("azimuth bin")
            axis.set_ylabel("zenith bin")
            axis.set_title(f"{row_label} - BS {rx_idx}", fontsize=8)
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(title)
    return _save_figure(figure, output_path, config)


def _plot_topology(
    h5: h5py.File,
    output_dir: Path,
    selection: dict[str, Any],
    config: VisualizationRunConfig,
    **_: Any,
) -> Path:
    bs_positions = _role_positions(h5, "bs", selection)
    ue_positions = _role_positions(h5, "ue", selection)
    bs = selection["bs_indices"]
    ue = selection["ue_indices"]
    figure, axis = plt.subplots(figsize=(7, 5))
    if ue_positions.size:
        axis.scatter(ue_positions[:, 0], ue_positions[:, 1], s=8, alpha=0.25, label="UE all")
    if bs_positions.size:
        axis.scatter(
            bs_positions[:, 0],
            bs_positions[:, 1],
            marker="^",
            s=70,
            color="tab:red",
            label="BS all",
        )
    if ue and ue_positions.size:
        axis.scatter(
            ue_positions[ue, 0],
            ue_positions[ue, 1],
            s=45,
            color="tab:blue",
            label="UE selected",
        )
    if bs and bs_positions.size:
        axis.scatter(
            bs_positions[bs, 0],
            bs_positions[bs, 1],
            marker="^",
            s=110,
            color="black",
            label="BS selected",
        )
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title("Topology Sample")
    axis.set_aspect("equal", adjustable="box")
    axis.legend(loc="best")
    return _save_figure(figure, output_dir / f"topology.{config.format}", config)


def _plot_link_overview(
    h5: h5py.File,
    output_dir: Path,
    selection: dict[str, Any],
    config: VisualizationRunConfig,
    **_: Any,
) -> Path:
    required = (
        "derived/link_valid_mask",
        "derived/los_flag",
        "derived/nlos_flag",
        "derived/path_count",
    )
    _require(h5, required)
    bs = selection["bs_indices"]
    ue = selection["ue_indices"]
    datasets = (
        ("valid", h5["derived/link_valid_mask"][()]),
        ("los", h5["derived/los_flag"][()]),
        ("nlos", h5["derived/nlos_flag"][()]),
        ("path_count", h5["derived/path_count"][()]),
    )
    figure, axes = plt.subplots(1, 4, figsize=(15, 4), squeeze=False)
    for axis, (title, data) in zip(axes[0], datasets, strict=True):
        image = axis.imshow(
            _link_matrix_for_bs_ue(np.asarray(data), selection),
            aspect="auto",
            origin="lower",
            interpolation="none",
        )
        axis.set_title(title)
        axis.set_xlabel("UE index")
        axis.set_ylabel("BS index")
        axis.set_xticks(range(len(ue)), ue, rotation=45)
        axis.set_yticks(range(len(bs)), bs)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    return _save_figure(figure, output_dir / f"link_overview.{config.format}", config)


def _plot_cfr_lines(
    h5: h5py.File,
    output_dir: Path,
    selection: dict[str, Any],
    config: VisualizationRunConfig,
    **_: Any,
) -> list[Path]:
    _require(h5, ("channel/truth/cfr",))
    cfr = h5["channel/truth/cfr"]
    freqs = h5["frequency/frequencies_hz"][()] if "frequency/frequencies_hz" in h5 else None
    magnitude = _plot_link_grid(
        cfr,
        output_dir / f"cfr_lines_magnitude.{config.format}",
        selection,
        config,
        lambda axis, bs, ue: _draw_cfr_lines(
            axis,
            cfr[_link_index_pair(selection, bs, ue)],
            freqs,
            value_kind="magnitude",
        ),
        "CFR magnitude lines",
    )
    phase = _plot_link_grid(
        cfr,
        output_dir / f"cfr_lines_phase.{config.format}",
        selection,
        config,
        lambda axis, bs, ue: _draw_cfr_lines(
            axis,
            cfr[_link_index_pair(selection, bs, ue)],
            freqs,
            value_kind="phase",
        ),
        "CFR phase lines",
    )
    return [magnitude, phase]


def _plot_cfr_heatmap(
    h5: h5py.File,
    output_dir: Path,
    selection: dict[str, Any],
    config: VisualizationRunConfig,
    **_: Any,
) -> list[Path]:
    _require(h5, ("channel/truth/cfr",))
    cfr = h5["channel/truth/cfr"]
    magnitude = _plot_link_grid(
        cfr,
        output_dir / f"cfr_heatmap_magnitude.{config.format}",
        selection,
        config,
        lambda axis, bs, ue: _draw_ant_subcarrier_heatmap(
            axis,
            cfr[_link_index_pair(selection, bs, ue)],
            "|CFR| [dB]",
            value_kind="magnitude_db",
        ),
        "CFR antenna-pair heatmaps",
    )
    phase = _plot_link_grid(
        cfr,
        output_dir / f"cfr_heatmap_phase.{config.format}",
        selection,
        config,
        lambda axis, bs, ue: _draw_ant_subcarrier_heatmap(
            axis,
            cfr[_link_index_pair(selection, bs, ue)],
            "CFR phase [rad]",
            value_kind="phase",
        ),
        "CFR phase antenna-pair heatmaps",
    )
    return [magnitude, phase]


def _plot_cfr_error(
    h5: h5py.File,
    output_dir: Path,
    selection: dict[str, Any],
    config: VisualizationRunConfig,
    **_: Any,
) -> list[Path]:
    _require(h5, ("channel/truth/cfr", "observation/cfr_est"))
    truth = h5["channel/truth/cfr"]
    estimate = h5["observation/cfr_est"]

    def draw_magnitude(axis: Any, bs: int, ue: int) -> None:
        tx_idx, rx_idx = _link_index_pair(selection, bs, ue)
        amplitude_error_db = (
            20.0 * np.log10(np.abs(estimate[0, tx_idx, rx_idx]) + _EPS)
            - 20.0 * np.log10(np.abs(truth[tx_idx, rx_idx]) + _EPS)
        )
        _draw_ant_subcarrier_heatmap(
            axis, amplitude_error_db, "CFR magnitude error [dB]", value_kind="real"
        )

    def draw_phase(axis: Any, bs: int, ue: int) -> None:
        tx_idx, rx_idx = _link_index_pair(selection, bs, ue)
        phase_error = _wrap_phase(
            np.angle(estimate[0, tx_idx, rx_idx]) - np.angle(truth[tx_idx, rx_idx])
        )
        _draw_ant_subcarrier_heatmap(
            axis, phase_error, "CFR phase error [rad]", value_kind="real"
        )

    magnitude = _plot_link_grid(
        truth,
        output_dir / f"cfr_error_magnitude.{config.format}",
        selection,
        config,
        draw_magnitude,
        "CFR magnitude estimate error",
    )
    phase = _plot_link_grid(
        truth,
        output_dir / f"cfr_error_phase.{config.format}",
        selection,
        config,
        draw_phase,
        "CFR phase estimate error",
    )
    return [magnitude, phase]


def _plot_waveform_grid(
    h5: h5py.File,
    output_dir: Path,
    selection: dict[str, Any],
    config: VisualizationRunConfig,
    **_: Any,
) -> Path:
    _require(h5, ("waveform/rx_grid",))
    rx_grid = h5["waveform/rx_grid"]

    def draw(axis: Any, bs: int, ue: int) -> None:
        tx_idx, rx_idx = _link_index_pair(selection, bs, ue)
        data = rx_grid[0, tx_idx, rx_idx]
        power = 10.0 * np.log10(np.mean(np.abs(data) ** 2, axis=0) + _EPS)
        image = axis.imshow(
            power.T,
            aspect="auto",
            origin="lower",
            interpolation="none",
        )
        axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        axis.set_xlabel("OFDM symbol")
        axis.set_ylabel("subcarrier")

    return _plot_link_grid(
        rx_grid,
        output_dir / f"waveform_rx_grid.{config.format}",
        selection,
        config,
        draw,
        "NR PUSCH RX grid power",
    )


def _plot_aoa(
    h5: h5py.File,
    output_dir: Path,
    selection: dict[str, Any],
    config: VisualizationRunConfig,
    **_: Any,
) -> Path:
    _require(h5, ("derived/first_path_aoa_azimuth_rad", "derived/first_path_aoa_zenith_rad"))
    figure, axis = plt.subplots(figsize=(7, 5))
    for label, az_path, ze_path in (
        ("first", "derived/first_path_aoa_azimuth_rad", "derived/first_path_aoa_zenith_rad"),
        ("strongest", "derived/strongest_aoa_azimuth_rad", "derived/strongest_aoa_zenith_rad"),
        ("los", "derived/los_aoa_azimuth_rad", "derived/los_aoa_zenith_rad"),
    ):
        if az_path not in h5 or ze_path not in h5:
            continue
        az = _link_matrix_for_bs_ue(h5[az_path][()], selection).ravel()
        ze = _link_matrix_for_bs_ue(h5[ze_path][()], selection).ravel()
        mask = np.isfinite(az) & np.isfinite(ze)
        if np.any(mask):
            axis.scatter(az[mask], ze[mask], label=label, alpha=0.75)
    axis.set_xlabel("azimuth [rad]")
    axis.set_ylabel("zenith [rad]")
    axis.set_title("AoA Labels")
    axis.legend(loc="best")
    axis.grid(True, alpha=0.3)
    return _save_figure(figure, output_dir / f"aoa_labels.{config.format}", config)


def _plot_nlos_paths(
    h5: h5py.File,
    output_dir: Path,
    selection: dict[str, Any],
    config: VisualizationRunConfig,
    **_: Any,
) -> Path:
    _require(
        h5,
        (
            "paths/nlos_truth/valid",
            "paths/nlos_truth/delay_s",
            "paths/nlos_truth/path_power_db",
        ),
    )
    valid = h5["paths/nlos_truth/valid"][()]
    delays = h5["paths/nlos_truth/delay_s"][()]
    powers = h5["paths/nlos_truth/path_power_db"][()]
    azimuth = (
        h5["paths/nlos_truth/aoa_azimuth_rad"][()]
        if "paths/nlos_truth/aoa_azimuth_rad" in h5
        else None
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    selected_paths = _selected_path_arrays(selection, valid, delays, powers, azimuth)
    mask = selected_paths["valid"]
    delay_sel = selected_paths["delay_s"]
    power_sel = selected_paths["path_power_db"]
    axes[0].scatter(delay_sel[mask] * 1e9, power_sel[mask], s=8, alpha=0.6)
    axes[0].set_xlabel("delay [ns]")
    axes[0].set_ylabel("power [dB]")
    axes[0].set_title("NLoS delay-power")
    if azimuth is not None:
        az_sel = selected_paths["aoa_azimuth_rad"]
        axes[1].hist(az_sel[mask & np.isfinite(az_sel)], bins=36)
    axes[1].set_xlabel("AoA azimuth [rad]")
    axes[1].set_ylabel("count")
    axes[1].set_title("NLoS AoA azimuth")
    return _save_figure(figure, output_dir / f"nlos_paths.{config.format}", config)


def _plot_spatial_spectrum(
    h5: h5py.File,
    output_dir: Path,
    selection: dict[str, Any],
    config: VisualizationRunConfig,
    **_: Any,
) -> list[Path]:
    _require(h5, ("array/angle_grid_rad",))
    angle_grid = np.asarray(h5["array/angle_grid_rad"][()])
    candidates = (
        (
            "aoa_heatmap_label",
            "array/aoa_heatmap_label",
            "AoA heatmap label / aoa_heatmap_label",
        ),
        (
            "truth",
            "array/spatial_spectrum_truth",
            "Bartlett spectrum from truth CFR / spatial_spectrum_truth",
        ),
        (
            "cfr_est",
            "array/spatial_spectrum_cfr_est",
            "Bartlett spectrum from estimated CFR / spatial_spectrum_cfr_est",
        ),
        (
            "observation",
            "array/spatial_spectrum_observation",
            "Bartlett spectrum from RX grid / spatial_spectrum_observation",
        ),
    )
    generated: list[Path] = []
    for suffix, dataset_path, title in candidates:
        if dataset_path not in h5:
            continue
        data = h5[dataset_path]
        row_limits = _spatial_spectrum_row_limits(data, selection)

        def draw(
            axis: Any,
            bs: int,
            ue: int,
            *,
            source: h5py.Dataset = data,
            limits: dict[int, tuple[float, float]] = row_limits,
        ) -> None:
            tx_idx, rx_idx = _link_index_pair(selection, bs, ue)
            spectrum = source[0, tx_idx, rx_idx]
            vmin, vmax = limits[int(ue)]
            image = axis.imshow(
                spectrum,
                aspect="auto",
                origin="lower",
                interpolation="none",
                vmin=vmin,
                vmax=vmax,
            )
            axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
            axis.set_xlabel("azimuth bin")
            axis.set_ylabel("zenith bin")

        generated.extend(
            (
                _plot_link_grid(
                    data,
                    output_dir / f"spatial_spectrum_{suffix}.{config.format}",
                    selection,
                    config,
                    draw,
                    title,
                ),
                _plot_spatial_spectrum_polar_grid(
                    data,
                    angle_grid,
                    output_dir / f"spatial_spectrum_{suffix}_polar.{config.format}",
                    selection,
                    config,
                    f"{title} polar hemispheres",
                    row_limits,
                ),
            )
        )
    if not generated:
        raise _SkipPlot("no spatial spectrum dataset present")
    return generated


def _plot_spatial_spectrum_polar_grid(
    dataset: h5py.Dataset,
    angle_grid: np.ndarray,
    output_path: Path,
    selection: dict[str, Any],
    config: VisualizationRunConfig,
    title: str,
    row_limits: dict[int, tuple[float, float]],
) -> Path:
    bs = selection["bs_indices"]
    ue = selection["ue_indices"]
    rows = max(len(ue), 1)
    cols = max(len(bs), 1)
    spectra: dict[tuple[int, int], np.ndarray] = {}
    for ue_idx in ue:
        for bs_idx in bs:
            tx_idx, rx_idx = _link_index_pair(selection, int(bs_idx), int(ue_idx))
            spectrum = np.asarray(dataset[0, tx_idx, rx_idx], dtype=np.float32)
            spectra[(int(ue_idx), int(bs_idx))] = spectrum

    figure, axes = plt.subplots(
        rows,
        cols * 2,
        figsize=(5.2 * cols, 3.2 * rows),
        squeeze=False,
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )
    for row, ue_idx in enumerate(ue):
        for col, bs_idx in enumerate(bs):
            upper_axis = axes[row, col * 2]
            lower_axis = axes[row, col * 2 + 1]
            spectrum = spectra[(int(ue_idx), int(bs_idx))]
            vmin, vmax = row_limits[int(ue_idx)]
            _draw_spatial_spectrum_polar_pair(
                upper_axis,
                lower_axis,
                spectrum,
                angle_grid,
                vmin=vmin,
                vmax=vmax,
            )
            upper_axis.set_title(f"UE {ue_idx} - BS {bs_idx}\nupper", fontsize=8)
            lower_axis.set_title(f"UE {ue_idx} - BS {bs_idx}\nlower", fontsize=8)
    figure.suptitle(title)
    return _save_figure(figure, output_path, config)


def _draw_spatial_spectrum_polar_pair(
    upper_axis: Any,
    lower_axis: Any,
    spectrum: np.ndarray,
    angle_grid: np.ndarray,
    *,
    vmin: float,
    vmax: float,
) -> Any:
    zenith = np.asarray(angle_grid[:, 0, 0], dtype=np.float64)
    azimuth = np.asarray(angle_grid[0, :, 1], dtype=np.float64)
    zenith_mid = 0.5 * (float(np.nanmin(zenith)) + float(np.nanmax(zenith)))

    upper_mask = zenith <= zenith_mid + 1e-12
    lower_mask = zenith >= zenith_mid - 1e-12
    azimuth_edges = _grid_edges(azimuth)

    upper_values = np.asarray(spectrum[upper_mask, :], dtype=np.float32)
    upper_radius = zenith[upper_mask]
    mappable = _draw_spatial_spectrum_polar_half(
        upper_axis,
        azimuth_edges,
        upper_radius,
        upper_values,
        title_label="r=zenith",
        vmin=vmin,
        vmax=vmax,
    )

    lower_values = np.asarray(spectrum[lower_mask, :], dtype=np.float32)[::-1, :]
    lower_radius = (float(np.nanmax(zenith)) - zenith[lower_mask])[::-1]
    _draw_spatial_spectrum_polar_half(
        lower_axis,
        azimuth_edges,
        lower_radius,
        lower_values,
        title_label="r=pi-zenith",
        vmin=vmin,
        vmax=vmax,
    )
    return mappable


def _draw_spatial_spectrum_polar_half(
    axis: Any,
    azimuth_edges: np.ndarray,
    radius_centers: np.ndarray,
    values: np.ndarray,
    *,
    title_label: str,
    vmin: float,
    vmax: float,
) -> Any:
    radius_edges = _grid_edges(radius_centers)
    mappable = axis.pcolormesh(
        azimuth_edges,
        radius_edges,
        values,
        shading="flat",
        vmin=vmin,
        vmax=vmax,
    )
    axis.set_theta_zero_location("E")
    axis.set_theta_direction(1)
    axis.set_ylim(max(0.0, float(radius_edges[0])), float(radius_edges[-1]))
    axis.set_thetagrids([0, 90, 180, 270], labels=["0", "90", "180", "270"])
    axis.set_rticks([0.0, np.pi / 4.0, np.pi / 2.0])
    axis.set_yticklabels(["0", "45", "90"])
    axis.tick_params(labelsize=7, pad=1)
    axis.set_xlabel(title_label, fontsize=7, labelpad=-2)
    axis.grid(True, alpha=0.3)
    return mappable


def _plot_nmse_snr(
    h5: h5py.File,
    output_dir: Path,
    selection: dict[str, Any],
    config: VisualizationRunConfig,
    **_: Any,
) -> Path:
    _require(h5, ("observation/snr_db", "evaluation/nmse_db"))
    snr = h5["observation/snr_db"][()].ravel()
    nmse = h5["evaluation/nmse_db"][()].ravel()
    mask = np.isfinite(snr) & np.isfinite(nmse)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(snr[mask], nmse[mask], s=8, alpha=0.5)
    axes[0].set_xlabel("SNR [dB]")
    axes[0].set_ylabel("NMSE [dB]")
    axes[0].set_title("NMSE vs SNR")
    axes[1].hist(nmse[mask], bins=40)
    axes[1].set_xlabel("NMSE [dB]")
    axes[1].set_ylabel("count")
    axes[1].set_title("NMSE distribution")
    return _save_figure(figure, output_dir / f"nmse_snr.{config.format}", config)


def _plot_path_samples(
    h5: h5py.File,
    output_dir: Path,
    selection: dict[str, Any],
    config: VisualizationRunConfig,
    **_: Any,
) -> Path:
    _require(
        h5,
        (
            "paths/samples/vertices_m",
            "paths/samples/vertex_count",
            "paths/samples/sampled_link_indices",
        ),
    )
    vertices = h5["paths/samples/vertices_m"][()]
    counts = h5["paths/samples/vertex_count"][()]
    links = h5["paths/samples/sampled_link_indices"][()]
    bs_idx, ue_idx, tx_idx, rx_idx = _first_selected_link_pair(selection)
    figure = plt.figure(figsize=(7, 5))
    axis = figure.add_subplot(111, projection="3d")
    plotted = 0
    for sample in range(vertices.shape[0]):
        if (int(links[sample, 0]), int(links[sample, 1])) != (tx_idx, rx_idx):
            continue
        for path_idx in range(vertices.shape[1]):
            count = int(counts[sample, path_idx])
            if count > 1:
                points = vertices[sample, path_idx, :count]
                axis.plot(points[:, 0], points[:, 1], points[:, 2], alpha=0.65)
                plotted += 1
    if plotted == 0:
        raise _SkipPlot(f"no sampled paths for UE {ue_idx} - BS {bs_idx}")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    axis.set_title(f"Sampled Path Geometry UE {ue_idx} - BS {bs_idx}")
    return _save_figure(figure, output_dir / f"path_samples.{config.format}", config)


def _plot_full_summary(
    h5: h5py.File,
    output_dir: Path,
    selection: dict[str, Any],
    config: VisualizationRunConfig,
    **_: Any,
) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    axes = axes.ravel()
    if "derived/link_valid_mask" in h5:
        valid = h5["derived/link_valid_mask"][()].astype(np.float32)
        axes[0].imshow(valid, aspect="auto", origin="lower", interpolation="none")
        axes[0].set_title("link_valid_mask")
    if "derived/path_count" in h5:
        axes[1].hist(h5["derived/path_count"][()].ravel(), bins=40)
        axes[1].set_title("path_count distribution")
    if "evaluation/nmse_db" in h5:
        axes[2].hist(h5["evaluation/nmse_db"][()].ravel(), bins=40)
        axes[2].set_title("NMSE distribution")
    if "channel/truth/path_power_db" in h5:
        axes[3].hist(h5["channel/truth/path_power_db"][()].ravel(), bins=40)
        axes[3].set_title("path power distribution")
    return _save_figure(figure, output_dir / f"full_summary.{config.format}", config)


def _plot_dataset_preview(
    h5: h5py.File,
    output_dir: Path,
    selection: dict[str, Any],
    config: VisualizationRunConfig,
    *,
    dataset_path: str | None,
    plot_type: str,
) -> Path:
    if not dataset_path:
        raise _SkipPlot("dataset_path is required for dataset mode")
    path = dataset_path.lstrip("/")
    if path not in h5:
        raise _SkipPlot(f"dataset not found: /{path}")
    values = np.asarray(h5[path][()])
    if values.dtype.kind == "c":
        values = np.abs(values)
    values = np.squeeze(values)
    figure, axis = plt.subplots(figsize=(7, 5))
    kind = plot_type
    if kind == "auto":
        kind = "hist" if values.ndim > 2 else "heatmap" if values.ndim == 2 else "line"
    if kind == "hist":
        axis.hist(values[np.isfinite(values)].ravel(), bins=50)
    elif kind == "heatmap":
        view = _last_two_dim_view(values)
        image = axis.imshow(view, aspect="auto", origin="lower", interpolation="none")
        figure.colorbar(image, ax=axis)
    elif kind == "line":
        axis.plot(np.ravel(values))
    else:
        raise _SkipPlot(f"unsupported plot_type {plot_type!r}")
    axis.set_title(f"/{path}")
    return _save_figure(figure, output_dir / f"dataset_preview.{config.format}", config)


def _plot_link_grid(
    dataset: h5py.Dataset,
    output_path: Path,
    selection: dict[str, Any],
    config: VisualizationRunConfig,
    draw_fn: Any,
    title: str,
) -> Path:
    bs = selection["bs_indices"]
    ue = selection["ue_indices"]
    rows = max(len(ue), 1)
    cols = max(len(bs), 1)
    figure, axes = plt.subplots(rows, cols, figsize=(4.1 * cols, 3.0 * rows), squeeze=False)
    for row, ue_idx in enumerate(ue):
        for col, bs_idx in enumerate(bs):
            axis = axes[row, col]
            draw_fn(axis, int(bs_idx), int(ue_idx))
            axis.set_title(f"UE {ue_idx} - BS {bs_idx}", fontsize=9)
    figure.suptitle(title)
    return _save_figure(figure, output_path, config)


def _draw_cfr_lines(
    axis: Any,
    link_cfr: np.ndarray,
    freqs: np.ndarray | None,
    *,
    value_kind: str,
) -> None:
    _ = freqs
    subcarrier = link_cfr.shape[-1]
    x = np.arange(subcarrier)
    if value_kind == "magnitude":
        values = 20.0 * np.log10(np.abs(link_cfr.reshape(-1, subcarrier)) + _EPS)
        axis.set_ylabel("|H| [dB]")
    elif value_kind == "phase":
        values = np.angle(link_cfr.reshape(-1, subcarrier))
        axis.set_ylabel("phase [rad]")
    else:
        raise ValueError(f"unsupported CFR line value kind: {value_kind}")
    for row in values:
        axis.plot(x, row, alpha=0.55, linewidth=0.8)
    axis.set_xlabel("subcarrier")


def _draw_ant_subcarrier_heatmap(
    axis: Any,
    value: np.ndarray,
    label: str,
    *,
    value_kind: str,
) -> None:
    subcarrier = value.shape[-1]
    if value_kind == "magnitude_db":
        heatmap = 20.0 * np.log10(np.abs(value.reshape(-1, subcarrier)) + _EPS)
    elif value_kind == "phase":
        heatmap = np.angle(value.reshape(-1, subcarrier))
    elif value_kind == "real":
        heatmap = np.asarray(value).reshape(-1, subcarrier)
    else:
        raise ValueError(f"unsupported antenna-subcarrier heatmap value kind: {value_kind}")
    image = axis.imshow(
        heatmap.T,
        aspect="auto",
        origin="lower",
        interpolation="none",
    )
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label=label)
    axis.set_xlabel("antenna pair")
    axis.set_ylabel("subcarrier")


def _wrap_phase(value: np.ndarray) -> np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def _spatial_spectrum_row_limits(
    dataset: h5py.Dataset,
    selection: dict[str, Any],
) -> dict[int, tuple[float, float]]:
    """Return one color scale per selected UE across all selected BS."""

    bs_indices = [int(value) for value in selection["bs_indices"]]
    limits: dict[int, tuple[float, float]] = {}
    for ue_idx in selection["ue_indices"]:
        finite_chunks: list[np.ndarray] = []
        for bs_idx in bs_indices:
            tx_idx, rx_idx = _link_index_pair(selection, int(bs_idx), int(ue_idx))
            values = np.asarray(dataset[0, tx_idx, rx_idx], dtype=np.float32)
            finite = values[np.isfinite(values)]
            if finite.size:
                finite_chunks.append(finite)
        if not finite_chunks:
            limits[int(ue_idx)] = (0.0, 1.0)
            continue
        row_values = np.concatenate(finite_chunks)
        vmin = float(np.min(row_values))
        vmax = float(np.max(row_values))
        limits[int(ue_idx)] = _stable_color_limits(vmin, vmax)
    return limits


def _stable_color_limits(vmin: float, vmax: float) -> tuple[float, float]:
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return (0.0, 1.0)
    if vmax > vmin:
        return (vmin, vmax)
    if vmax > 0.0:
        return (0.0, vmax)
    if vmin < 0.0:
        return (vmin, 0.0)
    return (0.0, 1.0)


def _grid_edges(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=np.float64)
    if centers.size == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5], dtype=np.float64)
    edges = np.empty(centers.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges


def _as_path_list(value: Path | list[Path]) -> list[Path]:
    if isinstance(value, list):
        return value
    return [value]


def _last_two_dim_view(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    while array.ndim > 2:
        array = array[0]
    if array.ndim == 0:
        return array.reshape(1, 1)
    if array.ndim == 1:
        return array.reshape(1, -1)
    return array


def _selected_path_arrays(
    selection: dict[str, Any],
    valid: np.ndarray,
    delays: np.ndarray,
    powers: np.ndarray,
    azimuth: np.ndarray | None,
) -> dict[str, np.ndarray]:
    pairs = _selected_link_pairs(selection)
    if not pairs:
        empty_bool = np.zeros((0,), dtype=np.bool_)
        empty_float = np.zeros((0,), dtype=np.float32)
        return {
            "valid": empty_bool,
            "delay_s": empty_float,
            "path_power_db": empty_float,
            "aoa_azimuth_rad": empty_float,
        }
    valid_chunks = []
    delay_chunks = []
    power_chunks = []
    azimuth_chunks = []
    for tx_idx, rx_idx in pairs:
        valid_chunks.append(np.asarray(valid[tx_idx, rx_idx]).ravel())
        delay_chunks.append(np.asarray(delays[tx_idx, rx_idx]).ravel())
        power_chunks.append(np.asarray(powers[tx_idx, rx_idx]).ravel())
        if azimuth is not None:
            azimuth_chunks.append(np.asarray(azimuth[tx_idx, rx_idx]).ravel())
    result = {
        "valid": np.concatenate(valid_chunks).astype(np.bool_, copy=False),
        "delay_s": np.concatenate(delay_chunks).astype(np.float32, copy=False),
        "path_power_db": np.concatenate(power_chunks).astype(np.float32, copy=False),
        "aoa_azimuth_rad": np.zeros((0,), dtype=np.float32),
    }
    if azimuth_chunks:
        result["aoa_azimuth_rad"] = np.concatenate(azimuth_chunks).astype(
            np.float32,
            copy=False,
        )
    return result


def _nlos_selection_index(
    valid: np.ndarray,
    bs: list[int],
    ue: list[int],
) -> tuple[np.ndarray, ...]:
    return np.ix_(
        bs,
        ue,
        range(valid.shape[2]),
        range(valid.shape[3]),
        range(valid.shape[4]),
    )


def _save_figure(figure: Any, path: Path, config: VisualizationRunConfig) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not figure.get_constrained_layout():
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="This figure includes Axes that are not compatible with tight_layout.*",
                category=UserWarning,
            )
            figure.tight_layout()
    figure.savefig(path, dpi=config.dpi)
    plt.close(figure)
    return path


def _require(h5: h5py.File, paths: tuple[str, ...]) -> None:
    missing = [path for path in paths if path not in h5]
    if missing:
        raise _SkipPlot(f"missing required dataset(s): {', '.join('/' + p for p in missing)}")


def _clip_unique(values: list[int], upper: int) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        item = int(value)
        if 0 <= item < upper and item not in seen:
            seen.add(item)
            out.append(item)
    return out


class _SkipPlot(RuntimeError):
    """Raised internally when a plot cannot be generated for this HDF5 file."""
