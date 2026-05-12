"""Configuration-driven visualization reports for simulation HDF5 files."""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np

from sionna_measurement_sim.visualization.config import VisualizationRunConfig

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
                path = _dispatch_plot(
                    h5,
                    output_dir,
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

    num_bs = int(h5["topology/tx_positions_m"].shape[0])
    num_ue = int(h5["topology/rx_positions_m"].shape[0])
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
        ue = _sample_ue_indices(h5, config, bs, num_ue)
    if not ue and num_ue:
        ue = [0]

    return {
        "bs_indices": [int(value) for value in bs],
        "ue_indices": [int(value) for value in ue],
    }


def _sample_ue_indices(
    h5: h5py.File,
    config: VisualizationRunConfig,
    bs: list[int],
    num_ue: int,
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
        candidates = np.flatnonzero(np.any(valid[np.asarray(bs), :], axis=0))
    else:
        candidates = np.arange(num_ue)

    selected: list[int] = []
    if candidates.size:
        count = min(target, candidates.size)
        if (
            config.sample_policy == "spatially_spread_valid_links"
            and "topology/rx_positions_m" in h5
        ):
            selected.extend(_select_spatially_spread_ues(h5, candidates, count))
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
) -> list[int]:
    positions = np.asarray(h5["topology/rx_positions_m"][candidates, :2], dtype=np.float64)
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


def _plot_topology(
    h5: h5py.File,
    output_dir: Path,
    selection: dict[str, Any],
    config: VisualizationRunConfig,
    **_: Any,
) -> Path:
    tx = h5["topology/tx_positions_m"][()]
    rx = h5["topology/rx_positions_m"][()]
    bs = selection["bs_indices"]
    ue = selection["ue_indices"]
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(rx[:, 0], rx[:, 1], s=8, alpha=0.25, label="UE all")
    axis.scatter(tx[:, 0], tx[:, 1], marker="^", s=70, color="tab:red", label="BS all")
    if ue:
        axis.scatter(rx[ue, 0], rx[ue, 1], s=45, color="tab:blue", label="UE selected")
    if bs:
        axis.scatter(tx[bs, 0], tx[bs, 1], marker="^", s=110, color="black", label="BS selected")
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
            np.asarray(data)[np.ix_(bs, ue)],
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
        lambda axis, bs, ue: _draw_cfr_lines(axis, cfr[bs, ue], freqs, value_kind="magnitude"),
        "CFR magnitude lines",
    )
    phase = _plot_link_grid(
        cfr,
        output_dir / f"cfr_lines_phase.{config.format}",
        selection,
        config,
        lambda axis, bs, ue: _draw_cfr_lines(axis, cfr[bs, ue], freqs, value_kind="phase"),
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
            axis, cfr[bs, ue], "|CFR| [dB]", value_kind="magnitude_db"
        ),
        "CFR antenna-pair heatmaps",
    )
    phase = _plot_link_grid(
        cfr,
        output_dir / f"cfr_heatmap_phase.{config.format}",
        selection,
        config,
        lambda axis, bs, ue: _draw_ant_subcarrier_heatmap(
            axis, cfr[bs, ue], "CFR phase [rad]", value_kind="phase"
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
        amplitude_error_db = (
            20.0 * np.log10(np.abs(estimate[0, bs, ue]) + _EPS)
            - 20.0 * np.log10(np.abs(truth[bs, ue]) + _EPS)
        )
        _draw_ant_subcarrier_heatmap(
            axis, amplitude_error_db, "CFR magnitude error [dB]", value_kind="real"
        )

    def draw_phase(axis: Any, bs: int, ue: int) -> None:
        phase_error = _wrap_phase(np.angle(estimate[0, bs, ue]) - np.angle(truth[bs, ue]))
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
        data = rx_grid[0, ue, bs]
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
    bs = selection["bs_indices"]
    ue = selection["ue_indices"]
    figure, axis = plt.subplots(figsize=(7, 5))
    for label, az_path, ze_path in (
        ("first", "derived/first_path_aoa_azimuth_rad", "derived/first_path_aoa_zenith_rad"),
        ("strongest", "derived/strongest_aoa_azimuth_rad", "derived/strongest_aoa_zenith_rad"),
        ("los", "derived/los_aoa_azimuth_rad", "derived/los_aoa_zenith_rad"),
    ):
        if az_path not in h5 or ze_path not in h5:
            continue
        az = h5[az_path][()][np.ix_(bs, ue)].ravel()
        ze = h5[ze_path][()][np.ix_(bs, ue)].ravel()
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
    bs = selection["bs_indices"]
    ue = selection["ue_indices"]
    valid = h5["paths/nlos_truth/valid"][()]
    delays = h5["paths/nlos_truth/delay_s"][()]
    powers = h5["paths/nlos_truth/path_power_db"][()]
    azimuth = (
        h5["paths/nlos_truth/aoa_azimuth_rad"][()]
        if "paths/nlos_truth/aoa_azimuth_rad" in h5
        else None
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    nlos_index = _nlos_selection_index(valid, bs, ue)
    mask = valid[nlos_index]
    delay_sel = delays[nlos_index]
    power_sel = powers[nlos_index]
    axes[0].scatter(delay_sel[mask] * 1e9, power_sel[mask], s=8, alpha=0.6)
    axes[0].set_xlabel("delay [ns]")
    axes[0].set_ylabel("power [dB]")
    axes[0].set_title("NLoS delay-power")
    if azimuth is not None:
        az_sel = azimuth[nlos_index]
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
            "label",
            "array/spatial_spectrum_label",
            "AoA heatmap label / spatial_spectrum_label",
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
            spectrum = source[0, ue, bs]
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
            spectrum = np.asarray(dataset[0, ue_idx, bs_idx], dtype=np.float32)
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
    bs_set = set(selection["bs_indices"])
    ue_set = set(selection["ue_indices"])
    figure = plt.figure(figsize=(7, 5))
    axis = figure.add_subplot(111, projection="3d")
    plotted = 0
    for sample in range(vertices.shape[0]):
        if (int(links[sample, 0]) not in bs_set) or (int(links[sample, 1]) not in ue_set):
            continue
        for path_idx in range(vertices.shape[1]):
            count = int(counts[sample, path_idx])
            if count > 1:
                points = vertices[sample, path_idx, :count]
                axis.plot(points[:, 0], points[:, 1], points[:, 2], alpha=0.65)
                plotted += 1
    if plotted == 0:
        raise _SkipPlot("no sampled paths for selected links")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    axis.set_title("Sampled Path Geometry")
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
            values = np.asarray(dataset[0, int(ue_idx), bs_idx], dtype=np.float32)
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
