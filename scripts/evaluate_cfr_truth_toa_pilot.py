"""Evaluate ToA/range estimates from CFR truth against first-path RT truth.

This pilot intentionally estimates ToA only from `/channel/truth/cfr`.
Derived labels and positions are read only after estimation to compute errors.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from sionna_measurement_sim.domain.derived import SPEED_OF_LIGHT_MPS

DEFAULT_POWER_BINS_DB = (-np.inf, -117.0, -100.0, -80.0, -60.0, np.inf)


@dataclass(frozen=True)
class PdpEstimate:
    toa_s: float
    range_m: float
    success: bool
    selected_delay_bin: int
    peak_power_linear: float
    peak_snr_db: float


@dataclass(frozen=True)
class PhaseSlopeEstimate:
    toa_s: float
    range_m: float
    success: bool
    fit_residual_rad: float


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate RTT-like one-way range from CFR truth and compare against "
            "first-path RT truth labels."
        )
    )
    parser.add_argument("--input-root", required=True, help="Scene-index output root or run dir.")
    parser.add_argument("--output-dir", required=True, help="Directory for CSV/JSON outputs.")
    parser.add_argument("--pdp-oversampling-factor", type=int, default=8)
    parser.add_argument("--pdp-window", choices=["hann", "rect"], default="hann")
    parser.add_argument("--pdp-relative-threshold-db", type=float, default=-12.0)
    parser.add_argument("--pdp-min-peak-snr-db", type=float, default=6.0)
    parser.add_argument("--pdp-batch-size", type=int, default=32)
    parser.add_argument("--phase-min-mean-power", type=float, default=1.0e-12)
    parser.add_argument(
        "--include-invalid-gt",
        action="store_true",
        help="Also write rows whose first-path GT is invalid. They are excluded from metrics.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = discover_completed_run_dirs(input_root)
    if not run_dirs:
        raise SystemExit(f"No completed run dirs found under {input_root}")

    details: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        details.extend(
            process_run_dir(
                run_dir,
                pdp_oversampling_factor=args.pdp_oversampling_factor,
                pdp_window=args.pdp_window,
                pdp_relative_threshold_db=args.pdp_relative_threshold_db,
                pdp_min_peak_snr_db=args.pdp_min_peak_snr_db,
                pdp_batch_size=args.pdp_batch_size,
                phase_min_mean_power=args.phase_min_mean_power,
                include_invalid_gt=args.include_invalid_gt,
            )
        )

    summary = build_summary(
        details,
        input_root=input_root,
        run_dirs=run_dirs,
        config={
            "pdp_oversampling_factor": args.pdp_oversampling_factor,
            "pdp_window": args.pdp_window,
            "pdp_relative_threshold_db": args.pdp_relative_threshold_db,
            "pdp_min_peak_snr_db": args.pdp_min_peak_snr_db,
            "pdp_batch_size": args.pdp_batch_size,
            "phase_min_mean_power": args.phase_min_mean_power,
            "power_bins_db": [format_edge(value) for value in DEFAULT_POWER_BINS_DB],
        },
    )

    write_details_csv(output_dir / "toa_link_details.csv", details)
    write_details_jsonl(output_dir / "toa_link_details.jsonl", details)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(render_summary_markdown(summary), encoding="utf-8")
    print(f"Processed {len(run_dirs)} runs, {len(details)} detail rows")
    print(f"Summary: {output_dir / 'summary.json'}")
    print(f"CSV: {output_dir / 'toa_link_details.csv'}")
    return 0


def discover_completed_run_dirs(input_root: Path) -> list[Path]:
    scene_index_manifest = input_root / "scene_index_run_manifest.jsonl"
    if scene_index_manifest.exists():
        run_dirs: list[Path] = []
        for line in scene_index_manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "completed":
                continue
            run_dirs.append(Path(str(row["output_dir"])))
        return run_dirs
    if (input_root / "manifest" / "manifest.json").exists():
        return [input_root]
    runs_dir = input_root / "runs"
    if runs_dir.exists():
        return sorted(
            path
            for path in runs_dir.iterdir()
            if (path / "manifest/manifest.json").exists()
        )
    return []


def process_run_dir(
    run_dir: Path,
    *,
    pdp_oversampling_factor: int,
    pdp_window: str,
    pdp_relative_threshold_db: float,
    pdp_min_peak_snr_db: float,
    pdp_batch_size: int,
    phase_min_mean_power: float,
    include_invalid_gt: bool,
) -> list[dict[str, Any]]:
    manifest_path = run_dir / "manifest" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scene_id = str(manifest.get("scene_id") or run_dir.name)
    rows: list[dict[str, Any]] = []
    for item in manifest.get("results", []):
        h5_path = resolve_manifest_artifact(run_dir, Path(str(item["result_h5"])))
        shard_id = str(item.get("shard_id", h5_path.stem))
        with h5py.File(h5_path, "r") as h5:
            frequencies_hz = np.asarray(h5["frequency/frequencies_hz"][()], dtype=np.float64)
            cfr = np.asarray(h5["channel/truth/cfr"][()], dtype=np.complex64)
            first_range = np.asarray(
                h5["derived/first_path_propagation_range_m"][()],
                dtype=np.float64,
            )
            first_delay = np.asarray(h5["derived/first_path_delay_s"][()], dtype=np.float64)
            link_valid = np.asarray(h5["derived/link_valid_mask"][()], dtype=np.bool_)
            los_flag = np.asarray(h5["derived/los_flag"][()], dtype=np.bool_)
            nlos_flag = np.asarray(h5["derived/nlos_flag"][()], dtype=np.bool_)
            path_power_db = np.asarray(h5["derived/path_power_db"][()], dtype=np.float64)
            global_tx = read_global_indices(h5, "tx", cfr.shape[0])
            global_rx = read_global_indices(h5, "rx", cfr.shape[1])

            local_links = flatten_links(cfr)
            pdp = estimate_pdp_for_links(
                local_links,
                frequencies_hz,
                oversampling_factor=pdp_oversampling_factor,
                window_name=pdp_window,
                relative_threshold_db=pdp_relative_threshold_db,
                min_peak_snr_db=pdp_min_peak_snr_db,
                batch_size=pdp_batch_size,
            )
            phase = [
                estimate_phase_slope_one(link, frequencies_hz, min_mean_power=phase_min_mean_power)
                for link in local_links
            ]
            csi_power_db = cfr_power_db(local_links)

            rx_count = cfr.shape[1]
            for flat_index, (pdp_est, phase_est) in enumerate(zip(pdp, phase, strict=True)):
                local_tx = flat_index // rx_count
                local_rx = flat_index % rx_count
                valid_gt = bool(
                    link_valid[local_tx, local_rx]
                    and np.isfinite(first_range[local_tx, local_rx])
                )
                if not valid_gt and not include_invalid_gt:
                    continue
                gt_range_m = value_or_none(first_range[local_tx, local_rx])
                gt_delay_s = value_or_none(first_delay[local_tx, local_rx])
                row = {
                    "scene_id": scene_id,
                    "run_dir": run_dir.as_posix(),
                    "h5_path": h5_path.as_posix(),
                    "shard_id": shard_id,
                    "local_tx_index": int(local_tx),
                    "local_rx_index": int(local_rx),
                    "global_tx_index": int(global_tx[local_tx]),
                    "global_rx_index": int(global_rx[local_rx]),
                    "gt_valid": valid_gt,
                    "los_flag": bool(los_flag[local_tx, local_rx]),
                    "nlos_flag": bool(nlos_flag[local_tx, local_rx]),
                    "gt_first_path_delay_s": gt_delay_s,
                    "gt_first_path_range_m": gt_range_m,
                    "path_power_db": value_or_none(path_power_db[local_tx, local_rx]),
                    "csi_power_db": value_or_none(csi_power_db[flat_index]),
                    "pdp_success": bool(pdp_est.success),
                    "pdp_toa_est_s": value_or_none(pdp_est.toa_s),
                    "pdp_range_est_m": value_or_none(pdp_est.range_m),
                    "pdp_range_error_m": error_or_none(pdp_est.range_m, gt_range_m),
                    "pdp_abs_range_error_m": abs_error_or_none(pdp_est.range_m, gt_range_m),
                    "pdp_selected_delay_bin": int(pdp_est.selected_delay_bin),
                    "pdp_peak_power_linear": value_or_none(pdp_est.peak_power_linear),
                    "pdp_peak_snr_db": value_or_none(pdp_est.peak_snr_db),
                    "phase_success": bool(phase_est.success),
                    "phase_toa_est_s": value_or_none(phase_est.toa_s),
                    "phase_range_est_m": value_or_none(phase_est.range_m),
                    "phase_range_error_m": error_or_none(phase_est.range_m, gt_range_m),
                    "phase_abs_range_error_m": abs_error_or_none(phase_est.range_m, gt_range_m),
                    "phase_fit_residual_rad": value_or_none(phase_est.fit_residual_rad),
                }
                rows.append(row)
    return rows


def resolve_manifest_artifact(run_dir: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    candidate = run_dir / path
    if candidate.exists():
        return candidate
    return path


def read_global_indices(h5: h5py.File, role: str, count: int) -> np.ndarray:
    path = f"shard/global_{role}_indices"
    if path in h5:
        return np.asarray(h5[path][()], dtype=np.int64)
    return np.arange(count, dtype=np.int64)


def flatten_links(cfr: np.ndarray) -> np.ndarray:
    if cfr.ndim != 5:
        raise ValueError(f"Expected CFR shape [tx,rx,rx_ant,tx_ant,subcarrier], got {cfr.shape}")
    tx, rx, rx_ant, tx_ant, subcarrier = cfr.shape
    return cfr.reshape(tx * rx, rx_ant, tx_ant, subcarrier)


def estimate_pdp_for_links(
    links: np.ndarray,
    frequencies_hz: np.ndarray,
    *,
    oversampling_factor: int,
    window_name: str,
    relative_threshold_db: float,
    min_peak_snr_db: float,
    batch_size: int,
) -> list[PdpEstimate]:
    subcarriers = links.shape[-1]
    spacing_hz = subcarrier_spacing_hz(frequencies_hz)
    n_fft = int(subcarriers * oversampling_factor)
    delay_step_s = 1.0 / (spacing_hz * n_fft)
    threshold_ratio = 10.0 ** (relative_threshold_db / 10.0)
    window = window_vector(subcarriers, window_name).astype(np.float32)
    estimates: list[PdpEstimate] = []
    batch_size = max(1, int(batch_size))
    for start in range(0, links.shape[0], batch_size):
        batch = links[start : start + batch_size]
        finite = np.isfinite(batch).all(axis=(1, 2, 3))
        profile = np.fft.ifft(
            batch * window[np.newaxis, np.newaxis, np.newaxis, :],
            n=n_fft,
            axis=-1,
        )
        power_batch = np.mean(np.abs(profile) ** 2, axis=(1, 2)).astype(np.float64)
        for row, power in enumerate(power_batch):
            if not bool(finite[row]):
                estimates.append(failed_pdp())
                continue
            estimates.append(
                select_pdp_peak(
                    power,
                    delay_step_s=delay_step_s,
                    threshold_ratio=threshold_ratio,
                    min_peak_snr_db=min_peak_snr_db,
                    max_cluster_gap=max(1, oversampling_factor),
                )
            )
    return estimates


def select_pdp_peak(
    power: np.ndarray,
    *,
    delay_step_s: float,
    threshold_ratio: float,
    min_peak_snr_db: float,
    max_cluster_gap: int,
) -> PdpEstimate:
    if power.size == 0 or not np.any(np.isfinite(power)):
        return failed_pdp()
    strongest_bin = int(np.nanargmax(power))
    strongest_power = float(power[strongest_bin])
    if strongest_power <= 0.0 or not np.isfinite(strongest_power):
        return failed_pdp()
    candidate_bins = local_peak_indices(power)
    candidate_bins = candidate_bins[power[candidate_bins] >= strongest_power * threshold_ratio]
    candidate_bins = candidate_bins[candidate_bins <= strongest_bin]
    if candidate_bins.size == 0:
        return failed_pdp()
    coarse_bin = select_first_peak_cluster_max(
        candidate_bins,
        power,
        max_cluster_gap=max_cluster_gap,
    )
    noise_floor = float(np.nanmedian(power))
    snr_db = linear_to_db(strongest_power / max(noise_floor, np.finfo(np.float64).tiny))
    if snr_db < min_peak_snr_db:
        return failed_pdp()
    bin_offset = parabolic_log_power_offset(power, coarse_bin)
    toa_s = (float(coarse_bin) + bin_offset) * delay_step_s
    return PdpEstimate(
        toa_s=toa_s,
        range_m=toa_s * SPEED_OF_LIGHT_MPS,
        success=True,
        selected_delay_bin=coarse_bin,
        peak_power_linear=float(power[coarse_bin]),
        peak_snr_db=snr_db,
    )


def estimate_phase_slope_one(
    link_cfr: np.ndarray,
    frequencies_hz: np.ndarray,
    *,
    min_mean_power: float,
) -> PhaseSlopeEstimate:
    if not np.all(np.isfinite(link_cfr)):
        return failed_phase()
    centered_f = frequencies_hz - float(np.mean(frequencies_hz))
    estimates: list[float] = []
    weights: list[float] = []
    residuals: list[float] = []
    for rx_ant in range(link_cfr.shape[0]):
        for tx_ant in range(link_cfr.shape[1]):
            h = link_cfr[rx_ant, tx_ant]
            power = np.abs(h) ** 2
            mean_power = float(np.mean(power))
            if mean_power < min_mean_power or not np.all(np.isfinite(h)):
                continue
            phase = np.unwrap(np.angle(h))
            slope, intercept = weighted_line_fit(centered_f, phase, power)
            toa_s = -slope / (2.0 * np.pi)
            if not np.isfinite(toa_s) or toa_s < 0.0:
                continue
            fitted = slope * centered_f + intercept
            residual = float(np.sqrt(np.average((phase - fitted) ** 2, weights=power)))
            estimates.append(float(toa_s))
            weights.append(mean_power)
            residuals.append(residual)
    if not estimates:
        return failed_phase()
    weight_array = np.asarray(weights, dtype=np.float64)
    toa_s = weighted_median(np.asarray(estimates, dtype=np.float64), weight_array)
    residual = weighted_median(np.asarray(residuals, dtype=np.float64), weight_array)
    return PhaseSlopeEstimate(
        toa_s=toa_s,
        range_m=toa_s * SPEED_OF_LIGHT_MPS,
        success=True,
        fit_residual_rad=residual,
    )


def failed_pdp() -> PdpEstimate:
    return PdpEstimate(
        toa_s=np.nan,
        range_m=np.nan,
        success=False,
        selected_delay_bin=-1,
        peak_power_linear=np.nan,
        peak_snr_db=np.nan,
    )


def failed_phase() -> PhaseSlopeEstimate:
    return PhaseSlopeEstimate(
        toa_s=np.nan,
        range_m=np.nan,
        success=False,
        fit_residual_rad=np.nan,
    )


def subcarrier_spacing_hz(frequencies_hz: np.ndarray) -> float:
    diffs = np.diff(np.asarray(frequencies_hz, dtype=np.float64))
    if diffs.size == 0 or not np.all(diffs > 0):
        raise ValueError("frequency/frequencies_hz must be strictly increasing")
    return float(np.median(diffs))


def window_vector(size: int, name: str) -> np.ndarray:
    if name == "hann":
        return np.hanning(size)
    if name == "rect":
        return np.ones(size, dtype=np.float64)
    raise ValueError(f"Unsupported PDP window: {name}")


def local_peak_indices(power: np.ndarray) -> np.ndarray:
    if power.size == 1:
        return np.array([0], dtype=np.int64)
    peaks: list[int] = []
    for idx in range(power.size):
        left = power[idx - 1] if idx > 0 else -np.inf
        right = power[idx + 1] if idx + 1 < power.size else -np.inf
        if power[idx] >= left and power[idx] >= right:
            peaks.append(idx)
    return np.asarray(peaks, dtype=np.int64)


def select_first_peak_cluster_max(
    candidate_bins: np.ndarray,
    power: np.ndarray,
    *,
    max_cluster_gap: int,
) -> int:
    cluster = [int(candidate_bins[0])]
    for value in candidate_bins[1:]:
        value = int(value)
        if value - cluster[-1] > max_cluster_gap:
            break
        cluster.append(value)
    cluster_array = np.asarray(cluster, dtype=np.int64)
    return int(cluster_array[int(np.argmax(power[cluster_array]))])


def parabolic_log_power_offset(power: np.ndarray, bin_index: int) -> float:
    if bin_index <= 0 or bin_index >= power.size - 1:
        return 0.0
    eps = np.finfo(np.float64).tiny
    left = np.log(max(float(power[bin_index - 1]), eps))
    center = np.log(max(float(power[bin_index]), eps))
    right = np.log(max(float(power[bin_index + 1]), eps))
    denominator = left - 2.0 * center + right
    if abs(denominator) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denominator, -1.0, 1.0))


def weighted_line_fit(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    weights = np.asarray(weights, dtype=np.float64)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        raise ValueError("weights must contain positive power")
    x_mean = float(np.sum(weights * x) / weight_sum)
    y_mean = float(np.sum(weights * y) / weight_sum)
    centered_x = x - x_mean
    centered_y = y - y_mean
    denominator = float(np.sum(weights * centered_x * centered_x))
    if denominator <= 0.0:
        return 0.0, y_mean
    slope = float(np.sum(weights * centered_x * centered_y) / denominator)
    intercept = y_mean - slope * x_mean
    return slope, intercept


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cdf = np.cumsum(sorted_weights)
    cutoff = 0.5 * float(cdf[-1])
    return float(sorted_values[int(np.searchsorted(cdf, cutoff, side="left"))])


def cfr_power_db(links: np.ndarray) -> np.ndarray:
    power = np.mean(np.abs(links.astype(np.complex64)) ** 2, axis=(1, 2, 3))
    return linear_to_db(np.maximum(power, 1.0e-30))


def linear_to_db(value: np.ndarray | float) -> np.ndarray | float:
    return 10.0 * np.log10(value)


def value_or_none(value: float | np.floating[Any] | None) -> float | None:
    if value is None:
        return None
    scalar = float(value)
    if not np.isfinite(scalar):
        return None
    return scalar


def error_or_none(est: float, gt: float | None) -> float | None:
    if gt is None or not np.isfinite(est):
        return None
    return float(est - gt)


def abs_error_or_none(est: float, gt: float | None) -> float | None:
    err = error_or_none(est, gt)
    if err is None:
        return None
    return abs(err)


def build_summary(
    details: list[dict[str, Any]],
    *,
    input_root: Path,
    run_dirs: list[Path],
    config: dict[str, Any],
) -> dict[str, Any]:
    valid_rows = [row for row in details if row["gt_valid"]]
    summary: dict[str, Any] = {
        "input_root": input_root.as_posix(),
        "run_count": len(run_dirs),
        "run_dirs": [path.as_posix() for path in run_dirs],
        "detail_row_count": len(details),
        "valid_gt_link_count": len(valid_rows),
        "config": jsonable_config(config),
        "scene_counts": dict(Counter(row["scene_id"] for row in valid_rows)),
        "los_nlos_counts": dict(Counter(los_group(row) for row in valid_rows)),
        "estimators": {},
    }
    for estimator in ("pdp", "phase"):
        summary["estimators"][estimator] = {
            "overall": metric_block(valid_rows, estimator),
            "los_nlos": group_metric_blocks(valid_rows, estimator, los_group),
            "path_power_db_bins": binned_metric_blocks(
                valid_rows,
                estimator,
                field="path_power_db",
                bins=DEFAULT_POWER_BINS_DB,
            ),
            "csi_power_db_bins": binned_metric_blocks(
                valid_rows,
                estimator,
                field="csi_power_db",
                bins=DEFAULT_POWER_BINS_DB,
            ),
            "by_scene": group_metric_blocks(
                valid_rows,
                estimator,
                lambda row: str(row["scene_id"]),
            ),
        }
    return summary


def metric_block(rows: list[dict[str, Any]], estimator: str) -> dict[str, Any]:
    success_field = f"{estimator}_success"
    error_field = f"{estimator}_range_error_m"
    abs_error_field = f"{estimator}_abs_range_error_m"
    success_rows = [
        row
        for row in rows
        if row.get(success_field) and row.get(abs_error_field) is not None
    ]
    errors = np.asarray([row[error_field] for row in success_rows], dtype=np.float64)
    abs_errors = np.asarray([row[abs_error_field] for row in success_rows], dtype=np.float64)
    block = {
        "valid_count": len(rows),
        "success_count": len(success_rows),
        "detection_success_rate": (len(success_rows) / len(rows)) if rows else None,
    }
    if abs_errors.size == 0:
        block.update(
            {
                "signed_mean_error_m": None,
                "signed_median_error_m": None,
                "abs_mean_error_m": None,
                "abs_median_error_m": None,
                "abs_p90_error_m": None,
                "abs_p95_error_m": None,
            }
        )
        return block
    block.update(
        {
            "signed_mean_error_m": float(np.mean(errors)),
            "signed_median_error_m": float(np.median(errors)),
            "abs_mean_error_m": float(np.mean(abs_errors)),
            "abs_median_error_m": float(np.median(abs_errors)),
            "abs_p90_error_m": float(np.percentile(abs_errors, 90)),
            "abs_p95_error_m": float(np.percentile(abs_errors, 95)),
        }
    )
    return block


def group_metric_blocks(
    rows: list[dict[str, Any]],
    estimator: str,
    key_fn,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(key_fn(row))].append(row)
    return {key: metric_block(value, estimator) for key, value in sorted(grouped.items())}


def binned_metric_blocks(
    rows: list[dict[str, Any]],
    estimator: str,
    *,
    field: str,
    bins: tuple[float, ...],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is None:
            grouped["missing"].append(row)
            continue
        grouped[bin_label(float(value), bins)].append(row)
    return {key: metric_block(value, estimator) for key, value in sorted(grouped.items())}


def bin_label(value: float, bins: tuple[float, ...]) -> str:
    for low, high in zip(bins[:-1], bins[1:], strict=True):
        if low <= value < high:
            return f"[{format_edge(low)},{format_edge(high)})"
    return "out_of_range"


def format_edge(value: float) -> str:
    if value == -np.inf:
        return "-inf"
    if value == np.inf:
        return "inf"
    return f"{value:g}"


def los_group(row: dict[str, Any]) -> str:
    return "los" if row.get("los_flag") else "nlos"


def jsonable_config(config: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, (list, tuple)):
            out[key] = list(value)
        else:
            out[key] = value
    return out


def write_details_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_details_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def render_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# CFR Truth ToA Pilot Summary",
        "",
        f"- input_root: `{summary['input_root']}`",
        f"- run_count: {summary['run_count']}",
        f"- valid_gt_link_count: {summary['valid_gt_link_count']}",
        "",
    ]
    for estimator, estimator_summary in summary["estimators"].items():
        overall = estimator_summary["overall"]
        lines.extend(
            [
                f"## {estimator}",
                "",
                "| metric | value |",
                "|---|---:|",
                f"| success_rate | {fmt(overall['detection_success_rate'])} |",
                f"| abs_mean_error_m | {fmt(overall['abs_mean_error_m'])} |",
                f"| abs_median_error_m | {fmt(overall['abs_median_error_m'])} |",
                f"| abs_p90_error_m | {fmt(overall['abs_p90_error_m'])} |",
                f"| abs_p95_error_m | {fmt(overall['abs_p95_error_m'])} |",
                f"| signed_mean_error_m | {fmt(overall['signed_mean_error_m'])} |",
                "",
                "### LoS vs NLoS",
                "",
                "| group | valid | success_rate | abs_median_m | abs_p90_m | abs_p95_m |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for group, block in estimator_summary["los_nlos"].items():
            lines.append(
                "| "
                f"{group} | {block['valid_count']} | {fmt(block['detection_success_rate'])} | "
                f"{fmt(block['abs_median_error_m'])} | {fmt(block['abs_p90_error_m'])} | "
                f"{fmt(block['abs_p95_error_m'])} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
