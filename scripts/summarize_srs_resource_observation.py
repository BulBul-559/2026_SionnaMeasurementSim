#!/usr/bin/env python
"""Summarize NR SRS resource observation and v2 smoke outputs."""

from __future__ import annotations

import argparse
import csv
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
        cfr_resource = h5["observation/cfr_est_resource"][()]
        truth_resource = _truth_resource_for_srs_ports(h5)
        resource_error = cfr_resource - truth_resource
        resource_nmse = _nmse_db(truth_resource, cfr_resource)
        full_nmse = h5["evaluation/nmse_db"][()]
        eval_resource_nmse = h5["evaluation/srs_resource_nmse_db"][()]
        eval_interp_nmse = h5["evaluation/srs_interpolation_nmse_db"][()]
        mask = h5["waveform/srs_resource_mask"][()]
        cfo = h5["observation/cfo_hz"][()] if "observation/cfo_hz" in h5 else np.asarray([])
        sfo = h5["observation/sfo_ppm"][()] if "observation/sfo_ppm" in h5 else np.asarray([])
        timing = (
            h5["observation/timing_offset_samples"][()]
            if "observation/timing_offset_samples" in h5
            else np.asarray([])
        )
        phase = (
            h5["observation/phase_offset_rad"][()]
            if "observation/phase_offset_rad" in h5
            else np.asarray([])
        )
        scale = (
            h5["waveform/srs_power_scale_linear"][()]
            if "waveform/srs_power_scale_linear" in h5
            else np.asarray([1.0], dtype=np.float32)
        )
        tx_power = (
            h5["waveform/srs_tx_power_dbm"][()]
            if "waveform/srs_tx_power_dbm" in h5
            else np.asarray([np.nan], dtype=np.float32)
        )
        return {
            "path": str(path),
            "schema_pass": True,
            "resource_re_count": int(cfr_resource.shape[-1]),
            "resource_mask_count": int(np.count_nonzero(mask)),
            "tx_grid_shape": list(h5["waveform/tx_grid"].shape),
            "rx_grid_shape": list(h5["waveform/rx_grid"].shape),
            "cfr_est_resource_shape": list(cfr_resource.shape),
            "srs_port_count": int(cfr_resource.shape[-2]),
            "srs_symbol_count": int(h5["waveform/srs_symbol_indices"].shape[0]),
            "prb_count_per_symbol": h5["waveform/srs_prb_count_per_symbol"][()].astype(int).tolist()
            if "waveform/srs_prb_count_per_symbol" in h5
            else [],
            "resource_nmse_db_mean": _finite_mean(resource_nmse),
            "resource_nmse_db_median": _finite_median(resource_nmse),
            "resource_abs_error_mean": _finite_mean(np.abs(resource_error)),
            "full_nmse_db_mean": _finite_mean(full_nmse),
            "full_nmse_db_median": _finite_median(full_nmse),
            "eval_resource_nmse_db_mean": _finite_mean(eval_resource_nmse),
            "eval_interpolation_nmse_db_mean": _finite_mean(eval_interp_nmse),
            "cfo_nonzero": _any_nonzero(cfo),
            "sfo_nonzero": _any_nonzero(sfo),
            "timing_nonzero": _any_nonzero(timing),
            "phase_nonzero": _any_nonzero(phase),
            "power_scale_min": _finite_min(scale),
            "power_scale_mean": _finite_mean(scale),
            "power_scale_max": _finite_max(scale),
            "tx_power_dbm_min": _finite_min(tx_power),
            "tx_power_dbm_mean": _finite_mean(tx_power),
            "tx_power_dbm_max": _finite_max(tx_power),
            "ranging_pdp_finite_rate": _ranging_finite_rate(h5, "pdp_peak"),
            "ranging_phase_finite_rate": _ranging_finite_rate(h5, "phase_slope"),
            "has_spatial_spectrum_cfr_est": "array/spatial_spectrum_cfr_est" in h5,
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
        "all_impairment_metadata_nonzero": all(
            bool(record["cfo_nonzero"])
            and bool(record["sfo_nonzero"])
            and bool(record["timing_nonzero"])
            and bool(record["phase_nonzero"])
            for record in records
        ),
        "all_spatial_spectrum_cfr_est_present": all(
            bool(record["has_spatial_spectrum_cfr_est"]) for record in records
        ),
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({column: record[column] for column in columns})


def _truth_resource_for_srs_ports(h5: h5py.File) -> np.ndarray:
    truth = h5["channel/truth/cfr"][()]
    cfr_resource = h5["observation/cfr_est_resource"]
    re_indices = h5["waveform/srs_re_subcarrier_indices"][()]
    if "waveform/srs_re_symbol_indices" not in h5 or "waveform/srs_port_tx_ant_map" not in h5:
        return truth[np.newaxis, ..., re_indices]
    re_symbols = h5["waveform/srs_re_symbol_indices"][()]
    srs_symbols = h5["waveform/srs_symbol_indices"][()]
    port_map = h5["waveform/srs_port_tx_ant_map"][()]
    local_by_symbol = {int(symbol): idx for idx, symbol in enumerate(srs_symbols)}
    local_symbols = np.asarray([local_by_symbol[int(symbol)] for symbol in re_symbols])
    out = np.zeros(
        (*truth[np.newaxis, ...].shape[:4], cfr_resource.shape[-2], re_indices.size),
        dtype=np.complex64,
    )
    truth_snap = truth[np.newaxis, ...]
    for port in range(cfr_resource.shape[-2]):
        for flat_idx, local_symbol in enumerate(local_symbols):
            tx_ant = int(port_map[port, local_symbol])
            if tx_ant < 0:
                continue
            out[..., port, flat_idx] = truth_snap[..., tx_ant, int(re_indices[flat_idx])]
    return out


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


def _finite_min(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return float(np.min(finite)) if finite.size else float("nan")


def _finite_max(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    return float(np.max(finite)) if finite.size else float("nan")


def _any_nonzero(values: np.ndarray) -> bool:
    arr = np.asarray(values)
    if arr.size == 0:
        return False
    return bool(np.any(np.abs(arr.astype(np.float64, copy=False)) > 0.0))


def _ranging_finite_rate(h5: h5py.File, estimator: str) -> float:
    path = f"ranging/{estimator}/range_error_m"
    success_path = f"ranging/{estimator}/detection_success"
    if path not in h5 or success_path not in h5:
        return float("nan")
    errors = np.asarray(h5[path][()], dtype=np.float32)
    success = np.asarray(h5[success_path][()], dtype=np.bool_)
    finite = np.isfinite(errors) & success
    return float(np.mean(finite)) if finite.size else float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
