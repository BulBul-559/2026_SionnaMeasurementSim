"""Compare PHY CSI outputs from two HDF5 files or sharded output directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path, help="First results.h5 or sharded output dir")
    parser.add_argument("right", type=Path, help="Second results.h5 or sharded output dir")
    parser.add_argument("--label-left", default="left")
    parser.add_argument("--label-right", default="right")
    args = parser.parse_args()

    left = summarize(args.left)
    right = summarize(args.right)
    print(_format_summary(args.label_left, left))
    print()
    print(_format_summary(args.label_right, right))
    return 0


def summarize(path: Path) -> dict[str, object]:
    files = _resolve_hdf5_files(path)
    summaries = [_summarize_hdf5(file) for file in files]
    return {
        "path": path.as_posix(),
        "file_count": len(files),
        "standards": sorted({str(item["standard"]) for item in summaries}),
        "link_count": int(sum(int(item["link_count"]) for item in summaries)),
        "median_nmse_db": _weighted_median(summaries, "median_nmse_db"),
        "p95_nmse_db": _weighted_percentile(summaries, "p95_nmse_db", 95.0),
        "valid_rate": _weighted_average(summaries, "valid_rate"),
        "spectra": sorted({name for item in summaries for name in item["spectra"]}),
    }


def _resolve_hdf5_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    manifest = path / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        results = data.get("results", [])
        if results:
            return [Path(item["result_h5"]) for item in results]
        if data.get("results_h5"):
            return [Path(data["results_h5"])]
    return sorted(path.glob("result_*.h5"))


def _summarize_hdf5(path: Path) -> dict[str, object]:
    with h5py.File(path, "r") as h5:
        standard = h5["waveform/standard"][()]
        if isinstance(standard, bytes):
            standard = standard.decode("utf-8")
        nmse = np.asarray(h5["evaluation/nmse_db"][()], dtype=np.float32)
        valid = np.asarray(h5["observation/valid_mask"][()], dtype=np.bool_)
        spectra = [
            name
            for name in (
                "spatial_spectrum_truth",
                "spatial_spectrum_cfr_est",
                "spatial_spectrum_observation",
            )
            if f"array/{name}" in h5
        ]
    finite_nmse = nmse[np.isfinite(nmse)]
    return {
        "standard": standard,
        "link_count": int(valid.size),
        "median_nmse_db": float(np.median(finite_nmse)) if finite_nmse.size else float("nan"),
        "p95_nmse_db": float(np.percentile(finite_nmse, 95)) if finite_nmse.size else float("nan"),
        "valid_rate": float(np.mean(valid)) if valid.size else 0.0,
        "spectra": spectra,
    }


def _weighted_average(items: list[dict[str, object]], key: str) -> float:
    weights = np.asarray([int(item["link_count"]) for item in items], dtype=np.float64)
    values = np.asarray([float(item[key]) for item in items], dtype=np.float64)
    if weights.size == 0 or np.sum(weights) == 0:
        return float("nan")
    return float(np.sum(values * weights) / np.sum(weights))


def _weighted_median(items: list[dict[str, object]], key: str) -> float:
    values = [float(item[key]) for item in items if np.isfinite(float(item[key]))]
    return float(np.median(values)) if values else float("nan")


def _weighted_percentile(items: list[dict[str, object]], key: str, percentile: float) -> float:
    values = [float(item[key]) for item in items if np.isfinite(float(item[key]))]
    return float(np.percentile(values, percentile)) if values else float("nan")


def _format_summary(label: str, summary: dict[str, object]) -> str:
    return "\n".join(
        [
            f"{label}: {summary['path']}",
            f"  files: {summary['file_count']}",
            f"  standards: {', '.join(summary['standards'])}",
            f"  links: {summary['link_count']}",
            f"  median_nmse_db: {summary['median_nmse_db']:.3f}",
            f"  p95_nmse_db: {summary['p95_nmse_db']:.3f}",
            f"  valid_rate: {summary['valid_rate']:.4f}",
            f"  spectra: {', '.join(summary['spectra'])}",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
