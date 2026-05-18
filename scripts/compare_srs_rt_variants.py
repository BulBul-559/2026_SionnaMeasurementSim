"""Compare SRS-like RT variant outputs and plot selected-link CFR views."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

_EPS = 1e-30


@dataclass(frozen=True)
class Variant:
    label: str
    path: Path
    links: dict[tuple[int, int], np.ndarray]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Sharded output directories or HDF5 files to compare.",
    )
    parser.add_argument(
        "--labels",
        default="",
        help="Comma-separated labels. Defaults to path names.",
    )
    parser.add_argument(
        "--dataset",
        default="/observation/cfr_est",
        help="Complex CFR dataset path to compare.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for metrics and PNG outputs.",
    )
    parser.add_argument("--reference", default=None, help="Reference label; defaults to first.")
    parser.add_argument("--bs-index", type=int, default=0, help="Global BS/TX index to plot.")
    parser.add_argument("--ue-index", type=int, default=0, help="Global UE/RX index to plot.")
    args = parser.parse_args()

    labels = _parse_labels(args.labels, args.paths)
    variants = [
        Variant(label=label, path=path, links=_load_links(path, args.dataset))
        for label, path in zip(labels, args.paths, strict=True)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    reference = args.reference or variants[0].label
    if reference not in {variant.label for variant in variants}:
        msg = f"Unknown reference label: {reference}"
        raise ValueError(msg)

    common_links = _common_links(variants)
    if not common_links:
        msg = "No common global BS/UE links found across variants."
        raise ValueError(msg)
    selected_link = (args.bs_index, args.ue_index)
    if selected_link not in common_links:
        selected_link = sorted(common_links)[0]

    metrics = _build_metrics(variants, reference, common_links, selected_link)
    _write_metrics(args.output_dir, metrics)
    _plot_selected_link(
        variants,
        selected_link=selected_link,
        output_dir=args.output_dir,
    )
    print(args.output_dir / "metrics.json")
    return 0


def _parse_labels(raw: str, paths: list[Path]) -> list[str]:
    if raw:
        labels = [item.strip() for item in raw.split(",") if item.strip()]
        if len(labels) != len(paths):
            msg = "--labels count must match number of paths"
            raise ValueError(msg)
        return labels
    return [path.name or f"variant_{index}" for index, path in enumerate(paths)]


def _resolve_hdf5_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    manifest = path / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        results = data.get("results", [])
        if results:
            files: list[Path] = []
            for item in results:
                result_path = Path(item["result_h5"])
                files.append(result_path if result_path.is_absolute() else path / result_path)
            return files
        if data.get("results_h5"):
            result_path = Path(data["results_h5"])
            return [result_path if result_path.is_absolute() else path / result_path]
    files = sorted(path.glob("result_*.h5"))
    if files:
        return files
    return sorted(path.glob("bs*_ue*/results.h5")) + sorted(path.glob("bs*_ue*/result_*.h5"))


def _load_links(path: Path, dataset_path: str) -> dict[tuple[int, int], np.ndarray]:
    links: dict[tuple[int, int], np.ndarray] = {}
    files = _resolve_hdf5_files(path)
    if not files:
        msg = f"No HDF5 files found under {path}"
        raise FileNotFoundError(msg)
    key = dataset_path.strip("/")
    for file in files:
        with h5py.File(file, "r") as h5:
            if key not in h5:
                msg = f"{dataset_path} not found in {file}"
                raise KeyError(msg)
            data = np.asarray(h5[key][()])
            if data.ndim == 6:
                data = data[0]
            if data.ndim != 5:
                msg = (
                    f"{dataset_path} must have shape [snapshot,tx,rx,rx_ant,tx_ant,sc] "
                    f"or [tx,rx,rx_ant,tx_ant,sc], got {data.shape}"
                )
                raise ValueError(msg)
            tx_indices = _read_indices(h5, "shard/global_tx_indices", data.shape[0])
            rx_indices = _read_indices(h5, "shard/global_rx_indices", data.shape[1])
            if "shard/global_tx_indices" not in h5 and "shard/global_rx_indices" not in h5:
                parsed = _parse_link_indices_from_path(file)
                if parsed is not None:
                    tx_indices = np.asarray([parsed[0]], dtype=np.int64)
                    rx_indices = np.asarray([parsed[1]], dtype=np.int64)
            for tx_local, tx_global in enumerate(tx_indices):
                for rx_local, rx_global in enumerate(rx_indices):
                    links[(int(tx_global), int(rx_global))] = np.asarray(
                        data[tx_local, rx_local],
                        dtype=np.complex64,
                    )
    return links


def _read_indices(h5: h5py.File, path: str, length: int) -> np.ndarray:
    if path in h5:
        return np.asarray(h5[path][()], dtype=np.int64)
    return np.arange(length, dtype=np.int64)


def _parse_link_indices_from_path(path: Path) -> tuple[int, int] | None:
    match = re.search(r"bs(\d+)_ue(\d+)", path.parent.name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _common_links(variants: list[Variant]) -> set[tuple[int, int]]:
    common = set(variants[0].links)
    for variant in variants[1:]:
        common &= set(variant.links)
    return common


def _build_metrics(
    variants: list[Variant],
    reference_label: str,
    common_links: set[tuple[int, int]],
    selected_link: tuple[int, int],
) -> dict[str, Any]:
    reference = next(variant for variant in variants if variant.label == reference_label)
    by_label = {variant.label: variant for variant in variants}
    metrics: dict[str, Any] = {
        "reference": reference_label,
        "common_link_count": len(common_links),
        "selected_link": {"bs_index": selected_link[0], "ue_index": selected_link[1]},
        "against_reference": {},
        "selected_link_against_reference": {},
        "pairwise": {},
    }
    for variant in variants:
        metrics["against_reference"][variant.label] = _compare_link_sets(
            reference.links,
            variant.links,
            common_links,
        )
        metrics["selected_link_against_reference"][variant.label] = _compare_arrays(
            reference.links[selected_link],
            variant.links[selected_link],
        )
    for left_label, left in by_label.items():
        metrics["pairwise"][left_label] = {}
        for right_label, right in by_label.items():
            metrics["pairwise"][left_label][right_label] = _compare_link_sets(
                left.links,
                right.links,
                common_links,
            )
    return metrics


def _compare_link_sets(
    reference: dict[tuple[int, int], np.ndarray],
    candidate: dict[tuple[int, int], np.ndarray],
    links: set[tuple[int, int]],
) -> dict[str, float]:
    ref_values = np.concatenate([reference[key].reshape(-1) for key in sorted(links)])
    cand_values = np.concatenate([candidate[key].reshape(-1) for key in sorted(links)])
    return _compare_vectors(ref_values, cand_values)


def _compare_arrays(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    return _compare_vectors(reference.reshape(-1), candidate.reshape(-1))


def _compare_vectors(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(reference.real) & np.isfinite(reference.imag)
    mask &= np.isfinite(candidate.real) & np.isfinite(candidate.imag)
    ref = reference[mask].astype(np.complex128)
    cand = candidate[mask].astype(np.complex128)
    if ref.size == 0:
        return {
            "sample_count": 0.0,
            "nmse_db": float("nan"),
            "complex_correlation": float("nan"),
            "magnitude_mae_db": float("nan"),
            "phase_circular_mae_rad": float("nan"),
            "phase_circular_rmse_rad": float("nan"),
        }
    error = cand - ref
    ref_power = float(np.sum(np.abs(ref) ** 2))
    cand_power = float(np.sum(np.abs(cand) ** 2))
    nmse = float(np.sum(np.abs(error) ** 2) / max(ref_power, _EPS))
    corr = float(
        np.abs(np.vdot(ref, cand))
        / max(np.sqrt(ref_power * cand_power), _EPS)
    )
    mag_ref = 20.0 * np.log10(np.abs(ref) + _EPS)
    mag_cand = 20.0 * np.log10(np.abs(cand) + _EPS)
    phase_delta = np.angle(cand * np.conj(ref))
    return {
        "sample_count": float(ref.size),
        "nmse_db": float(10.0 * np.log10(max(nmse, _EPS))),
        "complex_correlation": corr,
        "magnitude_mae_db": float(np.mean(np.abs(mag_cand - mag_ref))),
        "phase_circular_mae_rad": float(np.mean(np.abs(phase_delta))),
        "phase_circular_rmse_rad": float(np.sqrt(np.mean(phase_delta ** 2))),
    }


def _write_metrics(output_dir: Path, metrics: dict[str, Any]) -> None:
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    rows = []
    for scope, values in (
        ("all_common_links", metrics["against_reference"]),
        ("selected_link", metrics["selected_link_against_reference"]),
    ):
        for label, row in values.items():
            rows.append({"scope": scope, "label": label, **row})
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "scope",
                "label",
                "sample_count",
                "nmse_db",
                "complex_correlation",
                "magnitude_mae_db",
                "phase_circular_mae_rad",
                "phase_circular_rmse_rad",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _plot_selected_link(
    variants: list[Variant],
    *,
    selected_link: tuple[int, int],
    output_dir: Path,
) -> None:
    link_arrays = [variant.links[selected_link] for variant in variants]
    magnitude = [
        20.0 * np.log10(np.abs(_antenna_pair_matrix(array)) + _EPS)
        for array in link_arrays
    ]
    phase = [np.angle(_antenna_pair_matrix(array)) for array in link_arrays]
    _plot_heatmap_row(
        magnitude,
        labels=[variant.label for variant in variants],
        title=(
            f"SRS CFR magnitude variants "
            f"(BS {selected_link[0]}, UE {selected_link[1]})"
        ),
        colorbar_label="|CFR| [dB]",
        output_path=output_dir / "cfr_magnitude_variants.png",
    )
    _plot_heatmap_row(
        phase,
        labels=[variant.label for variant in variants],
        title=(
            f"SRS CFR phase variants "
            f"(BS {selected_link[0]}, UE {selected_link[1]})"
        ),
        colorbar_label="phase [rad]",
        output_path=output_dir / "cfr_phase_variants.png",
        vmin=-np.pi,
        vmax=np.pi,
    )


def _antenna_pair_matrix(link: np.ndarray) -> np.ndarray:
    # link shape: [rx_ant, tx_ant, subcarrier]; plot as [subcarrier, antenna_pair].
    rx_ant, tx_ant, subcarriers = link.shape
    return link.reshape(rx_ant * tx_ant, subcarriers).T


def _plot_heatmap_row(
    matrices: list[np.ndarray],
    *,
    labels: list[str],
    title: str,
    colorbar_label: str,
    output_path: Path,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    if vmin is None:
        vmin = min(float(np.nanmin(matrix)) for matrix in matrices)
    if vmax is None:
        vmax = max(float(np.nanmax(matrix)) for matrix in matrices)
    fig, axes = plt.subplots(1, len(matrices), figsize=(4.2 * len(matrices), 4.2), squeeze=False)
    fig.suptitle(title)
    image = None
    for axis, matrix, label in zip(axes[0], matrices, labels, strict=True):
        image = axis.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            interpolation="none",
            vmin=vmin,
            vmax=vmax,
        )
        axis.set_title(label)
        axis.set_xlabel("antenna pair")
        axis.set_ylabel("subcarrier")
    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.88)
        cbar.set_label(colorbar_label)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
