"""Plot per-UE CFR similarity heatmaps over a floorplan image.

The default arguments compare the two medium SRS-like shard20 baselines in the
local outputs directory. Similarity is computed per UE by aggregating all BS
links, antenna pairs, and subcarriers for the selected CFR dataset.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEFT = REPO_ROOT / "outputs/nr_srs_medium_0000_label0p2_full_baseline_shard20"
DEFAULT_RIGHT = REPO_ROOT / "outputs/nr_srs_medium_0001_label0p2_full_baseline_shard20"
DEFAULT_FLOORPLAN_IMAGE = REPO_ROOT / "data/medium/medium_0000/floorplan/000_z_1.60.png"
DEFAULT_FLOORPLAN_META = REPO_ROOT / "data/medium/medium_0000/floorplan/meta.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/cfr_similarity_medium_0000_vs_medium_0001"
DEFAULT_DATASET = "/observation/cfr_est"
EPS = 1e-30


@dataclass(frozen=True)
class ShardItem:
    shard_index: int
    path: Path


@dataclass(frozen=True)
class ShardPair:
    shard_index: int
    left: Path | None
    right: Path | None


@dataclass(frozen=True)
class ShardResult:
    shard_index: int
    ue_indices: np.ndarray
    bs_indices: np.ndarray
    positions_m: np.ndarray
    magnitude_similarity: np.ndarray
    phase_similarity: np.ndarray
    i_similarity: np.ndarray
    q_similarity: np.ndarray
    magnitude_similarity_per_bs: np.ndarray
    phase_similarity_per_bs: np.ndarray
    i_similarity_per_bs: np.ndarray
    q_similarity_per_bs: np.ndarray


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--left", type=Path, default=DEFAULT_LEFT, help="Left HDF5 file or sharded run directory."
    )
    parser.add_argument(
        "--right",
        type=Path,
        default=DEFAULT_RIGHT,
        help="Right HDF5 file or sharded run directory.",
    )
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET, help="Complex CFR dataset path to compare."
    )
    parser.add_argument(
        "--snapshot-index", type=int, default=0, help="Snapshot index for 6-D observation datasets."
    )
    parser.add_argument("--floorplan-image", type=Path, default=DEFAULT_FLOORPLAN_IMAGE)
    parser.add_argument("--floorplan-meta", type=Path, default=DEFAULT_FLOORPLAN_META)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=1, help="Parallel shard workers.")
    parser.add_argument(
        "--max-shards", type=int, default=None, help="Optional limit for quick debugging."
    )
    parser.add_argument(
        "--image-origin",
        choices=("upper", "lower"),
        default="upper",
        help="How row 0 of the floorplan PNG maps to the plot extent.",
    )
    parser.add_argument("--heatmap-alpha", type=float, default=0.68)
    parser.add_argument("--point-size", type=float, default=5.0)
    parser.add_argument(
        "--bs-subset",
        default="0,1,2",
        help="Comma-separated global BS indices for the reduced-BS mean outputs.",
    )
    parser.add_argument(
        "--show-samples",
        action="store_true",
        help="Overlay original UE sample points on top of the rectangular heatmap.",
    )
    parser.add_argument("--position-atol-m", type=float, default=1e-3)
    args = parser.parse_args()

    left_items = _resolve_shards(args.left)
    right_items = _resolve_shards(args.right)
    shard_pairs = _pair_shards(left_items, right_items)
    if args.max_shards is not None:
        shard_pairs = shard_pairs[: args.max_shards]
    if not shard_pairs:
        msg = "No paired HDF5 shards found."
        raise FileNotFoundError(msg)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = _compute_all_shards(
        shard_pairs,
        dataset_path=args.dataset,
        snapshot_index=args.snapshot_index,
        workers=max(1, args.workers),
        position_atol_m=args.position_atol_m,
    )
    table = _merge_results(results)
    output_paths = _write_outputs(args, table, shard_pairs)
    print(json.dumps({key: value.as_posix() for key, value in output_paths.items()}, indent=2))
    return 0


def _resolve_shards(path: Path) -> list[ShardItem]:
    path = path.expanduser()
    if path.is_file():
        return [ShardItem(shard_index=0, path=path)]

    manifest = path / "manifest.json"
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        items = data.get("results", [])
        if items:
            shards: list[ShardItem] = []
            for index, item in enumerate(items):
                result_path = _resolve_manifest_path(path, Path(item["result_h5"]))
                shards.append(
                    ShardItem(shard_index=int(item.get("shard_index", index)), path=result_path)
                )
            return sorted(shards, key=lambda item: item.shard_index)
        result_h5 = data.get("results_h5")
        if result_h5:
            return [ShardItem(shard_index=0, path=_resolve_manifest_path(path, Path(result_h5)))]

    files = sorted(path.glob("result_*.h5"))
    if not files:
        files = sorted(path.glob("results.h5"))
    return [_shard_item_from_file(index, file) for index, file in enumerate(files)]


def _shard_item_from_file(default_index: int, path: Path) -> ShardItem:
    match = re.fullmatch(r"result_(\d+)\.h5", path.name)
    if match is not None:
        return ShardItem(shard_index=int(match.group(1)), path=path)
    return ShardItem(shard_index=default_index, path=path)


def _resolve_manifest_path(run_dir: Path, result_path: Path) -> Path:
    if result_path.is_absolute():
        return result_path
    if result_path.exists():
        return result_path
    candidate = run_dir / result_path
    if candidate.exists():
        return candidate
    return result_path


def _pair_shards(left: list[ShardItem], right: list[ShardItem]) -> list[ShardPair]:
    left_by_index = {item.shard_index: item.path for item in left}
    right_by_index = {item.shard_index: item.path for item in right}
    indices = sorted(set(left_by_index) | set(right_by_index))
    return [
        ShardPair(index, left_by_index.get(index), right_by_index.get(index)) for index in indices
    ]


def _compute_all_shards(
    shard_pairs: list[ShardPair],
    *,
    dataset_path: str,
    snapshot_index: int,
    workers: int,
    position_atol_m: float,
) -> list[ShardResult]:
    if workers == 1:
        return [
            _compute_shard_pair(pair, dataset_path, snapshot_index, position_atol_m)
            for pair in shard_pairs
        ]

    results: list[ShardResult] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _compute_shard_pair,
                pair,
                dataset_path,
                snapshot_index,
                position_atol_m,
            ): pair.shard_index
            for pair in shard_pairs
        }
        for future in as_completed(futures):
            results.append(future.result())
    return sorted(results, key=lambda result: result.shard_index)


def _compute_shard_pair(
    pair: ShardPair,
    dataset_path: str,
    snapshot_index: int,
    position_atol_m: float,
) -> ShardResult:
    if pair.left is None and pair.right is None:
        msg = f"Shard {pair.shard_index} has no HDF5 file on either side."
        raise ValueError(msg)
    if pair.left is None:
        metadata = _load_ue_metadata(pair.right)
        return _zero_similarity_result(pair.shard_index, metadata)
    if pair.right is None:
        metadata = _load_ue_metadata(pair.left)
        return _zero_similarity_result(pair.shard_index, metadata)

    left = _load_ue_major_cfr(pair.left, dataset_path, snapshot_index)
    right = _load_ue_major_cfr(pair.right, dataset_path, snapshot_index)

    if not np.array_equal(left["ue_indices"], right["ue_indices"]):
        msg = f"Shard {pair.shard_index} UE indices differ between runs."
        raise ValueError(msg)
    if left["cfr"].shape != right["cfr"].shape:
        msg = (
            f"Shard {pair.shard_index} CFR shapes differ: "
            f"{left['cfr'].shape} vs {right['cfr'].shape}"
        )
        raise ValueError(msg)
    if not np.allclose(left["positions_m"], right["positions_m"], atol=position_atol_m, rtol=0.0):
        max_delta = float(np.max(np.abs(left["positions_m"] - right["positions_m"])))
        msg = f"Shard {pair.shard_index} UE positions differ by up to {max_delta:.6g} m."
        raise ValueError(msg)

    left_cfr = left["cfr"]
    right_cfr = right["cfr"]
    valid = left["valid"] & right["valid"]
    sample_mask = _finite_complex_mask(left_cfr, right_cfr, valid)

    mag_left = np.abs(left_cfr).astype(np.float32, copy=False)
    mag_right = np.abs(right_cfr).astype(np.float32, copy=False)
    magnitude_similarity = _normalized_l2_similarity(mag_left, mag_right, sample_mask)
    phase_similarity = _weighted_phase_similarity(
        left_cfr, right_cfr, mag_left, mag_right, sample_mask
    )
    i_similarity = _normalized_l2_similarity(left_cfr.real, right_cfr.real, sample_mask)
    q_similarity = _normalized_l2_similarity(left_cfr.imag, right_cfr.imag, sample_mask)
    per_bs_axes = tuple(range(2, left_cfr.ndim))
    magnitude_similarity_per_bs = _normalized_l2_similarity(
        mag_left, mag_right, sample_mask, axes=per_bs_axes
    )
    phase_similarity_per_bs = _weighted_phase_similarity(
        left_cfr,
        right_cfr,
        mag_left,
        mag_right,
        sample_mask,
        axes=per_bs_axes,
    )
    i_similarity_per_bs = _normalized_l2_similarity(
        left_cfr.real, right_cfr.real, sample_mask, axes=per_bs_axes
    )
    q_similarity_per_bs = _normalized_l2_similarity(
        left_cfr.imag, right_cfr.imag, sample_mask, axes=per_bs_axes
    )

    return ShardResult(
        shard_index=pair.shard_index,
        ue_indices=np.asarray(left["ue_indices"], dtype=np.int64),
        bs_indices=np.asarray(left["bs_indices"], dtype=np.int64),
        positions_m=np.asarray(left["positions_m"], dtype=np.float32),
        magnitude_similarity=magnitude_similarity,
        phase_similarity=phase_similarity,
        i_similarity=i_similarity,
        q_similarity=q_similarity,
        magnitude_similarity_per_bs=magnitude_similarity_per_bs,
        phase_similarity_per_bs=phase_similarity_per_bs,
        i_similarity_per_bs=i_similarity_per_bs,
        q_similarity_per_bs=q_similarity_per_bs,
    )


def _zero_similarity_result(shard_index: int, metadata: dict[str, Any]) -> ShardResult:
    ue_indices = np.asarray(metadata["ue_indices"], dtype=np.int64)
    bs_indices = np.asarray(metadata["bs_indices"], dtype=np.int64)
    zeros = np.zeros(ue_indices.shape[0], dtype=np.float32)
    zeros_per_bs = np.zeros((ue_indices.shape[0], bs_indices.shape[0]), dtype=np.float32)
    return ShardResult(
        shard_index=shard_index,
        ue_indices=ue_indices,
        bs_indices=bs_indices,
        positions_m=np.asarray(metadata["positions_m"], dtype=np.float32),
        magnitude_similarity=zeros.copy(),
        phase_similarity=zeros.copy(),
        i_similarity=zeros.copy(),
        q_similarity=zeros.copy(),
        magnitude_similarity_per_bs=zeros_per_bs.copy(),
        phase_similarity_per_bs=zeros_per_bs.copy(),
        i_similarity_per_bs=zeros_per_bs.copy(),
        q_similarity_per_bs=zeros_per_bs.copy(),
    )


def _load_ue_major_cfr(path: Path, dataset_path: str, snapshot_index: int) -> dict[str, Any]:
    key = dataset_path.strip("/")
    with h5py.File(path, "r") as h5:
        if key not in h5:
            msg = f"{dataset_path} not found in {path}"
            raise KeyError(msg)
        cfr = _read_cfr(h5[key], snapshot_index)
        valid = _read_link_valid_mask(h5, snapshot_index, cfr.shape[:2])
        tx_role = _decode_scalar(h5["link/tx_role"][()]).lower()
        rx_role = _decode_scalar(h5["link/rx_role"][()]).lower()
        tx_indices = _read_indices(h5, "shard/global_tx_indices", cfr.shape[0])
        rx_indices = _read_indices(h5, "shard/global_rx_indices", cfr.shape[1])
        tx_positions = np.asarray(h5["topology/tx_positions_m"][()], dtype=np.float32)
        rx_positions = np.asarray(h5["topology/rx_positions_m"][()], dtype=np.float32)

    if tx_role == "ue":
        return {
            "cfr": cfr,
            "valid": valid,
            "ue_indices": tx_indices,
            "bs_indices": rx_indices,
            "positions_m": tx_positions,
        }
    if rx_role == "ue":
        return {
            "cfr": np.moveaxis(cfr, 1, 0),
            "valid": valid.T,
            "ue_indices": rx_indices,
            "bs_indices": tx_indices,
            "positions_m": rx_positions,
        }

    msg = f"Neither TX nor RX role is UE in {path}: tx_role={tx_role}, rx_role={rx_role}"
    raise ValueError(msg)


def _load_ue_metadata(path: Path | None) -> dict[str, Any]:
    if path is None:
        msg = "Cannot load UE metadata from a missing shard."
        raise ValueError(msg)
    with h5py.File(path, "r") as h5:
        tx_role = _decode_scalar(h5["link/tx_role"][()]).lower()
        rx_role = _decode_scalar(h5["link/rx_role"][()]).lower()
        tx_positions = np.asarray(h5["topology/tx_positions_m"][()], dtype=np.float32)
        rx_positions = np.asarray(h5["topology/rx_positions_m"][()], dtype=np.float32)
        tx_indices = _read_indices(h5, "shard/global_tx_indices", tx_positions.shape[0])
        rx_indices = _read_indices(h5, "shard/global_rx_indices", rx_positions.shape[0])

    if tx_role == "ue":
        return {"ue_indices": tx_indices, "bs_indices": rx_indices, "positions_m": tx_positions}
    if rx_role == "ue":
        return {"ue_indices": rx_indices, "bs_indices": tx_indices, "positions_m": rx_positions}
    msg = f"Neither TX nor RX role is UE in {path}: tx_role={tx_role}, rx_role={rx_role}"
    raise ValueError(msg)


def _read_cfr(dataset: h5py.Dataset, snapshot_index: int) -> np.ndarray:
    if dataset.ndim == 6:
        if snapshot_index < 0 or snapshot_index >= dataset.shape[0]:
            msg = f"snapshot_index={snapshot_index} outside dataset shape {dataset.shape}"
            raise IndexError(msg)
        cfr = dataset[snapshot_index]
    elif dataset.ndim == 5:
        cfr = dataset[()]
    else:
        msg = f"CFR dataset must be 5-D or 6-D, got shape {dataset.shape}"
        raise ValueError(msg)
    return np.asarray(cfr, dtype=np.complex64)


def _read_link_valid_mask(
    h5: h5py.File, snapshot_index: int, link_shape: tuple[int, int]
) -> np.ndarray:
    valid = np.ones(link_shape, dtype=np.bool_)
    if "derived/link_valid_mask" in h5:
        valid &= np.asarray(h5["derived/link_valid_mask"][()], dtype=np.bool_)
    if "observation/valid_mask" in h5:
        obs_valid = h5["observation/valid_mask"]
        if obs_valid.ndim == 3:
            valid &= np.asarray(obs_valid[snapshot_index], dtype=np.bool_)
        elif obs_valid.ndim == 2:
            valid &= np.asarray(obs_valid[()], dtype=np.bool_)
    return valid


def _read_indices(h5: h5py.File, key: str, length: int) -> np.ndarray:
    if key in h5:
        return np.asarray(h5[key][()], dtype=np.int64)
    return np.arange(length, dtype=np.int64)


def _decode_scalar(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _finite_complex_mask(left: np.ndarray, right: np.ndarray, link_valid: np.ndarray) -> np.ndarray:
    mask = link_valid[:, :, None, None, None]
    mask = mask & np.isfinite(left.real) & np.isfinite(left.imag)
    return mask & np.isfinite(right.real) & np.isfinite(right.imag)


def _normalized_l2_similarity(
    left: np.ndarray,
    right: np.ndarray,
    mask: np.ndarray,
    *,
    axes: tuple[int, ...] | None = None,
) -> np.ndarray:
    if axes is None:
        axes = tuple(range(1, left.ndim))
    left_values = np.where(mask, left, 0.0).astype(np.float64, copy=False)
    right_values = np.where(mask, right, 0.0).astype(np.float64, copy=False)
    numerator = np.sqrt(np.sum((left_values - right_values) ** 2, axis=axes))
    left_norm = np.sqrt(np.sum(left_values**2, axis=axes))
    right_norm = np.sqrt(np.sum(right_values**2, axis=axes))
    denom = left_norm + right_norm
    count = np.sum(mask, axis=axes)
    similarity = np.full(numerator.shape, np.nan, dtype=np.float32)
    valid = (count > 0) & (denom > EPS)
    similarity[valid] = np.clip(1.0 - numerator[valid] / denom[valid], 0.0, 1.0)
    both_zero = (count > 0) & (denom <= EPS)
    similarity[both_zero] = 1.0
    return similarity


def _weighted_phase_similarity(
    left: np.ndarray,
    right: np.ndarray,
    mag_left: np.ndarray,
    mag_right: np.ndarray,
    mask: np.ndarray,
    *,
    axes: tuple[int, ...] | None = None,
) -> np.ndarray:
    if axes is None:
        axes = tuple(range(1, left.ndim))
    phase_score = 0.5 * (1.0 + np.cos(np.angle(right * np.conj(left))))
    weights = np.sqrt(mag_left * mag_right)
    weights = np.where(mask, weights, 0.0).astype(np.float64, copy=False)
    numerator = np.sum(weights * phase_score, axis=axes)
    denom = np.sum(weights, axis=axes)
    count = np.sum(mask, axis=axes)
    similarity = np.full(numerator.shape, np.nan, dtype=np.float32)
    weighted = denom > EPS
    similarity[weighted] = np.clip(numerator[weighted] / denom[weighted], 0.0, 1.0)

    unweighted = (~weighted) & (count > 0)
    if np.any(unweighted):
        fallback = np.sum(np.where(mask, phase_score, 0.0), axis=axes)
        similarity[unweighted] = np.clip(fallback[unweighted] / count[unweighted], 0.0, 1.0)
    return similarity


def _merge_results(results: list[ShardResult]) -> dict[str, np.ndarray]:
    ordered = sorted(results, key=lambda item: item.shard_index)
    bs_indices = ordered[0].bs_indices
    for item in ordered[1:]:
        if not np.array_equal(item.bs_indices, bs_indices):
            msg = f"Shard {item.shard_index} has different BS indices: {item.bs_indices}"
            raise ValueError(msg)
    return {
        "ue_index": np.concatenate([item.ue_indices for item in ordered]),
        "bs_indices": bs_indices,
        "position_m": np.concatenate([item.positions_m for item in ordered], axis=0),
        "magnitude": np.concatenate([item.magnitude_similarity for item in ordered]),
        "phase": np.concatenate([item.phase_similarity for item in ordered]),
        "i": np.concatenate([item.i_similarity for item in ordered]),
        "q": np.concatenate([item.q_similarity for item in ordered]),
        "magnitude_per_bs": np.concatenate(
            [item.magnitude_similarity_per_bs for item in ordered], axis=0
        ),
        "phase_per_bs": np.concatenate([item.phase_similarity_per_bs for item in ordered], axis=0),
        "i_per_bs": np.concatenate([item.i_similarity_per_bs for item in ordered], axis=0),
        "q_per_bs": np.concatenate([item.q_similarity_per_bs for item in ordered], axis=0),
    }


def _write_outputs(
    args: argparse.Namespace, table: dict[str, np.ndarray], shard_pairs: list[ShardPair]
) -> dict[str, Path]:
    output_paths = _write_metric_outputs(args.output_dir, args, table, shard_pairs)
    output_paths.update(_write_bs_breakdown_outputs(args, table, shard_pairs))
    return output_paths


def _write_metric_outputs(
    output_dir: Path,
    args: argparse.Namespace,
    table: dict[str, np.ndarray],
    shard_pairs: list[ShardPair],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "cfr_similarity_by_ue.csv"
    stats_path = output_dir / "cfr_similarity_stats.csv"
    summary_path = output_dir / "summary.json"
    _write_csv(csv_path, table)
    _write_stats_csv(stats_path, table)
    figure_paths = _plot_heatmaps(args, table, output_dir=output_dir)
    histogram_paths = _plot_histograms(output_dir, table)
    figure_paths = {**figure_paths, **histogram_paths}
    _write_summary(summary_path, args, table, shard_pairs, figure_paths, csv_path, stats_path)
    return {"csv": csv_path, "stats": stats_path, "summary": summary_path, **figure_paths}


def _write_bs_breakdown_outputs(
    args: argparse.Namespace,
    table: dict[str, np.ndarray],
    shard_pairs: list[ShardPair],
) -> dict[str, Path]:
    bs_indices = np.asarray(table["bs_indices"], dtype=np.int64)
    output_paths: dict[str, Path] = {}

    all_columns = np.arange(bs_indices.shape[0], dtype=np.int64)
    all_mean_dir = args.output_dir / "bs_mean_all7"
    all_mean_table = _aggregate_per_bs_table(table, all_columns)
    _write_metric_outputs(all_mean_dir, args, all_mean_table, shard_pairs)
    output_paths["bs_mean_all7"] = all_mean_dir

    subset_columns = _parse_bs_subset(args.bs_subset, bs_indices)
    subset_label = "_".join(str(int(bs_indices[col])) for col in subset_columns)
    subset_dir = args.output_dir / f"bs_mean_subset_{subset_label}"
    subset_table = _aggregate_per_bs_table(table, subset_columns)
    _write_metric_outputs(subset_dir, args, subset_table, shard_pairs)
    output_paths["bs_mean_subset"] = subset_dir

    per_bs_root = args.output_dir / "per_bs"
    for column, global_bs in enumerate(bs_indices):
        bs_dir = per_bs_root / f"bs_{int(global_bs):02d}"
        bs_table = _single_bs_table(table, column)
        _write_metric_outputs(bs_dir, args, bs_table, shard_pairs)
    output_paths["per_bs_root"] = per_bs_root
    return output_paths


def _aggregate_per_bs_table(
    table: dict[str, np.ndarray],
    columns: np.ndarray,
) -> dict[str, np.ndarray]:
    return {
        "ue_index": table["ue_index"],
        "position_m": table["position_m"],
        "magnitude": np.mean(table["magnitude_per_bs"][:, columns], axis=1),
        "phase": np.mean(table["phase_per_bs"][:, columns], axis=1),
        "i": np.mean(table["i_per_bs"][:, columns], axis=1),
        "q": np.mean(table["q_per_bs"][:, columns], axis=1),
    }


def _single_bs_table(table: dict[str, np.ndarray], column: int) -> dict[str, np.ndarray]:
    return {
        "ue_index": table["ue_index"],
        "position_m": table["position_m"],
        "magnitude": table["magnitude_per_bs"][:, column],
        "phase": table["phase_per_bs"][:, column],
        "i": table["i_per_bs"][:, column],
        "q": table["q_per_bs"][:, column],
    }


def _parse_bs_subset(raw: str, bs_indices: np.ndarray) -> np.ndarray:
    requested = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not requested:
        msg = "--bs-subset must contain at least one BS index."
        raise ValueError(msg)
    columns: list[int] = []
    for bs_index in requested:
        matches = np.flatnonzero(bs_indices == bs_index)
        if matches.size == 0:
            msg = (
                f"Requested BS {bs_index} not found in available BS indices {bs_indices.tolist()}."
            )
            raise ValueError(msg)
        columns.append(int(matches[0]))
    return np.asarray(columns, dtype=np.int64)


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
        for index, position, mag, phase, i_value, q_value in zip(
            table["ue_index"],
            positions,
            table["magnitude"],
            table["phase"],
            table["i"],
            table["q"],
            strict=True,
        ):
            writer.writerow(
                {
                    "ue_index": int(index),
                    "x_m": float(position[0]),
                    "y_m": float(position[1]),
                    "z_m": float(position[2]),
                    "magnitude_similarity": float(mag),
                    "phase_similarity": float(phase),
                    "i_similarity": float(i_value),
                    "q_similarity": float(q_value),
                }
            )


def _write_stats_csv(path: Path, table: dict[str, np.ndarray]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["metric", "min", "max", "mean"],
        )
        writer.writeheader()
        for key in ("magnitude", "phase", "i", "q"):
            stats = _metric_summary(table[key])
            writer.writerow(
                {
                    "metric": key,
                    "min": stats["min"],
                    "max": stats["max"],
                    "mean": stats["mean"],
                }
            )


def _write_summary(
    path: Path,
    args: argparse.Namespace,
    table: dict[str, np.ndarray],
    shard_pairs: list[ShardPair],
    figure_paths: dict[str, Path],
    csv_path: Path,
    stats_path: Path,
) -> None:
    metric_keys = ("magnitude", "phase", "i", "q")
    summary = {
        "left": args.left.as_posix(),
        "right": args.right.as_posix(),
        "dataset": args.dataset,
        "snapshot_index": args.snapshot_index,
        "floorplan_image": args.floorplan_image.as_posix(),
        "floorplan_meta": args.floorplan_meta.as_posix(),
        "shard_count": len(shard_pairs),
        "missing_shards": {
            "left": [pair.shard_index for pair in shard_pairs if pair.left is None],
            "right": [pair.shard_index for pair in shard_pairs if pair.right is None],
            "policy": "UEs from a shard missing on either side are assigned similarity 0.0.",
        },
        "ue_count": int(table["ue_index"].size),
        "similarity_range": [0.0, 1.0],
        "rendering": {
            "heatmap_grid": "rectangular UE x/y bounding box",
            "interpolation": "bilinear over a regular grid with missing UE cells set to zero",
            "show_samples": bool(args.show_samples),
            "x_range_m": [
                float(np.min(table["position_m"][:, 0])),
                float(np.max(table["position_m"][:, 0])),
            ],
            "y_range_m": [
                float(np.min(table["position_m"][:, 1])),
                float(np.max(table["position_m"][:, 1])),
            ],
            "x_spacing_m": _infer_grid_spacing(table["position_m"][:, 0]),
            "y_spacing_m": _infer_grid_spacing(table["position_m"][:, 1]),
            "missing_grid_cell_value": 0.0,
        },
        "similarity_definitions": {
            "magnitude": (
                "1 - ||abs(H_left)-abs(H_right)||_2 / "
                "(||abs(H_left)||_2 + ||abs(H_right)||_2), clipped to [0,1]"
            ),
            "phase": (
                "magnitude-weighted mean of "
                "(1 + cos(angle(H_right*conj(H_left)))) / 2, clipped to [0,1]"
            ),
            "i": (
                "1 - ||real(H_left)-real(H_right)||_2 / "
                "(||real(H_left)||_2 + ||real(H_right)||_2), clipped to [0,1]"
            ),
            "q": (
                "1 - ||imag(H_left)-imag(H_right)||_2 / "
                "(||imag(H_left)||_2 + ||imag(H_right)||_2), clipped to [0,1]"
            ),
        },
        "metrics": {key: _metric_summary(table[key]) for key in metric_keys},
        "outputs": {
            "csv": csv_path.as_posix(),
            "stats": stats_path.as_posix(),
            **{key: value.as_posix() for key, value in figure_paths.items()},
        },
    }
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def _metric_summary(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "finite_count": 0,
            "min": float("nan"),
            "median": float("nan"),
            "mean": float("nan"),
            "max": float("nan"),
        }
    return {
        "finite_count": int(finite.size),
        "min": float(np.min(finite)),
        "median": float(np.median(finite)),
        "mean": float(np.mean(finite)),
        "max": float(np.max(finite)),
    }


def _plot_heatmaps(
    args: argparse.Namespace,
    table: dict[str, np.ndarray],
    *,
    output_dir: Path,
) -> dict[str, Path]:
    floorplan = np.asarray(Image.open(args.floorplan_image).convert("RGB"))
    meta = json.loads(args.floorplan_meta.read_text(encoding="utf-8"))
    extent = _floorplan_extent(meta, floorplan)

    specs = [
        ("magnitude", "Magnitude Similarity", "cfr_similarity_magnitude_floorplan.png"),
        ("phase", "Phase Similarity", "cfr_similarity_phase_floorplan.png"),
        ("i", "I Similarity", "cfr_similarity_i_floorplan.png"),
        ("q", "Q Similarity", "cfr_similarity_q_floorplan.png"),
    ]
    paths: dict[str, Path] = {}
    for key, title, filename in specs:
        output_path = output_dir / filename
        _plot_single_heatmap(
            floorplan,
            extent,
            table["position_m"][:, 0],
            table["position_m"][:, 1],
            table[key],
            title=title,
            output_path=output_path,
            image_origin=args.image_origin,
            heatmap_alpha=args.heatmap_alpha,
            point_size=args.point_size,
            show_samples=args.show_samples,
        )
        paths[key] = output_path

    combined_path = output_dir / "cfr_similarity_four_panel_floorplan.png"
    _plot_combined_heatmap(
        floorplan,
        extent,
        table,
        specs=specs,
        output_path=combined_path,
        image_origin=args.image_origin,
        heatmap_alpha=args.heatmap_alpha,
        point_size=args.point_size,
        show_samples=args.show_samples,
    )
    paths["combined"] = combined_path
    return paths


def _plot_histograms(output_dir: Path, table: dict[str, np.ndarray]) -> dict[str, Path]:
    specs = [
        ("magnitude", "Magnitude Similarity", "cfr_similarity_magnitude_histogram.png"),
        ("phase", "Phase Similarity", "cfr_similarity_phase_histogram.png"),
        ("i", "I Similarity", "cfr_similarity_i_histogram.png"),
        ("q", "Q Similarity", "cfr_similarity_q_histogram.png"),
    ]
    paths: dict[str, Path] = {}
    for key, title, filename in specs:
        output_path = output_dir / filename
        _plot_single_histogram(table[key], title=title, output_path=output_path)
        paths[f"{key}_histogram"] = output_path

    combined_path = output_dir / "cfr_similarity_histograms.png"
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), squeeze=False)
    for axis, (key, title, _filename) in zip(axes.ravel(), specs, strict=True):
        _draw_histogram(axis, table[key], title=title)
    fig.savefig(combined_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    paths["histograms"] = combined_path
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
    axis.set_ylabel("UE count")
    axis.set_xlim(0.0, 1.0)
    axis.grid(axis="y", alpha=0.25)


def _floorplan_extent(
    meta: dict[str, Any], floorplan: np.ndarray
) -> tuple[float, float, float, float]:
    origin_x, origin_y = [float(value) for value in meta.get("origin_xy_m", [0.0, 0.0])]
    if "extent_xy_m" in meta:
        width_m, height_m = [float(value) for value in meta["extent_xy_m"]]
    else:
        resolution = float(meta["resolution_m_per_pixel"])
        height_px, width_px = floorplan.shape[:2]
        width_m = width_px * resolution
        height_m = height_px * resolution
    return origin_x, origin_x + width_m, origin_y, origin_y + height_m


def _plot_single_heatmap(
    floorplan: np.ndarray,
    extent: tuple[float, float, float, float],
    x_m: np.ndarray,
    y_m: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    output_path: Path,
    image_origin: str,
    heatmap_alpha: float,
    point_size: float,
    show_samples: bool,
) -> None:
    fig, axis = plt.subplots(figsize=(8.0, 8.5))
    image = _draw_floorplan_similarity(
        axis,
        floorplan,
        extent,
        x_m,
        y_m,
        values,
        title=title,
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
    specs: list[tuple[str, str, str]],
    output_path: Path,
    image_origin: str,
    heatmap_alpha: float,
    point_size: float,
    show_samples: bool,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 13.5), squeeze=False)
    image = None
    for axis, (key, title, _filename) in zip(axes.ravel(), specs, strict=True):
        image = _draw_floorplan_similarity(
            axis,
            floorplan,
            extent,
            table["position_m"][:, 0],
            table["position_m"][:, 1],
            table[key],
            title=title,
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


def _draw_floorplan_similarity(
    axis: plt.Axes,
    floorplan: np.ndarray,
    extent: tuple[float, float, float, float],
    x_m: np.ndarray,
    y_m: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    image_origin: str,
    heatmap_alpha: float,
    point_size: float,
    show_samples: bool,
) -> Any:
    axis.imshow(floorplan, extent=extent, origin=image_origin)
    grid, grid_extent = _rasterize_similarity_grid(x_m, y_m, values)
    image = axis.imshow(
        grid,
        extent=grid_extent,
        origin="lower",
        cmap="viridis",
        interpolation="bilinear",
        alpha=heatmap_alpha,
        vmin=0.0,
        vmax=1.0,
    )
    if show_samples:
        axis.scatter(
            x_m,
            y_m,
            c=values,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            s=point_size,
            linewidths=0.0,
            alpha=0.9,
        )
    axis.set_title(title)
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_xlim(extent[0], extent[1])
    axis.set_ylim(extent[2], extent[3])
    axis.set_aspect("equal", adjustable="box")
    return image


def _rasterize_similarity_grid(
    x_m: np.ndarray,
    y_m: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    finite = np.isfinite(x_m) & np.isfinite(y_m) & np.isfinite(values)
    if np.count_nonzero(finite) < 3:
        msg = "Need at least three finite UE similarity samples to render a heatmap."
        raise ValueError(msg)

    x = x_m[finite]
    y = y_m[finite]
    z = np.clip(values[finite], 0.0, 1.0)

    x_min = float(np.min(x))
    x_max = float(np.max(x))
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    dx = _infer_grid_spacing(x)
    dy = _infer_grid_spacing(y)
    x_grid = _regular_axis(x_min, x_max, dx)
    y_grid = _regular_axis(y_min, y_max, dy)
    nx = x_grid.size
    ny = y_grid.size

    grid_sum = np.zeros((ny, nx), dtype=np.float64)
    grid_count = np.zeros((ny, nx), dtype=np.int64)
    ix = np.rint((x - x_min) / dx).astype(np.int64)
    iy = np.rint((y - y_min) / dy).astype(np.int64)
    inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    np.add.at(grid_sum, (iy[inside], ix[inside]), z[inside])
    np.add.at(grid_count, (iy[inside], ix[inside]), 1)

    grid = np.zeros((ny, nx), dtype=np.float32)
    sampled = grid_count > 0
    grid[sampled] = (grid_sum[sampled] / grid_count[sampled]).astype(np.float32)
    return grid, (float(x_grid[0]), float(x_grid[-1]), float(y_grid[0]), float(y_grid[-1]))


def _regular_axis(min_value: float, max_value: float, spacing: float) -> np.ndarray:
    count = int(round((max_value - min_value) / spacing)) + 1
    return min_value + np.arange(count, dtype=np.float64) * spacing


def _infer_grid_spacing(values: np.ndarray) -> float:
    unique = np.unique(np.round(values.astype(np.float64), decimals=6))
    deltas = np.diff(unique)
    positive = deltas[deltas > 1e-6]
    if positive.size == 0:
        return 1.0
    return float(np.median(positive))


if __name__ == "__main__":
    raise SystemExit(main())
