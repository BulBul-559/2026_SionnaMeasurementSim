"""HDF5 reader helpers."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from sionna_measurement_sim.io.schema_validator import validate_hdf5_contract


@dataclass(frozen=True)
class ManifestDatasetFragment:
    """Dataset readback for one manifest result entry."""

    source_path: Path
    dataset_path: str
    data: Any
    global_ue_indices: tuple[int, ...]
    global_tx_indices: tuple[int, ...]
    global_rx_indices: tuple[int, ...]
    shard_id: str = ""
    fragment_id: str = ""
    append_start: int | None = None
    append_count: int | None = None
    bundled: bool = False


def read_dataset(path: str | Path, dataset_path: str) -> Any:
    """Read a dataset value from an HDF5 file."""

    validate_hdf5_contract(path)
    with h5py.File(path, "r") as h5:
        value = h5[_clean_dataset_path(dataset_path)][()]
    return _decode(value)


def read_metadata(path: str | Path) -> dict[str, Any]:
    """Read all scalar metadata fields."""

    validate_hdf5_contract(path)
    with h5py.File(path, "r") as h5:
        return {name: _decode(dataset[()]) for name, dataset in h5["meta"].items()}


def read_truth_cfr(path: str | Path) -> Any:
    """Read the contract-compliant truth CFR dataset."""

    validate_hdf5_contract(path)
    with h5py.File(path, "r") as h5:
        return h5["channel/truth/cfr"][()]


def read_bundle_index(path: str | Path) -> dict[str, Any]:
    """Read appendable bundle fragment offsets and global UE indices."""

    validate_hdf5_contract(path)
    with h5py.File(path, "r") as h5:
        if "bundle" not in h5:
            msg = f"{path} is not an HDF5 result bundle"
            raise KeyError(msg)
        return {
            "fragment_count": int(h5["bundle/fragment_count"][()]),
            "ue_count": int(h5["bundle/ue_count"][()]),
            "ue_axis_role": _decode(h5["bundle/ue_axis_role"][()]),
            "fragment_id": [_decode(value) for value in h5["bundle/fragment_id"][()]],
            "shard_offsets": np.asarray(h5["bundle/shard_offsets"][()]),
            "global_ue_indices": np.asarray(h5["bundle/global_ue_indices"][()]),
        }


def read_bundle_fragment_dataset(
    path: str | Path,
    dataset_path: str,
    *,
    fragment_index: int | None = None,
    fragment_id: str | None = None,
) -> Any:
    """Read one dataset for a single appendable bundle fragment.

    Datasets that are appended along the resolved UE axis are sliced with
    `/bundle/shard_offsets`. Shared datasets are returned whole, and
    per-fragment sidecar datasets under `/bundle/fragments/<id>/...` take
    precedence when a value could not be safely appended at the root.
    """

    validate_hdf5_contract(path)
    with h5py.File(path, "r") as h5:
        if "bundle" not in h5:
            msg = f"{path} is not an HDF5 result bundle"
            raise KeyError(msg)
        row = _resolve_bundle_fragment_row(
            h5,
            fragment_index=fragment_index,
            fragment_id=fragment_id,
        )
        return _read_bundle_fragment_dataset_from_open_h5(h5, dataset_path, row)


def iter_manifest_dataset(
    path: str | Path,
    dataset_path: str,
) -> Iterator[ManifestDatasetFragment]:
    """Yield one dataset chunk per result entry from a file, run dir, or manifest.

    Default shard manifests read each `result_h5` directly. Bundle manifests read
    the referenced `bundle_h5` and `bundle_fragment_id`, slicing appendable root
    datasets through `/bundle/shard_offsets`.
    """

    dataset_path = _clean_dataset_path(dataset_path)
    root = Path(path)
    manifest_path = _find_manifest_path(root)
    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_dir = _manifest_run_dir(manifest_path)
        yield from _iter_manifest_entries(manifest, run_dir, dataset_path)
        return
    if root.is_file():
        yield _single_h5_fragment(root, dataset_path)
        return
    search_root = root / "results" if (root / "results").is_dir() else root
    for h5_path in sorted(search_root.glob("result*.h5")):
        yield _single_h5_fragment(h5_path, dataset_path)
    if not (search_root / "results.h5").exists():
        return
    yield _single_h5_fragment(search_root / "results.h5", dataset_path)


def iter_link_labels(path: str | Path):
    """Yield compact `/labels/link` tables from one HDF5 file or sharded run dir."""

    root = Path(path)
    if root.is_file():
        yield read_link_labels(root)
        return
    manifest = root / "manifest" / "manifest.json"
    if manifest.exists():
        import json

        data = json.loads(manifest.read_text(encoding="utf-8"))
        for item in data.get("results", []):
            result_h5 = item.get("result_h5", "")
            if result_h5:
                yield read_link_labels(result_h5)
        return
    results_dir = root / "results"
    for h5_path in sorted(results_dir.glob("*.h5")):
        yield read_link_labels(h5_path)


def read_link_labels(path: str | Path) -> dict[str, Any]:
    """Read the compact RT labels-only `/labels/link` table."""

    validate_hdf5_contract(path)
    with h5py.File(path, "r") as h5:
        if "labels/link" not in h5:
            msg = f"{path} does not contain /labels/link"
            raise KeyError(msg)
        return {
            name: np.asarray(dataset[()])
            for name, dataset in h5["labels/link"].items()
        }


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _find_manifest_path(path: Path) -> Path | None:
    if path.is_file() and path.name == "manifest.json":
        return path
    direct_manifest = path / "manifest.json"
    if direct_manifest.exists():
        return direct_manifest
    nested_manifest = path / "manifest" / "manifest.json"
    if nested_manifest.exists():
        return nested_manifest
    return None


def _manifest_run_dir(manifest_path: Path) -> Path:
    if manifest_path.parent.name == "manifest":
        return manifest_path.parent.parent
    return manifest_path.parent


def _iter_manifest_entries(
    manifest: dict[str, Any],
    run_dir: Path,
    dataset_path: str,
) -> Iterator[ManifestDatasetFragment]:
    entries = list(manifest.get("results", []))
    if entries:
        for item in entries:
            yield _manifest_entry_fragment(item, run_dir, dataset_path)
        return
    result_h5 = manifest.get("results_h5")
    if result_h5:
        yield _single_h5_fragment(
            _resolve_manifest_artifact(run_dir, Path(str(result_h5))),
            dataset_path,
        )


def _manifest_entry_fragment(
    item: dict[str, Any],
    run_dir: Path,
    dataset_path: str,
) -> ManifestDatasetFragment:
    bundle_h5 = item.get("bundle_h5")
    if bundle_h5:
        source_path = _resolve_manifest_artifact(run_dir, Path(str(bundle_h5)))
        fragment_id = str(item.get("bundle_fragment_id", ""))
        if not fragment_id:
            msg = f"Bundled manifest result for {source_path} is missing bundle_fragment_id"
            raise KeyError(msg)
        return ManifestDatasetFragment(
            source_path=source_path,
            dataset_path=dataset_path,
            data=read_bundle_fragment_dataset(
                source_path,
                dataset_path,
                fragment_id=fragment_id,
            ),
            global_ue_indices=_int_tuple(item.get("global_ue_indices")),
            global_tx_indices=_int_tuple(item.get("global_tx_indices")),
            global_rx_indices=_int_tuple(item.get("global_rx_indices")),
            shard_id=str(item.get("shard_id", "")),
            fragment_id=fragment_id,
            append_start=_optional_int(item.get("append_start")),
            append_count=_optional_int(item.get("append_count")),
            bundled=True,
        )
    result_h5 = item.get("result_h5")
    if not result_h5:
        msg = "Manifest result entry must contain result_h5 or bundle_h5"
        raise KeyError(msg)
    source_path = _resolve_manifest_artifact(run_dir, Path(str(result_h5)))
    return ManifestDatasetFragment(
        source_path=source_path,
        dataset_path=dataset_path,
        data=read_dataset(source_path, dataset_path),
        global_ue_indices=_int_tuple(item.get("global_ue_indices")),
        global_tx_indices=_int_tuple(item.get("global_tx_indices")),
        global_rx_indices=_int_tuple(item.get("global_rx_indices")),
        shard_id=str(item.get("shard_id", "")),
        append_start=_optional_int(item.get("append_start")),
        append_count=_optional_int(item.get("append_count")),
        bundled=False,
    )


def _single_h5_fragment(path: Path, dataset_path: str) -> ManifestDatasetFragment:
    return ManifestDatasetFragment(
        source_path=path,
        dataset_path=dataset_path,
        data=read_dataset(path, dataset_path),
        global_ue_indices=(),
        global_tx_indices=(),
        global_rx_indices=(),
    )


def _resolve_manifest_artifact(run_dir: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return run_dir / path


def _int_tuple(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    return tuple(int(item) for item in value)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _read_bundle_fragment_dataset_from_open_h5(
    h5: h5py.File,
    dataset_path: str,
    row: int,
) -> Any:
    dataset_path = _clean_dataset_path(dataset_path)
    fragment_id = _decode(h5["bundle/fragment_id"][row])
    sidecar_path = f"bundle/fragments/{fragment_id}/{dataset_path}"
    if sidecar_path in h5:
        return h5[sidecar_path][()]
    if dataset_path not in h5:
        msg = f"{h5.filename} does not contain /{dataset_path}"
        raise KeyError(msg)
    dataset = h5[dataset_path]
    offset = np.asarray(h5["bundle/shard_offsets"][row], dtype=np.int64)
    selection = _bundle_fragment_selection(
        h5,
        dataset_path,
        dataset,
        append_start=int(offset[0]),
        append_count=int(offset[1]),
    )
    if selection is None:
        return dataset[()]
    return dataset[selection]


def _resolve_bundle_fragment_row(
    h5: h5py.File,
    *,
    fragment_index: int | None,
    fragment_id: str | None,
) -> int:
    if (fragment_index is None) == (fragment_id is None):
        msg = "Provide exactly one of fragment_index or fragment_id"
        raise ValueError(msg)
    fragment_count = int(h5["bundle/fragment_count"][()])
    if fragment_index is not None:
        row = int(fragment_index)
        if row < 0 or row >= fragment_count:
            msg = f"fragment_index {row} is outside [0, {fragment_count})"
            raise IndexError(msg)
        return row
    fragment_ids = [_decode(value) for value in h5["bundle/fragment_id"][()]]
    try:
        return fragment_ids.index(str(fragment_id))
    except ValueError as exc:
        msg = f"Unknown bundle fragment_id {fragment_id!r}"
        raise KeyError(msg) from exc


def _bundle_fragment_selection(
    h5: h5py.File,
    dataset_path: str,
    dataset: h5py.Dataset,
    *,
    append_start: int,
    append_count: int,
) -> tuple[slice, ...] | None:
    axis, scale = _bundle_dataset_append_axis(h5, dataset_path, dataset)
    if axis is None:
        return None
    scaled_start = append_start * scale
    scaled_count = append_count * scale
    selection = [slice(None)] * dataset.ndim
    selection[axis] = slice(scaled_start, scaled_start + scaled_count)
    return tuple(selection)


def _bundle_dataset_append_axis(
    h5: h5py.File,
    dataset_path: str,
    dataset: h5py.Dataset,
) -> tuple[int | None, int]:
    if dataset.ndim == 0:
        return None, 1
    role = _decode(h5["bundle/ue_axis_role"][()])
    path_axes = {
        "topology/tx_positions_m": ("tx", 0),
        "topology/tx_labels": ("tx", 0),
        "topology/rx_positions_m": ("rx", 0),
        "topology/rx_labels": ("rx", 0),
        "devices/tx_velocity_mps": ("tx", 1),
        "devices/tx_orientation_rad": ("tx", 1),
        "devices/rx_velocity_mps": ("rx", 1),
        "devices/rx_orientation_rad": ("rx", 1),
    }
    if dataset_path in path_axes:
        axis_role, axis = path_axes[dataset_path]
        if axis_role == role:
            return axis, 1
        return None, 1

    tx_count = int(h5["topology/tx_positions_m"].shape[0]) if "topology" in h5 else 0
    rx_count = int(h5["topology/rx_positions_m"].shape[0]) if "topology" in h5 else 0
    if (
        dataset_path.startswith("labels/link/")
        and dataset.ndim >= 1
        and tx_count > 0
        and rx_count > 0
        and dataset.shape[0] == tx_count * rx_count
    ):
        scale = rx_count if role == "tx" else tx_count
        return 0, int(scale)

    order = _attr_to_string(dataset.attrs.get("index_order", ""))
    if not order:
        return None, 1
    wanted = {"tx", "ul_tx"} if role == "tx" else {"rx", "ul_rx"}
    bundle_ue_count = int(h5["bundle/ue_count"][()])
    for axis, token in enumerate(part.strip() for part in order.split(",")):
        if axis >= dataset.ndim:
            continue
        if token in wanted and int(dataset.shape[axis]) == bundle_ue_count:
            return axis, 1
    return None, 1


def _clean_dataset_path(dataset_path: str) -> str:
    path = str(dataset_path).strip("/")
    if not path:
        msg = "dataset_path must not be empty"
        raise ValueError(msg)
    return path


def _attr_to_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)
