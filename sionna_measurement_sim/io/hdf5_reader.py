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
    append_axis: int | None = None
    ue_axis_role: str = ""
    bundled: bool = False


@dataclass(frozen=True)
class ManifestDatasetBatch:
    """Dataset readback for a training-style batch of manifest fragments."""

    dataset_path: str
    data: Any
    fragments: tuple[ManifestDatasetFragment, ...]
    global_ue_indices: tuple[int, ...]
    global_tx_indices: tuple[int, ...]
    global_rx_indices: tuple[int, ...]
    shard_ids: tuple[str, ...]
    fragment_ids: tuple[str, ...]
    source_paths: tuple[Path, ...]
    append_axis: int | None = None
    ue_axis_role: str = ""


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


def iter_manifest_dataset_batches(
    path: str | Path,
    dataset_path: str,
    *,
    max_fragments: int | None = 32,
    max_ue: int | None = None,
) -> Iterator[ManifestDatasetBatch]:
    """Yield concatenated manifest dataset batches for training-style reads.

    Fragments are concatenated along the resolved UE append axis when the dataset
    carries one. Shared metadata datasets are returned once per batch when all
    fragment values are identical.
    """

    if max_fragments is not None and int(max_fragments) < 1:
        max_fragments = None
    if max_ue is not None and int(max_ue) < 1:
        max_ue = None
    pending: list[ManifestDatasetFragment] = []
    pending_ue = 0
    for fragment in iter_manifest_dataset(path, dataset_path):
        fragment_ue = len(fragment.global_ue_indices)
        would_exceed_fragments = (
            max_fragments is not None and len(pending) >= int(max_fragments)
        )
        would_exceed_ue = (
            max_ue is not None
            and pending
            and pending_ue + fragment_ue > int(max_ue)
        )
        if pending and (would_exceed_fragments or would_exceed_ue):
            yield _make_manifest_dataset_batch(pending)
            pending = []
            pending_ue = 0
        pending.append(fragment)
        pending_ue += fragment_ue
    if pending:
        yield _make_manifest_dataset_batch(pending)


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
        current_bundle_path: Path | None = None
        current_bundle_h5: h5py.File | None = None
        try:
            for item in entries:
                bundle_h5 = item.get("bundle_h5")
                if bundle_h5:
                    source_path = _resolve_manifest_artifact(
                        run_dir,
                        Path(str(bundle_h5)),
                    )
                    if current_bundle_path != source_path:
                        if current_bundle_h5 is not None:
                            current_bundle_h5.close()
                        validate_hdf5_contract(source_path)
                        current_bundle_h5 = h5py.File(source_path, "r")
                        current_bundle_path = source_path
                    yield _manifest_bundle_entry_fragment(
                        item,
                        source_path,
                        dataset_path,
                        current_bundle_h5,
                    )
                    continue
                if current_bundle_h5 is not None:
                    current_bundle_h5.close()
                    current_bundle_h5 = None
                    current_bundle_path = None
                yield _manifest_shard_entry_fragment(item, run_dir, dataset_path)
        finally:
            if current_bundle_h5 is not None:
                current_bundle_h5.close()
        return
    result_h5 = manifest.get("results_h5")
    if result_h5:
        yield _single_h5_fragment(
            _resolve_manifest_artifact(run_dir, Path(str(result_h5))),
            dataset_path,
        )


def _manifest_bundle_entry_fragment(
    item: dict[str, Any],
    source_path: Path,
    dataset_path: str,
    h5: h5py.File,
) -> ManifestDatasetFragment:
    fragment_id = str(item.get("bundle_fragment_id", ""))
    if not fragment_id:
        msg = f"Bundled manifest result for {source_path} is missing bundle_fragment_id"
        raise KeyError(msg)
    row = _resolve_bundle_fragment_row(h5, fragment_index=None, fragment_id=fragment_id)
    data, append_axis = _read_bundle_fragment_dataset_with_axis_from_open_h5(
        h5,
        dataset_path,
        row,
    )
    return ManifestDatasetFragment(
        source_path=source_path,
        dataset_path=dataset_path,
        data=_decode(data),
        global_ue_indices=_int_tuple(item.get("global_ue_indices")),
        global_tx_indices=_int_tuple(item.get("global_tx_indices")),
        global_rx_indices=_int_tuple(item.get("global_rx_indices")),
        shard_id=str(item.get("shard_id", "")),
        fragment_id=fragment_id,
        append_start=_optional_int(item.get("append_start")),
        append_count=_optional_int(item.get("append_count")),
        append_axis=append_axis,
        ue_axis_role=_decode(h5["bundle/ue_axis_role"][()]),
        bundled=True,
    )


def _manifest_shard_entry_fragment(
    item: dict[str, Any],
    run_dir: Path,
    dataset_path: str,
) -> ManifestDatasetFragment:
    result_h5 = item.get("result_h5")
    if not result_h5:
        msg = "Manifest result entry must contain result_h5 or bundle_h5"
        raise KeyError(msg)
    source_path = _resolve_manifest_artifact(run_dir, Path(str(result_h5)))
    return _single_h5_fragment(
        source_path,
        dataset_path,
        global_ue_indices=_int_tuple(item.get("global_ue_indices")),
        global_tx_indices=_int_tuple(item.get("global_tx_indices")),
        global_rx_indices=_int_tuple(item.get("global_rx_indices")),
        shard_id=str(item.get("shard_id", "")),
        append_start=_optional_int(item.get("append_start")),
        append_count=_optional_int(item.get("append_count")),
    )


def _single_h5_fragment(
    path: Path,
    dataset_path: str,
    *,
    global_ue_indices: tuple[int, ...] = (),
    global_tx_indices: tuple[int, ...] = (),
    global_rx_indices: tuple[int, ...] = (),
    shard_id: str = "",
    append_start: int | None = None,
    append_count: int | None = None,
) -> ManifestDatasetFragment:
    validate_hdf5_contract(path)
    dataset_path = _clean_dataset_path(dataset_path)
    with h5py.File(path, "r") as h5:
        if dataset_path not in h5:
            msg = f"{path} does not contain /{dataset_path}"
            raise KeyError(msg)
        dataset = h5[dataset_path]
        data = dataset[()]
        ue_axis_role = _h5_ue_axis_role(h5)
        append_axis = _h5_dataset_append_axis(
            h5,
            dataset_path,
            dataset,
            ue_axis_role=ue_axis_role,
        )
    return ManifestDatasetFragment(
        source_path=path,
        dataset_path=dataset_path,
        data=_decode(data),
        global_ue_indices=global_ue_indices,
        global_tx_indices=global_tx_indices,
        global_rx_indices=global_rx_indices,
        shard_id=shard_id,
        append_start=append_start,
        append_count=append_count,
        append_axis=append_axis,
        ue_axis_role=ue_axis_role,
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


def _make_manifest_dataset_batch(
    fragments: list[ManifestDatasetFragment],
) -> ManifestDatasetBatch:
    if not fragments:
        msg = "Cannot build a manifest dataset batch from no fragments"
        raise ValueError(msg)
    first = fragments[0]
    append_axis = first.append_axis
    ue_axis_role = first.ue_axis_role
    if any(fragment.ue_axis_role != ue_axis_role for fragment in fragments):
        msg = f"Cannot batch /{first.dataset_path}: mixed resolved UE axis roles"
        raise ValueError(msg)
    arrays = [np.asarray(fragment.data) for fragment in fragments]
    if append_axis is not None and all(
        fragment.append_axis == append_axis for fragment in fragments
    ):
        data = _concatenate_batch_arrays(arrays, append_axis)
    elif len(arrays) == 1 or all(
        _array_equal_for_batch(arrays[0], array) for array in arrays[1:]
    ):
        data = fragments[0].data
        append_axis = None
    else:
        msg = (
            f"Cannot batch /{first.dataset_path}: fragments do not share a resolved "
            "UE append axis and are not identical shared values"
        )
        raise ValueError(msg)
    return ManifestDatasetBatch(
        dataset_path=first.dataset_path,
        data=data,
        fragments=tuple(fragments),
        global_ue_indices=tuple(
            int(index)
            for fragment in fragments
            for index in fragment.global_ue_indices
        ),
        global_tx_indices=_batch_role_indices(
            fragments,
            role="tx",
            ue_axis_role=ue_axis_role,
        ),
        global_rx_indices=_batch_role_indices(
            fragments,
            role="rx",
            ue_axis_role=ue_axis_role,
        ),
        shard_ids=tuple(fragment.shard_id for fragment in fragments),
        fragment_ids=tuple(fragment.fragment_id for fragment in fragments),
        source_paths=tuple(fragment.source_path for fragment in fragments),
        append_axis=append_axis,
        ue_axis_role=ue_axis_role,
    )


def _array_equal_for_batch(left: np.ndarray, right: np.ndarray) -> bool:
    try:
        return bool(np.array_equal(left, right, equal_nan=True))
    except TypeError:
        return bool(np.array_equal(left, right))


def _concatenate_batch_arrays(arrays: list[np.ndarray], append_axis: int) -> np.ndarray:
    reference = arrays[0]
    if append_axis >= reference.ndim:
        msg = f"append_axis {append_axis} is out of bounds for rank {reference.ndim}"
        raise ValueError(msg)
    for array in arrays[1:]:
        if array.ndim != reference.ndim:
            msg = "Cannot batch arrays with different ranks"
            raise ValueError(msg)
        for axis, (left, right) in enumerate(zip(reference.shape, array.shape, strict=True)):
            if axis == append_axis:
                continue
            if int(left) != int(right):
                msg = (
                    "Cannot batch arrays whose non-append dimensions differ: "
                    f"axis {axis} has {left} != {right}"
                )
                raise ValueError(msg)
    return np.concatenate(arrays, axis=append_axis)


def _batch_role_indices(
    fragments: list[ManifestDatasetFragment],
    *,
    role: str,
    ue_axis_role: str,
) -> tuple[int, ...]:
    attr = "global_tx_indices" if role == "tx" else "global_rx_indices"
    values = [getattr(fragment, attr) for fragment in fragments]
    if role == ue_axis_role:
        return tuple(int(index) for item in values for index in item)
    return _unique_int_tuple(index for item in values for index in item)


def _unique_int_tuple(values) -> tuple[int, ...]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        item = int(value)
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return tuple(ordered)


def _read_bundle_fragment_dataset_with_axis_from_open_h5(
    h5: h5py.File,
    dataset_path: str,
    row: int,
) -> tuple[Any, int | None]:
    dataset_path = _clean_dataset_path(dataset_path)
    fragment_id = _decode(h5["bundle/fragment_id"][row])
    sidecar_path = f"bundle/fragments/{fragment_id}/{dataset_path}"
    if sidecar_path in h5:
        dataset = h5[sidecar_path]
        return dataset[()], _h5_dataset_append_axis(
            h5,
            dataset_path,
            dataset,
            ue_axis_role=_decode(h5["bundle/ue_axis_role"][()]),
        )
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
    append_axis, _ = _bundle_dataset_append_axis(h5, dataset_path, dataset)
    if selection is None:
        return dataset[()], None
    return dataset[selection], append_axis


def _read_bundle_fragment_dataset_from_open_h5(
    h5: h5py.File,
    dataset_path: str,
    row: int,
) -> Any:
    data, _ = _read_bundle_fragment_dataset_with_axis_from_open_h5(
        h5,
        dataset_path,
        row,
    )
    return data


def _h5_ue_axis_role(h5: h5py.File) -> str:
    tx_role = _read_optional_h5_string(h5, "link/tx_role", "tx")
    rx_role = _read_optional_h5_string(h5, "link/rx_role", "rx")
    if tx_role == "ue":
        return "tx"
    if rx_role == "ue":
        return "rx"
    if "bundle/ue_axis_role" in h5:
        return _decode(h5["bundle/ue_axis_role"][()])
    return "tx"


def _read_optional_h5_string(h5: h5py.File, path: str, default: str) -> str:
    if path not in h5:
        return default
    return _decode(h5[path][()])


def _h5_dataset_append_axis(
    h5: h5py.File,
    dataset_path: str,
    dataset: h5py.Dataset,
    *,
    ue_axis_role: str,
) -> int | None:
    if dataset.ndim == 0:
        return None
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
        return axis if axis_role == ue_axis_role else None

    tx_count = int(h5["topology/tx_positions_m"].shape[0]) if "topology" in h5 else 0
    rx_count = int(h5["topology/rx_positions_m"].shape[0]) if "topology" in h5 else 0
    if (
        dataset_path.startswith("labels/link/")
        and dataset.ndim >= 1
        and tx_count > 0
        and rx_count > 0
        and dataset.shape[0] == tx_count * rx_count
    ):
        return 0

    order = _attr_to_string(dataset.attrs.get("index_order", ""))
    if not order:
        return None
    wanted = {"tx", "ul_tx"} if ue_axis_role == "tx" else {"rx", "ul_rx"}
    expected = tx_count if ue_axis_role == "tx" else rx_count
    for axis, token in enumerate(part.strip() for part in order.split(",")):
        if axis >= dataset.ndim:
            continue
        if token in wanted and int(dataset.shape[axis]) == expected:
            return axis
    return None


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
