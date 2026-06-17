"""Appendable HDF5 bundle writer for experimental sharded outputs."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from sionna_measurement_sim.domain.constants import BUNDLE_CONTRACT_NAME
from sionna_measurement_sim.domain.results import (
    IQLinkLibraryResult,
    MeasurementSimulationResult,
    RTLabelsOnlyResult,
    ShardSpec,
)
from sionna_measurement_sim.io.hdf5_writer import (
    _ACTIVE_COMPRESSION,
    _ACTIVE_GZIP_LEVEL,
    _ACTIVE_TRACER,
    UTF8_DTYPE,
    _resolve_dataset_compression,
    _validate_compression_args,
    write_iq_link_library_result_to_h5,
    write_measurement_result_to_h5,
    write_rt_labels_result_to_h5,
)

BundleResult = MeasurementSimulationResult | RTLabelsOnlyResult | IQLinkLibraryResult


@dataclass(frozen=True)
class BundleAppendSummary:
    """Summary for one appended shard fragment."""

    fragment_id: str
    append_start: int
    append_count: int
    global_ue_indices: tuple[int, ...]
    shard_index: int
    parent_shard_index: int
    fallback_level: int
    fallback_reason: str


class HDF5ResultBundleWriter:
    """Write multiple shard results into one appendable HDF5 bundle.

    The writer records each domain result through the existing HDF5 writer into
    a lightweight in-memory fragment, then appends datasets that carry the resolved UE axis.
    This keeps PHY-standard-specific field handling in the existing writer while
    the bundle layer stays focused on HDF5 layout.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        compression: str = "gzip",
        gzip_level: int = 4,
        tracer: Any | None = None,
    ) -> None:
        _validate_compression_args(compression, gzip_level)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self.gzip_level = int(gzip_level)
        self.tracer = tracer
        self._h5 = h5py.File(self.path, "w")
        self._fragment_count = 0
        self._ue_count = 0
        self._ue_axis_role: str | None = None
        self._source_contract_name = ""
        self._shared_dataset_cache: dict[str, np.ndarray] = {}
        self._closed = False

    def __enter__(self) -> HDF5ResultBundleWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def append_result(
        self,
        result: BundleResult,
        *,
        shard_spec: ShardSpec | None = None,
        fragment_id: str | None = None,
    ) -> BundleAppendSummary:
        """Append one computed shard result to the bundle."""

        if self._closed:
            raise ValueError("Cannot append to a closed HDF5ResultBundleWriter")
        fragment_id = fragment_id or _default_fragment_id(self._fragment_count, shard_spec)
        with _result_as_recorded_h5(result) as source:
            if self._fragment_count == 0:
                self._initialize_from_first_fragment(source)
            context = _bundle_context(source, shard_spec, fragment_id)
            if self._ue_axis_role is not None and context.ue_axis_role != self._ue_axis_role:
                msg = (
                    "All bundle fragments must use the same resolved UE axis role; "
                    f"got {context.ue_axis_role!r}, expected {self._ue_axis_role!r}"
                )
                raise ValueError(msg)
            append_start = self._ue_count
            self._append_datasets(source, context)
            self._append_bundle_metadata(context, append_start)
            self._fragment_count += 1
            self._ue_count += len(context.global_ue_indices)
            return BundleAppendSummary(
                fragment_id=fragment_id,
                append_start=append_start,
                append_count=len(context.global_ue_indices),
                global_ue_indices=tuple(int(v) for v in context.global_ue_indices),
                shard_index=context.shard_index,
                parent_shard_index=context.parent_shard_index,
                fallback_level=context.fallback_level,
                fallback_reason=context.fallback_reason,
            )

    def close(self) -> None:
        if self._closed:
            return
        bundle = self._h5.require_group("bundle")
        _upsert_scalar(bundle, "fragment_count", np.int32(self._fragment_count))
        _upsert_scalar(bundle, "ue_count", np.int64(self._ue_count))
        self._h5.close()
        self._closed = True

    def _initialize_from_first_fragment(self, source: _RecordedH5File) -> None:
        contract_name = _read_string(source["meta/contract_name"])
        self._source_contract_name = contract_name
        context = _bundle_context(source, None, "probe")
        self._ue_axis_role = context.ue_axis_role
        meta = self._h5.require_group("meta")
        for name, dataset in source["meta"].items():
            if name == "contract_name":
                _copy_scalar(meta, name, BUNDLE_CONTRACT_NAME)
            else:
                _copy_dataset(meta, name, dataset)
        bundle = self._h5.require_group("bundle")
        _copy_scalar(bundle, "source_contract_name", contract_name)
        _copy_scalar(bundle, "ue_axis_role", self._ue_axis_role)
        _copy_scalar(bundle, "append_axis_policy", "resolved_ue_axis")
        _copy_scalar(bundle, "layout_version", "experimental_append_v1")

    def _append_datasets(self, source: _RecordedH5File, context: _FragmentContext) -> None:
        for dataset in _iter_datasets(source):
            path = _strip_root(dataset.name)
            if _skip_source_dataset(path):
                continue
            append_axis = _append_axis_for_dataset(path, dataset, context)
            if append_axis is None:
                self._copy_shared_or_fragment(dataset, context)
                continue
            self._append_dataset(path, dataset, append_axis)

    def _copy_shared_or_fragment(
        self,
        dataset: h5py.Dataset | _RecordedDataset,
        context: _FragmentContext,
    ) -> None:
        path = _strip_root(dataset.name)
        if path in self._h5:
            cached = self._shared_dataset_cache.get(path)
            if cached is None:
                cached = np.asarray(self._h5[path][()])
                self._shared_dataset_cache[path] = cached
            if _array_values_equal(cached, dataset):
                return
            fragment_group = self._h5.require_group(f"bundle/fragments/{context.fragment_id}")
            _copy_dataset_path(fragment_group, path, dataset)
            return
        _copy_dataset_path(self._h5, path, dataset)
        self._shared_dataset_cache[path] = np.asarray(dataset[()])

    def _append_dataset(
        self,
        path: str,
        source: h5py.Dataset | _RecordedDataset,
        axis: int,
    ) -> None:
        array = source[()]
        if axis >= np.ndim(array):
            msg = f"Append axis {axis} is out of bounds for /{path}"
            raise ValueError(msg)
        parent = _require_parent(self._h5, path)
        name = Path(path).name
        if path not in self._h5:
            kwargs = self._dataset_create_kwargs(f"/{path}", array, axis, source)
            maxshape = list(array.shape)
            maxshape[axis] = None
            start = time.perf_counter()
            dataset = parent.create_dataset(
                name,
                data=array,
                maxshape=tuple(maxshape),
                **kwargs,
            )
            duration_s = time.perf_counter() - start
            _copy_attrs(source, dataset)
            self._record_write(dataset, array, duration_s)
            return

        dataset = self._h5[path]
        if dataset.ndim != np.ndim(array):
            msg = f"/{path} rank changed across bundle fragments"
            raise ValueError(msg)
        for dim, (existing, incoming) in enumerate(zip(dataset.shape, array.shape, strict=True)):
            if dim == axis:
                continue
            if existing != incoming:
                msg = f"/{path} non-append dimension {dim} changed: {existing} != {incoming}"
                raise ValueError(msg)
        old_size = dataset.shape[axis]
        new_size = old_size + array.shape[axis]
        selection = [slice(None)] * dataset.ndim
        selection[axis] = slice(old_size, new_size)
        start = time.perf_counter()
        dataset.resize(new_size, axis=axis)
        dataset[tuple(selection)] = array
        duration_s = time.perf_counter() - start
        self._record_write(dataset, array, duration_s)

    def _dataset_create_kwargs(
        self,
        path: str,
        array: np.ndarray,
        axis: int,
        source: h5py.Dataset | _RecordedDataset,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"chunks": _chunk_shape(array.shape, axis)}
        if h5py.check_string_dtype(source.dtype) is not None:
            kwargs["dtype"] = source.dtype
            return kwargs
        compression = _resolve_dataset_compression(path, np.asarray(array), self.compression)
        if compression is not None and np.ndim(array) > 0 and np.size(array) > 0:
            kwargs["compression"] = compression
            if compression == "gzip":
                kwargs["compression_opts"] = self.gzip_level
            kwargs["shuffle"] = True
        return kwargs

    def _append_bundle_metadata(
        self,
        context: _FragmentContext,
        append_start: int,
    ) -> None:
        group = self._h5.require_group("bundle")
        _append_1d(group, "fragment_id", np.asarray([context.fragment_id], dtype=object))
        _append_1d(group, "shard_index", np.asarray([context.shard_index], dtype=np.int32))
        _append_1d(
            group,
            "parent_shard_index",
            np.asarray([context.parent_shard_index], dtype=np.int32),
        )
        _append_1d(
            group,
            "fallback_level",
            np.asarray([context.fallback_level], dtype=np.int32),
        )
        _append_1d(
            group,
            "fallback_reason",
            np.asarray([context.fallback_reason], dtype=object),
        )
        _append_2d(
            group,
            "shard_offsets",
            np.asarray([[append_start, len(context.global_ue_indices)]], dtype=np.int64),
        )
        _append_1d(
            group,
            "global_ue_indices",
            np.asarray(context.global_ue_indices, dtype=np.int64),
        )
        _append_1d(
            group,
            "global_tx_indices",
            np.asarray(context.global_tx_indices, dtype=np.int64),
        )
        _append_1d(
            group,
            "global_rx_indices",
            np.asarray(context.global_rx_indices, dtype=np.int64),
        )

    def _record_write(
        self,
        dataset: h5py.Dataset,
        array: np.ndarray,
        duration_s: float,
    ) -> None:
        if self.tracer is None:
            return
        try:
            storage_bytes = int(dataset.id.get_storage_size())
        except Exception:
            storage_bytes = -1
        self.tracer.record_event(
            "hdf5.dataset_write",
            path=str(dataset.name),
            shape=tuple(int(dim) for dim in np.shape(array)),
            dtype=str(np.asarray(array).dtype),
            raw_bytes=int(np.asarray(array).nbytes),
            storage_bytes=storage_bytes,
            compression=str(dataset.compression or "none"),
            compression_opts=dataset.compression_opts,
            duration_s=float(duration_s),
            bundle=True,
        )


@dataclass(frozen=True)
class _FragmentContext:
    fragment_id: str
    ue_axis_role: str
    global_ue_indices: np.ndarray
    global_tx_indices: np.ndarray
    global_rx_indices: np.ndarray
    tx_count: int
    rx_count: int
    shard_index: int
    parent_shard_index: int
    fallback_level: int
    fallback_reason: str


class _RecordedDataset:
    """Small dataset facade used to avoid serializing bundle fragments to HDF5."""

    def __init__(self, name: str, data: Any, dtype: Any | None = None) -> None:
        self.name = name
        self.attrs: dict[str, Any] = {}
        self._data = np.asarray(data, dtype=dtype) if dtype is not None else np.asarray(data)
        self.dtype = self._data.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(dim) for dim in self._data.shape)

    @property
    def ndim(self) -> int:
        return int(self._data.ndim)

    def __getitem__(self, key: Any) -> Any:
        return self._data[key]


class _RecordedGroup:
    """Minimal h5py-like group facade for existing writer helper reuse."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.attrs: dict[str, Any] = {}
        self._children: dict[str, _RecordedGroup | _RecordedDataset] = {}

    def require_group(self, name: str) -> _RecordedGroup:
        group = self
        for part in _path_parts(name):
            child = group._children.get(part)
            if child is None:
                child = _RecordedGroup(_join_h5_path(group.name, part))
                group._children[part] = child
            if not isinstance(child, _RecordedGroup):
                msg = f"/{name} already exists as a dataset"
                raise TypeError(msg)
            group = child
        return group

    def create_dataset(
        self,
        name: str,
        data: Any,
        dtype: Any | None = None,
        **_: Any,
    ) -> _RecordedDataset:
        parts = _path_parts(name)
        if not parts:
            msg = "Dataset name must not be empty"
            raise ValueError(msg)
        group = self.require_group("/".join(parts[:-1])) if len(parts) > 1 else self
        dataset_name = parts[-1]
        if dataset_name in group._children:
            msg = f"/{name} already exists"
            raise ValueError(msg)
        dataset = _RecordedDataset(_join_h5_path(group.name, dataset_name), data, dtype=dtype)
        group._children[dataset_name] = dataset
        return dataset

    def __contains__(self, path: str) -> bool:
        try:
            self[path]
        except KeyError:
            return False
        return True

    def __getitem__(self, path: str) -> _RecordedGroup | _RecordedDataset:
        item: _RecordedGroup | _RecordedDataset = self
        for part in _path_parts(path):
            if not isinstance(item, _RecordedGroup):
                raise KeyError(path)
            try:
                item = item._children[part]
            except KeyError as exc:
                raise KeyError(path) from exc
        return item

    def items(self):
        return self._children.items()

    def values(self):
        return self._children.values()


class _RecordedH5File(_RecordedGroup):
    """Root in-memory fragment facade."""

    def __init__(self) -> None:
        super().__init__("/")

    def __enter__(self) -> _RecordedH5File:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def close(self) -> None:
        return


def _result_as_recorded_h5(result: BundleResult) -> _RecordedH5File:
    compression_token = _ACTIVE_COMPRESSION.set("none")
    gzip_token = _ACTIVE_GZIP_LEVEL.set(1)
    tracer_token = _ACTIVE_TRACER.set(None)
    h5 = _RecordedH5File()
    try:
        if isinstance(result, MeasurementSimulationResult):
            write_measurement_result_to_h5(h5, result)
        elif isinstance(result, RTLabelsOnlyResult):
            write_rt_labels_result_to_h5(h5, result)
        elif isinstance(result, IQLinkLibraryResult):
            write_iq_link_library_result_to_h5(h5, result)
        else:  # pragma: no cover - defensive for future result types
            msg = f"Unsupported bundle result type: {type(result)!r}"
            raise TypeError(msg)
        return h5
    finally:
        _ACTIVE_TRACER.reset(tracer_token)
        _ACTIVE_GZIP_LEVEL.reset(gzip_token)
        _ACTIVE_COMPRESSION.reset(compression_token)


def _bundle_context(
    source: _RecordedH5File,
    shard_spec: ShardSpec | None,
    fragment_id: str,
) -> _FragmentContext:
    tx_role = _read_string(source["link/tx_role"]) if "link/tx_role" in source else "tx"
    rx_role = _read_string(source["link/rx_role"]) if "link/rx_role" in source else "rx"
    tx_count = int(source["topology/tx_positions_m"].shape[0])
    rx_count = int(source["topology/rx_positions_m"].shape[0])
    tx_indices = (
        np.asarray(source["shard/global_tx_indices"][()], dtype=np.int64)
        if "shard/global_tx_indices" in source
        else np.arange(tx_count, dtype=np.int64)
    )
    rx_indices = (
        np.asarray(source["shard/global_rx_indices"][()], dtype=np.int64)
        if "shard/global_rx_indices" in source
        else np.arange(rx_count, dtype=np.int64)
    )
    if tx_role == "ue":
        ue_axis_role = "tx"
        global_ue_indices = tx_indices
    elif rx_role == "ue":
        ue_axis_role = "rx"
        global_ue_indices = rx_indices
    else:
        ue_axis_role = "tx"
        global_ue_indices = tx_indices
    shard_index = (
        int(source["shard/shard_index"][()])
        if "shard/shard_index" in source
        else (int(shard_spec.shard_index) if shard_spec else 0)
    )
    parent = (
        shard_spec.parent_shard_index
        if shard_spec is not None and shard_spec.parent_shard_index is not None
        else shard_index
    )
    return _FragmentContext(
        fragment_id=fragment_id,
        ue_axis_role=ue_axis_role,
        global_ue_indices=np.asarray(global_ue_indices, dtype=np.int64),
        global_tx_indices=tx_indices,
        global_rx_indices=rx_indices,
        tx_count=tx_count,
        rx_count=rx_count,
        shard_index=shard_index,
        parent_shard_index=int(parent),
        fallback_level=int(shard_spec.fallback_level if shard_spec else 0),
        fallback_reason=str(shard_spec.fallback_reason if shard_spec else ""),
    )


def _append_axis_for_dataset(
    path: str,
    dataset: h5py.Dataset | _RecordedDataset,
    context: _FragmentContext,
) -> int | None:
    if dataset.ndim == 0:
        return None
    role = context.ue_axis_role
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
    if path in path_axes:
        axis_role, axis = path_axes[path]
        if axis_role == role and _can_append_shape(dataset.shape, axis):
            return axis
        return None
    if path.startswith("labels/link/") and dataset.shape[0] == context.tx_count * context.rx_count:
        return 0 if _can_append_shape(dataset.shape, 0) else None

    order = _attr_string(dataset.attrs.get("index_order", ""))
    if not order:
        return None
    tokens = [token.strip() for token in order.split(",")]
    wanted = {"tx", "ul_tx"} if role == "tx" else {"rx", "ul_rx"}
    expected = context.tx_count if role == "tx" else context.rx_count
    for axis, token in enumerate(tokens):
        if axis >= dataset.ndim:
            continue
        if token in wanted and dataset.shape[axis] == expected:
            return axis if _can_append_shape(dataset.shape, axis) else None
    return None


def _can_append_shape(shape: tuple[int, ...], axis: int) -> bool:
    return shape[axis] > 0 and all(dim > 0 for idx, dim in enumerate(shape) if idx != axis)


def _skip_source_dataset(path: str) -> bool:
    return path.startswith("meta/") or path.startswith("shard/")


def _iter_datasets(group: h5py.Group | _RecordedGroup):
    for item in group.values():
        if isinstance(item, (h5py.Dataset, _RecordedDataset)):
            yield item
        elif isinstance(item, (h5py.Group, _RecordedGroup)):
            yield from _iter_datasets(item)


def _copy_dataset_path(
    parent: h5py.Group,
    path: str,
    source: h5py.Dataset | _RecordedDataset,
) -> h5py.Dataset:
    group = _require_parent(parent, path)
    return _copy_dataset(group, Path(path).name, source)


def _copy_dataset(
    group: h5py.Group,
    name: str,
    source: h5py.Dataset | _RecordedDataset,
) -> h5py.Dataset:
    data = source[()]
    kwargs: dict[str, Any] = {}
    if h5py.check_string_dtype(source.dtype) is not None:
        kwargs["dtype"] = source.dtype
    dataset = group.create_dataset(name, data=data, **kwargs)
    _copy_attrs(source, dataset)
    return dataset


def _copy_scalar(group: h5py.Group, name: str, value: Any) -> h5py.Dataset:
    if isinstance(value, str):
        return group.create_dataset(name, data=value, dtype=UTF8_DTYPE)
    return group.create_dataset(name, data=value)


def _copy_attrs(source: h5py.Dataset | _RecordedDataset, target: h5py.Dataset) -> None:
    for key, value in source.attrs.items():
        target.attrs[key] = value


def _require_parent(root: h5py.Group, path: str) -> h5py.Group:
    parent = Path(path).parent.as_posix()
    if parent in ("", "."):
        return root
    return root.require_group(parent)


def _append_1d(group: h5py.Group, name: str, values: np.ndarray) -> None:
    values = np.asarray(values)
    if name not in group:
        dtype = UTF8_DTYPE if values.dtype == object else values.dtype
        group.create_dataset(name, data=values, maxshape=(None,), dtype=dtype, chunks=True)
        return
    dataset = group[name]
    old = dataset.shape[0]
    dataset.resize(old + values.shape[0], axis=0)
    dataset[old:] = values


def _append_2d(group: h5py.Group, name: str, values: np.ndarray) -> None:
    values = np.asarray(values)
    if name not in group:
        group.create_dataset(
            name,
            data=values,
            maxshape=(None, values.shape[1]),
            chunks=True,
        )
        return
    dataset = group[name]
    old = dataset.shape[0]
    dataset.resize(old + values.shape[0], axis=0)
    dataset[old:] = values


def _upsert_scalar(group: h5py.Group, name: str, value: Any) -> None:
    if name in group:
        group[name][()] = value
        return
    _copy_scalar(group, name, value)


def _array_values_equal(left: np.ndarray, right: h5py.Dataset | _RecordedDataset) -> bool:
    if tuple(int(dim) for dim in left.shape) != right.shape:
        return False
    if (
        h5py.check_string_dtype(left.dtype) is not None
        or h5py.check_string_dtype(right.dtype) is not None
    ):
        return np.array_equal(left, right[()])
    return np.array_equal(left, right[()], equal_nan=True)


def _chunk_shape(shape: tuple[int, ...], append_axis: int) -> tuple[int, ...]:
    if not shape:
        return ()
    chunks = [max(1, min(int(dim), 64)) for dim in shape]
    chunks[append_axis] = max(1, min(int(shape[append_axis]), 16))
    return tuple(chunks)


def _read_string(dataset: h5py.Dataset | _RecordedDataset) -> str:
    value = dataset[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _attr_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _path_parts(path: str) -> list[str]:
    return [part for part in path.strip("/").split("/") if part]


def _join_h5_path(parent: str, child: str) -> str:
    if parent == "/":
        return f"/{child}"
    return f"{parent}/{child}"


def _strip_root(path: str) -> str:
    return path[1:] if path.startswith("/") else path


def _default_fragment_id(index: int, shard_spec: ShardSpec | None) -> str:
    if shard_spec is None:
        return f"fragment_{index:06d}"
    return shard_spec.shard_id or f"{shard_spec.shard_index:03d}"
