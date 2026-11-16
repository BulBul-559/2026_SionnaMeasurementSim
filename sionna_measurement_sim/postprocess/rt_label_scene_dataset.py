"""Build compact per-scene RT labels datasets from raw shard HDF5 runs.

This module is the labels-only sibling of ``cfr_scene_dataset``. It reads
``sionna_measurement_rt_labels`` shard outputs and writes a separate
scene-level dataset without modifying the source run directories.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from sionna_measurement_sim import __version__
from sionna_measurement_sim.postprocess.cfr_scene_dataset import (
    SOURCE_SHARD_ID_DTYPE,
    SourceSceneRun,
    _collect_scene_assets,
    _compression_kwargs,
    _copy_dataset_if_present,
    _copy_scene_assets,
    _create_dataset,
    _decode_h5_string,
    _extract_room_layout,
    _optional_int,
    _optional_path,
    _read_optional_scalar,
    _read_string,
    _relative_or_absolute,
    _resolve_manifest_artifact,
    _resolve_optional_existing,
    _room_indices_for_ids,
    _room_values_for_indices,
    _strings_for_room_indices,
    _write_assets,
    _write_rooms,
    _write_scalar,
    _write_string,
    discover_source_scene_runs,
)

RT_LABEL_SCENE_CONTRACT_NAME = "sionna_rt_label_scene_dataset"
RT_LABEL_SCENE_SCHEMA_VERSION = "0.1.0"

_REQUIRED_LINK_LABEL_DATASETS = (
    "link_index",
    "tx_index",
    "rx_index",
    "global_tx_index",
    "global_rx_index",
    "tx_xy_m",
    "rx_xy_m",
    "link_valid_mask",
    "geometric_distance_m",
    "first_path_delay_s",
    "first_path_propagation_range_m",
    "strongest_path_delay_s",
    "path_power_db",
    "los_flag",
    "nlos_flag",
    "path_count",
    "first_path_aoa_azimuth_rad",
    "first_path_aoa_zenith_rad",
    "tx_rx_bearing_rad",
    "tx_rx_distance_m",
)

_REQUIRED_SCENE_DATASETS = (
    "scene_id",
    "scene_metadata",
    "link_id",
    "tx_id",
    "rx_id",
    "ue_id",
    "bs_id",
    "target_xy",
    "valid_mask",
    "source_shard_id",
    "source_shard_index",
    "source_local_index",
    "source_local_tx_index",
    "source_local_rx_index",
    "labels/link/link_index",
    "labels/link/tx_index",
    "labels/link/rx_index",
    "labels/link/global_tx_index",
    "labels/link/global_rx_index",
    "labels/link/link_valid_mask",
    "labels/link/geometric_distance_m",
    "labels/link/first_path_delay_s",
    "labels/link/first_path_propagation_range_m",
    "labels/link/strongest_path_delay_s",
    "labels/link/path_power_db",
    "labels/link/los_flag",
    "labels/link/nlos_flag",
    "labels/link/path_count",
    "labels/link/tx_xy_m",
    "labels/link/rx_xy_m",
    "rooms/room_id",
    "rooms/room_type",
    "rooms/room_type_id",
    "rooms/floor_area_m2",
    "rooms/bounds_xy",
    "rooms/ue_count",
    "rooms/bs_count",
    "link/ue_room_index",
    "link/bs_room_index",
    "link/target_room_index",
    "link/target_room_area_m2",
    "link/target_room_type_id",
)


@dataclass(frozen=True)
class RTLabelSceneBuildConfig:
    """Configuration for offline RT labels scene dataset construction."""

    input_root: Path | None = None
    output_root: Path = Path("outputs/rt_label_scene_dataset")
    run_dirs: tuple[Path, ...] = ()
    start_index: int = 0
    limit: int | None = None
    scene_ids: tuple[str, ...] = ()
    workers: int = 1
    overwrite: bool = False
    skip_existing: bool = True
    validate: bool = True
    copy_assets: bool = True
    embed_assets: bool = True
    label_name: str = "label_panel_0p5.json"
    label_compression: str = "gzip"
    gzip_level: int = 1
    metadata_compression: str = "gzip"
    link_index_format: str = "pt"
    command: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RTLabelSceneResult:
    """Post-processing result for a single RT labels scene."""

    scene_id: str
    status: str
    h5_path: Path | None = None
    num_links: int = 0
    valid_link_count: int = 0
    label_dataset_count: int = 0
    source_shards: tuple[str, ...] = ()
    source_run_dir: Path | None = None
    label_file: Path | None = None
    scene_file: Path | None = None
    floorplan_found: bool = False
    elapsed_seconds: float = 0.0
    output_bytes: int = 0
    source_bytes: int = 0
    path_power_min_db: float | None = None
    path_power_max_db: float | None = None
    path_power_quantiles_db: dict[str, float] = field(default_factory=dict)
    error: str = ""


@dataclass(frozen=True)
class RTLabelSceneDatasetResult:
    """Result for a RT labels scene dataset build run."""

    output_root: Path
    manifest_path: Path
    link_index_path: Path | None
    link_index_parquet_path: Path | None
    room_index_path: Path | None
    room_index_parquet_path: Path | None
    scene_results: tuple[RTLabelSceneResult, ...]
    elapsed_seconds: float

    @property
    def completed(self) -> int:
        return sum(1 for result in self.scene_results if result.status == "completed")

    @property
    def failed(self) -> int:
        return sum(1 for result in self.scene_results if result.status == "failed")

    @property
    def skipped(self) -> int:
        return sum(1 for result in self.scene_results if result.status == "skipped")


def build_rt_label_scene_dataset(
    config: RTLabelSceneBuildConfig,
) -> RTLabelSceneDatasetResult:
    """Build a scene-level RT labels dataset from raw sharded run outputs."""

    start = time.perf_counter()
    _validate_build_config(config)
    output_root = Path(config.output_root)
    (output_root / "scenes").mkdir(parents=True, exist_ok=True)
    if config.copy_assets:
        (output_root / "assets").mkdir(parents=True, exist_ok=True)

    runs = discover_source_scene_runs(
        input_root=config.input_root,
        run_dirs=config.run_dirs,
        scene_ids=config.scene_ids,
        start_index=config.start_index,
        limit=config.limit,
    )

    worker_payloads = [
        _WorkerConfig(
            run=run,
            output_root=output_root,
            overwrite=config.overwrite,
            skip_existing=config.skip_existing,
            validate=config.validate,
            copy_assets=config.copy_assets,
            embed_assets=config.embed_assets,
            label_name=config.label_name,
            label_compression=config.label_compression,
            gzip_level=config.gzip_level,
            metadata_compression=config.metadata_compression,
        )
        for run in runs
    ]

    if config.workers == 1 or len(worker_payloads) <= 1:
        scene_results = tuple(_process_scene_worker(payload) for payload in worker_payloads)
    else:
        results: list[RTLabelSceneResult | None] = [None] * len(worker_payloads)
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            futures = {
                executor.submit(_process_scene_worker, payload): index
                for index, payload in enumerate(worker_payloads)
            }
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        scene_results = tuple(result for result in results if result is not None)

    index_paths = _write_global_indexes(output_root, scene_results, config.link_index_format)
    manifest_path = _write_dataset_manifest(
        output_root=output_root,
        config=config,
        scene_results=scene_results,
        index_paths=index_paths,
        elapsed_seconds=time.perf_counter() - start,
    )
    return RTLabelSceneDatasetResult(
        output_root=output_root,
        manifest_path=manifest_path,
        link_index_path=index_paths.get("link_index_pt"),
        link_index_parquet_path=index_paths.get("link_index_parquet"),
        room_index_path=index_paths.get("room_index_pt"),
        room_index_parquet_path=index_paths.get("room_index_parquet"),
        scene_results=scene_results,
        elapsed_seconds=time.perf_counter() - start,
    )


def validate_rt_label_scene_h5(path: str | Path) -> None:
    """Validate the compact RT label scene HDF5 contract."""

    with h5py.File(path, "r") as h5:
        contract = _read_string(h5["meta/contract_name"])
        if contract != RT_LABEL_SCENE_CONTRACT_NAME:
            msg = f"Unsupported RT label scene contract: {contract!r}"
            raise ValueError(msg)
        for dataset_path in _REQUIRED_SCENE_DATASETS:
            if dataset_path not in h5:
                msg = f"Missing required RT label scene dataset /{dataset_path}"
                raise ValueError(msg)
        link_count = int(h5["link_id"].shape[0])
        for name in (
            "tx_id",
            "rx_id",
            "ue_id",
            "bs_id",
            "valid_mask",
            "source_shard_id",
            "source_shard_index",
            "source_local_index",
            "source_local_tx_index",
            "source_local_rx_index",
        ):
            if h5[name].shape != (link_count,):
                msg = f"/{name} must match link count {link_count}, got {h5[name].shape}"
                raise ValueError(msg)
        if h5["target_xy"].shape != (link_count, 2):
            msg = f"/target_xy must be [link,2], got {h5['target_xy'].shape}"
            raise ValueError(msg)
        for name in _REQUIRED_LINK_LABEL_DATASETS:
            shape = h5[f"labels/link/{name}"].shape
            expected_tail = (2,) if name in ("tx_xy_m", "rx_xy_m") else ()
            if shape != (link_count, *expected_tail):
                msg = f"/labels/link/{name} has invalid shape {shape}"
                raise ValueError(msg)
        if not np.array_equal(h5["valid_mask"][()], h5["labels/link/link_valid_mask"][()]):
            msg = "/valid_mask must mirror /labels/link/link_valid_mask"
            raise ValueError(msg)
        room_count = int(h5["rooms/room_id"].shape[0])
        for name in ("room_type", "room_type_id", "floor_area_m2", "ue_count", "bs_count"):
            if h5[f"rooms/{name}"].shape != (room_count,):
                msg = f"/rooms/{name} must match room count {room_count}"
                raise ValueError(msg)
        if h5["rooms/bounds_xy"].shape != (room_count, 4):
            msg = f"/rooms/bounds_xy must be [room,4], got {h5['rooms/bounds_xy'].shape}"
            raise ValueError(msg)
        for name in (
            "ue_room_index",
            "bs_room_index",
            "target_room_index",
            "target_room_area_m2",
            "target_room_type_id",
        ):
            if h5[f"link/{name}"].shape != (link_count,):
                msg = f"/link/{name} must match link count {link_count}"
                raise ValueError(msg)


@dataclass(frozen=True)
class _WorkerConfig:
    run: SourceSceneRun
    output_root: Path
    overwrite: bool
    skip_existing: bool
    validate: bool
    copy_assets: bool
    embed_assets: bool
    label_name: str
    label_compression: str
    gzip_level: int
    metadata_compression: str


@dataclass(frozen=True)
class _DatasetSpec:
    name: str
    dtype: np.dtype
    tail_shape: tuple[int, ...]
    attrs: dict[str, Any]


@dataclass(frozen=True)
class _SourceLayout:
    link_count: int
    tx_count: int
    rx_count: int
    tx_role: str
    rx_role: str
    label_specs: tuple[_DatasetSpec, ...]


def _process_scene_worker(config: _WorkerConfig) -> RTLabelSceneResult:
    start = time.perf_counter()
    try:
        return _process_scene(config, start)
    except Exception as exc:  # noqa: BLE001 - worker must report scene failure.
        return RTLabelSceneResult(
            scene_id=config.run.scene_id,
            status="failed",
            source_run_dir=config.run.run_dir,
            label_file=config.run.label_file,
            scene_file=config.run.scene_file,
            elapsed_seconds=time.perf_counter() - start,
            error=f"{type(exc).__name__}: {exc}",
        )


def _process_scene(config: _WorkerConfig, start: float) -> RTLabelSceneResult:
    run = config.run
    scene_id = run.scene_id
    output_path = config.output_root / "scenes" / f"{scene_id}.h5"
    if output_path.exists():
        if config.overwrite:
            output_path.unlink()
        elif config.skip_existing:
            return _result_from_existing_scene(run, output_path, start)
        else:
            msg = f"Output scene H5 already exists: {output_path}"
            raise FileExistsError(msg)

    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    source_entries = list(manifest.get("results", []))
    if not source_entries:
        msg = f"Source run has no shard results: {run.manifest_path}"
        raise ValueError(msg)
    source_paths = tuple(
        _resolve_manifest_artifact(run.run_dir, Path(str(entry["result_h5"])))
        for entry in source_entries
    )
    source_bytes = sum(path.stat().st_size for path in source_paths if path.exists())
    label_file = _resolve_optional_existing(
        run.label_file or _optional_path(manifest.get("label_file"))
    )
    scene_file = _resolve_optional_existing(
        run.scene_file or _optional_path(manifest.get("scene_file"))
    )

    source_layout = _inspect_source_layout(source_entries, source_paths)
    assets = _collect_scene_assets(
        label_file=label_file,
        scene_file=scene_file,
        label_name=config.label_name,
    )
    room_layout = _extract_room_layout(assets.label_file)

    tmp_path = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _write_scene_h5(
            tmp_path,
            run=run,
            manifest=manifest,
            source_entries=source_entries,
            source_paths=source_paths,
            source_layout=source_layout,
            assets=assets,
            room_layout=room_layout,
            config=config,
        )
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    if config.copy_assets:
        _copy_scene_assets(config.output_root, scene_id, assets)
    if config.validate:
        validate_rt_label_scene_h5(output_path)

    with h5py.File(output_path, "r") as h5:
        power = np.asarray(h5["labels/link/path_power_db"][()])
        valid = np.asarray(h5["valid_mask"][()], dtype=np.bool_)
        return RTLabelSceneResult(
            scene_id=scene_id,
            status="completed",
            h5_path=output_path,
            num_links=int(h5["link_id"].shape[0]),
            valid_link_count=int(np.count_nonzero(valid)),
            label_dataset_count=len(h5["labels/link"]),
            source_shards=tuple(str(item.get("shard_id", "")) for item in source_entries),
            source_run_dir=run.run_dir,
            label_file=label_file,
            scene_file=scene_file,
            floorplan_found=assets.floorplan_png is not None,
            elapsed_seconds=time.perf_counter() - start,
            output_bytes=output_path.stat().st_size,
            source_bytes=source_bytes,
            path_power_min_db=float(np.min(power)) if power.size else None,
            path_power_max_db=float(np.max(power)) if power.size else None,
            path_power_quantiles_db=_quantiles(power),
        )


def _inspect_source_layout(
    entries: list[dict[str, Any]],
    source_paths: tuple[Path, ...],
) -> _SourceLayout:
    link_count = 0
    tx_total = 0
    rx_count = -1
    tx_role = rx_role = ""
    label_specs: tuple[_DatasetSpec, ...] | None = None
    expected_names: tuple[str, ...] | None = None
    for _entry, source_path in zip(entries, source_paths, strict=True):
        with h5py.File(source_path, "r") as h5:
            if "labels/link" not in h5:
                msg = f"{source_path}: missing /labels/link"
                raise ValueError(msg)
            group = h5["labels/link"]
            for name in _REQUIRED_LINK_LABEL_DATASETS:
                if name not in group:
                    msg = f"{source_path}: missing /labels/link/{name}"
                    raise ValueError(msg)
            shard_link_count = int(group["link_index"].shape[0])
            for name, dataset in group.items():
                if dataset.shape[0] != shard_link_count:
                    msg = f"{source_path}: /labels/link/{name} first dim must be link"
                    raise ValueError(msg)
            names = tuple(sorted(group.keys()))
            specs = tuple(
                _DatasetSpec(
                    name=name,
                    dtype=np.dtype(group[name].dtype),
                    tail_shape=tuple(int(dim) for dim in group[name].shape[1:]),
                    attrs=_decode_attrs(dict(group[name].attrs)),
                )
                for name in names
            )
            if expected_names is None:
                expected_names = names
                label_specs = specs
                tx_role = _read_string(h5["link/tx_role"]) if "link/tx_role" in h5 else "tx"
                rx_role = _read_string(h5["link/rx_role"]) if "link/rx_role" in h5 else "rx"
                rx_count = (
                    int(h5["topology/rx_positions_m"].shape[0])
                    if "topology/rx_positions_m" in h5
                    else 0
                )
            elif names != expected_names:
                msg = f"{source_path}: /labels/link dataset set differs across shards"
                raise ValueError(msg)
            else:
                assert label_specs is not None
                _check_specs_match(source_path, label_specs, specs)
            link_count += shard_link_count
            if "topology/tx_positions_m" in h5:
                tx_total += int(h5["topology/tx_positions_m"].shape[0])
    if label_specs is None:
        msg = "No source label datasets found"
        raise ValueError(msg)
    return _SourceLayout(
        link_count=link_count,
        tx_count=tx_total,
        rx_count=rx_count,
        tx_role=tx_role,
        rx_role=rx_role,
        label_specs=label_specs,
    )


def _check_specs_match(
    source_path: Path,
    expected: tuple[_DatasetSpec, ...],
    actual: tuple[_DatasetSpec, ...],
) -> None:
    for left, right in zip(expected, actual, strict=True):
        if (
            left.name != right.name
            or left.dtype != right.dtype
            or left.tail_shape != right.tail_shape
        ):
            msg = f"{source_path}: /labels/link/{right.name} schema differs across shards"
            raise ValueError(msg)


def _write_scene_h5(
    path: Path,
    *,
    run: SourceSceneRun,
    manifest: dict[str, Any],
    source_entries: list[dict[str, Any]],
    source_paths: tuple[Path, ...],
    source_layout: _SourceLayout,
    assets: Any,
    room_layout: Any,
    config: _WorkerConfig,
) -> None:
    metadata_kwargs = _compression_kwargs(config.metadata_compression, config.gzip_level)
    label_kwargs = _compression_kwargs(config.label_compression, config.gzip_level)
    with h5py.File(path, "w") as h5:
        _write_core_metadata(h5, run, manifest, source_layout, assets, config)
        _write_rooms(h5, room_layout, metadata_kwargs)
        _write_link_label_datasets(h5, source_layout, label_kwargs)
        _create_dataset(h5, "link_id", shape=(source_layout.link_count,), dtype=np.int64)
        for name in ("tx_id", "rx_id", "ue_id", "bs_id"):
            _create_dataset(h5, name, shape=(source_layout.link_count,), dtype=np.int64)
        _create_dataset(
            h5,
            "target_xy",
            shape=(source_layout.link_count, 2),
            dtype=np.float32,
            attrs={"unit": "m", "index_order": "link,xy"},
            **metadata_kwargs,
        )
        _create_dataset(h5, "valid_mask", shape=(source_layout.link_count,), dtype=np.bool_)
        _create_dataset(
            h5,
            "source_shard_id",
            shape=(source_layout.link_count,),
            dtype=SOURCE_SHARD_ID_DTYPE,
        )
        for name in (
            "source_shard_index",
            "source_local_index",
            "source_local_tx_index",
            "source_local_rx_index",
        ):
            _create_dataset(h5, name, shape=(source_layout.link_count,), dtype=np.int32)
        for name in ("ue_room_index", "bs_room_index", "target_room_index"):
            _create_dataset(h5, f"link/{name}", shape=(source_layout.link_count,), dtype=np.int32)
        _create_dataset(
            h5,
            "link/target_room_area_m2",
            shape=(source_layout.link_count,),
            dtype=np.float32,
            attrs={"unit": "m^2"},
            **metadata_kwargs,
        )
        _create_dataset(
            h5,
            "link/target_room_type_id",
            shape=(source_layout.link_count,),
            dtype=np.int32,
        )

        tx_positions_by_id: dict[int, np.ndarray] = {}
        rx_positions_by_id: dict[int, np.ndarray] = {}
        offset = 0
        for entry, source_path in zip(source_entries, source_paths, strict=True):
            with h5py.File(source_path, "r") as src:
                offset = _append_source_shard(
                    h5,
                    src,
                    entry=entry,
                    offset=offset,
                    tx_role=source_layout.tx_role,
                    rx_role=source_layout.rx_role,
                    room_layout=room_layout,
                    tx_positions_by_id=tx_positions_by_id,
                    rx_positions_by_id=rx_positions_by_id,
                )
        if offset != source_layout.link_count:
            msg = f"scene link count mismatch: wrote {offset}, expected {source_layout.link_count}"
            raise RuntimeError(msg)
        _write_unique_topology(h5, tx_positions_by_id, rx_positions_by_id, metadata_kwargs)


def _write_link_label_datasets(
    h5: h5py.File,
    source_layout: _SourceLayout,
    label_kwargs: dict[str, Any],
) -> None:
    for spec in source_layout.label_specs:
        dataset = _create_dataset(
            h5,
            f"labels/link/{spec.name}",
            shape=(source_layout.link_count, *spec.tail_shape),
            dtype=spec.dtype,
            **label_kwargs,
        )
        for key, value in spec.attrs.items():
            dataset.attrs[key] = value


def _append_source_shard(
    out: h5py.File,
    src: h5py.File,
    *,
    entry: dict[str, Any],
    offset: int,
    tx_role: str,
    rx_role: str,
    room_layout: Any,
    tx_positions_by_id: dict[int, np.ndarray],
    rx_positions_by_id: dict[int, np.ndarray],
) -> int:
    labels = {
        name: np.asarray(dataset[()])
        for name, dataset in src["labels/link"].items()
    }
    link_count = int(labels["link_index"].shape[0])
    slc = slice(offset, offset + link_count)
    for name, values in labels.items():
        out[f"labels/link/{name}"][slc] = values

    tx_ids = labels["global_tx_index"].astype(np.int64, copy=False)
    rx_ids = labels["global_rx_index"].astype(np.int64, copy=False)
    tx_xy = labels["tx_xy_m"].astype(np.float32, copy=False)
    rx_xy = labels["rx_xy_m"].astype(np.float32, copy=False)
    valid = labels["link_valid_mask"].astype(np.bool_, copy=False)
    tx_local = labels["tx_index"].astype(np.int32, copy=False)
    rx_local = labels["rx_index"].astype(np.int32, copy=False)
    local_link = labels["link_index"].astype(np.int32, copy=False)

    if tx_role == "ue":
        ue_id, bs_id = tx_ids, rx_ids
        target_xy = tx_xy
    elif rx_role == "ue":
        ue_id, bs_id = rx_ids, tx_ids
        target_xy = rx_xy
    else:
        ue_id, bs_id = tx_ids, rx_ids
        target_xy = tx_xy
    ue_room_index = _room_indices_for_ids(ue_id, room_layout.ue_room_index)
    bs_room_index = _room_indices_for_ids(bs_id, room_layout.bs_room_index)
    target_room_index = ue_room_index
    target_room_area = _room_values_for_indices(
        target_room_index,
        room_layout.floor_area_by_index,
        fill_value=np.nan,
        dtype=np.float32,
    )
    target_room_type_id = _room_values_for_indices(
        target_room_index,
        room_layout.type_id_by_index,
        fill_value=-1,
        dtype=np.int32,
    )

    shard_id = str(entry.get("shard_id") or _read_optional_scalar(src, "shard/shard_index"))
    shard_index = _optional_int(entry.get("shard_index"))
    if shard_index is None:
        shard_index = int(_read_optional_scalar(src, "shard/shard_index", 0))
    out["link_id"][slc] = np.arange(offset, offset + link_count, dtype=np.int64)
    out["tx_id"][slc] = tx_ids
    out["rx_id"][slc] = rx_ids
    out["ue_id"][slc] = ue_id
    out["bs_id"][slc] = bs_id
    out["target_xy"][slc] = target_xy
    out["valid_mask"][slc] = valid
    out["link/ue_room_index"][slc] = ue_room_index
    out["link/bs_room_index"][slc] = bs_room_index
    out["link/target_room_index"][slc] = target_room_index
    out["link/target_room_area_m2"][slc] = target_room_area
    out["link/target_room_type_id"][slc] = target_room_type_id
    out["source_shard_id"][slc] = np.full(link_count, shard_id.encode("utf-8"), dtype="S32")
    out["source_shard_index"][slc] = np.full(link_count, shard_index, dtype=np.int32)
    out["source_local_index"][slc] = local_link
    out["source_local_tx_index"][slc] = tx_local
    out["source_local_rx_index"][slc] = rx_local

    _collect_unique_positions(src, tx_ids, tx_local, "tx", tx_positions_by_id)
    _collect_unique_positions(src, rx_ids, rx_local, "rx", rx_positions_by_id)
    return offset + link_count


def _collect_unique_positions(
    src: h5py.File,
    global_ids: np.ndarray,
    local_ids: np.ndarray,
    role: str,
    target: dict[int, np.ndarray],
) -> None:
    path = f"topology/{role}_positions_m"
    if path not in src:
        return
    positions = np.asarray(src[path][()], dtype=np.float32)
    for global_id, local_id in zip(global_ids, local_ids, strict=True):
        local = int(local_id)
        if 0 <= local < positions.shape[0]:
            target.setdefault(int(global_id), positions[local])


def _write_unique_topology(
    h5: h5py.File,
    tx_positions_by_id: dict[int, np.ndarray],
    rx_positions_by_id: dict[int, np.ndarray],
    metadata_kwargs: dict[str, Any],
) -> None:
    topology = h5.require_group("topology")
    for role, mapping in (("tx", tx_positions_by_id), ("rx", rx_positions_by_id)):
        ids = np.asarray(sorted(mapping), dtype=np.int64)
        positions = np.asarray([mapping[int(item)] for item in ids], dtype=np.float32)
        if positions.size == 0:
            positions = np.empty((0, 3), dtype=np.float32)
        topology.create_dataset(f"global_{role}_index", data=ids)
        dataset = topology.create_dataset(
            f"{role}_positions_m",
            data=positions,
            **metadata_kwargs,
        )
        dataset.attrs["unit"] = "m"
        dataset.attrs["index_order"] = f"{role},xyz"


def _write_core_metadata(
    h5: h5py.File,
    run: SourceSceneRun,
    manifest: dict[str, Any],
    source_layout: _SourceLayout,
    assets: Any,
    config: _WorkerConfig,
) -> None:
    metadata = _scene_metadata(run, manifest, source_layout, assets, config)
    _write_string(h5, "scene_id", run.scene_id)
    _write_string(h5, "scene_metadata", json.dumps(metadata, sort_keys=True))
    meta = h5.require_group("meta")
    _write_string(meta, "contract_name", RT_LABEL_SCENE_CONTRACT_NAME)
    _write_string(meta, "schema_version", RT_LABEL_SCENE_SCHEMA_VERSION)
    _write_string(meta, "producer", f"sionna_measurement_sim {__version__}")
    _write_string(meta, "created_at", datetime.now(UTC).isoformat())
    _write_string(meta, "source_run_dir", run.run_dir.as_posix())
    _write_string(meta, "source_manifest", run.manifest_path.as_posix())
    _write_string(meta, "label_compression", config.label_compression)
    _write_string(meta, "metadata_compression", config.metadata_compression)
    _write_string(meta, "generation_command", " ".join(sys.argv))
    _write_string(meta, "scene_metadata_json", json.dumps(metadata, sort_keys=True))

    scene = h5.require_group("scene")
    _write_string(scene, "scene_id", run.scene_id)
    _write_string(scene, "scene_class", run.scene_class)
    _write_scalar(scene, "source_index", -1 if run.index is None else int(run.index), np.int64)
    _write_scalar(scene, "num_links", source_layout.link_count, np.int64)
    _write_scalar(scene, "tx_count", source_layout.tx_count, np.int32)
    _write_scalar(scene, "rx_count", source_layout.rx_count, np.int32)
    if assets.label_file is not None:
        _write_string(scene, "label_file", assets.label_file.as_posix())
    if run.scene_file is not None:
        _write_string(scene, "scene_file", run.scene_file.as_posix())

    with h5py.File(_first_source_h5(manifest, run.run_dir), "r") as src:
        for name in (
            "center_frequency_hz",
            "bandwidth_hz",
            "num_subcarriers",
            "subcarrier_spacing_hz",
        ):
            _copy_dataset_if_present(src, h5, f"frequency/{name}")
        _copy_dataset_if_present(src, h5, "frequency/frequencies_hz", compression="gzip")
        for path in (
            "link/tx_role",
            "link/rx_role",
            "link/phy_link_direction",
            "link/duplex_mode",
            "antenna/tx_num_ant",
            "antenna/rx_num_ant",
            "antenna/tx_num_rows",
            "antenna/tx_num_cols",
            "antenna/rx_num_rows",
            "antenna/rx_num_cols",
            "antenna/tx_spacing_lambda",
            "antenna/rx_spacing_lambda",
            "antenna/tx_pattern",
            "antenna/rx_pattern",
            "antenna/tx_polarization",
            "antenna/rx_polarization",
        ):
            _copy_dataset_if_present(src, h5, path, compression="gzip")

    source = h5.require_group("source")
    _write_string(source, "manifest_json", json.dumps(manifest, sort_keys=True))
    _write_string(source, "config_snapshot_json", json.dumps(manifest.get("config_snapshot", {})))
    if config.embed_assets:
        _write_assets(h5, assets)


def _scene_metadata(
    run: SourceSceneRun,
    manifest: dict[str, Any],
    source_layout: _SourceLayout,
    assets: Any,
    config: _WorkerConfig,
) -> dict[str, Any]:
    label_summary: dict[str, Any] = {}
    if assets.label_file is not None:
        try:
            label = json.loads(assets.label_file.read_text(encoding="utf-8"))
            label_summary = {
                "label_version": label.get("label_version") or label.get("label_schema"),
                "generator": label.get("generator"),
                "scene_id": label.get("scene_id"),
                "bs_count": len(label.get("bs_points", [])),
                "ue_count": len(label.get("ue_points", [])),
                "group_count": len(label.get("groups", [])),
            }
        except (OSError, json.JSONDecodeError):
            label_summary = {"error": "failed_to_read_label_summary"}
    return {
        "scene_id": run.scene_id,
        "scene_class": run.scene_class,
        "source_index": run.index,
        "source_run_dir": run.run_dir.as_posix(),
        "source_manifest": run.manifest_path.as_posix(),
        "label_file": assets.label_file.as_posix() if assets.label_file else "",
        "scene_file": run.scene_file.as_posix() if run.scene_file else "",
        "label_summary": label_summary,
        "floorplan_found": assets.floorplan_png is not None,
        "floorplan_file": assets.floorplan_png.as_posix() if assets.floorplan_png else "",
        "label_floorplan_file": assets.label_floorplan_png.as_posix()
        if assets.label_floorplan_png
        else "",
        "link_shape": {
            "num_links": source_layout.link_count,
            "tx_count": source_layout.tx_count,
            "rx_count": source_layout.rx_count,
            "tx_role": source_layout.tx_role,
            "rx_role": source_layout.rx_role,
        },
        "source_config": manifest.get("config_snapshot", {}),
        "postprocess": {
            "label_compression": config.label_compression,
            "metadata_compression": config.metadata_compression,
        },
    }


def _write_global_indexes(
    output_root: Path,
    scene_results: tuple[RTLabelSceneResult, ...],
    link_index_format: str,
) -> dict[str, Path]:
    indexed = [
        result
        for result in scene_results
        if result.status in ("completed", "skipped") and result.h5_path
    ]
    if not indexed:
        return {}
    if link_index_format != "pt":
        msg = f"Unsupported link_index_format={link_index_format!r}; currently only 'pt'"
        raise ValueError(msg)

    import torch

    scene_ids: list[str] = []
    h5_paths: list[str] = []
    global_link_id: list[np.ndarray] = []
    scene_row: list[np.ndarray] = []
    local_link_index: list[np.ndarray] = []
    tx_id: list[np.ndarray] = []
    rx_id: list[np.ndarray] = []
    bs_id: list[np.ndarray] = []
    ue_id: list[np.ndarray] = []
    target_xy: list[np.ndarray] = []
    valid_mask: list[np.ndarray] = []
    geometric_distance_m: list[np.ndarray] = []
    first_path_delay_s: list[np.ndarray] = []
    first_path_propagation_range_m: list[np.ndarray] = []
    strongest_path_delay_s: list[np.ndarray] = []
    path_power_db: list[np.ndarray] = []
    los_flag: list[np.ndarray] = []
    nlos_flag: list[np.ndarray] = []
    path_count: list[np.ndarray] = []
    ue_room_index: list[np.ndarray] = []
    bs_room_index: list[np.ndarray] = []
    target_room_index: list[np.ndarray] = []
    target_room_type_id: list[np.ndarray] = []
    target_room_area_m2: list[np.ndarray] = []
    parquet_scene_id: list[str] = []
    parquet_h5_path: list[str] = []
    parquet_target_room_id: list[str] = []
    parquet_target_room_type: list[str] = []

    global_room_id: list[np.ndarray] = []
    room_scene_row: list[np.ndarray] = []
    local_room_index: list[np.ndarray] = []
    room_id: list[str] = []
    room_type: list[str] = []
    room_type_id: list[np.ndarray] = []
    room_floor_area_m2: list[np.ndarray] = []
    room_bounds_xy: list[np.ndarray] = []
    room_ue_count: list[np.ndarray] = []
    room_bs_count: list[np.ndarray] = []
    cursor = 0
    room_cursor = 0
    for row, result in enumerate(indexed):
        assert result.h5_path is not None
        h5_path = _relative_or_absolute(result.h5_path, output_root)
        scene_ids.append(result.scene_id)
        h5_paths.append(h5_path)
        with h5py.File(result.h5_path, "r") as h5:
            count = int(h5["link_id"].shape[0])
            room_count = int(h5["rooms/room_id"].shape[0])
            local_rooms = np.arange(room_count, dtype=np.int32)
            room_ids = [_decode_h5_string(value) for value in h5["rooms/room_id"][()]]
            room_types = [_decode_h5_string(value) for value in h5["rooms/room_type"][()]]

            global_room_id.append(np.arange(room_cursor, room_cursor + room_count, dtype=np.int64))
            room_scene_row.append(np.full(room_count, row, dtype=np.int32))
            local_room_index.append(local_rooms)
            room_id.extend(room_ids)
            room_type.extend(room_types)
            room_type_id.append(np.asarray(h5["rooms/room_type_id"][()], dtype=np.int32))
            room_floor_area_m2.append(np.asarray(h5["rooms/floor_area_m2"][()], dtype=np.float32))
            room_bounds_xy.append(np.asarray(h5["rooms/bounds_xy"][()], dtype=np.float32))
            room_ue_count.append(np.asarray(h5["rooms/ue_count"][()], dtype=np.int32))
            room_bs_count.append(np.asarray(h5["rooms/bs_count"][()], dtype=np.int32))

            labels = h5["labels/link"]
            target_room = np.asarray(h5["link/target_room_index"][()], dtype=np.int32)
            target_type = np.asarray(h5["link/target_room_type_id"][()], dtype=np.int32)
            target_area = np.asarray(h5["link/target_room_area_m2"][()], dtype=np.float32)
            global_link_id.append(np.arange(cursor, cursor + count, dtype=np.int64))
            scene_row.append(np.full(count, row, dtype=np.int32))
            local_link_index.append(np.asarray(h5["link_id"][()], dtype=np.int64))
            tx_id.append(np.asarray(h5["tx_id"][()], dtype=np.int64))
            rx_id.append(np.asarray(h5["rx_id"][()], dtype=np.int64))
            bs_id.append(np.asarray(h5["bs_id"][()], dtype=np.int64))
            ue_id.append(np.asarray(h5["ue_id"][()], dtype=np.int64))
            target_xy.append(np.asarray(h5["target_xy"][()], dtype=np.float32))
            valid_mask.append(np.asarray(h5["valid_mask"][()], dtype=np.bool_))
            geometric_distance_m.append(
                np.asarray(labels["geometric_distance_m"][()], dtype=np.float32)
            )
            first_path_delay_s.append(
                np.asarray(labels["first_path_delay_s"][()], dtype=np.float32)
            )
            first_path_propagation_range_m.append(
                np.asarray(labels["first_path_propagation_range_m"][()], dtype=np.float32)
            )
            strongest_path_delay_s.append(
                np.asarray(labels["strongest_path_delay_s"][()], dtype=np.float32)
            )
            path_power_db.append(np.asarray(labels["path_power_db"][()], dtype=np.float32))
            los_flag.append(np.asarray(labels["los_flag"][()], dtype=np.bool_))
            nlos_flag.append(np.asarray(labels["nlos_flag"][()], dtype=np.bool_))
            path_count.append(np.asarray(labels["path_count"][()], dtype=np.int32))
            ue_room_index.append(np.asarray(h5["link/ue_room_index"][()], dtype=np.int32))
            bs_room_index.append(np.asarray(h5["link/bs_room_index"][()], dtype=np.int32))
            target_room_index.append(target_room)
            target_room_type_id.append(target_type)
            target_room_area_m2.append(target_area)
            parquet_scene_id.extend([result.scene_id] * count)
            parquet_h5_path.extend([h5_path] * count)
            parquet_target_room_id.extend(_strings_for_room_indices(target_room, room_ids))
            parquet_target_room_type.extend(_strings_for_room_indices(target_room, room_types))
        cursor += count
        room_cursor += room_count

    payload = {
        "schema_version": RT_LABEL_SCENE_SCHEMA_VERSION,
        "contract_name": "sionna_rt_label_scene_link_index",
        "scene_ids": scene_ids,
        "scene_h5_paths": h5_paths,
        "global_link_id": torch.from_numpy(np.concatenate(global_link_id)),
        "scene_row": torch.from_numpy(np.concatenate(scene_row)),
        "local_link_index": torch.from_numpy(np.concatenate(local_link_index)),
        "tx_id": torch.from_numpy(np.concatenate(tx_id)),
        "rx_id": torch.from_numpy(np.concatenate(rx_id)),
        "bs_id": torch.from_numpy(np.concatenate(bs_id)),
        "ue_id": torch.from_numpy(np.concatenate(ue_id)),
        "target_xy": torch.from_numpy(np.concatenate(target_xy)),
        "valid_mask": torch.from_numpy(np.concatenate(valid_mask)),
        "geometric_distance_m": torch.from_numpy(np.concatenate(geometric_distance_m)),
        "first_path_delay_s": torch.from_numpy(np.concatenate(first_path_delay_s)),
        "first_path_propagation_range_m": torch.from_numpy(
            np.concatenate(first_path_propagation_range_m)
        ),
        "strongest_path_delay_s": torch.from_numpy(np.concatenate(strongest_path_delay_s)),
        "path_power_db": torch.from_numpy(np.concatenate(path_power_db)),
        "los_flag": torch.from_numpy(np.concatenate(los_flag)),
        "nlos_flag": torch.from_numpy(np.concatenate(nlos_flag)),
        "path_count": torch.from_numpy(np.concatenate(path_count)),
        "ue_room_index": torch.from_numpy(np.concatenate(ue_room_index)),
        "bs_room_index": torch.from_numpy(np.concatenate(bs_room_index)),
        "target_room_index": torch.from_numpy(np.concatenate(target_room_index)),
        "target_room_type_id": torch.from_numpy(np.concatenate(target_room_type_id)),
        "target_room_area_m2": torch.from_numpy(np.concatenate(target_room_area_m2)),
    }
    link_pt_path = output_root / "link_index.pt"
    torch.save(payload, link_pt_path)

    room_payload = {
        "schema_version": RT_LABEL_SCENE_SCHEMA_VERSION,
        "contract_name": "sionna_rt_label_scene_room_index",
        "scene_ids": scene_ids,
        "scene_h5_paths": h5_paths,
        "global_room_id": torch.from_numpy(np.concatenate(global_room_id))
        if global_room_id
        else torch.empty(0, dtype=torch.int64),
        "scene_row": torch.from_numpy(np.concatenate(room_scene_row))
        if room_scene_row
        else torch.empty(0, dtype=torch.int32),
        "local_room_index": torch.from_numpy(np.concatenate(local_room_index))
        if local_room_index
        else torch.empty(0, dtype=torch.int32),
        "room_id": room_id,
        "room_type": room_type,
        "room_type_id": torch.from_numpy(np.concatenate(room_type_id))
        if room_type_id
        else torch.empty(0, dtype=torch.int32),
        "floor_area_m2": torch.from_numpy(np.concatenate(room_floor_area_m2))
        if room_floor_area_m2
        else torch.empty(0, dtype=torch.float32),
        "bounds_xy": torch.from_numpy(np.concatenate(room_bounds_xy))
        if room_bounds_xy
        else torch.empty((0, 4), dtype=torch.float32),
        "ue_count": torch.from_numpy(np.concatenate(room_ue_count))
        if room_ue_count
        else torch.empty(0, dtype=torch.int32),
        "bs_count": torch.from_numpy(np.concatenate(room_bs_count))
        if room_bs_count
        else torch.empty(0, dtype=torch.int32),
    }
    room_pt_path = output_root / "room_index.pt"
    torch.save(room_payload, room_pt_path)

    link_parquet_path = output_root / "link_index.parquet"
    room_parquet_path = output_root / "room_index.parquet"
    _write_link_index_parquet(
        link_parquet_path,
        scene_id=parquet_scene_id,
        scene_h5_path=parquet_h5_path,
        global_link_id=np.concatenate(global_link_id),
        local_link_index=np.concatenate(local_link_index),
        tx_id=np.concatenate(tx_id),
        rx_id=np.concatenate(rx_id),
        bs_id=np.concatenate(bs_id),
        ue_id=np.concatenate(ue_id),
        target_xy=np.concatenate(target_xy),
        valid_mask=np.concatenate(valid_mask),
        geometric_distance_m=np.concatenate(geometric_distance_m),
        first_path_delay_s=np.concatenate(first_path_delay_s),
        first_path_propagation_range_m=np.concatenate(first_path_propagation_range_m),
        strongest_path_delay_s=np.concatenate(strongest_path_delay_s),
        path_power_db=np.concatenate(path_power_db),
        los_flag=np.concatenate(los_flag),
        nlos_flag=np.concatenate(nlos_flag),
        path_count=np.concatenate(path_count),
        target_room_index=np.concatenate(target_room_index),
        target_room_id=parquet_target_room_id,
        target_room_type=parquet_target_room_type,
        target_room_type_id=np.concatenate(target_room_type_id),
        target_room_area_m2=np.concatenate(target_room_area_m2),
    )
    _write_room_index_parquet(
        room_parquet_path,
        scene_ids=scene_ids,
        h5_paths=h5_paths,
        scene_row=np.concatenate(room_scene_row) if room_scene_row else np.empty(0, np.int32),
        global_room_id=np.concatenate(global_room_id) if global_room_id else np.empty(0, np.int64),
        local_room_index=np.concatenate(local_room_index)
        if local_room_index
        else np.empty(0, np.int32),
        room_id=room_id,
        room_type=room_type,
        room_type_id=np.concatenate(room_type_id) if room_type_id else np.empty(0, np.int32),
        floor_area_m2=np.concatenate(room_floor_area_m2)
        if room_floor_area_m2
        else np.empty(0, np.float32),
        bounds_xy=np.concatenate(room_bounds_xy)
        if room_bounds_xy
        else np.empty((0, 4), np.float32),
        ue_count=np.concatenate(room_ue_count) if room_ue_count else np.empty(0, np.int32),
        bs_count=np.concatenate(room_bs_count) if room_bs_count else np.empty(0, np.int32),
    )
    return {
        "link_index_pt": link_pt_path,
        "link_index_parquet": link_parquet_path,
        "room_index_pt": room_pt_path,
        "room_index_parquet": room_parquet_path,
    }


def _write_link_index_parquet(
    path: Path,
    *,
    scene_id: list[str],
    scene_h5_path: list[str],
    global_link_id: np.ndarray,
    local_link_index: np.ndarray,
    tx_id: np.ndarray,
    rx_id: np.ndarray,
    bs_id: np.ndarray,
    ue_id: np.ndarray,
    target_xy: np.ndarray,
    valid_mask: np.ndarray,
    geometric_distance_m: np.ndarray,
    first_path_delay_s: np.ndarray,
    first_path_propagation_range_m: np.ndarray,
    strongest_path_delay_s: np.ndarray,
    path_power_db: np.ndarray,
    los_flag: np.ndarray,
    nlos_flag: np.ndarray,
    path_count: np.ndarray,
    target_room_index: np.ndarray,
    target_room_id: list[str],
    target_room_type: list[str],
    target_room_type_id: np.ndarray,
    target_room_area_m2: np.ndarray,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {
            "global_link_id": global_link_id,
            "scene_id": scene_id,
            "scene_h5_path": scene_h5_path,
            "local_link_index": local_link_index,
            "tx_id": tx_id,
            "rx_id": rx_id,
            "bs_id": bs_id,
            "ue_id": ue_id,
            "target_x_m": target_xy[:, 0],
            "target_y_m": target_xy[:, 1],
            "valid_mask": valid_mask,
            "geometric_distance_m": geometric_distance_m,
            "first_path_delay_s": first_path_delay_s,
            "first_path_propagation_range_m": first_path_propagation_range_m,
            "strongest_path_delay_s": strongest_path_delay_s,
            "path_power_db": path_power_db,
            "los_flag": los_flag,
            "nlos_flag": nlos_flag,
            "path_count": path_count,
            "target_room_index": target_room_index,
            "target_room_id": target_room_id,
            "target_room_type": target_room_type,
            "target_room_type_id": target_room_type_id,
            "target_room_area_m2": target_room_area_m2,
        }
    )
    pq.write_table(table, path)


def _write_room_index_parquet(
    path: Path,
    *,
    scene_ids: list[str],
    h5_paths: list[str],
    scene_row: np.ndarray,
    global_room_id: np.ndarray,
    local_room_index: np.ndarray,
    room_id: list[str],
    room_type: list[str],
    room_type_id: np.ndarray,
    floor_area_m2: np.ndarray,
    bounds_xy: np.ndarray,
    ue_count: np.ndarray,
    bs_count: np.ndarray,
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    scene_id = [scene_ids[int(row)] for row in scene_row]
    scene_h5_path = [h5_paths[int(row)] for row in scene_row]
    table = pa.table(
        {
            "global_room_id": global_room_id,
            "scene_id": scene_id,
            "scene_h5_path": scene_h5_path,
            "local_room_index": local_room_index,
            "room_id": room_id,
            "room_type": room_type,
            "room_type_id": room_type_id,
            "floor_area_m2": floor_area_m2,
            "bounds_xmin_m": bounds_xy[:, 0],
            "bounds_ymin_m": bounds_xy[:, 1],
            "bounds_xmax_m": bounds_xy[:, 2],
            "bounds_ymax_m": bounds_xy[:, 3],
            "ue_count": ue_count,
            "bs_count": bs_count,
        }
    )
    pq.write_table(table, path)


def _write_dataset_manifest(
    *,
    output_root: Path,
    config: RTLabelSceneBuildConfig,
    scene_results: tuple[RTLabelSceneResult, ...],
    index_paths: dict[str, Path],
    elapsed_seconds: float,
) -> Path:
    manifest = {
        "schema_version": RT_LABEL_SCENE_SCHEMA_VERSION,
        "contract_name": "sionna_rt_label_scene_dataset_manifest",
        "created_at": datetime.now(UTC).isoformat(),
        "producer": f"sionna_measurement_sim {__version__}",
        "command": list(config.command or sys.argv),
        "input_root": config.input_root.as_posix() if config.input_root else "",
        "output_root": output_root.as_posix(),
        "scene_count": len(scene_results),
        "completed": sum(1 for result in scene_results if result.status == "completed"),
        "failed": sum(1 for result in scene_results if result.status == "failed"),
        "skipped": sum(1 for result in scene_results if result.status == "skipped"),
        "elapsed_seconds": elapsed_seconds,
        "workers": config.workers,
        "label_compression": config.label_compression,
        "link_index": _relative_or_absolute(index_paths.get("link_index_pt"), output_root),
        "link_index_pt": _relative_or_absolute(index_paths.get("link_index_pt"), output_root),
        "link_index_parquet": _relative_or_absolute(
            index_paths.get("link_index_parquet"), output_root
        ),
        "room_index_pt": _relative_or_absolute(index_paths.get("room_index_pt"), output_root),
        "room_index_parquet": _relative_or_absolute(
            index_paths.get("room_index_parquet"), output_root
        ),
        "scenes": [_scene_result_to_manifest(result, output_root) for result in scene_results],
    }
    path = output_root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _scene_result_to_manifest(result: RTLabelSceneResult, output_root: Path) -> dict[str, Any]:
    return {
        "scene_id": result.scene_id,
        "status": result.status,
        "h5_path": _relative_or_absolute(result.h5_path, output_root) if result.h5_path else "",
        "num_links": result.num_links,
        "valid_link_count": result.valid_link_count,
        "label_dataset_count": result.label_dataset_count,
        "source_shards": list(result.source_shards),
        "source_run_dir": result.source_run_dir.as_posix() if result.source_run_dir else "",
        "label_file": result.label_file.as_posix() if result.label_file else "",
        "scene_file": result.scene_file.as_posix() if result.scene_file else "",
        "floorplan_found": result.floorplan_found,
        "elapsed_seconds": result.elapsed_seconds,
        "source_bytes": result.source_bytes,
        "output_bytes": result.output_bytes,
        "path_power_min_db": result.path_power_min_db,
        "path_power_max_db": result.path_power_max_db,
        "path_power_quantiles_db": result.path_power_quantiles_db,
        "error": result.error,
    }


def _result_from_existing_scene(
    run: SourceSceneRun,
    output_path: Path,
    start: float,
) -> RTLabelSceneResult:
    validate_rt_label_scene_h5(output_path)
    with h5py.File(output_path, "r") as h5:
        power = np.asarray(h5["labels/link/path_power_db"][()])
        valid = np.asarray(h5["valid_mask"][()], dtype=np.bool_)
        return RTLabelSceneResult(
            scene_id=run.scene_id,
            status="skipped",
            h5_path=output_path,
            num_links=int(h5["link_id"].shape[0]),
            valid_link_count=int(np.count_nonzero(valid)),
            label_dataset_count=len(h5["labels/link"]),
            source_run_dir=run.run_dir,
            label_file=run.label_file,
            scene_file=run.scene_file,
            floorplan_found=bool(h5["floorplan/found"][()])
            if "floorplan/found" in h5
            else False,
            elapsed_seconds=time.perf_counter() - start,
            output_bytes=output_path.stat().st_size,
            path_power_min_db=float(np.min(power)) if power.size else None,
            path_power_max_db=float(np.max(power)) if power.size else None,
            path_power_quantiles_db=_quantiles(power),
        )


def _first_source_h5(manifest: dict[str, Any], run_dir: Path) -> Path:
    results = manifest.get("results", [])
    if not results:
        raise ValueError("manifest has no result entries")
    return _resolve_manifest_artifact(run_dir, Path(str(results[0]["result_h5"])))


def _decode_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in attrs.items():
        if isinstance(value, bytes):
            out[key] = value.decode("utf-8")
        else:
            out[key] = value
    return out


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    quantiles = np.percentile(values, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    keys = ("p0", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "p100")
    return {key: float(value) for key, value in zip(keys, quantiles, strict=True)}


def _validate_build_config(config: RTLabelSceneBuildConfig) -> None:
    if config.workers < 1:
        raise ValueError("workers must be >= 1")
    if config.limit is not None and config.limit < 1:
        raise ValueError("limit must be positive when provided")
    if config.label_compression not in ("none", "gzip", "lzf"):
        raise ValueError("label_compression must be none/gzip/lzf")
    if config.metadata_compression not in ("none", "gzip", "lzf"):
        raise ValueError("metadata_compression must be none/gzip/lzf")
    if not (1 <= int(config.gzip_level) <= 9):
        raise ValueError("gzip_level must be in [1, 9]")
