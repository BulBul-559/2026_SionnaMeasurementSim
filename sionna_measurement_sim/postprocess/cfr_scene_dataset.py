"""Build compact per-scene CFR truth datasets from raw shard HDF5 runs.

This module is intentionally offline-only. It reads the production shard HDF5
outputs and writes a separate training-oriented dataset without modifying the
source run directories.
"""

from __future__ import annotations

import json
import os
import shutil
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

CFR_SCENE_CONTRACT_NAME = "sionna_cfr_scene_dataset"
CFR_SCENE_SCHEMA_VERSION = "0.2.0"
UTF8_DTYPE = h5py.string_dtype(encoding="utf-8")
SOURCE_SHARD_ID_DTYPE = "S32"
POWER_FLOOR_LINEAR = 1e-30

_REQUIRED_SCENE_DATASETS = (
    "cfr",
    "link_id",
    "tx_id",
    "rx_id",
    "ue_id",
    "bs_id",
    "tx_pos",
    "rx_pos",
    "ue_pos",
    "bs_pos",
    "target_xy",
    "valid_mask",
    "source_shard_id",
    "source_shard_index",
    "source_local_index",
    "source_local_tx_index",
    "source_local_rx_index",
    "csi_power_db",
    "cfr_power_linear",
    "cfr_norm",
    "link_power_bucket",
    "is_finite",
    "nan_count",
    "scene_id",
    "scene_metadata",
    "frequency/frequencies_hz",
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
class SourceSceneRun:
    """One completed raw sharded scene run to post-process."""

    scene_id: str
    run_dir: Path
    manifest_path: Path
    label_file: Path | None = None
    scene_file: Path | None = None
    scene_class: str = ""
    index: int | None = None


@dataclass(frozen=True)
class CFRSceneBuildConfig:
    """Configuration for offline CFR scene dataset construction."""

    input_root: Path | None = None
    output_root: Path = Path("outputs/cfr_scene_dataset")
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
    cfr_compression: str = "none"
    gzip_level: int = 1
    metadata_compression: str = "gzip"
    link_index_format: str = "pt"
    command: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class CFRSceneResult:
    """Post-processing result for a single scene."""

    scene_id: str
    status: str
    h5_path: Path | None = None
    num_links: int = 0
    valid_link_count: int = 0
    cfr_shape: tuple[int, ...] = ()
    dtype: str = "complex64"
    source_shards: tuple[str, ...] = ()
    source_run_dir: Path | None = None
    label_file: Path | None = None
    scene_file: Path | None = None
    floorplan_found: bool = False
    elapsed_seconds: float = 0.0
    output_bytes: int = 0
    source_bytes: int = 0
    power_min_db: float | None = None
    power_max_db: float | None = None
    power_quantiles_db: dict[str, float] = field(default_factory=dict)
    error: str = ""


@dataclass(frozen=True)
class CFRSceneDatasetResult:
    """Result for a dataset build run."""

    output_root: Path
    manifest_path: Path
    link_index_path: Path | None
    link_index_parquet_path: Path | None
    room_index_path: Path | None
    room_index_parquet_path: Path | None
    scene_results: tuple[CFRSceneResult, ...]
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


def build_cfr_scene_dataset(config: CFRSceneBuildConfig) -> CFRSceneDatasetResult:
    """Build a scene-level CFR truth dataset from raw sharded run outputs."""

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
            cfr_compression=config.cfr_compression,
            gzip_level=config.gzip_level,
            metadata_compression=config.metadata_compression,
        )
        for run in runs
    ]

    if config.workers == 1 or len(worker_payloads) <= 1:
        scene_results = tuple(_process_scene_worker(payload) for payload in worker_payloads)
    else:
        results: list[CFRSceneResult | None] = [None] * len(worker_payloads)
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
    return CFRSceneDatasetResult(
        output_root=output_root,
        manifest_path=manifest_path,
        link_index_path=index_paths.get("link_index_pt"),
        link_index_parquet_path=index_paths.get("link_index_parquet"),
        room_index_path=index_paths.get("room_index_pt"),
        room_index_parquet_path=index_paths.get("room_index_parquet"),
        scene_results=scene_results,
        elapsed_seconds=time.perf_counter() - start,
    )


def discover_source_scene_runs(
    *,
    input_root: Path | None,
    run_dirs: tuple[Path, ...] = (),
    scene_ids: tuple[str, ...] = (),
    start_index: int = 0,
    limit: int | None = None,
) -> tuple[SourceSceneRun, ...]:
    """Discover completed source scene runs from explicit dirs or a queue root."""

    selected_ids = set(scene_ids)
    runs: list[SourceSceneRun] = []
    if run_dirs:
        runs.extend(_run_from_dir(Path(run_dir)) for run_dir in run_dirs)
    elif input_root is not None:
        root = Path(input_root)
        queue_manifest = root / "scene_index_run_manifest.jsonl"
        if queue_manifest.exists():
            for line in queue_manifest.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status") != "completed":
                    continue
                scene_id = str(row.get("scene_key") or row.get("scene_id") or "")
                if selected_ids and scene_id not in selected_ids:
                    continue
                run_dir = Path(str(row["output_dir"]))
                runs.append(
                    SourceSceneRun(
                        scene_id=scene_id,
                        run_dir=run_dir,
                        manifest_path=run_dir / "manifest" / "manifest.json",
                        label_file=Path(str(row["label_file"]))
                        if row.get("label_file")
                        else None,
                        scene_file=Path(str(row["scene_file"]))
                        if row.get("scene_file")
                        else None,
                        scene_class=str(row.get("scene_class", "")),
                        index=_optional_int(row.get("index")),
                    )
                )
        elif (root / "manifest" / "manifest.json").exists():
            runs.append(_run_from_dir(root))
        elif (root / "runs").is_dir():
            for run_dir in sorted((root / "runs").iterdir()):
                if (run_dir / "manifest" / "manifest.json").exists():
                    run = _run_from_dir(run_dir)
                    if not selected_ids or run.scene_id in selected_ids:
                        runs.append(run)
        else:
            msg = f"Could not discover scene runs under {root}"
            raise FileNotFoundError(msg)
    else:
        msg = "Either input_root or run_dirs must be provided"
        raise ValueError(msg)

    runs = [run for run in runs if run.manifest_path.exists()]
    if selected_ids:
        runs = [run for run in runs if run.scene_id in selected_ids]
    if start_index:
        runs = runs[int(start_index) :]
    if limit is not None:
        runs = runs[: int(limit)]
    return tuple(runs)


def validate_cfr_scene_h5(path: str | Path) -> None:
    """Validate the compact CFR scene HDF5 contract."""

    with h5py.File(path, "r") as h5:
        contract = _read_string(h5["meta/contract_name"])
        if contract != CFR_SCENE_CONTRACT_NAME:
            msg = f"Unsupported CFR scene contract: {contract!r}"
            raise ValueError(msg)
        for dataset_path in _REQUIRED_SCENE_DATASETS:
            if dataset_path not in h5:
                msg = f"Missing required CFR scene dataset /{dataset_path}"
                raise ValueError(msg)
        cfr = h5["cfr"]
        if cfr.ndim != 4:
            msg = f"/cfr must be [link,rx_ant,tx_ant,subcarrier], got {cfr.shape}"
            raise ValueError(msg)
        link_count = cfr.shape[0]
        for name in (
            "link_id",
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
            "csi_power_db",
            "cfr_power_linear",
            "cfr_norm",
            "link_power_bucket",
            "is_finite",
            "nan_count",
        ):
            if h5[name].shape != (link_count,):
                msg = f"/{name} must match link count {link_count}, got {h5[name].shape}"
                raise ValueError(msg)
        for name in ("tx_pos", "rx_pos", "ue_pos", "bs_pos"):
            if h5[name].shape != (link_count, 3):
                msg = f"/{name} must be [link,3], got {h5[name].shape}"
                raise ValueError(msg)
        if h5["target_xy"].shape != (link_count, 2):
            msg = f"/target_xy must be [link,2], got {h5['target_xy'].shape}"
            raise ValueError(msg)
        if h5["frequency/frequencies_hz"].shape[0] != cfr.shape[-1]:
            msg = "/frequency/frequencies_hz length must match /cfr subcarrier dimension"
            raise ValueError(msg)
        room_count = h5["rooms/room_id"].shape[0]
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
    cfr_compression: str
    gzip_level: int
    metadata_compression: str


def _process_scene_worker(config: _WorkerConfig) -> CFRSceneResult:
    start = time.perf_counter()
    try:
        return _process_scene(config, start)
    except Exception as exc:  # noqa: BLE001 - worker must report scene failure.
        return CFRSceneResult(
            scene_id=config.run.scene_id,
            status="failed",
            source_run_dir=config.run.run_dir,
            label_file=config.run.label_file,
            scene_file=config.run.scene_file,
            elapsed_seconds=time.perf_counter() - start,
            error=f"{type(exc).__name__}: {exc}",
        )


def _process_scene(config: _WorkerConfig, start: float) -> CFRSceneResult:
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
    label_file = run.label_file or _optional_path(manifest.get("label_file"))
    scene_file = run.scene_file or _optional_path(manifest.get("scene_file"))
    label_file = _resolve_optional_existing(label_file)
    scene_file = _resolve_optional_existing(scene_file)

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
        validate_cfr_scene_h5(output_path)

    with h5py.File(output_path, "r") as h5:
        power = np.asarray(h5["csi_power_db"][()])
        valid = np.asarray(h5["valid_mask"][()], dtype=np.bool_)
        quantiles = _power_quantiles(power)
        return CFRSceneResult(
            scene_id=scene_id,
            status="completed",
            h5_path=output_path,
            num_links=int(h5["cfr"].shape[0]),
            valid_link_count=int(np.count_nonzero(valid)),
            cfr_shape=tuple(int(dim) for dim in h5["cfr"].shape),
            dtype=str(h5["cfr"].dtype),
            source_shards=tuple(str(item.get("shard_id", "")) for item in source_entries),
            source_run_dir=run.run_dir,
            label_file=label_file,
            scene_file=scene_file,
            floorplan_found=assets.floorplan_png is not None,
            elapsed_seconds=time.perf_counter() - start,
            output_bytes=output_path.stat().st_size,
            source_bytes=source_bytes,
            power_min_db=float(np.min(power)) if power.size else None,
            power_max_db=float(np.max(power)) if power.size else None,
            power_quantiles_db=quantiles,
        )


@dataclass(frozen=True)
class _SourceLayout:
    link_count: int
    cfr_shape: tuple[int, int, int, int]
    tx_count: int
    rx_count: int
    rx_ant: int
    tx_ant: int
    subcarriers: int
    tx_role: str
    rx_role: str


@dataclass(frozen=True)
class _SceneAssets:
    label_file: Path | None
    label_report_json: Path | None
    floorplan_png: Path | None
    floorplan_meta_json: Path | None
    label_floorplan_png: Path | None


@dataclass(frozen=True)
class _RoomRecord:
    local_index: int
    room_id: str
    room_type: str
    room_type_id: int
    floor_area_m2: float
    bounds_xy: tuple[float, float, float, float]
    ue_count: int
    bs_count: int


@dataclass(frozen=True)
class _RoomLayout:
    rooms: tuple[_RoomRecord, ...]
    ue_room_index: dict[int, int]
    bs_room_index: dict[int, int]

    @property
    def floor_area_by_index(self) -> np.ndarray:
        return np.asarray([room.floor_area_m2 for room in self.rooms], dtype=np.float32)

    @property
    def type_id_by_index(self) -> np.ndarray:
        return np.asarray([room.room_type_id for room in self.rooms], dtype=np.int32)


def _write_scene_h5(
    path: Path,
    *,
    run: SourceSceneRun,
    manifest: dict[str, Any],
    source_entries: list[dict[str, Any]],
    source_paths: tuple[Path, ...],
    source_layout: _SourceLayout,
    assets: _SceneAssets,
    room_layout: _RoomLayout,
    config: _WorkerConfig,
) -> None:
    metadata_kwargs = _compression_kwargs(config.metadata_compression, config.gzip_level)
    cfr_kwargs = _compression_kwargs(config.cfr_compression, config.gzip_level)
    if config.cfr_compression == "none":
        cfr_kwargs = {}

    with h5py.File(path, "w") as h5:
        _write_core_metadata(h5, run, manifest, source_layout, assets, config)
        _write_rooms(h5, room_layout, metadata_kwargs)
        _create_dataset(
            h5,
            "cfr",
            shape=source_layout.cfr_shape,
            dtype=np.complex64,
            attrs={
                "unit": "linear_complex",
                "index_order": "link,rx_ant,tx_ant,subcarrier",
            },
            **cfr_kwargs,
        )
        _create_dataset(h5, "link_id", shape=(source_layout.link_count,), dtype=np.int64)
        for name in ("tx_id", "rx_id", "ue_id", "bs_id"):
            _create_dataset(h5, name, shape=(source_layout.link_count,), dtype=np.int64)
        for name in ("tx_pos", "rx_pos", "ue_pos", "bs_pos"):
            _create_dataset(
                h5,
                name,
                shape=(source_layout.link_count, 3),
                dtype=np.float32,
                attrs={"unit": "m", "index_order": "link,xyz"},
                **metadata_kwargs,
            )
        _create_dataset(
            h5,
            "target_xy",
            shape=(source_layout.link_count, 2),
            dtype=np.float32,
            attrs={"unit": "m", "index_order": "link,xy"},
            **metadata_kwargs,
        )
        for name in ("valid_mask", "is_finite"):
            _create_dataset(h5, name, shape=(source_layout.link_count,), dtype=np.bool_)
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
            "nan_count",
        ):
            _create_dataset(h5, name, shape=(source_layout.link_count,), dtype=np.int32)
        for name in ("csi_power_db", "cfr_power_linear", "cfr_norm"):
            _create_dataset(
                h5,
                name,
                shape=(source_layout.link_count,),
                dtype=np.float32,
                attrs={"unit": "dB" if name == "csi_power_db" else "linear"},
                **metadata_kwargs,
            )
        _create_dataset(
            h5,
            "link_power_bucket",
            shape=(source_layout.link_count,),
            dtype=np.int8,
        )
        for name in ("ue_room_index", "bs_room_index", "target_room_index"):
            _create_dataset(
                h5,
                f"link/{name}",
                shape=(source_layout.link_count,),
                dtype=np.int32,
            )
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
                )
        if offset != source_layout.link_count:
            msg = f"scene link count mismatch: wrote {offset}, expected {source_layout.link_count}"
            raise RuntimeError(msg)


def _append_source_shard(
    out: h5py.File,
    src: h5py.File,
    *,
    entry: dict[str, Any],
    offset: int,
    tx_role: str,
    rx_role: str,
    room_layout: _RoomLayout,
) -> int:
    cfr = np.asarray(src["channel/truth/cfr"][()], dtype=np.complex64)
    tx_count, rx_count, rx_ant, tx_ant, subcarriers = cfr.shape
    link_count = tx_count * rx_count
    cfr_links = cfr.reshape(link_count, rx_ant, tx_ant, subcarriers)
    out["cfr"][offset : offset + link_count] = cfr_links

    global_tx = _read_global_indices(src, "tx", entry, tx_count)
    global_rx = _read_global_indices(src, "rx", entry, rx_count)
    tx_ids = np.repeat(global_tx, rx_count).astype(np.int64, copy=False)
    rx_ids = np.tile(global_rx, tx_count).astype(np.int64, copy=False)
    tx_positions = np.asarray(src["topology/tx_positions_m"][()], dtype=np.float32)
    rx_positions = np.asarray(src["topology/rx_positions_m"][()], dtype=np.float32)
    tx_pos = np.repeat(tx_positions, rx_count, axis=0)
    rx_pos = np.tile(rx_positions, (tx_count, 1))

    finite = np.isfinite(cfr_links)
    is_finite = np.all(finite.reshape(link_count, -1), axis=1)
    nan_count = np.count_nonzero(~finite.reshape(link_count, -1), axis=1).astype(np.int32)
    power_linear = np.mean(np.abs(cfr_links) ** 2, axis=(1, 2, 3)).astype(np.float32)
    power_db = (10.0 * np.log10(np.maximum(power_linear, POWER_FLOOR_LINEAR))).astype(
        np.float32
    )
    cfr_norm = np.sqrt(np.sum(np.abs(cfr_links) ** 2, axis=(1, 2, 3))).astype(np.float32)
    has_signal = np.asarray(src["channel/truth/has_geometric_signal"][()]).reshape(link_count)
    valid = np.asarray(has_signal, dtype=np.bool_) & is_finite

    if tx_role == "ue":
        ue_id, bs_id = tx_ids, rx_ids
        ue_pos, bs_pos = tx_pos, rx_pos
    elif rx_role == "ue":
        ue_id, bs_id = rx_ids, tx_ids
        ue_pos, bs_pos = rx_pos, tx_pos
    else:
        ue_id, bs_id = tx_ids, rx_ids
        ue_pos, bs_pos = tx_pos, rx_pos
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
    local_tx = np.repeat(np.arange(tx_count, dtype=np.int32), rx_count)
    local_rx = np.tile(np.arange(rx_count, dtype=np.int32), tx_count)
    local_link = (local_tx * rx_count + local_rx).astype(np.int32)
    slc = slice(offset, offset + link_count)
    out["link_id"][slc] = np.arange(offset, offset + link_count, dtype=np.int64)
    out["tx_id"][slc] = tx_ids
    out["rx_id"][slc] = rx_ids
    out["ue_id"][slc] = ue_id
    out["bs_id"][slc] = bs_id
    out["tx_pos"][slc] = tx_pos
    out["rx_pos"][slc] = rx_pos
    out["ue_pos"][slc] = ue_pos
    out["bs_pos"][slc] = bs_pos
    out["target_xy"][slc] = ue_pos[:, :2]
    out["valid_mask"][slc] = valid
    out["is_finite"][slc] = is_finite
    out["nan_count"][slc] = nan_count
    out["csi_power_db"][slc] = power_db
    out["cfr_power_linear"][slc] = power_linear
    out["cfr_norm"][slc] = cfr_norm
    out["link_power_bucket"][slc] = _power_buckets(power_db)
    out["link/ue_room_index"][slc] = ue_room_index
    out["link/bs_room_index"][slc] = bs_room_index
    out["link/target_room_index"][slc] = target_room_index
    out["link/target_room_area_m2"][slc] = target_room_area
    out["link/target_room_type_id"][slc] = target_room_type_id
    out["source_shard_id"][slc] = np.full(link_count, shard_id.encode("utf-8"), dtype="S32")
    out["source_shard_index"][slc] = np.full(link_count, shard_index, dtype=np.int32)
    out["source_local_index"][slc] = local_link
    out["source_local_tx_index"][slc] = local_tx
    out["source_local_rx_index"][slc] = local_rx
    return offset + link_count


def _write_core_metadata(
    h5: h5py.File,
    run: SourceSceneRun,
    manifest: dict[str, Any],
    source_layout: _SourceLayout,
    assets: _SceneAssets,
    config: _WorkerConfig,
) -> None:
    metadata = _scene_metadata(run, manifest, source_layout, assets, config)
    _write_string(h5, "scene_id", run.scene_id)
    _write_string(h5, "scene_metadata", json.dumps(metadata, sort_keys=True))
    meta = h5.require_group("meta")
    _write_string(meta, "contract_name", CFR_SCENE_CONTRACT_NAME)
    _write_string(meta, "schema_version", CFR_SCENE_SCHEMA_VERSION)
    _write_string(meta, "producer", f"sionna_measurement_sim {__version__}")
    _write_string(meta, "created_at", datetime.now(UTC).isoformat())
    _write_string(meta, "source_run_dir", run.run_dir.as_posix())
    _write_string(meta, "source_manifest", run.manifest_path.as_posix())
    _write_string(meta, "cfr_compression", config.cfr_compression)
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
    _write_scalar(scene, "rx_ant", source_layout.rx_ant, np.int32)
    _write_scalar(scene, "tx_ant", source_layout.tx_ant, np.int32)
    _write_scalar(scene, "subcarriers", source_layout.subcarriers, np.int32)
    if run.label_file is not None:
        _write_string(scene, "label_file", run.label_file.as_posix())
    if run.scene_file is not None:
        _write_string(scene, "scene_file", run.scene_file.as_posix())

    with h5py.File(_first_source_h5(manifest, run.run_dir), "r") as src:
        _copy_dataset_if_present(src, h5, "frequency/frequencies_hz", compression="gzip")
        for name in (
            "center_frequency_hz",
            "bandwidth_hz",
            "num_subcarriers",
            "subcarrier_spacing_hz",
        ):
            _copy_dataset_if_present(src, h5, f"frequency/{name}")
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


def _write_assets(h5: h5py.File, assets: _SceneAssets) -> None:
    group = h5.require_group("assets")
    for name, path in (
        ("label_json", assets.label_file),
        ("label_report_json", assets.label_report_json),
        ("label_floorplan_png", assets.label_floorplan_png),
    ):
        _write_asset_bytes(group, name, path)
    floorplan = h5.require_group("floorplan")
    _write_scalar(floorplan, "found", assets.floorplan_png is not None, np.bool_)
    _write_asset_bytes(floorplan, "image_png", assets.floorplan_png)
    _write_asset_bytes(floorplan, "meta_json", assets.floorplan_meta_json)


def _write_asset_bytes(group: h5py.Group, name: str, path: Path | None) -> None:
    path_group = group.require_group(name)
    _write_scalar(path_group, "present", path is not None and path.exists(), np.bool_)
    if path is None or not path.exists():
        _write_string(path_group, "source_path", "")
        return
    _write_string(path_group, "source_path", path.as_posix())
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    path_group.create_dataset("bytes", data=data, compression="gzip", compression_opts=1)


def _write_rooms(
    h5: h5py.File,
    room_layout: _RoomLayout,
    metadata_kwargs: dict[str, Any],
) -> None:
    rooms = h5.require_group("rooms")
    room_records = room_layout.rooms
    room_count = len(room_records)
    _write_string_vector(rooms, "room_id", [room.room_id for room in room_records])
    _write_string_vector(rooms, "room_type", [room.room_type for room in room_records])
    rooms.create_dataset(
        "room_type_id",
        data=np.asarray([room.room_type_id for room in room_records], dtype=np.int32),
    )
    rooms.create_dataset(
        "floor_area_m2",
        data=np.asarray([room.floor_area_m2 for room in room_records], dtype=np.float32),
        **metadata_kwargs,
    ).attrs["unit"] = "m^2"
    bounds = np.asarray([room.bounds_xy for room in room_records], dtype=np.float32)
    if not room_count:
        bounds = np.empty((0, 4), dtype=np.float32)
    bounds_dataset = rooms.create_dataset("bounds_xy", data=bounds, **metadata_kwargs)
    bounds_dataset.attrs["unit"] = "m"
    bounds_dataset.attrs["index_order"] = "room,xmin,ymin,xmax,ymax"
    rooms.create_dataset(
        "ue_count",
        data=np.asarray([room.ue_count for room in room_records], dtype=np.int32),
    )
    rooms.create_dataset(
        "bs_count",
        data=np.asarray([room.bs_count for room in room_records], dtype=np.int32),
    )


def _write_string_vector(group: h5py.Group, name: str, values: list[str]) -> None:
    if values:
        group.create_dataset(name, data=[str(value) for value in values], dtype=UTF8_DTYPE)
    else:
        group.create_dataset(name, shape=(0,), dtype=UTF8_DTYPE)


def _extract_room_layout(label_file: Path | None) -> _RoomLayout:
    if label_file is None or not label_file.exists():
        return _RoomLayout(rooms=(), ue_room_index={}, bs_room_index={})
    try:
        label = json.loads(label_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _RoomLayout(rooms=(), ue_room_index={}, bs_room_index={})

    ue_points = list(label.get("ue_points", []))
    bs_points = list(label.get("bs_points", []))
    ue_counts = _count_points_by_room(ue_points)
    bs_counts = _count_points_by_room(bs_points)
    top_level_points_by_room: dict[str, list[Any]] = {}
    for point in [*ue_points, *bs_points]:
        room_id = _point_room_id(point)
        if room_id:
            top_level_points_by_room.setdefault(room_id, []).append(point)

    room_order: list[str] = []
    room_source: dict[str, dict[str, Any]] = {}
    for index, group in enumerate(label.get("groups", [])):
        if not isinstance(group, dict):
            continue
        room_id = str(group.get("room_id") or group.get("name") or f"room_{index}")
        if room_id not in room_source:
            room_order.append(room_id)
        room_source[room_id] = group
    for point in [*ue_points, *bs_points]:
        room_id = _point_room_id(point)
        if room_id and room_id not in room_source:
            room_order.append(room_id)
            room_source[room_id] = {
                "room_id": room_id,
                "room_type": _point_room_type(point),
            }

    room_type_to_id: dict[str, int] = {}
    rooms: list[_RoomRecord] = []
    for local_index, room_id in enumerate(room_order):
        source = room_source.get(room_id, {})
        room_type = str(source.get("room_type") or "unknown")
        if room_type not in room_type_to_id:
            room_type_to_id[room_type] = len(room_type_to_id)
        floor_area = _optional_float(source.get("floor_area_m2"))
        bounds_xy = _room_bounds_xy(source, top_level_points_by_room.get(room_id, []))
        rooms.append(
            _RoomRecord(
                local_index=local_index,
                room_id=room_id,
                room_type=room_type,
                room_type_id=room_type_to_id[room_type],
                floor_area_m2=floor_area,
                bounds_xy=bounds_xy,
                ue_count=ue_counts.get(room_id, len(source.get("ue_points", []))),
                bs_count=bs_counts.get(room_id, len(source.get("bs_points", []))),
            )
        )

    room_id_to_index = {room.room_id: room.local_index for room in rooms}
    return _RoomLayout(
        rooms=tuple(rooms),
        ue_room_index=_point_room_index_by_global_id(ue_points, room_id_to_index),
        bs_room_index=_point_room_index_by_global_id(bs_points, room_id_to_index),
    )


def _count_points_by_room(points: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for point in points:
        room_id = _point_room_id(point)
        if room_id:
            counts[room_id] = counts.get(room_id, 0) + 1
    return counts


def _point_room_index_by_global_id(
    points: list[Any],
    room_id_to_index: dict[str, int],
) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for index, point in enumerate(points):
        room_id = _point_room_id(point)
        if room_id in room_id_to_index:
            mapping[index] = room_id_to_index[room_id]
    return mapping


def _point_room_id(point: Any) -> str:
    if isinstance(point, dict):
        return str(point.get("room_id") or "")
    return ""


def _point_room_type(point: Any) -> str:
    if isinstance(point, dict):
        return str(point.get("room_type") or "unknown")
    return "unknown"


def _room_bounds_xy(
    room: dict[str, Any],
    top_level_points: list[Any],
) -> tuple[float, float, float, float]:
    explicit = _parse_bounds_xy(room.get("bounds_xy") or room.get("bounds") or room.get("bbox"))
    if explicit is not None:
        return explicit
    xy: list[tuple[float, float]] = []
    for key in ("ue_points", "bs_points", "ue_positions", "bs_positions"):
        xy.extend(_xy_values(room.get(key, [])))
    xy.extend(_xy_values(top_level_points))
    if not xy:
        nan = float("nan")
        return (nan, nan, nan, nan)
    xs = [point[0] for point in xy]
    ys = [point[1] for point in xy]
    return (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))


def _parse_bounds_xy(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        keys = ("xmin", "ymin", "xmax", "ymax")
        if all(key in value for key in keys):
            return tuple(float(value[key]) for key in keys)  # type: ignore[return-value]
        keys = ("min_x", "min_y", "max_x", "max_y")
        if all(key in value for key in keys):
            return tuple(float(value[key]) for key in keys)  # type: ignore[return-value]
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return tuple(float(item) for item in value)  # type: ignore[return-value]
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(isinstance(item, (list, tuple)) and len(item) >= 2 for item in value)
    ):
        return (
            float(value[0][0]),
            float(value[0][1]),
            float(value[1][0]),
            float(value[1][1]),
        )
    return None


def _xy_values(values: Any) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    if not isinstance(values, list):
        return points
    for value in values:
        xy = _xy_from_point(value)
        if xy is not None:
            points.append(xy)
    return points


def _xy_from_point(value: Any) -> tuple[float, float] | None:
    if isinstance(value, dict):
        if "position" in value and isinstance(value["position"], (list, tuple)):
            position = value["position"]
            if len(position) >= 2:
                return (float(position[0]), float(position[1]))
        if "x" in value and "y" in value:
            return (float(value["x"]), float(value["y"]))
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return (float(value[0]), float(value[1]))
    return None


def _room_indices_for_ids(ids: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    return np.asarray([mapping.get(int(item), -1) for item in ids], dtype=np.int32)


def _room_values_for_indices(
    indices: np.ndarray,
    values: np.ndarray,
    *,
    fill_value: float | int,
    dtype: Any,
) -> np.ndarray:
    out = np.full(indices.shape, fill_value, dtype=dtype)
    valid = (indices >= 0) & (indices < len(values))
    out[valid] = values[indices[valid]]
    return out


def _optional_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _inspect_source_layout(
    entries: list[dict[str, Any]],
    source_paths: tuple[Path, ...],
) -> _SourceLayout:
    link_count = 0
    tx_total = 0
    rx_count = rx_ant = tx_ant = subcarriers = -1
    tx_role = rx_role = ""
    for entry, source_path in zip(entries, source_paths, strict=True):
        with h5py.File(source_path, "r") as h5:
            cfr_shape = tuple(int(dim) for dim in h5["channel/truth/cfr"].shape)
            if len(cfr_shape) != 5:
                msg = f"{source_path}: /channel/truth/cfr must be rank 5"
                raise ValueError(msg)
            tx, rx, local_rx_ant, local_tx_ant, local_subcarriers = cfr_shape
            if rx_count < 0:
                rx_count = rx
                rx_ant = local_rx_ant
                tx_ant = local_tx_ant
                subcarriers = local_subcarriers
                tx_role = _read_string(h5["link/tx_role"]) if "link/tx_role" in h5 else "tx"
                rx_role = _read_string(h5["link/rx_role"]) if "link/rx_role" in h5 else "rx"
            elif (rx, local_rx_ant, local_tx_ant, local_subcarriers) != (
                rx_count,
                rx_ant,
                tx_ant,
                subcarriers,
            ):
                msg = f"{source_path}: CFR non-TX dimensions differ across shards"
                raise ValueError(msg)
            link_count += tx * rx
            tx_total += tx
            if int(entry.get("ue_count", tx)) != tx:
                # Fallback shards can have custom names but still expose local TX shape.
                pass
    return _SourceLayout(
        link_count=link_count,
        cfr_shape=(link_count, rx_ant, tx_ant, subcarriers),
        tx_count=tx_total,
        rx_count=rx_count,
        rx_ant=rx_ant,
        tx_ant=tx_ant,
        subcarriers=subcarriers,
        tx_role=tx_role,
        rx_role=rx_role,
    )


def _collect_scene_assets(
    *,
    label_file: Path | None,
    scene_file: Path | None,
    label_name: str,
) -> _SceneAssets:
    label_report = None
    label_floorplan = None
    floorplan_png = None
    floorplan_meta = None
    if label_file is not None and label_file.exists():
        report_candidate = label_file.parent / "report" / f"{label_file.stem}_report.json"
        if report_candidate.exists():
            label_report = report_candidate
        scene_root = label_file.parent.parent
        label_floorplan_candidate = scene_root / "label_floorplan" / f"{label_file.stem}.png"
        if label_floorplan_candidate.exists():
            label_floorplan = label_floorplan_candidate
        floorplan_png = _select_floorplan_png(scene_root)
        floorplan_meta_candidate = scene_root / "floorplan" / "meta.json"
        if floorplan_meta_candidate.exists():
            floorplan_meta = floorplan_meta_candidate
    if floorplan_png is None and scene_file is not None and scene_file.exists():
        floorplan_png = _select_floorplan_png(scene_file.parent)
        floorplan_meta_candidate = scene_file.parent / "floorplan" / "meta.json"
        if floorplan_meta is None and floorplan_meta_candidate.exists():
            floorplan_meta = floorplan_meta_candidate
    if label_file is None and scene_file is not None:
        candidate = scene_file.parent / "label" / label_name
        if candidate.exists():
            label_file = candidate
    return _SceneAssets(
        label_file=label_file if label_file is not None and label_file.exists() else None,
        label_report_json=label_report,
        floorplan_png=floorplan_png,
        floorplan_meta_json=floorplan_meta,
        label_floorplan_png=label_floorplan,
    )


def _copy_scene_assets(output_root: Path, scene_id: str, assets: _SceneAssets) -> None:
    target = output_root / "assets" / scene_id
    target.mkdir(parents=True, exist_ok=True)
    for path in (
        assets.label_file,
        assets.label_report_json,
        assets.floorplan_png,
        assets.floorplan_meta_json,
        assets.label_floorplan_png,
    ):
        if path is not None and path.exists():
            shutil.copy2(path, target / path.name)


def _write_global_indexes(
    output_root: Path,
    scene_results: tuple[CFRSceneResult, ...],
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
    bs_id: list[np.ndarray] = []
    ue_id: list[np.ndarray] = []
    rx_pos: list[np.ndarray] = []
    tx_pos: list[np.ndarray] = []
    target_xy: list[np.ndarray] = []
    csi_power_db: list[np.ndarray] = []
    valid_mask: list[np.ndarray] = []
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
            room_type_ids = np.asarray(h5["rooms/room_type_id"][()], dtype=np.int32)
            room_areas = np.asarray(h5["rooms/floor_area_m2"][()], dtype=np.float32)

            global_room_id.append(np.arange(room_cursor, room_cursor + room_count, dtype=np.int64))
            room_scene_row.append(np.full(room_count, row, dtype=np.int32))
            local_room_index.append(local_rooms)
            room_id.extend(room_ids)
            room_type.extend(room_types)
            room_type_id.append(room_type_ids)
            room_floor_area_m2.append(room_areas)
            room_bounds_xy.append(np.asarray(h5["rooms/bounds_xy"][()], dtype=np.float32))
            room_ue_count.append(np.asarray(h5["rooms/ue_count"][()], dtype=np.int32))
            room_bs_count.append(np.asarray(h5["rooms/bs_count"][()], dtype=np.int32))

            target_room = np.asarray(h5["link/target_room_index"][()], dtype=np.int32)
            target_type = np.asarray(h5["link/target_room_type_id"][()], dtype=np.int32)
            target_area = np.asarray(h5["link/target_room_area_m2"][()], dtype=np.float32)
            global_link_id.append(np.arange(cursor, cursor + count, dtype=np.int64))
            scene_row.append(np.full(count, row, dtype=np.int32))
            local_link_index.append(np.asarray(h5["link_id"][()], dtype=np.int64))
            bs_id.append(np.asarray(h5["bs_id"][()], dtype=np.int64))
            ue_id.append(np.asarray(h5["ue_id"][()], dtype=np.int64))
            rx_pos.append(np.asarray(h5["rx_pos"][()], dtype=np.float32))
            tx_pos.append(np.asarray(h5["tx_pos"][()], dtype=np.float32))
            target_xy.append(np.asarray(h5["target_xy"][()], dtype=np.float32))
            csi_power_db.append(np.asarray(h5["csi_power_db"][()], dtype=np.float32))
            valid_mask.append(np.asarray(h5["valid_mask"][()], dtype=np.bool_))
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
        "schema_version": CFR_SCENE_SCHEMA_VERSION,
        "contract_name": "sionna_cfr_scene_link_index",
        "scene_ids": scene_ids,
        "scene_h5_paths": h5_paths,
        "global_link_id": torch.from_numpy(np.concatenate(global_link_id)),
        "scene_row": torch.from_numpy(np.concatenate(scene_row)),
        "local_link_index": torch.from_numpy(np.concatenate(local_link_index)),
        "bs_id": torch.from_numpy(np.concatenate(bs_id)),
        "ue_id": torch.from_numpy(np.concatenate(ue_id)),
        "rx_pos": torch.from_numpy(np.concatenate(rx_pos)),
        "tx_pos": torch.from_numpy(np.concatenate(tx_pos)),
        "target_xy": torch.from_numpy(np.concatenate(target_xy)),
        "csi_power_db": torch.from_numpy(np.concatenate(csi_power_db)),
        "valid_mask": torch.from_numpy(np.concatenate(valid_mask)),
        "ue_room_index": torch.from_numpy(np.concatenate(ue_room_index)),
        "bs_room_index": torch.from_numpy(np.concatenate(bs_room_index)),
        "target_room_index": torch.from_numpy(np.concatenate(target_room_index)),
        "target_room_type_id": torch.from_numpy(np.concatenate(target_room_type_id)),
        "target_room_area_m2": torch.from_numpy(np.concatenate(target_room_area_m2)),
    }
    link_pt_path = output_root / "link_index.pt"
    torch.save(payload, link_pt_path)

    room_payload = {
        "schema_version": CFR_SCENE_SCHEMA_VERSION,
        "contract_name": "sionna_cfr_scene_room_index",
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
        bs_id=np.concatenate(bs_id),
        ue_id=np.concatenate(ue_id),
        rx_pos=np.concatenate(rx_pos),
        tx_pos=np.concatenate(tx_pos),
        target_xy=np.concatenate(target_xy),
        csi_power_db=np.concatenate(csi_power_db),
        valid_mask=np.concatenate(valid_mask),
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
    bs_id: np.ndarray,
    ue_id: np.ndarray,
    rx_pos: np.ndarray,
    tx_pos: np.ndarray,
    target_xy: np.ndarray,
    csi_power_db: np.ndarray,
    valid_mask: np.ndarray,
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
            "bs_id": bs_id,
            "ue_id": ue_id,
            "rx_x_m": rx_pos[:, 0],
            "rx_y_m": rx_pos[:, 1],
            "rx_z_m": rx_pos[:, 2],
            "tx_x_m": tx_pos[:, 0],
            "tx_y_m": tx_pos[:, 1],
            "tx_z_m": tx_pos[:, 2],
            "target_x_m": target_xy[:, 0],
            "target_y_m": target_xy[:, 1],
            "csi_power_db": csi_power_db,
            "valid_mask": valid_mask,
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


def _strings_for_room_indices(indices: np.ndarray, values: list[str]) -> list[str]:
    return [values[int(index)] if 0 <= int(index) < len(values) else "" for index in indices]


def _decode_h5_string(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _write_dataset_manifest(
    *,
    output_root: Path,
    config: CFRSceneBuildConfig,
    scene_results: tuple[CFRSceneResult, ...],
    index_paths: dict[str, Path],
    elapsed_seconds: float,
) -> Path:
    manifest = {
        "schema_version": CFR_SCENE_SCHEMA_VERSION,
        "contract_name": "sionna_cfr_scene_dataset_manifest",
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
        "cfr_compression": config.cfr_compression,
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


def _scene_result_to_manifest(result: CFRSceneResult, output_root: Path) -> dict[str, Any]:
    return {
        "scene_id": result.scene_id,
        "status": result.status,
        "h5_path": _relative_or_absolute(result.h5_path, output_root) if result.h5_path else "",
        "num_links": result.num_links,
        "valid_link_count": result.valid_link_count,
        "cfr_shape": list(result.cfr_shape),
        "dtype": result.dtype,
        "source_shards": list(result.source_shards),
        "source_run_dir": result.source_run_dir.as_posix() if result.source_run_dir else "",
        "label_file": result.label_file.as_posix() if result.label_file else "",
        "scene_file": result.scene_file.as_posix() if result.scene_file else "",
        "floorplan_found": result.floorplan_found,
        "elapsed_seconds": result.elapsed_seconds,
        "source_bytes": result.source_bytes,
        "output_bytes": result.output_bytes,
        "power_min_db": result.power_min_db,
        "power_max_db": result.power_max_db,
        "power_quantiles_db": result.power_quantiles_db,
        "error": result.error,
    }


def _result_from_existing_scene(
    run: SourceSceneRun,
    output_path: Path,
    start: float,
) -> CFRSceneResult:
    validate_cfr_scene_h5(output_path)
    with h5py.File(output_path, "r") as h5:
        power = np.asarray(h5["csi_power_db"][()])
        valid = np.asarray(h5["valid_mask"][()], dtype=np.bool_)
        return CFRSceneResult(
            scene_id=run.scene_id,
            status="skipped",
            h5_path=output_path,
            num_links=int(h5["cfr"].shape[0]),
            valid_link_count=int(np.count_nonzero(valid)),
            cfr_shape=tuple(int(dim) for dim in h5["cfr"].shape),
            dtype=str(h5["cfr"].dtype),
            source_run_dir=run.run_dir,
            label_file=run.label_file,
            scene_file=run.scene_file,
            floorplan_found=bool(h5["floorplan/found"][()])
            if "floorplan/found" in h5
            else False,
            elapsed_seconds=time.perf_counter() - start,
            output_bytes=output_path.stat().st_size,
            power_min_db=float(np.min(power)) if power.size else None,
            power_max_db=float(np.max(power)) if power.size else None,
            power_quantiles_db=_power_quantiles(power),
        )


def _run_from_dir(run_dir: Path) -> SourceSceneRun:
    manifest_path = run_dir / "manifest" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    scene_id = str(manifest.get("scene_id") or run_dir.name)
    return SourceSceneRun(
        scene_id=scene_id,
        run_dir=run_dir,
        manifest_path=manifest_path,
        label_file=_optional_path(manifest.get("label_file")),
        scene_file=_optional_path(manifest.get("scene_file")),
    )


def _first_source_h5(manifest: dict[str, Any], run_dir: Path) -> Path:
    results = manifest.get("results", [])
    if not results:
        raise ValueError("manifest has no result entries")
    return _resolve_manifest_artifact(run_dir, Path(str(results[0]["result_h5"])))


def _resolve_manifest_artifact(run_dir: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    if path.exists():
        return path
    candidate = run_dir / path
    if candidate.exists():
        return candidate
    candidate = run_dir.parent / path
    if candidate.exists():
        return candidate
    return path


def _read_global_indices(
    h5: h5py.File,
    role: str,
    entry: dict[str, Any],
    expected: int,
) -> np.ndarray:
    h5_path = f"shard/global_{role}_indices"
    if h5_path in h5:
        values = np.asarray(h5[h5_path][()], dtype=np.int64)
    else:
        values = np.asarray(entry.get(f"global_{role}_indices", []), dtype=np.int64)
    if values.shape != (expected,):
        values = np.arange(expected, dtype=np.int64)
    return values


def _scene_metadata(
    run: SourceSceneRun,
    manifest: dict[str, Any],
    source_layout: _SourceLayout,
    assets: _SceneAssets,
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
            "rx_ant": source_layout.rx_ant,
            "tx_ant": source_layout.tx_ant,
            "subcarriers": source_layout.subcarriers,
            "tx_role": source_layout.tx_role,
            "rx_role": source_layout.rx_role,
        },
        "source_config": manifest.get("config_snapshot", {}),
        "postprocess": {
            "cfr_compression": config.cfr_compression,
            "metadata_compression": config.metadata_compression,
        },
    }


def _select_floorplan_png(scene_root: Path) -> Path | None:
    floorplan_dir = scene_root / "floorplan"
    preferred = floorplan_dir / "floorplan_1p60.png"
    if preferred.exists():
        return preferred
    if floorplan_dir.exists():
        candidates = sorted(floorplan_dir.glob("floorplan_*.png"))
        if candidates:
            return candidates[0]
        preview = floorplan_dir / "preview.png"
        if preview.exists():
            return preview
    return None


def _create_dataset(
    h5: h5py.File,
    path: str,
    *,
    shape: tuple[int, ...],
    dtype: Any,
    attrs: dict[str, Any] | None = None,
    **kwargs: Any,
) -> h5py.Dataset:
    parent = _ensure_parent(h5, path)
    dataset = parent.create_dataset(path.split("/")[-1], shape=shape, dtype=dtype, **kwargs)
    for key, value in (attrs or {}).items():
        dataset.attrs[key] = value
    return dataset


def _copy_dataset_if_present(
    src: h5py.File,
    dst: h5py.File,
    path: str,
    *,
    compression: str = "none",
) -> None:
    if path not in src:
        return
    parent = _ensure_parent(dst, path)
    data = src[path][()]
    kwargs = _compression_kwargs(compression, 1) if np.asarray(data).shape else {}
    dataset = parent.create_dataset(path.split("/")[-1], data=data, **kwargs)
    for key, value in src[path].attrs.items():
        dataset.attrs[key] = value


def _write_string(group: h5py.File | h5py.Group, name: str, value: str) -> None:
    group.create_dataset(name, data=str(value), dtype=UTF8_DTYPE)


def _write_scalar(group: h5py.File | h5py.Group, name: str, value: Any, dtype: Any) -> None:
    group.create_dataset(name, data=np.asarray(value, dtype=dtype))


def _ensure_parent(h5: h5py.File, path: str) -> h5py.Group:
    parent_path = "/".join(path.split("/")[:-1])
    if not parent_path:
        return h5
    return h5.require_group(parent_path)


def _compression_kwargs(compression: str, gzip_level: int) -> dict[str, Any]:
    if compression == "none":
        return {}
    if compression == "gzip":
        return {"compression": "gzip", "compression_opts": int(gzip_level), "shuffle": True}
    if compression == "lzf":
        return {"compression": "lzf", "shuffle": True}
    msg = f"Unsupported compression {compression!r}"
    raise ValueError(msg)


def _power_buckets(power_db: np.ndarray) -> np.ndarray:
    buckets = np.zeros(power_db.shape, dtype=np.int8)
    buckets[power_db > -110.0] = 1
    buckets[power_db > -95.0] = 2
    buckets[power_db > -80.0] = 3
    buckets[power_db > -65.0] = 4
    buckets[power_db > -50.0] = 5
    return buckets


def _power_quantiles(power: np.ndarray) -> dict[str, float]:
    if power.size == 0:
        return {}
    quantiles = np.percentile(power, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    keys = ("p0", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "p100")
    return {key: float(value) for key, value in zip(keys, quantiles, strict=True)}


def _relative_or_absolute(path: Path | None, root: Path) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _validate_build_config(config: CFRSceneBuildConfig) -> None:
    if config.workers < 1:
        raise ValueError("workers must be >= 1")
    if config.limit is not None and config.limit < 1:
        raise ValueError("limit must be positive when provided")
    if config.cfr_compression not in ("none", "gzip", "lzf"):
        raise ValueError("cfr_compression must be none/gzip/lzf")
    if config.metadata_compression not in ("none", "gzip", "lzf"):
        raise ValueError("metadata_compression must be none/gzip/lzf")
    if not (1 <= int(config.gzip_level) <= 9):
        raise ValueError("gzip_level must be in [1, 9]")


def _read_string(dataset: h5py.Dataset) -> str:
    value = dataset[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _optional_path(value: Any) -> Path | None:
    if value is None or value == "":
        return None
    return Path(str(value))


def _resolve_optional_existing(path: Path | None) -> Path | None:
    if path is None:
        return None
    if path.exists():
        return path
    resolved = Path(path.as_posix()).resolve()
    return resolved if resolved.exists() else path


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_optional_scalar(h5: h5py.File, path: str, default: Any = "") -> Any:
    if path not in h5:
        return default
    value = h5[path][()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value
