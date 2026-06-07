"""HDF5 reader helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from sionna_measurement_sim.io.schema_validator import validate_hdf5_contract


def read_dataset(path: str | Path, dataset_path: str) -> Any:
    """Read a dataset value from an HDF5 file."""

    with h5py.File(path, "r") as h5:
        value = h5[dataset_path][()]
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
