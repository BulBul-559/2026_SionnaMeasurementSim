"""HDF5 reader helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py

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


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value
