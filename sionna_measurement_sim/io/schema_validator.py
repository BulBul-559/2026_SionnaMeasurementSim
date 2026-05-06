"""Minimal HDF5 contract validator used by tests and readback."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


class SchemaValidationError(ValueError):
    """Raised when an HDF5 file violates the project data contract."""


REQUIRED_TRUTH_GROUPS = (
    "meta",
    "input",
    "topology",
    "devices",
    "antenna",
    "scene",
    "frequency",
    "channel/truth",
    "paths/samples",
    "runtime",
)

REQUIRED_TRUTH_DATASETS = (
    "meta/schema_version",
    "meta/contract_name",
    "meta/index_order",
    "meta/unit_convention",
    "meta/config_snapshot",
    "topology/tx_positions_m",
    "topology/rx_positions_m",
    "devices/tx_velocity_mps",
    "devices/rx_velocity_mps",
    "devices/tx_orientation_rad",
    "devices/rx_orientation_rad",
    "antenna/tx_polarization",
    "antenna/rx_polarization",
    "frequency/frequencies_hz",
    "channel/truth/cfr",
    "paths/samples/vertices_m",
    "paths/samples/interaction_type",
    "paths/samples/object_id",
    "paths/samples/primitive_id",
    "paths/samples/doppler_hz",
    "paths/samples/tau_s",
)

PATH_SAMPLE_DATASETS = (
    "paths/samples/sampled_link_indices",
    "paths/samples/sampled_path_indices",
    "paths/samples/path_gain_db",
    "paths/samples/path_type",
    "paths/samples/vertices_m",
    "paths/samples/interaction_type",
    "paths/samples/object_id",
    "paths/samples/primitive_id",
    "paths/samples/doppler_hz",
    "paths/samples/tau_s",
)


def validate_hdf5_contract(path: str | Path) -> None:
    """Validate the minimal truth HDF5 contract."""

    with h5py.File(path, "r") as h5:
        _require_absent(h5, "channel/cfr")
        _require_present(h5, ("meta/schema_version",), kind=h5py.Dataset)
        _require_present(h5, REQUIRED_TRUTH_GROUPS, kind=h5py.Group)
        _require_present(h5, REQUIRED_TRUTH_DATASETS, kind=h5py.Dataset)
        _require_present(h5, PATH_SAMPLE_DATASETS, kind=h5py.Dataset)
        _validate_truth_shapes(h5)
        _validate_path_sample_shapes(h5)
        _validate_units(h5)
        _validate_values(h5)


def _require_absent(h5: h5py.File, dataset_path: str) -> None:
    if dataset_path in h5:
        msg = f"Forbidden dataset exists: /{dataset_path}"
        raise SchemaValidationError(msg)


def _require_present(h5: h5py.File, paths: tuple[str, ...], *, kind: type) -> None:
    for required_path in paths:
        if required_path not in h5:
            msg = f"Missing required path: /{required_path}"
            raise SchemaValidationError(msg)
        if not isinstance(h5[required_path], kind):
            msg = f"Required path has wrong HDF5 object type: /{required_path}"
            raise SchemaValidationError(msg)


def _validate_truth_shapes(h5: h5py.File) -> None:
    cfr = h5["channel/truth/cfr"]
    tx_positions = h5["topology/tx_positions_m"]
    rx_positions = h5["topology/rx_positions_m"]
    frequencies = h5["frequency/frequencies_hz"]

    if cfr.ndim != 5:
        msg = f"/channel/truth/cfr must be rank 5, got {cfr.shape}"
        raise SchemaValidationError(msg)
    if cfr.dtype.kind != "c":
        msg = "/channel/truth/cfr must be a complex dtype"
        raise SchemaValidationError(msg)
    if cfr.shape[0] != tx_positions.shape[0] or cfr.shape[1] != rx_positions.shape[0]:
        msg = "/channel/truth/cfr tx/rx dimensions must match topology"
        raise SchemaValidationError(msg)
    if cfr.shape[-1] != frequencies.shape[-1]:
        msg = "frequencies_hz length must match truth cfr subcarrier dimension"
        raise SchemaValidationError(msg)

    if "observation/cfr_est" in h5:
        cfr_est = h5["observation/cfr_est"]
        if cfr_est.ndim != 6:
            msg = f"/observation/cfr_est must be rank 6, got {cfr_est.shape}"
            raise SchemaValidationError(msg)
        if cfr_est.shape[1:] != cfr.shape:
            msg = "/observation/cfr_est shape[1:] must match /channel/truth/cfr"
            raise SchemaValidationError(msg)


def _validate_path_sample_shapes(h5: h5py.File) -> None:
    sampled_links = h5["paths/samples/sampled_link_indices"]
    vertices = h5["paths/samples/vertices_m"]
    interactions = h5["paths/samples/interaction_type"]
    object_id = h5["paths/samples/object_id"]
    primitive_id = h5["paths/samples/primitive_id"]
    doppler = h5["paths/samples/doppler_hz"]
    tau = h5["paths/samples/tau_s"]

    if sampled_links.ndim != 2 or sampled_links.shape[-1] != 2:
        msg = "/paths/samples/sampled_link_indices must have shape [sample, 2]"
        raise SchemaValidationError(msg)
    if vertices.ndim != 4 or vertices.shape[-1] != 3:
        msg = "/paths/samples/vertices_m must have shape [sample, sample_path, max_vertices, 3]"
        raise SchemaValidationError(msg)
    if interactions.ndim != 3:
        msg = "/paths/samples/interaction_type must have rank 3"
        raise SchemaValidationError(msg)
    for name, dataset in (
        ("object_id", object_id),
        ("primitive_id", primitive_id),
    ):
        if dataset.shape != interactions.shape:
            msg = f"/paths/samples/{name} shape must match interaction_type"
            raise SchemaValidationError(msg)
    expected_scalar_shape = vertices.shape[:2]
    if doppler.shape != expected_scalar_shape or tau.shape != expected_scalar_shape:
        msg = "path doppler_hz and tau_s must match [sample, sample_path]"
        raise SchemaValidationError(msg)


def _validate_units(h5: h5py.File) -> None:
    for dataset_path in (
        "topology/tx_positions_m",
        "topology/rx_positions_m",
        "frequency/frequencies_hz",
        "channel/truth/cfr",
        "paths/samples/vertices_m",
        "paths/samples/doppler_hz",
        "paths/samples/tau_s",
    ):
        if "unit" not in h5[dataset_path].attrs:
            msg = f"Missing unit attribute on /{dataset_path}"
            raise SchemaValidationError(msg)


def _validate_values(h5: h5py.File) -> None:
    cfr = h5["channel/truth/cfr"][()]
    frequencies = h5["frequency/frequencies_hz"][()]
    if not np.any(np.isfinite(cfr)):
        msg = "/channel/truth/cfr must contain at least one finite value"
        raise SchemaValidationError(msg)
    if not np.all(np.isfinite(frequencies)) or np.any(np.diff(frequencies) <= 0):
        msg = "/frequency/frequencies_hz must be finite and strictly increasing"
        raise SchemaValidationError(msg)

    for dataset_path in (
        "paths/samples/vertices_m",
        "paths/samples/doppler_hz",
        "paths/samples/tau_s",
    ):
        values = h5[dataset_path][()]
        if values.size and not np.all(np.isfinite(values)):
            msg = f"/{dataset_path} must contain finite values"
            raise SchemaValidationError(msg)
