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
    "link",
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
    "channel/truth/geometric_path_count",
    "channel/truth/los_exists",
    "channel/truth/nlos_exists",
    "paths/samples/vertices_m",
    "paths/samples/interaction_type",
    "paths/samples/object_id",
    "paths/samples/primitive_id",
    "paths/samples/doppler_hz",
    "paths/samples/tau_s",
)

PATH_SAMPLE_DATASETS = (
    "paths/samples/sampled_link_indices",
    "paths/samples/sampled_rx_ant_indices",
    "paths/samples/sampled_tx_ant_indices",
    "paths/samples/sampled_path_indices",
    "paths/samples/path_count",
    "paths/samples/path_gain_db",
    "paths/samples/path_type",
    "paths/samples/vertices_m",
    "paths/samples/vertex_count",
    "paths/samples/interaction_type",
    "paths/samples/object_id",
    "paths/samples/primitive_id",
    "paths/samples/doppler_hz",
    "paths/samples/tau_s",
    "link/duplex_mode",
    "link/phy_link_direction",
    "link/rt_trace_direction",
    "link/reciprocity_mode",
    "link/reciprocity_applied",
)

REQUIRED_OBSERVATION_GROUPS = (
    "waveform",
    "observation",
    "impairments",
    "receiver",
    "evaluation",
)

REQUIRED_CALIBRATION_DATASETS = (
    "calibration/profile_id",
    "calibration/fitted_parameters",
    "calibration/validation_metrics",
)

REQUIRED_OBSERVATION_DATASETS = (
    "waveform/standard",
    "waveform/fft_size",
    "waveform/pilot_indices",
    "waveform/pilot_symbols",
    "receiver/estimator_type",
    "observation/cfr_est",
    "observation/valid_mask",
    "observation/detection_success",
    "observation/estimation_success",
    "observation/snr_db",
    "observation/cfo_hz",
    "observation/sfo_ppm",
    "observation/timing_offset_samples",
    "observation/phase_offset_rad",
    "observation/agc_gain_db",
    "observation/clipping_flag",
    "impairments/model_version",
    "impairments/random_seed",
    "evaluation/nmse_db",
    "evaluation/nmse_db_total",
    "evaluation/ber",
    "evaluation/bler",
    "evaluation/num_bit_errors",
    "evaluation/num_blocks",
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
        if _observation_enabled(h5):
            _validate_observation_cfr_est_shape_if_present(h5)
            _require_present(h5, REQUIRED_OBSERVATION_GROUPS, kind=h5py.Group)
            _require_present(h5, REQUIRED_OBSERVATION_DATASETS, kind=h5py.Dataset)
            _validate_observation_shapes(h5)
            if "calibration" in h5:
                _require_present(h5, REQUIRED_CALIBRATION_DATASETS, kind=h5py.Dataset)
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
    if "channel/truth/cfr_snapshots" in h5:
        ss = h5["channel/truth/cfr_snapshots"]
        if ss.ndim != 6:
            msg = f"/channel/truth/cfr_snapshots must be rank 6, got {ss.shape}"
            raise SchemaValidationError(msg)
        if ss.shape[1:] != cfr.shape:
            msg = "cfr_snapshots.shape[1:] must match truth cfr shape"
            raise SchemaValidationError(msg)

    _validate_cir_shapes(h5, cfr.shape[0], cfr.shape[1])


def _validate_cir_shapes(
    h5: h5py.File, num_tx: int, num_rx: int,
) -> None:
    """Validate CIR dataset shapes and consistency with topology."""
    cir_paths = {
        "cir_coefficients": "channel/truth/cir_coefficients",
        "cir_delays_s": "channel/truth/cir_delays_s",
        "cir_valid": "channel/truth/cir_valid",
    }
    present = [p for p in cir_paths.values() if p in h5]
    if not present:
        return  # CIR all absent — OK for backward compat
    if len(present) != 3:
        missing = [n for n, p in cir_paths.items() if p not in h5]
        msg = f"CIR datasets partially present; missing: {missing}"
        raise SchemaValidationError(msg)
    for _name, path in cir_paths.items():
        ds = h5[path]
        if ds.ndim != 6:
            msg = f"/{path} must be rank 6, got {ds.shape}"
            raise SchemaValidationError(msg)
        if ds.shape[1] != num_tx or ds.shape[2] != num_rx:
            msg = f"/{path} tx/rx dimensions must match topology"
            raise SchemaValidationError(msg)
    cir_coeff = h5["channel/truth/cir_coefficients"]
    cir_delays = h5["channel/truth/cir_delays_s"]
    cir_valid = h5["channel/truth/cir_valid"]
    if cir_coeff.shape != cir_delays.shape or cir_coeff.shape != cir_valid.shape:
        msg = "CIR dataset shapes must all match"
        raise SchemaValidationError(msg)
    if cir_coeff.dtype.kind != "c":
        msg = "/channel/truth/cir_coefficients must be a complex dtype"
        raise SchemaValidationError(msg)
    if cir_delays.dtype.kind != "f":
        msg = "/channel/truth/cir_delays_s must be a float dtype"
        raise SchemaValidationError(msg)


def _validate_path_sample_shapes(h5: h5py.File) -> None:
    sampled_links = h5["paths/samples/sampled_link_indices"]
    vertices = h5["paths/samples/vertices_m"]
    vertex_count = h5["paths/samples/vertex_count"]
    interactions = h5["paths/samples/interaction_type"]
    object_id = h5["paths/samples/object_id"]
    primitive_id = h5["paths/samples/primitive_id"]

    # Antenna index field shape validation
    for name in ("sampled_rx_ant_indices", "sampled_tx_ant_indices"):
        if name in h5["paths/samples"]:
            arr = h5[f"paths/samples/{name}"]
            if arr.ndim != 1 or arr.shape[0] != sampled_links.shape[0]:
                msg = f"/paths/samples/{name} must be 1D [sample]"
                raise SchemaValidationError(msg)
    doppler = h5["paths/samples/doppler_hz"]
    tau = h5["paths/samples/tau_s"]

    if sampled_links.ndim != 2 or sampled_links.shape[-1] != 2:
        msg = "/paths/samples/sampled_link_indices must have shape [sample, 2]"
        raise SchemaValidationError(msg)
    if vertices.ndim != 4 or vertices.shape[-1] != 3:
        msg = "/paths/samples/vertices_m must have shape [sample, sample_path, max_vertices, 3]"
        raise SchemaValidationError(msg)
    if vertex_count.shape != vertices.shape[:2]:
        msg = "/paths/samples/vertex_count must match [sample, sample_path]"
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
    if interactions.shape[:2] != expected_scalar_shape:
        msg = "path interaction_type must match [sample, sample_path, max_depth]"
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
    if _observation_enabled(h5):
        for dataset_path in (
            "waveform/pilot_symbols",
            "observation/cfr_est",
            "observation/snr_db",
            "evaluation/nmse_db",
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

    if "paths/samples/vertex_count" in h5:
        vertex_count = h5["paths/samples/vertex_count"][()]
        interactions = h5["paths/samples/interaction_type"][()]
        interaction_count = np.count_nonzero(interactions != 0, axis=-1)
        active = vertex_count > 0
        if np.any(vertex_count[active] < interaction_count[active] + 2):
            msg = "sample path vertex_count must include TX/RX endpoints"
            raise SchemaValidationError(msg)
    if _observation_enabled(h5):
        for dataset_path in ("observation/cfr_est", "observation/snr_db", "evaluation/nmse_db"):
            values = h5[dataset_path][()]
            if not np.all(np.isfinite(values)):
                msg = f"/{dataset_path} must contain finite values"
                raise SchemaValidationError(msg)


def _observation_enabled(h5: h5py.File) -> bool:
    if "observation/cfr_est" in h5:
        return True
    if "meta/observation_branch_enabled" in h5:
        return bool(h5["meta/observation_branch_enabled"][()])
    return False


def _validate_observation_shapes(h5: h5py.File) -> None:
    cfr_est = h5["observation/cfr_est"]
    valid_mask = h5["observation/valid_mask"]
    detection_success = h5["observation/detection_success"]
    estimation_success = h5["observation/estimation_success"]
    snr_db = h5["observation/snr_db"]
    nmse_db = h5["evaluation/nmse_db"]

    _validate_observation_cfr_est_shape_if_present(h5)
    link_shape = cfr_est.shape[:3]
    for name, dataset in (
        ("valid_mask", valid_mask),
        ("detection_success", detection_success),
        ("estimation_success", estimation_success),
        ("snr_db", snr_db),
        ("nmse_db", nmse_db),
    ):
        if dataset.shape != link_shape:
            msg = f"/observation or /evaluation {name} must match [snapshot, tx, rx]"
            raise SchemaValidationError(msg)


def _validate_observation_cfr_est_shape_if_present(h5: h5py.File) -> None:
    if "observation/cfr_est" not in h5:
        return
    truth_cfr = h5["channel/truth/cfr"]
    cfr_est = h5["observation/cfr_est"]
    if cfr_est.ndim != 6:
        msg = f"/observation/cfr_est must be rank 6, got {cfr_est.shape}"
        raise SchemaValidationError(msg)
    if cfr_est.shape[-5:] != truth_cfr.shape[-5:]:
        msg = "/observation/cfr_est shape[-5:] must match /channel/truth/cfr"
        raise SchemaValidationError(msg)
