"""Minimal HDF5 contract validator used by tests and readback."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from sionna_measurement_sim.domain.constants import (
    FULL_CONTRACT_NAME,
    RT_LABELS_CONTRACT_NAME,
)


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
    "derived",
    "channel/truth",
    "paths/samples",
    "paths/nlos_truth",
    "runtime",
    "link",
)

REQUIRED_TRUTH_DATASETS = (
    "meta/schema_version",
    "meta/contract_name",
    "meta/index_order",
    "meta/unit_convention",
    "meta/output_profile",
    "meta/config_snapshot",
    "topology/tx_positions_m",
    "topology/rx_positions_m",
    "devices/tx_velocity_mps",
    "devices/rx_velocity_mps",
    "devices/tx_orientation_rad",
    "devices/rx_orientation_rad",
    "antenna/tx_polarization",
    "antenna/rx_polarization",
    "scene/scene_id",
    "scene/map_id",
    "frequency/frequencies_hz",
    "derived/geometric_distance_m",
    "derived/los_distance_m",
    "derived/first_path_delay_s",
    "derived/first_path_propagation_range_m",
    "derived/strongest_path_delay_s",
    "derived/los_aoa_azimuth_rad",
    "derived/los_aoa_zenith_rad",
    "derived/strongest_aoa_azimuth_rad",
    "derived/strongest_aoa_zenith_rad",
    "derived/first_path_aoa_azimuth_rad",
    "derived/first_path_aoa_zenith_rad",
    "derived/los_flag",
    "derived/nlos_flag",
    "derived/path_count",
    "derived/path_power_db",
    "derived/link_valid_mask",
    "derived/tx_rx_midpoint_m",
    "derived/tx_rx_bearing_rad",
    "derived/tx_rx_distance_m",
    "derived/path_selection_policy",
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
    "paths/nlos_truth/valid",
    "paths/nlos_truth/aoa_zenith_rad",
    "paths/nlos_truth/aoa_azimuth_rad",
    "paths/nlos_truth/aod_zenith_rad",
    "paths/nlos_truth/aod_azimuth_rad",
    "paths/nlos_truth/path_power_db",
    "paths/nlos_truth/delay_s",
    "paths/nlos_truth/path_depth",
    "paths/nlos_truth/path_type",
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
    "link/tx_role",
    "link/rx_role",
)

OPTIONAL_SHARD_DATASETS = (
    "shard/shard_index",
    "shard/shard_count",
    "shard/axis",
    "shard/global_rx_start",
    "shard/global_rx_indices",
    "shard/global_tx_indices",
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
    "evaluation/num_bits",
    "evaluation/num_block_errors",
    "evaluation/num_blocks",
)

PDP_PEAK_RANGING_DATASETS = (
    "ranging/pdp_peak/toa_est_s",
    "ranging/pdp_peak/one_way_range_est_m",
    "ranging/pdp_peak/rtt_equiv_s",
    "ranging/pdp_peak/range_error_m",
    "ranging/pdp_peak/detection_success",
    "ranging/pdp_peak/selected_delay_bin",
    "ranging/pdp_peak/peak_power_linear",
    "ranging/pdp_peak/peak_snr_db",
)

PHASE_SLOPE_RANGING_DATASETS = (
    "ranging/phase_slope/toa_est_s",
    "ranging/phase_slope/one_way_range_est_m",
    "ranging/phase_slope/rtt_equiv_s",
    "ranging/phase_slope/range_error_m",
    "ranging/phase_slope/detection_success",
    "ranging/phase_slope/fit_residual_rad",
)

RT_LABELS_REQUIRED_GROUPS = (
    "meta",
    "input",
    "topology",
    "devices",
    "antenna",
    "scene",
    "frequency",
    "derived",
    "labels/link",
    "runtime",
    "link",
)

RT_LABELS_REQUIRED_DATASETS = (
    "meta/schema_version",
    "meta/contract_name",
    "meta/index_order",
    "meta/unit_convention",
    "meta/output_profile",
    "meta/config_snapshot",
    "topology/tx_positions_m",
    "topology/rx_positions_m",
    "devices/tx_velocity_mps",
    "devices/rx_velocity_mps",
    "devices/tx_orientation_rad",
    "devices/rx_orientation_rad",
    "antenna/tx_polarization",
    "antenna/rx_polarization",
    "scene/scene_id",
    "scene/map_id",
    "frequency/frequencies_hz",
    "derived/geometric_distance_m",
    "derived/los_distance_m",
    "derived/first_path_delay_s",
    "derived/first_path_propagation_range_m",
    "derived/strongest_path_delay_s",
    "derived/los_aoa_azimuth_rad",
    "derived/los_aoa_zenith_rad",
    "derived/strongest_aoa_azimuth_rad",
    "derived/strongest_aoa_zenith_rad",
    "derived/first_path_aoa_azimuth_rad",
    "derived/first_path_aoa_zenith_rad",
    "derived/los_flag",
    "derived/nlos_flag",
    "derived/path_count",
    "derived/path_power_db",
    "derived/link_valid_mask",
    "derived/tx_rx_midpoint_m",
    "derived/tx_rx_bearing_rad",
    "derived/tx_rx_distance_m",
    "derived/path_selection_policy",
    "labels/link/link_index",
    "labels/link/tx_index",
    "labels/link/rx_index",
    "labels/link/global_tx_index",
    "labels/link/global_rx_index",
    "labels/link/tx_xy_m",
    "labels/link/rx_xy_m",
    "labels/link/link_valid_mask",
    "labels/link/geometric_distance_m",
    "labels/link/first_path_delay_s",
    "labels/link/first_path_propagation_range_m",
    "labels/link/strongest_path_delay_s",
    "labels/link/path_power_db",
    "labels/link/los_flag",
    "labels/link/nlos_flag",
    "labels/link/path_count",
    "labels/link/first_path_aoa_azimuth_rad",
    "labels/link/first_path_aoa_zenith_rad",
    "labels/link/tx_rx_bearing_rad",
    "labels/link/tx_rx_distance_m",
    "link/duplex_mode",
    "link/phy_link_direction",
    "link/tx_role",
    "link/rx_role",
)


def validate_hdf5_contract(path: str | Path) -> None:
    """Validate the minimal truth HDF5 contract."""

    with h5py.File(path, "r") as h5:
        _require_absent(h5, "channel/cfr")
        _require_absent(h5, "derived/rtt_like_m")
        _require_absent(h5, "derived/rtt_like_s")
        _require_absent(h5, "array/spatial_spectrum_label")
        _require_absent(h5, "array/spatial_spectrum_srs")
        _require_present(h5, ("meta/schema_version",), kind=h5py.Dataset)
        _require_present(h5, ("meta/contract_name",), kind=h5py.Dataset)
        contract_name = _read_string(h5["meta/contract_name"])
        if contract_name == RT_LABELS_CONTRACT_NAME:
            _validate_rt_labels_contract(h5)
            return
        if contract_name != FULL_CONTRACT_NAME:
            msg = f"Unsupported HDF5 contract_name: {contract_name!r}"
            raise SchemaValidationError(msg)
        _require_present(h5, REQUIRED_TRUTH_GROUPS, kind=h5py.Group)
        _require_present(h5, REQUIRED_TRUTH_DATASETS, kind=h5py.Dataset)
        _require_present(h5, PATH_SAMPLE_DATASETS, kind=h5py.Dataset)
        _validate_truth_shapes(h5)
        _validate_derived_shapes(h5)
        _validate_path_sample_shapes(h5)
        _validate_nlos_truth_shapes(h5)
        _validate_shard_if_present(h5)
        _validate_array_outputs_if_present(h5)
        if _observation_enabled(h5):
            _validate_observation_cfr_est_shape_if_present(h5)
            _require_present(h5, REQUIRED_OBSERVATION_GROUPS, kind=h5py.Group)
            _require_present(h5, REQUIRED_OBSERVATION_DATASETS, kind=h5py.Dataset)
            _validate_observation_shapes(h5)
            _validate_bler_contract(h5)
            _validate_nr_pusch_fields_if_applicable(h5)
            _validate_nr_srs_fields_if_applicable(h5)
            if "calibration" in h5:
                _require_present(h5, REQUIRED_CALIBRATION_DATASETS, kind=h5py.Dataset)
        if "ranging" in h5:
            _validate_ranging(h5)
        if "iq" in h5:
            _validate_iq(h5)
        if "multiuser" in h5:
            _validate_multiuser_srs(h5)
        _validate_units(h5)
        _validate_values(h5)


def _read_string(dataset: h5py.Dataset) -> str:
    value = dataset[()]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _validate_rt_labels_contract(h5: h5py.File) -> None:
    _require_absent(h5, "channel")
    _require_absent(h5, "paths")
    _require_absent(h5, "waveform")
    _require_absent(h5, "observation")
    _require_absent(h5, "array")
    _require_absent(h5, "ranging")
    _require_present(h5, RT_LABELS_REQUIRED_GROUPS, kind=h5py.Group)
    _require_present(h5, RT_LABELS_REQUIRED_DATASETS, kind=h5py.Dataset)
    if _read_string(h5["meta/output_profile"]) != "rt_labels_only":
        msg = "/meta/output_profile must be rt_labels_only for RT labels contract"
        raise SchemaValidationError(msg)
    _validate_derived_shapes(h5)
    _validate_shard_if_present(h5)
    _validate_rt_link_labels_shapes(h5)
    _validate_rt_labels_units(h5)
    _validate_rt_labels_values(h5)


def _validate_rt_link_labels_shapes(h5: h5py.File) -> None:
    tx = h5["topology/tx_positions_m"].shape[0]
    rx = h5["topology/rx_positions_m"].shape[0]
    link_count = tx * rx
    group = h5["labels/link"]
    for name in (
        "link_index",
        "tx_index",
        "rx_index",
        "global_tx_index",
        "global_rx_index",
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
    ):
        if group[name].shape != (link_count,):
            msg = f"/labels/link/{name} must have shape [link]"
            raise SchemaValidationError(msg)
        if "index_order" not in group[name].attrs:
            msg = f"Missing index_order attribute on /labels/link/{name}"
            raise SchemaValidationError(msg)
    for name in ("tx_xy_m", "rx_xy_m"):
        if group[name].shape != (link_count, 2):
            msg = f"/labels/link/{name} must have shape [link,2]"
            raise SchemaValidationError(msg)
        if "index_order" not in group[name].attrs:
            msg = f"Missing index_order attribute on /labels/link/{name}"
            raise SchemaValidationError(msg)


def _validate_rt_labels_units(h5: h5py.File) -> None:
    for dataset_path in (
        "topology/tx_positions_m",
        "topology/rx_positions_m",
        "frequency/frequencies_hz",
        "derived/geometric_distance_m",
        "derived/first_path_delay_s",
        "derived/first_path_propagation_range_m",
        "derived/tx_rx_midpoint_m",
        "derived/tx_rx_bearing_rad",
        "derived/tx_rx_distance_m",
        "labels/link/tx_xy_m",
        "labels/link/rx_xy_m",
        "labels/link/geometric_distance_m",
        "labels/link/first_path_delay_s",
        "labels/link/first_path_propagation_range_m",
        "labels/link/strongest_path_delay_s",
        "labels/link/path_power_db",
        "labels/link/first_path_aoa_azimuth_rad",
        "labels/link/first_path_aoa_zenith_rad",
        "labels/link/tx_rx_bearing_rad",
        "labels/link/tx_rx_distance_m",
    ):
        if "unit" not in h5[dataset_path].attrs:
            msg = f"Missing unit attribute on /{dataset_path}"
            raise SchemaValidationError(msg)


def _validate_rt_labels_values(h5: h5py.File) -> None:
    frequencies = h5["frequency/frequencies_hz"][()]
    if not np.all(np.isfinite(frequencies)) or np.any(np.diff(frequencies) <= 0):
        msg = "/frequency/frequencies_hz must be finite and strictly increasing"
        raise SchemaValidationError(msg)
    group = h5["labels/link"]
    valid = group["link_valid_mask"][()].astype(np.bool_)
    for name in ("geometric_distance_m", "tx_rx_distance_m", "path_power_db"):
        values = group[name][()]
        if not np.all(np.isfinite(values)):
            msg = f"/labels/link/{name} must contain finite values"
            raise SchemaValidationError(msg)
    for name in (
        "first_path_delay_s",
        "first_path_propagation_range_m",
        "strongest_path_delay_s",
    ):
        values = group[name][()]
        if not np.all(np.isfinite(values[valid])):
            msg = f"/labels/link/{name} must be finite for valid links"
            raise SchemaValidationError(msg)


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


def _validate_derived_shapes(h5: h5py.File) -> None:
    tx = h5["topology/tx_positions_m"].shape[0]
    rx = h5["topology/rx_positions_m"].shape[0]
    link_shape = (tx, rx)
    for dataset_path in (
        "derived/geometric_distance_m",
        "derived/los_distance_m",
        "derived/first_path_delay_s",
        "derived/first_path_propagation_range_m",
        "derived/strongest_path_delay_s",
        "derived/los_aoa_azimuth_rad",
        "derived/los_aoa_zenith_rad",
        "derived/strongest_aoa_azimuth_rad",
        "derived/strongest_aoa_zenith_rad",
        "derived/first_path_aoa_azimuth_rad",
        "derived/first_path_aoa_zenith_rad",
        "derived/los_flag",
        "derived/nlos_flag",
        "derived/path_count",
        "derived/path_power_db",
        "derived/link_valid_mask",
        "derived/tx_rx_bearing_rad",
        "derived/tx_rx_distance_m",
    ):
        if h5[dataset_path].shape != link_shape:
            msg = f"/{dataset_path} must match [tx, rx]"
            raise SchemaValidationError(msg)
    if h5["derived/tx_rx_midpoint_m"].shape != (*link_shape, 2):
        msg = "/derived/tx_rx_midpoint_m must match [tx, rx, 2]"
        raise SchemaValidationError(msg)


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


def _validate_nlos_truth_shapes(h5: h5py.File) -> None:
    truth_cfr = h5["channel/truth/cfr"]
    expected_prefix = truth_cfr.shape[:4]
    valid = h5["paths/nlos_truth/valid"]
    if valid.ndim != 5:
        msg = f"/paths/nlos_truth/valid must be rank 5, got {valid.shape}"
        raise SchemaValidationError(msg)
    if valid.shape[:4] != expected_prefix:
        msg = "/paths/nlos_truth dimensions must match [tx,rx,rx_ant,tx_ant,path]"
        raise SchemaValidationError(msg)
    for dataset_path in (
        "paths/nlos_truth/aoa_zenith_rad",
        "paths/nlos_truth/aoa_azimuth_rad",
        "paths/nlos_truth/aod_zenith_rad",
        "paths/nlos_truth/aod_azimuth_rad",
        "paths/nlos_truth/path_power_db",
        "paths/nlos_truth/delay_s",
        "paths/nlos_truth/path_depth",
        "paths/nlos_truth/path_type",
    ):
        if h5[dataset_path].shape != valid.shape:
            msg = f"/{dataset_path} must match /paths/nlos_truth/valid shape"
            raise SchemaValidationError(msg)


def _validate_shard_if_present(h5: h5py.File) -> None:
    if "shard" not in h5:
        return
    _require_present(h5, ("shard",), kind=h5py.Group)
    _require_present(h5, OPTIONAL_SHARD_DATASETS, kind=h5py.Dataset)

    shard_index = int(h5["shard/shard_index"][()])
    shard_count = int(h5["shard/shard_count"][()])
    global_rx_start = int(h5["shard/global_rx_start"][()])
    axis = h5["shard/axis"][()]
    if isinstance(axis, bytes):
        axis = axis.decode("utf-8")

    if shard_count < 1:
        msg = "/shard/shard_count must be positive"
        raise SchemaValidationError(msg)
    if shard_index < 0 or shard_index >= shard_count:
        msg = "/shard/shard_index must be in [0, shard_count)"
        raise SchemaValidationError(msg)
    if not axis:
        msg = "/shard/axis must not be empty"
        raise SchemaValidationError(msg)
    if global_rx_start < 0:
        msg = "/shard/global_rx_start must be non-negative"
        raise SchemaValidationError(msg)

    rx_indices = h5["shard/global_rx_indices"]
    tx_indices = h5["shard/global_tx_indices"]
    if rx_indices.ndim != 1 or tx_indices.ndim != 1:
        msg = "/shard global index datasets must be 1D"
        raise SchemaValidationError(msg)
    if rx_indices.dtype.kind not in ("i", "u") or tx_indices.dtype.kind not in ("i", "u"):
        msg = "/shard global index datasets must have integer dtype"
        raise SchemaValidationError(msg)
    if rx_indices.shape[0] != h5["topology/rx_positions_m"].shape[0]:
        msg = "/shard/global_rx_indices length must match local RX topology"
        raise SchemaValidationError(msg)
    if tx_indices.shape[0] != h5["topology/tx_positions_m"].shape[0]:
        msg = "/shard/global_tx_indices length must match local TX topology"
        raise SchemaValidationError(msg)
    if np.any(rx_indices[()] < 0) or np.any(tx_indices[()] < 0):
        msg = "/shard global index datasets must be non-negative"
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
        "paths/nlos_truth/aoa_zenith_rad",
        "paths/nlos_truth/aoa_azimuth_rad",
        "paths/nlos_truth/aod_zenith_rad",
        "paths/nlos_truth/aod_azimuth_rad",
        "paths/nlos_truth/path_power_db",
        "paths/nlos_truth/delay_s",
        "derived/geometric_distance_m",
        "derived/first_path_delay_s",
        "derived/first_path_propagation_range_m",
        "derived/tx_rx_midpoint_m",
        "derived/tx_rx_bearing_rad",
        "derived/tx_rx_distance_m",
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
    if "ranging" in h5:
        for dataset_path in (
            "ranging/pdp_peak/toa_est_s",
            "ranging/pdp_peak/one_way_range_est_m",
            "ranging/pdp_peak/rtt_equiv_s",
            "ranging/pdp_peak/range_error_m",
            "ranging/pdp_peak/peak_power_linear",
            "ranging/pdp_peak/peak_snr_db",
            "ranging/phase_slope/toa_est_s",
            "ranging/phase_slope/one_way_range_est_m",
            "ranging/phase_slope/rtt_equiv_s",
            "ranging/phase_slope/range_error_m",
            "ranging/phase_slope/fit_residual_rad",
        ):
            if dataset_path in h5 and "unit" not in h5[dataset_path].attrs:
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


def _validate_ranging(h5: h5py.File) -> None:
    if "observation/cfr_est" not in h5:
        msg = "/ranging requires /observation/cfr_est"
        raise SchemaValidationError(msg)
    _require_present(h5, ("ranging/default_estimator",), kind=h5py.Dataset)
    default_estimator = h5["ranging/default_estimator"][()]
    if isinstance(default_estimator, bytes):
        default_estimator = default_estimator.decode("utf-8")
    if default_estimator not in ("pdp_peak", "phase_slope"):
        msg = "/ranging/default_estimator must be pdp_peak or phase_slope"
        raise SchemaValidationError(msg)
    if default_estimator not in h5["ranging"]:
        msg = f"/ranging/{default_estimator} is required by default_estimator"
        raise SchemaValidationError(msg)
    if "pdp_peak" in h5["ranging"]:
        _require_present(h5, PDP_PEAK_RANGING_DATASETS, kind=h5py.Dataset)
        _validate_ranging_estimator_common(
            h5,
            "ranging/pdp_peak",
            extra=("selected_delay_bin", "peak_power_linear", "peak_snr_db"),
        )
    if "phase_slope" in h5["ranging"]:
        _require_present(h5, PHASE_SLOPE_RANGING_DATASETS, kind=h5py.Dataset)
        _validate_ranging_estimator_common(
            h5,
            "ranging/phase_slope",
            extra=("fit_residual_rad",),
        )


def _validate_ranging_estimator_common(
    h5: h5py.File,
    group_path: str,
    *,
    extra: tuple[str, ...],
) -> None:
    link_shape = h5["observation/cfr_est"].shape[:3]
    group = h5[group_path]
    for name in (
        "toa_est_s",
        "one_way_range_est_m",
        "rtt_equiv_s",
        "range_error_m",
        "detection_success",
        *extra,
    ):
        dataset = group[name]
        if dataset.shape != link_shape:
            msg = f"/{group_path}/{name} must match [snapshot,tx,rx]"
            raise SchemaValidationError(msg)
        if "index_order" not in dataset.attrs:
            msg = f"Missing index_order attribute on /{group_path}/{name}"
            raise SchemaValidationError(msg)
    success = group["detection_success"][()].astype(np.bool_)
    for name in ("toa_est_s", "one_way_range_est_m", "rtt_equiv_s", "range_error_m"):
        values = group[name][()]
        if not np.all(np.isfinite(values[success])):
            msg = f"/{group_path}/{name} must be finite where detection_success is true"
            raise SchemaValidationError(msg)
        if not np.all(np.isnan(values[~success])):
            msg = f"/{group_path}/{name} must be NaN where detection_success is false"
            raise SchemaValidationError(msg)
    if "selected_delay_bin" in group:
        selected = group["selected_delay_bin"][()]
        if not np.all(selected[success] >= 0):
            msg = f"/{group_path}/selected_delay_bin must be non-negative on success"
            raise SchemaValidationError(msg)
        if not np.all(selected[~success] == -1):
            msg = f"/{group_path}/selected_delay_bin must be -1 on failure"
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


NR_PUSCH_REQUIRED_FIELDS = (
    "waveform/num_prb",
    "waveform/subcarrier_spacing_khz",
    "waveform/num_layers",
    "waveform/num_antenna_ports",
    "waveform/mcs_index",
    "waveform/mcs_table",
    "waveform/dmrs_config_type",
    "waveform/dmrs_length",
    "waveform/dmrs_additional_position",
    "waveform/num_cdm_groups_without_data",
    "waveform/tx_grid",
    "waveform/rx_grid",
    "waveform/noise_variance",
    "waveform/tx_power_dbm_per_port",
    "waveform/tx_power_scale_linear",
    "waveform/serving_rx_index",
    "waveform/path_loss_db",
    "waveform/power_clipped_flag",
    "array/rx_snapshot_matrix",
    "array/aoa_label_rad",
    "array/aoa_heatmap_label",
    "array/angle_grid_rad",
    "array/spectrum_policy",
    "receiver/mimo_detector",
)


def _validate_nr_pusch_fields_if_applicable(h5: h5py.File) -> None:
    """When waveform/standard == 'nr_pusch', enforce NR-specific fields."""
    if "waveform/standard" not in h5:
        return
    std = h5["waveform/standard"][()]
    if isinstance(std, bytes):
        std = std.decode()
    if std != "nr_pusch":
        return

    _require_present(
        h5, NR_PUSCH_REQUIRED_FIELDS, kind=h5py.Dataset,
    )

    # num_layers >= 1
    nl = int(h5["waveform/num_layers"][()])
    if nl < 1:
        raise SchemaValidationError(
            f"/waveform/num_layers must be >= 1, got {nl}"
        )

    # num_antenna_ports >= num_layers
    nap = int(h5["waveform/num_antenna_ports"][()])
    if nap < nl:
        raise SchemaValidationError(
            f"/waveform/num_antenna_ports ({nap}) must be >= num_layers ({nl})"
        )

    # mimo_detector must be a recognized value
    det = h5["receiver/mimo_detector"][()]
    if isinstance(det, bytes):
        det = det.decode()
    if det not in ("lmmse", "kbest"):
        raise SchemaValidationError(
            f"/receiver/mimo_detector must be 'lmmse' or 'kbest', got {det!r}"
        )

    # receiver_type must be pusch_receiver
    if "receiver/receiver_type" in h5:
        rt = h5["receiver/receiver_type"][()]
        if isinstance(rt, bytes):
            rt = rt.decode()
        if rt != "pusch_receiver":
            raise SchemaValidationError(
                f"/receiver/receiver_type must be 'pusch_receiver' for NR PUSCH, got {rt!r}"
            )
    _validate_nr_pusch_waveform_grid_shapes(h5)
    _validate_common_power_fields(h5)
    _validate_nr_pusch_array_shapes(h5)


def _validate_nr_pusch_waveform_grid_shapes(h5: h5py.File) -> None:
    tx_grid = h5["waveform/tx_grid"]
    rx_grid = h5["waveform/rx_grid"]
    noise_variance = h5["waveform/noise_variance"]
    if tx_grid.ndim != 6:
        msg = f"/waveform/tx_grid must be rank 6, got {tx_grid.shape}"
        raise SchemaValidationError(msg)
    if rx_grid.ndim != 6:
        msg = f"/waveform/rx_grid must be rank 6, got {rx_grid.shape}"
        raise SchemaValidationError(msg)
    if noise_variance.ndim != 3:
        msg = f"/waveform/noise_variance must be rank 3, got {noise_variance.shape}"
        raise SchemaValidationError(msg)
    if tx_grid.shape[:3] != rx_grid.shape[:3]:
        msg = "/waveform/tx_grid and /waveform/rx_grid link dimensions must match"
        raise SchemaValidationError(msg)
    if noise_variance.shape != tx_grid.shape[:3]:
        msg = "/waveform/noise_variance must match waveform grid [snapshot,ul_tx,ul_rx]"
        raise SchemaValidationError(msg)
    for dataset_path in (
        "waveform/tx_grid",
        "waveform/rx_grid",
        "waveform/noise_variance",
    ):
        ds = h5[dataset_path]
        if "unit" not in ds.attrs or "index_order" not in ds.attrs:
            msg = f"Missing unit/index_order attribute on /{dataset_path}"
            raise SchemaValidationError(msg)


def _validate_common_power_fields(h5: h5py.File) -> None:
    tx_grid = h5["waveform/tx_grid"]
    tx_power = h5["waveform/tx_power_dbm_per_port"]
    scale = h5["waveform/tx_power_scale_linear"]
    serving = h5["waveform/serving_rx_index"]
    path_loss = h5["waveform/path_loss_db"]
    clipped = h5["waveform/power_clipped_flag"]
    link_tx_shape = tx_grid.shape[:2]
    if tx_power.ndim != 3:
        raise SchemaValidationError(
            f"/waveform/tx_power_dbm_per_port must be rank 3, got {tx_power.shape}"
        )
    if scale.shape != tx_power.shape or clipped.shape != tx_power.shape:
        msg = "/waveform TX power, scale, and clipped flag must share [snapshot,tx,port]"
        raise SchemaValidationError(msg)
    if tx_power.shape[:2] != link_tx_shape:
        msg = "/waveform/tx_power_dbm_per_port must match tx_grid [snapshot,tx]"
        raise SchemaValidationError(msg)
    if serving.shape != link_tx_shape or path_loss.shape != link_tx_shape:
        msg = "/waveform serving_rx_index/path_loss_db must match [snapshot,tx]"
        raise SchemaValidationError(msg)
    for dataset_path in (
        "waveform/tx_power_dbm_per_port",
        "waveform/tx_power_scale_linear",
        "waveform/serving_rx_index",
        "waveform/path_loss_db",
        "waveform/power_clipped_flag",
    ):
        ds = h5[dataset_path]
        if "unit" not in ds.attrs or "index_order" not in ds.attrs:
            msg = f"Missing unit/index_order attribute on /{dataset_path}"
            raise SchemaValidationError(msg)


def _validate_nr_pusch_array_shapes(h5: h5py.File) -> None:
    rx_grid = h5["waveform/rx_grid"]
    snapshot_matrix = h5["array/rx_snapshot_matrix"]
    aoa = h5["array/aoa_label_rad"]
    heatmap = h5["array/aoa_heatmap_label"]
    angle_grid = h5["array/angle_grid_rad"]
    link_shape = rx_grid.shape[:3]
    num_rx_ant = rx_grid.shape[3]

    if snapshot_matrix.shape != (*link_shape, num_rx_ant, num_rx_ant):
        msg = (
            "/array/rx_snapshot_matrix must match "
            "[snapshot,ul_tx,ul_rx,ul_rx_ant,ul_rx_ant]"
        )
        raise SchemaValidationError(msg)
    if aoa.shape != (*link_shape, 2):
        msg = "/array/aoa_label_rad must match [snapshot,ul_tx,ul_rx,angle_component]"
        raise SchemaValidationError(msg)
    if angle_grid.ndim != 3 or angle_grid.shape[-1] != 2:
        msg = "/array/angle_grid_rad must have shape [zenith,azimuth,2]"
        raise SchemaValidationError(msg)
    spectrum_shape = (*link_shape, *angle_grid.shape[:2])
    if heatmap.shape != spectrum_shape:
        msg = "/array/aoa_heatmap_label must match [snapshot,ul_tx,ul_rx,zenith,azimuth]"
        raise SchemaValidationError(msg)
    for optional_path in (
        "array/spatial_spectrum_truth",
        "array/spatial_spectrum_cfr_est",
        "array/spatial_spectrum_observation",
    ):
        if optional_path in h5 and h5[optional_path].shape != spectrum_shape:
            msg = f"/{optional_path} must match [snapshot,ul_tx,ul_rx,zenith,azimuth]"
            raise SchemaValidationError(msg)
    for dataset_path in (
        "array/rx_snapshot_matrix",
        "array/aoa_label_rad",
        "array/aoa_heatmap_label",
        "array/spatial_spectrum_truth",
        "array/spatial_spectrum_cfr_est",
        "array/spatial_spectrum_observation",
        "array/angle_grid_rad",
    ):
        if dataset_path not in h5:
            continue
        ds = h5[dataset_path]
        if "unit" not in ds.attrs or "index_order" not in ds.attrs:
            msg = f"Missing unit/index_order attribute on /{dataset_path}"
            raise SchemaValidationError(msg)


NR_SRS_REQUIRED_FIELDS = (
    "waveform/tx_grid",
    "waveform/rx_grid",
    "waveform/noise_variance",
    "waveform/tx_power_dbm_per_port",
    "waveform/tx_power_scale_linear",
    "waveform/serving_rx_index",
    "waveform/path_loss_db",
    "waveform/power_clipped_flag",
    "waveform/srs_resource_mask",
    "waveform/srs_pilot_symbols",
    "waveform/srs_re_symbol_indices",
    "waveform/srs_re_subcarrier_indices",
    "waveform/srs_port_tx_ant_map",
    "waveform/srs_prb_start_per_symbol",
    "waveform/srs_prb_count_per_symbol",
    "waveform/srs_cyclic_shift_indices",
    "waveform/srs_tx_power_dbm",
    "waveform/srs_power_scale_linear",
    "observation/cfr_est_resource",
    "array/rx_snapshot_matrix",
    "array/aoa_label_rad",
    "array/aoa_heatmap_label",
    "array/angle_grid_rad",
    "array/spectrum_policy",
)


MULTIUSER_SRS_REQUIRED_FIELDS = (
    "multiuser/standard",
    "multiuser/resource_strategy",
    "multiuser/rx_grid_shared",
    "multiuser/noise_variance",
    "multiuser/snr_db",
    "multiuser/rssi_dbm",
    "multiuser/noise_power_dbm",
    "multiuser/active_tx_indices",
    "multiuser/active_tx_global_indices",
    "multiuser/active_tx_mask",
    "multiuser/active_tx_positions_m",
    "multiuser/comb_offset",
    "multiuser/prb_start",
    "multiuser/prb_count",
    "multiuser/re_symbol_indices",
    "multiuser/re_subcarrier_indices",
    "multiuser/re_mask",
    "multiuser/allocated_subcarrier_indices",
    "multiuser/allocated_subcarrier_mask",
    "multiuser/resource_occupancy_count",
    "multiuser/resource_collision_mask",
    "multiuser/cfr_est_resource",
    "multiuser/cfr_est_allocated",
)


def _validate_nr_srs_fields_if_applicable(h5: h5py.File) -> None:
    """When waveform/standard == 'nr_srs', enforce stage-2 SRS subset fields."""

    if "waveform/standard" not in h5:
        return
    std = h5["waveform/standard"][()]
    if isinstance(std, bytes):
        std = std.decode()
    if std != "nr_srs":
        return

    _require_absent(h5, "waveform/pilot_code")
    _require_absent(h5, "waveform/srs_tx_grid")
    _require_absent(h5, "observation/srs_cfr_est")
    _require_present(h5, NR_SRS_REQUIRED_FIELDS, kind=h5py.Dataset)
    _validate_common_power_fields(h5)
    tx_grid = h5["waveform/tx_grid"]
    rx_grid = h5["waveform/rx_grid"]
    noise_variance = h5["waveform/noise_variance"]
    resource_mask = h5["waveform/srs_resource_mask"]
    pilot_symbols = h5["waveform/srs_pilot_symbols"]
    re_symbols = h5["waveform/srs_re_symbol_indices"]
    re_indices = h5["waveform/srs_re_subcarrier_indices"]
    port_map = h5["waveform/srs_port_tx_ant_map"]
    prb_start = h5["waveform/srs_prb_start_per_symbol"]
    prb_count = h5["waveform/srs_prb_count_per_symbol"]
    cyclic_shifts = h5["waveform/srs_cyclic_shift_indices"]
    tx_power = h5["waveform/srs_tx_power_dbm"]
    power_scale = h5["waveform/srs_power_scale_linear"]
    cfr_resource = h5["observation/cfr_est_resource"]
    if tx_grid.ndim != 6:
        raise SchemaValidationError(f"/waveform/tx_grid must be rank 6, got {tx_grid.shape}")
    if rx_grid.ndim != 6:
        raise SchemaValidationError(f"/waveform/rx_grid must be rank 6, got {rx_grid.shape}")
    if noise_variance.shape != tx_grid.shape[:3]:
        msg = "/waveform/noise_variance must match [snapshot,ul_tx,ul_rx]"
        raise SchemaValidationError(msg)
    if resource_mask.shape != tx_grid.shape[4:6]:
        msg = "/waveform/srs_resource_mask must match [ofdm_symbol,subcarrier]"
        raise SchemaValidationError(msg)
    if pilot_symbols.shape[1:] != tx_grid.shape[4:6]:
        msg = "/waveform/srs_pilot_symbols must match [srs_port,ofdm_symbol,subcarrier]"
        raise SchemaValidationError(msg)
    num_ports = int(pilot_symbols.shape[0])
    if re_symbols.ndim != 1 or re_indices.ndim != 1 or re_symbols.shape != re_indices.shape:
        msg = "/waveform/srs_re_symbol_indices and srs_re_subcarrier_indices must match [srs_re]"
        raise SchemaValidationError(msg)
    resource_re_count = int(np.asarray(resource_mask).sum())
    if re_indices.shape[0] != resource_re_count:
        msg = "/waveform/srs_re_* indices must match flattened resource mask RE count"
        raise SchemaValidationError(msg)
    re_symbol_values = np.asarray(re_symbols, dtype=np.int64)
    re_subcarrier_values = np.asarray(re_indices, dtype=np.int64)
    if (
        np.any(re_symbol_values < 0)
        or np.any(re_symbol_values >= resource_mask.shape[0])
        or np.any(re_subcarrier_values < 0)
        or np.any(re_subcarrier_values >= resource_mask.shape[1])
    ):
        msg = "/waveform/srs_re_* indices must be inside resource mask bounds"
        raise SchemaValidationError(msg)
    if not np.all(np.asarray(resource_mask)[re_symbol_values, re_subcarrier_values]):
        msg = "/waveform/srs_re_* indices must point to active resource mask entries"
        raise SchemaValidationError(msg)
    srs_symbol_count = int(np.asarray(resource_mask).any(axis=1).sum())
    if port_map.shape != (num_ports, srs_symbol_count):
        msg = "/waveform/srs_port_tx_ant_map must match [srs_port,srs_symbol]"
        raise SchemaValidationError(msg)
    if prb_start.shape != (srs_symbol_count,) or prb_count.shape != (srs_symbol_count,):
        msg = "/waveform/srs_prb_*_per_symbol must match [srs_symbol]"
        raise SchemaValidationError(msg)
    if cyclic_shifts.shape != (num_ports,):
        msg = "/waveform/srs_cyclic_shift_indices must match [srs_port]"
        raise SchemaValidationError(msg)
    if tx_power.shape != (*tx_grid.shape[:2], num_ports):
        msg = "/waveform/srs_tx_power_dbm must match [snapshot,tx,srs_port]"
        raise SchemaValidationError(msg)
    if power_scale.shape != (*tx_grid.shape[:2], num_ports):
        msg = "/waveform/srs_power_scale_linear must match [snapshot,tx,srs_port]"
        raise SchemaValidationError(msg)
    expected_resource_shape = (
        *tx_grid.shape[:3],
        rx_grid.shape[3],
        num_ports,
        re_indices.shape[0],
    )
    if cfr_resource.shape != expected_resource_shape:
        msg = "/observation/cfr_est_resource must match [snapshot,tx,rx,rx_ant,srs_port,srs_re]"
        raise SchemaValidationError(msg)
    expected_full_shape = (
        *tx_grid.shape[:3],
        rx_grid.shape[3],
        tx_grid.shape[3],
        tx_grid.shape[5],
    )
    if h5["observation/cfr_est"].shape != expected_full_shape:
        msg = "/observation/cfr_est must match full-band SRS link/antenna/subcarrier shape"
        raise SchemaValidationError(msg)
    if rx_grid.shape[:3] != tx_grid.shape[:3]:
        msg = "/waveform/tx_grid and /waveform/rx_grid link dimensions must match"
        raise SchemaValidationError(msg)
    if rx_grid.shape[4:6] != tx_grid.shape[4:6]:
        msg = "/waveform/tx_grid and /waveform/rx_grid OFDM dimensions must match"
        raise SchemaValidationError(msg)

    link_shape = rx_grid.shape[:3]
    num_rx_ant = rx_grid.shape[3]
    snapshot_matrix = h5["array/rx_snapshot_matrix"]
    aoa = h5["array/aoa_label_rad"]
    angle_grid = h5["array/angle_grid_rad"]
    if snapshot_matrix.shape != (*link_shape, num_rx_ant, num_rx_ant):
        msg = "/array/rx_snapshot_matrix must match SRS RX grid receiver antennas"
        raise SchemaValidationError(msg)
    if aoa.shape != (*link_shape, 2):
        msg = "/array/aoa_label_rad must match SRS link dimensions"
        raise SchemaValidationError(msg)
    if angle_grid.ndim != 3 or angle_grid.shape[-1] != 2:
        msg = "/array/angle_grid_rad must have shape [zenith,azimuth,2]"
        raise SchemaValidationError(msg)
    for dataset_path in (
        "waveform/tx_grid",
        "waveform/rx_grid",
        "waveform/noise_variance",
        "waveform/tx_power_dbm_per_port",
        "waveform/tx_power_scale_linear",
        "waveform/serving_rx_index",
        "waveform/path_loss_db",
        "waveform/power_clipped_flag",
        "waveform/srs_resource_mask",
        "waveform/srs_pilot_symbols",
        "waveform/srs_re_symbol_indices",
        "waveform/srs_re_subcarrier_indices",
        "waveform/srs_port_tx_ant_map",
        "waveform/srs_prb_start_per_symbol",
        "waveform/srs_prb_count_per_symbol",
        "waveform/srs_cyclic_shift_indices",
        "waveform/srs_tx_power_dbm",
        "waveform/srs_power_scale_linear",
        "observation/cfr_est_resource",
    ):
        if dataset_path not in h5:
            continue
        ds = h5[dataset_path]
        if "unit" not in ds.attrs or "index_order" not in ds.attrs:
            msg = f"Missing unit/index_order attribute on /{dataset_path}"
            raise SchemaValidationError(msg)


def _validate_multiuser_srs(h5: h5py.File) -> None:
    _require_present(h5, ("multiuser",), kind=h5py.Group)
    _require_present(h5, MULTIUSER_SRS_REQUIRED_FIELDS, kind=h5py.Dataset)
    standard = _read_string(h5["multiuser/standard"])
    if standard != "nr_srs":
        msg = "/multiuser/standard must be nr_srs"
        raise SchemaValidationError(msg)
    strategy = _read_string(h5["multiuser/resource_strategy"])
    if strategy not in ("comb_offset", "prb_split"):
        msg = "/multiuser/resource_strategy must be comb_offset or prb_split"
        raise SchemaValidationError(msg)
    rx_grid = h5["multiuser/rx_grid_shared"]
    if rx_grid.ndim != 6:
        msg = "/multiuser/rx_grid_shared must be [snapshot,frame,rx,rx_ant,symbol,subcarrier]"
        raise SchemaValidationError(msg)
    snap, frame, rx, rx_ant, symbols, subcarriers = rx_grid.shape
    if h5["topology/rx_positions_m"].shape[0] != rx:
        msg = "/multiuser/rx_grid_shared RX dimension must match topology"
        raise SchemaValidationError(msg)
    if h5["antenna/rx_num_ant"][()] != rx_ant:
        msg = "/multiuser/rx_grid_shared rx_ant dimension must match antenna"
        raise SchemaValidationError(msg)
    for dataset_path in (
        "multiuser/noise_variance",
        "multiuser/snr_db",
        "multiuser/rssi_dbm",
        "multiuser/noise_power_dbm",
    ):
        if h5[dataset_path].shape != (snap, frame, rx):
            msg = f"/{dataset_path} must match [snapshot,frame,rx]"
            raise SchemaValidationError(msg)
    active_tx = h5["multiuser/active_tx_indices"]
    active_mask = h5["multiuser/active_tx_mask"]
    if active_tx.ndim != 2 or active_tx.shape[0] != frame:
        msg = "/multiuser/active_tx_indices must be [frame,active_ue]"
        raise SchemaValidationError(msg)
    if active_mask.shape != active_tx.shape:
        msg = "/multiuser/active_tx_mask must match active_tx_indices"
        raise SchemaValidationError(msg)
    active_ue = active_tx.shape[1]
    for dataset_path in (
        "multiuser/active_tx_global_indices",
        "multiuser/comb_offset",
    ):
        if h5[dataset_path].shape != active_tx.shape:
            msg = f"/{dataset_path} must match [frame,active_ue]"
            raise SchemaValidationError(msg)
    if h5["multiuser/active_tx_positions_m"].shape != (*active_tx.shape, 3):
        msg = "/multiuser/active_tx_positions_m must match [frame,active_ue,3]"
        raise SchemaValidationError(msg)
    for dataset_path in ("multiuser/prb_start", "multiuser/prb_count"):
        ds = h5[dataset_path]
        if ds.ndim != 3 or ds.shape[:2] != (frame, active_ue):
            msg = f"/{dataset_path} must be [frame,active_ue,srs_symbol]"
            raise SchemaValidationError(msg)
    re_symbols = h5["multiuser/re_symbol_indices"]
    re_subcarriers = h5["multiuser/re_subcarrier_indices"]
    re_mask = h5["multiuser/re_mask"]
    if (
        re_symbols.ndim != 3
        or re_symbols.shape[:2] != (frame, active_ue)
        or re_subcarriers.shape != re_symbols.shape
        or re_mask.shape != re_symbols.shape
    ):
        msg = "/multiuser/re_* datasets must match [frame,active_ue,max_srs_re]"
        raise SchemaValidationError(msg)
    allocated = h5["multiuser/allocated_subcarrier_indices"]
    allocated_mask = h5["multiuser/allocated_subcarrier_mask"]
    if allocated.ndim != 3 or allocated.shape[:2] != (frame, active_ue):
        msg = "/multiuser/allocated_subcarrier_indices must be [frame,active_ue,max_alloc]"
        raise SchemaValidationError(msg)
    if allocated_mask.shape != allocated.shape:
        msg = "/multiuser/allocated_subcarrier_mask must match allocated indices"
        raise SchemaValidationError(msg)
    occupancy = h5["multiuser/resource_occupancy_count"]
    collision = h5["multiuser/resource_collision_mask"]
    if occupancy.shape != (frame, symbols, subcarriers):
        msg = "/multiuser/resource_occupancy_count must match [frame,symbol,subcarrier]"
        raise SchemaValidationError(msg)
    if collision.shape != occupancy.shape:
        msg = "/multiuser/resource_collision_mask must match occupancy"
        raise SchemaValidationError(msg)
    if not np.array_equal(collision[()], occupancy[()] > 1):
        msg = "/multiuser/resource_collision_mask must equal resource_occupancy_count > 1"
        raise SchemaValidationError(msg)
    cfr_resource = h5["multiuser/cfr_est_resource"]
    if cfr_resource.ndim != 7 or cfr_resource.shape[:5] != (snap, frame, active_ue, rx, rx_ant):
        msg = (
            "/multiuser/cfr_est_resource must be "
            "[snapshot,frame,active_ue,rx,rx_ant,srs_port,max_srs_re]"
        )
        raise SchemaValidationError(msg)
    cfr_allocated = h5["multiuser/cfr_est_allocated"]
    if cfr_allocated.ndim != 7 or cfr_allocated.shape[:5] != (snap, frame, active_ue, rx, rx_ant):
        msg = (
            "/multiuser/cfr_est_allocated must be "
            "[snapshot,frame,active_ue,rx,rx_ant,tx_ant,max_alloc]"
        )
        raise SchemaValidationError(msg)
    if cfr_allocated.shape[5] != h5["antenna/tx_num_ant"][()]:
        msg = "/multiuser/cfr_est_allocated tx_ant dimension must match antenna"
        raise SchemaValidationError(msg)
    _validate_multiuser_indices(h5, active_tx[()], active_mask[()])
    for dataset_path in MULTIUSER_SRS_REQUIRED_FIELDS:
        if dataset_path in ("multiuser/standard", "multiuser/resource_strategy"):
            continue
        ds = h5[dataset_path]
        if "unit" not in ds.attrs or "index_order" not in ds.attrs:
            msg = f"Missing unit/index_order attribute on /{dataset_path}"
            raise SchemaValidationError(msg)


def _validate_iq(h5: h5py.File) -> None:
    _require_present(
        h5,
        (
            "iq/sample_rate_hz",
            "iq/fft_size",
            "iq/cp_length",
            "iq/num_ofdm_symbols",
            "iq/time_domain_convention",
        ),
        kind=h5py.Dataset,
    )
    fft_size = int(h5["iq/fft_size"][()])
    cp_length = int(h5["iq/cp_length"][()])
    num_symbols = int(h5["iq/num_ofdm_symbols"][()])
    if fft_size < 2:
        raise SchemaValidationError("/iq/fft_size must be >= 2")
    if cp_length < 0:
        raise SchemaValidationError("/iq/cp_length must be non-negative")
    if num_symbols < 1:
        raise SchemaValidationError("/iq/num_ofdm_symbols must be positive")
    if "link" in h5["iq"]:
        _validate_iq_link(h5, fft_size, cp_length, num_symbols)
    if "noncooperative" in h5["iq"]:
        _validate_iq_noncooperative(h5, fft_size, cp_length, num_symbols)


def _validate_iq_link(
    h5: h5py.File,
    fft_size: int,
    cp_length: int,
    num_symbols: int,
) -> None:
    link = h5["iq/link"]
    frequency_shape = None
    for name in ("frequency_clean", "frequency_observed"):
        if name not in link:
            continue
        ds = link[name]
        if ds.ndim != 6:
            raise SchemaValidationError(f"/iq/link/{name} must be rank 6")
        if ds.shape[-2:] != (num_symbols, fft_size):
            msg = f"/iq/link/{name} symbol/subcarrier dimensions must match /iq metadata"
            raise SchemaValidationError(msg)
        if frequency_shape is None:
            frequency_shape = ds.shape
        elif ds.shape != frequency_shape:
            raise SchemaValidationError("/iq/link frequency datasets must share shape")
        _require_unit_order(ds, f"iq/link/{name}")

    time_shape = None
    expected_samples = num_symbols * (fft_size + cp_length)
    for name in ("time_clean", "time_observed"):
        if name not in link:
            continue
        ds = link[name]
        if ds.ndim != 5:
            raise SchemaValidationError(f"/iq/link/{name} must be rank 5")
        if ds.shape[-1] != expected_samples:
            msg = f"/iq/link/{name} sample dimension must be {expected_samples}"
            raise SchemaValidationError(msg)
        if time_shape is None:
            time_shape = ds.shape
        elif ds.shape != time_shape:
            raise SchemaValidationError("/iq/link time datasets must share shape")
        _require_unit_order(ds, f"iq/link/{name}")
    if frequency_shape is not None and time_shape is not None:
        if frequency_shape[:4] != time_shape[:4]:
            raise SchemaValidationError("/iq/link frequency/time leading dims must match")


def _validate_iq_noncooperative(
    h5: h5py.File,
    fft_size: int,
    cp_length: int,
    num_symbols: int,
) -> None:
    group = h5["iq/noncooperative"]
    _require_present(
        h5,
        (
            "iq/noncooperative/noise_variance",
            "iq/noncooperative/snr_db",
            "iq/noncooperative/rssi_dbm",
            "iq/noncooperative/noise_power_dbm",
            "iq/noncooperative/active_tx_indices",
            "iq/noncooperative/active_tx_global_indices",
            "iq/noncooperative/active_tx_mask",
            "iq/noncooperative/active_tx_positions_m",
            "iq/noncooperative/resource_occupancy_count",
            "iq/noncooperative/resource_collision_mask",
        ),
        kind=h5py.Dataset,
    )
    expected_samples = num_symbols * (fft_size + cp_length)
    time_shape = None
    for name in ("rx_time_clean", "rx_time_observed"):
        if name not in group:
            continue
        ds = group[name]
        if ds.ndim != 5:
            raise SchemaValidationError(f"/iq/noncooperative/{name} must be rank 5")
        if ds.shape[-1] != expected_samples:
            msg = f"/iq/noncooperative/{name} sample dimension must be {expected_samples}"
            raise SchemaValidationError(msg)
        if time_shape is None:
            time_shape = ds.shape
        elif ds.shape != time_shape:
            raise SchemaValidationError("/iq/noncooperative time datasets must share shape")
        _require_unit_order(ds, f"iq/noncooperative/{name}")
    if time_shape is None:
        raise SchemaValidationError("/iq/noncooperative requires rx_time_clean or rx_time_observed")
    snap, frame, rx, _rx_ant, _samples = time_shape
    for name in ("noise_variance", "snr_db", "rssi_dbm", "noise_power_dbm"):
        ds = group[name]
        if ds.shape != (snap, frame, rx):
            msg = f"/iq/noncooperative/{name} must be [snapshot,frame,rx]"
            raise SchemaValidationError(msg)
        _require_unit_order(ds, f"iq/noncooperative/{name}")
    active_tx = group["active_tx_indices"]
    if active_tx.ndim != 2 or active_tx.shape[0] != frame:
        raise SchemaValidationError(
            "/iq/noncooperative/active_tx_indices must be [frame,active_tx]"
        )
    active_shape = active_tx.shape
    for name in ("active_tx_global_indices", "active_tx_mask"):
        if group[name].shape != active_shape:
            msg = f"/iq/noncooperative/{name} must match active_tx_indices"
            raise SchemaValidationError(msg)
    if group["active_tx_positions_m"].shape != (*active_shape, 3):
        raise SchemaValidationError(
            "/iq/noncooperative/active_tx_positions_m must be [frame,active_tx,3]"
        )
    occupancy = group["resource_occupancy_count"]
    collision = group["resource_collision_mask"]
    if occupancy.shape != (frame, num_symbols, fft_size):
        raise SchemaValidationError(
            "/iq/noncooperative/resource_occupancy_count must be "
            "[frame,ofdm_symbol,subcarrier]"
        )
    if collision.shape != occupancy.shape:
        raise SchemaValidationError(
            "/iq/noncooperative/resource_collision_mask must match occupancy"
        )
    if not np.array_equal(collision[()], occupancy[()] > 1):
        raise SchemaValidationError(
            "/iq/noncooperative/resource_collision_mask must equal occupancy > 1"
        )


def _require_unit_order(dataset: h5py.Dataset, path: str) -> None:
    if "unit" not in dataset.attrs or "index_order" not in dataset.attrs:
        raise SchemaValidationError(f"Missing unit/index_order attribute on /{path}")


def _validate_multiuser_indices(
    h5: h5py.File,
    active_tx: np.ndarray,
    active_mask: np.ndarray,
) -> None:
    tx_count = h5["topology/tx_positions_m"].shape[0]
    valid = active_mask.astype(np.bool_)
    if np.any(active_tx[valid] < 0) or np.any(active_tx[valid] >= tx_count):
        msg = "/multiuser/active_tx_indices contains out-of-range active TX index"
        raise SchemaValidationError(msg)
    if np.any(active_tx[~valid] != -1):
        msg = "/multiuser inactive active_tx_indices entries must be -1"
        raise SchemaValidationError(msg)


def _validate_array_outputs_if_present(h5: h5py.File) -> None:
    if "array" not in h5:
        return
    group = h5["array"]
    if "angle_grid_rad" not in group:
        return

    angle_grid = group["angle_grid_rad"]
    if angle_grid.ndim != 3 or angle_grid.shape[-1] != 2:
        msg = "/array/angle_grid_rad must have shape [zenith,azimuth,2]"
        raise SchemaValidationError(msg)

    link_shape = None
    if "aoa_label_rad" in group:
        aoa = group["aoa_label_rad"]
        if aoa.ndim != 4 or aoa.shape[-1] != 2:
            msg = "/array/aoa_label_rad must match [snapshot,ul_tx,ul_rx,angle_component]"
            raise SchemaValidationError(msg)
        link_shape = aoa.shape[:3]

    for dataset_path in (
        "array/aoa_heatmap_label",
        "array/spatial_spectrum_truth",
        "array/spatial_spectrum_cfr_est",
        "array/spatial_spectrum_observation",
    ):
        if dataset_path not in h5:
            continue
        ds = h5[dataset_path]
        if ds.ndim != 5:
            msg = f"/{dataset_path} must have rank 5"
            raise SchemaValidationError(msg)
        if ds.shape[-2:] != angle_grid.shape[:2]:
            msg = f"/{dataset_path} angle dimensions must match /array/angle_grid_rad"
            raise SchemaValidationError(msg)
        if link_shape is None:
            link_shape = ds.shape[:3]
        elif ds.shape[:3] != link_shape:
            msg = f"/{dataset_path} link dimensions must match /array/aoa_label_rad"
            raise SchemaValidationError(msg)

    for dataset_path in (
        "array/rx_snapshot_matrix",
        "array/aoa_label_rad",
        "array/aoa_heatmap_label",
        "array/spatial_spectrum_truth",
        "array/spatial_spectrum_cfr_est",
        "array/spatial_spectrum_observation",
        "array/angle_grid_rad",
    ):
        if dataset_path not in h5:
            continue
        ds = h5[dataset_path]
        if "unit" not in ds.attrs or "index_order" not in ds.attrs:
            msg = f"Missing unit/index_order attribute on /{dataset_path}"
            raise SchemaValidationError(msg)


def _validate_bler_contract(h5: h5py.File) -> None:
    """Validate TB/CRC BLER contract for NR PUSCH output.

    Only enforced when waveform/standard == "nr_pusch" because
    custom_ofdm uses legacy BLER semantics (num_blocks may be 0).
    """
    if "evaluation/num_blocks" not in h5:
        return
    if "evaluation/num_block_errors" not in h5:
        return
    if "evaluation/bler" not in h5:
        return

    # Only enforce strict BLER contract for NR PUSCH
    if "waveform/standard" in h5:
        std = h5["waveform/standard"][()]
        if isinstance(std, bytes):
            std = std.decode()
        if std != "nr_pusch":
            return
    else:
        return

    num_blocks = int(h5["evaluation/num_blocks"][()])
    num_block_errors = int(h5["evaluation/num_block_errors"][()])
    bler = float(h5["evaluation/bler"][()])

    if num_blocks <= 0:
        raise SchemaValidationError(
            f"/evaluation/num_blocks must be > 0, got {num_blocks}"
        )
    if num_block_errors < 0:
        raise SchemaValidationError(
            f"/evaluation/num_block_errors must be >= 0, got {num_block_errors}"
        )
    if num_block_errors > num_blocks:
        raise SchemaValidationError(
            f"/evaluation/num_block_errors ({num_block_errors}) "
            f"must be <= num_blocks ({num_blocks})"
        )
    expected_bler = num_block_errors / num_blocks
    if not np.isclose(bler, expected_bler, rtol=1e-4, atol=1e-6):
        raise SchemaValidationError(
            f"/evaluation/bler ({bler}) must equal "
            f"num_block_errors/num_blocks ({expected_bler})"
        )
