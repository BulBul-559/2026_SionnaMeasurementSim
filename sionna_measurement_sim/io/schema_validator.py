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


def validate_hdf5_contract(path: str | Path) -> None:
    """Validate the minimal truth HDF5 contract."""

    with h5py.File(path, "r") as h5:
        _require_absent(h5, "channel/cfr")
        _require_absent(h5, "derived/rtt_like_m")
        _require_absent(h5, "derived/rtt_like_s")
        _require_present(h5, ("meta/schema_version",), kind=h5py.Dataset)
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
    "array/rx_snapshot_matrix",
    "array/aoa_label_rad",
    "array/aoa_heatmap_label",
    "array/spatial_spectrum_label",
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


def _validate_nr_pusch_array_shapes(h5: h5py.File) -> None:
    rx_grid = h5["waveform/rx_grid"]
    snapshot_matrix = h5["array/rx_snapshot_matrix"]
    aoa = h5["array/aoa_label_rad"]
    heatmap = h5["array/aoa_heatmap_label"]
    spectrum = h5["array/spatial_spectrum_label"]
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
    if spectrum.shape != spectrum_shape:
        msg = (
            "/array/spatial_spectrum_label must match "
            "[snapshot,ul_tx,ul_rx,zenith,azimuth]"
        )
        raise SchemaValidationError(msg)
    for optional_path in (
        "array/spatial_spectrum_truth",
        "array/spatial_spectrum_cfr_est",
        "array/spatial_spectrum_observation",
        "array/spatial_spectrum_srs",
    ):
        if optional_path in h5 and h5[optional_path].shape != spectrum_shape:
            msg = f"/{optional_path} must match [snapshot,ul_tx,ul_rx,zenith,azimuth]"
            raise SchemaValidationError(msg)
    for dataset_path in (
        "array/rx_snapshot_matrix",
        "array/aoa_label_rad",
        "array/aoa_heatmap_label",
        "array/spatial_spectrum_label",
        "array/spatial_spectrum_truth",
        "array/spatial_spectrum_cfr_est",
        "array/spatial_spectrum_observation",
        "array/spatial_spectrum_srs",
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
    "waveform/srs_resource_mask",
    "waveform/srs_pilot_symbols",
    "waveform/srs_port_index",
    "waveform/srs_re_subcarrier_indices",
    "observation/cfr_est_resource",
    "array/rx_snapshot_matrix",
    "array/aoa_label_rad",
    "array/aoa_heatmap_label",
    "array/spatial_spectrum_label",
    "array/angle_grid_rad",
    "array/spectrum_policy",
)


def _validate_nr_srs_fields_if_applicable(h5: h5py.File) -> None:
    """When waveform/standard == 'nr_srs', enforce stage-1 SRS subset fields."""

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
    tx_grid = h5["waveform/tx_grid"]
    rx_grid = h5["waveform/rx_grid"]
    noise_variance = h5["waveform/noise_variance"]
    resource_mask = h5["waveform/srs_resource_mask"]
    pilot_symbols = h5["waveform/srs_pilot_symbols"]
    port_index = h5["waveform/srs_port_index"]
    re_indices = h5["waveform/srs_re_subcarrier_indices"]
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
    if port_index.shape != (tx_grid.shape[3],):
        msg = "/waveform/srs_port_index must match [ul_tx_ant]"
        raise SchemaValidationError(msg)
    resource_subcarrier_count = int(np.asarray(resource_mask).any(axis=0).sum())
    if re_indices.ndim != 1 or re_indices.shape[0] != resource_subcarrier_count:
        msg = "/waveform/srs_re_subcarrier_indices must match resource mask subcarriers"
        raise SchemaValidationError(msg)
    expected_resource_shape = (
        *tx_grid.shape[:3],
        rx_grid.shape[3],
        tx_grid.shape[3],
        re_indices.shape[0],
    )
    if cfr_resource.shape != expected_resource_shape:
        msg = "/observation/cfr_est_resource must match [snapshot,tx,rx,rx_ant,tx_ant,srs_re]"
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
    spectrum_shape = (*link_shape, *angle_grid.shape[:2])
    if snapshot_matrix.shape != (*link_shape, num_rx_ant, num_rx_ant):
        msg = "/array/rx_snapshot_matrix must match SRS RX grid receiver antennas"
        raise SchemaValidationError(msg)
    if aoa.shape != (*link_shape, 2):
        msg = "/array/aoa_label_rad must match SRS link dimensions"
        raise SchemaValidationError(msg)
    if (
        "array/spatial_spectrum_srs" in h5
        and h5["array/spatial_spectrum_srs"].shape != spectrum_shape
    ):
        msg = "/array/spatial_spectrum_srs must match [snapshot,ul_tx,ul_rx,zenith,azimuth]"
        raise SchemaValidationError(msg)
    for dataset_path in (
        "waveform/tx_grid",
        "waveform/rx_grid",
        "waveform/noise_variance",
        "waveform/srs_resource_mask",
        "waveform/srs_pilot_symbols",
        "waveform/srs_port_index",
        "waveform/srs_re_subcarrier_indices",
        "observation/cfr_est_resource",
        "array/spatial_spectrum_srs",
    ):
        if dataset_path not in h5:
            continue
        ds = h5[dataset_path]
        if "unit" not in ds.attrs or "index_order" not in ds.attrs:
            msg = f"Missing unit/index_order attribute on /{dataset_path}"
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
        "array/spatial_spectrum_label",
        "array/spatial_spectrum_truth",
        "array/spatial_spectrum_cfr_est",
        "array/spatial_spectrum_observation",
        "array/spatial_spectrum_srs",
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
        "array/spatial_spectrum_label",
        "array/spatial_spectrum_truth",
        "array/spatial_spectrum_cfr_est",
        "array/spatial_spectrum_observation",
        "array/spatial_spectrum_srs",
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
