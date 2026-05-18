"""HDF5 writer for domain results.

The writer consumes only domain models. It must not import or inspect Sionna
native objects.
"""

from __future__ import annotations

import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from sionna_measurement_sim.domain.results import MeasurementSimulationResult

UTF8_DTYPE = h5py.string_dtype(encoding="utf-8")
_ACTIVE_COMPRESSION: ContextVar[str] = ContextVar(
    "_ACTIVE_COMPRESSION",
    default="gzip",
)
_ACTIVE_TRACER: ContextVar[Any | None] = ContextVar("_ACTIVE_TRACER", default=None)


def write_measurement_result(
    path: str | Path,
    result: MeasurementSimulationResult,
    *,
    compression: str = "gzip",
    tracer: Any | None = None,
) -> Path:
    """Write a truth-only result to an HDF5 file."""

    if compression not in ("gzip", "lzf", "none"):
        msg = f"Unsupported HDF5 compression: {compression!r}"
        raise ValueError(msg)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    compression_token = _ACTIVE_COMPRESSION.set(compression)
    tracer_token = _ACTIVE_TRACER.set(tracer)
    try:
        with h5py.File(output_path, "w") as h5:
            _write_meta(h5, result)
            _write_input(h5, result)
            _write_shard(h5, result)
            _write_topology(h5, result)
            _write_devices(h5, result)
            _write_antenna(h5, result)
            _write_scene(h5, result)
            _write_frequency(h5, result)
            _write_derived(h5, result)
            _write_truth(h5, result)
            _write_path_samples(h5, result)
            _write_nlos_path_truth(h5, result)
            _write_path_full(h5, result)
            _write_cir_truth(h5, result)
            _write_waveform(h5, result)
            _write_array(h5, result)
            _write_observation(h5, result)
            _write_impairments(h5, result)
            _write_receiver(h5, result)
            _write_evaluation(h5, result)
            _write_calibration(h5, result)
            _write_motion(h5, result)
            _write_link(h5, result)
            _write_runtime(h5, result)
    finally:
        _ACTIVE_TRACER.reset(tracer_token)
        _ACTIVE_COMPRESSION.reset(compression_token)

    return output_path


def _write_meta(h5: h5py.File, result: MeasurementSimulationResult) -> None:
    meta = result.metadata
    group = h5.require_group("meta")
    _write_scalar(group, "schema_version", meta.schema_version)
    _write_scalar(group, "contract_name", meta.contract_name)
    _write_scalar(group, "producer", meta.producer)
    _write_scalar(group, "created_at", meta.created_at)
    _write_scalar(group, "run_id", meta.run_id)
    _write_scalar(group, "git_commit", meta.git_commit)
    _write_scalar(group, "random_seed", np.int64(meta.random_seed))
    _write_scalar(group, "coordinate_system", meta.coordinate_system)
    _write_scalar(group, "unit_convention", meta.unit_convention)
    _write_scalar(group, "index_order", meta.index_order)
    _write_scalar(group, "truth_branch_enabled", bool(meta.truth_branch_enabled))
    _write_scalar(group, "observation_branch_enabled", bool(meta.observation_branch_enabled))
    _write_scalar(group, "measurement_realism_level", meta.measurement_realism_level)
    _write_scalar(group, "config_snapshot", meta.config_snapshot)
    _write_scalar(group, "software_versions", meta.software_versions)


def _write_input(h5: h5py.File, result: MeasurementSimulationResult) -> None:
    input_spec = result.input_spec
    group = h5.require_group("input")
    _write_scalar(group, "label_file", input_spec.label_file)
    _write_scalar(group, "scene_file", input_spec.scene_file)
    _write_scalar(group, "input_dataset_id", input_spec.input_dataset_id)
    _write_scalar(group, "input_schema", input_spec.input_schema)


def _write_shard(h5: h5py.File, result: MeasurementSimulationResult) -> None:
    shard = result.shard
    if shard is None:
        return
    group = h5.require_group("shard")
    _write_scalar(group, "shard_index", np.int32(shard.shard_index))
    _write_scalar(group, "shard_count", np.int32(shard.shard_count))
    _write_scalar(group, "axis", shard.axis)
    _write_scalar(group, "global_rx_start", np.int64(shard.global_rx_start))
    _write_dataset(group, "global_rx_indices", shard.global_rx_indices)
    _write_dataset(group, "global_tx_indices", shard.global_tx_indices)


def _write_topology(h5: h5py.File, result: MeasurementSimulationResult) -> None:
    topology = result.topology
    group = h5.require_group("topology")
    _write_dataset(group, "tx_positions_m", topology.tx_positions_m, unit="m")
    _write_dataset(group, "rx_positions_m", topology.rx_positions_m, unit="m")
    _write_string_array(group, "tx_labels", topology.tx_labels)
    _write_string_array(group, "rx_labels", topology.rx_labels)


def _write_devices(h5: h5py.File, result: MeasurementSimulationResult) -> None:
    devices = result.devices
    group = h5.require_group("devices")
    _write_dataset(group, "tx_velocity_mps", devices.tx_velocity_mps, unit="m/s")
    _write_dataset(group, "rx_velocity_mps", devices.rx_velocity_mps, unit="m/s")
    _write_dataset(group, "tx_orientation_rad", devices.tx_orientation_rad, unit="rad")
    _write_dataset(group, "rx_orientation_rad", devices.rx_orientation_rad, unit="rad")


def _write_antenna(h5: h5py.File, result: MeasurementSimulationResult) -> None:
    antenna = result.antenna
    group = h5.require_group("antenna")
    _write_scalar(group, "tx_array_type", antenna.tx_array_type)
    _write_scalar(group, "rx_array_type", antenna.rx_array_type)
    _write_scalar(group, "tx_num_rows", np.int32(antenna.tx_num_rows))
    _write_scalar(group, "tx_num_cols", np.int32(antenna.tx_num_cols))
    _write_scalar(group, "rx_num_rows", np.int32(antenna.rx_num_rows))
    _write_scalar(group, "rx_num_cols", np.int32(antenna.rx_num_cols))
    _write_scalar(group, "tx_num_ant", np.int32(antenna.tx_num_ant))
    _write_scalar(group, "rx_num_ant", np.int32(antenna.rx_num_ant))
    _write_dataset(group, "tx_spacing_lambda", antenna.tx_spacing_lambda, unit="lambda")
    _write_dataset(group, "rx_spacing_lambda", antenna.rx_spacing_lambda, unit="lambda")
    _write_scalar(group, "tx_polarization", antenna.tx_polarization)
    _write_scalar(group, "rx_polarization", antenna.rx_polarization)
    _write_scalar(group, "tx_pattern", antenna.tx_pattern)
    _write_scalar(group, "rx_pattern", antenna.rx_pattern)
    _write_scalar(group, "synthetic_array", bool(antenna.synthetic_array))
    _write_scalar(group, "tx_orientation_mode", antenna.tx_orientation_mode)
    _write_scalar(group, "rx_orientation_mode", antenna.rx_orientation_mode)


def _write_scene(h5: h5py.File, result: MeasurementSimulationResult) -> None:
    scene = result.scene
    group = h5.require_group("scene")
    _write_scalar(group, "scene_name", scene.scene_name)
    _write_scalar(group, "scene_file", scene.scene_file)
    _write_scalar(group, "scene_id", scene.scene_id)
    _write_scalar(group, "map_id", scene.map_id)
    _write_scalar(group, "material_policy", scene.material_policy)


def _write_frequency(h5: h5py.File, result: MeasurementSimulationResult) -> None:
    frequency = result.frequency
    group = h5.require_group("frequency")
    _write_scalar(group, "center_frequency_hz", np.float64(frequency.center_frequency_hz))
    _write_scalar(group, "bandwidth_hz", np.float64(frequency.bandwidth_hz))
    _write_scalar(group, "num_subcarriers", np.int32(frequency.num_subcarriers))
    _write_scalar(group, "subcarrier_spacing_hz", np.float64(frequency.subcarrier_spacing_hz))
    _write_dataset(group, "frequencies_hz", frequency.frequencies_hz, unit="Hz")


def _write_derived(h5: h5py.File, result: MeasurementSimulationResult) -> None:
    derived = result.derived
    if derived is None:
        return
    group = h5.require_group("derived")
    link_order = "tx,rx"
    _write_dataset(
        group, "geometric_distance_m", derived.geometric_distance_m,
        unit="m", index_order=link_order,
    )
    _write_dataset(
        group, "los_distance_m", derived.los_distance_m,
        unit="m", index_order=link_order,
    )
    _write_dataset(
        group, "first_path_delay_s", derived.first_path_delay_s,
        unit="s", index_order=link_order,
    )
    _write_dataset(
        group, "strongest_path_delay_s", derived.strongest_path_delay_s,
        unit="s", index_order=link_order,
    )
    _write_dataset(
        group, "rtt_like_m", derived.rtt_like_m, unit="m", index_order=link_order,
    )
    _write_dataset(
        group, "rtt_like_s", derived.rtt_like_s, unit="s", index_order=link_order,
    )
    _write_dataset(
        group, "los_aoa_azimuth_rad", derived.los_aoa_azimuth_rad,
        unit="rad", index_order=link_order,
    )
    _write_dataset(
        group, "los_aoa_zenith_rad", derived.los_aoa_zenith_rad,
        unit="rad", index_order=link_order,
    )
    _write_dataset(
        group, "strongest_aoa_azimuth_rad", derived.strongest_aoa_azimuth_rad,
        unit="rad", index_order=link_order,
    )
    _write_dataset(
        group, "strongest_aoa_zenith_rad", derived.strongest_aoa_zenith_rad,
        unit="rad", index_order=link_order,
    )
    _write_dataset(
        group, "first_path_aoa_azimuth_rad", derived.first_path_aoa_azimuth_rad,
        unit="rad", index_order=link_order,
    )
    _write_dataset(
        group, "first_path_aoa_zenith_rad", derived.first_path_aoa_zenith_rad,
        unit="rad", index_order=link_order,
    )
    _write_dataset(group, "los_flag", derived.los_flag, index_order=link_order)
    _write_dataset(group, "nlos_flag", derived.nlos_flag, index_order=link_order)
    _write_dataset(group, "path_count", derived.path_count, index_order=link_order)
    _write_dataset(
        group, "path_power_db", derived.path_power_db, unit="dB", index_order=link_order,
    )
    _write_dataset(group, "link_valid_mask", derived.link_valid_mask, index_order=link_order)
    _write_dataset(
        group, "tx_rx_midpoint_m", derived.tx_rx_midpoint_m,
        unit="m", index_order="tx,rx,xy",
    )
    _write_dataset(
        group, "tx_rx_bearing_rad", derived.tx_rx_bearing_rad,
        unit="rad", index_order=link_order,
    )
    _write_dataset(
        group, "tx_rx_distance_m", derived.tx_rx_distance_m,
        unit="m", index_order=link_order,
    )
    _write_scalar(group, "path_selection_policy", derived.path_selection_policy)


def _write_truth(h5: h5py.File, result: MeasurementSimulationResult) -> None:
    truth = result.truth
    group = h5.require_group("channel").require_group("truth")
    _write_dataset(
        group,
        "cfr",
        truth.cfr,
        unit="linear_complex",
        index_order="tx,rx,rx_ant,tx_ant,subcarrier",
    )
    _write_dataset(group, "path_power_db", truth.path_power_db, unit="dB")
    _write_dataset(group, "has_geometric_signal", truth.has_geometric_signal)
    _write_dataset(group, "geometric_path_count", truth.geometric_path_count)
    _write_dataset(group, "los_exists", truth.los_exists)
    _write_dataset(group, "nlos_exists", truth.nlos_exists)
    if truth.cfr_snapshots is not None:
        _write_dataset(
            group,
            "cfr_snapshots",
            truth.cfr_snapshots,
            unit="linear_complex",
            index_order="snapshot,tx,rx,rx_ant,tx_ant,subcarrier",
        )


def _write_cir_truth(h5: h5py.File, result: MeasurementSimulationResult) -> None:
    cir = result.cir_truth
    if cir is None:
        return
    group = h5.require_group("channel").require_group("truth")
    _write_dataset(group, "cir_coefficients", cir.coefficients, unit="linear_complex")
    _write_dataset(group, "cir_delays_s", cir.delays_s, unit="s")
    _write_dataset(group, "cir_valid", cir.valid)


def _write_path_samples(h5: h5py.File, result: MeasurementSimulationResult) -> None:
    samples = result.path_samples
    group = h5.require_group("paths").require_group("samples")
    _write_dataset(group, "sampled_link_indices", samples.sampled_link_indices)
    _write_dataset(group, "sampled_rx_ant_indices", samples.sampled_rx_ant_indices)
    _write_dataset(group, "sampled_tx_ant_indices", samples.sampled_tx_ant_indices)
    _write_dataset(group, "sampled_path_indices", samples.sampled_path_indices)
    _write_dataset(group, "path_count", samples.path_count)
    _write_dataset(group, "path_gain_db", samples.path_gain_db, unit="dB")
    _write_string_array(group, "path_type", samples.path_type)
    _write_dataset(group, "vertices_m", samples.vertices_m, unit="m")
    _write_dataset(group, "vertex_count", samples.vertex_count)
    _write_dataset(group, "interaction_type", samples.interaction_type)
    _write_dataset(group, "object_id", samples.object_id)
    _write_dataset(group, "primitive_id", samples.primitive_id)
    _write_dataset(group, "doppler_hz", samples.doppler_hz, unit="Hz")
    _write_dataset(group, "tau_s", samples.tau_s, unit="s")


def _write_nlos_path_truth(h5: h5py.File, result: MeasurementSimulationResult) -> None:
    truth = result.nlos_path_truth
    if truth is None:
        return

    group = h5.require_group("paths").require_group("nlos_truth")
    order = "tx,rx,rx_ant,tx_ant,path"
    _write_dataset(group, "valid", truth.valid, index_order=order)
    _write_dataset(
        group, "aoa_zenith_rad", truth.aoa_zenith_rad, unit="rad", index_order=order,
    )
    _write_dataset(
        group, "aoa_azimuth_rad", truth.aoa_azimuth_rad, unit="rad", index_order=order,
    )
    _write_dataset(
        group, "aod_zenith_rad", truth.aod_zenith_rad, unit="rad", index_order=order,
    )
    _write_dataset(
        group, "aod_azimuth_rad", truth.aod_azimuth_rad, unit="rad", index_order=order,
    )
    _write_dataset(group, "path_power_db", truth.path_power_db, unit="dB", index_order=order)
    _write_dataset(group, "delay_s", truth.delay_s, unit="s", index_order=order)
    _write_dataset(group, "path_depth", truth.path_depth, index_order=order)
    _write_string_array(group, "path_type", truth.path_type)
    group["path_type"].attrs["index_order"] = order


def _write_path_full(h5: h5py.File, result: MeasurementSimulationResult) -> None:
    table = result.path_table
    if table is None:
        return

    group = h5.require_group("paths").require_group("full")
    _write_dataset(group, "valid", table.valid)
    _write_dataset(group, "a", table.a, unit="linear_complex")
    _write_dataset(group, "tau_s", table.tau_s, unit="s")
    _write_dataset(group, "doppler_hz", table.doppler_hz, unit="Hz")
    _write_dataset(group, "theta_t_rad", table.theta_t_rad, unit="rad")
    _write_dataset(group, "phi_t_rad", table.phi_t_rad, unit="rad")
    _write_dataset(group, "theta_r_rad", table.theta_r_rad, unit="rad")
    _write_dataset(group, "phi_r_rad", table.phi_r_rad, unit="rad")
    _write_dataset(group, "interaction_type", table.interaction_type)
    _write_dataset(group, "object_id", table.object_id)
    _write_dataset(group, "primitive_id", table.primitive_id)
    _write_dataset(group, "vertices_m", table.vertices_m, unit="m")
    _write_string_array(group, "path_type", table.path_type)
    _write_dataset(group, "path_depth", table.path_depth)


def _write_waveform(h5: h5py.File, result: MeasurementSimulationResult) -> None:
    waveform = result.waveform
    if waveform is None:
        return
    group = h5.require_group("waveform")
    _write_scalar(group, "standard", waveform.standard)
    _write_scalar(group, "sample_rate_hz", np.float64(waveform.sample_rate_hz))
    _write_scalar(group, "fft_size", np.int32(waveform.fft_size))
    _write_scalar(group, "cp_length", np.int32(waveform.cp_length))
    _write_scalar(group, "num_ofdm_symbols", np.int32(waveform.num_ofdm_symbols))
    _write_dataset(group, "pilot_indices", waveform.pilot_indices)
    _write_dataset(group, "data_subcarrier_indices", waveform.data_subcarrier_indices)
    _write_dataset(group, "pilot_symbols", waveform.pilot_symbols, unit="linear_complex")
    _write_scalar(group, "tx_power_dbm", np.float32(waveform.tx_power_dbm))
    # NR PUSCH extras
    extras = result.waveform_extras
    if extras:
        for key in (
            "num_prb", "subcarrier_spacing_khz", "num_layers",
            "num_antenna_ports", "mcs_index", "mcs_table",
            "dmrs_config_type", "dmrs_length", "dmrs_additional_position",
            "num_cdm_groups_without_data",
        ):
            if key in extras:
                _write_scalar(group, key, np.int32(extras[key]))
        if "subcarrier_spacing_hz" in extras:
            _write_scalar(
                group, "subcarrier_spacing_hz", np.float64(extras["subcarrier_spacing_hz"]),
            )
        if "slot_number" in extras:
            _write_scalar(group, "slot_number", np.int32(extras["slot_number"]))
        for key in ("cyclic_prefix", "target_coderate", "modulation"):
            if key in extras:
                _write_scalar(group, key, str(extras[key]))
        if waveform.standard == "nr_pusch":
            if "tx_grid" in extras:
                _write_dataset(
                    group,
                    "tx_grid",
                    extras["tx_grid"],
                    unit="linear_complex",
                    index_order="snapshot,ul_tx,ul_rx,ul_tx_ant,ofdm_symbol,subcarrier",
                )
            if "rx_grid" in extras:
                _write_dataset(
                    group,
                    "rx_grid",
                    extras["rx_grid"],
                    unit="linear_complex",
                    index_order="snapshot,ul_tx,ul_rx,ul_rx_ant,ofdm_symbol,subcarrier",
                )
            if "noise_variance" in extras:
                _write_dataset(
                    group,
                    "noise_variance",
                    extras["noise_variance"],
                    unit="linear",
                    index_order="snapshot,ul_tx,ul_rx",
                )
        if waveform.standard == "nr_srs":
            if "srs_tx_grid" in extras:
                _write_dataset(
                    group,
                    "srs_tx_grid",
                    extras["srs_tx_grid"],
                    unit="linear_complex",
                    index_order="snapshot,ul_tx,ul_rx,ul_tx_ant,ofdm_symbol,subcarrier",
                )
            if "srs_rx_grid" in extras:
                _write_dataset(
                    group,
                    "srs_rx_grid",
                    extras["srs_rx_grid"],
                    unit="linear_complex",
                    index_order="snapshot,ul_tx,ul_rx,ul_rx_ant,ofdm_symbol,subcarrier",
                )
            if "srs_noise_variance" in extras:
                _write_dataset(
                    group,
                    "srs_noise_variance",
                    extras["srs_noise_variance"],
                    unit="linear",
                    index_order="snapshot,ul_tx,ul_rx",
                )
            if "srs_pilot_code" in extras:
                _write_dataset(
                    group,
                    "srs_pilot_code",
                    extras["srs_pilot_code"],
                    unit="linear_complex",
                    index_order="ul_tx_ant,ofdm_symbol",
                )
        # TODO: export custom OFDM tx_grid/rx_grid only after that path carries
        # real generated frequency-domain waveform tensors.


def _write_array(h5: h5py.File, result: MeasurementSimulationResult) -> None:
    outputs = result.array_outputs
    if not outputs:
        return
    group = h5.require_group("array")
    if "rx_snapshot_matrix" in outputs:
        _write_dataset(
            group,
            "rx_snapshot_matrix",
            outputs["rx_snapshot_matrix"],
            unit="linear_complex",
            index_order="snapshot,ul_tx,ul_rx,ul_rx_ant,ul_rx_ant",
        )
    if "aoa_label_rad" in outputs:
        _write_dataset(
            group,
            "aoa_label_rad",
            outputs["aoa_label_rad"],
            unit="rad",
            index_order="snapshot,ul_tx,ul_rx,angle_component",
        )
    if "spatial_spectrum_label" in outputs:
        _write_dataset(
            group,
            "spatial_spectrum_label",
            outputs["spatial_spectrum_label"],
            unit="linear",
            index_order="snapshot,ul_tx,ul_rx,zenith,azimuth",
        )
    if "aoa_heatmap_label" in outputs:
        _write_dataset(
            group,
            "aoa_heatmap_label",
            outputs["aoa_heatmap_label"],
            unit="linear",
            index_order="snapshot,ul_tx,ul_rx,zenith,azimuth",
        )
    if "spatial_spectrum_truth" in outputs:
        _write_dataset(
            group,
            "spatial_spectrum_truth",
            outputs["spatial_spectrum_truth"],
            unit="linear",
            index_order="snapshot,ul_tx,ul_rx,zenith,azimuth",
        )
    if "spatial_spectrum_cfr_est" in outputs:
        _write_dataset(
            group,
            "spatial_spectrum_cfr_est",
            outputs["spatial_spectrum_cfr_est"],
            unit="linear",
            index_order="snapshot,ul_tx,ul_rx,zenith,azimuth",
        )
    if "spatial_spectrum_observation" in outputs:
        _write_dataset(
            group,
            "spatial_spectrum_observation",
            outputs["spatial_spectrum_observation"],
            unit="linear",
            index_order="snapshot,ul_tx,ul_rx,zenith,azimuth",
        )
    if "spatial_spectrum_srs" in outputs:
        _write_dataset(
            group,
            "spatial_spectrum_srs",
            outputs["spatial_spectrum_srs"],
            unit="linear",
            index_order="snapshot,ul_tx,ul_rx,zenith,azimuth",
        )
    if "angle_grid_rad" in outputs:
        _write_dataset(
            group,
            "angle_grid_rad",
            outputs["angle_grid_rad"],
            unit="rad",
            index_order="zenith,azimuth,angle_component",
        )
    if "spectrum_policy" in outputs:
        _write_scalar(group, "spectrum_policy", str(outputs["spectrum_policy"]))


def _write_observation(h5: h5py.File, result: MeasurementSimulationResult) -> None:
    observation = result.observation
    if observation is None:
        return
    group = h5.require_group("observation")
    _write_dataset(
        group,
        "cfr_est",
        observation.cfr_est,
        unit="linear_complex",
        index_order="snapshot,tx,rx,rx_ant,tx_ant,subcarrier",
    )
    if result.waveform is not None and result.waveform.standard == "nr_srs":
        _write_dataset(
            group,
            "srs_cfr_est",
            observation.cfr_est,
            unit="linear_complex",
            index_order="snapshot,tx,rx,rx_ant,tx_ant,subcarrier",
        )
    _write_dataset(group, "valid_mask", observation.valid_mask)
    _write_dataset(group, "detection_success", observation.detection_success)
    _write_dataset(group, "estimation_success", observation.estimation_success)
    _write_dataset(group, "snr_db", observation.snr_db, unit="dB")
    _write_dataset(group, "rssi_dbm", observation.rssi_dbm, unit="dBm")
    _write_dataset(group, "noise_power_dbm", observation.noise_power_dbm, unit="dBm")
    _write_dataset(group, "cfo_hz", observation.cfo_hz, unit="Hz")
    _write_dataset(group, "sfo_ppm", observation.sfo_ppm, unit="ppm")
    _write_dataset(group, "timing_offset_samples", observation.timing_offset_samples, unit="sample")
    _write_dataset(group, "phase_offset_rad", observation.phase_offset_rad, unit="rad")
    _write_dataset(group, "agc_gain_db", observation.agc_gain_db, unit="dB")
    _write_dataset(group, "clipping_flag", observation.clipping_flag)


def _write_impairments(h5: h5py.File, result: MeasurementSimulationResult) -> None:
    impairments = result.impairments
    if impairments is None:
        return
    group = h5.require_group("impairments")
    _write_scalar(group, "model_version", impairments.model_version)
    _write_scalar(group, "random_seed", np.int64(impairments.random_seed))
    _write_scalar(group, "awgn_config", impairments.awgn_config)
    _write_scalar(group, "cfo_sfo_config", impairments.cfo_sfo_config)
    _write_scalar(group, "phase_noise_config", impairments.phase_noise_config)
    _write_scalar(group, "iq_imbalance_config", impairments.iq_imbalance_config)
    _write_scalar(group, "agc_adc_config", impairments.agc_adc_config)


def _write_receiver(h5: h5py.File, result: MeasurementSimulationResult) -> None:
    receiver = result.receiver
    if receiver is None:
        return
    group = h5.require_group("receiver")
    _write_scalar(group, "receiver_type", receiver.receiver_type)
    _write_scalar(group, "estimator_type", receiver.estimator_type)
    _write_scalar(group, "sync_method", receiver.sync_method)
    _write_scalar(group, "mimo_detector", receiver.mimo_detector)
    _write_scalar(group, "input_domain", receiver.input_domain)
    _write_scalar(group, "interpolation_method", receiver.interpolation_method)
    _write_scalar(
        group,
        "packet_detection_threshold",
        np.float32(receiver.packet_detection_threshold),
    )
    _write_scalar(group, "failure_policy", receiver.failure_policy)
    _write_scalar(group, "calibration_profile_id", receiver.calibration_profile_id)


def _write_evaluation(h5: h5py.File, result: MeasurementSimulationResult) -> None:
    evaluation = result.evaluation
    if evaluation is None:
        return
    group = h5.require_group("evaluation")
    _write_dataset(group, "nmse_db", evaluation.nmse_db, unit="dB")
    _write_dataset(group, "nmse_db_total", evaluation.nmse_db_total, unit="dB")
    _write_dataset(group, "amplitude_error_db", evaluation.amplitude_error_db, unit="dB")
    _write_dataset(group, "phase_error_rad", evaluation.phase_error_rad, unit="rad")
    _write_dataset(group, "correlation", evaluation.correlation)
    _write_scalar(group, "detection_rate", np.float32(evaluation.detection_rate))
    _write_scalar(
        group,
        "estimation_failure_rate",
        np.float32(evaluation.estimation_failure_rate),
    )
    _write_scalar(group, "ber", np.float32(evaluation.ber))
    _write_scalar(group, "bler", np.float32(evaluation.bler))
    _write_scalar(group, "num_bit_errors", np.int64(evaluation.num_bit_errors))
    _write_scalar(group, "num_bits", np.int64(evaluation.num_bits))
    _write_scalar(group, "num_block_errors", np.int64(evaluation.num_block_errors))
    _write_scalar(group, "num_blocks", np.int64(evaluation.num_blocks))


def _write_calibration(h5: h5py.File, result: MeasurementSimulationResult) -> None:
    calibration = result.calibration
    if calibration is None:
        return
    group = h5.require_group("calibration")
    _write_scalar(group, "profile_id", calibration.profile_id)
    _write_scalar(group, "fitted_parameters", calibration.fitted_parameters)
    _write_scalar(group, "validation_metrics", calibration.validation_metrics)


def _write_motion(h5: h5py.File, result: MeasurementSimulationResult) -> None:
    motion = result.motion
    if motion is None:
        return
    group = h5.require_group("motion")
    _write_dataset(group, "snapshot_id", motion.snapshot_id)
    _write_dataset(group, "timestamp_s", motion.timestamp_s, unit="s")
    _write_scalar(group, "sampling_frequency_hz", np.float64(motion.sampling_frequency_hz))
    _write_scalar(group, "num_time_steps", np.int32(motion.num_time_steps))
    _write_scalar(group, "mobility_mode", motion.mobility_mode)


def _write_link(h5: h5py.File, result: MeasurementSimulationResult) -> None:
    link = result.link
    if link is None:
        return
    group = h5.require_group("link")
    _write_scalar(group, "duplex_mode", link.duplex_mode)
    _write_scalar(group, "phy_link_direction", link.phy_link_direction)
    _write_scalar(group, "rt_trace_direction", link.rt_trace_direction)
    _write_scalar(group, "reciprocity_mode", link.reciprocity_mode)
    _write_scalar(group, "reciprocity_applied", bool(link.reciprocity_applied))


def _write_runtime(h5: h5py.File, result: MeasurementSimulationResult) -> None:
    runtime = result.runtime
    group = h5.require_group("runtime")
    _write_scalar(group, "python_version", runtime.python_version)
    _write_scalar(group, "sionna_version", runtime.sionna_version)
    _write_scalar(group, "sionna_rt_version", runtime.sionna_rt_version)
    _write_scalar(group, "torch_version", runtime.torch_version)
    _write_scalar(group, "mitsuba_version", runtime.mitsuba_version)
    _write_scalar(group, "drjit_version", runtime.drjit_version)
    _write_scalar(group, "cuda_available", bool(runtime.cuda_available))
    _write_scalar(group, "cuda_device_name", runtime.cuda_device_name)
    _write_scalar(group, "command_line", runtime.command_line)
    _write_scalar(group, "elapsed_seconds", np.float64(runtime.elapsed_seconds))


def _write_scalar(group: h5py.Group, name: str, value: Any) -> h5py.Dataset:
    if isinstance(value, str):
        return group.create_dataset(name, data=value, dtype=UTF8_DTYPE)
    return group.create_dataset(name, data=value)


def _write_string_array(group: h5py.Group, name: str, values: Any) -> h5py.Dataset:
    array = np.asarray(values, dtype=object)
    return group.create_dataset(name, data=array, dtype=UTF8_DTYPE)


def _write_dataset(
    group: h5py.Group,
    name: str,
    value: Any,
    *,
    unit: str | None = None,
    index_order: str | None = None,
) -> h5py.Dataset:
    array = np.asarray(value)
    kwargs: dict[str, Any] = {}
    compression = _ACTIVE_COMPRESSION.get()
    if array.ndim > 0 and array.size > 0 and compression != "none":
        kwargs["compression"] = compression
        kwargs["shuffle"] = True

    start = time.perf_counter()
    dataset = group.create_dataset(name, data=array, **kwargs)
    duration_s = time.perf_counter() - start
    if unit is not None:
        dataset.attrs["unit"] = unit
    if index_order is not None:
        dataset.attrs["index_order"] = index_order
    _record_dataset_write(dataset, array, duration_s, compression)
    return dataset


def _record_dataset_write(
    dataset: h5py.Dataset,
    array: np.ndarray,
    duration_s: float,
    compression: str,
) -> None:
    tracer = _ACTIVE_TRACER.get()
    if tracer is None:
        return
    try:
        storage_bytes = int(dataset.id.get_storage_size())
    except Exception:
        storage_bytes = -1
    tracer.record_event(
        "hdf5.dataset_write",
        path=str(dataset.name),
        shape=tuple(int(dim) for dim in array.shape),
        dtype=str(array.dtype),
        raw_bytes=int(array.nbytes),
        storage_bytes=storage_bytes,
        compression=compression,
        duration_s=float(duration_s),
    )
