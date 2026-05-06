"""HDF5 writer for domain results.

The writer consumes only domain models. It must not import or inspect Sionna
native objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np

from sionna_measurement_sim.domain.results import MeasurementSimulationResult

UTF8_DTYPE = h5py.string_dtype(encoding="utf-8")


def write_measurement_result(path: str | Path, result: MeasurementSimulationResult) -> Path:
    """Write a truth-only result to an HDF5 file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as h5:
        _write_meta(h5, result)
        _write_input(h5, result)
        _write_topology(h5, result)
        _write_devices(h5, result)
        _write_antenna(h5, result)
        _write_scene(h5, result)
        _write_frequency(h5, result)
        _write_truth(h5, result)
        _write_path_samples(h5, result)
        _write_runtime(h5, result)

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


def _write_scene(h5: h5py.File, result: MeasurementSimulationResult) -> None:
    scene = result.scene
    group = h5.require_group("scene")
    _write_scalar(group, "scene_name", scene.scene_name)
    _write_scalar(group, "scene_file", scene.scene_file)
    _write_scalar(group, "material_policy", scene.material_policy)


def _write_frequency(h5: h5py.File, result: MeasurementSimulationResult) -> None:
    frequency = result.frequency
    group = h5.require_group("frequency")
    _write_scalar(group, "center_frequency_hz", np.float64(frequency.center_frequency_hz))
    _write_scalar(group, "bandwidth_hz", np.float64(frequency.bandwidth_hz))
    _write_scalar(group, "num_subcarriers", np.int32(frequency.num_subcarriers))
    _write_scalar(group, "subcarrier_spacing_hz", np.float64(frequency.subcarrier_spacing_hz))
    _write_dataset(group, "frequencies_hz", frequency.frequencies_hz, unit="Hz")


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


def _write_path_samples(h5: h5py.File, result: MeasurementSimulationResult) -> None:
    samples = result.path_samples
    group = h5.require_group("paths").require_group("samples")
    _write_dataset(group, "sampled_link_indices", samples.sampled_link_indices)
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
    if array.ndim > 0 and array.size > 0:
        kwargs["compression"] = "gzip"
        kwargs["shuffle"] = True

    dataset = group.create_dataset(name, data=array, **kwargs)
    if unit is not None:
        dataset.attrs["unit"] = unit
    if index_order is not None:
        dataset.attrs["index_order"] = index_order
    return dataset
