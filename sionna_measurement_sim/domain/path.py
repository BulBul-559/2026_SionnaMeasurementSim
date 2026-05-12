"""Path domain models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sionna_measurement_sim.domain.validation import require_finite, require_shape


@dataclass(frozen=True)
class PathTable:
    """Full TX-first path table converted from Sionna RT paths."""

    valid: np.ndarray
    a: np.ndarray
    tau_s: np.ndarray
    doppler_hz: np.ndarray
    theta_t_rad: np.ndarray
    phi_t_rad: np.ndarray
    theta_r_rad: np.ndarray
    phi_r_rad: np.ndarray
    interaction_type: np.ndarray
    object_id: np.ndarray
    primitive_id: np.ndarray
    vertices_m: np.ndarray
    path_type: np.ndarray
    path_depth: np.ndarray

    def __post_init__(self) -> None:
        valid = np.asarray(self.valid, dtype=np.bool_)
        a = np.asarray(self.a, dtype=np.complex64)
        tau_s = np.asarray(self.tau_s, dtype=np.float32)
        doppler_hz = np.asarray(self.doppler_hz, dtype=np.float32)
        theta_t_rad = np.asarray(self.theta_t_rad, dtype=np.float32)
        phi_t_rad = np.asarray(self.phi_t_rad, dtype=np.float32)
        theta_r_rad = np.asarray(self.theta_r_rad, dtype=np.float32)
        phi_r_rad = np.asarray(self.phi_r_rad, dtype=np.float32)
        interaction_type = np.asarray(self.interaction_type, dtype=np.uint32)
        object_id = np.asarray(self.object_id, dtype=np.uint32)
        primitive_id = np.asarray(self.primitive_id, dtype=np.uint32)
        vertices_m = np.asarray(self.vertices_m, dtype=np.float32)
        path_type = np.asarray(self.path_type, dtype=object)
        path_depth = np.asarray(self.path_depth, dtype=np.int32)

        require_shape("valid", valid, (None, None, None, None, None))
        scalar_shape = valid.shape
        for name, value in (
            ("a", a),
            ("tau_s", tau_s),
            ("doppler_hz", doppler_hz),
            ("theta_t_rad", theta_t_rad),
            ("phi_t_rad", phi_t_rad),
            ("theta_r_rad", theta_r_rad),
            ("phi_r_rad", phi_r_rad),
            ("path_type", path_type),
            ("path_depth", path_depth),
        ):
            require_shape(name, value, scalar_shape)

        require_shape("interaction_type", interaction_type, (*scalar_shape, None))
        depth = interaction_type.shape[-1]
        require_shape("object_id", object_id, (*scalar_shape, depth))
        require_shape("primitive_id", primitive_id, (*scalar_shape, depth))
        require_shape("vertices_m", vertices_m, (*scalar_shape, depth, 3))

        require_finite("a", a)
        require_finite("tau_s", tau_s)
        require_finite("doppler_hz", doppler_hz)
        require_finite("theta_t_rad", theta_t_rad)
        require_finite("phi_t_rad", phi_t_rad)
        require_finite("theta_r_rad", theta_r_rad)
        require_finite("phi_r_rad", phi_r_rad)
        require_finite("vertices_m", vertices_m)

        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "a", a)
        object.__setattr__(self, "tau_s", tau_s)
        object.__setattr__(self, "doppler_hz", doppler_hz)
        object.__setattr__(self, "theta_t_rad", theta_t_rad)
        object.__setattr__(self, "phi_t_rad", phi_t_rad)
        object.__setattr__(self, "theta_r_rad", theta_r_rad)
        object.__setattr__(self, "phi_r_rad", phi_r_rad)
        object.__setattr__(self, "interaction_type", interaction_type)
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "primitive_id", primitive_id)
        object.__setattr__(self, "vertices_m", vertices_m)
        object.__setattr__(self, "path_type", path_type)
        object.__setattr__(self, "path_depth", path_depth)


@dataclass(frozen=True)
class NLoSPathTruth:
    """Default lightweight NLoS path AoA/AoD truth export."""

    valid: np.ndarray
    aoa_zenith_rad: np.ndarray
    aoa_azimuth_rad: np.ndarray
    aod_zenith_rad: np.ndarray
    aod_azimuth_rad: np.ndarray
    path_power_db: np.ndarray
    delay_s: np.ndarray
    path_depth: np.ndarray
    path_type: np.ndarray

    def __post_init__(self) -> None:
        valid = np.asarray(self.valid, dtype=np.bool_)
        require_shape("valid", valid, (None, None, None, None, None))
        scalar_shape = valid.shape
        for name in (
            "aoa_zenith_rad",
            "aoa_azimuth_rad",
            "aod_zenith_rad",
            "aod_azimuth_rad",
            "path_power_db",
            "delay_s",
        ):
            value = np.asarray(getattr(self, name), dtype=np.float32)
            require_shape(name, value, scalar_shape)
            object.__setattr__(self, name, value)
        path_depth = np.asarray(self.path_depth, dtype=np.int32)
        path_type = np.asarray(self.path_type, dtype=object)
        require_shape("path_depth", path_depth, scalar_shape)
        require_shape("path_type", path_type, scalar_shape)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "path_depth", path_depth)
        object.__setattr__(self, "path_type", path_type)

    @classmethod
    def empty(cls, num_tx: int, num_rx: int, num_rx_ant: int, num_tx_ant: int) -> NLoSPathTruth:
        shape = (num_tx, num_rx, num_rx_ant, num_tx_ant, 0)
        return cls(
            valid=np.zeros(shape, dtype=np.bool_),
            aoa_zenith_rad=np.zeros(shape, dtype=np.float32),
            aoa_azimuth_rad=np.zeros(shape, dtype=np.float32),
            aod_zenith_rad=np.zeros(shape, dtype=np.float32),
            aod_azimuth_rad=np.zeros(shape, dtype=np.float32),
            path_power_db=np.zeros(shape, dtype=np.float32),
            delay_s=np.zeros(shape, dtype=np.float32),
            path_depth=np.zeros(shape, dtype=np.int32),
            path_type=np.empty(shape, dtype=object),
        )


def build_nlos_path_truth(table: PathTable) -> NLoSPathTruth:
    """Build default NLoS AoA/AoD truth from a full path table."""

    valid = table.valid & (table.path_type != "los")
    float_shape = table.valid.shape
    path_type = np.empty(float_shape, dtype=object)
    path_type[:, :, :, :, :] = "invalid"
    path_type[valid] = table.path_type[valid]

    def masked_float(values: np.ndarray) -> np.ndarray:
        out = np.full(float_shape, np.nan, dtype=np.float32)
        out[valid] = np.asarray(values, dtype=np.float32)[valid]
        return out

    power = np.full(float_shape, np.nan, dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        power[valid] = 10.0 * np.log10(np.maximum(np.abs(table.a[valid]) ** 2, 1e-30))

    path_depth = np.zeros(float_shape, dtype=np.int32)
    path_depth[valid] = table.path_depth[valid]

    return NLoSPathTruth(
        valid=valid,
        aoa_zenith_rad=masked_float(table.theta_r_rad),
        aoa_azimuth_rad=masked_float(table.phi_r_rad),
        aod_zenith_rad=masked_float(table.theta_t_rad),
        aod_azimuth_rad=masked_float(table.phi_t_rad),
        path_power_db=power,
        delay_s=masked_float(table.tau_s),
        path_depth=path_depth,
        path_type=path_type,
    )


@dataclass(frozen=True)
class PathSamples:
    """Lightweight path samples for HDF5 `/paths/samples`."""

    sampled_link_indices: np.ndarray  # [sample, 2] = [tx, rx]
    sampled_rx_ant_indices: np.ndarray  # [sample] rx antenna index
    sampled_tx_ant_indices: np.ndarray  # [sample] tx antenna index
    sampled_path_indices: np.ndarray
    path_count: np.ndarray
    path_gain_db: np.ndarray
    path_type: np.ndarray
    vertices_m: np.ndarray
    vertex_count: np.ndarray
    interaction_type: np.ndarray
    object_id: np.ndarray
    primitive_id: np.ndarray
    doppler_hz: np.ndarray
    tau_s: np.ndarray

    def __post_init__(self) -> None:
        sampled_link_indices = np.asarray(self.sampled_link_indices, dtype=np.int32)
        sampled_rx_ant = np.asarray(self.sampled_rx_ant_indices, dtype=np.int32)
        sampled_tx_ant = np.asarray(self.sampled_tx_ant_indices, dtype=np.int32)
        sampled_path_indices = np.asarray(self.sampled_path_indices, dtype=np.int32)
        path_count = np.asarray(self.path_count, dtype=np.int32)
        path_gain_db = np.asarray(self.path_gain_db, dtype=np.float32)
        path_type = np.asarray(self.path_type, dtype=object)
        vertices_m = np.asarray(self.vertices_m, dtype=np.float32)
        vertex_count = np.asarray(self.vertex_count, dtype=np.int32)
        interaction_type = np.asarray(self.interaction_type, dtype=np.uint32)
        object_id = np.asarray(self.object_id, dtype=np.uint32)
        primitive_id = np.asarray(self.primitive_id, dtype=np.uint32)
        doppler_hz = np.asarray(self.doppler_hz, dtype=np.float32)
        tau_s = np.asarray(self.tau_s, dtype=np.float32)

        require_shape("sampled_link_indices", sampled_link_indices, (None, 2))
        sample_count = sampled_link_indices.shape[0]
        require_shape("sampled_rx_ant_indices", sampled_rx_ant, (sample_count,))
        require_shape("sampled_tx_ant_indices", sampled_tx_ant, (sample_count,))
        require_shape("sampled_path_indices", sampled_path_indices, (sample_count, None))
        sample_path_count = sampled_path_indices.shape[1]
        require_shape("path_count", path_count, (sample_count,))
        require_shape("path_gain_db", path_gain_db, (sample_count, sample_path_count))
        require_shape("path_type", path_type, (sample_count, sample_path_count))
        require_shape("vertices_m", vertices_m, (sample_count, sample_path_count, None, 3))
        max_vertices = vertices_m.shape[2]
        require_shape("vertex_count", vertex_count, (sample_count, sample_path_count))
        require_shape("interaction_type", interaction_type, (sample_count, sample_path_count, None))
        max_depth = interaction_type.shape[2]
        require_shape("object_id", object_id, (sample_count, sample_path_count, max_depth))
        require_shape("primitive_id", primitive_id, (sample_count, sample_path_count, max_depth))
        require_shape("doppler_hz", doppler_hz, (sample_count, sample_path_count))
        require_shape("tau_s", tau_s, (sample_count, sample_path_count))
        require_finite("path_gain_db", path_gain_db)
        require_finite("vertices_m", vertices_m)
        require_finite("doppler_hz", doppler_hz)
        require_finite("tau_s", tau_s)

        if np.any(vertex_count > max_vertices):
            msg = "vertex_count cannot exceed vertices_m max_vertices dimension"
            raise ValueError(msg)
        if np.any(path_count > sample_path_count):
            msg = "path_count cannot exceed sampled_path_indices sample_path dimension"
            raise ValueError(msg)

        object.__setattr__(self, "sampled_link_indices", sampled_link_indices)
        object.__setattr__(self, "sampled_rx_ant_indices", sampled_rx_ant)
        object.__setattr__(self, "sampled_tx_ant_indices", sampled_tx_ant)
        object.__setattr__(self, "sampled_path_indices", sampled_path_indices)
        object.__setattr__(self, "path_count", path_count)
        object.__setattr__(self, "path_gain_db", path_gain_db)
        object.__setattr__(self, "path_type", path_type)
        object.__setattr__(self, "vertices_m", vertices_m)
        object.__setattr__(self, "vertex_count", vertex_count)
        object.__setattr__(self, "interaction_type", interaction_type)
        object.__setattr__(self, "object_id", object_id)
        object.__setattr__(self, "primitive_id", primitive_id)
        object.__setattr__(self, "doppler_hz", doppler_hz)
        object.__setattr__(self, "tau_s", tau_s)

    @classmethod
    def empty(cls) -> PathSamples:
        """Create valid empty path samples for Phase 1 no-RT files."""

        return cls(
            sampled_link_indices=np.zeros((0, 2), dtype=np.int32),
            sampled_rx_ant_indices=np.zeros((0,), dtype=np.int32),
            sampled_tx_ant_indices=np.zeros((0,), dtype=np.int32),
            sampled_path_indices=np.zeros((0, 0), dtype=np.int32),
            path_count=np.zeros((0,), dtype=np.int32),
            path_gain_db=np.zeros((0, 0), dtype=np.float32),
            path_type=np.empty((0, 0), dtype=object),
            vertices_m=np.zeros((0, 0, 0, 3), dtype=np.float32),
            vertex_count=np.zeros((0, 0), dtype=np.int32),
            interaction_type=np.zeros((0, 0, 0), dtype=np.uint32),
            object_id=np.zeros((0, 0, 0), dtype=np.uint32),
            primitive_id=np.zeros((0, 0, 0), dtype=np.uint32),
            doppler_hz=np.zeros((0, 0), dtype=np.float32),
            tau_s=np.zeros((0, 0), dtype=np.float32),
        )
