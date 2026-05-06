"""Path sample domain model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sionna_measurement_sim.domain.validation import require_finite, require_shape


@dataclass(frozen=True)
class PathSamples:
    """Lightweight path samples for HDF5 `/paths/samples`."""

    sampled_link_indices: np.ndarray
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
