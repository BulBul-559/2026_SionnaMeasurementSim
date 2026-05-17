"""Convert Sionna RT Paths objects into domain path models."""

from __future__ import annotations

from typing import Any

import numpy as np

from sionna_measurement_sim.adapters.sionna_rt.shape_contracts import (
    to_project_path_interaction,
    to_project_path_scalar,
    to_project_path_vertices,
)
from sionna_measurement_sim.domain.path import PathSamples, PathTable
from sionna_measurement_sim.domain.topology import Topology

INTERACTION_NONE = 0
INTERACTION_SPECULAR = 1
INTERACTION_DIFFUSE = 2
INTERACTION_REFRACTION = 4
INTERACTION_DIFFRACTION = 8


def paths_to_table(paths: Any) -> tuple[PathTable, np.ndarray, np.ndarray, np.ndarray]:
    """Convert Sionna paths to the project TX-first PathTable contract.

    Returns (path_table, geometric_path_count, los_exists, nlos_exists).
    geometric_path_count: int32 [tx, rx] — number of valid geometric paths per link.
    los_exists: bool [tx, rx] — at least one LoS path exists.
    nlos_exists: bool [tx, rx] — at least one NLoS path exists.
    """

    valid = _path_scalar_to_tx_first(_to_numpy(paths.valid), "valid").astype(np.bool_)
    a = _complex_path_coefficients(_to_numpy(paths.a))
    tau_s = _path_scalar_to_tx_first(_to_numpy(paths.tau), "tau").astype(np.float32)
    doppler_hz = _path_scalar_to_tx_first(_to_numpy(paths.doppler), "doppler").astype(np.float32)
    theta_t_rad = _path_scalar_to_tx_first(_to_numpy(paths.theta_t), "theta_t").astype(np.float32)
    phi_t_rad = _path_scalar_to_tx_first(_to_numpy(paths.phi_t), "phi_t").astype(np.float32)
    theta_r_rad = _path_scalar_to_tx_first(_to_numpy(paths.theta_r), "theta_r").astype(np.float32)
    phi_r_rad = _path_scalar_to_tx_first(_to_numpy(paths.phi_r), "phi_r").astype(np.float32)

    interaction_type = _interaction_to_tx_first(_to_numpy(paths.interactions), "interactions")
    object_id = _interaction_to_tx_first(_to_numpy(paths.objects), "objects")
    primitive_id = _interaction_to_tx_first(_to_numpy(paths.primitives), "primitives")
    vertices_m = _vertices_to_tx_first(_to_numpy(paths.vertices))
    path_depth = np.count_nonzero(interaction_type != INTERACTION_NONE, axis=-1).astype(np.int32)
    path_type = _classify_path_types(interaction_type)
    (
        valid,
        tau_s,
        doppler_hz,
        theta_t_rad,
        phi_t_rad,
        theta_r_rad,
        phi_r_rad,
        interaction_type,
        object_id,
        primitive_id,
        vertices_m,
        path_type,
        path_depth,
    ) = _broadcast_path_metadata_to_coefficients(
        a.shape,
        valid,
        tau_s,
        doppler_hz,
        theta_t_rad,
        phi_t_rad,
        theta_r_rad,
        phi_r_rad,
        interaction_type,
        object_id,
        primitive_id,
        vertices_m,
        path_type,
        path_depth,
    )

    path_table = PathTable(
        valid=valid,
        a=a,
        tau_s=tau_s,
        doppler_hz=doppler_hz,
        theta_t_rad=theta_t_rad,
        phi_t_rad=phi_t_rad,
        theta_r_rad=theta_r_rad,
        phi_r_rad=phi_r_rad,
        interaction_type=interaction_type,
        object_id=object_id,
        primitive_id=primitive_id,
        vertices_m=vertices_m,
        path_type=path_type,
        path_depth=path_depth,
    )

    # Collapse antenna dims: any antenna pair with a valid path means the link is active
    tx, rx, rx_ant, tx_ant, path_count = valid.shape
    link_valid = np.any(valid, axis=(2, 3))  # [tx, rx, path_count]

    geometric_path_count = np.count_nonzero(link_valid, axis=-1).astype(np.int32)
    # Aggregate over ALL antenna pairs (not just index 0) for los/nlos detection
    los_exists = np.any(valid & (path_type == "los"), axis=(2, 3, 4))  # [tx, rx]
    nlos_exists = np.any(valid & (path_type != "los"), axis=(2, 3, 4))  # [tx, rx]

    return path_table, geometric_path_count, los_exists, nlos_exists


def path_table_to_samples(
    table: PathTable,
    topology: Topology,
    *,
    max_paths_per_link: int | None = None,
) -> PathSamples:
    """Build lightweight path samples with TX/interactions/RX vertices."""

    tx_count, rx_count, rx_ant_count, tx_ant_count, path_count = table.valid.shape
    depth = table.interaction_type.shape[-1]
    sample_count = tx_count * rx_count * rx_ant_count * tx_ant_count
    max_vertices = depth + 2
    selected_paths = max_paths_per_link or path_count

    sampled_link_indices = np.zeros((sample_count, 2), dtype=np.int32)
    sampled_rx_ant_indices = np.zeros((sample_count,), dtype=np.int32)
    sampled_tx_ant_indices = np.zeros((sample_count,), dtype=np.int32)
    sampled_path_indices = np.full((sample_count, selected_paths), -1, dtype=np.int32)
    sample_path_count = np.zeros((sample_count,), dtype=np.int32)
    path_gain_db = np.zeros((sample_count, selected_paths), dtype=np.float32)
    path_type = np.empty((sample_count, selected_paths), dtype=object)
    path_type[:, :] = "invalid"
    vertices_m = np.zeros((sample_count, selected_paths, max_vertices, 3), dtype=np.float32)
    vertex_count = np.zeros((sample_count, selected_paths), dtype=np.int32)
    interaction_type = np.zeros((sample_count, selected_paths, depth), dtype=np.uint32)
    object_id = np.zeros((sample_count, selected_paths, depth), dtype=np.uint32)
    primitive_id = np.zeros((sample_count, selected_paths, depth), dtype=np.uint32)
    doppler_hz = np.zeros((sample_count, selected_paths), dtype=np.float32)
    tau_s = np.zeros((sample_count, selected_paths), dtype=np.float32)

    sample = 0
    for tx in range(tx_count):
        for rx in range(rx_count):
            for rx_ant in range(rx_ant_count):
                for tx_ant in range(tx_ant_count):
                    sampled_link_indices[sample] = [tx, rx]
                    sampled_rx_ant_indices[sample] = rx_ant
                    sampled_tx_ant_indices[sample] = tx_ant
                    v = np.flatnonzero(
                        table.valid[tx, rx, rx_ant, tx_ant]
                    )[:selected_paths]
                    sample_path_count[sample] = len(v)
                    for oi, pi in enumerate(v):
                        sampled_path_indices[sample, oi] = pi
                        a = table.a[tx, rx, rx_ant, tx_ant, pi]
                        path_gain_db[sample, oi] = _path_gain_db(a)
                        path_type[sample, oi] = table.path_type[tx, rx, rx_ant, tx_ant, pi]
                        ic = int(table.path_depth[tx, rx, rx_ant, tx_ant, pi])
                        vc = ic + 2
                        vertex_count[sample, oi] = vc
                        vertices_m[sample, oi, 0] = topology.tx_positions_m[tx]
                        if ic:
                            vertices_m[sample, oi, 1:1 + ic] = table.vertices_m[
                                tx, rx, rx_ant, tx_ant, pi, :ic
                            ]
                        vertices_m[sample, oi, ic + 1] = topology.rx_positions_m[rx]
                        interaction_type[sample, oi] = table.interaction_type[
                            tx, rx, rx_ant, tx_ant, pi
                        ]
                        object_id[sample, oi] = table.object_id[tx, rx, rx_ant, tx_ant, pi]
                        primitive_id[sample, oi] = table.primitive_id[tx, rx, rx_ant, tx_ant, pi]
                        doppler_hz[sample, oi] = table.doppler_hz[tx, rx, rx_ant, tx_ant, pi]
                        tau_s[sample, oi] = table.tau_s[tx, rx, rx_ant, tx_ant, pi]
                    sample += 1

    return PathSamples(
        sampled_link_indices=sampled_link_indices,
        sampled_rx_ant_indices=sampled_rx_ant_indices,
        sampled_tx_ant_indices=sampled_tx_ant_indices,
        sampled_path_indices=sampled_path_indices,
        path_count=sample_path_count,
        path_gain_db=path_gain_db,
        path_type=path_type,
        vertices_m=vertices_m,
        vertex_count=vertex_count,
        interaction_type=interaction_type,
        object_id=object_id,
        primitive_id=primitive_id,
        doppler_hz=doppler_hz,
        tau_s=tau_s,
    )


def _to_numpy(value: Any) -> np.ndarray:
    return np.asarray(value)


def _complex_path_coefficients(value: np.ndarray) -> np.ndarray:
    if value.ndim == 6 and value.shape[0] == 2:
        complex_value = value[0] + 1j * value[1]
    elif np.iscomplexobj(value):
        complex_value = value
    else:
        msg = f"Unsupported paths.a shape: {value.shape}"
        raise ValueError(msg)
    return _path_scalar_to_tx_first(complex_value, "a").astype(np.complex64)


def _path_scalar_to_tx_first(value: np.ndarray, name: str) -> np.ndarray:
    """Convert Sionna path scalar to TX-first 5D.

    Delegates to :func:`shape_contracts.to_project_path_scalar`.
    """
    return to_project_path_scalar(value, name)


def _interaction_to_tx_first(value: np.ndarray, name: str) -> np.ndarray:
    """Convert Sionna interaction data to TX-first 6D.

    Delegates to :func:`shape_contracts.to_project_path_interaction`.
    """
    return to_project_path_interaction(value, name)


def _vertices_to_tx_first(value: np.ndarray) -> np.ndarray:
    """Convert Sionna vertices to TX-first 7D.

    Delegates to :func:`shape_contracts.to_project_path_vertices`.
    """
    return to_project_path_vertices(value)


def _classify_path_types(interaction_type: np.ndarray) -> np.ndarray:
    output = np.empty(interaction_type.shape[:-1], dtype=object)
    for index in np.ndindex(output.shape):
        nonzero = set(int(v) for v in interaction_type[index] if int(v) != INTERACTION_NONE)
        if not nonzero:
            output[index] = "los"
        elif len(nonzero) > 1:
            output[index] = "mixed"
        elif INTERACTION_SPECULAR in nonzero:
            output[index] = "reflection"
        elif INTERACTION_DIFFUSE in nonzero:
            output[index] = "diffuse"
        elif INTERACTION_REFRACTION in nonzero:
            output[index] = "refraction"
        elif INTERACTION_DIFFRACTION in nonzero:
            output[index] = "diffraction"
        else:
            output[index] = "unknown"
    return output


def _broadcast_path_metadata_to_coefficients(
    coefficient_shape: tuple[int, int, int, int, int],
    valid: np.ndarray,
    tau_s: np.ndarray,
    doppler_hz: np.ndarray,
    theta_t_rad: np.ndarray,
    phi_t_rad: np.ndarray,
    theta_r_rad: np.ndarray,
    phi_r_rad: np.ndarray,
    interaction_type: np.ndarray,
    object_id: np.ndarray,
    primitive_id: np.ndarray,
    vertices_m: np.ndarray,
    path_type: np.ndarray,
    path_depth: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Broadcast synthetic-array path metadata to element-level coefficients."""

    scalar_arrays = (
        valid,
        tau_s,
        doppler_hz,
        theta_t_rad,
        phi_t_rad,
        theta_r_rad,
        phi_r_rad,
        path_type,
        path_depth,
    )
    broadcast_scalars = tuple(
        np.broadcast_to(array, coefficient_shape).copy()
        if array.shape != coefficient_shape
        else array
        for array in scalar_arrays
    )
    depth = interaction_type.shape[-1]
    interaction_shape = (*coefficient_shape, depth)
    broadcast_interactions = tuple(
        np.broadcast_to(array, interaction_shape).copy()
        if array.shape != interaction_shape
        else array
        for array in (interaction_type, object_id, primitive_id)
    )
    vertex_shape = (*coefficient_shape, depth, 3)
    if vertices_m.shape != vertex_shape:
        vertices_m = np.broadcast_to(vertices_m, vertex_shape).copy()
    return (*broadcast_scalars[:7], *broadcast_interactions, vertices_m, *broadcast_scalars[7:])


def _path_gain_db(value: np.complex64) -> np.float32:
    return np.float32(20.0 * np.log10(max(float(np.abs(value)), 1e-30)))
