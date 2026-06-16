"""Spatial spectrum helpers for array observations."""

from __future__ import annotations

import numpy as np

from sionna_measurement_sim.domain.array import ArraySpectrumConfig


def build_angle_grid_rad(config: ArraySpectrumConfig) -> np.ndarray:
    """Build [zenith, azimuth, angle_component] grid in radians."""

    zenith = np.linspace(
        config.zenith_min_rad,
        config.zenith_max_rad,
        config.zenith_bins,
        dtype=np.float32,
    )
    azimuth = np.linspace(
        config.azimuth_min_rad,
        config.azimuth_max_rad,
        config.azimuth_bins,
        dtype=np.float32,
    )
    zz, aa = np.meshgrid(zenith, azimuth, indexing="ij")
    return np.stack((zz, aa), axis=-1).astype(np.float32, copy=False)


def build_aoa_heatmap_label(
    aoa_label_rad: np.ndarray | None,
    angle_grid_rad: np.ndarray,
    link_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Build nearest-bin AoA heatmap labels and sanitized AoA labels."""

    labels = np.zeros((*link_shape, 2), dtype=np.float32)
    spectrum = np.zeros((*link_shape, *angle_grid_rad.shape[:2]), dtype=np.float32)
    if aoa_label_rad is None:
        return labels, spectrum

    aoa = np.asarray(aoa_label_rad, dtype=np.float32)
    if aoa.shape != labels.shape:
        raise ValueError(f"aoa_label_rad must have shape {labels.shape}, got {aoa.shape}")
    labels[...] = np.nan_to_num(aoa, nan=0.0)

    zenith_grid = angle_grid_rad[:, 0, 0]
    azimuth_grid = angle_grid_rad[0, :, 1]
    for index in np.ndindex(link_shape):
        zenith, azimuth = aoa[index]
        if not (np.isfinite(zenith) and np.isfinite(azimuth)):
            continue
        zenith_idx = int(np.argmin(np.abs(zenith_grid - zenith)))
        azimuth_idx = int(np.argmin(np.abs(azimuth_grid - azimuth)))
        spectrum[index + (zenith_idx, azimuth_idx)] = 1.0
    return labels, spectrum


def build_rx_snapshot_matrix(samples: np.ndarray) -> np.ndarray:
    """Build per-link RX covariance matrices from array samples."""

    array = np.asarray(samples, dtype=np.complex64)
    if array.ndim < 5:
        raise ValueError(f"samples must have rank >= 5, got {array.shape}")
    return _covariance_from_samples(array)


def project_cfr_to_ul_receiver_samples(cfr: np.ndarray) -> np.ndarray:
    """Convert link-view CFR tensors to receiver-array samples.

    Truth CFR shape:
    [tx, rx, rx_ant, tx_ant, subcarrier]
    -> [snapshot=1, tx, rx, rx_ant, tx_ant, subcarrier]

    Observation CFR shape:
    [snapshot, tx, rx, rx_ant, tx_ant, subcarrier]
    -> unchanged.
    """

    array = np.asarray(cfr, dtype=np.complex64)
    if array.ndim == 5:
        return array[np.newaxis, ...]
    if array.ndim == 6:
        return array
    raise ValueError(f"cfr must have rank 5 or 6, got {array.shape}")


def build_bartlett_spectrum(
    samples: np.ndarray,
    *,
    rx_num_rows: int,
    rx_num_cols: int,
    rx_spacing_lambda: tuple[float, float],
    config: ArraySpectrumConfig,
    rx_orientation_rad: np.ndarray | None = None,
) -> np.ndarray:
    """Build Bartlett spatial spectra from receiver-array samples.

    samples shape: [snapshot, tx, rx, rx_ant, ...].

    The angle grid is expressed in the scene/global frame. If
    ``rx_orientation_rad`` is provided, the receiver array element positions are
    rotated into that scene frame before building steering vectors.
    """

    array = np.asarray(samples, dtype=np.complex64)
    if array.ndim < 5:
        raise ValueError(f"samples must have rank >= 5, got {array.shape}")
    num_rx_ant = int(rx_num_rows) * int(rx_num_cols)
    if array.shape[3] != num_rx_ant:
        raise ValueError(
            f"samples rx_ant dimension {array.shape[3]} does not match "
            f"rx_num_rows*rx_num_cols {num_rx_ant}"
        )

    covariance = _covariance_from_samples(array, link_chunk_size=config.link_chunk_size)
    return build_bartlett_spectrum_from_covariance(
        covariance,
        rx_num_rows=rx_num_rows,
        rx_num_cols=rx_num_cols,
        rx_spacing_lambda=rx_spacing_lambda,
        config=config,
        rx_orientation_rad=rx_orientation_rad,
    )


def build_bartlett_spectrum_from_covariance(
    covariance: np.ndarray,
    *,
    rx_num_rows: int,
    rx_num_cols: int,
    rx_spacing_lambda: tuple[float, float],
    config: ArraySpectrumConfig,
    rx_orientation_rad: np.ndarray | None = None,
) -> np.ndarray:
    """Build Bartlett spatial spectra from receiver covariance matrices.

    covariance shape: [snapshot, tx, rx, rx_ant, rx_ant].
    """

    cov = np.asarray(covariance, dtype=np.complex64)
    if cov.ndim != 5:
        raise ValueError(f"covariance must have rank 5, got {cov.shape}")
    num_rx_ant = int(rx_num_rows) * int(rx_num_cols)
    if cov.shape[3:] != (num_rx_ant, num_rx_ant):
        raise ValueError(
            f"covariance antenna dimensions {cov.shape[3:]} do not match "
            f"rx_num_rows*rx_num_cols {num_rx_ant}"
        )

    angle_grid = build_angle_grid_rad(config)
    orientations = _normalize_rx_orientation(
        rx_orientation_rad,
        snapshot_count=cov.shape[0],
        rx_count=cov.shape[2],
    )
    if orientations is not None and not np.allclose(orientations, 0.0):
        return _build_bartlett_spectrum_from_covariance_oriented(
            cov,
            angle_grid,
            rx_num_rows=rx_num_rows,
            rx_num_cols=rx_num_cols,
            rx_spacing_lambda=rx_spacing_lambda,
            config=config,
            rx_orientation_rad=orientations,
        )

    steering = _steering_matrix(
        angle_grid,
        rx_num_rows=rx_num_rows,
        rx_num_cols=rx_num_cols,
        rx_spacing_lambda=rx_spacing_lambda,
    )
    return _build_bartlett_spectrum_with_covariance(cov, angle_grid, steering, config)


def _build_bartlett_spectrum_from_covariance_oriented(
    covariance: np.ndarray,
    angle_grid: np.ndarray,
    *,
    rx_num_rows: int,
    rx_num_cols: int,
    rx_spacing_lambda: tuple[float, float],
    config: ArraySpectrumConfig,
    rx_orientation_rad: np.ndarray,
) -> np.ndarray:
    output = np.zeros((*covariance.shape[:3], *angle_grid.shape[:2]), dtype=np.float32)
    for snapshot_idx in range(covariance.shape[0]):
        for rx_idx in range(covariance.shape[2]):
            steering = _steering_matrix(
                angle_grid,
                rx_num_rows=rx_num_rows,
                rx_num_cols=rx_num_cols,
                rx_spacing_lambda=rx_spacing_lambda,
                orientation_rad=rx_orientation_rad[snapshot_idx, rx_idx],
            )
            link_covariance = covariance[
                snapshot_idx : snapshot_idx + 1,
                :,
                rx_idx : rx_idx + 1,
            ]
            output[snapshot_idx : snapshot_idx + 1, :, rx_idx : rx_idx + 1] = (
                _build_bartlett_spectrum_with_covariance(
                    link_covariance,
                    angle_grid,
                    steering,
                    config,
                )
            )
    return output


def _build_bartlett_spectrum_with_covariance(
    covariance: np.ndarray,
    angle_grid: np.ndarray,
    steering: np.ndarray,
    config: ArraySpectrumConfig,
) -> np.ndarray:
    num_rx_ant = steering.shape[-1]
    flat_steering = steering.reshape(-1, num_rx_ant)
    output = np.zeros((*covariance.shape[:3], *angle_grid.shape[:2]), dtype=np.float32)

    num_links = int(np.prod(covariance.shape[:3]))
    flat_covariance = covariance.reshape(num_links, num_rx_ant, num_rx_ant)
    flat_output = output.reshape(num_links, -1)
    chunk_size = int(config.link_chunk_size)

    for start in range(0, flat_covariance.shape[0], chunk_size):
        stop = min(start + chunk_size, flat_covariance.shape[0])
        cov = flat_covariance[start:stop]
        if cov.size == 0:
            continue

        active = np.any(np.isfinite(cov), axis=(1, 2)) & np.any(
            np.abs(cov) > 0.0,
            axis=(1, 2),
        )
        if not np.any(active):
            continue

        active_covariance = np.nan_to_num(cov[active], copy=True)
        projected = np.einsum(
            "ba,cad,bd->cb",
            np.conjugate(flat_steering),
            active_covariance,
            flat_steering,
            optimize=True,
        ).real
        projected = np.maximum(projected, 0.0).astype(np.float32, copy=False)

        peaks = np.max(projected, axis=1)
        valid_peaks = (peaks > 0.0) & np.isfinite(peaks)
        normalized = np.zeros_like(projected)
        normalized[valid_peaks] = (
            projected[valid_peaks] / peaks[valid_peaks, np.newaxis]
        )

        chunk_output = np.zeros((stop - start, flat_output.shape[1]), dtype=np.float32)
        chunk_output[active] = normalized
        flat_output[start:stop] = chunk_output
    return output


def _covariance_from_samples(
    array: np.ndarray,
    *,
    link_chunk_size: int = 512,
) -> np.ndarray:
    num_rx_ant = array.shape[3]
    sample_count = int(np.prod(array.shape[4:]))
    covariance = np.zeros((*array.shape[:3], num_rx_ant, num_rx_ant), dtype=np.complex64)
    if sample_count == 0:
        return covariance

    num_links = int(np.prod(array.shape[:3]))
    flat_samples = array.reshape(num_links, num_rx_ant, sample_count)
    flat_covariance = covariance.reshape(num_links, num_rx_ant, num_rx_ant)
    chunk_size = max(1, int(link_chunk_size))
    for start in range(0, num_links, chunk_size):
        stop = min(start + chunk_size, num_links)
        x = flat_samples[start:stop]
        covariance_chunk = flat_covariance[start:stop]
        active = np.any(np.isfinite(x), axis=(1, 2)) & np.any(
            np.abs(x) > 0.0,
            axis=(1, 2),
        )
        if not np.any(active):
            continue
        active_x = np.nan_to_num(x[active], copy=True)
        covariance_chunk[active] = (
            np.matmul(active_x, np.conjugate(np.swapaxes(active_x, -1, -2)))
            / np.float32(sample_count)
        )
    return covariance


def _steering_matrix(
    angle_grid_rad: np.ndarray,
    *,
    rx_num_rows: int,
    rx_num_cols: int,
    rx_spacing_lambda: tuple[float, float],
    orientation_rad: np.ndarray | None = None,
) -> np.ndarray:
    element_positions = _planar_array_positions(
        rx_num_rows,
        rx_num_cols,
        rx_spacing_lambda,
    )
    if orientation_rad is not None:
        rotation = _rotation_matrix(np.asarray(orientation_rad, dtype=np.float32))
        element_positions = element_positions @ rotation.T

    zenith = angle_grid_rad[..., 0][..., np.newaxis]
    azimuth = angle_grid_rad[..., 1][..., np.newaxis]
    direction_x = np.sin(zenith) * np.cos(azimuth)
    direction_y = np.sin(zenith) * np.sin(azimuth)
    direction_z = np.cos(zenith)
    phase = 2.0 * np.pi * (
        element_positions[:, 0] * direction_x
        + element_positions[:, 1] * direction_y
        + element_positions[:, 2] * direction_z
    )
    steering = np.exp(1j * phase).astype(np.complex64)
    return steering / np.sqrt(np.float32(rx_num_rows * rx_num_cols))


def _planar_array_positions(
    rx_num_rows: int,
    rx_num_cols: int,
    rx_spacing_lambda: tuple[float, float],
) -> np.ndarray:
    """Return Sionna PlanarArray element positions in local x-y-z order."""

    vertical_spacing, horizontal_spacing = rx_spacing_lambda
    col, row = np.meshgrid(
        np.arange(rx_num_cols, dtype=np.float32),
        np.arange(rx_num_rows, dtype=np.float32),
        indexing="ij",
    )
    element_x = np.zeros((rx_num_rows * rx_num_cols,), dtype=np.float32)
    element_y = (
        col.reshape(-1) - np.float32((rx_num_cols - 1) / 2.0)
    ) * np.float32(horizontal_spacing)
    element_z = (
        np.float32((rx_num_rows - 1) / 2.0) - row.reshape(-1)
    ) * np.float32(vertical_spacing)
    return np.stack((element_x, element_y, element_z), axis=-1)


def _rotation_matrix(orientation_rad: np.ndarray) -> np.ndarray:
    """Numpy equivalent of Sionna RT's z-y-x orientation rotation matrix."""

    if orientation_rad.shape != (3,):
        raise ValueError(f"orientation_rad must have shape (3,), got {orientation_rad.shape}")
    alpha, beta, gamma = [float(value) for value in orientation_rad]
    sin_a, cos_a = np.sin(alpha), np.cos(alpha)
    sin_b, cos_b = np.sin(beta), np.cos(beta)
    sin_c, cos_c = np.sin(gamma), np.cos(gamma)
    return np.asarray(
        [
            [
                cos_a * cos_b,
                cos_a * sin_b * sin_c - sin_a * cos_c,
                cos_a * sin_b * cos_c + sin_a * sin_c,
            ],
            [
                sin_a * cos_b,
                sin_a * sin_b * sin_c + cos_a * cos_c,
                sin_a * sin_b * cos_c - cos_a * sin_c,
            ],
            [-sin_b, cos_b * sin_c, cos_b * cos_c],
        ],
        dtype=np.float32,
    )


def _normalize_rx_orientation(
    rx_orientation_rad: np.ndarray | None,
    *,
    snapshot_count: int,
    rx_count: int,
) -> np.ndarray | None:
    if rx_orientation_rad is None:
        return None
    orientation = np.asarray(rx_orientation_rad, dtype=np.float32)
    if orientation.ndim >= 2 and orientation.shape[-2:] == (3, 1):
        orientation = np.squeeze(orientation, axis=-1)
    if orientation.shape == (3,):
        orientation = np.broadcast_to(orientation, (snapshot_count, rx_count, 3))
    elif orientation.shape == (rx_count, 3):
        orientation = np.broadcast_to(
            orientation[np.newaxis, :, :],
            (snapshot_count, rx_count, 3),
        )
    elif orientation.shape == (snapshot_count, rx_count, 3):
        pass
    else:
        raise ValueError(
            "rx_orientation_rad must have shape [3], [rx,3], or [snapshot,rx,3], "
            f"got {orientation.shape}"
        )
    return np.asarray(orientation, dtype=np.float32)
