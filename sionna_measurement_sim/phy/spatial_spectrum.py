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
    link_shape = array.shape[:3]
    num_rx_ant = array.shape[3]
    snapshot_matrix = np.zeros((*link_shape, num_rx_ant, num_rx_ant), dtype=np.complex64)
    for index in np.ndindex(link_shape):
        x = array[index].reshape(num_rx_ant, -1)
        if x.shape[1] > 0:
            snapshot_matrix[index] = (x @ np.conjugate(x.T)) / np.float32(x.shape[1])
    return snapshot_matrix


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
) -> np.ndarray:
    """Build Bartlett spatial spectra from receiver-array samples.

    samples shape: [snapshot, ul_tx, ul_rx, rx_ant, ...].
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

    angle_grid = build_angle_grid_rad(config)
    steering = _steering_matrix(
        angle_grid,
        rx_num_rows=rx_num_rows,
        rx_num_cols=rx_num_cols,
        rx_spacing_lambda=rx_spacing_lambda,
    )
    flat_steering = steering.reshape(-1, num_rx_ant)
    output = np.zeros((*array.shape[:3], *angle_grid.shape[:2]), dtype=np.float32)

    num_links = int(np.prod(array.shape[:3]))
    sample_count = int(np.prod(array.shape[4:]))
    flat_samples = array.reshape(num_links, num_rx_ant, sample_count)
    flat_output = output.reshape(flat_samples.shape[0], -1)
    chunk_size = int(config.link_chunk_size)

    for start in range(0, flat_samples.shape[0], chunk_size):
        stop = min(start + chunk_size, flat_samples.shape[0])
        x = flat_samples[start:stop]
        if x.size == 0 or x.shape[2] == 0:
            continue

        active = np.any(np.isfinite(x), axis=(1, 2)) & np.any(
            np.abs(x) > 0.0,
            axis=(1, 2),
        )
        if not np.any(active):
            continue

        active_x = np.nan_to_num(x[active], copy=True)
        covariance = np.matmul(
            active_x,
            np.conjugate(np.swapaxes(active_x, -1, -2)),
        ) / np.float32(active_x.shape[2])
        projected = np.einsum(
            "ba,cad,bd->cb",
            np.conjugate(flat_steering),
            covariance,
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


def _steering_matrix(
    angle_grid_rad: np.ndarray,
    *,
    rx_num_rows: int,
    rx_num_cols: int,
    rx_spacing_lambda: tuple[float, float],
) -> np.ndarray:
    vertical_spacing, horizontal_spacing = rx_spacing_lambda
    col, row = np.meshgrid(
        np.arange(rx_num_cols, dtype=np.float32),
        np.arange(rx_num_rows, dtype=np.float32),
        indexing="ij",
    )
    element_y = (
        col.reshape(-1) - np.float32((rx_num_cols - 1) / 2.0)
    ) * np.float32(horizontal_spacing)
    element_z = (
        np.float32((rx_num_rows - 1) / 2.0) - row.reshape(-1)
    ) * np.float32(vertical_spacing)

    zenith = angle_grid_rad[..., 0][..., np.newaxis]
    azimuth = angle_grid_rad[..., 1][..., np.newaxis]
    direction_y = np.sin(zenith) * np.sin(azimuth)
    direction_z = np.cos(zenith)
    phase = 2.0 * np.pi * (element_y * direction_y + element_z * direction_z)
    steering = np.exp(1j * phase).astype(np.complex64)
    return steering / np.sqrt(np.float32(rx_num_rows * rx_num_cols))
