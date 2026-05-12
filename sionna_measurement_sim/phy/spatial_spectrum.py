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
    """Convert project-view CFR tensors to UL receiver-array samples.

    Project truth CFR shape:
    [bs, ue, ue_ant, bs_ant, subcarrier]
    -> [snapshot=1, ue, bs, bs_ant, ue_ant, subcarrier]

    Project observation CFR shape:
    [snapshot, bs, ue, ue_ant, bs_ant, subcarrier]
    -> [snapshot, ue, bs, bs_ant, ue_ant, subcarrier]
    """

    array = np.asarray(cfr, dtype=np.complex64)
    if array.ndim == 5:
        return np.transpose(array, (1, 0, 3, 2, 4))[np.newaxis, ...]
    if array.ndim == 6:
        return np.transpose(array, (0, 2, 1, 4, 3, 5))
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

    for index in np.ndindex(array.shape[:3]):
        x = array[index].reshape(num_rx_ant, -1)
        if x.size == 0 or not np.any(np.isfinite(x)) or not np.any(np.abs(x) > 0.0):
            continue
        x = np.nan_to_num(x, copy=False)
        covariance = (x @ np.conjugate(x.T)) / np.float32(x.shape[1])
        projected = np.einsum(
            "bi,ij,bj->b",
            np.conjugate(flat_steering),
            covariance,
            flat_steering,
            optimize=True,
        ).real
        projected = np.maximum(projected, 0.0).astype(np.float32, copy=False)
        peak = float(np.max(projected)) if projected.size else 0.0
        if peak > 0.0 and np.isfinite(peak):
            projected = projected / np.float32(peak)
        else:
            projected = np.zeros_like(projected)
        output[index] = projected.reshape(angle_grid.shape[:2])
    return output


def _steering_matrix(
    angle_grid_rad: np.ndarray,
    *,
    rx_num_rows: int,
    rx_num_cols: int,
    rx_spacing_lambda: tuple[float, float],
) -> np.ndarray:
    vertical_spacing, horizontal_spacing = rx_spacing_lambda
    row = np.arange(rx_num_rows, dtype=np.float32) - (rx_num_rows - 1) / 2.0
    col = np.arange(rx_num_cols, dtype=np.float32) - (rx_num_cols - 1) / 2.0
    rr, cc = np.meshgrid(row, col, indexing="ij")
    element_y = cc.reshape(-1) * np.float32(horizontal_spacing)
    element_z = rr.reshape(-1) * np.float32(vertical_spacing)

    zenith = angle_grid_rad[..., 0][..., np.newaxis]
    azimuth = angle_grid_rad[..., 1][..., np.newaxis]
    direction_y = np.sin(zenith) * np.sin(azimuth)
    direction_z = np.cos(zenith)
    phase = 2.0 * np.pi * (element_y * direction_y + element_z * direction_z)
    steering = np.exp(1j * phase).astype(np.complex64)
    return steering / np.sqrt(np.float32(rx_num_rows * rx_num_cols))
