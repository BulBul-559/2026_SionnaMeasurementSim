import numpy as np

from sionna_measurement_sim.domain.array import ArraySpectrumConfig
from sionna_measurement_sim.phy.spatial_spectrum import (
    _steering_matrix,
    build_angle_grid_rad,
    build_bartlett_spectrum,
    project_cfr_to_ul_receiver_samples,
)


def _build_bartlett_spectrum_per_link(
    samples: np.ndarray,
    *,
    rx_num_rows: int,
    rx_num_cols: int,
    rx_spacing_lambda: tuple[float, float],
    config: ArraySpectrumConfig,
) -> np.ndarray:
    array = np.asarray(samples, dtype=np.complex64)
    num_rx_ant = int(rx_num_rows) * int(rx_num_cols)
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


def test_array_spectrum_config_defaults_to_link_chunk_size_512():
    assert ArraySpectrumConfig().link_chunk_size == 512


def test_bartlett_spectrum_peak_matches_synthetic_direction():
    config = ArraySpectrumConfig(
        enabled=True,
        zenith_bins=5,
        azimuth_bins=5,
        zenith_min_rad=0.0,
        zenith_max_rad=np.pi,
        azimuth_min_rad=-np.pi,
        azimuth_max_rad=np.pi,
    )
    angle_grid = build_angle_grid_rad(config)
    target_zenith_idx = 2
    target_azimuth_idx = 3
    zenith, azimuth = angle_grid[target_zenith_idx, target_azimuth_idx]

    rows, cols = 2, 2
    row = np.arange(rows, dtype=np.float32) - (rows - 1) / 2.0
    col = np.arange(cols, dtype=np.float32) - (cols - 1) / 2.0
    rr, cc = np.meshgrid(row, col, indexing="ij")
    direction_y = np.sin(zenith) * np.sin(azimuth)
    direction_z = np.cos(zenith)
    phase = np.pi * (cc.reshape(-1) * direction_y + rr.reshape(-1) * direction_z)
    steering = np.exp(1j * phase).astype(np.complex64) / np.sqrt(np.float32(rows * cols))
    samples = np.repeat(steering[:, np.newaxis], 8, axis=1)
    samples = samples.reshape(1, 1, 1, rows * cols, 8)

    spectrum = build_bartlett_spectrum(
        samples,
        rx_num_rows=rows,
        rx_num_cols=cols,
        rx_spacing_lambda=(0.5, 0.5),
        config=config,
    )

    assert spectrum[0, 0, 0, target_zenith_idx, target_azimuth_idx] == np.float32(1.0)


def test_bartlett_spectrum_matches_per_link_reference_across_chunks():
    rng = np.random.default_rng(20240514)
    samples = (
        rng.normal(size=(2, 3, 2, 4, 2, 3))
        + 1j * rng.normal(size=(2, 3, 2, 4, 2, 3))
    ).astype(np.complex64)
    samples[0, 1, 1] = 0.0
    config = ArraySpectrumConfig(
        enabled=True,
        zenith_bins=4,
        azimuth_bins=5,
        link_chunk_size=3,
    )

    spectrum = build_bartlett_spectrum(
        samples,
        rx_num_rows=2,
        rx_num_cols=2,
        rx_spacing_lambda=(0.5, 0.5),
        config=config,
    )
    reference = _build_bartlett_spectrum_per_link(
        samples,
        rx_num_rows=2,
        rx_num_cols=2,
        rx_spacing_lambda=(0.5, 0.5),
        config=config,
    )

    np.testing.assert_allclose(spectrum, reference, rtol=2e-5, atol=2e-6)
    assert np.all(spectrum[0, 1, 1] == 0.0)
    assert np.allclose(np.max(spectrum, axis=(-2, -1))[spectrum.any(axis=(-2, -1))], 1.0)


def test_bartlett_spectrum_zero_input_is_zero():
    config = ArraySpectrumConfig(enabled=True, zenith_bins=3, azimuth_bins=4)
    spectrum = build_bartlett_spectrum(
        np.zeros((1, 2, 1, 4, 3), dtype=np.complex64),
        rx_num_rows=2,
        rx_num_cols=2,
        rx_spacing_lambda=(0.5, 0.5),
        config=config,
    )

    assert spectrum.shape == (1, 2, 1, 3, 4)
    assert np.all(spectrum == 0.0)


def test_project_cfr_to_ul_receiver_samples_accepts_truth_and_estimate_cfr():
    truth = np.arange(2 * 3 * 4 * 5 * 6, dtype=np.float32).reshape(2, 3, 4, 5, 6)
    truth_samples = project_cfr_to_ul_receiver_samples(truth.astype(np.complex64))

    assert truth_samples.shape == (1, 3, 2, 5, 4, 6)
    np.testing.assert_allclose(truth_samples[0, 1, 0, 2, 3, 4], truth[0, 1, 3, 2, 4])

    estimate = np.arange(7 * 2 * 3 * 4 * 5 * 6, dtype=np.float32).reshape(
        7, 2, 3, 4, 5, 6
    )
    estimate_samples = project_cfr_to_ul_receiver_samples(estimate.astype(np.complex64))

    assert estimate_samples.shape == (7, 3, 2, 5, 4, 6)
    np.testing.assert_allclose(
        estimate_samples[6, 1, 0, 2, 3, 4],
        estimate[6, 0, 1, 3, 2, 4],
    )
