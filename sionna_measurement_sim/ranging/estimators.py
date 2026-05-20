"""Waveform-level ToA/range estimators operating on estimated CFR."""

from __future__ import annotations

import numpy as np

from sionna_measurement_sim.domain.derived import SPEED_OF_LIGHT_MPS
from sionna_measurement_sim.domain.frequency import FrequencyGrid
from sionna_measurement_sim.ranging.config import (
    PdpPeakRangingConfig,
    PhaseSlopeRangingConfig,
)
from sionna_measurement_sim.ranging.result import PdpPeakResult, PhaseSlopeResult


def estimate_pdp_peak(
    cfr_est: np.ndarray,
    frequency_grid: FrequencyGrid,
    truth_first_path_delay_s: np.ndarray,
    config: PdpPeakRangingConfig,
    *,
    write_rtt_equivalent: bool = True,
) -> PdpPeakResult:
    """Estimate ToA by locating the first detectable PDP peak."""

    cfr = _as_cfr6(cfr_est)
    snapshots, tx_count, rx_count = cfr.shape[:3]
    subcarriers = cfr.shape[-1]
    spacing_hz = _subcarrier_spacing_hz(frequency_grid)
    n_fft = int(subcarriers * config.oversampling_factor)
    delay_step_s = 1.0 / (spacing_hz * n_fft)
    max_bin = n_fft
    if config.max_delay_s is not None:
        max_bin = max(1, min(n_fft, int(np.floor(config.max_delay_s / delay_step_s)) + 1))

    window = _window(subcarriers, config.window).astype(np.float32)
    toa = _nan_float((snapshots, tx_count, rx_count))
    one_way = _nan_float((snapshots, tx_count, rx_count))
    rtt_equiv = _nan_float((snapshots, tx_count, rx_count))
    error = _nan_float((snapshots, tx_count, rx_count))
    selected_bin = np.full((snapshots, tx_count, rx_count), -1, dtype=np.int32)
    peak_power = _nan_float((snapshots, tx_count, rx_count))
    peak_snr = _nan_float((snapshots, tx_count, rx_count))
    success = np.zeros((snapshots, tx_count, rx_count), dtype=np.bool_)

    truth_range = _truth_range_by_snapshot(truth_first_path_delay_s, snapshots)
    threshold_ratio = 10.0 ** (config.relative_threshold_db / 10.0)
    for snapshot in range(snapshots):
        for tx in range(tx_count):
            for rx in range(rx_count):
                link_cfr = cfr[snapshot, tx, rx]
                if not np.any(np.isfinite(link_cfr)):
                    continue
                profile = np.fft.ifft(
                    link_cfr * window[np.newaxis, np.newaxis, :],
                    n=n_fft,
                    axis=-1,
                )
                power = np.mean(np.abs(profile) ** 2, axis=(0, 1)).astype(np.float64)
                power = power[:max_bin]
                if power.size == 0 or not np.any(np.isfinite(power)):
                    continue

                strongest_bin = int(np.nanargmax(power))
                strongest_power = float(power[strongest_bin])
                if strongest_power <= 0.0 or not np.isfinite(strongest_power):
                    continue
                candidate_bins = _local_peak_indices(power)
                candidate_bins = candidate_bins[
                    power[candidate_bins] >= strongest_power * threshold_ratio
                ]
                candidate_bins = candidate_bins[candidate_bins <= strongest_bin]
                if candidate_bins.size == 0:
                    continue
                coarse_bin = _select_first_peak_cluster_max(
                    candidate_bins,
                    power,
                    max_cluster_gap=max(1, config.oversampling_factor),
                )
                noise_floor = float(np.nanmedian(power))
                snr_db = _linear_to_db(
                    strongest_power / max(noise_floor, np.finfo(np.float64).tiny)
                )
                if snr_db < config.min_peak_snr_db:
                    continue

                bin_offset = (
                    _parabolic_log_power_offset(power, coarse_bin)
                    if config.interpolation == "parabolic_log_power"
                    else 0.0
                )
                est_toa = (float(coarse_bin) + bin_offset) * delay_step_s
                est_range = est_toa * SPEED_OF_LIGHT_MPS
                toa[snapshot, tx, rx] = est_toa
                one_way[snapshot, tx, rx] = est_range
                if write_rtt_equivalent:
                    rtt_equiv[snapshot, tx, rx] = 2.0 * est_toa
                error[snapshot, tx, rx] = est_range - truth_range[snapshot, tx, rx]
                selected_bin[snapshot, tx, rx] = coarse_bin
                peak_power[snapshot, tx, rx] = float(power[coarse_bin])
                peak_snr[snapshot, tx, rx] = snr_db
                success[snapshot, tx, rx] = True

    return PdpPeakResult(
        toa_est_s=toa,
        one_way_range_est_m=one_way,
        rtt_equiv_s=rtt_equiv,
        range_error_m=error,
        detection_success=success,
        selected_delay_bin=selected_bin,
        peak_power_linear=peak_power,
        peak_snr_db=peak_snr,
    )


def estimate_phase_slope(
    cfr_est: np.ndarray,
    frequency_grid: FrequencyGrid,
    truth_first_path_delay_s: np.ndarray,
    config: PhaseSlopeRangingConfig,
    *,
    write_rtt_equivalent: bool = True,
) -> PhaseSlopeResult:
    """Estimate ToA from the CFR phase slope across subcarriers."""

    cfr = _as_cfr6(cfr_est)
    snapshots, tx_count, rx_count = cfr.shape[:3]
    frequencies = np.asarray(frequency_grid.frequencies_hz, dtype=np.float64)
    centered_f = frequencies - float(np.mean(frequencies))
    toa = _nan_float((snapshots, tx_count, rx_count))
    one_way = _nan_float((snapshots, tx_count, rx_count))
    rtt_equiv = _nan_float((snapshots, tx_count, rx_count))
    error = _nan_float((snapshots, tx_count, rx_count))
    residual = _nan_float((snapshots, tx_count, rx_count))
    success = np.zeros((snapshots, tx_count, rx_count), dtype=np.bool_)
    truth_range = _truth_range_by_snapshot(truth_first_path_delay_s, snapshots)

    for snapshot in range(snapshots):
        for tx in range(tx_count):
            for rx in range(rx_count):
                estimates: list[float] = []
                estimate_weights: list[float] = []
                residuals: list[float] = []
                for rx_ant in range(cfr.shape[3]):
                    for tx_ant in range(cfr.shape[4]):
                        h = cfr[snapshot, tx, rx, rx_ant, tx_ant]
                        power = np.abs(h) ** 2
                        mean_power = float(np.mean(power))
                        if mean_power < config.min_mean_power or not np.all(np.isfinite(h)):
                            continue
                        phase = np.angle(h)
                        if config.unwrap:
                            phase = np.unwrap(phase)
                        slope, intercept = _weighted_line_fit(centered_f, phase, power)
                        est_toa = -slope / (2.0 * np.pi)
                        if not np.isfinite(est_toa) or est_toa < 0.0:
                            continue
                        fitted = slope * centered_f + intercept
                        residual_rad = float(
                            np.sqrt(np.average((phase - fitted) ** 2, weights=power))
                        )
                        estimates.append(float(est_toa))
                        estimate_weights.append(mean_power)
                        residuals.append(residual_rad)
                if not estimates:
                    continue
                est_toa = _weighted_median(
                    np.asarray(estimates, dtype=np.float64),
                    np.asarray(estimate_weights, dtype=np.float64),
                )
                est_range = est_toa * SPEED_OF_LIGHT_MPS
                toa[snapshot, tx, rx] = est_toa
                one_way[snapshot, tx, rx] = est_range
                if write_rtt_equivalent:
                    rtt_equiv[snapshot, tx, rx] = 2.0 * est_toa
                error[snapshot, tx, rx] = est_range - truth_range[snapshot, tx, rx]
                residual[snapshot, tx, rx] = _weighted_median(
                    np.asarray(residuals, dtype=np.float64),
                    np.asarray(estimate_weights, dtype=np.float64),
                )
                success[snapshot, tx, rx] = True

    return PhaseSlopeResult(
        toa_est_s=toa,
        one_way_range_est_m=one_way,
        rtt_equiv_s=rtt_equiv,
        range_error_m=error,
        detection_success=success,
        fit_residual_rad=residual,
    )


def _as_cfr6(cfr_est: np.ndarray) -> np.ndarray:
    cfr = np.asarray(cfr_est, dtype=np.complex64)
    if cfr.ndim != 6:
        msg = f"cfr_est must have shape [snapshot,tx,rx,rx_ant,tx_ant,subcarrier], got {cfr.shape}"
        raise ValueError(msg)
    return cfr


def _subcarrier_spacing_hz(frequency_grid: FrequencyGrid) -> float:
    frequencies = np.asarray(frequency_grid.frequencies_hz, dtype=np.float64)
    diffs = np.diff(frequencies)
    if diffs.size == 0 or not np.all(diffs > 0):
        msg = "frequency_grid must contain strictly increasing subcarrier frequencies"
        raise ValueError(msg)
    return float(np.median(diffs))


def _window(size: int, name: str) -> np.ndarray:
    if name == "hann":
        return np.hanning(size)
    if name == "rect":
        return np.ones(size, dtype=np.float64)
    msg = f"unsupported PDP window {name!r}"
    raise ValueError(msg)


def _local_peak_indices(power: np.ndarray) -> np.ndarray:
    if power.size == 1:
        return np.array([0], dtype=np.int64)
    peaks = []
    for idx in range(power.size):
        left = power[idx - 1] if idx > 0 else -np.inf
        right = power[idx + 1] if idx + 1 < power.size else -np.inf
        if power[idx] >= left and power[idx] >= right:
            peaks.append(idx)
    return np.asarray(peaks, dtype=np.int64)


def _parabolic_log_power_offset(power: np.ndarray, bin_index: int) -> float:
    if bin_index <= 0 or bin_index >= power.size - 1:
        return 0.0
    eps = np.finfo(np.float64).tiny
    left = np.log(max(float(power[bin_index - 1]), eps))
    center = np.log(max(float(power[bin_index]), eps))
    right = np.log(max(float(power[bin_index + 1]), eps))
    denominator = left - 2.0 * center + right
    if abs(denominator) < 1e-12:
        return 0.0
    return float(np.clip(0.5 * (left - right) / denominator, -1.0, 1.0))


def _select_first_peak_cluster_max(
    candidate_bins: np.ndarray,
    power: np.ndarray,
    *,
    max_cluster_gap: int,
) -> int:
    cluster = [int(candidate_bins[0])]
    for value in candidate_bins[1:]:
        value = int(value)
        if value - cluster[-1] > max_cluster_gap:
            break
        cluster.append(value)
    cluster_array = np.asarray(cluster, dtype=np.int64)
    return int(cluster_array[int(np.argmax(power[cluster_array]))])


def _weighted_line_fit(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    weights = np.asarray(weights, dtype=np.float64)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        msg = "weights must contain positive power"
        raise ValueError(msg)
    x_mean = float(np.sum(weights * x) / weight_sum)
    y_mean = float(np.sum(weights * y) / weight_sum)
    centered_x = x - x_mean
    centered_y = y - y_mean
    denominator = float(np.sum(weights * centered_x * centered_x))
    if denominator <= 0.0:
        return 0.0, y_mean
    slope = float(np.sum(weights * centered_x * centered_y) / denominator)
    intercept = y_mean - slope * x_mean
    return slope, intercept


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cdf = np.cumsum(sorted_weights)
    cutoff = 0.5 * float(cdf[-1])
    return float(sorted_values[int(np.searchsorted(cdf, cutoff, side="left"))])


def _truth_range_by_snapshot(truth_first_path_delay_s: np.ndarray, snapshots: int) -> np.ndarray:
    truth = np.asarray(truth_first_path_delay_s, dtype=np.float32)
    if truth.ndim != 2:
        msg = "truth_first_path_delay_s must have shape [tx,rx]"
        raise ValueError(msg)
    return np.broadcast_to(truth[np.newaxis, ...] * SPEED_OF_LIGHT_MPS, (snapshots, *truth.shape))


def _nan_float(shape: tuple[int, ...]) -> np.ndarray:
    return np.full(shape, np.nan, dtype=np.float32)


def _linear_to_db(value: float) -> float:
    return float(10.0 * np.log10(max(value, np.finfo(np.float64).tiny)))
