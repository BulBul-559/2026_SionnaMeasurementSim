"""Similarity metrics between the Sionna reference CFR and the DiffeRT CFR.

Two engines will never agree exactly, so the interesting quantities are:

* how much of the difference is a pure scale offset (``nmse_db`` after optimal complex
  scaling vs ``nmse_mag_db`` after real-only scaling),
* whether the frequency selectivity and delay structure agree (``rms_delay_spread``,
  ``freq_selectivity``, ``pdp_corr``),
* whether the spatial phase across the 32 ports agrees (``port_corr``), which is what a
  downstream beamformer / angle estimator actually consumes.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "score_pair",
    "pdp",
    "rms_delay_spread_ns",
    "freq_selectivity_db",
]

_EPS = 1e-30


def _flatten(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h, dtype=np.complex128)
    return h.reshape(-1)


def _db(x: float) -> float:
    return float(10.0 * np.log10(max(float(x), _EPS)))


def pdp(h: np.ndarray) -> np.ndarray:
    """Power delay profile from the CFR (one-sided, normalised to unit energy)."""
    h = np.asarray(h, dtype=np.complex128)
    if h.ndim == 1:
        h = h[None, :]
    ir = np.fft.ifft(h, axis=-1)
    prof = np.mean(np.abs(ir) ** 2, axis=0)
    total = prof.sum()
    return prof / total if total > 0 else prof


def rms_delay_spread_ns(h: np.ndarray, bandwidth_hz: float, num_subcarriers: int) -> float:
    """RMS delay spread, computed *circularly*.

    The delay axis of an N-point IFFT is circular: a channel whose energy sits at bin 0
    leaks a little energy into the last bins, so a plain linear first/second moment puts
    the mean in the middle of the window and reports a delay spread of microseconds for a
    single-tap channel. Using the circular mean (angle of the first Fourier moment) and
    angle-wrapped deviations keeps the metric bounded by the window/2 and correct.
    """
    prof = pdp(h)
    n = prof.size
    if n == 0 or prof.sum() <= 0:
        return 0.0

    dt = 1.0 / (num_subcarriers * (bandwidth_hz / (num_subcarriers - 1)))
    phases = 2.0 * np.pi * np.arange(n) / n

    moment = np.sum(prof * np.exp(1j * phases))
    if np.abs(moment) <= 0:
        return float(1e9 * dt * n / 2.0)

    mean_phase = np.angle(moment) % (2.0 * np.pi)
    # wrapped deviation in bins, in [-n/2, n/2)
    dev = np.angle(np.exp(1j * (phases - mean_phase))) * n / (2.0 * np.pi)
    var_bins = float(np.sum(prof * dev**2))
    return float(1e9 * dt * np.sqrt(max(var_bins, 0.0)))


def freq_selectivity_db(h: np.ndarray) -> float:
    """Mean over ports of the std over frequency of the narrowband power, in dB."""
    h = np.asarray(h, dtype=np.complex128)
    if h.ndim == 1:
        h = h[None, :]
    power_db = 10.0 * np.log10(np.abs(h) ** 2 + _EPS)
    return float(np.mean(np.std(power_db, axis=-1)))


def score_pair(
    h_ref: np.ndarray,
    h_dut: np.ndarray,
    *,
    bandwidth_hz: float,
    num_subcarriers: int,
) -> dict[str, float]:
    """Compare one link's CFR (``(ports, freq)``) from both engines."""
    ref = _flatten(h_ref)
    dut = _flatten(h_dut)

    ref_power = float(np.vdot(ref, ref).real)
    dut_power = float(np.vdot(dut, dut).real)

    # optimal complex scalar (amplitude + phase)
    alpha = np.vdot(dut, ref) / (dut_power if dut_power > 0 else 1.0)
    nmse_complex = float(np.linalg.norm(ref - alpha * dut) ** 2 / max(ref_power, _EPS))

    # best real scalar, then best per-link constant phase removed -> "shape only"
    beta = float(np.sqrt(ref_power / dut_power)) if dut_power > 0 else 0.0
    nmse_mag = float(np.linalg.norm(np.abs(ref) - np.abs(beta * dut)) ** 2 / max(ref_power, _EPS))

    # unit-norm cosine similarity
    if ref_power > 0 and dut_power > 0:
        cos_sim = float(np.abs(np.vdot(ref, dut)) / np.sqrt(ref_power * dut_power))
    else:
        cos_sim = 0.0

    # magnitude-only correlation
    mag_ref = np.abs(ref)
    mag_dut = np.abs(dut)
    if mag_ref.std() > 0 and mag_dut.std() > 0:
        mag_corr = float(np.corrcoef(mag_ref, mag_dut)[0, 1])
    else:
        mag_corr = 0.0

    # per-port correlation of the frequency response (spatial consistency)
    ref2 = np.asarray(h_ref, dtype=np.complex128)
    dut2 = np.asarray(h_dut, dtype=np.complex128)
    if ref2.ndim == 1:
        ref2 = ref2[None, :]
        dut2 = dut2[None, :]
    port_corrs = []
    for i in range(min(ref2.shape[0], dut2.shape[0])):
        a, b = ref2[i], dut2[i]
        if a.std() > 0 and b.std() > 0:
            port_corrs.append(float(np.corrcoef(np.abs(a), np.abs(b))[0, 1]))
    port_corr = float(np.mean(port_corrs)) if port_corrs else 0.0

    p_ref = pdp(ref2)
    p_dut = pdp(dut2)
    pdp_corr = (
        float(np.corrcoef(p_ref, p_dut)[0, 1]) if p_ref.std() > 0 and p_dut.std() > 0 else 0.0
    )
    pdp_l1 = float(np.abs(p_ref - p_dut).sum() / 2.0)

    ref_power_db = _db(ref_power / ref.size)
    dut_power_db = _db(dut_power / dut.size)

    return {
        "cos_sim": cos_sim,
        "nmse_complex_db": _db(nmse_complex),
        "nmse_magnitude_db": _db(nmse_mag),
        "mag_corr": mag_corr,
        "port_corr": port_corr,
        "pdp_corr": pdp_corr,
        "pdp_l1": pdp_l1,
        "power_ref_db": ref_power_db,
        "power_dut_db": dut_power_db,
        "power_offset_db": dut_power_db - ref_power_db,
        "rms_ds_ref_ns": rms_delay_spread_ns(ref2, bandwidth_hz, num_subcarriers),
        "rms_ds_dut_ns": rms_delay_spread_ns(dut2, bandwidth_hz, num_subcarriers),
        "freq_sel_ref_db": freq_selectivity_db(ref2),
        "freq_sel_dut_db": freq_selectivity_db(dut2),
    }
