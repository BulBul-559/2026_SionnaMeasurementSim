"""Tests for the HDF5 writer and the similarity metrics (no ray tracing involved)."""

from __future__ import annotations

import numpy as np

from differt_crosscheck.config import DEFAULT_GRID
from differt_crosscheck.io import SCHEMA_VERSION, write_scene_h5
from differt_crosscheck.metrics import (
    freq_selectivity_db,
    pdp,
    rms_delay_spread_ns,
    score_pair,
)


def _fake_cfr(num_links=2, ports=32, freqs=64, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(num_links, ports, 1, freqs)) + 1j * rng.normal(size=(num_links, ports, 1, freqs))).astype(np.complex64)


def test_h5_roundtrip(tmp_path=None) -> None:
    import h5py
    import tempfile
    from pathlib import Path

    grid = DEFAULT_GRID
    freqs = grid.frequencies_hz()
    cfr = _fake_cfr(freqs=freqs.size)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "scene.h5"
        write_scene_h5(
            path,
            scene_key="toy",
            scene_class="small_room",
            scene_file="toy.xml",
            cfr=cfr,
            bs_id=np.array([0, 0]),
            ue_id=np.array([3, 4]),
            bs_pos=np.zeros((2, 3)),
            ue_pos=np.ones((2, 3)),
            rx_pos=np.zeros((32, 3)),
            frequencies_hz=freqs,
            grid=grid.as_dict(),
            antenna={
                "tx": {"num_rows": 1, "num_cols": 1, "num_elements": 1, "spacing_m": 0.0,
                       "spacing_lambda": 0.5, "pattern": "iso", "polarization": "V"},
                "rx": {"num_rows": 4, "num_cols": 8, "num_elements": 32, "spacing_m": 0.04364,
                       "spacing_lambda": 0.5005, "pattern": "iso", "polarization": "V"},
            },
            solver={"tracer": "sbr"},
            mesh={"num_triangles": 1},
        )
        with h5py.File(path, "r") as f:
            assert f["cfr"].shape == (2, 32, 1, freqs.size)
            assert f["cfr"].dtype == np.complex64
            assert np.allclose(f["frequency/frequencies_hz"][()], freqs)
            assert f["meta/schema_version"][()].decode() == SCHEMA_VERSION
            assert f["scene/ports"][()] == 32
            assert np.allclose(np.asarray(f["cfr"][()]), cfr)


def test_identical_links_score_perfectly() -> None:
    grid = DEFAULT_GRID
    freqs = grid.frequencies_hz()
    h = _fake_cfr(num_links=1, freqs=freqs.size)[0, :, 0, :]
    score = score_pair(h, h.copy(), bandwidth_hz=grid.bandwidth_hz, num_subcarriers=grid.num_subcarriers)
    assert abs(score["cos_sim"] - 1.0) < 1e-6
    assert score["nmse_complex_db"] < -30
    assert abs(score["power_offset_db"]) < 1e-6


def test_scale_and_phase_invariance() -> None:
    grid = DEFAULT_GRID
    freqs = grid.frequencies_hz()
    h = _fake_cfr(num_links=1, freqs=freqs.size)[0, :, 0, :]
    shifted = 3.5 * h * np.exp(1j * 0.7)
    score = score_pair(h, shifted, bandwidth_hz=grid.bandwidth_hz, num_subcarriers=grid.num_subcarriers)
    assert score["cos_sim"] > 0.999
    assert abs(score["power_offset_db"] - 20 * np.log10(3.5)) < 1e-3


def test_noise_degrades_scores() -> None:
    grid = DEFAULT_GRID
    freqs = grid.frequencies_hz()
    h = _fake_cfr(num_links=1, freqs=freqs.size)[0, :, 0, :]
    rng = np.random.default_rng(7)
    noisy = h + 0.5 * (rng.normal(size=h.shape) + 1j * rng.normal(size=h.shape))
    score = score_pair(h, noisy, bandwidth_hz=grid.bandwidth_hz, num_subcarriers=grid.num_subcarriers)
    assert score["cos_sim"] < 0.999
    assert score["nmse_complex_db"] > -30


def test_pdp_and_spread() -> None:
    grid = DEFAULT_GRID
    freqs = grid.frequencies_hz()
    # single tap at zero delay -> zero spread
    h = np.ones((2, freqs.size), dtype=complex)
    prof = pdp(h)
    assert np.isclose(prof.sum(), 1.0)
    assert rms_delay_spread_ns(h, grid.bandwidth_hz, grid.num_subcarriers) < 1e-6
    sel = freq_selectivity_db(np.ones((2, freqs.size), dtype=complex))
    assert sel < 1e-6


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
