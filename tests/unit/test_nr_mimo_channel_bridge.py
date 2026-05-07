"""Unit tests for the NR MIMO channel bridge module."""

from __future__ import annotations

import numpy as np

from sionna_measurement_sim.domain.link import LinkConfig
from sionna_measurement_sim.phy.nr_mimo_channel import (
    PUSCHMIMOChannel,
    build_mimo_cfr_from_cir,
    cfr_to_pusch_perfect_h,
    pusch_h_to_cfr_est,
    reverse_reciprocity_cfr,
)


def _make_cir(snap=1, tx=2, rx=1, rx_ant=4, tx_ant=2, path=5, seed=42):
    rng = np.random.RandomState(seed)
    coeff = (
        rng.randn(snap, tx, rx, rx_ant, tx_ant, path)
        + 1j * rng.randn(snap, tx, rx, rx_ant, tx_ant, path)
    ).astype(np.complex64)
    delays = np.linspace(0, 1e-7, path, dtype=np.float32).reshape(
        1, 1, 1, 1, 1, path
    )
    delays = np.broadcast_to(
        delays, (snap, tx, rx, rx_ant, tx_ant, path)
    ).copy()
    return coeff, delays


class TestBuildMIMOCFRFromCIR:
    """Tests for build_mimo_cfr_from_cir."""

    def test_returns_pusch_mimo_channel(self):
        coeff, delays = _make_cir()
        link = LinkConfig()
        ch = build_mimo_cfr_from_cir(coeff, delays, link, 30000.0, 48)
        assert isinstance(ch, PUSCHMIMOChannel)
        assert ch.cfr.dtype == np.complex64
        assert ch.cfr.ndim == 6

    def test_no_reciprocity_converts_to_ul(self):
        """Without reciprocity, DL CIR is converted to UL convention."""
        snap, tx, rx, rx_ant, tx_ant, path = 2, 3, 1, 4, 2, 5
        coeff, delays = _make_cir(snap, tx, rx, rx_ant, tx_ant, path)
        link = LinkConfig(reciprocity_applied=False)
        ch = build_mimo_cfr_from_cir(coeff, delays, link, 15000.0, 48)
        assert ch.num_snap == snap
        # DL: tx→UL rx, DL: rx→UL tx
        assert ch.num_ul_tx == rx  # UL tx = UE = project rx
        assert ch.num_ul_rx == tx  # UL rx = BS = project tx
        assert ch.num_ul_tx_ant == rx_ant  # UE antennas
        assert ch.num_ul_rx_ant == tx_ant  # BS antennas
        assert not ch.reciprocity_applied

    def test_with_reciprocity_swaps_dimensions(self):
        """TDD reciprocity also converts to UL convention."""
        snap, tx, rx, rx_ant, tx_ant, path = 2, 3, 1, 4, 2, 5
        coeff, delays = _make_cir(snap, tx, rx, rx_ant, tx_ant, path)
        link = LinkConfig(reciprocity_applied=True)
        ch = build_mimo_cfr_from_cir(coeff, delays, link, 15000.0, 48)
        assert ch.reciprocity_applied
        # After reciprocity: ul_tx was project rx, ul_rx was project tx
        assert ch.num_ul_tx == rx
        assert ch.num_ul_rx == tx
        # UL tx ant = project rx_ant (UE antennas), UL rx ant = project tx_ant (BS antennas)
        assert ch.num_ul_tx_ant == rx_ant
        assert ch.num_ul_rx_ant == tx_ant

    def test_cfr_shape_matches_input(self):
        snap, tx, rx, rx_ant, tx_ant, path = 1, 2, 1, 4, 4, 3
        coeff, delays = _make_cir(snap, tx, rx, rx_ant, tx_ant, path)
        link = LinkConfig(reciprocity_applied=True)
        ch = build_mimo_cfr_from_cir(coeff, delays, link, 30000.0, 48)
        # Reciprocity: DL [snap=1, tx=2, rx=1, rx_ant=4, tx_ant=4, path=3]
        #           → UL [snap=1, ul_tx=1, ul_rx=2, ul_rx_ant=4, ul_tx_ant=4, subcarrier=48]
        assert ch.cfr.shape[0] == snap
        assert ch.cfr.shape[1] == rx  # ul_tx
        assert ch.cfr.shape[2] == tx  # ul_rx
        assert ch.cfr.shape[3] == tx_ant  # ul_rx_ant = BS antennas = project tx_ant
        assert ch.cfr.shape[4] == rx_ant  # ul_tx_ant = UE antennas = project rx_ant
        assert ch.cfr.shape[5] == 48

    def test_cfr_is_finite(self):
        coeff, delays = _make_cir()
        link = LinkConfig()
        ch = build_mimo_cfr_from_cir(coeff, delays, link, 30000.0, 48)
        assert np.all(np.isfinite(ch.cfr))


class TestCFRToPUSCHPerfectH:
    """Tests for cfr_to_pusch_perfect_h."""

    def test_shape_4x4(self):
        """4x4 MIMO: 4 rx antennas, 4 tx antennas."""
        coeff, delays = _make_cir(snap=1, tx=1, rx=1, rx_ant=4, tx_ant=4, path=3)
        link = LinkConfig(reciprocity_applied=False)
        ch = build_mimo_cfr_from_cir(coeff, delays, link, 30000.0, 48)
        h = cfr_to_pusch_perfect_h(ch, snap_idx=0, ul_tx_idx=0, ul_rx_idx=0, num_ofdm_symbols=14)
        assert h.ndim == 7
        assert h.shape[0] == 1  # batch
        assert h.shape[1] == 1  # num_rx
        assert h.shape[2] == 4  # num_rx_ant
        assert h.shape[3] == 1  # num_tx
        assert h.shape[4] == 4  # num_tx_ant
        assert h.shape[5] == 14  # num_ofdm_symbols
        assert h.shape[6] == 48  # subcarriers

    def test_shape_2x2(self):
        coeff, delays = _make_cir(snap=1, tx=1, rx=1, rx_ant=2, tx_ant=2, path=3)
        link = LinkConfig(reciprocity_applied=False)
        ch = build_mimo_cfr_from_cir(coeff, delays, link, 30000.0, 48)
        h = cfr_to_pusch_perfect_h(ch, 0, 0, 0, 14)
        assert h.shape[2] == 2
        assert h.shape[4] == 2

    def test_different_antenna_counts(self):
        """Asymmetric: 8 UE ant (project rx_ant), 2 BS ant (project tx_ant).

        Without reciprocity, these map to ul_tx_ant=8 and ul_rx_ant=2.
        """
        coeff, delays = _make_cir(snap=1, tx=1, rx=1, rx_ant=8, tx_ant=2, path=3)
        link = LinkConfig(reciprocity_applied=False)
        ch = build_mimo_cfr_from_cir(coeff, delays, link, 30000.0, 48)
        # UL: num_ul_tx_ant=8 (UE), num_ul_rx_ant=2 (BS)
        h = cfr_to_pusch_perfect_h(ch, 0, 0, 0, 14)
        assert h.shape[2] == 2  # num_rx_ant = ul_rx_ant = BS antennas
        assert h.shape[4] == 8  # num_tx_ant = ul_tx_ant = UE antennas

    def test_h_is_finite(self):
        coeff, delays = _make_cir(snap=1, tx=1, rx=1, rx_ant=4, tx_ant=4, path=3)
        link = LinkConfig(reciprocity_applied=False)
        ch = build_mimo_cfr_from_cir(coeff, delays, link, 30000.0, 48)
        import torch
        h = cfr_to_pusch_perfect_h(ch, 0, 0, 0, 14)
        assert torch.all(torch.isfinite(h))

    def test_not_broadcast_across_antenna_pairs(self):
        """Verify different antenna pairs have different channel values."""
        coeff, delays = _make_cir(snap=1, tx=1, rx=1, rx_ant=4, tx_ant=4, path=3, seed=123)
        link = LinkConfig(reciprocity_applied=False)
        ch = build_mimo_cfr_from_cir(coeff, delays, link, 30000.0, 48)
        h = cfr_to_pusch_perfect_h(ch, 0, 0, 0, 14)
        # Extract CFR for different antenna pairs
        h_np = h[0, 0, :, 0, :, 0, 0].cpu().numpy()  # [rx_ant, tx_ant]
        # Check that not all pairs are identical (should be different channels)
        # Use a generous tolerance since some pairs may be similar
        variations = np.std(np.abs(h_np))
        assert variations > 0, "All antenna pairs have identical channel gains"


class TestPUSCHHToCFREst:
    """Tests for pusch_h_to_cfr_est."""

    def test_shape_4x4(self):
        import torch
        h = torch.randn(1, 1, 4, 1, 4, 14, 48, dtype=torch.complex64)
        cfr_slice = pusch_h_to_cfr_est(h)
        assert cfr_slice.ndim == 3
        assert cfr_slice.shape == (4, 4, 48)
        assert cfr_slice.dtype == np.complex64

    def test_shape_asymmetric(self):
        import torch
        h = torch.randn(1, 1, 8, 1, 2, 14, 48, dtype=torch.complex64)
        cfr_slice = pusch_h_to_cfr_est(h)
        assert cfr_slice.shape == (8, 2, 48)


class TestReverseReciprocityCFR:
    """Tests for reverse_reciprocity_cfr."""

    def test_roundtrip(self):
        """reverse_reciprocity_cfr on UL CFR restores DL shape."""
        snap, tx, rx, rx_ant, tx_ant, sc = 1, 2, 3, 4, 5, 48
        # UL CFR: [snap, ul_tx, ul_rx, ul_rx_ant, ul_tx_ant, subcarrier]
        ul_cfr = np.random.randn(snap, tx, rx, rx_ant, tx_ant, sc).astype(
            np.complex64
        ) + 1j * np.random.randn(snap, tx, rx, rx_ant, tx_ant, sc).astype(np.complex64)

        # Reverse to DL: [snap, tx, rx, rx_ant, tx_ant, subcarrier]
        dl_cfr = reverse_reciprocity_cfr(ul_cfr)
        assert dl_cfr.shape == (snap, rx, tx, tx_ant, rx_ant, sc)

        # Re-apply to get back UL
        ul_cfr2 = reverse_reciprocity_cfr(dl_cfr)
        assert ul_cfr2.shape == ul_cfr.shape
        np.testing.assert_array_equal(ul_cfr, ul_cfr2)
