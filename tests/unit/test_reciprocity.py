"""Tests for TDD reciprocity transforms."""

import numpy as np

from sionna_measurement_sim.phy.reciprocity import (
    apply_tdd_reciprocity,
    apply_tdd_reciprocity_cir,
)


class TestReciprocity:
    def test_cfr_5d_transform(self):
        # [tx, rx, rx_ant, tx_ant, sub] -> [tx, rx, tx_ant, rx_ant, sub]
        cfr = np.ones((2, 3, 4, 4, 64))  # 2tx, 3rx, 4x4 MIMO
        result = apply_tdd_reciprocity(cfr)
        assert result.shape == (3, 2, 4, 4, 64)  # swapped tx<->rx, rx_ant<->tx_ant

    def test_cir_6d_transform(self):
        cir = np.ones((1, 2, 3, 4, 4, 10))  # snap, tx, rx, rx_ant, tx_ant, path
        result = apply_tdd_reciprocity_cir(cir)
        assert result.shape == (1, 3, 2, 4, 4, 10)
