"""TDD reciprocity transforms for CFR and CIR."""

from __future__ import annotations

import numpy as np


def apply_tdd_reciprocity(cfr: np.ndarray) -> np.ndarray:
    """Transform BS->UE truth to UE->BS uplink via TDD reciprocity.

    For CFR [tx, rx, rx_ant, tx_ant, subcarrier]:
    TX/RX roles swap: tx->rx, rx->tx, rx_ant<->tx_ant.
    Returns: [tx, rx, tx_ant, rx_ant, subcarrier] (uplink perspective)
    """
    return np.transpose(cfr, (1, 0, 3, 2, 4))


def apply_tdd_reciprocity_cir(cir: np.ndarray) -> np.ndarray:
    """Same for CIR [snapshot, tx, rx, rx_ant, tx_ant, path]."""
    return np.transpose(cir, (0, 2, 1, 4, 3, 5))
