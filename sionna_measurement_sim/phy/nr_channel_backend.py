"""Pluggable NR PUSCH channel backends.

Each backend encapsulates channel building, perfect-CSI extraction, and
channel application.  This lets the main observation function swap between
the current ``ApplyOFDMChannel`` route and the official
``CIRDataset + OFDMChannel`` route without touching per-link processing.
"""

from __future__ import annotations

import numpy as np
import torch

from sionna_measurement_sim.domain.link import LinkConfig
from sionna_measurement_sim.phy.nr_mimo_channel import (
    PUSCHMIMOChannel,
    build_mimo_cfr_from_cir,
    cfr_to_pusch_perfect_h,
)

# ── backend interface (duck-typed, not a strict Protocol) ──────────────


class ApplyOFDMChannelBackend:
    """Channel backend using ``ApplyOFDMChannel`` on pre-computed CFR.

    This is the current stable backend.  It converts the project CIR to a
    6-D CFR via ``cir_to_ofdm_channel``, then uses ``ApplyOFDMChannel``
    to apply the frequency-domain channel per (snap, ul_tx, ul_rx) link.
    """

    def __init__(self, channel: PUSCHMIMOChannel) -> None:
        self._channel = channel
        self._apply = None  # lazy init

    # ── factory ──────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        cir_coefficients: np.ndarray,
        cir_delays_s: np.ndarray,
        link_config: LinkConfig,
        subcarrier_spacing_hz: float,
        num_subcarriers: int,
    ) -> ApplyOFDMChannelBackend:
        """Build backend from project CIR arrays."""
        channel = build_mimo_cfr_from_cir(
            cir_coefficients, cir_delays_s, link_config,
            subcarrier_spacing_hz, num_subcarriers,
        )
        return cls(channel)

    # ── properties ───────────────────────────────────────────────────

    @property
    def cfr(self) -> np.ndarray:
        """6-D CFR in UL convention."""
        return self._channel.cfr

    @property
    def num_snap(self) -> int:
        return self._channel.num_snap

    @property
    def num_ul_tx(self) -> int:
        return self._channel.num_ul_tx

    @property
    def num_ul_rx(self) -> int:
        return self._channel.num_ul_rx

    @property
    def num_ul_tx_ant(self) -> int:
        return self._channel.num_ul_tx_ant

    @property
    def num_ul_rx_ant(self) -> int:
        return self._channel.num_ul_rx_ant

    @property
    def num_subcarriers(self) -> int:
        return self._channel.num_subcarriers

    @property
    def reciprocity_applied(self) -> bool:
        return self._channel.reciprocity_applied

    # ── channel operations ───────────────────────────────────────────

    def perfect_h(
        self,
        snap_idx: int = 0,
        ul_tx_idx: int = 0,
        ul_rx_idx: int = 0,
        num_ofdm_symbols: int = 14,
    ) -> torch.Tensor:
        """Perfect-CSI tensor for a single (snap, ul_tx, ul_rx) link.

        Returns tensor of shape
        ``[1, 1, num_ul_rx_ant, 1, num_ul_tx_ant, num_ofdm_symbols, fft_size]``.
        """
        return cfr_to_pusch_perfect_h(
            self._channel,
            snap_idx=snap_idx,
            ul_tx_idx=ul_tx_idx,
            ul_rx_idx=ul_rx_idx,
            num_ofdm_symbols=num_ofdm_symbols,
        )

    def apply(
        self,
        x: torch.Tensor,
        no: torch.Tensor,
        snap_idx: int = 0,
        ul_tx_idx: int = 0,
        ul_rx_idx: int = 0,
        num_ofdm_symbols: int = 14,
    ) -> torch.Tensor:
        """Apply MIMO OFDM channel to TX signal ``x``.

        Parameters
        ----------
        x : torch.Tensor
            PUSCH TX signal, shape ``[batch, num_tx, num_streams, num_ofdm_symbols, fft_size]``.
        no : torch.Tensor
            Noise power (scalar or broadcastable).
        snap_idx, ul_tx_idx, ul_rx_idx : int
            Link indices into the pre-computed CFR.
        num_ofdm_symbols : int
            Number of OFDM symbols in the slot.

        Returns
        -------
        y : torch.Tensor
            ``[batch, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]``.
        """
        if self._apply is None:
            from sionna.phy.channel import ApplyOFDMChannel

            self._apply = ApplyOFDMChannel()

        h = self.perfect_h(snap_idx, ul_tx_idx, ul_rx_idx, num_ofdm_symbols)
        return self._apply(x, h, no)


# ── backend factory ─────────────────────────────────────────────────────


def create_channel_backend(
    cir_coefficients: np.ndarray,
    cir_delays_s: np.ndarray,
    link_config: LinkConfig,
    subcarrier_spacing_hz: float,
    num_subcarriers: int,
    *,
    backend_name: str = "apply_ofdm",
) -> ApplyOFDMChannelBackend:
    """Create a channel backend by name.

    Currently only ``"apply_ofdm"`` is supported.
    Future: ``"cir_dataset_ofdm"``.
    """
    if backend_name == "apply_ofdm":
        return ApplyOFDMChannelBackend.build(
            cir_coefficients, cir_delays_s, link_config,
            subcarrier_spacing_hz, num_subcarriers,
        )
    raise NotImplementedError(
        f"Unknown channel_backend: {backend_name!r}. "
        f"Supported: 'apply_ofdm'"
    )
