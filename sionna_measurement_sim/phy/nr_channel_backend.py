"""Pluggable NR PUSCH channel backends.

Each backend encapsulates channel building, perfect-CSI extraction, and
channel application.  This lets the main observation function swap between
the current ``ApplyOFDMChannel`` route and the official
``CIRDataset + OFDMChannel`` route without touching per-link processing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from sionna_measurement_sim.domain.link import LinkConfig
from sionna_measurement_sim.phy.nr_mimo_channel import (
    PUSCHMIMOChannel,
    build_mimo_cfr_from_cir,
    cfr_to_pusch_perfect_h,
)

# ── result type ───────────────────────────────────────────────────────


class ChannelApplyResult:
    """Result of applying the MIMO OFDM channel.

    Attributes
    ----------
    y : torch.Tensor
        Received signal ``[batch, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]``.
    h : torch.Tensor
        Channel frequency response that was applied
        ``[batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols, fft_size]``.
    """

    __slots__ = ("y", "h")

    def __init__(self, y: torch.Tensor, h: torch.Tensor) -> None:
        self.y = y
        self.h = h


# ── ApplyOFDMChannel backend (current stable) ───────────────────────────


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
        """Perfect-CSI tensor for a single link.

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
        resource_grid: Any = None,  # unused by this backend
    ) -> torch.Tensor:
        """Apply MIMO OFDM channel to TX signal ``x``.

        Returns ``[batch, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]``.
        """
        if self._apply is None:
            from sionna.phy.channel import ApplyOFDMChannel
            self._apply = ApplyOFDMChannel()

        h = self.perfect_h(snap_idx, ul_tx_idx, ul_rx_idx, num_ofdm_symbols)
        return self._apply(x, h, no)

    def apply_with_h(
        self,
        x: torch.Tensor,
        no: torch.Tensor,
        snap_idx: int = 0,
        ul_tx_idx: int = 0,
        ul_rx_idx: int = 0,
        num_ofdm_symbols: int = 14,
        resource_grid: Any = None,
    ) -> ChannelApplyResult:
        """Apply channel and return both ``y`` and ``h``.

        The returned ``h`` is the channel tensor that was applied
        (from pre-computed CFR via ``cfr_to_pusch_perfect_h``).
        """
        if self._apply is None:
            from sionna.phy.channel import ApplyOFDMChannel
            self._apply = ApplyOFDMChannel()

        h = self.perfect_h(snap_idx, ul_tx_idx, ul_rx_idx, num_ofdm_symbols)
        y = self._apply(x, h, no)
        return ChannelApplyResult(y=y, h=h)

    def apply_full_with_h(
        self,
        x: torch.Tensor,
        no: torch.Tensor,
        snap_idx: int = 0,
        num_ofdm_symbols: int = 14,
        resource_grid: Any = None,
    ) -> ChannelApplyResult:
        """Apply full multi-TX/RX MIMO channel (MU-MIMO).

        Uses ``cfr_to_full_mimo_h`` for the channel tensor.
        """
        from sionna_measurement_sim.phy.nr_mimo_channel import (
            PUSCHMIMOChannel,
            cfr_to_full_mimo_h,
        )

        if self._apply is None:
            from sionna.phy.channel import ApplyOFDMChannel
            self._apply = ApplyOFDMChannel()

        channel = PUSCHMIMOChannel(
            cfr=self._channel.cfr,
            num_snap=self._channel.num_snap,
            num_ul_tx=self._channel.num_ul_tx,
            num_ul_rx=self._channel.num_ul_rx,
            num_ul_tx_ant=self._channel.num_ul_tx_ant,
            num_ul_rx_ant=self._channel.num_ul_rx_ant,
            num_subcarriers=self._channel.num_subcarriers,
            reciprocity_applied=self._channel.reciprocity_applied,
        )
        h_full = cfr_to_full_mimo_h(channel, snap_idx, num_ofdm_symbols)
        y = self._apply(x, h_full, no)
        return ChannelApplyResult(y=y, h=h_full)


# ── CIRDataset + OFDMChannel backend ──────────────────────────────────


class CIRDatasetOFDMChannelBackend:
    """Channel backend using the official ``CIRDataset + OFDMChannel`` API.

    Follows the Sionna RT link-level tutorial pattern:
    project CIR → UL conversion → ``CIRDataset`` →
    ``OFDMChannel(return_channel=True)``.

    The pre-computed CFR (via ``build_mimo_cfr_from_cir``) is still stored
    for the ``cfr`` property (NMSE, shape consistency).  Channel application
    goes through ``OFDMChannel`` per (snap, ul_tx, ul_rx) link.
    """

    def __init__(
        self,
        cir_ul: np.ndarray,
        tau_ul: np.ndarray,
        cfr: np.ndarray,
        num_snap: int,
        num_ul_tx: int,
        num_ul_rx: int,
        num_ul_rx_ant: int,
        num_ul_tx_ant: int,
        num_paths: int,
        subcarrier_spacing_hz: float,
        num_subcarriers: int,
        reciprocity_applied: bool,
    ) -> None:
        self._cir_ul = cir_ul  # UL-convention CIR coefficients
        self._tau_ul = tau_ul  # UL-convention CIR delays
        self._cfr = cfr
        self._num_snap = num_snap
        self._num_ul_tx = num_ul_tx
        self._num_ul_rx = num_ul_rx
        self._num_ul_rx_ant = num_ul_rx_ant
        self._num_ul_tx_ant = num_ul_tx_ant
        self._num_paths = num_paths
        self._sc_spacing_hz = subcarrier_spacing_hz
        self._num_subcarriers = num_subcarriers
        self._reciprocity_applied = reciprocity_applied

    # ── factory ──────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        cir_coefficients: np.ndarray,
        cir_delays_s: np.ndarray,
        link_config: LinkConfig,
        subcarrier_spacing_hz: float,
        num_subcarriers: int,
    ) -> CIRDatasetOFDMChannelBackend:
        """Build backend from project CIR arrays.

        Converts to UL convention (with or without TDD reciprocity),
        then stores both the UL CIR and a pre-computed CFR for the
        ``cfr`` property.
        """
        # Apply TDD reciprocity if configured
        reciprocity_applied = False
        if (
            link_config.reciprocity_mode == "transpose_rt_channel"
            and link_config.reciprocity_applied
        ):
            try:
                from sionna_measurement_sim.phy.reciprocity import (
                    apply_tdd_reciprocity_cir,
                )
                cir_coefficients = apply_tdd_reciprocity_cir(cir_coefficients)
                cir_delays_s = apply_tdd_reciprocity_cir(cir_delays_s)
                reciprocity_applied = True
            except ImportError:
                pass

        # Convert to UL convention if reciprocity not applied
        if not reciprocity_applied:
            cir_ul = np.transpose(cir_coefficients, (0, 2, 1, 4, 3, 5))
            tau_ul = np.transpose(cir_delays_s, (0, 2, 1, 4, 3, 5))
        else:
            cir_ul = cir_coefficients
            tau_ul = cir_delays_s

        num_snap = cir_ul.shape[0]
        num_ul_tx = cir_ul.shape[1]
        num_ul_rx = cir_ul.shape[2]
        num_ul_rx_ant = cir_ul.shape[3]
        num_ul_tx_ant = cir_ul.shape[4]
        num_paths = cir_ul.shape[5]

        # CIRDataset expects tau: [num_rx, num_tx, num_paths] (link-level,
        # not per-antenna-pair).  Real RT data has antenna-dependent delays.
        # We compute per-link median delay as the shared approximation.
        # The OFDMChannel output may differ slightly from the pre-computed
        # per-antenna-pair CFR; this is the standard Sionna link-level tradeoff.

        # Pre-compute CFR (same as ApplyOFDMChannelBackend path)
        from sionna_measurement_sim.phy.nr_mimo_channel import (
            _cir_to_cfr_internal,
        )
        cfr = _cir_to_cfr_internal(
            cir_ul, tau_ul, subcarrier_spacing_hz, num_subcarriers,
        )

        return cls(
            cir_ul=cir_ul,
            tau_ul=tau_ul,
            cfr=cfr,
            num_snap=num_snap,
            num_ul_tx=num_ul_tx,
            num_ul_rx=num_ul_rx,
            num_ul_rx_ant=num_ul_rx_ant,
            num_ul_tx_ant=num_ul_tx_ant,
            num_paths=num_paths,
            subcarrier_spacing_hz=subcarrier_spacing_hz,
            num_subcarriers=num_subcarriers,
            reciprocity_applied=reciprocity_applied,
        )

    # ── properties ───────────────────────────────────────────────────

    @property
    def cfr(self) -> np.ndarray:
        return self._cfr

    @property
    def num_snap(self) -> int:
        return self._num_snap

    @property
    def num_ul_tx(self) -> int:
        return self._num_ul_tx

    @property
    def num_ul_rx(self) -> int:
        return self._num_ul_rx

    @property
    def num_ul_tx_ant(self) -> int:
        return self._num_ul_tx_ant

    @property
    def num_ul_rx_ant(self) -> int:
        return self._num_ul_rx_ant

    @property
    def num_subcarriers(self) -> int:
        return self._num_subcarriers

    @property
    def reciprocity_applied(self) -> bool:
        return self._reciprocity_applied

    # ── channel operations ───────────────────────────────────────────

    def _make_cir_generator(self, snap_idx: int, ul_tx_idx: int, ul_rx_idx: int):
        """Create a single-batch CIR generator for one (snap, ul_tx, ul_rx)."""

        # Extract CIR slice for this link
        # UL CIR: [snap, ul_tx, ul_rx, ul_rx_ant, ul_tx_ant, path]
        a_slice = self._cir_ul[
            snap_idx, ul_tx_idx, ul_rx_idx, ...
        ]  # [ul_rx_ant, ul_tx_ant, path]
        tau_slice = np.median(
            self._tau_ul[snap_idx, ul_tx_idx, ul_rx_idx, :, :, :],
            axis=(0, 1),
        )  # [path] — median across antenna pairs

        def _gen():
            # CIRDataset expects:
            #   a:   [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
            #   tau: [num_rx, num_tx, num_paths]
            a = torch.as_tensor(a_slice, dtype=torch.complex64)
            a = a.unsqueeze(0).unsqueeze(2).unsqueeze(-1)
            # → [1, ul_rx_ant, 1, ul_tx_ant, path, 1]
            tau = torch.as_tensor(tau_slice, dtype=torch.float32)
            tau = tau.reshape(1, 1, -1)  # [1, 1, path]
            yield a, tau

        return _gen

    def perfect_h(
        self,
        snap_idx: int = 0,
        ul_tx_idx: int = 0,
        ul_rx_idx: int = 0,
        num_ofdm_symbols: int = 14,
    ) -> torch.Tensor:
        """Perfect-CSI tensor from pre-computed CFR.

        Uses the same ``cfr_to_pusch_perfect_h`` path as
        ``ApplyOFDMChannelBackend`` for consistency.
        """
        channel = PUSCHMIMOChannel(
            cfr=self._cfr,
            num_snap=self._num_snap,
            num_ul_tx=self._num_ul_tx,
            num_ul_rx=self._num_ul_rx,
            num_ul_tx_ant=self._num_ul_tx_ant,
            num_ul_rx_ant=self._num_ul_rx_ant,
            num_subcarriers=self._num_subcarriers,
            reciprocity_applied=self._reciprocity_applied,
        )
        return cfr_to_pusch_perfect_h(
            channel,
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
        resource_grid: Any = None,
    ) -> torch.Tensor:
        """Apply MIMO OFDM channel via ``OFDMChannel`` with ``CIRDataset``.

        Creates a fresh ``CIRDataset`` and ``OFDMChannel`` per call
        (per-link mode).  ``resource_grid`` must be the PUSCH resource grid.
        """
        from sionna.phy.channel import CIRDataset, OFDMChannel

        if resource_grid is None:
            raise ValueError(
                "CIRDatasetOFDMChannelBackend.apply() requires resource_grid"
            )

        generator = self._make_cir_generator(snap_idx, ul_tx_idx, ul_rx_idx)
        dataset = CIRDataset(
            cir_generator=generator,
            batch_size=1,
            num_rx=1,
            num_rx_ant=self._num_ul_rx_ant,
            num_tx=1,
            num_tx_ant=self._num_ul_tx_ant,
            num_paths=self._num_paths,
            num_time_steps=1,
        )
        ofdm_ch = OFDMChannel(
            dataset, resource_grid,
            normalize_channel=False,
            return_channel=True,
        )
        y, h = ofdm_ch(x, no)
        return y

    def apply_with_h(
        self,
        x: torch.Tensor,
        no: torch.Tensor,
        snap_idx: int = 0,
        ul_tx_idx: int = 0,
        ul_rx_idx: int = 0,
        num_ofdm_symbols: int = 14,
        resource_grid: Any = None,
    ) -> ChannelApplyResult:
        """Apply channel via ``OFDMChannel`` and return both ``y`` and ``h``.

        The returned ``h`` comes directly from
        ``OFDMChannel(return_channel=True)``, closing the official
        backend loop for perfect CSI.
        """
        from sionna.phy.channel import CIRDataset, OFDMChannel

        if resource_grid is None:
            raise ValueError(
                "CIRDatasetOFDMChannelBackend.apply_with_h() requires resource_grid"
            )

        generator = self._make_cir_generator(snap_idx, ul_tx_idx, ul_rx_idx)
        dataset = CIRDataset(
            cir_generator=generator,
            batch_size=1,
            num_rx=1,
            num_rx_ant=self._num_ul_rx_ant,
            num_tx=1,
            num_tx_ant=self._num_ul_tx_ant,
            num_paths=self._num_paths,
            num_time_steps=1,
        )
        ofdm_ch = OFDMChannel(
            dataset, resource_grid,
            normalize_channel=False,
            return_channel=True,
        )
        y, h = ofdm_ch(x, no)
        return ChannelApplyResult(y=y, h=h)

    def apply_full_with_h(
        self,
        x: torch.Tensor,
        no: torch.Tensor,
        snap_idx: int = 0,
        num_ofdm_symbols: int = 14,
        resource_grid: Any = None,
    ) -> ChannelApplyResult:
        """Full multi-TX/RX MIMO channel (MU-MIMO) — not yet supported.

        The CIRDataset backend currently only handles per-link SU-MIMO.
        Full MU-MIMO would require a multi-TX/RX generator and a
        single ``OFDMChannel`` covering all links.
        """
        raise NotImplementedError(
            "CIRDatasetOFDMChannelBackend does not yet support "
            "full multi-TX/RX MU-MIMO. Use channel_backend='apply_ofdm' "
            "for MU-MIMO."
        )


# ── backend factory ─────────────────────────────────────────────────────


def create_channel_backend(
    cir_coefficients: np.ndarray,
    cir_delays_s: np.ndarray,
    link_config: LinkConfig,
    subcarrier_spacing_hz: float,
    num_subcarriers: int,
    *,
    backend_name: str = "apply_ofdm",
) -> ApplyOFDMChannelBackend | CIRDatasetOFDMChannelBackend:
    """Create a channel backend by name.

    Supported backends:
    - ``"apply_ofdm"`` — :class:`ApplyOFDMChannelBackend` (current stable)
    - ``"cir_dataset_ofdm"`` — :class:`CIRDatasetOFDMChannelBackend`
    """
    if backend_name == "apply_ofdm":
        return ApplyOFDMChannelBackend.build(
            cir_coefficients, cir_delays_s, link_config,
            subcarrier_spacing_hz, num_subcarriers,
        )
    if backend_name == "cir_dataset_ofdm":
        return CIRDatasetOFDMChannelBackend.build(
            cir_coefficients, cir_delays_s, link_config,
            subcarrier_spacing_hz, num_subcarriers,
        )
    raise NotImplementedError(
        f"Unknown channel_backend: {backend_name!r}. "
        f"Supported: 'apply_ofdm', 'cir_dataset_ofdm'"
    )
