"""NR MIMO channel bridge: project CIR to Sionna PUSCH-compatible CFR.

Converts project CIR arrays (6-D [snap, tx, rx, rx_ant, tx_ant, path]) to
formats usable by Sionna ApplyOFDMChannel and PUSCHReceiver.  All dimension
mappings are explicit and testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from sionna_measurement_sim.domain.link import LinkConfig


@dataclass(frozen=True)
class PUSCHMIMOChannel:
    """MIMO channel bundle for PUSCH processing.

    Holds the frequency-domain channel response in project CFR format,
    plus metadata about the UL/DL orientation and reciprocity.
    """

    cfr: np.ndarray
    """6-D CFR in project orientation
    ``[snap, ul_tx, ul_rx, ul_rx_ant, ul_tx_ant, subcarrier]``
    where ul_tx = UE (was project rx), ul_rx = BS (was project tx)."""

    num_snap: int
    num_ul_tx: int
    num_ul_rx: int
    num_ul_tx_ant: int
    num_ul_rx_ant: int
    num_subcarriers: int
    reciprocity_applied: bool


def build_mimo_cfr_from_cir(
    cir_coefficients: np.ndarray,
    cir_delays_s: np.ndarray,
    link_config: LinkConfig,
    subcarrier_spacing_hz: float,
    num_subcarriers: int,
) -> PUSCHMIMOChannel:
    """Convert project CIR to CFR for use with PUSCH MIMO processing.

    Applies TDD reciprocity if configured, then converts the CIR to
    frequency domain via ``sionna.phy.channel.cir_to_ofdm_channel``.

    Parameters
    ----------
    cir_coefficients : np.ndarray
        6-D ``[snap, tx, rx, rx_ant, tx_ant, path]`` complex CIR.
    cir_delays_s : np.ndarray
        6-D ``[snap, tx, rx, rx_ant, tx_ant, path]`` float delays.
    link_config : LinkConfig
    subcarrier_spacing_hz : float
    num_subcarriers : int

    Returns
    -------
    PUSCHMIMOChannel
    """
    # 1. Apply TDD reciprocity for uplink
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

    # 2. Convert to UL convention if reciprocity not applied.
    #    DL CIR:  [snap, tx, rx, rx_ant, tx_ant, path]
    #    UL CIR:  [snap, ul_tx(=rx), ul_rx(=tx), ul_rx_ant(=tx_ant), ul_tx_ant(=rx_ant), path]
    #    Without reciprocity, swap (dim1↔dim2) and (dim3↔dim4) to get UL.
    if not reciprocity_applied:
        cir_a_ul = np.transpose(cir_coefficients, (0, 2, 1, 4, 3, 5))
        cir_tau_ul = np.transpose(cir_delays_s, (0, 2, 1, 4, 3, 5))
    else:
        cir_a_ul = cir_coefficients
        cir_tau_ul = cir_delays_s

    # 3. Extract shapes
    # UL CIR: [snap, ul_tx, ul_rx, ul_rx_ant, ul_tx_ant, path]
    num_snap, num_ul_tx, num_ul_rx = cir_a_ul.shape[:3]
    num_ul_rx_ant = cir_a_ul.shape[3]  # BS antennas
    num_ul_tx_ant = cir_a_ul.shape[4]  # UE antennas

    # 4. Convert CIR to CFR using Sionna (reuses same permutation as old _cir_to_cfr)
    cfr = _cir_to_cfr_internal(cir_a_ul, cir_tau_ul, subcarrier_spacing_hz, num_subcarriers)

    return PUSCHMIMOChannel(
        cfr=cfr,
        num_snap=num_snap,
        num_ul_tx=num_ul_tx,
        num_ul_rx=num_ul_rx,
        num_ul_tx_ant=num_ul_tx_ant,
        num_ul_rx_ant=num_ul_rx_ant,
        num_subcarriers=num_subcarriers,
        reciprocity_applied=reciprocity_applied,
    )


def cfr_to_pusch_perfect_h(
    channel: PUSCHMIMOChannel,
    snap_idx: int = 0,
    ul_tx_idx: int = 0,
    ul_rx_idx: int = 0,
    num_ofdm_symbols: int = 14,
) -> torch.Tensor:
    """Extract a single (ul_tx, ul_rx) MIMO CFR and reshape to PUSCH perfect-CSI ``h``.

    Returns a tensor with shape
    ``[1, 1, num_ul_rx_ant, 1, num_ul_tx_ant, num_ofdm_symbols, num_subcarriers]``
    suitable for ``PUSCHReceiver(..., channel_estimator="perfect")``.

    Parameters
    ----------
    channel : PUSCHMIMOChannel
    snap_idx : int
        Snapshot index (default 0).
    ul_tx_idx : int
        Uplink TX (UE) index in the post-reciprocity CIR.
    ul_rx_idx : int
        Uplink RX (BS) index in the post-reciprocity CIR.
    num_ofdm_symbols : int
        Number of OFDM symbols in the slot (default 14 for NR).

    Returns
    -------
    torch.Tensor
        Complex tensor of shape
        ``[1, 1, num_ul_rx_ant, 1, num_ul_tx_ant, num_ofdm_symbols, num_subcarriers]``.
    """
    # channel.cfr: [snap, ul_tx, ul_rx, ul_rx_ant, ul_tx_ant, subcarrier]
    # Extract single (snap, ul_tx, ul_rx) slice:
    h_slice = channel.cfr[snap_idx, ul_tx_idx, ul_rx_idx, ...]
    # h_slice: [num_ul_rx_ant, num_ul_tx_ant, subcarrier]

    h_t = torch.as_tensor(h_slice, dtype=torch.complex64)
    # Add batch, num_rx(=1), num_tx(=1) dims and expand subcarrier to symbols
    # Target: [batch=1, num_rx=1, num_rx_ant, num_tx=1, num_tx_ant, num_ofdm_symbols, fft_size]
    h_t = h_t.unsqueeze(0).unsqueeze(0).unsqueeze(3)  # [1, 1, rx_ant, 1, tx_ant, subcarrier]
    # Expand across OFDM symbols (static channel)
    h_t = h_t.unsqueeze(-2)  # [1, 1, rx_ant, 1, tx_ant, 1, subcarrier]
    h_t = h_t.expand(-1, -1, -1, -1, -1, num_ofdm_symbols, -1)
    return h_t


def cfr_to_full_mimo_h(
    channel: PUSCHMIMOChannel,
    snap_idx: int = 0,
    num_ofdm_symbols: int = 14,
) -> torch.Tensor:
    """Extract the full multi-TX/RX MIMO CFR and reshape to PUSCH perfect-CSI ``h``.

    Unlike :func:`cfr_to_pusch_perfect_h` which returns a single-link
    tensor, this returns a tensor covering ALL (ul_tx, ul_rx) pairs
    simultaneously, suitable for MU-MIMO joint PUSCH processing.

    Returns a tensor of shape
    ``[1, num_ul_rx, num_ul_rx_ant, num_ul_tx, num_ul_tx_ant,
      num_ofdm_symbols, num_subcarriers]``.
    """
    # channel.cfr: [snap, ul_tx, ul_rx, ul_rx_ant, ul_tx_ant, subcarrier]
    h_full = channel.cfr[snap_idx, ...]  # [ul_tx, ul_rx, ul_rx_ant, ul_tx_ant, subcarrier]

    # Permute to PUSCH h order:
    # [ul_rx, ul_rx_ant, ul_tx, ul_tx_ant, subcarrier]
    h_t = torch.as_tensor(h_full, dtype=torch.complex64)
    h_t = h_t.permute(1, 2, 0, 3, 4)  # ul_rx, ul_rx_ant, ul_tx, ul_tx_ant, sub

    # Add batch and symbol dims
    h_t = h_t.unsqueeze(0).unsqueeze(-2)  # [1, ul_rx, ul_rx_ant, ul_tx, ul_tx_ant, 1, sub]
    h_t = h_t.expand(-1, -1, -1, -1, -1, num_ofdm_symbols, -1)

    return h_t


def pusch_h_to_cfr_est(
    h: torch.Tensor,
) -> np.ndarray:
    """Convert a PUSCH perfect-CSI ``h`` tensor to a 3-D CFR slice.

    The returned array has shape
    ``[num_rx_ant, num_tx_ant, num_subcarriers]``
    suitable for assignment into the per-link position of
    ``/observation/cfr_est``.

    Parameters
    ----------
    h : torch.Tensor
        Shape ``[batch, num_rx, num_rx_ant, num_tx, num_tx_ant,
        num_ofdm_symbols, fft_size]``.

    Returns
    -------
    np.ndarray
        3-D complex64 ``[num_rx_ant, num_tx_ant, num_subcarriers]``.
    """
    h_np = h.cpu().numpy() if isinstance(h, torch.Tensor) else np.asarray(h)
    # Take first OFDM symbol (static channel) and first batch/rx/tx
    return h_np[0, 0, :, 0, :, 0, :].astype(np.complex64, copy=False)


def reverse_reciprocity_cfr(cfr: np.ndarray) -> np.ndarray:
    """Reverse TDD reciprocity to restore project orientation.

    ``[snap, ul_tx, ul_rx, ul_rx_ant, ul_tx_ant, subcarrier]``
    → ``[snap, tx, rx, rx_ant, tx_ant, subcarrier]`` (DL project view).
    """
    return np.transpose(cfr, (0, 2, 1, 4, 3, 5))


# ── internal helpers ────────────────────────────────────────────────────


def _cir_to_cfr_internal(
    cir_coefficients: np.ndarray,
    cir_delays: np.ndarray,
    subcarrier_spacing_hz: float,
    num_subcarriers: int,
) -> np.ndarray:
    """Convert 6-D CIR to 6-D CFR using Sionna cir_to_ofdm_channel.

    Input CIR is in POST-reciprocity UL convention:
      coefficients: [snap, ul_tx, ul_rx, ul_rx_ant, ul_tx_ant, path]
      delays:       [snap, ul_tx, ul_rx, ul_rx_ant, ul_tx_ant, path]

    Returns 6-D CFR in same convention:
      [snap, ul_tx, ul_rx, ul_rx_ant, ul_tx_ant, subcarrier]

    Uses the same permutation logic as the original _cir_to_cfr
    (permute 0,2,3,1,4,5) which maps project dims to Sionna (rx-first) dims.
    """
    from sionna.phy.channel import cir_to_ofdm_channel, subcarrier_frequencies

    a = torch.as_tensor(cir_coefficients, dtype=torch.complex64)
    tau = torch.as_tensor(cir_delays, dtype=torch.float32)

    # Sionna expects: a:  [..., rx, rx_ant, tx, tx_ant, path, 1]
    #                 tau: [..., rx, rx_ant, tx, tx_ant, path]
    # Input: [snap, ul_tx, ul_rx, ul_rx_ant, ul_tx_ant, path]
    # Permute (0,2,3,1,4,5) -> [snap, ul_rx, ul_rx_ant, ul_tx, ul_tx_ant, path]
    a_perm = a.permute(0, 2, 3, 1, 4, 5).unsqueeze(-1)
    tau_perm = tau.permute(0, 2, 3, 1, 4, 5)

    freqs = subcarrier_frequencies(num_subcarriers, subcarrier_spacing_hz)
    # cir_to_ofdm_channel returns: [snap, rx, rx_ant, tx, tx_ant, 1, subcarrier]
    h_f = cir_to_ofdm_channel(freqs, a_perm, tau_perm)
    h_f = h_f.squeeze(-2)  # [snap, ul_rx, ul_rx_ant, ul_tx, ul_tx_ant, subcarrier]

    # Back to project convention: permute (0,3,1,2,4,5)
    # -> [snap, ul_tx, ul_rx, ul_rx_ant, ul_tx_ant, subcarrier]
    h_f = h_f.permute(0, 3, 1, 2, 4, 5)
    return h_f.detach().cpu().numpy()
