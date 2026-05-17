"""Dimension-aware shape contracts for the Sionna RT adapter layer.

Each function documents exact input-to-output axis semantics, so callers
can verify dimension conventions at a glance without tracing transpose
calls.
"""

from __future__ import annotations

import numpy as np


def to_project_cfr(
    raw_cfr: np.ndarray, num_time_steps: int
) -> tuple[np.ndarray, np.ndarray | None]:
    """Convert Sionna CFR to TX-first 5D with optional time-step snapshots.

    Input  (Sionna):  [rx, rx_ant, tx, tx_ant, time, subcarrier]        (6D)
    Output (project):
      - cfr_5d:       [tx, rx, rx_ant, tx_ant, subcarrier]             (5D)
      - snapshots:    [time, tx, rx, rx_ant, tx_ant, subcarrier] | None (6D)

    When *num_time_steps == 1* the snapshots array is ``None``.
    """
    cfr = np.asarray(raw_cfr)
    if cfr.ndim != 6:
        msg = f"Sionna cfr must have rank 6, got {cfr.shape}"
        raise ValueError(msg)
    if cfr.shape[4] != num_time_steps:
        msg = f"Expected {num_time_steps} time steps, got {cfr.shape[4]}"
        raise ValueError(msg)
    # [rx, rx_ant, tx, tx_ant, time, subcarrier] -> [tx, rx, rx_ant, tx_ant, time, subcarrier]
    cfr_tx_first = np.transpose(cfr, (2, 0, 1, 3, 4, 5))
    cfr_5d = cfr_tx_first[..., 0, :]  # first time step -> 5D
    if num_time_steps == 1:
        return cfr_5d, None
    # 6D snapshots: [time, tx, rx, rx_ant, tx_ant, subcarrier]
    snapshots = np.transpose(cfr_tx_first, (4, 0, 1, 2, 3, 5))
    return cfr_5d, snapshots


def to_project_path_scalar(value: np.ndarray, name: str) -> np.ndarray:
    """Convert a Sionna path scalar to TX-first 5D ``[tx, rx, rx_ant, tx_ant, path]``.

    Sionna shapes handled:
      - With antennas:  ``[rx, rx_ant, tx, tx_ant, path]``         (5D)
      - No antennas:    ``[rx, tx, path]``                          (3D)

    Raises ``ValueError`` for unrecognised dimensionality.
    """
    if value.ndim == 5:
        return np.transpose(value, (2, 0, 1, 3, 4))
    if value.ndim == 3:
        expanded = value[:, np.newaxis, :, np.newaxis, :]
        return np.transpose(expanded, (2, 0, 1, 3, 4))
    msg = f"Unsupported Sionna path scalar shape for {name}: {value.shape}"
    raise ValueError(msg)


def to_project_path_interaction(value: np.ndarray, name: str) -> np.ndarray:
    """Convert Sionna interaction data to TX-first 6D ``[tx, rx, rx_ant, tx_ant, path, depth]``.

    Sionna shapes handled:
      - With antennas:  ``[depth, rx, rx_ant, tx, tx_ant, path]``   (6D)
      - No antennas:    ``[depth, rx, tx, path]``                    (4D)

    Output is cast to ``np.uint32``.
    """
    if value.ndim == 6:
        return np.transpose(value, (3, 1, 2, 4, 5, 0)).astype(np.uint32)
    if value.ndim == 4:
        expanded = value[:, :, np.newaxis, :, np.newaxis, :]
        return np.transpose(expanded, (3, 1, 2, 4, 5, 0)).astype(np.uint32)
    msg = f"Unsupported Sionna interaction shape for {name}: {value.shape}"
    raise ValueError(msg)


def to_project_path_vertices(value: np.ndarray) -> np.ndarray:
    """Convert Sionna vertices to TX-first 7D ``[tx, rx, rx_ant, tx_ant, path, depth, 3]``.

    Sionna shapes handled:
      - With antennas:  ``[depth, rx, rx_ant, tx, tx_ant, path, 3]``  (7D)
      - No antennas:    ``[depth, rx, tx, path, 3]``                   (5D)

    Output is cast to ``np.float32``.
    """
    if value.ndim == 7:
        return np.transpose(value, (3, 1, 2, 4, 5, 0, 6)).astype(np.float32)
    if value.ndim == 5:
        expanded = value[:, :, np.newaxis, :, np.newaxis, :, :]
        return np.transpose(expanded, (3, 1, 2, 4, 5, 0, 6)).astype(np.float32)
    msg = f"Unsupported Sionna vertices shape: {value.shape}"
    raise ValueError(msg)


def to_project_cir(
    raw_a: np.ndarray,
    raw_tau: np.ndarray,
    valid_5d: np.ndarray,
    num_time_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert Sionna CIR to TX-first 6D.

    Sionna raw: a [rx, rx_ant, tx, tx_ant, path, time],
                tau [rx, rx_ant, tx, tx_ant, path].
    Returns (coefficients, delays_s, valid) — all
    [snapshot, tx, rx, rx_ant, tx_ant, path].
    """
    a = np.asarray(raw_a)
    if a.ndim == 8 and a.shape[0] == 2:  # real/imag split
        a = a[0] + 1j * a[1]
    if a.ndim == 7:
        a = a[0]  # drop batch dim
    tau = np.asarray(raw_tau)
    if tau.ndim == 6:
        tau = tau[0]
    if tau.ndim == 3:
        tau = tau[:, np.newaxis, :, np.newaxis, :]
    if tau.ndim != 5:
        msg = f"Sionna CIR tau must be rank 5 or synthetic rank 3, got {tau.shape}"
        raise ValueError(msg)

    a_tx = np.transpose(a, (5, 2, 0, 1, 3, 4))  # [time, tx, rx, rx_ant, tx_ant, path]
    tau_tx = np.transpose(tau, (2, 0, 1, 3, 4))  # [tx, rx, rx_ant, tx_ant, path]
    if tau_tx.shape != valid_5d.shape:
        tau_tx = np.broadcast_to(tau_tx, valid_5d.shape)

    tau_6d = np.broadcast_to(
        tau_tx[np.newaxis, ...],
        (num_time_steps, *tau_tx.shape),
    ).astype(np.float32, copy=False)

    valid_6d = np.broadcast_to(
        valid_5d[np.newaxis, ...],
        (num_time_steps, *valid_5d.shape),
    ).copy()

    a_clean = np.where(valid_6d, a_tx, np.zeros_like(a_tx))
    return a_clean.astype(np.complex64, copy=False), tau_6d, valid_6d


def assert_tx_first_scalar(arr: np.ndarray, name: str) -> None:
    """Raise if *arr* does not have expected TX-first 5D shape.

    Expected: ``[tx, rx, rx_ant, tx_ant, path]``.
    """
    if arr.ndim != 5:
        msg = f"{name} must be 5D [tx, rx, rx_ant, tx_ant, path], got shape {arr.shape}"
        raise ValueError(msg)
