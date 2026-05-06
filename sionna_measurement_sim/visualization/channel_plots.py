"""CFR / channel-transfer visualization helpers."""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


def plot_cfr_magnitude(hdf5_path: str | Path, output_path: str | Path) -> Path:
    """Plot |CFR| vs frequency for the first few TX-RX links (max 4)."""

    hdf5_path = Path(hdf5_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as h5:
        cfr = h5["/channel/truth/cfr"][()]  # (num_tx, num_rx, ..., num_sub)
        freqs = h5["/frequency/frequencies_hz"][()]

    # Squeeze singleton dimensions, then ensure at least 2D
    cfr_sq = cfr.squeeze()
    if cfr_sq.ndim < 2:
        cfr_sq = cfr_sq.reshape(-1, cfr_sq.shape[-1])

    # Collapse everything except the last axis (subcarriers) into a flat list of links
    # After squeeze, shape is either (num_tx, num_rx, num_sub) or (num_tx, num_sub) or (num_sub,)
    link_mags = []
    if cfr_sq.ndim == 2:
        # Each row is a link
        for i in range(min(cfr_sq.shape[0], 4)):
            link_mags.append((f"Link {i}", 20.0 * np.log10(np.abs(cfr_sq[i, :]) + 1e-30)))
    else:
        # 3D: (num_tx, num_rx, num_sub)
        count = 0
        for tx_idx in range(cfr_sq.shape[0]):
            for rx_idx in range(cfr_sq.shape[1]):
                if count >= 4:
                    break
                mag_db = 20.0 * np.log10(np.abs(cfr_sq[tx_idx, rx_idx, :]) + 1e-30)
                link_mags.append((f"TX{tx_idx}-RX{rx_idx}", mag_db))
                count += 1
            if count >= 4:
                break

    figure, axis = plt.subplots(figsize=(8, 5))
    for label, mag_db in link_mags:
        axis.plot(freqs * 1e-9, mag_db, label=label)

    axis.set_xlabel("Frequency [GHz]")
    axis.set_ylabel("|CFR| [dB]")
    axis.set_title("CFR Magnitude")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path
