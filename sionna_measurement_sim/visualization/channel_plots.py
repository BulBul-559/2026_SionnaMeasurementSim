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

    # Flatten leading dims to [link, subcarrier], keeping the last axis
    sub = cfr.shape[-1]
    cfr_flat = np.reshape(cfr, (-1, sub))
    if cfr_flat.shape[0] == 0:
        # Empty CFR, create empty plot
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.text(0.5, 0.5, "No CFR data", ha="center", va="center", transform=axis.transAxes)
        figure.savefig(output_path)
        plt.close(figure)
        return output_path

    # Plot up to 4 links
    n_links = min(cfr_flat.shape[0], 4)
    link_mags = []
    for i in range(n_links):
        mag_db = 20.0 * np.log10(np.abs(cfr_flat[i, :]) + 1e-30)
        # Use original multi-dim indices for label
        idx = np.unravel_index(i, cfr.shape[:-1])
        label = "TX" + ",".join(str(d) for d in idx[:2])
        link_mags.append((label, mag_db))

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
