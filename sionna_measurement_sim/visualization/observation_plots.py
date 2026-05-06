"""Observed-signal evaluation visualization helpers."""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt


def plot_nmse_snr(hdf5_path: str | Path, output_path: str | Path) -> Path:
    """Scatter plot of NMSE vs SNR per link (if observation data exists)."""

    hdf5_path = Path(hdf5_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(7, 5))

    with h5py.File(hdf5_path, "r") as h5:
        # Graceful skip when observation group is absent
        if "observation" not in h5 or "evaluation" not in h5:
            axis.text(
                0.5, 0.5, "Observation / evaluation data not available",
                ha="center", va="center", transform=axis.transAxes, fontsize=12,
            )
            axis.set_title("NMSE vs SNR")
            figure.tight_layout()
            figure.savefig(output_path)
            plt.close(figure)
            return output_path

        snr_db = h5["/observation/snr_db"][()]
        nmse_db = h5["/evaluation/nmse_db"][()]

    # Flatten all leading dims
    snr_flat = snr_db.ravel()
    nmse_flat = nmse_db.ravel()

    axis.scatter(snr_flat, nmse_flat, alpha=0.6, s=20)
    axis.set_xlabel("SNR [dB]")
    axis.set_ylabel("NMSE [dB]")
    axis.set_title("NMSE vs SNR per Link")
    axis.grid(True, linestyle="--", alpha=0.4)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path
