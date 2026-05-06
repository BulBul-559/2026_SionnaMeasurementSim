"""Path sample visualization smoke helpers."""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt


def plot_path_samples(hdf5_path: str | Path, output_path: str | Path) -> Path:
    """Plot sampled path polylines from an HDF5 result."""

    hdf5_path = Path(hdf5_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as h5:
        vertices = h5["paths/samples/vertices_m"][()]
        vertex_count = h5["paths/samples/vertex_count"][()]

    figure = plt.figure(figsize=(6, 4))
    axis = figure.add_subplot(111, projection="3d")
    for sample in range(vertices.shape[0]):
        for path in range(vertices.shape[1]):
            count = int(vertex_count[sample, path])
            if count <= 1:
                continue
            points = vertices[sample, path, :count]
            axis.plot(points[:, 0], points[:, 1], points[:, 2], alpha=0.65)

    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_zlabel("z [m]")
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def plot_delay_doppler(hdf5_path: str | Path, output_path: str | Path) -> Path:
    """Delay-Doppler scatter plot from path samples."""

    hdf5_path = Path(hdf5_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as h5:
        tau = h5["paths/samples/tau_s"][()]
        doppler = h5["paths/samples/doppler_hz"][()]
        path_gain = h5["paths/samples/path_gain_db"][()]

    figure, axis = plt.subplots(figsize=(7, 5))
    points = axis.scatter(
        tau.ravel() * 1e9, doppler.ravel(),
        c=path_gain.ravel(), cmap="viridis", alpha=0.7, s=20,
    )
    axis.set_xlabel("Delay [ns]")
    axis.set_ylabel("Doppler [Hz]")
    axis.set_title("Delay-Doppler Scatter")
    figure.colorbar(points, ax=axis, label="Path Gain [dB]")
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def plot_topology(hdf5_path: str | Path, output_path: str | Path) -> Path:
    """2D top-down topology plot of TX (red triangles) and RX (blue circles)."""

    hdf5_path = Path(hdf5_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as h5:
        tx_pos = h5["/topology/tx_positions_m"][()]
        rx_pos = h5["/topology/rx_positions_m"][()]

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(tx_pos[:, 0], tx_pos[:, 1], marker="^", color="red", s=60, label="TX")
    axis.scatter(rx_pos[:, 0], rx_pos[:, 1], marker="o", color="blue", s=30, label="RX")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.set_title("Topology")
    axis.legend()
    axis.set_aspect("equal")
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path

