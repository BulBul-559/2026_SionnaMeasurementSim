"""Path sample visualization smoke helper."""

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
