"""Scene loading and array placement.

Element geometry must match Sionna's ``PlanarArray`` exactly, otherwise the spatial
phase across the 32 ports is wrong and every downstream comparison is meaningless.
Sionna places elements in its local y-z plane (see ``PlanarArray.__init__``):

    y =  d_h * j - (C - 1) * d_h / 2
    z = -d_v * i + (R - 1) * d_v / 2
    x =  0

with ``i`` the row index (0 = top), ``j`` the column index, and flat element index
``k = i * C + j``. The panel is then rotated by the BS yaw about +z, which is exactly
Sionna's ``rotation_matrix`` for angles ``(alpha, beta, gamma) = (yaw, 0, 0)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import AntennaSpec, PlanarArraySpec

__all__ = [
    "load_front3d_scene",
    "subset_link_scene",
    "panel_element_positions",
    "rotation_z",
    "build_link_scene",
    "mesh_stats",
]


def rotation_z(yaw_rad: float) -> np.ndarray:
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def panel_element_positions(
    center: np.ndarray,
    yaw_rad: float,
    spec: PlanarArraySpec,
) -> np.ndarray:
    """Global coordinates of the panel elements, in Sionna element order."""
    rows, cols = spec.num_rows, spec.num_cols
    d_v = d_h = spec.spacing_m

    ii, jj = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    y = d_h * jj - (cols - 1) * d_h / 2.0
    z = -d_v * ii + (rows - 1) * d_v / 2.0
    local = np.stack([np.zeros_like(y), y, z], axis=-1).reshape(-1, 3)

    return (rotation_z(yaw_rad) @ local.T).T + np.asarray(center, dtype=float)[None, :]


def load_front3d_scene(scene_xml: str | Path):
    """Load a Front3D Mitsuba/Sionna ``scene.xml`` as a DiffeRT triangle scene."""
    from differt.geometry import Scene  # noqa: PLC0415

    scene_xml = Path(scene_xml)
    if not scene_xml.is_file():
        raise FileNotFoundError(scene_xml)
    return Scene.load_xml(str(scene_xml))


@dataclass(frozen=True)
class LinkScene:
    """A DiffeRT scene holding ``num_tx`` UEs and one 32-element BS panel."""

    scene: object
    tx_positions: np.ndarray  # (num_tx, 3)
    rx_positions: np.ndarray  # (num_elements, 3)
    bs_center: np.ndarray
    bs_yaw_rad: float

    @property
    def num_tx(self) -> int:
        return int(self.tx_positions.shape[0])

    @property
    def num_elements(self) -> int:
        return int(self.rx_positions.shape[0])


def build_link_scene(
    base_scene,
    tx_positions: np.ndarray,
    bs_center: np.ndarray,
    bs_yaw_rad: float,
    antenna: AntennaSpec,
) -> LinkScene:
    """Attach ``num_tx`` UE transmitters and one BS panel to ``base_scene``."""
    import jax.numpy as jnp  # noqa: PLC0415
    from differt.geometry import Scene  # noqa: PLC0415

    tx_positions = np.asarray(tx_positions, dtype=float).reshape(-1, 3)
    if tx_positions.shape[0] == 0:
        raise ValueError("at least one transmitter is required")

    rx_positions = panel_element_positions(bs_center, bs_yaw_rad, antenna.bs)

    scene = Scene(
        transmitters=jnp.asarray(tx_positions),
        receivers=jnp.asarray(rx_positions),
        mesh=base_scene.mesh,
    )
    return LinkScene(
        scene=scene,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        bs_center=np.asarray(bs_center, dtype=float),
        bs_yaw_rad=float(bs_yaw_rad),
    )


def mesh_stats(scene) -> dict[str, object]:  # noqa: ANN001
    """Small diagnostic summary of the loaded geometry."""
    mesh = scene.mesh
    return {
        "num_triangles": int(mesh.triangles.shape[0]),
        "num_vertices": int(mesh.vertices.shape[0]),
        "material_names": list(mesh.material_names),
    }


def subset_link_scene(link_scene: LinkScene, indices) -> LinkScene:
    """A copy of ``link_scene`` restricted to the given transmitter indices.

    Graph-based tracers (``hybrid``/``exhaustive``) enumerate candidates for every
    transmitter at once, so ten simultaneous UEs multiply both the candidate array and
    the memory by ten. Tracing one transmitter at a time keeps the peak bounded.
    """
    import jax.numpy as jnp  # noqa: PLC0415
    from differt.geometry import Scene  # noqa: PLC0415

    idx = list(indices)
    tx = link_scene.tx_positions[idx]
    scene = Scene(
        transmitters=jnp.asarray(tx),
        receivers=jnp.asarray(link_scene.rx_positions),
        mesh=link_scene.scene.mesh,
    )
    return LinkScene(
        scene=scene,
        tx_positions=tx,
        rx_positions=link_scene.rx_positions,
        bs_center=link_scene.bs_center,
        bs_yaw_rad=link_scene.bs_yaw_rad,
    )
