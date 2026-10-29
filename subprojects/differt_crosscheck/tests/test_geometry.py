"""Array geometry tests.

The expected numbers are taken directly from Sionna's ``PlanarArray.__init__`` formula
(``y = d_h*j - (C-1)*d_h/2``, ``z = -d_v*i + (R-1)*d_v/2``, ``x = 0``, flat index
``k = i*C + j``). If DiffeRT's kernel and Sionna disagree on element order or on which
axis is vertical, every downstream comparison would be silently wrong, so this is pinned
against the formula rather than against our own implementation.
"""

from __future__ import annotations

import numpy as np

from differt_crosscheck.config import PlanarArraySpec
from differt_crosscheck.scene import panel_element_positions, rotation_z

SPEC = PlanarArraySpec()
D = SPEC.spacing_m


def _reference_positions(center, yaw, rows, cols, d):
    ii, jj = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    y = d * jj - (cols - 1) * d / 2.0
    z = -d * ii + (rows - 1) * d / 2.0
    local = np.stack([np.zeros_like(y), y, z], axis=-1).reshape(-1, 3)
    return (rotation_z(yaw) @ local.T).T + np.asarray(center)[None, :]


def test_element_count() -> None:
    pos = panel_element_positions(np.zeros(3), 0.0, SPEC)
    assert pos.shape == (SPEC.num_elements, 3), pos.shape


def test_matches_sionna_formula() -> None:
    center = np.array([2.0, 9.224, 2.4])
    yaw = -0.92
    pos = panel_element_positions(center, yaw, SPEC)
    ref = _reference_positions(center, yaw, SPEC.num_rows, SPEC.num_cols, D)
    assert np.allclose(pos, ref, atol=1e-12)


def test_panel_is_vertical_and_planar() -> None:
    pos = panel_element_positions(np.zeros(3), 0.0, SPEC)
    assert np.allclose(pos[:, 0], 0.0), "panel must lie in the y-z plane"
    zs = np.unique(np.round(pos[:, 2], 9))
    ys = np.unique(np.round(pos[:, 1], 9))
    assert zs.size == SPEC.num_rows
    assert ys.size == SPEC.num_cols
    assert np.allclose(np.diff(ys), D)
    # np.unique sorts ascending; the Sionna row order itself runs top-down
    assert np.allclose(np.diff(zs), D)
    assert pos[0, 2] > pos[SPEC.num_cols, 2], "row 0 must be the top row"


def test_yaw_rotates_without_changing_spacing() -> None:
    yaw = np.pi / 3
    pos = panel_element_positions(np.zeros(3), yaw, SPEC)
    first_row = pos[: SPEC.num_cols]
    steps = np.linalg.norm(np.diff(first_row, axis=0), axis=1)
    assert np.allclose(steps, D, atol=1e-12)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
