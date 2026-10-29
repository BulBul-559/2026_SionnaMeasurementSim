"""Shared helpers for the PyLayers cross-check subproject.

Self-contained: nothing here imports or modifies sionna_measurement_sim.
Only reads Front3D scene XML/OBJ files and the prepared labels.
"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

WORKSPACE = Path('/home/sunmeiyuan/projects/SionnaMeasurementSim')
SUBROOT = WORKSPACE / 'subprojects' / 'pylayers_crosscheck'
SCENE_ROOT = Path('/data/sunmeiyuan/projects/SceneGen/front3d_full_all_6813_20260617_215824')

# Sionna ITU-R P.2040-3 coefficients, copied from
# sionna/rt/radio_materials/itu.py so both engines share one material corpus.
ITU_P2040 = {
    'concrete': (5.24, 0.0, 0.0462, 0.7822),
    'wood': (1.99, 0.0, 0.0047, 1.0718),
    'glass': (6.31, 0.0, 0.0036, 1.3394),
    'metal': (1.0, 0.0, 1e7, 0.0),
    'brick': (3.91, 0.0, 0.0238, 0.16),
    'plasterboard': (2.73, 0.0, 0.0085, 0.9395),
    'ceiling_board': (1.48, 0.0, 0.0011, 1.0750),
    'floorboard': (3.66, 0.0, 0.0044, 1.3515),
}

CENTER_FREQ_HZ = 3.438e9
BANDWIDTH_HZ = 23.04e6
NUM_SUBCARRIERS = 768
SUBCARRIER_SPACING_HZ = 30e3


def itu_eps_sigma(name: str, f_ghz: float) -> tuple[float, float]:
    """Relative permittivity and conductivity (S/m) at f_ghz for an ITU material."""
    a, b, c, d = ITU_P2040[name]
    eps = a * f_ghz ** b
    sigma = c * f_ghz ** d
    return float(eps), float(sigma)


def frequency_grid_ghz() -> np.ndarray:
    """768-point grid used by the DICHASUS-like contract (30 kHz spacing)."""
    start = CENTER_FREQ_HZ - SUBCARRIER_SPACING_HZ * (NUM_SUBCARRIERS - 1) / 2.0
    return (start + SUBCARRIER_SPACING_HZ * np.arange(NUM_SUBCARRIERS)) / 1e9


@dataclass
class SceneShape:
    shape_id: str
    obj_path: Path
    material: str
    matrix: np.ndarray | None


@dataclass
class SceneMesh:
    scene_id: str
    scene_file: Path
    verts: np.ndarray
    faces: np.ndarray
    face_material: list[str]
    shapes: list[SceneShape] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    verts: list[tuple[float, float, float]] = []
    tris: list[tuple[int, int, int]] = []
    with path.open('r', errors='ignore') as fh:
        for line in fh:
            if line.startswith('v '):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith('f '):
                idx = []
                for tok in line.split()[1:]:
                    i = int(tok.split('/')[0])
                    idx.append(i - 1 if i > 0 else len(verts) + i)
                for k in range(1, len(idx) - 1):
                    tris.append((idx[0], idx[k], idx[k + 1]))
    if not verts or not tris:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=int)
    return np.asarray(verts, float), np.asarray(tris, int)


def scene_shapes(scene_file: Path, include_furniture: bool = False) -> tuple[list[SceneShape], list[str]]:
    root = ET.parse(scene_file).getroot()
    base = scene_file.parent
    shapes: list[SceneShape] = []
    skipped: list[str] = []
    for sh in root.findall('shape'):
        sid = sh.get('id', '')
        is_arch = 'empty_scene' in sid or 'architecture' in sid
        is_furn = 'furniture' in sid
        if not (is_arch or (include_furniture and is_furn)):
            skipped.append(sid)
            continue
        strings = {c.get('name'): c.get('value') for c in sh.findall('string')}
        fn = strings.get('filename')
        if not fn:
            continue
        ref = sh.find('ref')
        mat_id = ref.get('id') if ref is not None else ''
        material = mat_id.replace('itu-', '').strip() or 'concrete'
        matrix = None
        tf = sh.find('transform')
        if tf is not None:
            mx = tf.find('matrix')
            if mx is not None:
                matrix = np.asarray([float(x) for x in mx.get('value').split()], float).reshape(4, 4)
        shapes.append(SceneShape(sid, base / fn, material, matrix))
    return shapes, skipped


def load_scene(scene_file: Path, include_furniture: bool = False) -> SceneMesh:
    shapes, skipped = scene_shapes(scene_file, include_furniture=include_furniture)
    verts_all: list[np.ndarray] = []
    faces_all: list[np.ndarray] = []
    mats: list[str] = []
    used: list[SceneShape] = []
    offset = 0
    for shape in shapes:
        if not shape.obj_path.exists():
            skipped.append(f'MISSING:{shape.obj_path.name}')
            continue
        v, f = load_obj(shape.obj_path)
        if len(f) == 0:
            continue
        if shape.matrix is not None:
            v = v @ shape.matrix[:3, :3].T + shape.matrix[:3, 3]
        verts_all.append(v)
        faces_all.append(f + offset)
        mats.extend([shape.material] * len(f))
        used.append(shape)
        offset += len(v)
    if not faces_all:
        raise RuntimeError(f'no geometry found in {scene_file}')
    return SceneMesh(
        scene_id=scene_file.parent.name,
        scene_file=scene_file,
        verts=np.vstack(verts_all),
        faces=np.vstack(faces_all),
        face_material=mats,
        shapes=used,
        skipped=skipped,
    )


def triangle_geometry(mesh: SceneMesh):
    p0 = mesh.verts[mesh.faces[:, 0]]
    p1 = mesh.verts[mesh.faces[:, 1]]
    p2 = mesh.verts[mesh.faces[:, 2]]
    nrm = np.cross(p1 - p0, p2 - p0)
    nlen = np.linalg.norm(nrm, axis=1)
    ok = nlen > 1e-12
    return p0[ok], p1[ok], p2[ok], nrm[ok], nlen[ok]


def label_path(scene_key: str) -> Path:
    return SUBROOT / 'inputs' / 'labels' / scene_key / 'label_panel_0p5_bs_center_yaw.json'


def sionna_scene_h5(scene_key: str) -> Path:
    return Path('/data/sunmeiyuan/projects/sionna/outputs/front3d_339_dichasus_like_cfr_v1'
                f'/dataset/scenes/{scene_key}.h5')


def resolved_label_path(scene_key: str) -> Path:
    """Prefer the copy inside the subproject, fall back to the prepared 339 label."""
    local = label_path(scene_key)
    if local.exists():
        return local
    return Path('/data/sunmeiyuan/projects/sionna/outputs/front3d_339_dichasus_like_cfr_v1'
                f'/prepared/labels/{scene_key}/label_panel_0p5_bs_center_yaw.json')


def load_label(scene_key: str) -> dict:
    return json.loads(resolved_label_path(scene_key).read_text())


def pylayers_project_dir() -> Path:
    """PyLayers project directory (the one written into ~/.pylayers)."""
    env = os.environ.get('PYLAYERS_PROJECT')
    if env:
        return Path(env)
    probe = WORKSPACE / 'tmp' / 'pylayers_probe' / 'pylayers_project'
    return probe


def install_layout(lay_path: Path) -> str:
    """Copy a generated .lay into <project>/struc/lay and return its short name.

    PyLayers resolves layout files with `pyu.getlong(name, pstruc['DIRLAY'])`,
    which always joins `basename/struc/lay/<name>`: an absolute path outside the
    project directory silently yields a void outdoor layout. So the layout has to
    be installed under the project directory and referenced by short name.
    """
    import shutil

    dest_dir = pylayers_project_dir() / 'struc' / 'lay'
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / lay_path.name
    shutil.copy2(lay_path, dest)
    return lay_path.name

