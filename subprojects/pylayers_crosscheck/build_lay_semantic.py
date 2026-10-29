"""Semantic reconstruction: SceneGen normalized architecture -> PyLayers `.lay`.

The normalized architecture (`3D-Front/scenegen_architecture_normalized/<uuid>/`)
keeps the semantic decomposition that the raw mesh loses:

    WallInner(30) WallOuter(18) WallTop(24) WallBottom(24)
    Door(2) Window(4) Floor(4) Ceiling(4) SlabSide(35) SlabBottom(1) ...

Each wall is a *single box panel*, so its centreline is exact - no sheet pairing,
no raster guessing. `WallTop`/`WallBottom` are the pieces above/below openings,
and `Door`/`Window` panels give the aperture footprint, which is bridged in plan
by a transparent AIR segment (PyLayers needs a closed subdivision, and an AIR
segment keeps the opening transparent like the mesh hole in Sionna).

Run:  python build_lay_semantic.py --scene front3d_0015
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import SCENE_ROOT, SUBROOT, itu_eps_sigma  # noqa: E402

ARCH_ROOT = Path('/data/sunmeiyuan/projects/3D-Front/scenegen_architecture_normalized')

WALL_TYPES = ('WallInner', 'WallOuter', 'WallTop', 'WallBottom')
OPENING_TYPES = ('Door', 'Window', 'Front', 'Back', 'BayWindow')
FLOOR_TYPES = ('Floor', 'SlabBottom')
CEIL_TYPES = ('Ceiling',)


@dataclass
class Panel:
    kind: str
    uid: str
    verts: np.ndarray      # (n,3)
    tris: np.ndarray       # (m,3) indices into verts


def parse_obj_groups(path: Path) -> list[Panel]:
    verts: list[tuple[float, float, float]] = []
    panels: list[Panel] = []
    cur_name = None
    cur_tris: list[tuple[int, int, int]] = []
    cur_vstart = 0

    def flush():
        nonlocal cur_name, cur_tris, cur_vstart
        if cur_name and cur_tris:
            idx = np.asarray(cur_tris, int)
            used = np.unique(idx)
            remap = {int(v): i for i, v in enumerate(used)}
            sub = np.asarray(verts, float)[used]
            tris = np.vectorize(remap.get)(idx)
            kind, _, uid = cur_name.partition('_')
            panels.append(Panel(kind=kind, uid=uid or cur_name, verts=sub, tris=tris))
        cur_tris = []

    with path.open('r', errors='ignore') as fh:
        for line in fh:
            if line.startswith('v '):
                p = line.split()
                verts.append((float(p[1]), float(p[2]), float(p[3])))
            elif line.startswith('g '):
                flush()
                cur_name = line[2:].strip()
            elif line.startswith('f '):
                idx = []
                for tok in line.split()[1:]:
                    i = int(tok.split('/')[0])
                    idx.append(i - 1 if i > 0 else len(verts) + i)
                for k in range(1, len(idx) - 1):
                    cur_tris.append((idx[0], idx[k], idx[k + 1]))
    flush()
    return panels


def panel_axis(points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Principal direction, centre, length and thickness of a plan footprint."""
    c = points_xy.mean(axis=0)
    q = points_xy - c
    cov = q.T @ q
    w, vec = np.linalg.eigh(cov)
    d = vec[:, np.argmax(w)]
    if abs(d[0]) >= abs(d[1]):
        d = np.array([np.sign(d[0]) or 1.0, 0.0])
    else:
        d = np.array([0.0, np.sign(d[1]) or 1.0])
    t = np.array([-d[1], d[0]])
    along = q @ d
    across = q @ t
    return d, c, float(along.max() - along.min()), float(across.max() - across.min())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='front3d_0015')
    ap.add_argument('--scene-id', default=None,
                    help='3D-FRONT uuid (default: read from the scene floorplan meta)')
    ap.add_argument('--wall-thickness', type=float, default=0.1)
    ap.add_argument('--out', default=None)
    ap.add_argument('--min-panel-area', type=float, default=0.25)
    ap.add_argument('--clean', action='store_true',
                    help='apply dangling/open-wall cleanup (may drop real walls)')
    args = ap.parse_args()

    import build_lay as bl

    scene_uuid = args.scene_id
    if scene_uuid is None:
        meta = json.loads((SCENE_ROOT / args.scene / 'floorplan' / 'class_mask_meta.json').read_text())
        scene_uuid = meta['scene_id']
    arch_dir = ARCH_ROOT / scene_uuid
    obj_path = arch_dir / f'{scene_uuid}.obj'
    meta_path = arch_dir / f'{scene_uuid}.json'
    print(f'architecture: {obj_path}')
    arch = json.loads(meta_path.read_text())
    counts = arch['scene']['mesh_type_counts']
    print(f"type counts: {counts}")

    # normalized frame -> scene frame used by scene.xml and the labels.
    # The normalization flips y (front3d_y_up_to_scenegen_z_up), so a plain
    # translation is wrong: verified by floorplan coverage, mirroring y lifts
    # wall-pixel coverage at 15 cm from 47 % to 95 %.
    bbox = arch['geometry']['bbox']
    x_min = float(bbox['min'][0])
    y_max = float(bbox['max'][1])
    print(f'frame map (normalized -> scene): x-{x_min:.4f}, y -> {y_max:.4f} - y')

    panels = parse_obj_groups(obj_path)
    for pnl in panels:
        v = pnl.verts.copy()
        v[:, 0] = v[:, 0] - x_min
        v[:, 1] = y_max - v[:, 1]
        pnl.verts = v
    by_type = defaultdict(list)
    for p in panels:
        by_type[p.kind].append(p)
    print('parsed panels:', {k: len(v) for k, v in sorted(by_type.items())})

    # ---- floor / ceiling levels
    def z_range(kinds):
        zs = [p.verts[:, 2] for k in kinds for p in by_type.get(k, [])]
        return (float(min(z.min() for z in zs)), float(max(z.max() for z in zs))) if zs else None

    floor_z = min(z.min() for p in by_type.get('Floor', []) for z in [p.verts[:, 2]]) \
        if by_type.get('Floor') else 0.0
    ceil_z = max(p.verts[:, 2].max() for p in by_type.get('Ceiling', [])) \
        if by_type.get('Ceiling') else 2.8
    print(f'floor_z={floor_z:.3f} ceil_z={ceil_z:.3f}')

    # ---- wall panels -> centreline segments
    segs: list[bl.Segment] = []
    wall_len = 0.0
    skipped = 0
    skip_why: dict[str, int] = {}
    for kind in WALL_TYPES:
        for p in by_type.get(kind, []):
            d, c, length, thick = panel_axis(p.verts[:, :2])
            z0, z1 = float(p.verts[:, 2].min()), float(p.verts[:, 2].max())
            if length < 0.15 or (z1 - z0) < 0.05:
                skip_why['too_small'] = skip_why.get('too_small', 0) + 1
                skipped += 1
                continue
            if length * (z1 - z0) < args.min_panel_area:
                skip_why['low_area'] = skip_why.get('low_area', 0) + 1
                skipped += 1
                continue
            proj = (p.verts[:, :2] - c) @ d
            a = c + d * float(proj.min())
            b = c + d * float(proj.max())
            segs.append(bl.Segment(p_a=(bl.snap(float(a[0])), bl.snap(float(a[1]))),
                                   p_b=(bl.snap(float(b[0])), bl.snap(float(b[1]))),
                                   z=(bl.snap(z0), bl.snap(z1))))
            wall_len += length
    print(f'wall panels -> {len(segs)} segments (skipped {skipped} {skip_why}), '
          f'total length {wall_len:.1f} m')

    # ---- openings: AIR bridges across their footprint
    n_air = 0
    for kind in OPENING_TYPES:
        for p in by_type.get(kind, []):
            d, c, length, thick = panel_axis(p.verts[:, :2])
            if length < 0.3:
                continue
            proj = (p.verts[:, :2] - c) @ d
            a = c + d * float(proj.min())
            b = c + d * float(proj.max())
            segs.append(bl.Segment(p_a=(bl.snap(float(a[0])), bl.snap(float(a[1]))),
                                   p_b=(bl.snap(float(b[0])), bl.snap(float(b[1]))),
                                   z=(bl.snap(float(floor_z)), bl.snap(float(ceil_z))),
                                   material='AIR'))
            n_air += 1
    print(f'openings -> {n_air} AIR segments')

    # ---- graph hygiene (same helpers as the mesh converter)
    registry: dict[tuple[float, float], int] = {}

    def nid(p):
        key = (bl.snap(float(p[0])), bl.snap(float(p[1])))
        if key not in registry:
            registry[key] = -(len(registry) + 1)
        return registry[key]

    rows = []
    for s in segs:
        a, b = nid(s.p_a), nid(s.p_b)
        if a != b:
            rows.append((a, b, s.z, s.material))
    nodes = {n: np.array(p, float) for p, n in registry.items()}
    print(f'raw: {len(rows)} segments, {len(nodes)} nodes')

    nodes, rows = bl.planarize(nodes, rows, tol=0.01)
    nodes, rows, n_corners = bl.close_corners(nodes, rows, max_ext=0.30)
    nodes, rows = bl.planarize(nodes, rows, tol=0.01)
    if args.clean:
        rows, n_pruned = bl.prune_dangling(nodes, rows)
        rows, plan = bl.clean_planar(nodes, rows, min_face_area=0.5)
    else:
        n_pruned = 0
        plan = dict(rooms=-1, removed=0, iterations=0)
    nodes, rows = bl.merge_coincident_nodes(nodes, rows)
    used = {n for r in rows for n in r[:2]}
    nodes = {n: p for n, p in nodes.items() if n in used}
    print(f'clean: {len(rows)} segments, {len(nodes)} nodes, corners {n_corners}, '
          f'pruned {n_pruned}, dropped {plan["removed"]}, faces {plan["rooms"]}')

    eps, sigma = itu_eps_sigma('concrete', 3.438)
    out_path = Path(args.out) if args.out else (
        SUBROOT / 'outputs' / 'layouts' / f'{args.scene}.lay')
    result = dict(ceil_z=ceil_z, floor_z=float(floor_z), segments=[
        bl.Segment(p_a=tuple(nodes[r[0]]), p_b=tuple(nodes[r[1]]), z=r[2], material=r[3])
        for r in rows])
    stats = bl.write_lay(result, out_path, args.wall_thickness, eps, sigma,
                         clean=args.clean)
    print(f'layout: {out_path}  points={stats["points"]} segments={stats["segments"]}')

    report = dict(scene=args.scene, scene_uuid=scene_uuid,
                  source='scenegen_architecture_normalized (semantic panels)',
                  mesh_type_counts=counts, floor_z=float(floor_z), ceil_z=ceil_z,
                  wall_panels=len(segs) - n_air, air_segments=n_air,
                  total_wall_length_m=round(wall_len, 2),
                  pruned=n_pruned, dropped=plan['removed'], faces=plan['rooms'], **stats)
    out_path.with_suffix('.report.json').write_text(json.dumps(report, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
