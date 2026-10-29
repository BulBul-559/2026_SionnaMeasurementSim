"""Front3D structure mesh -> PyLayers `.lay` converter.

Design notes
------------
* Only architecture shapes are used (furniture excluded by contract).
* Vertical triangles become wall lines, canonicalised by (azimuth mod 180,
  signed offset), so the two room-side sheets of one wall collapse into a single
  line instead of two coincident reflectors 0-30 mm apart.
* Each line is converted to an exact rectilinear occupancy grid in (u, z); cell
  boundaries come from real vertex coordinates (no raster resampling), so
  segment endpoints land on the original geometry.
* Only solid wall pieces are emitted. PyLayers' `.lay` reader complements each
  wall's missing z-intervals with AIR segments between zfloor and zceil
  (layout.py: `compint` + AIR iso segments), so door/window apertures need no
  explicit AIR geometry.
* Coordinates are snapped to millimetres, matching PyLayers' own precision
  limit in the layout reader.
* Collinear segments are split at every node lying on them, so the planar graph
  stays valid for `Layout.build()`.

Run:  python build_lay.py --scene front3d_0015
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import SCENE_ROOT, SUBROOT, itu_eps_sigma, load_scene, triangle_geometry  # noqa: E402

MM = 1000.0
SNAP_MM = 5.0          # node snap grid: 5 mm (walls are >= 65 mm apart, so this is safe)


def snap(v: float) -> float:
    return round(v * MM / SNAP_MM) * SNAP_MM / MM


@dataclass
class Segment:
    p_a: tuple[float, float]
    p_b: tuple[float, float]
    z: tuple[float, float]
    material: str = 'WALL'


@dataclass
class WallLine:
    az_deg: float
    offset: float
    triangles: list[int]
    area: float


def cluster_1d(values: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Greedy 1-D clustering; returns cluster centres and per-value assignment."""
    order = np.argsort(values, kind='stable')
    centres: list[float] = []
    assign = np.zeros(len(values), dtype=int)
    for idx in order:
        v = float(values[idx])
        if centres and abs(v - centres[-1]) <= tol:
            assign[idx] = len(centres) - 1
            centres[-1] = 0.5 * (centres[-1] + v)
        else:
            centres.append(v)
            assign[idx] = len(centres) - 1
    return np.asarray(centres), assign


def build_wall_lines(p0, p1, p2, nh, area, az_tol_deg=0.5, offset_tol=0.015) -> list[WallLine]:
    az = np.degrees(np.arctan2(nh[:, 1], nh[:, 0])) % 360.0
    az_folded = np.where(az >= 180.0, az - 180.0, az)
    sign = np.where(az >= 180.0, -1.0, 1.0)
    offset_signed = np.sum(nh[:, :2] * p0[:, :2], axis=1) * sign

    az_centres, az_assign = cluster_1d(az_folded, tol=az_tol_deg)
    lines: list[WallLine] = []
    for ia, az_c in enumerate(az_centres):
        sel = np.where(az_assign == ia)[0]
        if sel.size == 0:
            continue
        off_centres, off_assign = cluster_1d(offset_signed[sel], tol=offset_tol)
        for io, off_c in enumerate(off_centres):
            ids = sel[off_assign == io]
            if ids.size == 0:
                continue
            lines.append(WallLine(float(az_c), float(off_c),
                                  [int(i) for i in ids], float(area[ids].sum())))
    return lines


def line_basis(az_deg: float) -> tuple[np.ndarray, np.ndarray]:
    a = np.radians(az_deg)
    return np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])


def occupancy_rects(tri_uz: np.ndarray, u_tol=0.0015, z_tol=0.0015,
                    min_len=0.05, min_h=0.05):
    """Union of (u,z) triangles as axis-aligned rectangles with exact boundaries."""
    if len(tri_uz) == 0:
        return []
    u_centres, _ = cluster_1d(tri_uz[:, :, 0].ravel(), tol=u_tol)
    z_centres, _ = cluster_1d(tri_uz[:, :, 1].ravel(), tol=z_tol)
    if len(u_centres) < 2 or len(z_centres) < 2:
        return []

    nu, nz = len(u_centres) - 1, len(z_centres) - 1
    cu = 0.5 * (u_centres[:-1] + u_centres[1:])
    cz = 0.5 * (z_centres[:-1] + z_centres[1:])
    mask = np.zeros((nu, nz), dtype=bool)

    for t in tri_uz:
        (ax, az_), (bx, bz), (cx, cz_) = t
        den = (bz - cz_) * (ax - cx) + (cx - bx) * (az_ - cz_)
        if abs(den) < 1e-14:
            continue
        i0 = int(np.searchsorted(cu, min(ax, bx, cx)))
        i1 = int(np.searchsorted(cu, max(ax, bx, cx), side='right'))
        j0 = int(np.searchsorted(cz, min(az_, bz, cz_)))
        j1 = int(np.searchsorted(cz, max(az_, bz, cz_), side='right'))
        if i1 <= i0 or j1 <= j0:
            continue
        X = cu[i0:i1, None]
        Z = cz[None, j0:j1]
        w0 = ((bz - cz_) * (X - cx) + (cx - bx) * (Z - cz_)) / den
        w1 = ((cz_ - az_) * (X - cx) + (ax - cx) * (Z - cz_)) / den
        w2 = 1.0 - w0 - w1
        mask[i0:i1, j0:j1] |= (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)

    rects = []
    row_keys = [tuple(np.flatnonzero(mask[i, :])) for i in range(nu)]
    i = 0
    while i < nu:
        j = i
        while j + 1 < nu and row_keys[j + 1] == row_keys[i]:
            j += 1
        if row_keys[i]:
            covered = row_keys[i]
            k = 0
            while k < len(covered):
                k2 = k
                while k2 + 1 < len(covered) and covered[k2 + 1] == covered[k2] + 1:
                    k2 += 1
                z0, z1 = z_centres[covered[k]], z_centres[covered[k2] + 1]
                u0, u1 = u_centres[i], u_centres[j + 1]
                if (u1 - u0) >= min_len and (z1 - z0) >= min_h:
                    rects.append(((float(u0), float(u1)), (float(z0), float(z1))))
                k = k2 + 1
        i = j + 1
    return rects


def detect_levels(p0, p1, p2, nh, area, tol=0.05, min_area=8.0):
    hmask = np.sqrt(nh[:, 0] ** 2 + nh[:, 1] ** 2) < 0.05
    hi = np.where(hmask)[0]
    zc = (p0[hi, 2] + p1[hi, 2] + p2[hi, 2]) / 3.0
    centres, assign = cluster_1d(zc, tol=tol)
    totals = np.zeros(len(centres))
    for k, a in zip(assign, area[hi]):
        totals[k] += a
    keep = [(float(c), float(a)) for c, a in zip(centres, totals) if a >= min_area]
    keep.sort()
    return keep, hmask


def merge_close_lines(lines: list[WallLine], az_tol=0.6, min_sep=0.05,
                      min_area=0.3) -> list[WallLine]:
    """Drop sliver doubles: one wall must not appear as two lines a few cm apart.

    Front3D structure meshes carry mouldings, trims and reveal panels a couple of
    centimetres in front of the wall. Feeding both to PyLayers produces
    near-collinear overlapping constraints, which makes its constrained
    triangulation fail ("topological inconsistency after splitting a segment").
    Keep the larger line of each near-parallel pair.
    """
    kept: list[WallLine] = []
    for line in sorted(lines, key=lambda l: -l.area):
        if line.area < min_area:
            continue
        clash = False
        for k in kept:
            daz = abs((k.az_deg - line.az_deg + 90.0) % 180.0 - 90.0)
            if daz <= az_tol and abs(k.offset - line.offset) < min_sep:
                clash = True
                break
        if not clash:
            kept.append(line)
    return kept


def extract_segments(scene_key: str, floor_z: float | None, ceil_z: float | None,
                     include_furniture: bool = False, offset_tol: float = 0.015,
                     node_tol: float = 0.005, merge_sep: float = 0.05) -> dict:
    scene_file = SCENE_ROOT / scene_key / 'scene.xml'
    mesh = load_scene(scene_file, include_furniture=include_furniture)
    p0, p1, p2, nrm, nlen = triangle_geometry(mesh)
    area = 0.5 * nlen
    nh = nrm / nlen[:, None]

    vmask = np.abs(nh[:, 2]) < 0.05
    levels, hmask = detect_levels(p0, p1, p2, nh, area)

    if floor_z is None:
        floor_z = min(z for z, _ in levels) if levels else float(mesh.verts[:, 2].min())
    if ceil_z is None:
        ceil_z = max(z for z, _ in levels) if levels else float(mesh.verts[:, 2].max())

    vi = np.where(vmask)[0]
    lines = build_wall_lines(p0[vi], p1[vi], p2[vi], nh[vi], area[vi], offset_tol=offset_tol)
    n_lines_raw = len(lines)
    lines = merge_close_lines(lines, min_sep=merge_sep)

    segments: list[Segment] = []
    line_reports = []
    for line_id, line in enumerate(lines):
        n, t = line_basis(line.az_deg)
        ltri = vi[np.asarray(line.triangles)]
        pts = np.stack([p0[ltri], p1[ltri], p2[ltri]], axis=1)   # (m,3,3)
        u = pts[:, :, :2] @ t
        z = pts[:, :, 2]
        tri_uz = np.stack([np.stack([u[:, k], z[:, k]], axis=1) for k in range(3)], axis=1)
        rects = occupancy_rects(tri_uz)
        emitted = 0
        filled = 0
        for (u0, u1), (z0, z1) in rects:
            z0c, z1c = max(z0, floor_z), min(z1, ceil_z)
            if z1c - z0c < 0.05:
                continue
            pa = line.offset * n + u0 * t
            pb = line.offset * n + u1 * t
            segments.append(Segment(
                p_a=(snap(float(pa[0])), snap(float(pa[1]))),
                p_b=(snap(float(pb[0])), snap(float(pb[1]))),
                z=(snap(float(z0c)), snap(float(z1c))),
            ))
            emitted += 1

        # Fill interior plan gaps of this line with transparent AIR segments.
        # Front3D leaves real holes at doors/passages; a gap in plan would leave
        # the wall network open, and PyLayers' graph build needs a closed
        # subdivision. AIR segments keep the topology closed without adding
        # reflection or transmission loss, so the physics matches a hole.
        if emitted >= 2:
            spans = sorted(((min(u0, u1), max(u0, u1)) for (u0, u1), _ in rects
                            if (max(u0, u1) - min(u0, u1)) >= 0.05))
            merged: list[list[float]] = []
            for lo, hi in spans:
                if merged and lo <= merged[-1][1] + 0.05:
                    merged[-1][1] = max(merged[-1][1], hi)
                else:
                    merged.append([lo, hi])
            for k in range(len(merged) - 1):
                g0, g1 = merged[k][1], merged[k + 1][0]
                if g1 - g0 < 0.05:
                    continue
                pa = line.offset * n + g0 * t
                pb = line.offset * n + g1 * t
                segments.append(Segment(
                    p_a=(snap(float(pa[0])), snap(float(pa[1]))),
                    p_b=(snap(float(pb[0])), snap(float(pb[1]))),
                    z=(snap(float(floor_z)), snap(float(ceil_z))),
                    material='AIR',
                ))
                filled += 1
        line_reports.append(dict(line_id=line_id, az_deg=round(line.az_deg, 2),
                                 offset=round(line.offset, 3), area=round(line.area, 2),
                                 triangles=int(len(ltri)), segments=emitted,
                                 air_filled=filled))

    return dict(
        scene_key=scene_key,
        scene_file=str(scene_file),
        total_area=float(area.sum()),
        area_vertical=float(area[vmask].sum()),
        area_horizontal=float(area[hmask].sum()),
        area_other=float(area[~vmask & ~hmask].sum()),
        n_triangles=int(len(area)),
        levels=[dict(z=round(z, 3), area=round(a, 2)) for z, a in levels],
        floor_z=float(floor_z),
        ceil_z=float(ceil_z),
        n_wall_lines=len(lines),
        n_wall_lines_raw=n_lines_raw,
        lines=line_reports,
        segments=segments,
    )


def dedupe_and_split(segments: list[Segment], node_tol=0.002) -> tuple[dict, list]:
    """Build the node registry, drop duplicates and split collinear segments.

    PyLayers' `add_segment` only accepts *negative* point indices (`n1 < 0`, as in
    defstr.lay); positive indices are rejected with "error not a node". Point ids
    are therefore negative and segment ids positive, keeping the two namespaces
    disjoint the way the PyLayers layout reader expects.
    """
    registry: dict[tuple[float, float], int] = {}

    def nid(p):
        key = (snap(p[0]), snap(p[1]))
        if key not in registry:
            registry[key] = -(len(registry) + 1)
        return registry[key]

    raw = []
    seen = set()
    for seg in segments:
        a = nid(seg.p_a)
        b = nid(seg.p_b)
        if a == b:
            continue
        key = (min(a, b), max(a, b), seg.z)
        if key in seen:
            continue
        seen.add(key)
        raw.append((a, b, seg.z, seg.material))

    inv = {v: np.array(k) for k, v in registry.items()}
    nids = np.array(sorted(inv))
    coords = np.array([inv[i] for i in nids])

    out = []
    for a, b, z, mat in raw:
        pa, pb = inv[a], inv[b]
        d = pb - pa
        ln = float(np.linalg.norm(d))
        if ln < 1e-6:
            continue
        dd = d / ln
        rel = coords - pa
        proj = rel @ dd
        perp = np.abs(rel[:, 0] * dd[1] - rel[:, 1] * dd[0])
        on = (perp <= node_tol) & (proj > node_tol) & (proj < ln - node_tol)
        if not on.any():
            out.append((a, b, z, mat))
            continue
        chain = [tuple(coords[k]) for k in np.argsort(proj[on]) if on[k]]
        seq_nodes = [a] + [nid(p) for p in chain] + [b]
        for k in range(len(seq_nodes) - 1):
            if seq_nodes[k] != seq_nodes[k + 1]:
                out.append((seq_nodes[k], seq_nodes[k + 1], z, mat))

    # dedupe again (splitting may create repeats)
    final, seen2 = [], set()
    for a, b, z, mat in out:
        key = (min(a, b), max(a, b), z)
        if key in seen2:
            continue
        seen2.add(key)
        final.append((a, b, z, mat))
    return registry, final


def planarize(nodes: dict[int, np.ndarray], segs: list[tuple], tol=0.005,
              max_iter=200) -> tuple[dict[int, np.ndarray], list[tuple]]:
    """Insert a node at every proper crossing so the segment set is a planar graph.

    PyLayers' `Layout.buildGt` feeds the centrelines to a constrained Delaunay
    triangulation, which rejects any crossing that is not a vertex. Wall meshes
    produce plenty of them (partition walls butting into a run, reveal panels),
    so crossing points are materialised as nodes here.
    """
    segs = list(segs)
    nxt = min(nodes) - 1 if nodes else -1

    def get_node(p: np.ndarray) -> int:
        """Reuse the nearest existing node within `tol`, else create one."""
        nonlocal nxt
        key = np.array([snap(float(p[0])), snap(float(p[1]))], float)
        best, best_d = None, None
        for nid, q in nodes.items():
            d = float(np.linalg.norm(q - key))
            if d <= tol and (best_d is None or d < best_d):
                best, best_d = nid, d
        if best is not None:
            return best
        nid = nxt
        nxt -= 1
        nodes[nid] = key
        return nid

    for _ in range(max_iter):
        hit = None
        for i in range(len(segs)):
            a1, b1 = segs[i][0], segs[i][1]
            p, r = nodes[a1], nodes[b1] - nodes[a1]
            rn = float(np.linalg.norm(r))
            if rn < 1e-9:
                continue
            for j in range(i + 1, len(segs)):
                a2, b2 = segs[j][0], segs[j][1]
                if {a1, b1} & {a2, b2}:
                    continue
                q, s = nodes[a2], nodes[b2] - nodes[a2]
                sn = float(np.linalg.norm(s))
                if sn < 1e-9:
                    continue
                cross = r[0] * s[1] - r[1] * s[0]
                if abs(cross) <= 1e-9 * rn * sn:
                    continue
                qp = q - p
                t = (qp[0] * s[1] - qp[1] * s[0]) / cross
                u = (qp[0] * r[1] - qp[1] * r[0]) / cross
                et, eu = 0.001 / rn, 0.001 / sn
                if et < t < 1 - et and eu < u < 1 - eu:
                    hit = (i, j, p + t * r)
                    break
            if hit:
                break
        if hit is None:
            break
        i, j, x = hit
        nid = get_node(x)
        for k in (j, i):
            a, b, z, mat = segs[k]
            if nid in (a, b):
                continue
            segs[k:k + 1] = [(a, nid, z, mat), (nid, b, z, mat)]
    return nodes, segs


def prune_dangling(nodes: dict[int, np.ndarray], segs: list[tuple],
                   max_iter=100) -> tuple[list[tuple], int]:
    """Drop dangling wall chains so the segment set is a closed subdivision.

    PyLayers' `buildGt` merges convex faces across transparent segments and
    assumes each such segment separates two *distinct* faces. A wall that ends
    in mid-air (degree-1 point) makes a face adjacent to itself and the merge
    fails with `KeyError` in `dict(self.Gt[n1]).pop(n0)`. Front3D structure
    meshes contain such stubs (facade pieces, truncated partitions), so they are
    removed rather than left to break the graph build.
    """
    from collections import defaultdict

    removed = 0
    for _ in range(max_iter):
        deg: dict[int, int] = defaultdict(int)
        for a, b, _z, _m in segs:
            deg[a] += 1
            deg[b] += 1
        leaves = {n for n, d in deg.items() if d <= 1}
        if not leaves:
            break
        keep = [s for s in segs if s[0] not in leaves and s[1] not in leaves]
        removed += len(segs) - len(keep)
        segs = keep
    used = {n for s in segs for n in s[:2]}
    for n in list(nodes):
        if n not in used:
            nodes.pop(n)
    return segs, removed


def clean_planar(nodes: dict[int, np.ndarray], segs: list[tuple], max_iter=50,
                 min_face_area=0.5) -> tuple[list[tuple], dict]:
    """Remove walls that do not bound any room, and count the rooms.

    PyLayers requires every wall segment to take part in *two* cycles
    (`buildGt` asserts "Some segments are connected to LESS than 2 cycles").
    Leaf pruning alone does not guarantee that: a pair of walls can form a
    cul-de-sac whose segments all have degree >= 2 yet border the outer face
    twice.

    This walks the planar half-edge structure: at every node the outgoing
    half-edges are sorted by angle and faces are traced with the standard
    "next = clockwise neighbour of the reverse" rule. Segments that border no
    bounded face are dropped, repeatedly, until the subdivision is clean.
    """
    nodes = dict(nodes)
    segs = list(segs)
    info = dict(iterations=0, removed=0, rooms=0)

    def half_edges(seg_list):
        he = {}
        for i, (a, b, _z, _m) in enumerate(seg_list):
            he[(a, b)] = i
            he[(b, a)] = i
        return he

    for _ in range(max_iter):
        he = half_edges(segs)
        out: dict[int, list[int]] = {}
        for (u, v) in he:
            out.setdefault(u, []).append(v)
        for u, vs in out.items():
            vs.sort(key=lambda v: np.arctan2(nodes[v][1] - nodes[u][1],
                                             nodes[v][0] - nodes[u][0]))

        pos_in = {(u, v): out[u].index(v) for (u, v) in he}
        seen = set()
        faces = []
        for start in he:
            if start in seen:
                continue
            face = []
            cur = start
            guard = 0
            while cur not in seen and guard < 4 * len(segs) + 10:
                seen.add(cur)
                face.append(cur)
                r = (cur[1], cur[0])
                lst = out[cur[1]]
                nxt = lst[(pos_in[r] - 1) % len(lst)]
                cur = (cur[1], nxt)
                guard += 1
            if len(face) >= 3:
                xy = np.array([nodes[u] for (u, _v) in face], float)
                area = 0.5 * float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1)
                                           - np.roll(xy[:, 0], -1) * xy[:, 1]))
                faces.append((area, face))

        inner = [f for a, f in faces if abs(a) > min_face_area and a > 0]
        inner_segs = set()
        for face in inner:
            for (u, v) in face:
                inner_segs.add(he[(u, v)])

        keep = [s for i, s in enumerate(segs) if i in inner_segs]
        info['iterations'] += 1
        info['rooms'] = len(inner)
        if len(keep) == len(segs):
            break
        info['removed'] += len(segs) - len(keep)
        segs = keep

    used = {n for s in segs for n in s[:2]}
    for n in list(nodes):
        if n not in used:
            nodes.pop(n)
    return segs, info


def write_lay(result: dict, out_path: Path, wall_thickness: float, wall_eps: float,
              wall_sigma: float, roughness: float = 0.0,
              node_tol: float = 0.005) -> dict:
    registry, rows = dedupe_and_split(result['segments'], node_tol=node_tol)
    node_pos = {n: np.array(p, float) for p, n in registry.items()}
    node_pos, rows = planarize(node_pos, rows, tol=node_tol)
    rows, n_pruned = prune_dangling(node_pos, rows)
    rows, plan_info = clean_planar(node_pos, rows)
    registry = {(round(float(p[0]), 3), round(float(p[1]), 3)): n
                for n, p in node_pos.items()}

    lines = ['[info]', 'format = cart', 'version = 1.3', 'type = indoor', '']
    lines.append('[points]')
    for p, n in sorted(registry.items(), key=lambda kv: kv[1]):
        lines.append(f'{n} = ({p[0]:.3f}, {p[1]:.3f})')
    lines.append('')
    lines.append('[segments]')
    for i, (a, b, z, mat) in enumerate(rows, start=1):
        lines.append(f"{i} = {{'name': '{mat}', 'connect': [{a}, {b}], "
                     f"'z': ({z[0]:.3f}, {z[1]:.3f})}}")
    lines += ['', '[files]', 'furniture =', '']
    lines.append('[slabs]')
    lines.append(f"WALL = {{'color': 'grey30', 'linewidth': 3, 'lthick': [{wall_thickness}], "
                 f"'lmatname': ['CONCRETE']}}")
    lines.append("AIR = {'color': 'white', 'linewidth': 1, 'lthick': [0.01], 'lmatname': ['AIR']}")
    lines.append("_AIR = {'color': 'white', 'linewidth': 1, 'lthick': [0.01], 'lmatname': ['AIR']}")
    lines.append(f"CEIL = {{'color': 'grey20', 'linewidth': 1, 'lthick': [{wall_thickness}], "
                 f"'lmatname': ['CONCRETE']}}")
    lines.append(f"FLOOR = {{'color': 'grey40', 'linewidth': 1, 'lthick': [{wall_thickness}], "
                 f"'lmatname': ['CONCRETE']}}")
    lines += ['', '[materials]']
    lines.append(f"CONCRETE = {{'epr': ({wall_eps:.4f}+0j), 'mur': (1+0j), "
                 f"'sigma': {wall_sigma:.6f}, 'roughness': {roughness}}}")
    lines.append("AIR = {'epr': (1+0j), 'mur': (1+0j), 'sigma': 0.0, 'roughness': 0.0}")
    lines += ['', '[indoor]', f"zceil = {result['ceil_z']:.3f}",
              f"zfloor = {result['floor_z']:.3f}", '']

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(lines))
    return dict(points=len(registry), segments=len(rows), pruned_segments=n_pruned,
                rooms=plan_info["rooms"], plan_iterations=plan_info["iterations"],
                removed_open_walls=plan_info["removed"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='front3d_0015')
    ap.add_argument('--out', default=None)
    ap.add_argument('--wall-thickness', type=float, default=0.1,
                    help='slab thickness [m]; 0.1 matches Sionna DEFAULT_THICKNESS')
    ap.add_argument('--roughness', type=float, default=0.0)
    ap.add_argument('--include-furniture', action='store_true')
    ap.add_argument('--floor-z', type=float, default=None)
    ap.add_argument('--ceil-z', type=float, default=None)
    ap.add_argument('--offset-tol', type=float, default=0.015)
    ap.add_argument('--merge-sep', type=float, default=0.05,
                    help='drop one of two parallel wall sheets closer than this (m)')
    ap.add_argument('--node-tol', type=float, default=0.005)
    ap.add_argument('--report', default=None)
    args = ap.parse_args()

    result = extract_segments(args.scene, args.floor_z, args.ceil_z,
                              include_furniture=args.include_furniture,
                              offset_tol=args.offset_tol, merge_sep=args.merge_sep)
    eps, sigma = itu_eps_sigma('concrete', 3.438)
    out_path = Path(args.out) if args.out else (SUBROOT / 'outputs' / 'layouts'
                                                / f'{args.scene}.lay')
    stats = write_lay(result, out_path, args.wall_thickness, eps, sigma, args.roughness,
                      node_tol=args.node_tol)

    report = {k: v for k, v in result.items() if k != 'segments'}
    report.update(dict(
        lay_file=str(out_path),
        material=dict(name='CONCRETE', eps_r=eps, sigma_s_per_m=sigma,
                      thickness_m=args.wall_thickness,
                      source='ITU-R P.2040-3 coefficients copied from Sionna itu.py'),
        **stats,
    ))
    rpath = Path(args.report) if args.report else out_path.with_suffix('.report.json')
    rpath.write_text(json.dumps(report, indent=1))

    for k in ('scene_key', 'n_triangles', 'area_vertical', 'area_horizontal', 'area_other',
              'floor_z', 'ceil_z', 'n_wall_lines'):
        print(f'{k:16s}: {result[k] if k != "n_wall_lines" else result[k]}')
    print(f'{"levels":16s}: {[l["z"] for l in result["levels"]]}')
    print(f'{"wall lines>=0.5":16s}: {sum(1 for l in result["lines"] if l["area"] >= 0.5)}')
    print(f'{"lay file":16s}: {out_path}')
    print(f'{"points":16s}: {stats["points"]}')
    print(f'{"segments":16s}: {stats["segments"]} (pruned dangling: {stats.get("pruned_segments", 0)}, '
          f'dropped open walls: {stats.get("removed_open_walls", 0)})')
    print(f'{"rooms":16s}: {stats.get("rooms", 0)}')
    print(f'{"report":16s}: {rpath}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
