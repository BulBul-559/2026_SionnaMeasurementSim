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


def _line_u_range(line: WallLine, p0, p1, p2, vi) -> tuple[float, float]:
    n, t = line_basis(line.az_deg)
    tri = vi[np.asarray(line.triangles)]
    pts = np.stack([p0[tri], p1[tri], p2[tri]], axis=1)
    u = pts[:, :, :2] @ t
    return float(u.min()), float(u.max())


def pair_wall_sheets(lines: list[WallLine], p0, p1, p2, vi, min_gap=0.03,
                     max_gap=0.35, min_overlap=0.5, az_tol=0.8) -> tuple[list[WallLine], int]:
    """Collapse the two faces of one wall into a single centreline.

    Front3D models a 0.24 m wall as two separate sheets (the inner faces of the
    two rooms). Keeping both makes PyLayers enclose a thin cavity that it counts
    as a room, which breaks its graph build. Pairing the sheets and using the
    area-weighted mean offset yields one zero-thickness wall line - the
    representation PyLayers' 2.5D engine actually models (wall thickness then
    lives in the slab stack, not in the geometry).
    """
    ranges = [_line_u_range(l, p0, p1, p2, vi) for l in lines]
    used = [False] * len(lines)
    order = sorted(range(len(lines)), key=lambda i: -lines[i].area)
    merged: list[WallLine] = []
    n_pairs = 0

    for i in order:
        if used[i]:
            continue
        a = lines[i]
        best, best_gap = None, None
        for j in range(len(lines)):
            if j == i or used[j]:
                continue
            b = lines[j]
            daz = abs((a.az_deg - b.az_deg + 90.0) % 180.0 - 90.0)
            if daz > az_tol:
                continue
            gap = abs(b.offset - a.offset)
            if not (min_gap < gap <= max_gap):
                continue
            lo = max(ranges[i][0], ranges[j][0])
            hi = min(ranges[i][1], ranges[j][1])
            span = min(ranges[i][1] - ranges[i][0], ranges[j][1] - ranges[j][0])
            if span <= 0 or (hi - lo) / span < min_overlap:
                continue
            if best is None or gap < best_gap:
                best, best_gap = j, gap
        if best is None:
            merged.append(a)
            used[i] = True
            continue
        b = lines[best]
        w = a.area + b.area
        merged.append(WallLine(
            az_deg=a.az_deg,
            offset=(a.offset * a.area + b.offset * b.area) / w if w else a.offset,
            triangles=list(a.triangles) + list(b.triangles),
            area=w,
        ))
        used[i] = used[best] = True
        n_pairs += 1
    return merged, n_pairs


def close_corners(nodes: dict[int, np.ndarray], segs: list[tuple], max_ext=0.5,
                  min_angle_deg=30.0, passes=3):
    """Returns (nodes, segs, n_moved)."""
    """Extend wall ends to the perpendicular wall they were supposed to meet.

    After the sheets are collapsed to centrelines, every corner is short by about
    half a wall thickness on each side, so the network has small gaps. Each gap is
    closed by moving the wall end onto the intersection with the crossing wall -
    an extension of at most `max_ext`, which is far below any room dimension.
    """
    nodes = dict(nodes)
    n_moved = 0
    cos_min = np.cos(np.radians(min_angle_deg))

    for _ in range(passes):
        moved_this_pass = 0
        incident: dict[int, list[int]] = {}
        for k, (a, b, _z, _m) in enumerate(segs):
            incident.setdefault(a, []).append(k)
            incident.setdefault(b, []).append(k)
        for nid, seg_ids in list(incident.items()):
            if len(seg_ids) > 2:      # already a junction
                continue
            p = nodes[nid]
            best_target, best_shift = None, None
            for k in seg_ids:
                a, b = segs[k][0], segs[k][1]
                other = nodes[b] if a == nid else nodes[a]
                d = p - other
                ln = float(np.linalg.norm(d))
                if ln < 1e-6:
                    continue
                d = d / ln
                for k2, (c, e, _z2, _m2) in enumerate(segs):
                    if k2 == k:
                        continue
                    q = nodes[c]
                    s = nodes[e] - nodes[c]
                    sn = float(np.linalg.norm(s))
                    if sn < 1e-6:
                        continue
                    s = s / sn
                    if abs(float(d @ s)) > cos_min:
                        continue          # nearly parallel, not a corner
                    den = d[0] * (-s[1]) - d[1] * (-s[0])
                    if abs(den) < 1e-9:
                        continue
                    # solve p + t*d = q + u*s
                    rhs = q - p
                    t = (rhs[0] * (-s[1]) - rhs[1] * (-s[0])) / den
                    u = (d[0] * rhs[1] - d[1] * rhs[0]) / den
                    if not (0.0 < t <= max_ext):
                        continue
                    if not (-max_ext / sn <= u <= 1.0 + max_ext / sn):
                        continue
                    if best_shift is None or t < best_shift:
                        best_target = p + t * d
                        best_shift = t
            if best_target is not None:
                nodes[nid] = np.array([snap(float(best_target[0])), snap(float(best_target[1]))])
                moved_this_pass += 1
        n_moved += moved_this_pass
        if moved_this_pass == 0:
            break
    return nodes, segs, n_moved


def extract_segments(scene_key: str, floor_z: float | None, ceil_z: float | None,
                     include_furniture: bool = False, offset_tol: float = 0.015,
                     node_tol: float = 0.005, merge_sep: float = 0.05,
                     reconstruct: bool = True) -> dict:
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
    n_pairs = 0
    if reconstruct:
        lines, n_pairs = pair_wall_sheets(lines, p0, p1, p2, vi)

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
        n_sheet_pairs=n_pairs,
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

    for _ in range(max_iter):
        # work on unique endpoint pairs: iso segments (same pair, different z)
        # must stay together, and a half-edge map keyed by ids would collide
        pair_of = [(min(a, b), max(a, b)) for a, b, _z, _m in segs]
        pairs = sorted(set(pair_of))
        out: dict[int, list[int]] = {}
        for (u, v) in pairs:
            out.setdefault(u, []).append(v)
            out.setdefault(v, []).append(u)
        for u, vs in out.items():
            vs.sort(key=lambda v: np.arctan2(nodes[v][1] - nodes[u][1],
                                             nodes[v][0] - nodes[u][0]))

        pos_in = {(u, v): out[u].index(v) for (u, v) in
                  [(a, b) for a, b in pairs] + [(b, a) for a, b in pairs]}
        seen = set()
        faces = []
        for (u, v) in [(a, b) for a, b in pairs] + [(b, a) for a, b in pairs]:
            if (u, v) in seen:
                continue
            face = []
            cur = (u, v)
            guard = 0
            while cur not in seen and guard < 4 * len(pairs) + 10:
                seen.add(cur)
                face.append(cur)
                r = (cur[1], cur[0])
                lst = out[cur[1]]
                nxt = lst[(pos_in[r] - 1) % len(lst)]
                cur = (cur[1], nxt)
                guard += 1
            if len(face) >= 3:
                xy = np.array([nodes[a] for (a, _b) in face], float)
                area = 0.5 * float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1)
                                           - np.roll(xy[:, 0], -1) * xy[:, 1]))
                faces.append((area, face))

        inner = [f for a, f in faces if abs(a) > min_face_area and a > 0]
        keep_pairs = set()
        for face in inner:
            for (a, b) in face:
                keep_pairs.add((min(a, b), max(a, b)))

        keep = [s for s, pr in zip(segs, pair_of) if pr in keep_pairs]
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


def resolve_collinear_overlaps(nodes: dict[int, np.ndarray], segs: list[tuple],
                               tol=0.005, max_iter=50):
    """Split collinear overlapping segments so they only ever share endpoints.

    Wall panels from the architecture can overlap along the same line (corner
    panels, exterior vs interior walls, AIR bridges over a wall piece). A
    constrained triangulation rejects that, so the overlapping parts are split
    at the overlap boundaries; the duplicated piece then collapses in
    `merge_coincident_nodes`.
    """
    nodes = dict(nodes)
    segs = list(segs)
    nxt = min(nodes) - 1 if nodes else -1

    def nid_at(p):
        nonlocal nxt
        key = np.array([snap(float(p[0])), snap(float(p[1]))], float)
        best, bd = None, None
        for nid, q in nodes.items():
            d = float(np.linalg.norm(q - key))
            if d <= tol and (bd is None or d < bd):
                best, bd = nid, d
        if best is not None:
            return best
        nid = nxt
        nxt -= 1
        nodes[nid] = key
        return nid

    for _ in range(max_iter):
        changed = False
        for i in range(len(segs)):
            a1, b1 = segs[i][0], segs[i][1]
            p, r = nodes[a1], nodes[b1] - nodes[a1]
            rn = float(np.linalg.norm(r))
            if rn < 1e-9:
                continue
            u = r / rn
            for j in range(i + 1, len(segs)):
                a2, b2 = segs[j][0], segs[j][1]
                if {a1, b1} & {a2, b2}:
                    continue
                q, s = nodes[a2], nodes[b2] - nodes[a2]
                sn = float(np.linalg.norm(s))
                if sn < 1e-9:
                    continue
                if abs(float(u @ (s / sn))) < 0.9995:
                    continue
                dist = abs((q - p)[0] * u[1] - (q - p)[1] * u[0])
                if dist > tol:
                    continue
                t2 = np.sort([float((nodes[a2] - p) @ u), float((nodes[b2] - p) @ u)])
                lo, hi = max(0.0, t2[0]), min(rn, t2[1])
                if hi - lo <= tol:
                    continue
                for cut in (lo, hi):
                    if tol < cut < rn - tol:
                        nid = nid_at(p + cut * u)
                        a, b, z, m = segs[i]
                        segs[i:i + 1] = [(a, nid, z, m)]
                        segs.insert(i + 1, (nid, b, z, m))
                        changed = True
                        break
                if changed:
                    break
                for cut in (lo, hi):
                    if t2[0] + tol < cut < t2[1] - tol:
                        nid = nid_at(p + cut * u)
                        a, b, z, m = segs[j]
                        segs[j:j + 1] = [(a, nid, z, m)]
                        segs.insert(j + 1, (nid, b, z, m))
                        changed = True
                        break
                if changed:
                    break
            if changed:
                break
        if not changed:
            break
    return nodes, segs


def merge_coincident_nodes(nodes: dict[int, np.ndarray], segs: list[tuple],
                          ) -> tuple[dict[int, np.ndarray], list[tuple]]:
    """Merge nodes that landed on the same snapped position, remapping segments.

    Corner closing can push a wall end onto an existing node; without this the
    layout would carry two coincident points and the written file loses one of
    them (PyLayers then fails to find the segment's endpoint).
    """
    by_key: dict[tuple[float, float], int] = {}
    remap: dict[int, int] = {}
    for nid, p in nodes.items():
        key = (snap(float(p[0])), snap(float(p[1])))
        if key in by_key:
            remap[nid] = by_key[key]
        else:
            by_key[key] = nid
            remap[nid] = nid
    out_nodes = {nid: np.array(k, float) for k, nid in by_key.items()}
    out_segs, seen = [], set()
    for a, b, z, m in segs:
        a2, b2 = remap.get(a, a), remap.get(b, b)
        if a2 == b2:
            continue
        key = (min(a2, b2), max(a2, b2), z)
        if key in seen:
            continue
        seen.add(key)
        out_segs.append((a2, b2, z, m))
    return out_nodes, out_segs


def write_lay(result: dict, out_path: Path, wall_thickness: float, wall_eps: float,
              wall_sigma: float, roughness: float = 0.0,
              node_tol: float = 0.005, clean: bool = True) -> dict:
    registry, rows = dedupe_and_split(result['segments'], node_tol=node_tol)
    node_pos = {n: np.array(p, float) for p, n in registry.items()}
    node_pos, rows = planarize(node_pos, rows, tol=node_tol)
    if clean:
        node_pos, rows, n_corners = close_corners(node_pos, rows)
        node_pos, rows = planarize(node_pos, rows, tol=node_tol)
        rows, n_pruned = prune_dangling(node_pos, rows)
        rows, plan_info = clean_planar(node_pos, rows)
    else:
        n_corners, n_pruned = 0, 0
        plan_info = dict(rooms=-1, removed=0, iterations=0)
    node_pos, rows = resolve_collinear_overlaps(node_pos, rows)
    node_pos, rows = merge_coincident_nodes(node_pos, rows)
    # final consistency filter: every written segment must reference written points
    rows = [r for r in rows if r[0] in node_pos and r[1] in node_pos]
    used_final = {n for r in rows for n in r[:2]}
    node_pos = {n: p for n, p in node_pos.items() if n in used_final}
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
                corners_closed=n_corners,
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
    ap.add_argument('--no-reconstruct', action='store_true',
                    help='keep the mesh two-sheet walls instead of collapsing to centrelines')
    ap.add_argument('--node-tol', type=float, default=0.005)
    ap.add_argument('--report', default=None)
    args = ap.parse_args()

    result = extract_segments(args.scene, args.floor_z, args.ceil_z,
                              include_furniture=args.include_furniture,
                              offset_tol=args.offset_tol, merge_sep=args.merge_sep,
                              reconstruct=not args.no_reconstruct)
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
    print(f'{"sheet pairs":16s}: {result.get("n_sheet_pairs", 0)}')
    print(f'{"corners closed":16s}: {stats.get("corners_closed", 0)}')
    print(f'{"report":16s}: {rpath}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
