"""Build a PyLayers `.lay` from the SceneGen floorplan wall raster.

Why not the mesh: Front3D models every wall as two sheets 0.24 m apart. Feeding
both to PyLayers encloses a thin cavity that PyLayers counts as a room, and
collapsing the sheets post-hoc distorts corners (measured: 42 % of wall pixels
still covered, visibly skewed walls).

The SceneGen floorplan mask is a 5 cm raster whose `wall` class is 1-2 pixels
wide - effectively an already-thinned wall centreline raster produced by the
same pipeline that the dataset used for its LOS/NLOS labels. It therefore gives
a clean, single-line, closed plan by construction.

Pipeline:
  1. wall raster  = class_mask == 1  (furniture ignored, empty scene)
  2. extract axis-aligned runs (rows -> horizontal, columns -> vertical)
  3. cluster parallel runs into single centreline segments
  4. snap endpoints, close corners onto crossing walls, fill plan gaps with AIR
  5. write .lay with the same ITU concrete slab as the mesh-based converter

Run:  python build_lay_from_mask.py --scene front3d_0015
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import SCENE_ROOT, SUBROOT, itu_eps_sigma  # noqa: E402

SNAP = 0.05   # mask resolution


@dataclass
class Seg:
    p_a: tuple[float, float]
    p_b: tuple[float, float]
    z: tuple[float, float]
    material: str = 'WALL'


def load_mask(scene_key: str):
    root = SCENE_ROOT / scene_key / 'floorplan'
    mask = np.load(root / 'class_mask.npy')
    meta = json.loads((root / 'class_mask_meta.json').read_text())
    return mask, float(meta['resolution_m_per_pixel']), np.asarray(meta['origin_xy_m'], float)


def runs_1d(flags: np.ndarray) -> list[tuple[int, int]]:
    """Maximal [start, end] runs of True."""
    out = []
    i = 0
    n = len(flags)
    while i < n:
        if flags[i]:
            j = i
            while j + 1 < n and flags[j + 1]:
                j += 1
            out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def extract_axis_segments(wall: np.ndarray, res: float, origin: np.ndarray):
    """Axis-aligned wall centrelines from the raster, via union-find run merging.

    A wall is 1-2 px thick, so a line shows up as runs in neighbouring rows (or
    columns). Runs that overlap between adjacent lines are merged into one group;
    the group's mean index becomes the centreline and its union becomes the
    extent. O(n) with a per-line index, unlike the naive O(n^2) clustering.
    """
    h, w = wall.shape

    def merge_runs(runs_by_line, n_lines):
        parent = {}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        items = []
        for line in range(n_lines):
            for (s0, s1) in runs_by_line.get(line, []):
                parent[len(items)] = len(items)
                items.append((line, s0, s1))
        by_line: dict[int, list[int]] = {}
        for idx, (line, _s0, _s1) in enumerate(items):
            by_line.setdefault(line, []).append(idx)
        for line in range(n_lines - 1):
            for i in by_line.get(line, []):
                for j in by_line.get(line + 1, []):
                    ov = min(items[i][2], items[j][2]) - max(items[i][1], items[j][1])
                    long_ = max(items[i][2] - items[i][1], items[j][2] - items[j][1])
                    if long_ > 0 and ov >= 0.8 * long_:
                        union(i, j)
        groups: dict[int, list[tuple[int, int, int]]] = {}
        for idx, (line, s0, s1) in enumerate(items):
            groups.setdefault(find(idx), []).append((line, s0, s1))
        out = []
        for g in groups.values():
            line_mean = float(np.mean([x[0] for x in g]))
            lo = float(min(x[1] for x in g))
            hi = float(max(x[2] for x in g))
            out.append((line_mean, lo, hi))
        return out

    horiz_runs: dict[int, list[tuple[int, int]]] = {}
    for r in range(h):
        rr = runs_1d(wall[r, :])
        if rr:
            horiz_runs[r] = rr
    vert_runs: dict[int, list[tuple[int, int]]] = {}
    for c in range(w):
        rc = runs_1d(wall[:, c])
        if rc:
            vert_runs[c] = rc

    hsegs = merge_runs(horiz_runs, h)
    vsegs = merge_runs(vert_runs, w)

    segs: list[Seg] = []
    for r, c0, c1 in hsegs:
        if (c1 - c0) < 2:
            continue
        y = origin[1] + (r + 0.5) * res
        x0 = origin[0] + (c0 + 0.5) * res
        x1 = origin[0] + (c1 + 0.5) * res
        segs.append(Seg((round(x0, 3), round(y, 3)), (round(x1, 3), round(y, 3)), (0.0, 0.0)))
    for c, r0, r1 in vsegs:
        if (r1 - r0) < 2:
            continue
        x = origin[0] + (c + 0.5) * res
        y0 = origin[1] + (r0 + 0.5) * res
        y1 = origin[1] + (r1 + 0.5) * res
        segs.append(Seg((round(x, 3), round(y0, 3)), (round(x, 3), round(y1, 3)), (0.0, 0.0)))
    return segs, dict(h_runs=sum(len(v) for v in horiz_runs.values()),
                      v_runs=sum(len(v) for v in vert_runs.values()),
                      h_lines=len(hsegs), v_lines=len(vsegs))



def bridge_collinear_gaps(segs: list[Seg], tol=0.06) -> tuple[list[Seg], int]:
    """Close collinear gaps (doorways, passages) with transparent AIR segments.

    The wall raster is broken at every opening, which leaves the wall network an
    open tree; PyLayers needs a closed subdivision. An AIR segment spans the gap
    without reflection or transmission loss, so the physics still sees a hole.
    """
    out = list(segs)
    n_added = 0
    for axis in ('h', 'v'):
        groups: dict[float, list[Seg]] = {}
        for s in segs:
            a, b = s.p_a, s.p_b
            if axis == 'h' and abs(a[1] - b[1]) < 0.005:
                key = round(a[1] / tol) * tol
            elif axis == 'v' and abs(a[0] - b[0]) < 0.005:
                key = round(a[0] / tol) * tol
            else:
                continue
            groups.setdefault(key, []).append(s)
        for key, g in groups.items():
            spans = []
            for s in g:
                a, b = s.p_a, s.p_b
                lo, hi = (min(a[0], b[0]), max(a[0], b[0])) if axis == 'h' \
                    else (min(a[1], b[1]), max(a[1], b[1]))
                spans.append((lo, hi))
            spans.sort()
            merged = [list(spans[0])]
            for lo, hi in spans[1:]:
                if lo <= merged[-1][1] + 0.005:
                    merged[-1][1] = max(merged[-1][1], hi)
                else:
                    merged.append([lo, hi])
            for k in range(len(merged) - 1):
                g0, g1 = merged[k][1], merged[k + 1][0]
                if g1 - g0 < 0.03:
                    continue
                if axis == 'h':
                    out.append(Seg((round(g0, 3), key), (round(g1, 3), key), (0.0, 0.0), 'AIR'))
                else:
                    out.append(Seg((key, round(g0, 3)), (key, round(g1, 3)), (0.0, 0.0), 'AIR'))
                n_added += 1
    return out, n_added


def count_rooms(mask: np.ndarray) -> int:
    """Connected components of free space (4-neighbourhood), furniture counted as free."""
    free = (mask == 2) | (mask == 3)
    h, w = free.shape
    lab = np.zeros((h, w), int)
    cur = 0
    for r in range(h):
        for c in range(w):
            if free[r, c] and lab[r, c] == 0:
                cur += 1
                stack = [(r, c)]
                lab[r, c] = cur
                while stack:
                    y, x = stack.pop()
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and free[ny, nx] and lab[ny, nx] == 0:
                            lab[ny, nx] = cur
                            stack.append((ny, nx))
    areas = np.bincount(lab.ravel())[1:]
    return int(np.sum(areas * 0.0 + 1) - 1) if False else int((areas >= 20).sum())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='front3d_0015')
    ap.add_argument('--floor-z', type=float, default=-0.1)
    ap.add_argument('--ceil-z', type=float, default=2.8)
    ap.add_argument('--wall-thickness', type=float, default=0.1)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    import build_lay as bl

    mask, res, origin = load_mask(args.scene)
    wall = (mask == 1)
    segs, info = extract_axis_segments(wall, res, origin)
    print(f'raster runs: {info}')

    # reuse the mesh converter's graph hygiene: nodes -> planarize -> prune -> clean
    bl.SNAP_MM = 50.0   # everything already sits on a 5 cm grid
    registry: dict[tuple[float, float], int] = {}

    def nid(p):
        key = (bl.snap(float(p[0])), bl.snap(float(p[1])))
        if key not in registry:
            registry[key] = -(len(registry) + 1)
        return registry[key]

    segs, n_bridged = bridge_collinear_gaps(segs)
    print(f'bridged AIR gaps: {n_bridged}')
    rows = []
    for s in segs:
        a, b = nid(s.p_a), nid(s.p_b)
        if a != b:
            rows.append((a, b, (args.floor_z, args.ceil_z), s.material))
    nodes = {n: np.array(p, float) for p, n in registry.items()}
    print(f'raw segments: {len(rows)}, nodes: {len(nodes)}')

    nodes, rows = bl.planarize(nodes, rows, tol=0.06)
    nodes, rows, n_corners = bl.close_corners(nodes, rows, max_ext=0.12)
    nodes, rows = bl.planarize(nodes, rows, tol=0.06)
    rows, n_pruned = bl.prune_dangling(nodes, rows)
    rows, plan_info = bl.clean_planar(nodes, rows, min_face_area=1.0)
    nodes, rows = bl.merge_coincident_nodes(nodes, rows)
    used = {n for r in rows for n in r[:2]}
    nodes = {n: p for n, p in nodes.items() if n in used}
    print(f'after cleanup: segments {len(rows)}, nodes {len(nodes)}, '
          f'pruned {n_pruned}, dropped {plan_info["removed"]}, faces {plan_info["rooms"]}')
    print(f'rooms in mask free space: {count_rooms(mask)}')

    eps, sigma = itu_eps_sigma('concrete', 3.438)
    out_path = Path(args.out) if args.out else (
        SUBROOT / 'outputs' / 'layouts' / f'{args.scene}_mask.lay')
    result = dict(ceil_z=args.ceil_z, floor_z=args.floor_z, segments=[
        bl.Segment(p_a=tuple(nodes[r[0]]), p_b=tuple(nodes[r[1]]), z=r[2], material=r[3])
        for r in rows])
    stats = bl.write_lay(result, out_path, args.wall_thickness, eps, sigma)
    print(f'layout: {out_path}  points={stats["points"]} segments={stats["segments"]}')

    report = dict(scene=args.scene, source='SceneGen floorplan class_mask (wall class)',
                  resolution_m=res, raw=info, faces=plan_info['rooms'],
                  mask_rooms=count_rooms(mask), **stats)
    (out_path.with_suffix('.report.json')).write_text(json.dumps(report, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
