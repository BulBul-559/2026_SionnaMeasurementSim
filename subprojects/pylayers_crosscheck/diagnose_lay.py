"""Find the geometric defects that make PyLayers' constrained triangulation fail.

PyLayers builds the transmission graph with a constrained Delaunay triangulation
over the wall centrelines. That step only accepts a proper planar subdivision:
no crossings without a shared node, no collinear overlaps, no slivers.

Run:  python diagnose_lay.py --lay outputs/layouts/front3d_0015.lay
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np


def parse_lay(path: Path):
    pts, segs = {}, {}
    mode = None
    for line in path.read_text().splitlines():
        if line.startswith('['):
            mode = line.strip()
            continue
        if '=' not in line or mode not in ('[points]', '[segments]'):
            continue
        k, v = line.split('=', 1)
        if mode == '[points]':
            pts[int(k)] = np.array(ast.literal_eval(v), float)
        else:
            segs[int(k)] = ast.literal_eval(v.strip())
    return pts, segs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--lay', required=True)
    ap.add_argument('--tol', type=float, default=0.002)
    args = ap.parse_args()

    pts, segs = parse_lay(Path(args.lay))
    items = [(sid, s['connect'][0], s['connect'][1], s['z'], s['name'])
             for sid, s in segs.items()]

    dup_nodes = []
    pids = sorted(pts)
    for i, a in enumerate(pids):
        for b in pids[i + 1:]:
            if np.linalg.norm(pts[a] - pts[b]) <= args.tol:
                dup_nodes.append((a, b, float(np.linalg.norm(pts[a] - pts[b]))))

    crossings, collinear = [], []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            sid1, a1, b1, z1, m1 = items[i]
            sid2, a2, b2, z2, m2 = items[j]
            if {a1, b1} & {a2, b2}:
                continue
            p, r = pts[a1], pts[b1] - pts[a1]
            q, s = pts[a2], pts[b2] - pts[a2]
            rn, sn = np.linalg.norm(r), np.linalg.norm(s)
            if rn < 1e-9 or sn < 1e-9:
                continue
            cross = r[0] * s[1] - r[1] * s[0]
            qp = q - p
            if abs(cross) > 1e-9 * rn * sn:
                t = (qp[0] * s[1] - qp[1] * s[0]) / cross
                u = (qp[0] * r[1] - qp[1] * r[0]) / cross
                eps = args.tol / rn
                epsu = args.tol / sn
                if eps < t < 1 - eps and epsu < u < 1 - epsu:
                    x = p + t * r
                    near = min(np.linalg.norm(x - pts[k]) for k in pts)
                    crossings.append(dict(seg_a=sid1, seg_b=sid2,
                                          x=x.round(3).tolist(),
                                          nearest_node_m=round(float(near), 4)))
            else:
                dist = abs(qp[0] * r[1] - qp[1] * r[0]) / rn
                if dist > args.tol:
                    continue
                # collinear: project both onto r
                u1 = np.sort([0.0, rn])
                t2 = np.sort([(pts[a2] - p) @ r / rn, (pts[b2] - p) @ r / rn])
                lo, hi = max(u1[0], t2[0]), min(u1[1], t2[1])
                if hi - lo > args.tol:
                    collinear.append(dict(seg_a=sid1, seg_b=sid2,
                                          overlap_m=round(float(hi - lo), 3),
                                          z_a=z1, z_b=z2))

    print(f'nodes {len(pts)}, segments {len(items)}')
    print(f'duplicate/near-duplicate nodes (<= {args.tol * 1000:.0f} mm): {len(dup_nodes)}')
    for d in dup_nodes[:10]:
        print('   ', d)
    print(f'crossings without a shared node: {len(crossings)}')
    for c in crossings[:10]:
        print('   ', c)
    print(f'collinear overlaps: {len(collinear)}')
    for c in collinear[:10]:
        print('   ', c)
    return 0 if not (crossings or collinear) else 1


if __name__ == '__main__':
    raise SystemExit(main())
