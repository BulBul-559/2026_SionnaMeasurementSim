"""Pick LoS / 1-wall / 2-wall (and optionally 3-wall) links for the cross-check.

Wall crossings are counted on the SceneGen floorplan class mask (5 cm grid,
furniture ignored by contract), i.e. from a third, independent geometry source -
neither Sionna nor PyLayers decides which link is "through N walls".

Run:  python select_links.py --scene front3d_0015
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import SCENE_ROOT, SUBROOT, load_label  # noqa: E402


def load_mask(scene_key: str):
    root = SCENE_ROOT / scene_key / 'floorplan'
    mask = np.load(root / 'class_mask.npy')
    meta = json.loads((root / 'class_mask_meta.json').read_text())
    res = float(meta['resolution_m_per_pixel'])
    origin = np.asarray(meta['origin_xy_m'], float)
    return mask, res, origin


def count_wall_crossings(mask, res, origin, p_xy, q_xy, step=0.01):
    """Number of distinct wall runs crossed on the straight 2D line p->q."""
    d = np.asarray(q_xy, float) - np.asarray(p_xy, float)
    dist = float(np.linalg.norm(d))
    if dist < 1e-6:
        return 0, 0.0, False
    n = max(2, int(np.ceil(dist / step)) + 1)
    t = np.linspace(0.0, 1.0, n)
    pts = np.asarray(p_xy, float)[None, :] + t[:, None] * d[None, :]
    ij = np.floor((pts - origin) / res).astype(int)
    h, w = mask.shape[:2]
    inside = (ij[:, 0] >= 0) & (ij[:, 0] < w) & (ij[:, 1] >= 0) & (ij[:, 1] < h)
    is_wall = np.zeros(n, bool)
    is_wall[inside] = mask[ij[inside, 1], ij[inside, 0]] == 1
    runs = int(np.count_nonzero(is_wall[1:] & ~is_wall[:-1]) + (1 if is_wall[0] else 0))
    return runs, dist, bool(inside.all())


def is_free(mask, res, origin, xy):
    ij = np.floor((np.asarray(xy, float) - origin) / res).astype(int)
    h, w = mask.shape[:2]
    if not (0 <= ij[0] < w and 0 <= ij[1] < h):
        return False
    return int(mask[ij[1], ij[0]]) in (2, 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='front3d_0015')
    ap.add_argument('--min-dist', type=float, default=2.5)
    ap.add_argument('--max-dist', type=float, default=9.0)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    label = load_label(args.scene)
    bs = np.asarray(label['bs_positions'], float)
    ue = np.asarray(label['ue_positions'], float)
    mask, res, origin = load_mask(args.scene)

    cands = []
    for ib in range(len(bs)):
        for iu in range(len(ue)):
            if not is_free(mask, res, origin, bs[ib, :2]):
                continue
            if not is_free(mask, res, origin, ue[iu, :2]):
                continue
            runs, dist, inside = count_wall_crossings(mask, res, origin, bs[ib, :2], ue[iu, :2])
            if not inside or dist < args.min_dist or dist > args.max_dist:
                continue
            cands.append(dict(bs_id=ib, ue_id=iu, walls=runs, dist_2d=round(dist, 3)))

    print(f'candidate links: {len(cands)}')
    for w in sorted({c['walls'] for c in cands}):
        n = sum(1 for c in cands if c['walls'] == w)
        print(f'   walls={w}: {n}')

    # deterministic selection: closest to the median distance within each class
    chosen = []
    targets = [(0, 1), (1, 1), (2, 2), (3, 1)]
    for walls, k in targets:
        pool = [c for c in cands if c['walls'] == walls]
        if not pool:
            continue
        pool.sort(key=lambda c: abs(c['dist_2d'] - float(np.median([p['dist_2d'] for p in pool]))))
        used_ue = {c['ue_id'] for c in chosen}
        picked = 0
        for c in pool:
            if c['ue_id'] in used_ue:
                continue
            c = dict(c)
            c['bs_pos'] = bs[c['bs_id']].tolist()
            c['ue_pos'] = ue[c['ue_id']].tolist()
            c['bs_yaw_rad'] = list(label['bs_points'][c['bs_id']].get('orientation_rad', [0, 0, 0]))
            c['ue_yaw_rad'] = list(label['ue_points'][c['ue_id']].get('orientation_rad', [0, 0, 0]))
            c['bs_label'] = label['bs_points'][c['bs_id']].get('label')
            c['ue_label'] = label['ue_points'][c['ue_id']].get('label')
            chosen.append(c)
            used_ue.add(c['ue_id'])
            picked += 1
            if picked >= k:
                break

    for c in chosen:
        print(f"   picked walls={c['walls']} dist={c['dist_2d']:.2f} m "
              f"bs={c['bs_label']}({c['bs_id']}) ue={c['ue_label']}({c['ue_id']})")

    out = Path(args.out) if args.out else (SUBROOT / 'outputs' / 'links'
                                          / f'{args.scene}_links.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(scene_key=args.scene, policy=dict(
        wall_source='SceneGen floorplan class_mask (5 cm), furniture ignored',
        min_dist_m=args.min_dist, max_dist_m=args.max_dist,
        note='one LoS, one 1-wall, two 2-wall, one 3-wall where available'), links=chosen),
        indent=1))
    print('saved', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
