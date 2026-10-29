"""Visual sanity check: overlay the generated .lay walls on the SceneGen floorplan.

The class mask is the 5 cm plan the 339-scene pipeline used for its own
floorplan/mask products, so it is an independent reference for "did the
converter keep the same walls".

Run:  python check_layout.py --scene front3d_0015
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import SCENE_ROOT, SUBROOT  # noqa: E402


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
            pts[int(k)] = ast.literal_eval(v.strip())
        else:
            segs[int(k)] = ast.literal_eval(v.strip())
    return pts, segs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='front3d_0015')
    ap.add_argument('--lay', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    lay = Path(args.lay) if args.lay else SUBROOT / 'outputs' / 'layouts' / f'{args.scene}.lay'
    fp = SCENE_ROOT / args.scene / 'floorplan'
    mask = np.load(fp / 'class_mask.npy')
    import json
    meta = json.loads((fp / 'class_mask_meta.json').read_text())
    res = float(meta['resolution_m_per_pixel'])
    origin = np.asarray(meta['origin_xy_m'], float)

    pts, segs = parse_lay(lay)
    h, w = mask.shape[:2]
    extent = [origin[0], origin[0] + w * res, origin[1], origin[1] + h * res]

    fig, axes = plt.subplots(1, 2, figsize=(15, 9))
    cmap = matplotlib.colors.ListedColormap(['#f5f5f5', '#323232', '#68b3e8', '#dc8b37'])
    for ax, title in zip(axes, ('SceneGen class mask', 'class mask + generated .lay')):
        ax.imshow(mask, origin='lower', extent=extent, cmap=cmap, vmin=0, vmax=3,
                  interpolation='nearest')
        ax.set_title(title)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_aspect('equal')
    ax = axes[1]
    for seg in segs.values():
        a, b = seg['connect']
        pa, pb = pts[a], pts[b]
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]], '-', color='red', lw=1.2)
    ax.plot([], [], 'r-', label=f'{len(segs)} wall segments')
    ax.legend(loc='upper right')

    out = Path(args.out) if args.out else (
        SUBROOT / 'outputs' / 'reports' / f'{args.scene}_layout_overlay.png')
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    print('saved', out)

    # quantitative overlap: fraction of wall-mask pixels within 15 cm of a segment
    from matplotlib.path import Path as MPath
    ys, xs = np.nonzero(mask == 1)
    wx = origin[0] + (xs + 0.5) * res
    wy = origin[1] + (ys + 0.5) * res
    near = np.zeros(len(wx), bool)
    p_all = np.stack([wx, wy], axis=1)
    for seg in segs.values():
        a, b = seg['connect']
        pa, pb = np.array(pts[a]), np.array(pts[b])
        d = pb - pa
        ln2 = float(d @ d)
        if ln2 < 1e-9:
            continue
        tpar = np.clip(((p_all - pa) @ d) / ln2, 0.0, 1.0)
        proj = pa[None, :] + tpar[:, None] * d[None, :]
        near |= np.linalg.norm(p_all - proj, axis=1) <= 0.15
    print(f'wall-mask pixels: {len(wx)}, covered by a segment within 15 cm: '
          f'{near.sum()} ({near.mean() * 100:.1f}%)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
