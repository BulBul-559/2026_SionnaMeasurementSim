"""Diagnostics: what does a structure-only Front3D scene look like geometrically?

Prints the material composition, floor/ceiling levels, the grouped vertical
wall planes (azimuth / offset / extent / z-range) and interior apertures, so the
`.lay` conversion can be designed against real numbers instead of guesses.

Usage:  python inspect_geom.py front3d_0015 [--top 25]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import SCENE_ROOT, load_scene, triangle_geometry  # noqa: E402


def cluster_levels(zs, areas, tol=0.05, min_area=2.0):
    order = np.argsort(zs)
    levels = []
    for k in order:
        z, a = float(zs[k]), float(areas[k])
        if levels and abs(z - levels[-1][0]) <= tol:
            w = levels[-1][1] + a
            levels[-1][0] = (levels[-1][0] * levels[-1][1] + z * a) / w
            levels[-1][1] = w
        else:
            levels.append([z, a])
    return [l for l in levels if l[1] >= min_area]


def plane_groups(p0, p1, p2, nh, area, az_bin=1.0, off_bin=0.005):
    az = np.degrees(np.arctan2(nh[:, 1], nh[:, 0]))
    az = (np.round(az / az_bin).astype(int) % int(round(360 / az_bin))) * az_bin
    off = np.sum(nh[:, :2] * p0[:, :2], axis=1)
    off = np.round(off / off_bin).astype(int)
    groups: dict[tuple[float, int], list[int]] = {}
    for i in range(len(area)):
        groups.setdefault((float(az[i]), int(off[i])), []).append(i)
    return groups


def coverage(uv, faces, res):
    lo, hi = uv.min(0), uv.max(0)
    span = hi - lo
    if span[0] <= 1e-9 or span[1] <= 1e-9:
        return 1.0, 0.0
    nx = max(2, int(np.ceil(span[0] / res)))
    nz = max(2, int(np.ceil(span[1] / res)))
    du, dz = span[0] / nx, span[1] / nz
    mask = np.zeros((nx, nz), bool)
    gx = lo[0] + (np.arange(nx) + 0.5) * du
    gz = lo[1] + (np.arange(nz) + 0.5) * dz
    n = len(faces)
    for k in range(n):
        tri = uv[faces[k]]
        u0, v0 = tri.min(0)
        u1, v1 = tri.max(0)
        i0, i1 = max(0, int((u0 - lo[0]) / du) - 1), min(nx, int((u1 - lo[0]) / du) + 2)
        j0, j1 = max(0, int((v0 - lo[1]) / dz) - 1), min(nz, int((v1 - lo[1]) / dz) + 2)
        if i1 <= i0 or j1 <= j0:
            continue
        X, Z = gx[i0:i1, None], gz[None, j0:j1]
        ax, ay = tri[0]
        bx, by = tri[1]
        cx, cy = tri[2]
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-14:
            continue
        w0 = ((by - cy) * (X - cx) + (cx - bx) * (Z - cy)) / den
        w1 = ((cy - ay) * (X - cx) + (ax - cx) * (Z - cy)) / den
        w2 = 1.0 - w0 - w1
        mask[i0:i1, j0:j1] |= (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
    return float(mask.mean()), float(span[0] * span[1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('scene_key', nargs='?', default='front3d_0015')
    ap.add_argument('--top', type=int, default=25)
    args = ap.parse_args()

    scene_file = SCENE_ROOT / args.scene_key / 'scene.xml'
    mesh = load_scene(scene_file)
    p0, p1, p2, nrm, nlen = triangle_geometry(mesh)
    area = 0.5 * nlen
    nh = nrm / nlen[:, None]
    total = area.sum()

    print(f'scene        : {mesh.scene_id}')
    print(f'scene file   : {scene_file}')
    print(f'architecture : {len(mesh.shapes)} shapes, {len(area)} triangles')
    print(f'total area   : {total:.2f} m2')
    mats = {}
    for m, a in zip(mesh.face_material, area):
        mats[m] = mats.get(m, 0.0) + float(a)
    for m, a in sorted(mats.items(), key=lambda kv: -kv[1]):
        print(f'   material {m:10s} {a:8.2f} m2 ({a / total * 100:5.2f}%)')

    bbox = mesh.verts.min(0), mesh.verts.max(0)
    print(f'bbox min     : {np.round(bbox[0], 3)}')
    print(f'bbox max     : {np.round(bbox[1], 3)}')

    vmask = np.abs(nh[:, 2]) < 0.05
    hmask = np.sqrt(nh[:, 0] ** 2 + nh[:, 1] ** 2) < 0.05
    print(f'area split   : vertical {area[vmask].sum():.2f} '
          f'({area[vmask].sum() / total * 100:.2f}%) | horizontal {area[hmask].sum():.2f} '
          f'({area[hmask].sum() / total * 100:.2f}%) | other {area[~vmask & ~hmask].sum():.2f} '
          f'({area[~vmask & ~hmask].sum() / total * 100:.3f}%)')

    # horizontal levels
    hi = np.where(hmask)[0]
    zc = (p0[hi, 2] + p1[hi, 2] + p2[hi, 2]) / 3.0
    print('\nhorizontal levels (z, area):')
    for z, a in cluster_levels(zc, area[hi]):
        print(f'   z={z:7.3f} m   area={a:8.2f} m2')

    # vertical planes
    vi = np.where(vmask)[0]
    groups = plane_groups(p0[vi], p1[vi], p2[vi], nh[vi], area[vi])
    rows = []
    for (az, off), ids in groups.items():
        ids = np.asarray(ids)
        pa = float(area[vi][ids].sum())
        if pa < 0.5:
            continue
        a_rad = np.radians(az)
        n2 = np.array([np.cos(a_rad), np.sin(a_rad)])
        t2 = np.array([-n2[1], n2[0]])
        g0, g1, g2 = p0[vi][ids], p1[vi][ids], p2[vi][ids]
        pts = np.concatenate([g0, g1, g2], 0)
        u = pts[:, :2] @ t2
        z = pts[:, 2]
        n = len(ids)
        uv = np.stack([np.concatenate([g0[:, :2] @ t2, g1[:, :2] @ t2, g2[:, :2] @ t2]),
                       np.concatenate([g0[:, 2], g1[:, 2], g2[:, 2]])], axis=1)
        faces = np.stack([np.arange(n), np.arange(n) + n, np.arange(n) + 2 * n], axis=1)
        cov, bbox_area = coverage(uv, faces, res=0.05)
        rows.append(dict(az=az, off=off / 100.0, area=pa, u0=float(u.min()), u1=float(u.max()),
                         z0=float(z.min()), z1=float(z.max()), cov=cov, bbox=bbox_area))
    rows.sort(key=lambda r: -r['area'])
    print(f'\nvertical planes >= 0.5 m2: {len(rows)}  (showing {min(args.top, len(rows))})')
    print('   az[deg]  offset[m]   area[m2]   u-extent[m]        z-extent[m]      cover')
    for r in rows[:args.top]:
        print(f'   {r["az"]:7.1f} {r["off"]:9.3f} {r["area"]:10.2f}   '
              f'{r["u0"]:7.2f}..{r["u1"]:7.2f}   {r["z0"]:6.2f}..{r["z1"]:6.2f}   {r["cov"]:5.2f}')

    low_cover = [r for r in rows if r['cov'] < 0.85]
    print(f'\nplanes with coverage < 0.85 (apertures / non-rectangular): '
          f'{len(low_cover)} / {len(rows)}')
    print(f'p10/p50 coverage: {np.quantile([r["cov"] for r in rows], 0.1):.2f} / '
          f'{np.quantile([r["cov"] for r in rows], 0.5):.2f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
