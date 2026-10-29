"""Debug helper: run PyLayers' constrained triangulation and explain failures.

Compares the triangulation's vertices with the layout's own nodes to find which
polygon vertices do not map back to a Layout point (the condition that trips
`geomutil.setvnodes_new`'s assertion).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/sunmeiyuan/projects/SionnaMeasurementSim/subprojects/pylayers_crosscheck')

from common import install_layout  # noqa: E402


def main() -> int:
    from pylayers.gis.layout import Layout
    from pylayers.util import geomutil as geu

    lay = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        '/home/sunmeiyuan/projects/SionnaMeasurementSim/subprojects/pylayers_crosscheck'
        '/outputs/layouts/front3d_0015.lay')
    short = install_layout(lay)
    L = Layout(short)
    print(f'layout: {L.Ns} segments, {len(L.Gs.pos)} nodes')

    T, map_vertices = L._triangle()
    n_tri_vertices = len(T['vertices'])
    print(f'triangulation vertices: {n_tri_vertices}, layout mapped nodes: {len(map_vertices)}')
    if n_tri_vertices != len(map_vertices):
        print('  -> Triangle inserted extra vertices (Steiner points)')
        extra = np.asarray(T['vertices'])[len(map_vertices):]
        print('  extra count:', len(extra))
        d = []
        for p in extra:
            d.append(min(np.linalg.norm(np.array(q) - p) for q in L.Gs.pos.values()))
        print('  extra vertex distance to nearest layout node: '
              f'min={min(d):.3f} m, max={max(d):.3f} m')
        print('  sample extra:', np.round(extra[:5], 3).tolist())

    tri = T['triangles']
    print(f'triangles: {len(tri)}')
    allv = np.asarray(T['vertices'])
    node_xy = np.array(list(L.Gs.pos.values()), float)
    dmin = np.array([float(np.min(np.linalg.norm(node_xy - p, axis=1))) for p in allv])
    print(f'vertex distance to nearest layout node: min={dmin.min():.3f} '
          f'max={dmin.max():.3f} m, >50mm: {int((dmin > 0.05).sum())} / {len(dmin)}')
    for i in np.argsort(-dmin)[:5]:
        print(f'   vertex {i} {np.round(allv[i], 3).tolist()} d={dmin[i]:.3f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
