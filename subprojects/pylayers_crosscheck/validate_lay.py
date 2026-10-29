"""Validate a generated .lay inside PyLayers: parse, build, and report topology.

Usage:  python validate_lay.py --lay <path> [--link 0]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import install_layout, pylayers_project_dir  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--lay', required=True)
    ap.add_argument('--build', action='store_true')
    args = ap.parse_args()

    from pylayers.gis.layout import Layout

    lay = Path(args.lay)
    short = install_layout(lay)
    print(f'installed layout as {short} in {pylayers_project_dir()}')
    t0 = time.time()
    L = Layout(short)
    print(f'parsed in {time.time() - t0:.1f}s')
    print(f'  typ      : {L.typ}')
    print(f'  zfloor   : {L.zfloor}')
    print(f'  zceil    : {L.zceil}')
    print(f'  points   : {len(L.Gs.pos)}')
    print(f'  segments : {L.Ns}')
    for name, ids in sorted(L.name.items()):
        if ids:
            print(f'    slab {name:8s}: {len(ids)} segments')

    if not args.build:
        return 0

    t0 = time.time()
    L.build()
    dt = time.time() - t0
    print(f'built in {dt:.1f}s')
    for gname in ('Gs', 'Gt', 'Gi', 'Gv', 'Gr'):
        g = getattr(L, gname, None)
        if g is not None:
            print(f'  {gname}: {len(g.nodes())} nodes, {len(g.edges())} edges')
    try:
        cy = L.cycles()
        print(f'  cycles  : {len(cy)}')
    except Exception as e:  # noqa: BLE001
        print('  cycles failed:', repr(e)[:120])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
