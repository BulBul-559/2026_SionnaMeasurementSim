"""Refine a generated .lay until PyLayers' own triangulation stops adding vertices.

PyLayers builds its transmission graph with `triangle.triangulate(C, 'pa')` and
then asserts that every triangulation vertex maps back to a Layout node
(`geomutil.setvnodes_new`). The area option can insert a Steiner vertex, which
trips that assertion. Upstream layouts (e.g. defstr.lay) never trigger it; ours
does, on one long wall run.

Fix on the converter side, without touching PyLayers: parse the layout with the
real Layout class (so the auto-generated AIR/_AIR constraints are included),
run `_triangle()`, and whenever extra vertices appear, split the containing wall
segment in the .lay at those points and retry.

Run (inside the PyLayers env):  python refine_layout.py --lay <path>
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import install_layout  # noqa: E402


def parse_lay(path: Path):
    head: list[str] = []
    pts: dict[int, tuple[float, float]] = {}
    segs: dict[int, dict] = {}
    tail: list[str] = []
    mode = None
    for line in path.read_text().splitlines():
        if line.startswith('['):
            mode = line.strip()
            if mode in ('[files]', '[slabs]', '[materials]', '[indoor]'):
                tail.append(line)
            continue
        if mode == '[points]' and '=' in line:
            k, v = line.split('=', 1)
            pts[int(k)] = tuple(ast.literal_eval(v.strip()))
        elif mode == '[segments]' and '=' in line:
            k, v = line.split('=', 1)
            segs[int(k)] = ast.literal_eval(v.strip())
        elif mode in ('[files]', '[slabs]', '[materials]', '[indoor]'):
            tail.append(line)
        else:
            if mode is None:
                head.append(line)
    return pts, segs, tail


def write_lay(path: Path, pts: dict, segs: dict, tail: list[str]) -> None:
    out = ['[info]', 'format = cart', 'version = 1.3', 'type = indoor', '', '[points]']
    for n in sorted(pts):
        out.append(f'{n} = ({pts[n][0]:.3f}, {pts[n][1]:.3f})')
    out += ['', '[segments]']
    for i, key in enumerate(sorted(segs), start=1):
        d = segs[key]
        out.append(f"{i} = {{'name': '{d['name']}', 'connect': {list(d['connect'])}, "
                   f"'z': {tuple(d['z'])}}}")
    out.append('')
    out.extend(tail)
    path.write_text('\n'.join(out) + '\n')


def triangulate_with_pylayers(lay: Path):
    from pylayers.gis.layout import Layout

    short = install_layout(lay)
    L = Layout(short)
    T, map_vertices = L._triangle()
    return len(T['vertices']), len(map_vertices), np.asarray(T['vertices'])[len(map_vertices):]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--lay', required=True)
    ap.add_argument('--max-iter', type=int, default=20)
    args = ap.parse_args()

    lay = Path(args.lay)
    for it in range(args.max_iter):
        n_tri_v, n_mapped, extra = triangulate_with_pylayers(lay)
        print(f'iter {it}: mapped nodes {n_mapped}, triangulation vertices {n_tri_v} '
              f'(+{len(extra)})')
        if len(extra) == 0:
            print('clean: PyLayers needs no extra vertices')
            return 0
        pts, segs, tail = parse_lay(lay)
        changed = 0
        for p in extra:
            hit = None
            for sid, s in segs.items():
                a = np.array(pts[s['connect'][0]], float)
                b = np.array(pts[s['connect'][1]], float)
                ab = b - a
                ln2 = float(ab @ ab)
                if ln2 < 1e-12:
                    continue
                t = float((p - a) @ ab) / ln2
                if not (1e-4 < t < 1 - 1e-4):
                    continue
                if float(np.linalg.norm(a + t * ab - p)) <= 0.01:
                    hit = sid
                    break
            if hit is None:
                print('  WARNING extra vertex not on any wall segment:', np.round(p, 3).tolist())
                continue
            s = segs.pop(hit)
            nid = min(pts) - 1
            pts[nid] = (round(float(p[0]), 3), round(float(p[1]), 3))
            a, b = s['connect']
            nxt = max(segs) + 1 if segs else 1
            segs[nxt] = {'name': s['name'], 'connect': [a, nid], 'z': tuple(s['z'])}
            segs[nxt + 1] = {'name': s['name'], 'connect': [nid, b], 'z': tuple(s['z'])}
            changed += 1
        if changed == 0:
            print('  could not refine further')
            return 1
        write_lay(lay, pts, segs, tail)
        print(f'  split {changed} segment(s)')
    print('reached max iterations')
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
