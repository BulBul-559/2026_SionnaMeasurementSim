"""Make a generated .lay acceptable to `Layout.buildGt`.

PyLayers' transmission-graph builder is strict: every segment must take part in
at least two cycles, otherwise it raises

    AssertionError: Some segments are connected to LESS than 2 cycles [...]

Auto-converted Front3D geometry can still violate that after the converter's own
planar cleanup (the two sheets of a thick wall can enclose a face so thin that it
degenerates). This tool is the last mile: it runs the real `Layout.build()`,
reads the offending segment ids out of PyLayers' own message, removes the
corresponding .lay segments, and retries until the build succeeds.

Every removal is logged, so the final report can state exactly how much geometry
had to be trimmed.

Run (inside the PyLayers env):  python fit_pylayers.py --lay <path>
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import install_layout, pylayers_project_dir  # noqa: E402

ID_RE = re.compile(r'\[([0-9\s]+)\]')


def parse_lay(path: Path):
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


def try_build(lay: Path):
    from pylayers.gis.layout import Layout

    short = install_layout(lay)
    L = Layout(short)
    try:
        L.build()
        return True, L, None
    except Exception as e:  # noqa: BLE001
        return False, L, e


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--lay', required=True)
    ap.add_argument('--max-iter', type=int, default=12)
    ap.add_argument('--log', default=None)
    args = ap.parse_args()

    lay = Path(args.lay)
    log_path = Path(args.log) if args.log else lay.with_suffix('.fit.log')
    log: list[str] = []

    for it in range(args.max_iter):
        ok, L, err = try_build(lay)
        if ok:
            msg = (f'iter {it}: BUILD OK  points={len(L.Gs.pos)} segments={L.Ns} '
                   f'Gi={len(L.Gi.nodes())} cycles=n/a')
            print(msg)
            log.append(msg)
            log_path.write_text('\n'.join(log))
            return 0
        text = str(err)
        ids = []
        m = ID_RE.search(text)
        if m:
            ids = [int(x) for x in m.group(1).split()]
        msg = f'iter {it}: {type(err).__name__}: {text.splitlines()[0][:100]} ids={ids[:12]}'
        print(msg)
        log.append(msg)
        if not ids:
            log_path.write_text('\n'.join(log))
            print('cannot interpret failure, stopping')
            return 1

        pts, segs, tail = parse_lay(lay)
        coords_of = {}
        for sid in ids:
            if sid in L.Gs.node:
                conn = L.Gs.node[sid].get('connect')
                if conn:
                    a, b = conn
                    if a in L.Gs.pos and b in L.Gs.pos:
                        coords_of[sid] = (np.array(L.Gs.pos[a]), np.array(L.Gs.pos[b]))
        removed = []
        for sid, (pa, pb) in coords_of.items():
            best, best_d = None, 1e9
            for key, s in segs.items():
                qa = np.array(pts[s['connect'][0]], float)
                qb = np.array(pts[s['connect'][1]], float)
                d = (min(np.linalg.norm(qa - pa) + np.linalg.norm(qb - pb),
                         np.linalg.norm(qa - pb) + np.linalg.norm(qb - pa)))
                if d < best_d:
                    best, best_d = key, d
            if best is not None and best_d < 0.02:
                removed.append((best, segs[best]))
                del segs[best]
        if not removed:
            log_path.write_text('\n'.join(log))
            print('no matching .lay segment for the offending ids, stopping')
            return 1
        for key, d in removed:
            log.append(f'   removed lay segment {key} {d}')
            print(f'   removed {key} {d}')
        write_lay(lay, pts, segs, tail)
        log_path.write_text('\n'.join(log))

    print('max iterations reached')
    log_path.write_text('\n'.join(log))
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
