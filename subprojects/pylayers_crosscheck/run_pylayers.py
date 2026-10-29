"""Run PyLayers DLink for the selected links and dump 768-point CFRs.

Only the propagation side is used here: both ends are isotropic omni antennas so
the comparison isolates the ray-tracing engine (array/pattern conventions are a
separate question and are not mixed into this first check).

Two variants are produced per link:
  walls      : 2D only (ra_ceil_H=-1), no floor/ceiling reflection
  walls_cf   : walls + one floor/ceiling mirror pair (ra_ceil_H=<zceil>)

Diffraction is disabled because Sionna RT does not model diffraction at all.

Run:  python run_pylayers.py --scene front3d_0015
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import SUBROOT, frequency_grid_ghz  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='front3d_0015')
    ap.add_argument('--lay', default=None)
    ap.add_argument('--links', default=None)
    ap.add_argument('--out', default=None)
    ap.add_argument('--cutoff', type=int, default=3,
                    help='max 2D wall interactions (Sionna max_depth=4 total)')
    ap.add_argument('--n-mirror-cf', type=int, default=1)
    ap.add_argument('--variants', default='walls,walls_cf')
    args = ap.parse_args()

    from pylayers.antprop.antenna import Antenna
    from pylayers.gis.layout import Layout
    from pylayers.simul.link import DLink

    lay_path = Path(args.lay) if args.lay else SUBROOT / 'outputs' / 'layouts' / f'{args.scene}.lay'
    links_path = Path(args.links) if args.links else (
        SUBROOT / 'outputs' / 'links' / f'{args.scene}_links.json')
    out_dir = Path(args.out) if args.out else SUBROOT / 'outputs' / 'pylayers' / args.scene
    out_dir.mkdir(parents=True, exist_ok=True)

    links = json.loads(links_path.read_text())['links']
    fGHz = frequency_grid_ghz()

    t0 = time.time()
    L = Layout(str(lay_path))
    print(f'layout loaded in {time.time() - t0:.1f}s: {L.Ns} segments, '
          f'{len(L.Gs.pos)} points', flush=True)
    t0 = time.time()
    L.build()
    print(f'layout built in {time.time() - t0:.1f}s: Gi nodes {len(L.Gi.nodes())}', flush=True)

    summary = []
    for link in links:
        tag = f"w{link['walls']}_bs{link['bs_id']}_ue{link['ue_id']}"
        for variant in args.variants.split(','):
            if variant == 'walls':
                ceil_h, nmir = -1, 0
            elif variant == 'walls_cf':
                ceil_h, nmir = [], args.n_mirror_cf
            else:
                raise SystemExit(f'unknown variant {variant}')
            t0 = time.time()
            DL = DLink(L=L, fGHz=fGHz, cutoff=args.cutoff)
            DL.a = np.asarray(link['ue_pos'], float)
            DL.b = np.asarray(link['bs_pos'], float)
            DL.Aa = Antenna(typ='Omni', fGHz=fGHz)
            DL.Ab = Antenna(typ='Omni', fGHz=fGHz)
            DL.eval(cutoff=args.cutoff, diffraction=False, applywav=False,
                    si_progress=False, ra_ceil_H=ceil_h, ra_number_mirror_cf=nmir)
            elapsed = time.time() - t0

            tau_ns = np.asarray(DL.H.taud, float).ravel()
            y = np.asarray(DL.H.y, float)
            # composite channel transfer function over all rays
            h = np.sum(y * np.exp(-2j * np.pi * tau_ns[None, None, None, :]
                                  * fGHz[None, None, None, :]), axis=0) \
                if y.ndim == 4 else None
            if h is None:
                h = np.sum(y * np.exp(-2j * np.pi * tau_ns[:, None] * fGHz[None, :]), axis=0)
            cfr = np.asarray(h).reshape(-1, len(fGHz))[0]

            # per-ray data: delay (ns) and band-averaged power
            ray_power = np.mean(np.abs(y.reshape(len(tau_ns), -1) if y.ndim > 2 else y) ** 2, axis=1)

            np.savez_compressed(
                out_dir / f'{tag}__{variant}.npz',
                cfr=cfr.astype(np.complex128),
                fghz=fGHz,
                tau_ns=tau_ns,
                ray_power=ray_power,
                nray=len(tau_ns),
                elapsed_s=elapsed,
                link=json.dumps(link),
                variant=variant,
            )
            summary.append(dict(link=tag, variant=variant, nray=int(len(tau_ns)),
                                elapsed_s=round(elapsed, 2),
                                tau_min_ns=float(tau_ns.min()), tau_max_ns=float(tau_ns.max()),
                                cfr_dbfs=float(20 * np.log10(np.sqrt(np.mean(np.abs(cfr) ** 2)) + 1e-30))))
            print(f'{tag:24s} {variant:9s} nray={len(tau_ns):4d} '
                  f'tau={tau_ns.min():7.2f}..{tau_ns.max():7.2f} ns '
                  f'rms={20 * np.log10(np.sqrt(np.mean(np.abs(cfr) ** 2)) + 1e-30):7.2f} dB '
                  f'({elapsed:.1f}s)', flush=True)

    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=1))
    print('saved to', out_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
