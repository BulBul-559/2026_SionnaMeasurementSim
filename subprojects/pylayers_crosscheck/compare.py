"""Compare PyLayers vs Sionna CFRs on the same links.

All metrics are computed by the same code from the two 768-point CFRs, so no
engine-specific post-processing can hide a difference:

  - resolvable path count      : local maxima in the CFR-derived PDP (-20 dB)
  - first path excess delay    : first tap above -15 dB, minus d_3d/c
  - PDP shape correlation      : Pearson correlation of the normalised PDPs
  - CFR amplitude correlation  : Pearson correlation of |H(f)|
  - complex coherence          : |<h_a,h_b>| / (||h_a|| ||h_b||)
  - RMS delay spread           : from the same thresholded PDP
  - band-mean path gain        : 20log10 rms|H| (absolute level check)

Run:  python compare.py --scene front3d_0015
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import BANDWIDTH_HZ, NUM_SUBCARRIERS, SUBROOT  # noqa: E402

C0 = 299792458.0
DT_S = 1.0 / BANDWIDTH_HZ  # 43.4 ns tap spacing


def pdp_from_cfr(h: np.ndarray) -> np.ndarray:
    h = np.asarray(h).ravel()
    return np.abs(np.fft.ifft(h)) ** 2 / len(h)


def first_tap(pdp: np.ndarray, rel_db: float = -15.0) -> int:
    thr = pdp.max() * 10 ** (rel_db / 10.0)
    idx = np.flatnonzero(pdp >= thr)
    return int(idx[0]) if idx.size else 0


def count_peaks(pdp: np.ndarray, rel_db: float = -20.0) -> int:
    thr = pdp.max() * 10 ** (rel_db / 10.0)
    mask = pdp >= thr
    n = 0
    for k in range(1, len(pdp) - 1):
        if mask[k] and pdp[k] >= pdp[k - 1] and pdp[k] > pdp[k + 1]:
            n += 1
    return n


def rms_delay_spread(pdp: np.ndarray, rel_db: float = -15.0) -> float:
    thr = pdp.max() * 10 ** (rel_db / 10.0)
    idx = np.flatnonzero(pdp >= thr)
    if idx.size == 0:
        return 0.0
    tau = idx * DT_S
    p = pdp[idx]
    mean_t = float(np.sum(p * tau) / np.sum(p))
    return float(np.sqrt(max(0.0, np.sum(p * tau ** 2) / np.sum(p) - mean_t ** 2)))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, float).ravel()
    b = np.asarray(b, float).ravel()
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if a.std() < 1e-15 or b.std() < 1e-15:
        return float('nan')
    return float(np.corrcoef(a, b)[0, 1])


def load_sionna(h5_path: Path, bs_id: int, ue_id: int):
    with h5py.File(h5_path, 'r') as f:
        bs = f['bs_id'][:]
        ue = f['ue_id'][:]
        sel = np.flatnonzero((bs == bs_id) & (ue == ue_id))
        if sel.size == 0:
            return None
        cfr = np.asarray(f['cfr'][sel[0]]).reshape(-1)
        return cfr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='front3d_0015')
    ap.add_argument('--pylayers-dir', default=None)
    ap.add_argument('--sionna-h5', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    if args.pylayers_dir is None:
        for cand in (SUBROOT / 'outputs' / 'pylayers' / args.scene,):
            if cand.exists():
                args.pylayers_dir = cand
    py_dir = Path(args.pylayers_dir)
    sionna_h5 = Path(args.sionna_h5) if args.sionna_h5 else (
        SUBROOT / 'outputs' / 'sionna_1x1' / 'dataset' / 'scenes' / f'{args.scene}.h5')

    rows = []
    for npz_path in sorted(py_dir.glob('*.npz')):
        d = np.load(npz_path, allow_pickle=True)
        link = json.loads(str(d['link']))
        variant = str(d['variant'])
        h_py = np.asarray(d['cfr']).reshape(-1)
        h_si = load_sionna(sionna_h5, link['bs_id'], link['ue_id'])
        if h_si is None:
            continue
        dist = float(np.linalg.norm(np.asarray(link['bs_pos']) - np.asarray(link['ue_pos'])))
        pdp_py, pdp_si = pdp_from_cfr(h_py), pdp_from_cfr(h_si)
        los_delay = dist / C0
        t_py = first_tap(pdp_py) * DT_S
        t_si = first_tap(pdp_si) * DT_S
        rows.append(dict(
            link=f"w{link['walls']}_bs{link['bs_id']}_ue{link['ue_id']}",
            walls=int(link['walls']),
            dist_3d_m=round(dist, 3),
            variant=variant,
            pylayers_nray=int(d['nray']),
            sionna_paths_resolvable=count_peaks(pdp_si),
            pylayers_paths_resolvable=count_peaks(pdp_py),
            first_delay_pylayers_ns=round((t_py - los_delay) * 1e9, 2),
            first_delay_sionna_ns=round((t_si - los_delay) * 1e9, 2),
            first_delay_diff_ns=round((t_py - t_si) * 1e9, 2),
            rms_ds_pylayers_ns=round(rms_delay_spread(pdp_py) * 1e9, 2),
            rms_ds_sionna_ns=round(rms_delay_spread(pdp_si) * 1e9, 2),
            pdp_corr=round(corr(pdp_py / pdp_py.max(), pdp_si / pdp_si.max()), 4),
            cfr_amp_corr=round(corr(np.abs(h_py), np.abs(h_si)), 4),
            complex_coherence=round(abs(np.vdot(h_py, h_si)) /
                                    (np.linalg.norm(h_py) * np.linalg.norm(h_si)), 4),
            gain_pylayers_db=round(20 * np.log10(np.sqrt(np.mean(np.abs(h_py) ** 2)) + 1e-30), 2),
            gain_sionna_db=round(20 * np.log10(np.sqrt(np.mean(np.abs(h_si) ** 2)) + 1e-30), 2),
            gain_diff_db=round(20 * np.log10((np.sqrt(np.mean(np.abs(h_py) ** 2)) + 1e-30) /
                                             (np.sqrt(np.mean(np.abs(h_si) ** 2)) + 1e-30)), 2),
        ))

    rows.sort(key=lambda r: (r['variant'], r['walls'], r['link']))
    out = Path(args.out) if args.out else SUBROOT / 'outputs' / 'reports' / f'{args.scene}_metrics.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(
        scene=args.scene,
        pylayers_dir=str(py_dir),
        sionna_h5=str(sionna_h5),
        note='metrics computed identically from both 768-point CFRs; delay tap = %.1f ns'
             % (DT_S * 1e9),
        rows=rows), indent=1))

    hdr = ('link', 'variant', 'walls', 'dist', 'nray_py', 'paths_py', 'paths_si',
           'dFirst_py', 'dFirst_si', 'diff', 'rms_py', 'rms_si', 'pdp_r', 'amp_r', 'coh', 'dGain')
    print('  '.join(f'{h:>10s}' for h in hdr))
    for r in rows:
        print(f"{r['link']:>10s}  {r['variant']:>10s}  {r['walls']:>6d}  {r['dist_3d_m']:>7.2f}  "
              f"{r['pylayers_nray']:>7d}  {r['pylayers_paths_resolvable']:>8d}  "
              f"{r['sionna_paths_resolvable']:>8d}  {r['first_delay_pylayers_ns']:>10.2f}  "
              f"{r['first_delay_sionna_ns']:>10.2f}  {r['first_delay_diff_ns']:>6.2f}  "
              f"{r['rms_ds_pylayers_ns']:>6.2f}  {r['rms_ds_sionna_ns']:>6.2f}  "
              f"{r['pdp_corr']:>5.2f}  {r['cfr_amp_corr']:>5.2f}  {r['complex_coherence']:>4.2f}  "
              f"{r['gain_diff_db']:>6.2f}")
    print('saved', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
