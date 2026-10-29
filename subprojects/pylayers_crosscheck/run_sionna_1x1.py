"""Produce a single-antenna (1x1) Sionna baseline for the cross-check links.

Why re-run instead of reusing the 339-scene dataset: that dataset is 4x8 at the
BS, so its CFR mixes array geometry/orientation conventions with propagation.
For an engine-vs-engine comparison both sides must be single isotropic antennas.

Nothing in `sionna_measurement_sim` is modified; this only writes a config
copy and a one-row index inside the subproject and calls the existing CLI.

Run:  python run_sionna_1x1.py --scene front3d_0015 [--execute]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import SUBROOT, WORKSPACE  # noqa: E402

BASE_CONFIG = WORKSPACE / 'outputs' / 'front3d_339_dichasus_like_cfr_v1' / 'full_config.yaml'
BASE_INDEX = WORKSPACE / 'outputs' / 'front3d_339_dichasus_like_cfr_v1' / 'prepared' / 'test_scene_index.jsonl'
PYTHON = WORKSPACE / '.venv' / 'bin' / 'python'


def build(scene: str, gpus: list[int], workers: int) -> tuple[Path, Path, Path, Path]:
    root = SUBROOT / 'outputs' / 'sionna_1x1'
    root.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(x) for x in BASE_INDEX.read_text().splitlines()]
    row = next(r for r in rows if r['scene_key'] == scene)

    cfg = yaml.safe_load(BASE_CONFIG.read_text())
    cfg['runtime']['device'] = 'cuda'
    cfg['runtime']['require_gpu'] = True
    rt = cfg['runtime']
    if 'gpu' in rt and isinstance(rt['gpu'], dict):
        rt['gpu']['ids'] = gpus
    cfg.setdefault('runtime', {})['gpu_ids'] = gpus
    cfg['runtime']['parallel_workers'] = workers
    sched = cfg['runtime'].get('gpu_scheduler')
    if isinstance(sched, dict):
        sched['free_memory_threshold'] = 0.0
        sched['scan_interval_s'] = 2.0
    cfg['antenna']['bs_array']['num_rows'] = 1
    cfg['antenna']['bs_array']['num_cols'] = 1
    cfg['antenna']['ue_array']['num_rows'] = 1
    cfg['antenna']['ue_array']['num_cols'] = 1
    cfg['rt']['diffuse_reflection'] = False
    cfg['rt']['max_depth'] = 4
    cfg['output']['root_dir'] = str(root / 'runs')
    cfg['output']['products'] = ['cfr_truth']
    cfg['input']['scene_file'] = row['scene_file']
    cfg['input']['scene_id'] = row['scene_key']
    cfg['input']['label_file'] = row['label_file']

    cfg_path = root / f'{scene}_1x1_config.yaml'
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

    idx_path = root / f'{scene}_1x1_index.jsonl'
    idx_path.write_text(json.dumps(row) + '\n')
    return root, cfg_path, idx_path, row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default='front3d_0015')
    ap.add_argument('--gpus', default='2,3')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--execute', action='store_true')
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(',') if g.strip()]
    root, cfg_path, idx_path, row = build(args.scene, gpus, args.workers)
    print('config :', cfg_path)
    print('index  :', idx_path)
    print('scene  :', row['scene_file'])
    print('label  :', row['label_file'])
    print('gpus   :', gpus, 'workers', args.workers)

    if not args.execute:
        print('\n(dry run; pass --execute to launch the CLI)')
        return 0

    raw = root / 'raw_runs'
    ds = root / 'dataset'
    cmd = [str(PYTHON), '-u', '-m', 'sionna_measurement_sim.app.cli',
           '--config', str(cfg_path), 'run-scene-index',
           '--index-file', str(idx_path), '--output-root', str(raw)]
    print('\n$', ' '.join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=str(WORKSPACE))
    print('run-scene-index rc =', rc, flush=True)
    if rc != 0:
        return rc

    cmd2 = [str(PYTHON), '-u', '-m', 'sionna_measurement_sim.app.cli',
            'postprocess-cfr-scenes', '--input-root', str(raw), '--output-root', str(ds),
            '--workers', str(args.workers), '--cfr-compression', 'none']
    print('\n$', ' '.join(cmd2), flush=True)
    rc = subprocess.call(cmd2, cwd=str(WORKSPACE))
    print('postprocess rc =', rc, flush=True)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
