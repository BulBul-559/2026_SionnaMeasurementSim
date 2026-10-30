#!/usr/bin/env python
"""Compare a DiffeRT run against a reference Sionna dataset of the *same* contract.

The 2988 dataset stores one row per (bs, ue) pair with ``bs_pos``/``ue_pos``, so links are
matched by position rather than by row order. That matters because the DiffeRT run only
covers one BS and a handful of UEs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from differt_crosscheck.config import ensure_output_dirs, get_profile  # noqa: E402
from differt_crosscheck.metrics import score_pair  # noqa: E402

TOL_M = 1e-2


def _match_rows(ref: dict, bs_center: np.ndarray, ue_positions: np.ndarray) -> list[int]:
    rows = []
    for ue in ue_positions:
        d_bs = np.linalg.norm(ref["bs_pos"] - bs_center[None, :], axis=1)
        d_ue = np.linalg.norm(ref["ue_pos"] - ue[None, :], axis=1)
        cand = np.where((d_bs < TOL_M) & (d_ue < TOL_M))[0]
        rows.append(int(cand[0]) if cand.size else -1)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True, help="e.g. datasets/Front3d_2988_CFR_Only/scenes")
    parser.add_argument("--profile", default="2988")
    args = parser.parse_args()

    import h5py

    profile = get_profile(args.profile)
    paths = ensure_output_dirs(args.output_root)
    selection = json.loads(Path(args.selection).read_text())

    rows_out: list[dict] = []
    per_scene: list[dict] = []

    for scene in selection["scenes"]:
        key = scene["scene_key"]
        ours_path = paths["scenes"] / f"{key}.h5"
        ref_path = Path(args.reference_dir) / f"{key}.h5"
        if not ours_path.is_file() or not ref_path.is_file():
            print(f"[skip] {key}: missing {ours_path.name if not ours_path.is_file() else ref_path}")
            continue

        with h5py.File(ref_path, "r") as f:
            ref = {
                "bs_pos": np.asarray(f["bs_pos"][()], dtype=float),
                "ue_pos": np.asarray(f["ue_pos"][()], dtype=float),
                "cfr": f["cfr"],
            }
            ref_rows = _match_rows(ref, np.asarray(scene["bs_center"], dtype=float),
                                   np.asarray(scene["ue_positions"], dtype=float))
            if any(r < 0 for r in ref_rows):
                print(f"[warn] {key}: {sum(r < 0 for r in ref_rows)} links not found in the reference")
            valid = [(i, r) for i, r in enumerate(ref_rows) if r >= 0]
            ref_cfr = np.stack([np.asarray(ref["cfr"][r]) for _, r in valid]) if valid else None

        with h5py.File(ours_path, "r") as f:
            our_cfr = np.asarray(f["cfr"][()])

        scores = []
        for local, (our_index, ref_row) in enumerate(valid):
            ref_link = np.asarray(ref_cfr[local], dtype=np.complex128)[:, :, :]
            our_link = np.asarray(our_cfr[our_index], dtype=np.complex128)
            score = score_pair(
                ref_link,
                our_link,
                bandwidth_hz=profile.grid.bandwidth_hz,
                num_subcarriers=profile.grid.num_subcarriers,
            )
            score.update(
                {
                    "scene_key": key,
                    "link_index": int(our_index),
                    "reference_row": int(ref_row),
                    "power_ref_db": score["power_ref_db"],
                    "power_dut_db": score["power_dut_db"],
                    "power_offset_db": score["power_offset_db"],
                }
            )
            rows_out.append(score)
            scores.append(score)

        if scores:
            per_scene.append(
                {
                    "scene_key": key,
                    "num_links": len(scores),
                    **{
                        f"mean_{m}": float(np.mean([s[m] for s in scores]))
                        for m in ("cos_sim", "nmse_complex_db", "mag_corr", "pdp_corr", "power_offset_db")
                    },
                }
            )
            print(
                f"[{key}] links={len(scores)} cos_sim={np.mean([s['cos_sim'] for s in scores]):.3f} "
                f"power_offset={np.mean([s['power_offset_db'] for s in scores]):+.1f} dB "
                f"pdp_corr={np.mean([s['pdp_corr'] for s in scores]):.3f}",
                flush=True,
            )

    if rows_out:
        csv_path = paths["compare"] / "metrics_per_link.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows_out[0].keys()))
            writer.writeheader()
            writer.writerows(rows_out)
        summary = {
            "contract": profile.name,
            "reference_dir": str(args.reference_dir),
            "num_links": len(rows_out),
            "overall": {
                m: float(np.mean([r[m] for r in rows_out]))
                for m in ("cos_sim", "nmse_complex_db", "mag_corr", "pdp_corr", "power_offset_db")
            },
            "per_scene": per_scene,
        }
        (paths["compare"] / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"overall: {json.dumps(summary['overall'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
