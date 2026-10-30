#!/usr/bin/env python
"""Aggregate a per-scene DiffeRT run into one link index and compare with the reference.

Outputs (under ``--output-root``)::

    dataset/all_links_index.csv    one row per manifest link: identity, power, is_zero,
                                   and the similarity metrics against the reference
    dataset/aggregate_summary.json counts, means, and the zero-link breakdown
    compare/zero_links.csv         the links that came out empty
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

from differt_crosscheck.config import get_profile  # noqa: E402
from differt_crosscheck.metrics import score_pair  # noqa: E402

TOL_M = 1e-2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--profile", default="2988")
    args = parser.parse_args()

    import h5py

    profile = get_profile(args.profile)
    root = args.output_root
    scenes = json.loads((root / "scenes.json").read_text())
    rows = list(csv.DictReader(args.manifest.open()))

    by_scene: dict[str, list[dict]] = {}
    for row in rows:
        by_scene.setdefault(row["scene_id"], []).append(row)

    out_rows: list[dict] = []
    zero_rows: list[dict] = []
    missing: list[str] = []
    unreadable: list[str] = []

    for entry in scenes:
        scene = entry["scene_id"]
        h5_ours = root / "work" / scene / "dataset" / "scenes" / f"{scene}.h5"
        h5_ref = args.reference_dir / f"{scene}.h5"
        if not h5_ours.is_file():
            missing.append(scene)
            continue

        scene_rows = by_scene[scene]
        try:
            four = h5py.File(h5_ours, "r")
            fref = h5py.File(h5_ref, "r")
        except (OSError, RuntimeError) as exc:
            print(f"[skip] {scene}: unreadable ({type(exc).__name__})", flush=True)
            unreadable.append(scene)
            continue

        with four, fref:
            ours = np.asarray(four["cfr"][()])
            ref_cfr = fref["cfr"]
            ref_bs = np.asarray(fref["bs_pos"][()], dtype=float)
            ref_ue = np.asarray(fref["ue_pos"][()], dtype=float)

            for i, row in enumerate(scene_rows):
                if i >= ours.shape[0]:
                    break
                h = ours[i]
                power = float(np.mean(np.abs(h) ** 2))
                is_zero = bool(power <= 1e-28)

                bs = np.array([float(row["bs_x_m"]), float(row["bs_y_m"]), float(row["bs_z_m"])])
                ue = np.array([float(row["ue_x_m"]), float(row["ue_y_m"]), float(row["ue_z_m"])])
                d_bs = np.linalg.norm(ref_bs - bs[None, :], axis=1)
                d_ue = np.linalg.norm(ref_ue - ue[None, :], axis=1)
                cand = np.where((d_bs < TOL_M) & (d_ue < TOL_M))[0]
                ref_row = int(cand[0]) if cand.size else -1

                metrics: dict[str, float] = {}
                if ref_row >= 0 and not is_zero:
                    metrics = score_pair(
                        np.asarray(ref_cfr[ref_row], dtype=np.complex128),
                        np.asarray(h, dtype=np.complex128),
                        bandwidth_hz=profile.grid.bandwidth_hz,
                        num_subcarriers=profile.grid.num_subcarriers,
                    )

                record = {
                    "scene_id": scene,
                    "scene_class": row["scene_class"],
                    "global_link_id": row["global_link_id"],
                    "bs_id": row["bs_id"],
                    "ue_id": row["ue_id"],
                    "power_dut_db": float(10 * np.log10(power + 1e-30)),
                    "power_ref_db": float(row["csi_power_db"]),
                    "is_zero": int(is_zero),
                    "reference_row": ref_row,
                    "cos_sim": metrics.get("cos_sim", float("nan")),
                    "nmse_complex_db": metrics.get("nmse_complex_db", float("nan")),
                    "mag_corr": metrics.get("mag_corr", float("nan")),
                    "pdp_corr": metrics.get("pdp_corr", float("nan")),
                    "power_offset_db": metrics.get("power_offset_db", float("nan")),
                }
                out_rows.append(record)
                if is_zero:
                    zero_rows.append(record)

    ds = root / "dataset"
    cmp_dir = root / "compare"
    ds.mkdir(exist_ok=True)
    cmp_dir.mkdir(exist_ok=True)

    with (ds / "all_links_index.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    if zero_rows:
        with (cmp_dir / "zero_links.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(zero_rows[0].keys()))
            writer.writeheader()
            writer.writerows(zero_rows)

    valid = [r for r in out_rows if not r["is_zero"]]
    matched = [r for r in valid if r["reference_row"] >= 0]

    classes: dict[str, dict[str, int]] = {}
    for r in out_rows:
        c = classes.setdefault(r["scene_class"], {"total": 0, "zero": 0})
        c["total"] += 1
        c["zero"] += r["is_zero"]

    summary = {
        "contract": profile.name,
        "links_total": len(out_rows),
        "links_valid": len(valid),
        "links_zero": len(zero_rows),
        "links_matched_reference": len(matched),
        "scenes_total": len(scenes),
        "scenes_missing": missing,
        "scenes_unreadable": unreadable,
        "similarity_on_valid": {
            key: (float(np.nanmean([r[key] for r in matched])) if matched else None)
            for key in ("cos_sim", "nmse_complex_db", "mag_corr", "pdp_corr", "power_offset_db")
        },
        "zero_by_class": classes,
    }
    (ds / "aggregate_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps({k: summary[k] for k in ("links_total", "links_valid", "links_zero")}, indent=2))
    print("similarity on valid links:", summary["similarity_on_valid"])
    print("zero by class:", classes)
    if missing:
        print(f"missing scenes: {len(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
