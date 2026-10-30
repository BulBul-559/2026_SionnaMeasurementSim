#!/usr/bin/env python
"""Merge the two DiffeRT passes into one complete 7216-link dataset.

Inputs
------
* L1: ``contract2988_order1_7216_manifest`` -- order-1 CFR for every link of the cohort
  (3,622 non-zero, the rest all zero),
* L2: ``contract2988_rescue_r1`` -- order-2 transmission CFR for the links L1 left empty
  (1,728 recovered).

Output
------
A standalone, Sionna-style dataset::

    <root>/scenes/<scene>.h5     cfr [links,16,2,768] + per-link provenance arrays
    <root>/link_index.csv        7,216 rows: identity, source, power, validity
    <root>/manifest.json         contract, counts, provenance of both inputs
    <root>/README.md             how to read it

Per-link provenance is explicit so downstream code can decide whether to use the whole
set or only L1 (order-1) links:

    source = 0  nothing (still zero)
    source = 1  L1 order-1
    source = 2  L2 order-2 transmission rescue
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

FREQ_KEYS = ("center_frequency_hz", "bandwidth_hz", "num_subcarriers", "subcarrier_spacing_hz")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l1-root", type=Path, required=True)
    parser.add_argument("--l2-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import h5py

    scenes_dir = args.out / "scenes"
    scenes_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(args.manifest.open()))
    by_scene: dict[str, list[dict]] = {}
    for r in rows:
        by_scene.setdefault(r["scene_id"], []).append(r)

    index_rows = []
    totals = {"links": 0, "l1": 0, "l2": 0, "zero": 0}
    grid = None

    for scene, scene_rows in sorted(by_scene.items()):
        l1_path = args.l1_root / "work" / scene / "dataset" / "scenes" / f"{scene}.h5"
        l2_path = args.l2_root / "scenes" / f"{scene}.h5"
        status_path = args.l2_root / "status" / f"{scene}.csv"

        with h5py.File(l1_path, "r") as f:
            cfr = np.asarray(f["cfr"][()])                       # (n,16,2,768)
            if grid is None:
                grid = {k: float(f[f"frequency/{k}"][()]) for k in FREQ_KEYS}
                freqs = np.asarray(f["frequency/frequencies_hz"][()])
                antenna = {k: f[f"antenna/{k}"][()] for k in f["antenna"]}

        n = len(scene_rows)
        if cfr.shape[0] != n:
            print(f"[warn] {scene}: L1 has {cfr.shape[0]} links but manifest says {n}")

        # L2 covers only the links that were zero in L1, in zero_links.csv order
        zero_order = []
        if status_path.is_file():
            zero_order = list(csv.DictReader(status_path.open()))
        l2_map = {}
        if l2_path.is_file() and zero_order:
            with h5py.File(l2_path, "r") as f2:
                l2_cfr = np.asarray(f2["cfr"][()])
            for i, st in enumerate(zero_order):
                if i < l2_cfr.shape[0] and st["recovered"] == "1":
                    l2_map[st["global_link_id"]] = l2_cfr[i]

        source = np.zeros(n, dtype=np.uint8)
        power = np.zeros(n, dtype=np.float64)
        for i, row in enumerate(scene_rows):
            link_id = row["global_link_id"]
            p1 = float(np.mean(np.abs(cfr[i]) ** 2))
            if p1 > 1e-28:
                source[i] = 1
                power[i] = 10 * np.log10(p1)
            elif link_id in l2_map:
                cfr[i] = l2_map[link_id]
                source[i] = 2
                power[i] = 10 * np.log10(float(np.mean(np.abs(cfr[i]) ** 2)) + 1e-30)
            else:
                power[i] = -300.0

        out_path = scenes_dir / f"{scene}.h5"
        with h5py.File(out_path, "w") as f:
            f.create_dataset("cfr", data=cfr.astype(np.complex64))
            f.create_dataset("global_link_id", data=np.asarray([int(r["global_link_id"]) for r in scene_rows], dtype=np.int64))
            f.create_dataset("bs_id", data=np.asarray([int(r["bs_id"]) for r in scene_rows], dtype=np.int64))
            f.create_dataset("ue_id", data=np.asarray([int(r["ue_id"]) for r in scene_rows], dtype=np.int64))
            f.create_dataset("source", data=source)
            f.create_dataset("power_db", data=power.astype(np.float32))
            f.create_dataset("is_zero", data=(source == 0))
            f.create_dataset("valid_mask", data=(source > 0))
            f.create_dataset("ref_power_db", data=np.asarray([float(r["csi_power_db"]) for r in scene_rows], dtype=np.float32))
            f.create_dataset("bs_pos", data=np.asarray([[float(r["bs_x_m"]), float(r["bs_y_m"]), float(r["bs_z_m"])] for r in scene_rows], dtype=np.float32))
            f.create_dataset("ue_pos", data=np.asarray([[float(r["ue_x_m"]), float(r["ue_y_m"]), float(r["ue_z_m"])] for r in scene_rows], dtype=np.float32))

            g = f.create_group("frequency")
            g.create_dataset("frequencies_hz", data=freqs)
            for k, v in grid.items():
                g.create_dataset(k, data=v)
            ga = f.create_group("antenna")
            for k, v in antenna.items():
                ga.create_dataset(k, data=v)
            gs = f.create_group("scene")
            gs.create_dataset("scene_id", data=scene)
            gs.create_dataset("scene_class", data=scene_rows[0]["scene_class"])
            gs.create_dataset("num_links", data=n)
            gm = f.create_group("meta")
            gm.create_dataset("contract", data="2988")
            gm.create_dataset("engine", data="differt 0.12.0")
            gm.create_dataset("created_at", data=datetime.now(timezone.utc).isoformat())

        for i, row in enumerate(scene_rows):
            index_rows.append(
                {
                    "scene_id": scene,
                    "global_link_id": row["global_link_id"],
                    "bs_id": row["bs_id"],
                    "ue_id": row["ue_id"],
                    "scene_class": row["scene_class"],
                    "source": int(source[i]),
                    "is_zero": int(source[i] == 0),
                    "power_db": round(float(power[i]), 3),
                    "ref_power_db": float(row["csi_power_db"]),
                    "h5": str(out_path),
                    "h5_index": i,
                }
            )
        totals["links"] += n
        totals["l1"] += int((source == 1).sum())
        totals["l2"] += int((source == 2).sum())
        totals["zero"] += int((source == 0).sum())

    with (args.out / "link_index.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(index_rows[0].keys()))
        w.writeheader()
        w.writerows(index_rows)

    manifest = {
        "dataset": "front3d_7216_differt_2988_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "contract": "2988",
        "engine": "differt 0.12.0",
        "center_frequency_hz": grid["center_frequency_hz"],
        "num_subcarriers": grid["num_subcarriers"],
        "bs_array": "4x4 (16 ports), 0.5 lambda, iso, V",
        "ue_array": "1x2 (2 ports), 0.5 lambda, iso, V",
        "cfr_shape": ["num_links", 16, 2, 768],
        "num_scenes": len(by_scene),
        "num_links": totals["links"],
        "links_l1": totals["l1"],
        "links_l2": totals["l2"],
        "links_zero": totals["zero"],
        "coverage": round((totals["l1"] + totals["l2"]) / totals["links"], 4),
        "source": {
            "l1_root": str(args.l1_root),
            "l2_root": str(args.l2_root),
            "manifest_csv": str(args.manifest),
        },
        "provenance_field": {
            "source=1": "order-1 (LoS + single reflection/transmission)",
            "source=2": "order-2 transmission rescue",
            "source=0": "still empty",
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(json.dumps({k: totals[k] for k in ("links", "l1", "l2", "zero")}, indent=2))
    print(f"coverage {(totals['l1'] + totals['l2']) / totals['links'] * 100:.1f}%  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
