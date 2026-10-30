#!/usr/bin/env python
"""Turn the 7216-link CSI query manifest into per-scene selection files.

The manifest (``runs/analysis/csi_query_manifests/A001/csi_queries_7216.csv``) lists one
row per link with the full 2988 contract. This script groups it by scene, builds the
architecture-only scene XML for each scene (DiffeRT cannot enumerate
reflection+transmission candidates on the full Front3D mesh), and writes one selection
JSON per scene in the format ``split_stage.py`` expects.

Output layout (under ``--output-root``)::

    selections/<scene_id>.json
    scenes_reduced/<scene_id>.xml
    scenes.json                     # ordered scene list for the drivers
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from differt_crosscheck.config import get_profile  # noqa: E402

sys.path.insert(0, str(HERE / "scripts"))
from build_reduced_scene import build as build_reduced  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--profile", default="2988")
    args = parser.parse_args()

    profile = get_profile(args.profile)
    rows = list(csv.DictReader(args.manifest.open()))
    if not rows:
        raise SystemExit("empty manifest")

    sel_dir = args.output_root / "selections"
    scene_dir = args.output_root / "scenes_reduced"
    sel_dir.mkdir(parents=True, exist_ok=True)
    scene_dir.mkdir(parents=True, exist_ok=True)

    by_scene: dict[str, list[dict]] = {}
    for row in rows:
        by_scene.setdefault(row["scene_id"], []).append(row)

    scene_entries = []
    for scene_id, scene_rows in sorted(by_scene.items()):
        first = scene_rows[0]
        full_xml = Path(first["scene_file_abs"])
        reduced_xml = scene_dir / f"{scene_id}.xml"
        if not reduced_xml.is_file():
            build_reduced(full_xml, reduced_xml)

        bs = [float(first["bs_x_m"]), float(first["bs_y_m"]), float(first["bs_z_m"])]
        ue_positions = [
            [float(r["ue_x_m"]), float(r["ue_y_m"]), float(r["ue_z_m"])] for r in scene_rows
        ]
        selection = {
            "policy": "csi_query_manifest_a001_v1",
            "profile": profile.name,
            "scenes": [
                {
                    "scene_key": scene_id,
                    "scene_class": first["scene_class"],
                    "scene_xml": str(reduced_xml),
                    "scene_xml_full": str(full_xml),
                    "bs_center": bs,
                    "bs_yaw_rad": 0.0,  # 2988 contract keeps the BS array orientation fixed
                    "ue_positions": ue_positions,
                    "sionna_ue_ids": [int(r["ue_id"]) for r in scene_rows],
                    "manifest_rows": [r["global_link_id"] for r in scene_rows],
                }
            ],
            "num_links_total": len(scene_rows),
        }
        (sel_dir / f"{scene_id}.json").write_text(json.dumps(selection, indent=2))
        scene_entries.append(
            {
                "scene_id": scene_id,
                "scene_class": first["scene_class"],
                "num_links": len(scene_rows),
                "selection": str(sel_dir / f"{scene_id}.json"),
                "scene_xml": str(reduced_xml),
            }
        )

    (args.output_root / "scenes.json").write_text(json.dumps(scene_entries, indent=2))
    total = sum(e["num_links"] for e in scene_entries)
    print(f"scenes={len(scene_entries)} links={total} -> {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
