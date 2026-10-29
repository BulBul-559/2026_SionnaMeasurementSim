#!/usr/bin/env python
"""Resolve the three-scene / one-BS / ten-UE cross-check case and persist it."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from differt_crosscheck.config import DEFAULT_CASE, ensure_output_dirs  # noqa: E402
from differt_crosscheck.select import build_selection  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--bs-index", type=int, default=DEFAULT_CASE.bs_index)
    parser.add_argument("--num-ue", type=int, default=DEFAULT_CASE.num_ue_per_bs)
    args = parser.parse_args()

    paths = ensure_output_dirs(args.output_root)
    case = type(DEFAULT_CASE)(
        scene_keys=DEFAULT_CASE.scene_keys,
        size_labels=DEFAULT_CASE.size_labels,
        bs_index=args.bs_index,
        num_ue_per_bs=args.num_ue,
    )
    selection = build_selection(case)

    out = paths["links"] / "selection.json"
    out.write_text(json.dumps(selection, indent=2))

    for scene in selection["scenes"]:
        print(
            f"{scene['size_label']:>6}  {scene['scene_key']}  "
            f"class={scene['scene_class']:<11} area={scene['floor_area_m2']:.1f} m2  "
            f"bs_id={scene['bs_id']} yaw={scene['bs_yaw_rad']:+.3f}  "
            f"ues={len(scene['ue_positions'])}"
        )
    print(f"total links: {selection['num_links_total']}")
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
