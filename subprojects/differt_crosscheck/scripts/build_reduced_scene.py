#!/usr/bin/env python
"""Build a reduced (architecture-only) scene XML for DiffeRT feasibility work.

Front3D scenes are dominated by furniture geometry: for ``front3d_3682`` the
architecture is ~1k triangles while furniture accounts for ~46k. DiffeRT's
*complete* path solvers (needed for diffraction and transmission candidates) grow
with the primitive count and are unusable at 47k-500k triangles, but they are
perfectly tractable on the architecture-only subset.

The reduced scene keeps every ``empty_scene*`` / ``architecture*`` shape with its
original radio material and ``to_world`` transform, and rewrites the asset paths to
absolute ones so the XML can live outside the original asset folder.

Caveat: dropping furniture changes the channel. It is meant for feasibility and
mechanism checks, not as a drop-in replacement for the full scene.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

ARCH_MARKERS = ("empty_scene", "architecture")


def is_architecture(shape_id: str) -> bool:
    return any(marker in (shape_id or "") for marker in ARCH_MARKERS)


def build(scene_xml: Path, out: Path, *, keep_furniture: bool = False) -> dict:
    tree = ET.parse(scene_xml)
    root = tree.getroot()
    scene_dir = scene_xml.parent

    removed = 0
    for shape in list(root.findall("shape")):
        if keep_furniture or is_architecture(shape.get("id", "")):
            for string in shape.findall("string"):
                if string.get("name") == "filename":
                    value = string.get("value") or ""
                    path = Path(value)
                    if not path.is_absolute():
                        string.set("value", str((scene_dir / value).resolve()))
            continue
        root.remove(shape)
        removed += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return {"kept_shapes": len(root.findall("shape")), "removed_shapes": removed}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene_xml", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--keep-furniture", action="store_true")
    args = parser.parse_args()
    info = build(args.scene_xml, args.out, keep_furniture=args.keep_furniture)
    print(f"{args.scene_xml} -> {args.out}  {info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
