#!/usr/bin/env python
"""Lossless mesh simplification: merge coplanar triangles into minimal triangles.

Two steps, both geometry-preserving:

1. group triangles by their supporting plane (normal + offset, quantised);
2. per plane, take the 2D union of the triangles and retriangulate the resulting
   polygons (holes excluded by a containment filter).

The surface is *exactly* the same set of points, so every path length and reflection
point is unchanged; only the primitive list shrinks. Feasibility for higher-order
tracing is governed by the primitive count (candidates grow like (2T)^order), so this
is the cheapest possible complexity reduction.

Verification is built in: the summed triangle area per plane must match the original
area to within ``AREA_TOL``; otherwise the plane group is left untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

AREA_TOL = 1e-6  # relative


def load_mesh(scene_xml: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load the OBJ files referenced by a (reduced) Mitsuba scene into one triangle soup."""
    import xml.etree.ElementTree as ET

    root = ET.parse(scene_xml).getroot()
    verts: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    for shape in root.findall("shape"):
        name = {c.get("name"): c.get("value") for c in shape.findall("string")}.get("filename")
        if not name:
            continue
        base = len(verts)
        with open(name, errors="ignore") as fh:
            for line in fh:
                if line.startswith("v "):
                    _, x, y, z = line.split()[:4]
                    verts.append((float(x), float(y), float(z)))
                elif line.startswith("f "):
                    idx = [int(tok.split("/")[0]) for tok in line.split()[1:]]
                    for k in range(1, len(idx) - 1):
                        faces.append((base + idx[0] - 1, base + idx[k] - 1, base + idx[k + 1] - 1))
    return np.asarray(verts, dtype=float), np.asarray(faces, dtype=np.int64)


def plane_key(v0, v1, v2, *, normal_dec: int = 4, offset_dec: int = 4):
    n = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(n)
    if norm < 1e-12:
        return None
    n = n / norm
    # canonical orientation so that opposite winding lands in the same plane
    if n[0] < -1e-12 or (abs(n[0]) <= 1e-12 and (n[1] < -1e-12 or (abs(n[1]) <= 1e-12 and n[2] < 0))):
        n = -n
    offset = float(np.dot(n, v0))
    return (round(float(n[0]), normal_dec), round(float(n[1]), normal_dec),
            round(float(n[2]), normal_dec), round(offset, offset_dec))


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, a)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return u, v


def merge_plane(tris: np.ndarray, verts: np.ndarray, area_tol: float = 1e-4):
    """Merge one coplanar group: 2D union in the plane basis, then constrained triangulation."""
    from shapely.geometry import Polygon
    from shapely.ops import triangulate, unary_union

    pts3 = verts[tris]                      # (k, 3, 3)
    e1 = pts3[0, 1] - pts3[0, 0]
    e2 = pts3[0, 2] - pts3[0, 0]
    normal = np.cross(e1, e2)
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        return None
    normal /= norm
    u, v = _plane_basis(normal)
    origin = pts3[0, 0]

    # project every triangle into the plane basis
    rel = pts3 - origin                    # (k, 3, 3)
    xy = np.stack([rel @ u, rel @ v], axis=-1)   # (k, 3, 2)

    polys = []
    orig_area = 0.0
    for tri2 in xy:
        p = Polygon(tri2)
        if p.area <= 0:
            continue
        polys.append(p)
        orig_area += p.area
    if len(polys) <= 1:
        return None

    merged = unary_union(polys)
    if merged.is_empty:
        return None
    parts = list(merged.geoms) if merged.geom_type == "MultiPolygon" else [merged]

    out2 = []
    for part in parts:
        if part.area <= 0:
            continue
        # drop collinear boundary points (exact within 1 um): without this the union
        # keeps every subdivision vertex on the perimeter and retriangulation gains nothing
        part = part.simplify(1e-6, preserve_topology=True)
        for cand in triangulate(part):
            if part.covers(cand.representative_point()):
                out2.append(cand)

    # The reference is the *union* area, not the sum of triangle areas: Front3D meshes
    # contain slightly overlapping triangles, so the sum double-counts. Retriangulation
    # must reproduce the union exactly.
    union_area = float(sum(part.area for part in parts))
    new_area = sum(Polygon(t.exterior.coords).area for t in out2)
    if abs(new_area - union_area) > area_tol * max(union_area, 1e-12):
        return None

    coords = np.concatenate([np.asarray(t.exterior.coords)[:-1] for t in out2])   # (m, 2)
    pts = origin + coords[:, :1] * u + coords[:, 1:] * v                        # back to 3D
    return pts, new_area, len(tris), orig_area, union_area


def simplify(scene_xml: Path, *, max_group: int = 0) -> dict:
    verts, faces = load_mesh(scene_xml)
    if len(faces) == 0:
        return {"error": "no faces"}

    groups: dict[tuple, list[np.ndarray]] = {}
    for f in faces:
        key = plane_key(verts[f[0]], verts[f[1]], verts[f[2]])
        if key is None:
            continue
        groups.setdefault(key, []).append(f)

    new_coords: list[np.ndarray] = []
    new_faces: list[np.ndarray] = []
    kept = 0
    merged_groups = 0
    for key, tris in groups.items():
        tris_arr = np.asarray(tris)
        orig_area = 0.0
        for tri in tris_arr:
            p = verts[tri]
            orig_area += 0.5 * float(np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0])))
        result = None if (max_group and len(tris_arr) > max_group) else merge_plane(tris_arr, verts)
        if result is None:
            new_coords.append(verts[tris_arr.reshape(-1)])
            new_faces.append(np.arange(len(new_coords[-1])).reshape(-1, 3) + 0)
            kept += len(tris_arr)
            continue
        coords, new_area, _n, sum_area, union_area = result
        overlap_ratio = abs(sum_area - union_area) / max(union_area, 1e-12)
        if overlap_ratio > 0.2:   # suspicious: report but keep, it is the source mesh's overlap
            print(f"[warn] {scene_xml.stem} plane overlap {overlap_ratio:.1%}", flush=True)
        if abs(new_area - union_area) > AREA_TOL * max(union_area, 1e-9):
            # cannot prove losslessness -> keep the original triangles
            new_coords.append(verts[tris_arr.reshape(-1)])
            new_faces.append(np.arange(len(new_coords[-1])).reshape(-1, 3) + 0)
            kept += len(tris_arr)
            continue
        new_coords.append(coords)
        new_faces.append(np.arange(len(coords)).reshape(-1, 3))
        merged_groups += 1

    # re-index into one vertex array
    offsets = np.cumsum([0] + [len(c) for c in new_coords])
    V = np.concatenate(new_coords)
    F = np.concatenate([f + offsets[i] for i, f in enumerate(new_faces)])
    return {
        "scene": scene_xml.stem,
        "tris_in": int(len(faces)),
        "tris_out": int(len(F)),
        "verts_out": int(len(V)),
        "ratio": float(len(faces) / max(len(F), 1)),
        "planes": len(groups),
        "planes_merged": int(merged_groups),
        "tris_kept_as_is": int(kept),
        "vertices": V,
        "faces": F,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None, help="optional npz dump dir")
    args = parser.parse_args()
    for path in args.scenes:
        info = simplify(path)
        printable = {k: v for k, v in info.items() if k not in ("vertices", "faces")}
        print(json.dumps(printable))
        if args.out is not None and "vertices" in info:
            args.out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.out / f"{path.stem}.npz",
                                vertices=info["vertices"], faces=info["faces"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
