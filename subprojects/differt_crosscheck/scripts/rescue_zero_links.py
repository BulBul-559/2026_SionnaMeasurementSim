#!/usr/bin/env python
"""Rescue pass for links that order-1 left empty.

Strategy (validated on a 10-link prototype):

* take the architecture-only scene and apply **5 cm vertex clustering** -- the clusters
  stay inside a plane, so the plane equations (and therefore every path) are preserved
  while the triangle count drops 3-8x;
* trace **order 2 restricted to transmission**, the mechanism a cross-room link needs
  (the interaction alphabet halves, so candidates drop 4x versus {R, T});
* prune the candidate array to the valid paths before solving the fields, and solve in
  frequency chunks;
* write a per-scene HDF5 with the full contract shape ``[links, 16, 2, 768]`` plus a
  status row per link so the next round knows what is still missing.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from differt_crosscheck.config import get_profile  # noqa: E402

sys.path.insert(0, str(HERE / "scripts"))
from simplify_mesh import load_mesh  # noqa: E402


def decimate_cluster(verts: np.ndarray, faces: np.ndarray, cell: float):
    keys = np.floor(verts / cell).astype(np.int64)
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
    new_v = np.zeros((len(uniq), 3), dtype=float)
    counts = np.zeros(len(uniq), dtype=float)
    np.add.at(new_v, inverse, verts)
    np.add.at(counts, inverse, 1.0)
    new_v /= counts[:, None]
    shift = np.linalg.norm(new_v[inverse] - verts, axis=1)
    new_f = inverse[faces]
    good = (new_f[:, 0] != new_f[:, 1]) & (new_f[:, 1] != new_f[:, 2]) & (new_f[:, 0] != new_f[:, 2])
    new_f = new_f[good]
    _, first = np.unique(np.sort(new_f, axis=1), axis=0, return_index=True)
    return new_v, new_f[np.sort(first)], float(shift.max()) if len(shift) else 0.0


def crop_tunnel(verts, faces, a, b, margin):
    seg = b - a
    d2 = float(seg @ seg) + 1e-9
    t = np.clip(((verts - a) @ seg) / d2, 0.0, 1.0)
    closest = a + t[:, None] * seg
    keep = np.linalg.norm(verts - closest, axis=1) <= margin
    keep_f = keep[faces].any(axis=1)
    used = np.unique(faces[keep_f])
    remap = -np.ones(len(verts), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return verts[used], remap[faces[keep_f]]


def panel(center: np.ndarray, spacing: float) -> np.ndarray:
    ii, jj = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
    local = np.stack(
        [np.zeros(ii.size), spacing * jj.ravel() - 1.5 * spacing, -spacing * ii.ravel() + 1.5 * spacing],
        axis=-1,
    )
    return local + center[None, :]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--scene", default=None, help="only this scene (default: every scene with zero links)")
    parser.add_argument("--cell", type=float, default=0.05)
    parser.add_argument("--budget", type=int, default=2500, help="triangle budget before falling back to cropping")
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--types", default="t")
    parser.add_argument("--rx-chunk", type=int, default=4)
    parser.add_argument("--freq-chunk", type=int, default=8)
    parser.add_argument("--profile", default="2988")
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp

    import h5py
    from differt.em import InteractionType, TracedFields
    from differt.geometry import Mesh, Scene

    from differt_crosscheck.io import write_scene_h5
    from differt_crosscheck.solver import _prune_paths, build_radio_materials
    from differt_crosscheck.scene import mesh_stats

    profile = get_profile(args.profile)
    man = {r["global_link_id"]: r for r in csv.DictReader(args.manifest.open())}
    zeros = list(csv.DictReader((args.run_root / "compare" / "zero_links.csv").open()))
    by_scene: dict[str, list[dict]] = {}
    for row in zeros:
        by_scene.setdefault(row["scene_id"], []).append(row)
    scenes = [args.scene] if args.scene else sorted(by_scene)

    allowed = set()
    if "t" in args.types:
        allowed.add(InteractionType.TRANSMISSION)
    if "r" in args.types:
        allowed.add(InteractionType.REFLECTION)
    allowed = frozenset(allowed)

    spacing = 0.5 * 299_792_458.0 / profile.grid.center_frequency_hz
    freqs = profile.grid.frequencies_hz()
    mats = build_radio_materials(0.1)

    out_dir = args.out_root / "scenes"
    status_dir = args.out_root / "status"
    out_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)

    for scene_id in scenes:
        rows = by_scene.get(scene_id, [])
        if not rows:
            continue
        t_scene = time.time()
        xml = args.run_root / "scenes_reduced" / f"{scene_id}.xml"
        V, F = load_mesh(xml)
        # pick one clustering cell for the whole scene so that every link in it shares a
        # single mesh (the field solver is jit-compiled per mesh, so a shared mesh is the
        # difference between ~15 s and ~60 s per link)
        cell_used = args.cell
        cv_all, cf_all, shift_all = decimate_cluster(V, F, cell_used)
        if len(cf_all) > args.budget:
            for cand in (0.075, 0.10, 0.15, 0.20):
                v2, f2, s2 = decimate_cluster(V, F, cand)
                if len(f2) < len(cf_all):
                    cell_used, cv_all, cf_all, shift_all = cand, v2, f2, s2
                if len(f2) <= args.budget:
                    break

        links_out = []
        meta_rows = []
        status_fh = (status_dir / f"{scene_id}.csv").open("w", newline="")
        writer = csv.writer(status_fh)
        writer.writerow(["scene_id", "global_link_id", "recovered", "paths_valid", "tris_in", "tris_clustered", "seconds", "error"])
        for row in rows:
            m = man[row["global_link_id"]]
            a = np.array([float(m["bs_x_m"]), float(m["bs_y_m"]), float(m["bs_z_m"])])
            b = np.array([float(m["ue_x_m"]), float(m["ue_y_m"]), float(m["ue_z_m"])])
            rx = panel(a, spacing)
            tx = np.stack([b + np.array([0.0, -spacing / 2, 0.0]), b + np.array([0.0, spacing / 2, 0.0])])

            # geometry: cluster everything; if that is still above budget, crop a tunnel
            # around this specific bs-ue segment and re-cluster (cheap, keeps the walls
            # that matter for this link)
            gv, gf, gshift = cv_all, cf_all, shift_all
            geom = f"cluster {cell_used}"
            if len(gf) > args.budget:
                for margin in (1.5, 1.0):
                    cV, cF = crop_tunnel(V, F, a, b, margin)
                    gv, gf, gshift = decimate_cluster(cV, cF, cell_used)
                    geom = f"crop {margin} + cluster {cell_used}"
                    if len(gf) <= args.budget:
                        break
            link_mesh = Mesh(
                vertices=jnp.asarray(gv),
                triangles=jnp.asarray(gf),
                face_materials=jnp.zeros((len(gf),), dtype=jnp.int32),
                material_names=("itu_concrete",),
            )

            cfr = np.zeros((profile.bs.num_elements, 2, len(freqs)), dtype=np.complex64)
            valid_total = 0
            t_link = time.time()
            err = ""
            try:
                for tx_index in range(2):
                    for start in range(0, profile.bs.num_elements, args.rx_chunk):
                        ports = np.arange(start, min(start + args.rx_chunk, profile.bs.num_elements))
                        scene = Scene(
                            transmitters=jnp.asarray(tx[tx_index : tx_index + 1]),
                            receivers=jnp.asarray(rx[ports]),
                            mesh=link_mesh,
                        )
                        paths = scene.trace_paths(order=args.order, solver="exhaustive", allowed_interactions=allowed)
                        jax.block_until_ready(paths.vertices)
                        valid_total += int(np.sum(np.asarray(paths.mask)))
                        pruned, kept = _prune_paths(paths)
                        if kept == 0:
                            continue
                        solve = jax.jit(
                            jax.vmap(
                                lambda f: TracedFields.from_paths(
                                    pruned, link_mesh, frequency=f, radio_materials=mats
                                )
                            )
                        )
                        out = np.zeros((len(freqs), 1, len(ports)), dtype=np.complex128)
                        for fs in range(0, len(freqs), args.freq_chunk):
                            chunk = jnp.asarray(freqs[fs : fs + args.freq_chunk])
                            tf = solve(chunk)
                            jax.block_until_ready(tf.fields)
                            g = np.asarray(tf.fields)
                            mm = np.asarray(tf.mask)
                            out[fs : fs + args.freq_chunk] = np.sum(np.where(mm, g, 0.0), axis=-1)
                        cfr[ports, tx_index, :] = out[:, 0, :].T.astype(np.complex64)
            except Exception as exc:  # noqa: BLE001
                err = f"{type(exc).__name__}: {str(exc)[:100]}"
            recovered = valid_total > 0
            links_out.append(cfr)
            meta_rows.append((b, valid_total, recovered))
            writer.writerow(
                [
                    scene_id,
                    row["global_link_id"],
                    int(recovered),
                    valid_total,
                    len(F),
                    len(gf),
                    round(time.time() - t_link, 1),
                    err,
                ]
            )
            status_fh.flush()

        if links_out:
            cfr_arr = np.stack(links_out)
            h5 = out_dir / f"{scene_id}.h5"
            write_scene_h5(
                h5,
                scene_key=scene_id,
                scene_class=rows[0]["scene_class"],
                scene_file=str(xml),
                cfr=cfr_arr,
                bs_id=np.zeros(len(links_out), dtype=np.int64),
                ue_id=np.asarray([int(r["ue_id"]) for r in rows], dtype=np.int64),
                bs_pos=np.asarray([meta_rows[i][0] for i in range(len(rows))], dtype=float),
                ue_pos=np.asarray([meta_rows[i][0] for i in range(len(rows))], dtype=float),
                rx_pos=np.zeros((profile.bs.num_elements, 3)),
                frequencies_hz=freqs,
                grid=profile.grid.as_dict(),
                antenna={"tx": profile.ue.as_dict(), "rx": profile.bs.as_dict()},
                solver={
                    "tracer": "exhaustive",
                    "order": args.order,
                    "interactions": sorted(x.name.lower() for x in allowed),
                    "material_thickness_m": 0.1,
                    "geometry": f"arch-only + vertex clustering {args.cell} m",
                },
                mesh={"tris_in": len(F), "tris_clustered": len(cf_all), "max_vertex_shift_m": shift_all},
                extra_meta={"rescue_round": 1},
            )
        status_fh.close()
        print(
            f"[{scene_id}] links={len(rows)} recovered={sum(1 for r in meta_rows if r[2])} "
            f"tris {len(F)}->{len(cf_all)} {time.time() - t_scene:.1f}s",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
