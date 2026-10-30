#!/usr/bin/env python
"""Prototype: per-link mesh cropping + order-2 transmission tracing for links that
order-1 leaves empty.

For each requested link:
  1. load the architecture-only scene and crop it to a tunnel of ``--margin`` metres
     around the straight bs-ue segment (plus a floor/ceiling margin),
  2. trace order-2 paths restricted to transmission (the mechanism a cross-room link
     actually needs; the alphabet halves, so candidates drop 4x versus {R,T}),
  3. synthesise the CFR on the 768-point grid and compare it with the reference row.

Prints one JSON line per link plus a summary, so it can be driven over a sample.
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


def crop_mesh(verts: np.ndarray, faces: np.ndarray, a: np.ndarray, b: np.ndarray, margin: float):
    """Keep triangles whose vertices come within ``margin`` of the segment a-b."""
    seg = b - a
    d2 = float(seg @ seg) + 1e-9
    t = np.clip(((verts - a) @ seg) / d2, 0.0, 1.0)
    closest = a + t[:, None] * seg
    dist = np.linalg.norm(verts - closest, axis=1)
    keep_v = dist <= margin
    keep_f = keep_v[faces].any(axis=1)
    used = np.unique(faces[keep_f])
    remap = -np.ones(len(verts), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return verts[used], remap[faces[keep_f]]




def decimate_cluster(verts: np.ndarray, faces: np.ndarray, cell: float):
    """Lossy decimation by vertex clustering (weld vertices inside ``cell`` cubes).

    Every kept vertex moves by at most ``cell * sqrt(3) / 2``, so the geometry error is
    bounded and known. Degenerate triangles are dropped.
    """
    if cell <= 0:
        return verts, faces, 0.0
    keys = np.floor(verts / cell).astype(np.int64)
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
    # representative vertex = centroid of the cluster
    new_v = np.zeros((len(uniq), 3), dtype=float)
    counts = np.zeros(len(uniq), dtype=float)
    np.add.at(new_v, inverse, verts)
    np.add.at(counts, inverse, 1.0)
    new_v /= counts[:, None]
    shift = np.linalg.norm(new_v[inverse] - verts, axis=1)
    new_f = inverse[faces]
    good = (new_f[:, 0] != new_f[:, 1]) & (new_f[:, 1] != new_f[:, 2]) & (new_f[:, 0] != new_f[:, 2])
    new_f = new_f[good]
    # drop duplicate triangles
    _, first = np.unique(np.sort(new_f, axis=1), axis=0, return_index=True)
    new_f = new_f[np.sort(first)]
    return new_v, new_f, float(shift.max())

def build_scene(cropped_v: np.ndarray, cropped_f: np.ndarray, rx: np.ndarray, tx: np.ndarray):
    import jax.numpy as jnp

    from differt.geometry import Mesh, Scene

    mesh = Mesh(
        vertices=jnp.asarray(cropped_v),
        triangles=jnp.asarray(cropped_f),
        face_materials=jnp.zeros((len(cropped_f),), dtype=jnp.int32),
        material_names=("itu_concrete",),
    )
    return Scene(transmitters=jnp.asarray(tx), receivers=jnp.asarray(rx), mesh=mesh)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--margin", type=float, default=1.5)
    parser.add_argument("--decimate-cell", type=float, default=0.0)
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--types", default="t", help="t = transmission only, rt = reflection+transmission")
    parser.add_argument("--ports", type=int, default=4)
    parser.add_argument("--profile", default="2988")
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp

    import h5py

    from differt.em import InteractionType, TracedFields

    from differt_crosscheck.solver import _prune_paths, build_radio_materials
    from differt_crosscheck.metrics import score_pair

    profile = get_profile(args.profile)
    man = {r["global_link_id"]: r for r in csv.DictReader(args.manifest.open())}
    zeros = list(csv.DictReader((args.run_root / "compare" / "zero_links.csv").open()))[: args.limit]

    allowed = set()
    if "t" in args.types:
        allowed.add(InteractionType.TRANSMISSION)
    if "r" in args.types:
        allowed.add(InteractionType.REFLECTION)
    allowed = frozenset(allowed)

    freqs = jnp.asarray(profile.grid.frequencies_hz())
    lam = 299_792_458.0 / profile.grid.center_frequency_hz
    spacing = 0.5 * lam

    results = []
    for row in zeros:
        m = man[row["global_link_id"]]
        scene_id = row["scene_id"]
        a = np.array([float(m["bs_x_m"]), float(m["bs_y_m"]), float(m["bs_z_m"])])
        b = np.array([float(m["ue_x_m"]), float(m["ue_y_m"]), float(m["ue_z_m"])])

        # receiver panel: 4x4 in the y-z plane, 0.5 lambda spacing, orientation fixed
        ii, jj = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
        local = np.stack([np.zeros(ii.size), spacing * jj.ravel() - 1.5 * spacing,
                          -spacing * ii.ravel() + 1.5 * spacing], axis=-1)
        rx = local + a[None, :]
        tx = np.stack([b + np.array([0.0, -spacing / 2, 0.0]),
                       b + np.array([0.0, spacing / 2, 0.0])])

        xml = args.run_root / "scenes_reduced" / f"{scene_id}.xml"
        t0 = time.time()
        V, F = load_mesh(xml)
        if args.margin and args.margin > 0:
            cv, cf = crop_mesh(V, F, a, b, args.margin)
        else:
            cv, cf = V, F          # no cropping: keep every blocker

        cv, cf, shift = decimate_cluster(cv, cf, args.decimate_cell)
        t_crop = time.time() - t0

        cand_est = (len(cf) * len(allowed)) ** args.order
        entry = {
            "scene_id": scene_id,
            "link": row["global_link_id"],
            "tris_full": int(len(F)),
            "tris_cropped": int(len(cf)),
            "candidate_est": float(cand_est),
            "crop_seconds": round(t_crop, 1),
            "max_vertex_shift_m": round(shift, 4),
        }

        try:
            scene = build_scene(cv, cf, rx[: args.ports], tx[:1])
            t1 = time.time()
            paths = scene.trace_paths(order=args.order, solver="exhaustive", allowed_interactions=allowed)
            jax.block_until_ready(paths.vertices)
            entry["trace_seconds"] = round(time.time() - t1, 1)
            mask = np.asarray(paths.mask)
            entry["paths_valid"] = int(mask.sum())
            entry["paths_slots"] = int(mask.shape[-1])

            if entry["paths_valid"] > 0:
                mats = build_radio_materials(0.1)
                # 1M candidate slots but only a handful are valid: prune before solving,
                # then solve in frequency chunks (a single 768-frequency batch does not fit)
                pruned, n_kept = _prune_paths(paths)
                entry["slots_pruned"] = int(n_kept)
                solve = jax.jit(
                    jax.vmap(lambda f: TracedFields.from_paths(
                        pruned, scene.mesh, frequency=f, radio_materials=mats))
                )
                out = np.zeros((len(freqs), 1, args.ports), dtype=np.complex128)
                for start in range(0, len(freqs), 8):
                    chunk = freqs[start:start + 8]
                    tf = solve(chunk)
                    jax.block_until_ready(tf.fields)
                    g = np.asarray(tf.fields)
                    mm = np.asarray(tf.mask)
                    out[start:start + 8] = np.sum(np.where(mm, g, 0.0), axis=-1)
                cfr = np.transpose(out[:, 0, :])                    # (ports, nfreq)
                entry["power_db"] = float(10 * np.log10(np.mean(np.abs(cfr) ** 2) + 1e-30))
                np.save(args.run_root / "prototype" / f"{scene_id}_{row['global_link_id']}.npy", cfr)
            else:
                entry["power_db"] = -300.0
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
            print(json.dumps(entry), flush=True)
            results.append(entry)
            continue

        # reference
        ref_path = args.reference_dir / f"{scene_id}.h5"
        with h5py.File(ref_path, "r") as f:
            bs_pos = np.asarray(f["bs_pos"][()], dtype=float)
            ue_pos = np.asarray(f["ue_pos"][()], dtype=float)
            d_bs = np.linalg.norm(bs_pos - a[None, :], axis=1)
            d_ue = np.linalg.norm(ue_pos - b[None, :], axis=1)
            cand = np.where((d_bs < 1e-2) & (d_ue < 1e-2))[0]
            if cand.size:
                ref = np.asarray(f["cfr"][int(cand[0])], dtype=np.complex128)
                entry["ref_power_db"] = float(10 * np.log10(np.mean(np.abs(ref) ** 2) + 1e-30))
                if entry["paths_valid"] > 0:
                    sc = score_pair(
                        ref[:, : args.ports, :] if ref.ndim == 3 else ref,
                        cfr,
                        bandwidth_hz=profile.grid.bandwidth_hz,
                        num_subcarriers=profile.grid.num_subcarriers,
                    ) if ref.shape[0] == cfr.shape[0] else {}
                    entry["cos_sim"] = sc.get("cos_sim")
                    entry["power_offset_db"] = sc.get("power_offset_db")
                    entry["pdp_corr"] = sc.get("pdp_corr")
        print(json.dumps(entry), flush=True)
        results.append(entry)

    out = args.run_root / "prototype"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(results, indent=2))
    found = sum(1 for r in results if r.get("paths_valid", 0) > 0)
    print(f"--- links={len(results)} with_paths={found} ---")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
