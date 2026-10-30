#!/usr/bin/env python
"""Two-stage (GPU trace -> CPU fields) CFR generation for a contract profile.

Why two stages: DiffeRT's tracing is Warp ray/triangle arithmetic and runs ~5-10x faster
on a CUDA device, while the field solve (768 frequencies x few thousand paths) is *faster*
on CPU because the batches are small. Running both in one process forces a choice, so the
work is split:

  stage "trace"   (GPU): for every (UE port, receiver port-group) trace the path set,
                        prune it to the valid slots, and write a compact .npz.
  stage "fields"  (CPU): read the .npz files, solve the fields per frequency chunk and
                        write the contract HDF5 ([links, ports, tx_ant, subcarriers]).

The intermediate is small because pruning happens on the GPU side (candidate arrays are
thousands of slots wide, the valid set is hundreds).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from differt_crosscheck.config import (  # noqa: E402
    DEFAULT_SOLVER,
    AntennaSpec,
    SolverSpec,
    ensure_output_dirs,
    get_profile,
)
from differt_crosscheck.scene import (  # noqa: E402
    build_link_scene,
    load_front3d_scene,
    mesh_stats,
    subset_link_scene,
    ue_element_positions,
)


def _solver(args: argparse.Namespace) -> SolverSpec:
    return SolverSpec(
        tracer=args.tracer,
        order=args.order,
        num_rays=args.num_rays,
        max_num_candidates=args.max_num_candidates,
        include_los=not args.no_los,
        reflection=True,
        diffraction=args.diffraction,
        transmission=args.transmission,
        material_thickness_m=args.material_thickness,
    )


def plan_targets(selection: dict, profile, rx_chunk: int) -> list[dict]:
    """Expand the selection into per-(UE port, receiver port group) trace targets."""
    targets = []
    for scene in selection["scenes"]:
        bs_center = np.asarray(scene["bs_center"], dtype=float)
        yaw = float(scene["bs_yaw_rad"]) if profile.use_label_bs_yaw else 0.0
        num_ports = profile.bs.num_elements
        groups = [list(range(r, min(r + rx_chunk, num_ports))) for r in range(0, num_ports, rx_chunk)]
        for ue_index, ue_pos in enumerate(scene["ue_positions"]):
            ue_elems = ue_element_positions(
                np.asarray(ue_pos, dtype=float), profile.ue, profile.grid.center_frequency_hz, yaw
            )
            for tx_port in range(ue_elems.shape[0]):
                for group_index, ports in enumerate(groups):
                    targets.append(
                        {
                            "scene_key": scene["scene_key"],
                            "scene_xml": scene["scene_xml"],
                            "ue_index": ue_index,
                            "tx_port": tx_port,
                            "tx_position": ue_elems[tx_port].tolist(),
                            "group_index": group_index,
                            "group_ports": ports,
                            "bs_center": bs_center.tolist(),
                            "bs_yaw_rad": yaw,
                        }
                    )
    return targets


def stage_trace(args: argparse.Namespace, paths: dict, selection: dict, profile) -> int:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    from differt_crosscheck.solver import _prune_paths, trace_paths  # noqa: PLC0415

    solver = _solver(args)
    antenna = AntennaSpec(bs=profile.bs, ue=profile.ue)
    targets = plan_targets(selection, profile, args.rx_chunk)
    out_dir = paths["root"] / "paths"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[trace] device={jax.devices()[0]} targets={len(targets)} "
        f"solver={solver.tracer} order={solver.order} interactions={sorted(solver.interactions())}",
        flush=True,
    )

    scene_cache: dict[str, object] = {}
    kept = 0
    t_all = time.time()
    for i, target in enumerate(targets):
        key = target["scene_key"]
        if key not in scene_cache:
            scene_cache[key] = load_front3d_scene(target["scene_xml"])
        base = scene_cache[key]

        one = build_link_scene(
            base,
            np.asarray([target["tx_position"]], dtype=float),
            np.asarray(target["bs_center"], dtype=float),
            target["bs_yaw_rad"],
            antenna,
        )
        one = subset_link_scene(one, [0], target["group_ports"])

        for order in solver.orders():
            traced, _ = trace_paths(one, solver, order)
            pruned, n_valid = _prune_paths(traced)
            out = out_dir / (
                f"{key}_ue{target['ue_index']:03d}_tx{target['tx_port']:d}"
                f"_rx{target['group_index']:02d}_o{order}.npz"
            )
            np.savez_compressed(
                out,
                vertices=np.asarray(pruned.vertices),
                objects=np.asarray(pruned.objects),
                mask=np.asarray(pruned.mask),
                interaction_types=np.asarray(pruned.interaction_types),
                valid=n_valid,
            )
            kept += 1
        if (i + 1) % 10 == 0:
            print(f"[trace] {i + 1}/{len(targets)} targets, {time.time() - t_all:.1f}s", flush=True)

    print(f"[trace] done {kept} files in {time.time() - t_all:.1f}s", flush=True)
    (out_dir / "targets.json").write_text(json.dumps(targets, indent=2))
    return 0


def stage_fields(args: argparse.Namespace, paths: dict, selection: dict, profile) -> int:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    from differt.geometry import TracedPaths  # noqa: PLC0415

    from differt_crosscheck.antenna import resolve_polarization  # noqa: PLC0415
    from differt_crosscheck.io import write_scene_h5  # noqa: PLC0415
    from differt_crosscheck.solver import _cfr_for_paths, build_radio_materials  # noqa: PLC0415

    solver = _solver(args)
    frequencies = profile.grid.frequencies_hz()
    path_dir = paths["root"] / "paths"
    targets_file = args.targets_file or (path_dir / "targets.json")
    targets = json.loads(Path(targets_file).read_text())

    by_scene: dict[str, list[dict]] = {}
    for t in targets:
        by_scene.setdefault(t["scene_key"], []).append(t)

    mats = build_radio_materials(solver.material_thickness_m)
    tx_pol = resolve_polarization("V", profile.grid.center_frequency_hz, explicit=False)
    rx_pol = resolve_polarization("V", profile.grid.center_frequency_hz, explicit=False)

    summary = []
    for scene in selection["scenes"]:
        key = scene["scene_key"]
        num_ue = len(scene["ue_positions"])
        num_ports = profile.bs.num_elements
        num_tx_ant = profile.ue.num_elements
        cfr = np.zeros((num_ue, num_ports, num_tx_ant, frequencies.size), dtype=np.complex64)
        mesh = load_front3d_scene(scene["scene_xml"]).mesh

        t0 = time.time()
        for t in by_scene.get(key, []):
            ports = t["group_ports"]
            for order in solver.orders():
                f = path_dir / (
                    f"{key}_ue{t['ue_index']:03d}_tx{t['tx_port']:d}"
                    f"_rx{t['group_index']:02d}_o{order}.npz"
                )
                if not f.is_file():
                    continue
                data = np.load(f)
                if int(data["valid"]) == 0:
                    continue
                paths_obj = TracedPaths(
                    vertices=jnp.asarray(data["vertices"]),
                    objects=jnp.asarray(data["objects"]),
                    mask=jnp.asarray(data["mask"]),
                    interaction_types=jnp.asarray(data["interaction_types"]),
                )
                part, _ = _cfr_for_paths(
                    paths_obj,
                    mesh,
                    frequencies,
                    tx_pol,
                    rx_pol,
                    freq_chunk=args.freq_chunk,
                    num_tx=1,
                    num_el=len(ports),
                    tx_chunk=1,
                    radio_materials=mats,
                )
                cfr[t["ue_index"], ports, t["tx_port"], :] += part[0]
        elapsed = time.time() - t0

        out_path = paths["scenes"] / f"{key}.h5"
        write_scene_h5(
            out_path,
            scene_key=key,
            scene_class=scene["scene_class"],
            scene_file=scene["scene_xml"],
            cfr=cfr,
            bs_id=np.zeros(num_ue, dtype=np.int64),
            ue_id=np.asarray(scene["sionna_ue_ids"], dtype=np.int64),
            bs_pos=np.tile(np.asarray(scene["bs_center"], dtype=float), (num_ue, 1)),
            ue_pos=np.asarray(scene["ue_positions"], dtype=float),
            rx_pos=np.asarray(scene["bs_center"], dtype=float)[None, :],
            frequencies_hz=frequencies,
            grid=profile.grid.as_dict(),
            antenna={"tx": profile.ue.as_dict(), "rx": profile.bs.as_dict()},
            solver=solver.as_dict(),
            mesh=mesh_stats(load_front3d_scene(scene["scene_xml"])),
            extra_meta={"contract": profile.name, "stage": "split_gpu_trace_cpu_fields"},
        )
        power = 10 * np.log10(np.mean(np.abs(cfr) ** 2, axis=(1, 2, 3)) + 1e-30)
        print(f"[fields] {key}: shape={cfr.shape} {elapsed:.1f}s power_dB={np.round(power, 1).tolist()}", flush=True)
        summary.append({"scene_key": key, "h5": str(out_path), "seconds": elapsed, "shape": list(cfr.shape)})

    (paths["dataset"] / "split_summary.json").write_text(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["trace", "fields"])
    parser.add_argument("--profile", default="2988", choices=["2988", "339"])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--tracer", default="exhaustive")
    parser.add_argument("--order", type=int, default=2)
    parser.add_argument("--num-rays", type=int, default=DEFAULT_SOLVER.num_rays)
    parser.add_argument("--max-num-candidates", type=int, default=DEFAULT_SOLVER.max_num_candidates)
    parser.add_argument("--diffraction", action="store_true")
    parser.add_argument("--transmission", action="store_true")
    parser.add_argument("--no-los", action="store_true")
    parser.add_argument("--material-thickness", type=float, default=0.1)
    parser.add_argument("--rx-chunk", type=int, default=4)
    parser.add_argument("--freq-chunk", type=int, default=16)
    parser.add_argument("--targets-file", type=Path, default=None)
    args = parser.parse_args()

    profile = get_profile(args.profile)
    paths = ensure_output_dirs(args.output_root)
    selection = json.loads(Path(args.selection).read_text())
    args.selection = Path(args.selection)

    if args.stage == "trace":
        return stage_trace(args, paths, selection, profile)
    return stage_fields(args, paths, selection, profile)


if __name__ == "__main__":
    raise SystemExit(main())
