#!/usr/bin/env python
"""Run DiffeRT on the selected links and write one HDF5 per scene."""

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
    DEFAULT_ANTENNA,
    DEFAULT_GRID,
    DEFAULT_SOLVER,
    AntennaSpec,
    FrequencyGrid,
    SolverSpec,
    ensure_output_dirs,
)
from differt_crosscheck.io import write_scene_h5  # noqa: E402
from differt_crosscheck.scene import (  # noqa: E402
    build_link_scene,
    load_front3d_scene,
    mesh_stats,
)
from differt_crosscheck.solver import compute_cfr  # noqa: E402


def _parse_solver(args: argparse.Namespace) -> SolverSpec:
    return SolverSpec(
        tracer=args.tracer,
        order=args.order,
        num_rays=args.num_rays,
        max_num_candidates=args.max_num_candidates,
        include_los=not args.no_los,
        reflection=True,
        diffraction=args.diffraction,
        transmission=args.transmission,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--selection", type=Path, default=None)
    parser.add_argument("--scene", action="append", help="restrict to these scene keys")
    parser.add_argument(
        "--tracer", default=DEFAULT_SOLVER.tracer, choices=["sbr", "hybrid", "exhaustive"]
    )
    parser.add_argument("--order", type=int, default=DEFAULT_SOLVER.order)
    parser.add_argument("--num-rays", type=int, default=DEFAULT_SOLVER.num_rays)
    parser.add_argument("--max-num-candidates", type=int, default=DEFAULT_SOLVER.max_num_candidates)
    parser.add_argument("--diffraction", action="store_true")
    parser.add_argument("--transmission", action="store_true")
    parser.add_argument("--no-los", action="store_true")
    parser.add_argument("--freq-chunk", type=int, default=128)
    parser.add_argument("--tx-chunk", type=int, default=2)
    parser.add_argument("--explicit-antenna", action="store_true", default=False)
    parser.add_argument("--shortcut-antenna", dest="explicit_antenna", action="store_false")
    args = parser.parse_args()

    paths = ensure_output_dirs(args.output_root)
    selection_path = args.selection or (paths["links"] / "selection.json")
    selection = json.loads(Path(selection_path).read_text())

    grid: FrequencyGrid = DEFAULT_GRID
    antenna: AntennaSpec = DEFAULT_ANTENNA
    solver = _parse_solver(args)
    frequencies = grid.frequencies_hz()

    from differt_crosscheck.antenna import resolve_polarization  # noqa: PLC0415

    tx_pol = resolve_polarization("V", grid.center_frequency_hz, explicit=args.explicit_antenna)
    rx_pol = resolve_polarization("V", grid.center_frequency_hz, explicit=args.explicit_antenna)

    scenes = selection["scenes"]
    if args.scene:
        keep = set(args.scene)
        scenes = [s for s in scenes if s["scene_key"] in keep]

    summary = {
        "policy": selection.get("policy"),
        "solver": solver.as_dict(),
        "grid": grid.as_dict(),
        "antenna": {"bs": antenna.bs.as_dict(), "ue": antenna.ue.as_dict()},
        "explicit_antenna": bool(args.explicit_antenna),
        "scenes": [],
    }

    for scene_spec in scenes:
        key = scene_spec["scene_key"]
        t_scene = time.time()
        base = load_front3d_scene(scene_spec["scene_xml"])
        stats = mesh_stats(base)

        link_scene = build_link_scene(
            base,
            np.asarray(scene_spec["ue_positions"], dtype=float),
            np.asarray(scene_spec["bs_center"], dtype=float),
            float(scene_spec["bs_yaw_rad"]),
            antenna,
        )

        result = compute_cfr(
            link_scene,
            frequencies,
            solver,
            (tx_pol, rx_pol),
            freq_chunk=args.freq_chunk,
            tx_chunk=args.tx_chunk,
        )

        cfr = result.cfr  # (num_tx, ports, freq)
        num_links, ports, num_freq = cfr.shape
        cfr_contract = cfr.reshape(num_links, ports, 1, num_freq)

        out_path = paths["cfr"] / f"{key}.h5"
        write_scene_h5(
            out_path,
            scene_key=key,
            scene_class=scene_spec["scene_class"],
            scene_file=scene_spec["scene_xml"],
            cfr=cfr_contract,
            bs_id=np.full(num_links, scene_spec["bs_id"], dtype=np.int64),
            ue_id=np.asarray(scene_spec["sionna_ue_ids"], dtype=np.int64),
            bs_pos=np.tile(np.asarray(scene_spec["bs_center"], dtype=float), (num_links, 1)),
            ue_pos=np.asarray(scene_spec["ue_positions"], dtype=float),
            rx_pos=link_scene.rx_positions,
            frequencies_hz=frequencies,
            grid=grid.as_dict(),
            antenna={"tx": antenna.ue.as_dict(), "rx": antenna.bs.as_dict()},
            solver=solver.as_dict(),
            mesh=stats,
            extra_meta={
                "size_label": scene_spec["size_label"],
                "sionna_rows": scene_spec["sionna_rows"],
                "sionna_h5": scene_spec["sionna_h5"],
                "bs_yaw_rad": scene_spec["bs_yaw_rad"],
                "floor_area_m2": scene_spec["floor_area_m2"],
                "diagnostics": result.diagnostics,
            },
        )

        elapsed = time.time() - t_scene
        diag = result.diagnostics
        print(
            f"[{key}] links={num_links} ports={ports} freqs={num_freq} "
            f"tris={stats['num_triangles']} paths={diag['num_valid_paths_total']} "
            f"trace={diag['trace_seconds_total']:.1f}s fields={diag['fields_seconds_total']:.1f}s "
            f"total={elapsed:.1f}s -> {out_path}",
            flush=True,
        )
        summary["scenes"].append(
            {
                "scene_key": key,
                "size_label": scene_spec["size_label"],
                "h5": str(out_path),
                "elapsed_seconds": elapsed,
                "mesh": stats,
                "diagnostics": diag,
            }
        )

    out_summary = paths["cfr"] / "run_summary.json"
    out_summary.write_text(json.dumps(summary, indent=2))
    print(f"summary -> {out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
