"""Generate BS-wise RSS radio-map heatmaps over a floorplan.

The input can be a sharded run directory, a manifest, a results directory, or
a single HDF5 result file. Outputs are written to ``<run>/figures/heatmaps``
by default.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sionna_measurement_sim.visualization.radio_map import (  # noqa: E402
    DEFAULT_RSSI_DATASET,
    RadioMapRenderConfig,
    generate_radio_map_heatmaps,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_or_hdf5", type=Path, help="Run directory, manifest, or HDF5 file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <run>/figures/heatmaps.",
    )
    parser.add_argument("--floorplan-image", type=Path, default=None)
    parser.add_argument("--floorplan-meta", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=["interpolated", "samples", "both"],
        default="interpolated",
        help="Radio-map rendering mode.",
    )
    parser.add_argument("--dataset", default=DEFAULT_RSSI_DATASET)
    parser.add_argument("--snapshot-index", type=int, default=0)
    parser.add_argument("--grid-resolution-m", type=float, default=None)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--idw-power", type=float, default=2.0)
    parser.add_argument("--heatmap-alpha", type=float, default=0.68)
    parser.add_argument("--point-size", type=float, default=16.0)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--show-samples", action="store_true")
    args = parser.parse_args()

    summary = generate_radio_map_heatmaps(
        args.run_or_hdf5,
        args.output_dir,
        floorplan_image=args.floorplan_image,
        floorplan_meta=args.floorplan_meta,
        config=RadioMapRenderConfig(
            render_mode=args.mode,
            snapshot_index=args.snapshot_index,
            value_dataset=args.dataset,
            grid_resolution_m=args.grid_resolution_m,
            interpolation_neighbors=args.neighbors,
            interpolation_power=args.idw_power,
            heatmap_alpha=args.heatmap_alpha,
            point_size=args.point_size,
            dpi=args.dpi,
            show_samples_on_interpolated=args.show_samples,
        ),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
