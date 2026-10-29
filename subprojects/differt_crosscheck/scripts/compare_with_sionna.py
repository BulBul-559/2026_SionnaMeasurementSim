#!/usr/bin/env python
"""Score the DiffeRT CFR against the Sionna reference for the same links."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from differt_crosscheck.config import DEFAULT_GRID, ensure_output_dirs  # noqa: E402
from differt_crosscheck.metrics import score_pair  # noqa: E402

METRIC_KEYS = (
    "cos_sim",
    "nmse_complex_db",
    "nmse_magnitude_db",
    "mag_corr",
    "port_corr",
    "pdp_corr",
    "pdp_l1",
    "power_ref_db",
    "power_dut_db",
    "power_offset_db",
    "rms_ds_ref_ns",
    "rms_ds_dut_ns",
    "freq_sel_ref_db",
    "freq_sel_dut_db",
)


def _load(path: Path, rows: list[int]) -> np.ndarray:
    import h5py  # noqa: PLC0415

    with h5py.File(path, "r") as f:
        return np.asarray(f["cfr"][sorted(rows)], dtype=np.complex64)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--selection", type=Path, default=None)
    parser.add_argument("--figures", action="store_true", default=True)
    parser.add_argument("--no-figures", dest="figures", action="store_false")
    args = parser.parse_args()

    paths = ensure_output_dirs(args.output_root)
    selection_path = args.selection or (paths["links"] / "selection.json")
    selection = json.loads(Path(selection_path).read_text())
    grid = DEFAULT_GRID

    rows_out: list[dict] = []
    per_scene: list[dict] = []

    for scene in selection["scenes"]:
        key = scene["scene_key"]
        differt_path = paths["cfr"] / f"{key}.h5"
        sionna_path = Path(scene["sionna_h5"])
        if not differt_path.is_file():
            print(f"[skip] {key}: {differt_path} missing (run run_links.py first)")
            continue

        idx = list(range(len(scene["sionna_rows"])))
        dut = _load(differt_path, idx)  # (links, 32, 1, 768)
        ref_all = _load(sionna_path, scene["sionna_rows"])  # (links, 32, 1, 768)

        scores = []
        for local, (ue_id, sionna_row) in enumerate(
            zip(scene["sionna_ue_ids"], scene["sionna_rows"], strict=True)
        ):
            h_ref = np.asarray(ref_all[local, :, 0, :], dtype=np.complex128)
            h_dut = np.asarray(dut[local, :, 0, :], dtype=np.complex128)
            score = score_pair(
                h_ref,
                h_dut,
                bandwidth_hz=grid.bandwidth_hz,
                num_subcarriers=grid.num_subcarriers,
            )
            row = {
                "scene_key": key,
                "size_label": scene["size_label"],
                "link_index": local,
                "ue_id": int(ue_id),
                "sionna_row": int(sionna_row),
                **{k: score[k] for k in METRIC_KEYS},
            }
            rows_out.append(row)
            scores.append(score)

        def _mean(name: str, _scores=scores) -> float:
            return float(np.mean([s[name] for s in _scores]))

        per_scene.append(
            {
                "scene_key": key,
                "size_label": scene["size_label"],
                "num_links": len(scores),
                **{f"mean_{k}": _mean(k) for k in METRIC_KEYS},
            }
        )
        print(
            f"[{key}] cos_sim={_mean('cos_sim'):.3f} "
            f"nmse={_mean('nmse_complex_db'):.1f} dB "
            f"mag_corr={_mean('mag_corr'):.3f} "
            f"pdp_corr={_mean('pdp_corr'):.3f} "
            f"power_offset={_mean('power_offset_db'):+.1f} dB "
            f"ds_ref={_mean('rms_ds_ref_ns'):.0f}ns ds_dut={_mean('rms_ds_dut_ns'):.0f}ns",
            flush=True,
        )

        if args.figures:
            _plot_scene(paths["figures"] / f"{key}_cfr_compare.png", key, ref_all, dut)

    csv_path = paths["compare"] / "metrics_per_link.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows_out[0].keys()) if rows_out else ["scene_key"]
        )
        writer.writeheader()
        writer.writerows(rows_out)

    overall = {}
    if rows_out:
        for k in METRIC_KEYS:
            vals = np.array([r[k] for r in rows_out], dtype=float)
            overall[k] = {
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }

    summary = {
        "policy": selection.get("policy"),
        "engine_ref": "sionna_rt (front3d_339_dichasus_like_cfr_v1)",
        "engine_dut": "differt 0.12.0",
        "num_links": len(rows_out),
        "overall": overall,
        "per_scene": per_scene,
        "metrics_csv": str(csv_path),
    }
    summary_path = paths["compare"] / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"metrics -> {csv_path}")
    print(f"summary -> {summary_path}")
    if overall:
        print(
            "overall: "
            + "  ".join(
                f"{k}={overall[k]['mean']:.3f}"
                for k in ("cos_sim", "nmse_complex_db", "mag_corr", "port_corr", "pdp_corr")
            )
        )
    return 0


def _plot_scene(path: Path, key: str, ref: np.ndarray, dut: np.ndarray) -> None:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    h_ref = np.abs(ref[0, :, 0, :])
    h_dut = np.abs(dut[0, :, 0, :])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, data, title in (
        (axes[0], h_ref, "Sionna |H|"),
        (axes[1], h_dut, "DiffeRT |H|"),
        (axes[2], h_dut - h_ref, "difference"),
    ):
        im = ax.imshow(
            20 * np.log10(data + 1e-30),
            aspect="auto",
            origin="lower",
            cmap="viridis",
        )
        ax.set_title(f"{key} {title}")
        ax.set_xlabel("subcarrier")
        ax.set_ylabel("port")
        fig.colorbar(im, ax=ax, label="dB")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
