"""HDF5 writer mirroring the Sionna-side scene dataset layout.

The point of mirroring the layout is that ``compare_with_sionna.py`` (and any future
reader) can open both files with the same code path. Dataset names, shapes and dtypes
match ``outputs/front3d_339_dichasus_like_cfr_v1/dataset/scenes/<scene>.h5``:

    /cfr                      (num_links, 32, 1, 768) complex64
    /bs_id /ue_id /bs_pos /rx_pos /tx_pos /ue_pos /link_id
    /valid_mask /is_finite /nan_count
    /frequency/{frequencies_hz, center_frequency_hz, ...}
    /antenna/{tx,rx}_{num_rows,num_cols,num_ant,pattern,polarization,spacing_lambda}
    /meta/*

DiffeRT-specific information (solver configuration, path counts, timings) lives under
``/differt`` so it is impossible to confuse with the Sionna contract fields.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

__all__ = ["write_scene_h5", "SCHEMA_VERSION"]

SCHEMA_VERSION = "differt_crosscheck_scene_v1"


def _str_ds(group, name, value):  # noqa: ANN001
    if name in group:
        del group[name]
    group.create_dataset(name, data=str(value))


def write_scene_h5(
    path: str | Path,
    *,
    scene_key: str,
    scene_class: str,
    scene_file: str,
    cfr: np.ndarray,
    bs_id: np.ndarray,
    ue_id: np.ndarray,
    bs_pos: np.ndarray,
    ue_pos: np.ndarray,
    rx_pos: np.ndarray,
    frequencies_hz: np.ndarray,
    grid: dict,
    antenna: dict,
    solver: dict,
    mesh: dict,
    extra_meta: dict | None = None,
    compression: str | None = None,
) -> Path:
    """Write one scene worth of CFR plus the metadata needed to compare it."""
    import h5py  # noqa: PLC0415

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cfr = np.asarray(cfr, dtype=np.complex64)
    num_links = cfr.shape[0]

    if cfr.ndim != 4:
        raise ValueError(f"cfr must be (links, ports, tx, freq), got {cfr.shape}")

    kwargs = {}
    if compression not in (None, "none"):
        kwargs = {"compression": compression, "compression_opts": 4}

    with h5py.File(path, "w") as f:
        f.create_dataset("cfr", data=cfr, **kwargs)
        f.create_dataset("bs_id", data=np.asarray(bs_id, dtype=np.int64))
        f.create_dataset("ue_id", data=np.asarray(ue_id, dtype=np.int64))
        f.create_dataset("link_id", data=np.arange(num_links, dtype=np.int64))
        f.create_dataset("bs_pos", data=np.asarray(bs_pos, dtype=np.float32))
        f.create_dataset("ue_pos", data=np.asarray(ue_pos, dtype=np.float32))
        f.create_dataset("rx_pos", data=np.asarray(rx_pos, dtype=np.float32))
        f.create_dataset("tx_pos", data=np.asarray(ue_pos, dtype=np.float32))

        power = np.mean(np.abs(cfr) ** 2, axis=(1, 2, 3))
        f.create_dataset("cfr_power_linear", data=power.astype(np.float32))
        f.create_dataset("csi_power_db", data=(10 * np.log10(power + 1e-30)).astype(np.float32))
        f.create_dataset("valid_mask", data=np.ones(num_links, dtype=bool))
        f.create_dataset("is_finite", data=np.isfinite(cfr).all(axis=(1, 2, 3)))
        f.create_dataset("nan_count", data=np.isnan(cfr).sum(axis=(1, 2, 3)).astype(np.int32))

        g = f.create_group("frequency")
        g.create_dataset("frequencies_hz", data=np.asarray(frequencies_hz, dtype=np.float64))
        for key in ("center_frequency_hz", "bandwidth_hz", "subcarrier_spacing_hz"):
            g.create_dataset(key, data=float(grid[key]))
        g.create_dataset("num_subcarriers", data=int(grid["num_subcarriers"]))

        ga = f.create_group("antenna")
        for side in ("tx", "rx"):
            spec = antenna[side]
            ga.create_dataset(f"{side}_num_rows", data=int(spec["num_rows"]))
            ga.create_dataset(f"{side}_num_cols", data=int(spec["num_cols"]))
            ga.create_dataset(f"{side}_num_ant", data=int(spec["num_elements"]))
            _str_ds(ga, f"{side}_pattern", spec["pattern"])
            _str_ds(ga, f"{side}_polarization", spec["polarization"])
            ga.create_dataset(
                f"{side}_spacing_m",
                data=float(spec["spacing_m"]),
            )
            ga.create_dataset(
                f"{side}_spacing_lambda",
                data=float(spec["spacing_lambda"]),
            )

        gs = f.create_group("scene")
        _str_ds(gs, "scene_key", scene_key)
        _str_ds(gs, "scene_class", scene_class)
        _str_ds(gs, "scene_file", scene_file)
        gs.create_dataset("num_links", data=num_links)
        gs.create_dataset("subcarriers", data=int(cfr.shape[-1]))
        gs.create_dataset("ports", data=int(cfr.shape[1]))
        gs.create_dataset("tx_ant", data=int(cfr.shape[2]))

        gd = f.create_group("differt")
        _str_ds(gd, "solver_json", json.dumps(solver, sort_keys=True))
        _str_ds(gd, "mesh_json", json.dumps(mesh, sort_keys=True))

        gm = f.create_group("meta")
        _str_ds(gm, "producer", "differt_crosscheck")
        _str_ds(gm, "schema_version", SCHEMA_VERSION)
        _str_ds(gm, "created_at", datetime.now(UTC).isoformat())
        _str_ds(gm, "engine", "differt")
        if extra_meta:
            _str_ds(gm, "extra_json", json.dumps(extra_meta, sort_keys=True, default=str))

    return path
