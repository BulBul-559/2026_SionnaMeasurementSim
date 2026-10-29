"""Case selection: three scenes (small / medium / large) x one BS x ten UEs.

Scenes are ranked by summed floor area from the prepared label, which is a stable,
already-audited quantity in this workspace. Only links that exist in the Sionna dataset
with ``valid_mask=True`` are selectable, so the comparison never runs against a hole.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import (
    DEFAULT_CASE,
    SCENE_ROOT,
    SIONNA_PREPARED_INDEX,
    SIONNA_SCENE_DIR,
    CaseSpec,
)

__all__ = ["load_prepared_index", "rank_scenes", "choose_cases", "build_selection"]


def load_prepared_index(path: Path | None = None) -> list[dict]:
    path = SIONNA_PREPARED_INDEX if path is None else Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _floor_area(label_path: str | Path) -> float:
    try:
        data = json.loads(Path(label_path).read_text())
    except Exception:  # noqa: BLE001 - a broken label simply cannot be ranked
        return float("nan")
    return float(sum(group.get("floor_area_m2", 0.0) for group in data.get("groups", [])))


def rank_scenes(rows: list[dict]) -> list[dict]:
    ranked = []
    for row in rows:
        ranked.append(
            {
                "scene_key": row["scene_key"],
                "scene_class": row["scene_class"],
                "floor_area_m2": _floor_area(row["label_file"]),
                "label_file": row["label_file"],
                "scene_file": row["scene_file"],
                "bs_count": row.get("bs_count"),
                "ue_count": row.get("ue_count"),
            }
        )
    ranked = [r for r in ranked if np.isfinite(r["floor_area_m2"])]
    return sorted(ranked, key=lambda r: r["floor_area_m2"])


def _sionna_links(scene_key: str) -> dict:
    import h5py  # noqa: PLC0415

    path = SIONNA_SCENE_DIR / f"{scene_key}.h5"
    if not path.is_file():
        raise FileNotFoundError(f"Sionna dataset missing for {scene_key}: {path}")
    with h5py.File(path, "r") as f:
        return {
            "path": str(path),
            "bs_pos": np.asarray(f["bs_pos"][()], dtype=float),
            "ue_pos": np.asarray(f["ue_pos"][()], dtype=float),
            "bs_id": np.asarray(f["bs_id"][()]),
            "ue_id": np.asarray(f["ue_id"][()]),
            "valid": np.asarray(f["valid_mask"][()]),
            "num_links": int(f["cfr"].shape[0]),
        }


def _nearest(points: np.ndarray, target: np.ndarray) -> tuple[int, float]:
    d = np.linalg.norm(points - target[None, :], axis=1)
    idx = int(np.argmin(d))
    return idx, float(d[idx])


def build_selection(case: CaseSpec | None = None, index_path: Path | None = None) -> dict:
    """Resolve the case specification into concrete scene / BS / UE coordinates."""
    case = DEFAULT_CASE if case is None else case
    rows = load_prepared_index(index_path)
    by_key = {row["scene_key"]: row for row in rows}

    scenes_out = []
    for scene_key, size_label in zip(case.scene_keys, case.size_labels, strict=True):
        if scene_key not in by_key:
            raise KeyError(f"{scene_key} is not part of the 339-scene index")
        row = by_key[scene_key]
        label = json.loads(Path(row["label_file"]).read_text())
        links = _sionna_links(scene_key)

        bs_point = label["bs_points"][case.bs_index]
        bs_center = np.asarray(bs_point["position"], dtype=float)
        bs_yaw = float(bs_point["orientation_rad"][0])

        row_idx, dist = _nearest(links["bs_pos"], bs_center)
        if dist > 1e-2:
            raise ValueError(
                f"{scene_key}: BS position not found in Sionna dataset (d={dist:.3f} m)"
            )

        same_bs = np.where(np.linalg.norm(links["bs_pos"] - bs_center[None, :], axis=1) < 1e-3)[0]
        selectable = [i for i in same_bs if bool(links["valid"][i])]
        if len(selectable) < case.num_ue_per_bs:
            raise ValueError(
                f"{scene_key}: only {len(selectable)} valid links for BS {case.bs_index}, "
                f"need {case.num_ue_per_bs}"
            )

        stride = max(1, len(selectable) // case.num_ue_per_bs)
        picked = selectable[::stride][: case.num_ue_per_bs]
        if len(picked) < case.num_ue_per_bs:
            picked = selectable[: case.num_ue_per_bs]

        ue_positions = links["ue_pos"][picked]
        scenes_out.append(
            {
                "scene_key": scene_key,
                "size_label": size_label,
                "scene_class": row["scene_class"],
                "scene_file": row["scene_file"],
                "scene_xml": str(SCENE_ROOT / scene_key / "scene.xml"),
                "label_file": row["label_file"],
                "floor_area_m2": _floor_area(row["label_file"]),
                "bs_index": case.bs_index,
                "bs_id": int(links["bs_id"][row_idx]),
                "bs_center": bs_center.tolist(),
                "bs_yaw_rad": bs_yaw,
                "sionna_h5": links["path"],
                "sionna_rows": [int(i) for i in picked],
                "sionna_ue_ids": [int(links["ue_id"][i]) for i in picked],
                "ue_positions": np.asarray(ue_positions, dtype=float).tolist(),
            }
        )

    return {
        "policy": "differt_crosscheck_case_v1",
        "case": {
            "scene_keys": list(case.scene_keys),
            "size_labels": list(case.size_labels),
            "bs_index": case.bs_index,
            "num_ue_per_bs": case.num_ue_per_bs,
        },
        "scenes": scenes_out,
        "num_links_total": sum(len(s["ue_positions"]) for s in scenes_out),
    }


def choose_cases(num_scene: int = 3) -> CaseSpec:
    """Pick ``num_scene`` scenes spanning the size range (small / median / large)."""
    ranked = rank_scenes(load_prepared_index())
    if num_scene == 1:
        keys = [ranked[len(ranked) // 2]["scene_key"]]
    else:
        picks = np.linspace(0, len(ranked) - 1, num_scene).round().astype(int)
        keys = [ranked[i]["scene_key"] for i in picks]
    labels = ("small", "medium", "large")[: len(keys)]
    return CaseSpec(scene_keys=tuple(keys), size_labels=tuple(labels))
