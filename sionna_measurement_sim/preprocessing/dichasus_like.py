"""Prepare an existing scene split for the frozen ideal-CFR array comparison."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from sionna_measurement_sim.io.label_parser import load_role_topology_from_label
from sionna_measurement_sim.preprocessing.orientation import orient_label_to_map_center


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_dichasus_like_test_set(
    *,
    dataset_root: Path,
    source_project_root: Path,
    output_root: Path,
    split_id: str = "m117_seed2026_scene_level",
    expected_scene_count: int = 339,
) -> dict:
    """Write new labels and runner index, leaving every source file unchanged.

    Use the archived labels/floorplans actually consumed by the original model,
    and recover RT scene paths from the canonical split table. No resampling runs.
    A new output directory is required so earlier preparations stay recoverable.
    """
    dataset_root = Path(dataset_root).resolve()
    source_project_root = Path(source_project_root).resolve()
    output_root = Path(output_root).resolve()
    split_root = dataset_root / "splits" / split_id
    scene_list_path = split_root / "test_scenes.txt"
    scene_ids = [line.strip() for line in scene_list_path.read_text().splitlines() if line.strip()]
    if len(scene_ids) != expected_scene_count or len(set(scene_ids)) != len(scene_ids):
        raise ValueError("test scene count or uniqueness does not match the requested split")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", scene_id) for scene_id in scene_ids):
        raise ValueError("scene IDs must be safe directory names")
    table_path = split_root / "scene_split.jsonl"
    rows = [json.loads(line) for line in table_path.read_text().splitlines() if line.strip()]
    test_rows = [row for row in rows if row["split"] == "test"]
    by_id = {row["scene_id"]: row for row in test_rows}
    if len(test_rows) != len(by_id) or set(by_id) != set(scene_ids):
        raise ValueError("test_scenes.txt and scene_split.jsonl disagree")
    other_ids = {row["scene_id"] for row in rows if row["split"] != "test"}
    if set(scene_ids) & other_ids:
        raise ValueError("test scenes overlap another split")
    manifest_path = split_root / "split_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    scene_digest = _sha256(scene_list_path)
    if scene_digest != manifest["scene_list_sha256"]["test_scenes.txt"]:
        raise ValueError("canonical test scene list checksum mismatch")

    prepared = []
    for order, scene_id in enumerate(scene_ids):
        row = by_id[scene_id]
        assets = dataset_root / "assets" / scene_id
        label_path = assets / "label_panel_0p5.json"
        meta_path = assets / "meta.json"
        scene_path = (source_project_root / row["scene_file"]).resolve()
        h5_path = dataset_root / row["h5_path"]
        for required in (label_path, meta_path, scene_path, h5_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        label = json.loads(label_path.read_text())
        if (len(label["bs_points"]), len(label["ue_points"])) != (
            row["bs_count"], row["ue_count"]
        ):
            raise ValueError(f"archived label counts disagree with split table for {scene_id}")
        derived, center = orient_label_to_map_center(label, json.loads(meta_path.read_text()))
        provenance = {
            "policy": "map_center_xy_upright_bs_fixed_ue_v1",
            "map_center_xy_m": center,
            "source_label": str(label_path),
            "source_label_sha256": _sha256(label_path),
            "source_floorplan_meta_sha256": _sha256(meta_path),
        }
        derived["orientation_preprocessing"] = provenance
        destination = output_root / "labels" / scene_id / "label_panel_0p5_bs_center_yaw.json"
        entry = {
            "index": order,
            "index_id": f"{order:06d}",
            "scene_key": scene_id,
            "scene_class": row["scene_class"],
            "split": "test",
            "label_name": destination.name,
            "label_tag": "panel0p5_bs_center_yaw_v1",
            "label_file": str(destination),
            "scene_file": str(scene_path),
            "bs_count": row["bs_count"],
            "ue_count": row["ue_count"],
            "link_count": row["num_links"],
            "source_scene_index": row["source_scene_index"],
            "source_dataset_h5": str(h5_path),
            **provenance,
        }
        prepared.append((derived, entry))

    # All inputs and split membership are checked before creating any outputs.
    output_root.mkdir(parents=True, exist_ok=False)
    index_path = output_root / "test_scene_index.jsonl"
    with index_path.open("x", encoding="utf-8") as index:
        for derived, entry in prepared:
            destination = Path(entry["label_file"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(
                json.dumps(derived, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
            )
            topology = load_role_topology_from_label(
                destination, max_bs=entry["bs_count"], max_ue=entry["ue_count"]
            )
            assert topology.bs_orientation_rad is not None
            assert topology.ue_orientation_rad is not None
            entry["prepared_label_sha256"] = _sha256(destination)
            index.write(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "policy": "map_center_xy_upright_bs_fixed_ue_v1",
        "dataset_root": str(dataset_root),
        "split_id": split_id,
        "source_test_scenes": str(scene_list_path),
        "source_test_scenes_sha256": scene_digest,
        "source_scene_table_sha256": _sha256(table_path),
        "index_path": str(index_path),
        "index_sha256": _sha256(index_path),
        "scene_count": len(prepared),
        "scene_class_counts": dict(Counter(entry["scene_class"] for _, entry in prepared)),
        "raw_link_count": sum(entry["link_count"] for _, entry in prepared),
        "bs_count": sum(entry["bs_count"] for _, entry in prepared),
        "ue_count": sum(entry["ue_count"] for _, entry in prepared),
        "source_files_modified": False,
        "simulation_started": False,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
