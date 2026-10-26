from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from sionna_measurement_sim.adapters.sionna_rt.rt_solver import SionnaRTConfig, _build_scene
from sionna_measurement_sim.app.scene_index import _scene_config, read_scene_index
from sionna_measurement_sim.domain.antenna import AntennaSpec
from sionna_measurement_sim.domain.frequency import FrequencyGrid
from sionna_measurement_sim.domain.topology import resolve_link_roles, resolve_role_topology
from sionna_measurement_sim.io.label_parser import load_role_topology_from_label
from sionna_measurement_sim.preprocessing.dichasus_like import prepare_dichasus_like_test_set
from sionna_measurement_sim.preprocessing.orientation import orient_label_to_map_center
from sionna_measurement_sim.rt.truth_pipeline import ShardSpec, _split_shard_spec_for_fallback


def _label() -> dict:
    return {
        "bs_points": [
            {"label": "BS0", "position": [0, 2, 3]},
            {"label": "BS1", "position": [8, 2, 3], "orientation_rad": [0.7, 0, 0]},
            {"label": "BS2", "position": [4, 2, 3], "orientation_rad": [0, 0, 0]},
        ],
        "ue_points": [
            {"label": f"UE{i}", "x": i, "y": 1, "z": 1.6,
             "orientation_rad": [0.1 * i, 0, 0]} for i in range(4)
        ],
        "groups": [{"name": "unchanged", "metadata": [1, 2, 3]}],
    }


def _write_label(path: Path, label: dict) -> Path:
    path.write_text(json.dumps(label), encoding="utf-8")
    return path


@pytest.mark.parametrize("direction", ["uplink", "downlink"])
def test_orientation_follows_role_selection_and_reordering(tmp_path: Path, direction: str):
    path = _write_label(tmp_path / "label.json", _label())
    roles = load_role_topology_from_label(path, bs_indices=[2, 0, 1], ue_indices=[3, 1])
    assert roles.bs_orientation_rad == ((0, 0, 0), None, (0.7, 0, 0))
    np.testing.assert_allclose(roles.ue_orientation_rad, [[0.3, 0, 0], [0.1, 0, 0]])
    topology = resolve_role_topology(roles, resolve_link_roles(direction))
    expected_tx, expected_rx = (
        (roles.ue_orientation_rad, roles.bs_orientation_rad) if direction == "uplink"
        else (roles.bs_orientation_rad, roles.ue_orientation_rad)
    )
    assert topology.tx_orientation_rad == expected_tx
    assert topology.rx_orientation_rad == expected_rx
    np.testing.assert_array_equal(roles.bs_global_indices, [2, 0, 1])
    np.testing.assert_array_equal(roles.ue_global_indices, [3, 1])


def test_missing_orientation_preserves_legacy_and_shards_keep_global_rows(tmp_path: Path):
    label = _label()
    for point in label["bs_points"]:
        point.pop("orientation_rad", None)
    path = _write_label(tmp_path / "label.json", label)
    full = load_role_topology_from_label(path, max_bs=3, max_ue=4)
    shard = load_role_topology_from_label(path, max_bs=3, ue_start=2, ue_count=2)
    assert full.bs_orientation_rad is None and shard.bs_orientation_rad is None
    assert shard.ue_orientation_rad == full.ue_orientation_rad[2:]
    assert shard.bs_labels == full.bs_labels


@pytest.mark.parametrize("bad", [[0, 1], [0, 0, float("nan")], [float("inf"), 0, 0], "bad"])
def test_malformed_orientation_rejected_before_rt(tmp_path: Path, bad):
    label = _label()
    label["bs_points"][0]["orientation_rad"] = bad
    path = _write_label(tmp_path / "label.json", label)
    with pytest.raises(ValueError):
        load_role_topology_from_label(path, max_bs=3, max_ue=4)


def test_rt_applies_explicit_zero_and_preserves_legacy_tx_look_at(tmp_path: Path, monkeypatch):
    class Node:
        def __init__(self, *, name, position):
            self.name = name
            self.position = position
            self.orientation = [0.0, 0.0, 0.0]

        def look_at(self, target):
            self.orientation = [0.8, 0.2, 0.0]

    class Scene:
        def add(self, node):
            pass

    module = ModuleType("sionna.rt")
    module.PlanarArray = lambda **kwargs: kwargs
    module.Receiver = Node
    module.Transmitter = Node
    module.PathSolver = object
    module.load_scene = lambda *args, **kwargs: Scene()
    monkeypatch.setitem(sys.modules, "sionna", ModuleType("sionna"))
    monkeypatch.setitem(sys.modules, "sionna.rt", module)
    label = _label()
    # UE0 has an explicit zero; UE1 falls back to the old look_at mode.
    label["ue_points"][1].pop("orientation_rad")
    roles = load_role_topology_from_label(
        _write_label(tmp_path / "label.json", label), max_bs=3, max_ue=2
    )
    _, _, tx_angles, rx_angles = _build_scene(
        resolve_role_topology(roles, resolve_link_roles("uplink")),
        AntennaSpec(
            tx_orientation_mode="look_at_centroid", rx_orientation_rad=(1.2, 0, 0)
        ),
        FrequencyGrid.from_center_bandwidth(3.438e9, 23.04e6, 768),
        SionnaRTConfig(scene_file=tmp_path / "scene.xml"),
    )
    np.testing.assert_allclose(tx_angles, [[0, 0, 0], [0.8, 0.2, 0]])
    np.testing.assert_allclose(rx_angles, [[1.2, 0, 0], [0.7, 0, 0], [0, 0, 0]])


def test_preprocess_uses_world_map_center_and_keeps_input_and_groups():
    source = _label()
    original = copy.deepcopy(source)
    derived, center = orient_label_to_map_center(
        source, {"origin_xy_m": [0, 0], "extent_xy_m": [8, 4]},
        bs_fallback_rad=(0.3, 0, 0),
    )
    assert source == original
    assert center == [4, 2]
    np.testing.assert_allclose(
        [point["orientation_rad"] for point in derived["bs_points"]],
        [[0, 0, 0], [np.pi, 0, 0], [0.3, 0, 0]],
    )
    assert all(point["orientation_rad"] == [0, 0, 0] for point in derived["ue_points"])
    assert derived["groups"] == source["groups"]
    shifted = copy.deepcopy(source)
    for point in shifted["bs_points"]:
        point["position"][0] += 100
        point["position"][1] -= 20
    shifted, shifted_center = orient_label_to_map_center(
        shifted, {"origin_xy_m": [100, -20], "extent_xy_m": [8, 4]},
        bs_fallback_rad=(0.3, 0, 0),
    )
    assert shifted_center == [104, -18]
    assert [p["orientation_rad"] for p in shifted["bs_points"]] == [
        p["orientation_rad"] for p in derived["bs_points"]
    ]


@pytest.mark.parametrize("extent", [[0, 4], [-1, 4], [float("nan"), 4], [8, 4, 3]])
def test_preprocess_rejects_invalid_extent(extent):
    with pytest.raises(ValueError):
        orient_label_to_map_center(_label(), {"origin_xy_m": [0, 0], "extent_xy_m": extent})


def _fake_split(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "dataset"
    split = root / "splits" / "m117_seed2026_scene_level"
    split.mkdir(parents=True)
    scene_ids = ["front3d_0002", "front3d_0001"]
    source_project = tmp_path / "source"
    rows = []
    for scene_id in scene_ids:
        assets = root / "assets" / scene_id
        assets.mkdir(parents=True)
        (assets / "label_panel_0p5.json").write_text(json.dumps(_label()))
        (assets / "meta.json").write_text(
            json.dumps({"origin_xy_m": [0, 0], "extent_xy_m": [8, 4]})
        )
        scene = source_project / "data" / scene_id / "scene.xml"
        scene.parent.mkdir(parents=True)
        scene.write_text("<scene />")
        h5 = root / "scenes" / f"{scene_id}.h5"
        h5.parent.mkdir(exist_ok=True)
        h5.touch()
        rows.append({
            "scene_id": scene_id, "split": "test", "scene_class": "small_room",
            "bs_count": 3, "ue_count": 4, "num_links": 12,
            "source_scene_index": int(scene_id[-4:]),
            "scene_file": str(scene.relative_to(source_project)),
            "h5_path": str(h5.relative_to(root)),
        })
    scene_list = split / "test_scenes.txt"
    scene_list.write_text("\n".join(scene_ids) + "\n")
    (split / "scene_split.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    digest = hashlib.sha256(scene_list.read_bytes()).hexdigest()
    (split / "split_manifest.json").write_text(
        json.dumps({"scene_list_sha256": {"test_scenes.txt": digest}})
    )
    return root, source_project


def test_preparation_preserves_split_order_sources_and_runner_label_path(tmp_path: Path):
    root, source = _fake_split(tmp_path)
    original = (root / "assets/front3d_0002/label_panel_0p5.json").read_bytes()
    output = tmp_path / "prepared"
    result = prepare_dichasus_like_test_set(
        dataset_root=root, source_project_root=source, output_root=output, expected_scene_count=2
    )
    entries = read_scene_index(Path(result["index_path"]))
    assert [row["scene_key"] for row in entries] == ["front3d_0002", "front3d_0001"]
    assert result["raw_link_count"] == 24
    assert (root / "assets/front3d_0002/label_panel_0p5.json").read_bytes() == original
    template = {"input": {"label_file": "old.json"}, "output": {}}
    cfg = _scene_config(template, entries[0], tmp_path / "run")
    assert cfg["input"]["label_file"] == entries[0]["label_file"]
    roles = load_role_topology_from_label(Path(entries[0]["label_file"]), max_bs=3, max_ue=4)
    np.testing.assert_allclose(roles.bs_orientation_rad[1], [np.pi, 0, 0])
    with pytest.raises(FileExistsError):
        prepare_dichasus_like_test_set(
            dataset_root=root, source_project_root=source, output_root=output,
            expected_scene_count=2,
        )


def test_preparation_rejects_bad_split_before_writing(tmp_path: Path):
    root, source = _fake_split(tmp_path)
    with pytest.raises(ValueError, match="count"):
        prepare_dichasus_like_test_set(
            dataset_root=root, source_project_root=source, output_root=tmp_path / "prepared"
        )
    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize("indices", [None, (3, 1, 2, 0)])
def test_fallback_children_keep_selected_device_orientations(tmp_path: Path, indices):
    label = _label()
    path = _write_label(tmp_path / "label.json", label)
    spec = ShardSpec(
        shard_index=0, shard_count=1, ue_count=4, ue_indices=indices, bs_indices=(2, 1)
    )
    children = _split_shard_spec_for_fallback(spec, "cuda_oom")
    gathered = []
    for child in children:
        roles = load_role_topology_from_label(
            path, ue_start=child.ue_start, ue_count=child.ue_count,
            ue_indices=child.ue_indices, bs_indices=child.bs_indices,
        )
        assert roles.bs_orientation_rad == ((0, 0, 0), (0.7, 0, 0))
        gathered.extend(roles.ue_orientation_rad)
    order = indices if indices is not None else tuple(range(4))
    np.testing.assert_allclose(gathered, [label["ue_points"][i]["orientation_rad"] for i in order])
