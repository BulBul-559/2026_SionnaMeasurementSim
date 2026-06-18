from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import yaml

from sionna_measurement_sim.app import scene_index as scene_index_module
from sionna_measurement_sim.app.scene_index import (
    build_scene_index,
    read_scene_index,
    run_scene_index,
)


def test_build_scene_index_stratifies_and_writes_counts(tmp_path: Path):
    source_root = tmp_path / "front3d_full"
    _write_fake_dataset(source_root, "small_room", count=4, bs=2, ue=10)
    _write_fake_dataset(source_root, "normal_room", count=8, bs=3, ue=20)

    result = build_scene_index(
        source_root=source_root,
        output_path=tmp_path / "index.jsonl",
        classes=("small_room", "normal_room"),
        total_count=6,
        label_name="label_panel_0p5.json",
        seed=7,
    )

    entries = read_scene_index(result.index_path)
    counts = Counter(entry["scene_class"] for entry in entries)
    assert result.total_count == 6
    assert result.selected_counts == {"small_room": 2, "normal_room": 4}
    assert counts == {"small_room": 2, "normal_room": 4}
    assert [entry["index"] for entry in entries] == list(range(6))
    assert all(entry["label_file"].endswith("label/label_panel_0p5.json") for entry in entries)
    assert all(entry["scene_file"].endswith("scene.xml") for entry in entries)
    assert all(entry["link_count"] == entry["bs_count"] * entry["ue_count"] for entry in entries)
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["total_count"] == 6


def test_run_scene_index_dry_run_writes_per_scene_config(tmp_path: Path):
    index_path = tmp_path / "index.jsonl"
    label_file = tmp_path / "scene" / "label" / "label_panel_0p5.json"
    scene_file = tmp_path / "scene" / "scene.xml"
    label_file.parent.mkdir(parents=True)
    label_file.write_text(json.dumps({"bs_points": [{}], "ue_points": [{}, {}]}), encoding="utf-8")
    scene_file.write_text("<scene />", encoding="utf-8")
    index_path.write_text(
        json.dumps(
            {
                "index": 0,
                "scene_key": "front3d_0000",
                "scene_class": "normal_room",
                "label_name": "label_panel_0p5.json",
                "label_tag": "panel0p5",
                "label_file": label_file.as_posix(),
                "scene_file": scene_file.as_posix(),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "template.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "input": {
                    "label_file": "placeholder.json",
                    "scene_file": "placeholder.xml",
                    "scene_id": "placeholder",
                },
                "output": {"root_dir": "outputs/placeholder"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = run_scene_index(
        index_path=index_path,
        config_path=config_path,
        output_root=tmp_path / "runs",
        dry_run=True,
    )

    assert result.planned == 1
    manifest_entry = json.loads(result.manifest_path.read_text(encoding="utf-8").strip())
    assert manifest_entry["status"] == "planned"
    generated_config = yaml.safe_load(Path(manifest_entry["generated_config"]).read_text())
    assert generated_config["input"]["label_file"] == label_file.as_posix()
    assert generated_config["input"]["scene_file"] == scene_file.as_posix()
    assert generated_config["input"]["scene_id"] == "front3d_0000"
    assert generated_config["output"]["root_dir"].endswith(
        "runs/000000_front3d_0000_panel0p5_cfr_truth_srs64prb"
    )


def test_run_scene_index_pipeline_shards_fills_free_gpus_across_scenes(
    tmp_path: Path,
    monkeypatch,
):
    index_path = tmp_path / "index.jsonl"
    scene0_label, scene0_xml = _write_scene_fixture(tmp_path, "front3d_0000", ue=2)
    scene1_label, scene1_xml = _write_scene_fixture(tmp_path, "front3d_0001", ue=4)
    index_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "index": 0,
                        "scene_key": "front3d_0000",
                        "scene_class": "small_room",
                        "label_name": "label_panel_0p5.json",
                        "label_tag": "panel0p5",
                        "label_file": scene0_label.as_posix(),
                        "scene_file": scene0_xml.as_posix(),
                    },
                    sort_keys=True,
                ),
                json.dumps(
                    {
                        "index": 1,
                        "scene_key": "front3d_0001",
                        "scene_class": "normal_room",
                        "label_name": "label_panel_0p5.json",
                        "label_tag": "panel0p5",
                        "label_file": scene1_label.as_posix(),
                        "scene_file": scene1_xml.as_posix(),
                    },
                    sort_keys=True,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "template.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "runtime": {"device": "cuda", "seed": 1},
                "input": {
                    "label_file": "placeholder.json",
                    "scene_file": "placeholder.xml",
                    "scene_id": "placeholder",
                    "max_bs": 1,
                    "max_ue": 10,
                },
                "output": {
                    "root_dir": "outputs/placeholder",
                    "products": ["cfr_truth"],
                    "sharding": {
                        "enabled": True,
                        "axis": "ue",
                        "shard_size": 1,
                        "parallel_workers": 4,
                        "gpu_ids": [0, 1, 2, 3],
                        "visualization_mode": "none",
                        "gpu_scheduler": {
                            "enabled": True,
                            "free_memory_threshold": 0.6,
                            "scan_interval_s": 0.01,
                            "cross_scene_pipeline": True,
                        },
                        "fallback": {"enabled": False},
                    },
                },
                "phy": {"enabled": False},
                "motion": {"enabled": False},
                "visualization": {"enabled": False},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    submitted: list[tuple[str, int, int | None]] = []

    class _ImmediateFuture:
        def __init__(self, value):
            self._value = value

        def done(self):
            return True

        def result(self):
            return self._value

    class _ImmediateExecutor:
        def __init__(self, max_workers: int):
            self.max_workers = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def submit(self, fn, config, spec, gpu_id):
            submitted.append((str(config.scene_id), int(spec.shard_index), gpu_id))
            return _ImmediateFuture(fn(config, spec, gpu_id))

    def fake_run(config, spec, gpu_id):
        del config
        shard_id = spec.shard_id or f"{spec.shard_index:03d}"
        return {
            "results": [
                {
                    "shard_id": shard_id,
                    "shard_index": spec.shard_index,
                    "global_ue_indices": [int(spec.ue_start)],
                    "result_h5": "",
                    "manifest": "",
                }
            ],
            "attempts": [
                {
                    "shard_id": shard_id,
                    "shard_index": spec.shard_index,
                    "status": "succeeded",
                    "gpu_id": gpu_id,
                }
            ],
        }

    monkeypatch.setattr(scene_index_module, "ProcessPoolExecutor", _ImmediateExecutor)
    monkeypatch.setattr(
        scene_index_module,
        "wait",
        lambda futures, timeout=None, return_when=None: (set(futures), set()),
    )
    monkeypatch.setattr(
        scene_index_module.truth_pipeline,
        "_query_gpu_free_memory_ratios",
        lambda gpu_ids: {gpu_id: 0.9 for gpu_id in gpu_ids},
    )
    monkeypatch.setattr(
        scene_index_module.truth_pipeline,
        "_run_shard_spec_with_fallback",
        fake_run,
    )

    result = run_scene_index(
        index_path=index_path,
        config_path=config_path,
        output_root=tmp_path / "runs",
    )

    assert result.completed == 2
    assert result.failed == 0
    assert [item[0] for item in submitted[:4]] == [
        "front3d_0000",
        "front3d_0000",
        "front3d_0001",
        "front3d_0001",
    ]
    run_manifest_lines = [
        json.loads(line)
        for line in result.manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [line["status"] for line in run_manifest_lines] == ["completed", "completed"]
    scene0_manifest = json.loads(
        (
            tmp_path
            / "runs"
            / "runs"
            / "000000_front3d_0000_panel0p5_cfr_truth_srs64prb"
            / "manifest"
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert scene0_manifest["phase"] == "sharded_run_full_scene_index_pipeline"
    scheduler = scene0_manifest["sharding"]["gpu_scheduler"]
    assert scheduler["cross_scene_pipeline"] is True
    assert scheduler["candidate_gpu_ids"] == [0, 1, 2, 3]
    assert scheduler["scheduled_count"] == 2


def _write_fake_dataset(
    source_root: Path,
    class_name: str,
    *,
    count: int,
    bs: int,
    ue: int,
) -> None:
    manifest_path = source_root / "split_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    start = (
        len(manifest_path.read_text(encoding="utf-8").splitlines())
        if manifest_path.exists()
        else 0
    )
    with manifest_path.open("a", encoding="utf-8") as manifest:
        for offset in range(count):
            scene_index = start + offset
            scene_dir = f"front3d_{scene_index:04d}"
            scene_root = source_root / class_name / scene_dir
            label_dir = scene_root / "label"
            label_dir.mkdir(parents=True)
            (scene_root / "scene.xml").write_text("<scene />", encoding="utf-8")
            (label_dir / "label_panel_0p5.json").write_text(
                json.dumps({"bs_points": [{}] * bs, "ue_points": [{}] * ue}),
                encoding="utf-8",
            )
            manifest.write(
                json.dumps(
                    {
                        "split_class": class_name,
                        "scene_dir": scene_dir,
                        "scene_index": scene_index,
                        "split": "train",
                    },
                    sort_keys=True,
                )
                + "\n"
            )


def _write_scene_fixture(tmp_path: Path, scene_key: str, *, ue: int) -> tuple[Path, Path]:
    scene_root = tmp_path / "dataset" / scene_key
    label_dir = scene_root / "label"
    label_dir.mkdir(parents=True)
    scene_file = scene_root / "scene.xml"
    label_file = label_dir / "label_panel_0p5.json"
    scene_file.write_text("<scene />", encoding="utf-8")
    label_file.write_text(
        json.dumps({"bs_points": [{}], "ue_points": [{}] * ue}),
        encoding="utf-8",
    )
    return label_file, scene_file
