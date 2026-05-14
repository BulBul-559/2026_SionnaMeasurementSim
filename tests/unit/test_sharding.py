import json
from pathlib import Path

import numpy as np

from sionna_measurement_sim.config.schema import OutputShardingConfig
from sionna_measurement_sim.io.label_parser import count_topology_points, load_topology_from_label
from sionna_measurement_sim.rt.truth_pipeline import RTTruthRunConfig, _build_shard_specs


def test_load_topology_from_label_supports_rx_range(tmp_path: Path):
    label_path = _write_label(tmp_path)

    topology = load_topology_from_label(
        label_path,
        max_tx=2,
        max_rx=1,
        rx_start=2,
        rx_count=2,
    )

    assert topology.tx_labels == ("BS0", "BS1")
    assert topology.rx_labels == ("UE2", "UE3")
    np.testing.assert_allclose(
        topology.rx_positions_m,
        np.array(
            [
                [12.0, 22.0, 1.0],
                [13.0, 23.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )


def test_load_topology_from_label_keeps_max_count_as_cap(tmp_path: Path):
    label_path = _write_label(tmp_path)

    topology = load_topology_from_label(label_path, max_tx=10, max_rx=10)

    assert topology.tx_labels == ("BS0", "BS1", "BS2")
    assert topology.rx_labels == ("UE0", "UE1", "UE2", "UE3")


def test_load_topology_from_label_supports_explicit_rx_and_tx_indices(tmp_path: Path):
    label_path = _write_label(tmp_path)

    topology = load_topology_from_label(
        label_path,
        max_tx=1,
        max_rx=1,
        rx_indices=(3, 1),
        tx_indices=(2, 0),
    )

    assert topology.tx_labels == ("BS2", "BS0")
    assert topology.rx_labels == ("UE3", "UE1")
    np.testing.assert_allclose(
        topology.tx_positions_m,
        np.array(
            [
                [2.0, 3.0, 2.0],
                [0.0, 1.0, 2.0],
            ],
            dtype=np.float32,
        ),
    )


def test_count_topology_points_reads_available_label_size(tmp_path: Path):
    label_path = _write_label(tmp_path)

    assert count_topology_points(label_path) == (3, 4)


def test_build_shard_specs_splits_8884_ues_into_nine_shards(tmp_path: Path):
    label_path = _write_label_with_counts(tmp_path, tx_count=6, rx_count=8884)
    config = RTTruthRunConfig(
        label_file=label_path,
        scene_file=tmp_path / "scene.xml",
        output_dir=tmp_path / "out",
        max_tx=6,
        max_rx=8884,
        output_sharding_config=OutputShardingConfig(
            enabled=True,
            shard_size=1000,
        ),
    )

    specs = _build_shard_specs(config)

    assert len(specs) == 9
    assert [spec.rx_start for spec in specs] == [
        0,
        1000,
        2000,
        3000,
        4000,
        5000,
        6000,
        7000,
        8000,
    ]
    assert [spec.rx_count for spec in specs] == [
        1000,
        1000,
        1000,
        1000,
        1000,
        1000,
        1000,
        1000,
        884,
    ]


def test_build_shard_specs_caps_to_available_ues_and_normalizes_ue_axis(tmp_path: Path):
    label_path = _write_label(tmp_path)
    config = RTTruthRunConfig(
        label_file=label_path,
        scene_file=tmp_path / "scene.xml",
        output_dir=tmp_path / "out",
        max_tx=10,
        max_rx=10,
        output_sharding_config=OutputShardingConfig(
            enabled=True,
            axis="ue",
            shard_size=3,
        ),
    )

    specs = _build_shard_specs(config)

    assert len(specs) == 2
    assert [spec.axis for spec in specs] == ["rx", "rx"]
    assert [spec.rx_start for spec in specs] == [0, 3]
    assert [spec.rx_count for spec in specs] == [3, 1]


def _write_label(tmp_path: Path) -> Path:
    label_path = tmp_path / "label.json"
    label_path.write_text(
        json.dumps(
            {
                "groups": [
                    {
                        "bs_points": [
                            {"x": 0.0, "y": 1.0, "z": 2.0, "label": "BS0"},
                            {"x": 1.0, "y": 2.0, "z": 2.0, "label": "BS1"},
                            {"x": 2.0, "y": 3.0, "z": 2.0, "label": "BS2"},
                        ],
                        "ue_points": [
                            {"x": 10.0, "y": 20.0, "z": 1.0, "label": "UE0"},
                            {"x": 11.0, "y": 21.0, "z": 1.0, "label": "UE1"},
                            {"x": 12.0, "y": 22.0, "z": 1.0, "label": "UE2"},
                            {"x": 13.0, "y": 23.0, "z": 1.0, "label": "UE3"},
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return label_path


def _write_label_with_counts(tmp_path: Path, *, tx_count: int, rx_count: int) -> Path:
    label_path = tmp_path / f"label_{tx_count}_{rx_count}.json"
    label_path.write_text(
        json.dumps(
            {
                "groups": [
                    {
                        "bs_points": [
                            {
                                "x": float(index),
                                "y": float(index + 1),
                                "z": 2.0,
                                "label": f"BS{index}",
                            }
                            for index in range(tx_count)
                        ],
                        "ue_points": [
                            {
                                "x": float(index),
                                "y": float(index + 10),
                                "z": 1.0,
                                "label": f"UE{index}",
                            }
                            for index in range(rx_count)
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return label_path
