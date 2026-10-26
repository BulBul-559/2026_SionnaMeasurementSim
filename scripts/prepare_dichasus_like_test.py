"""Prepare upright BS orientations for the existing EnvMIMO-3K test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sionna_measurement_sim.preprocessing.dichasus_like import prepare_dichasus_like_test_set


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-project-root", type=Path, default=Path.cwd())
    parser.add_argument("--split-id", default="m117_seed2026_scene_level")
    parser.add_argument("--expected-scene-count", type=int, default=339)
    args = parser.parse_args()
    summary = prepare_dichasus_like_test_set(
        dataset_root=args.dataset_root,
        source_project_root=args.source_project_root,
        output_root=args.output_root,
        split_id=args.split_id,
        expected_scene_count=args.expected_scene_count,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
