"""Manifest writer for run directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_manifest(path: str | Path, manifest: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return output_path
