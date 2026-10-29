#!/usr/bin/env python
"""Collapse the per-scene run logs and the comparison into one markdown report."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ORDER_RE = re.compile(
    r"\[order (?P<order>\d+)\] candidates=(?P<cand>\d+) valid=(?P<valid>\d+) "
    r"trace=(?P<trace>[\d.]+)s fields=(?P<fields>[\d.]+)s"
)
SCENE_RE = re.compile(r"^\[(?P<key>\w+)\] links=(?P<links>\d+).*tris=(?P<tris>\d+).*paths=(?P<paths>\d+)")


def parse_log(path: Path) -> dict:
    orders = []
    scene = None
    for line in path.read_text(errors="ignore").splitlines():
        m = ORDER_RE.search(line)
        if m:
            orders.append({k: (int(v) if k in ("order", "cand", "valid") else float(v)) for k, v in m.groupdict().items()})
        m = SCENE_RE.search(line)
        if m:
            scene = {k: int(v) if k != "key" else v for k, v in m.groupdict().items()}
    return {"orders": orders, "scene": scene}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", nargs="+", required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="DiffeRT cross-check report")
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text()) if args.summary.is_file() else {}
    lines = [f"# {args.title}", ""]

    if summary:
        lines += [
            "## 与 Sionna 339 的逐链路相似度",
            "",
            f"- links: {summary.get('num_links')}",
            "",
            "| 指标 | mean | median | min | max |",
            "|---|---:|---:|---:|---:|",
        ]
        for key, stats in summary.get("overall", {}).items():
            lines.append(
                f"| `{key}` | {stats['mean']:.3f} | {stats['median']:.3f} | {stats['min']:.3f} | {stats['max']:.3f} |"
            )
        lines += ["", "### 逐场景", "", "| 场景 | 规模 | 链路 | cos_sim | NMSE(dB) | PDP 相关 | 功率偏差(dB) |", "|---|---|---:|---:|---:|---:|---:|"]
        for row in summary.get("per_scene", []):
            lines.append(
                f"| {row['scene_key']} | {row['size_label']} | {row['num_links']} | "
                f"{row['mean_cos_sim']:.3f} | {row['mean_nmse_complex_db']:.2f} | "
                f"{row['mean_pdp_corr']:.3f} | {row['mean_power_offset_db']:+.2f} |"
            )

    lines += ["", "## 路径与耗时", "", "| 场景 | 阶 | 候选 | 有效 | 追迹(s) | 场求解(s) |", "|---|---:|---:|---:|---:|---:|"]
    for log in args.logs:
        data = parse_log(Path(log))
        key = Path(log).stem
        for row in data["orders"]:
            lines.append(
                f"| {key} | {row['order']} | {row['cand']} | {row['valid']} | {row['trace']:.1f} | {row['fields']:.1f} |"
            )

    args.out.write_text("\n".join(lines) + "\n")
    print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
