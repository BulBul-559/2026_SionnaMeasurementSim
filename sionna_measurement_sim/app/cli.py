"""Command line entry point for SionnaMeasurementSim."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from sionna_measurement_sim import __version__
from sionna_measurement_sim.preflight.system import collect_basic_environment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sionna-measurement-sim",
        description="SionnaMeasurementSim command line interface.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "preflight",
        help="Print basic local environment information.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "preflight":
        for key, value in collect_basic_environment().items():
            print(f"{key}: {value}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
