"""Write a minimal Phase 1 HDF5 fixture under outputs/."""

from __future__ import annotations

from pathlib import Path

from sionna_measurement_sim.domain.results import create_phase1_minimal_result
from sionna_measurement_sim.io.hdf5_writer import write_measurement_result
from sionna_measurement_sim.io.schema_validator import validate_hdf5_contract


def main() -> int:
    output_path = Path("outputs/phase1_schema/results.h5")
    write_measurement_result(output_path, create_phase1_minimal_result())
    validate_hdf5_contract(output_path)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
