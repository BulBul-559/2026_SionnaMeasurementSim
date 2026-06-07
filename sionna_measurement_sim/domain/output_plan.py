"""Output profile planning for RT truth pipelines."""

from __future__ import annotations

from dataclasses import dataclass

from sionna_measurement_sim.domain.constants import (
    FULL_CONTRACT_NAME,
    OUTPUT_PROFILES,
    RT_LABELS_CONTRACT_NAME,
)


@dataclass(frozen=True)
class RTOutputPlan:
    """Concrete compute/write plan derived from ``output.profile``."""

    profile: str = "full"
    contract_name: str = FULL_CONTRACT_NAME
    compute_cfr: bool = True
    compute_cir: bool = True
    compute_path_samples: bool = True
    compute_nlos_truth: bool = True
    write_full_contract: bool = True
    write_compact_link_labels: bool = False


def build_rt_output_plan(profile: str) -> RTOutputPlan:
    """Return the concrete RT output plan for a validated profile string."""

    profile = str(profile)
    if profile not in OUTPUT_PROFILES:
        allowed = ", ".join(OUTPUT_PROFILES)
        msg = f"output.profile must be one of: {allowed}"
        raise ValueError(msg)
    if profile == "rt_labels_only":
        return RTOutputPlan(
            profile=profile,
            contract_name=RT_LABELS_CONTRACT_NAME,
            compute_cfr=False,
            compute_cir=False,
            compute_path_samples=False,
            compute_nlos_truth=False,
            write_full_contract=False,
            write_compact_link_labels=True,
        )
    # ``rt_lite`` keeps the full HDF5 contract in this phase, but the CLI/pipeline
    # uses the profile to disable PHY/ranging/spectrum/viz/calibration presets.
    return RTOutputPlan(profile=profile)
