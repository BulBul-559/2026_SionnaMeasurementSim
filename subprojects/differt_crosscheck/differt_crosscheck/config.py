"""Frozen configuration shared by every DiffeRT cross-check artefact.

The numeric contract mirrors the Sionna-side frozen contract documented in
``docs/proposals/dichasus_like_cfr_v1.md`` (centre frequency, NR grid, BS panel,
UE antenna, polarisation). Keeping both sides on the same contract is what makes
a link-for-link comparison meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------------------
# External locations (read-only inputs)
# --------------------------------------------------------------------------------------

SCENE_ROOT = Path("/data/sunmeiyuan/projects/SceneGen/front3d_full_all_6813_20260617_215824")

SIONNA_OUTPUT_ROOT = Path("/data/sunmeiyuan/projects/sionna/outputs")
SIONNA_339_ROOT = SIONNA_OUTPUT_ROOT / "front3d_339_dichasus_like_cfr_v1"
SIONNA_SCENE_DIR = SIONNA_339_ROOT / "dataset" / "scenes"
SIONNA_PREPARED_INDEX = SIONNA_339_ROOT / "prepared" / "test_scene_index.jsonl"

# --------------------------------------------------------------------------------------
# Where this cross-check writes (sibling of the Sionna outputs, clearly separated)
# --------------------------------------------------------------------------------------

OUTPUT_ROOT = SIONNA_OUTPUT_ROOT / "differt_rt_crosscheck_v1"


def output_paths(root: Path | None = None) -> dict[str, Path]:
    """Return the standard artefact layout below ``root``."""
    root = OUTPUT_ROOT if root is None else Path(root)
    return {
        "root": root,
        "links": root / "links",
        "dataset": root / "dataset",
        "scenes": root / "dataset" / "scenes",
        "compare": root / "compare",
        "figures": root / "compare" / "figures",
        "logs": root / "logs",
    }


def ensure_output_dirs(root: Path | None = None) -> dict[str, Path]:
    paths = output_paths(root)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


# --------------------------------------------------------------------------------------
# Case definition
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseSpec:
    """Three scenes (small / medium / large) x one BS x ten UEs."""

    scene_keys: tuple[str, ...] = ("front3d_3682", "front3d_2508", "front3d_5704")
    size_labels: tuple[str, ...] = ("small", "medium", "large")
    bs_index: int = 0
    num_ue_per_bs: int = 10

    def __post_init__(self) -> None:
        if len(self.scene_keys) != len(self.size_labels):
            raise ValueError("scene_keys and size_labels must have the same length")


# --------------------------------------------------------------------------------------
# Frequency grid
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FrequencyGrid:
    """NR-style uniform grid: 768 subcarriers, 30 kHz spacing, centred on 3.438 GHz."""

    center_frequency_hz: float = 3_438_000_000.0
    bandwidth_hz: float = 23_040_000.0
    num_subcarriers: int = 768
    subcarrier_spacing_hz: float = 30_000.0

    def __post_init__(self) -> None:
        # The Sionna contract stores bandwidth = N * dF (23.04 MHz for 768 x 30 kHz),
        # while the occupied span between the first and last subcarrier is (N-1) * dF.
        expected = self.num_subcarriers * self.subcarrier_spacing_hz
        if abs(expected - self.bandwidth_hz) > 1.0:
            raise ValueError(
                "grid inconsistent: "
                f"N*dF={expected:.1f} Hz but bandwidth={self.bandwidth_hz:.1f} Hz"
            )

    def frequencies_hz(self) -> np.ndarray:
        offsets = np.arange(self.num_subcarriers) - (self.num_subcarriers - 1) / 2.0
        return self.center_frequency_hz + self.subcarrier_spacing_hz * offsets

    def as_dict(self) -> dict[str, float | int]:
        return {
            "center_frequency_hz": self.center_frequency_hz,
            "bandwidth_hz": self.bandwidth_hz,
            "num_subcarriers": self.num_subcarriers,
            "subcarrier_spacing_hz": self.subcarrier_spacing_hz,
            "first_frequency_hz": float(self.frequencies_hz()[0]),
            "last_frequency_hz": float(self.frequencies_hz()[-1]),
        }


# --------------------------------------------------------------------------------------
# Antennas
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanarArraySpec:
    """Planar array, isotropic element, vertical polarisation (Sionna contract)."""

    num_rows: int = 4
    num_cols: int = 8
    #: Physical spacing; 0 means "derive from spacing_lambda at the contract frequency".
    spacing_m: float = 0.0
    spacing_lambda: float = 0.5
    pattern: str = "iso"
    polarization: str = "V"

    @property
    def num_elements(self) -> int:
        return self.num_rows * self.num_cols

    def element_spacing_m(self, center_frequency_hz: float) -> float:
        """Physical element spacing at ``center_frequency_hz`` (Sionna specifies 0.5 lambda)."""
        if self.spacing_m:
            return self.spacing_m
        return self.spacing_lambda * C0 / center_frequency_hz

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "num_elements": self.num_elements,
            "spacing_m": self.spacing_m,
            "spacing_lambda": self.spacing_lambda,
            "pattern": self.pattern,
            "polarization": self.polarization,
        }


@dataclass(frozen=True)
class AntennaSpec:
    bs: PlanarArraySpec = field(default_factory=PlanarArraySpec)
    ue: PlanarArraySpec = field(
        default_factory=lambda: PlanarArraySpec(
            num_rows=1, num_cols=1, spacing_m=0.0, spacing_lambda=0.5
        )
    )
    #: yaw (rotation about +z, radians) applied to the BS panel from the prepared label.
    bs_orientation_source: str = "label.bs_points[i].orientation_rad[0]"


# --------------------------------------------------------------------------------------
# Propagation solver
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SolverSpec:
    """DiffeRT path-tracer configuration.

    ``order`` is the maximum number of interactions, matching Sionna's ``max_depth``
    semantics for the reflection part of the path set.
    """

    #: "sbr" is the only tracer that scales to Front3D triangle counts.
    tracer: str = "sbr"
    order: int = 4
    num_rays: int = 400_000
    max_num_candidates: int = 200_000
    #: LoS is always requested by asking for order 0 as well.
    include_los: bool = True
    reflection: bool = True
    diffraction: bool = False
    transmission: bool = False
    #: Field solver defaults to V polarisation on both sides.
    tx_polarization: str = "V"
    rx_polarization: str = "V"
    #: Wall thickness used for transmission. Sionna's ``DEFAULT_THICKNESS`` is 0.1 m and
    #: our scene XMLs carry no per-material thickness, so 0.1 m keeps the two engines aligned.
    material_thickness_m: float = 0.1

    def interactions(self) -> frozenset[str]:
        names = []
        if self.reflection:
            names.append("reflection")
        if self.diffraction:
            names.append("diffraction")
        if self.transmission:
            names.append("transmission")
        return frozenset(names)

    def orders(self) -> tuple[int, ...]:
        """Orders to request in a single call (0 = LoS when enabled)."""
        first = 0 if self.include_los else 1
        return tuple(range(first, self.order + 1))

    def as_dict(self) -> dict[str, object]:
        return {
            "tracer": self.tracer,
            "order": self.order,
            "num_rays": self.num_rays,
            "max_num_candidates": self.max_num_candidates,
            "include_los": self.include_los,
            "interactions": sorted(self.interactions()),
            "tx_polarization": self.tx_polarization,
            "rx_polarization": self.rx_polarization,
            "material_thickness_m": self.material_thickness_m,
        }


C0 = 299_792_458.0

DEFAULT_GRID = FrequencyGrid()
DEFAULT_CASE = CaseSpec()
DEFAULT_ANTENNA = AntennaSpec()
DEFAULT_SOLVER = SolverSpec()

# --------------------------------------------------------------------------------------
# Contract profiles
# --------------------------------------------------------------------------------------
#
# Two antenna/carrier contracts exist in this workspace:
#
#   "2988"  the original Front3d_2988_CFR_Only dataset: 3.5 GHz, BS 4x4 (16 ports),
#           UE 1x2 (2 ports), BS orientation fixed at [0, 0, 0] (the dataset predates
#           per-device orientation), CFR shape [link, 16, 2, 768].
#   "339"   the DICHASUS-like regeneration: 3.438 GHz, BS 4x8 (32 ports), UE 1x1,
#           per-BS yaw from the prepared label, CFR shape [link, 32, 1, 768].
#
# Everything downstream (grid, arrays, link building, HDF5 shape) is derived from the
# selected profile so the two can never be mixed silently.


@dataclass(frozen=True)
class ContractProfile:
    name: str
    grid: FrequencyGrid
    bs: PlanarArraySpec
    ue: PlanarArraySpec
    #: false for the 2988 contract: the BS array kept its global orientation.
    use_label_bs_yaw: bool

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "grid": self.grid.as_dict(),
            "bs_array": self.bs.as_dict(),
            "ue_array": self.ue.as_dict(),
            "use_label_bs_yaw": self.use_label_bs_yaw,
            "cfr_shape": ["num_links", self.bs.num_elements, self.ue.num_elements, self.grid.num_subcarriers],
        }


PROFILE_2988 = ContractProfile(
    name="2988",
    grid=FrequencyGrid(center_frequency_hz=3_500_000_000.0),
    bs=PlanarArraySpec(num_rows=4, num_cols=4),
    ue=PlanarArraySpec(num_rows=1, num_cols=2, spacing_m=0.0),
    use_label_bs_yaw=False,
)

PROFILE_339 = ContractProfile(
    name="339",
    grid=FrequencyGrid(),
    bs=PlanarArraySpec(),
    ue=PlanarArraySpec(num_rows=1, num_cols=1, spacing_m=0.0),
    use_label_bs_yaw=True,
)

PROFILES = {"2988": PROFILE_2988, "339": PROFILE_339}


def get_profile(name: str) -> ContractProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"unknown contract profile {name!r}; choose from {sorted(PROFILES)}") from None
