"""Link-layer domain model for TDD reciprocity."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LinkConfig:
    """Configuration for the link direction and reciprocity behaviour.

    Stored under ``/link`` in the HDF5 file.
    """

    duplex_mode: str = "tdd"
    phy_link_direction: str = "uplink"
    rt_trace_direction: str = "bs_to_ue"
    reciprocity_mode: str = "transpose_rt_channel"
    reciprocity_applied: bool = True

    def __post_init__(self) -> None:
        if self.duplex_mode != "tdd":
            raise ValueError("Only TDD duplex mode is supported")
        if self.phy_link_direction != "uplink":
            raise ValueError("Only uplink PHY link direction is supported")
