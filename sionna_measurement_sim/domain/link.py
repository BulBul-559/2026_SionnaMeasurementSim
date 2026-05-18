"""Link-layer domain model."""

from __future__ import annotations

from dataclasses import dataclass

from sionna_measurement_sim.domain.topology import resolve_link_roles


@dataclass(frozen=True)
class LinkConfig:
    """Configuration for the link direction and resolved TX/RX roles.

    Stored under ``/link`` in the HDF5 file.
    """

    duplex_mode: str = "tdd"
    phy_link_direction: str = "uplink"
    tx_role: str = ""
    rx_role: str = ""
    # Internal legacy fallback only. Public YAML no longer exposes these fields.
    reciprocity_mode: str = "none"
    reciprocity_applied: bool = False

    def __post_init__(self) -> None:
        if self.duplex_mode != "tdd":
            raise ValueError("Only TDD duplex mode is supported")
        mapping = resolve_link_roles(self.phy_link_direction)
        tx_role = self.tx_role or mapping.tx_role
        rx_role = self.rx_role or mapping.rx_role
        if (tx_role, rx_role) != (mapping.tx_role, mapping.rx_role):
            raise ValueError("tx_role/rx_role must match phy_link_direction")
        reciprocity_mode = self.reciprocity_mode
        if self.reciprocity_applied and reciprocity_mode == "none":
            reciprocity_mode = "transpose_rt_channel"
        object.__setattr__(self, "phy_link_direction", mapping.phy_link_direction)
        object.__setattr__(self, "tx_role", tx_role)
        object.__setattr__(self, "rx_role", rx_role)
        object.__setattr__(self, "reciprocity_mode", reciprocity_mode)
