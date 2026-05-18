"""Topology domain models.

``RoleTopology`` is the input/config view: BS and UE are physical roles and do
not imply TX/RX. ``Topology`` is the link view consumed by RT/PHY/HDF5: TX/RX are
resolved from BS/UE using ``phy_link_direction``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sionna_measurement_sim.domain.validation import require_finite, require_shape


@dataclass(frozen=True)
class Topology:
    """TX/RX positions and labels in SI units."""

    tx_positions_m: np.ndarray
    rx_positions_m: np.ndarray
    tx_labels: tuple[str, ...]
    rx_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        tx_positions = np.asarray(self.tx_positions_m, dtype=np.float32)
        rx_positions = np.asarray(self.rx_positions_m, dtype=np.float32)

        require_shape("tx_positions_m", tx_positions, (None, 3))
        require_shape("rx_positions_m", rx_positions, (None, 3))
        require_finite("tx_positions_m", tx_positions)
        require_finite("rx_positions_m", rx_positions)

        if len(self.tx_labels) != tx_positions.shape[0]:
            msg = "tx_labels length must match tx_positions_m"
            raise ValueError(msg)
        if len(self.rx_labels) != rx_positions.shape[0]:
            msg = "rx_labels length must match rx_positions_m"
            raise ValueError(msg)

        object.__setattr__(self, "tx_positions_m", tx_positions)
        object.__setattr__(self, "rx_positions_m", rx_positions)
        object.__setattr__(self, "tx_labels", tuple(self.tx_labels))
        object.__setattr__(self, "rx_labels", tuple(self.rx_labels))

    @property
    def num_tx(self) -> int:
        return int(self.tx_positions_m.shape[0])

    @property
    def num_rx(self) -> int:
        return int(self.rx_positions_m.shape[0])


@dataclass(frozen=True)
class RoleTopology:
    """BS/UE positions and labels in SI units."""

    bs_positions_m: np.ndarray
    ue_positions_m: np.ndarray
    bs_labels: tuple[str, ...]
    ue_labels: tuple[str, ...]
    bs_global_indices: np.ndarray | tuple[int, ...] | None = None
    ue_global_indices: np.ndarray | tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        bs_positions = np.asarray(self.bs_positions_m, dtype=np.float32)
        ue_positions = np.asarray(self.ue_positions_m, dtype=np.float32)

        require_shape("bs_positions_m", bs_positions, (None, 3))
        require_shape("ue_positions_m", ue_positions, (None, 3))
        require_finite("bs_positions_m", bs_positions)
        require_finite("ue_positions_m", ue_positions)

        if len(self.bs_labels) != bs_positions.shape[0]:
            msg = "bs_labels length must match bs_positions_m"
            raise ValueError(msg)
        if len(self.ue_labels) != ue_positions.shape[0]:
            msg = "ue_labels length must match ue_positions_m"
            raise ValueError(msg)

        bs_indices = (
            np.arange(bs_positions.shape[0], dtype=np.int64)
            if self.bs_global_indices is None
            else np.asarray(self.bs_global_indices, dtype=np.int64)
        )
        ue_indices = (
            np.arange(ue_positions.shape[0], dtype=np.int64)
            if self.ue_global_indices is None
            else np.asarray(self.ue_global_indices, dtype=np.int64)
        )
        require_shape("bs_global_indices", bs_indices, (bs_positions.shape[0],))
        require_shape("ue_global_indices", ue_indices, (ue_positions.shape[0],))
        if np.any(bs_indices < 0):
            msg = "bs_global_indices must be non-negative"
            raise ValueError(msg)
        if np.any(ue_indices < 0):
            msg = "ue_global_indices must be non-negative"
            raise ValueError(msg)

        object.__setattr__(self, "bs_positions_m", bs_positions)
        object.__setattr__(self, "ue_positions_m", ue_positions)
        object.__setattr__(self, "bs_labels", tuple(self.bs_labels))
        object.__setattr__(self, "ue_labels", tuple(self.ue_labels))
        object.__setattr__(self, "bs_global_indices", bs_indices)
        object.__setattr__(self, "ue_global_indices", ue_indices)

    @property
    def num_bs(self) -> int:
        return int(self.bs_positions_m.shape[0])

    @property
    def num_ue(self) -> int:
        return int(self.ue_positions_m.shape[0])


@dataclass(frozen=True)
class LinkRoleMapping:
    """Resolved BS/UE to TX/RX mapping for one PHY link direction."""

    phy_link_direction: str
    tx_role: str
    rx_role: str

    def __post_init__(self) -> None:
        direction = str(self.phy_link_direction).lower()
        tx_role = str(self.tx_role).lower()
        rx_role = str(self.rx_role).lower()
        if direction not in ("uplink", "downlink"):
            msg = "phy_link_direction must be uplink or downlink"
            raise ValueError(msg)
        if {tx_role, rx_role} != {"bs", "ue"}:
            msg = "tx_role/rx_role must resolve to bs and ue"
            raise ValueError(msg)
        object.__setattr__(self, "phy_link_direction", direction)
        object.__setattr__(self, "tx_role", tx_role)
        object.__setattr__(self, "rx_role", rx_role)


def resolve_link_roles(phy_link_direction: str) -> LinkRoleMapping:
    """Return the TX/RX roles implied by a PHY link direction."""

    direction = str(phy_link_direction).lower()
    if direction == "uplink":
        return LinkRoleMapping(direction, tx_role="ue", rx_role="bs")
    if direction == "downlink":
        return LinkRoleMapping(direction, tx_role="bs", rx_role="ue")
    msg = "phy_link_direction must be uplink or downlink"
    raise ValueError(msg)


def resolve_role_topology(
    role_topology: RoleTopology,
    mapping: LinkRoleMapping,
) -> Topology:
    """Resolve a BS/UE topology into link-view TX/RX topology."""

    tx_positions = _role_value(
        mapping.tx_role,
        bs_value=role_topology.bs_positions_m,
        ue_value=role_topology.ue_positions_m,
    )
    rx_positions = _role_value(
        mapping.rx_role,
        bs_value=role_topology.bs_positions_m,
        ue_value=role_topology.ue_positions_m,
    )
    tx_labels = _role_value(
        mapping.tx_role,
        bs_value=role_topology.bs_labels,
        ue_value=role_topology.ue_labels,
    )
    rx_labels = _role_value(
        mapping.rx_role,
        bs_value=role_topology.bs_labels,
        ue_value=role_topology.ue_labels,
    )
    return Topology(
        tx_positions_m=tx_positions,
        rx_positions_m=rx_positions,
        tx_labels=tuple(tx_labels),
        rx_labels=tuple(rx_labels),
    )


def resolved_global_indices(
    role_topology: RoleTopology,
    mapping: LinkRoleMapping,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(tx_indices, rx_indices)`` in link-view order."""

    tx_indices = _role_value(
        mapping.tx_role,
        bs_value=role_topology.bs_global_indices,
        ue_value=role_topology.ue_global_indices,
    )
    rx_indices = _role_value(
        mapping.rx_role,
        bs_value=role_topology.bs_global_indices,
        ue_value=role_topology.ue_global_indices,
    )
    return np.asarray(tx_indices, dtype=np.int64), np.asarray(rx_indices, dtype=np.int64)


def resolve_role_pair(
    *,
    bs_value,
    ue_value,
    mapping: LinkRoleMapping,
):
    """Return ``(tx_value, rx_value)`` for arbitrary BS/UE values."""

    return (
        _role_value(mapping.tx_role, bs_value=bs_value, ue_value=ue_value),
        _role_value(mapping.rx_role, bs_value=bs_value, ue_value=ue_value),
    )


def _role_value(role: str, *, bs_value, ue_value):
    if role == "bs":
        return bs_value
    if role == "ue":
        return ue_value
    msg = f"Unknown role {role!r}"
    raise ValueError(msg)
