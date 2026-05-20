"""Standards-shaped NR SRS resource helpers.

This module implements the repository's NR SRS standards-shaped v2 subset.
It is deterministic and testable, but it is not a full 3GPP-compliant mapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, gcd
from typing import Any

import numpy as np


@dataclass(frozen=True)
class NRSRSHoppingConfig:
    enabled: bool = False
    frequency_offsets_prb: tuple[int, ...] = ()
    bandwidth_num_prb: tuple[int, ...] = ()


@dataclass(frozen=True)
class NRSRSPortsConfig:
    num_srs_ports: int | None = None
    mapping: str = "one_to_one"
    port_tx_ant_map: tuple[tuple[int, ...], ...] | None = None
    usage: str = "non_codebook"


@dataclass(frozen=True)
class NRSRSPowerControlConfig:
    enabled: bool = False
    p0_dbm: float = 0.0
    alpha: float = 0.8
    min_tx_power_dbm: float = -40.0
    max_tx_power_dbm: float = 23.0
    serving_rx_policy: str = "strongest_path"


@dataclass(frozen=True)
class NRSRSResourceConfig:
    slot_length_symbols: int = 14
    start_symbol: int = 12
    num_srs_symbols: int = 2
    comb_size: int = 2
    comb_offset: int = 0
    bwp_start_prb: int = 0
    bwp_num_prb: int | None = None
    trigger_mode: str = "aperiodic"
    periodicity_slots: int = 1
    slot_offset: int = 0
    slot_number: int = 0
    sequence_type: str = "zc_like"
    sequence_id: int = 0
    group_hopping: str = "disabled"
    sequence_hopping: str = "disabled"
    cyclic_shift_multiplexing: str = "cyclic_shift"
    cyclic_shift_indices: tuple[int, ...] | None = None
    hopping: NRSRSHoppingConfig = NRSRSHoppingConfig()
    ports: NRSRSPortsConfig = NRSRSPortsConfig()
    power_control: NRSRSPowerControlConfig = NRSRSPowerControlConfig()


@dataclass(frozen=True)
class NRSRSResourcePlan:
    config: NRSRSResourceConfig
    srs_symbol_indices: np.ndarray
    re_symbol_indices: np.ndarray
    re_subcarrier_indices: np.ndarray
    resource_mask: np.ndarray
    pilot_symbols: np.ndarray
    port_tx_ant_map: np.ndarray
    cyclic_shift_indices: np.ndarray
    prb_start_per_symbol: np.ndarray
    prb_count_per_symbol: np.ndarray
    sequence_group_indices: np.ndarray
    sequence_indices: np.ndarray
    zc_root_indices: np.ndarray

    @property
    def port_index(self) -> np.ndarray:
        """Legacy one-symbol port index view when every port maps one-to-one."""

        first_symbol_map = self.port_tx_ant_map[:, 0]
        return first_symbol_map.astype(np.int32, copy=True)


# Backward-compatible import name used by older tests/callers.
NRSRSResource = NRSRSResourcePlan


def resolve_srs_resource_config(
    phy_config: Any,
    *,
    num_subcarriers: int,
    num_ports: int | None = None,
    num_tx_ant: int | None = None,
) -> NRSRSResourceConfig:
    """Resolve SRS config from RTTruthRunConfig/Pydantic config-like objects."""

    tx_ant = int(num_tx_ant if num_tx_ant is not None else num_ports if num_ports else 1)
    srs = getattr(phy_config, "srs_config", None)
    if srs is None:
        srs = getattr(phy_config, "srs", None)
    legacy_num_symbols = int(getattr(phy_config, "num_ofdm_symbols", 2) or 2)
    default_num_srs_symbols = max(legacy_num_symbols, tx_ant) if srs is None else legacy_num_symbols
    default_start_symbol = max(0, 14 - default_num_srs_symbols) if srs is None else 12

    cfg = NRSRSResourceConfig(
        slot_length_symbols=_get_int(srs, "slot_length_symbols", 14),
        start_symbol=_get_int(srs, "start_symbol", default_start_symbol),
        num_srs_symbols=_get_int(srs, "num_srs_symbols", default_num_srs_symbols),
        comb_size=_get_int(srs, "comb_size", 2),
        comb_offset=_get_int(srs, "comb_offset", 0),
        bwp_start_prb=_get_int(srs, "bwp_start_prb", 0),
        bwp_num_prb=_get_optional_int(srs, "bwp_num_prb"),
        trigger_mode=str(_get(srs, "trigger_mode", "aperiodic")),
        periodicity_slots=_get_int(srs, "periodicity_slots", 1),
        slot_offset=_get_int(srs, "slot_offset", 0),
        slot_number=_get_int(srs, "slot_number", 0),
        sequence_type=str(_get(srs, "sequence_type", "zc_like")),
        sequence_id=_get_int(srs, "sequence_id", 0),
        group_hopping=str(_get(srs, "group_hopping", "disabled")),
        sequence_hopping=str(_get(srs, "sequence_hopping", "disabled")),
        cyclic_shift_multiplexing=str(_get(srs, "cyclic_shift_multiplexing", "cyclic_shift")),
        cyclic_shift_indices=_get_cyclic_shifts(srs),
        hopping=_get_hopping_config(_get(srs, "hopping", None)),
        ports=_get_ports_config(_get(srs, "ports", None)),
        power_control=_get_power_control_config(_get(srs, "power_control", None)),
    )
    validate_srs_resource_config(cfg, num_subcarriers=num_subcarriers, num_tx_ant=tx_ant)
    return cfg


def validate_srs_resource_config(
    config: NRSRSResourceConfig,
    *,
    num_subcarriers: int,
    num_ports: int | None = None,
    num_tx_ant: int | None = None,
) -> None:
    tx_ant = int(num_tx_ant if num_tx_ant is not None else num_ports if num_ports else 1)
    if config.slot_length_symbols < 1:
        raise ValueError("SRS slot_length_symbols must be positive")
    if config.start_symbol < 0:
        raise ValueError("SRS start_symbol must be non-negative")
    if config.num_srs_symbols < 1:
        raise ValueError("SRS num_srs_symbols must be positive")
    if config.start_symbol + config.num_srs_symbols > config.slot_length_symbols:
        raise ValueError("SRS symbols must fit within the slot")
    if config.comb_size not in (1, 2, 4):
        raise ValueError("SRS comb_size must be one of 1, 2, 4")
    if config.comb_offset < 0 or config.comb_offset >= config.comb_size:
        raise ValueError("SRS comb_offset must satisfy 0 <= offset < comb_size")
    if config.bwp_start_prb < 0:
        raise ValueError("SRS bwp_start_prb must be non-negative")
    if config.bwp_num_prb is not None and config.bwp_num_prb < 1:
        raise ValueError("SRS bwp_num_prb must be positive when set")
    if config.trigger_mode not in ("aperiodic", "periodic", "semipersistent"):
        raise ValueError("SRS trigger_mode must be aperiodic/periodic/semipersistent")
    if config.periodicity_slots < 1:
        raise ValueError("SRS periodicity_slots must be positive")
    if config.slot_offset < 0 or config.slot_number < 0:
        raise ValueError("SRS slot_offset and slot_number must be non-negative")
    if not is_srs_scheduled(config):
        raise ValueError(
            "SRS is not scheduled for the configured slot_number; current pipeline "
            "supports a single scheduled slot only"
        )
    if config.sequence_type not in ("zc_like", "nr_zc"):
        raise ValueError("SRS sequence_type must be zc_like/nr_zc")
    if config.group_hopping not in ("disabled", "enabled"):
        raise ValueError("SRS group_hopping must be disabled/enabled")
    if config.sequence_hopping not in ("disabled", "enabled"):
        raise ValueError("SRS sequence_hopping must be disabled/enabled")
    if config.cyclic_shift_multiplexing not in ("time", "cyclic_shift"):
        raise ValueError("SRS cyclic_shift_multiplexing must be time/cyclic_shift")
    port_count = _resolve_num_ports(config, tx_ant)
    if config.cyclic_shift_multiplexing == "time" and config.num_srs_symbols < port_count:
        raise ValueError(
            "time SRS port separation uses symbol-code orthogonality; "
            "num_srs_symbols must be >= number of SRS ports"
        )
    if config.cyclic_shift_indices is not None and len(config.cyclic_shift_indices) < port_count:
        raise ValueError("SRS cyclic_shift_indices must provide at least one value per port")
    if config.hopping.enabled:
        _validate_optional_symbol_list(
            config.hopping.frequency_offsets_prb,
            config.num_srs_symbols,
            "SRS hopping.frequency_offsets_prb",
        )
        _validate_optional_symbol_list(
            config.hopping.bandwidth_num_prb,
            config.num_srs_symbols,
            "SRS hopping.bandwidth_num_prb",
        )
    if config.ports.mapping not in ("one_to_one", "antenna_switching"):
        raise ValueError("SRS ports.mapping must be one_to_one/antenna_switching")
    if config.ports.usage not in ("codebook", "non_codebook"):
        raise ValueError("SRS ports.usage must be codebook/non_codebook")
    if num_subcarriers < 1:
        raise ValueError("num_subcarriers must be positive")


def is_srs_scheduled(config: NRSRSResourceConfig) -> bool:
    if config.trigger_mode == "aperiodic":
        return config.slot_number == config.slot_offset
    delta = config.slot_number - config.slot_offset
    return delta >= 0 and delta % config.periodicity_slots == 0


def build_srs_resource(
    config: NRSRSResourceConfig,
    *,
    num_subcarriers: int,
    num_ports: int | None = None,
    num_tx_ant: int | None = None,
    default_num_prb: int,
) -> NRSRSResourcePlan:
    """Build full-slot SRS resource plan and port pilot symbols."""

    tx_ant = int(num_tx_ant if num_tx_ant is not None else num_ports if num_ports else 1)
    validate_srs_resource_config(config, num_subcarriers=num_subcarriers, num_tx_ant=tx_ant)
    num_srs_ports = _resolve_num_ports(config, tx_ant)
    srs_symbols = np.arange(
        config.start_symbol,
        config.start_symbol + config.num_srs_symbols,
        dtype=np.int32,
    )
    port_tx_ant_map = _resolve_port_tx_ant_map(
        config,
        num_srs_ports=num_srs_ports,
        num_tx_ant=tx_ant,
        num_srs_symbols=srs_symbols.size,
    )
    prb_start, prb_count = _build_symbol_prbs(
        config,
        num_subcarriers=num_subcarriers,
        default_num_prb=default_num_prb,
    )
    re_symbols, re_subcarriers, resource_mask = _build_flat_resource_indices(
        config,
        srs_symbols=srs_symbols,
        prb_start_per_symbol=prb_start,
        prb_count_per_symbol=prb_count,
        num_subcarriers=num_subcarriers,
    )
    if re_subcarriers.size == 0:
        raise ValueError("SRS resource mapping produced no resource elements")

    cyclic_shifts = _resolve_cyclic_shifts(config, num_srs_ports)
    groups, sequences, roots = _build_sequence_metadata(config, srs_symbols, prb_count)
    pilot_symbols = _build_pilot_symbols(
        config,
        srs_symbols=srs_symbols,
        re_symbol_indices=re_symbols,
        re_subcarrier_indices=re_subcarriers,
        num_subcarriers=num_subcarriers,
        num_ports=num_srs_ports,
        cyclic_shifts=cyclic_shifts,
        port_tx_ant_map=port_tx_ant_map,
        sequence_group_indices=groups,
        sequence_indices=sequences,
        zc_root_indices=roots,
    )
    return NRSRSResourcePlan(
        config=config,
        srs_symbol_indices=srs_symbols,
        re_symbol_indices=re_symbols,
        re_subcarrier_indices=re_subcarriers,
        resource_mask=resource_mask,
        pilot_symbols=pilot_symbols,
        port_tx_ant_map=port_tx_ant_map,
        cyclic_shift_indices=cyclic_shifts,
        prb_start_per_symbol=prb_start,
        prb_count_per_symbol=prb_count,
        sequence_group_indices=groups,
        sequence_indices=sequences,
        zc_root_indices=roots,
    )


def _resolve_num_ports(config: NRSRSResourceConfig, num_tx_ant: int) -> int:
    if config.ports.num_srs_ports is not None:
        return int(config.ports.num_srs_ports)
    if config.ports.port_tx_ant_map is not None:
        return len(config.ports.port_tx_ant_map)
    return int(num_tx_ant)


def _resolve_port_tx_ant_map(
    config: NRSRSResourceConfig,
    *,
    num_srs_ports: int,
    num_tx_ant: int,
    num_srs_symbols: int,
) -> np.ndarray:
    ports = config.ports
    if ports.mapping == "one_to_one":
        if num_srs_ports > num_tx_ant:
            raise ValueError("SRS one_to_one ports require num_srs_ports <= num_tx_ant")
        mapping = np.broadcast_to(
            np.arange(num_srs_ports, dtype=np.int32)[:, np.newaxis],
            (num_srs_ports, num_srs_symbols),
        ).copy()
    else:
        if ports.port_tx_ant_map is None:
            raise ValueError("SRS antenna_switching requires ports.port_tx_ant_map")
        mapping = np.asarray(ports.port_tx_ant_map, dtype=np.int32)
        if mapping.shape != (num_srs_ports, num_srs_symbols):
            raise ValueError(
                "SRS ports.port_tx_ant_map must have shape "
                f"[num_srs_ports,num_srs_symbols]={num_srs_ports,num_srs_symbols}"
            )
        if np.any(mapping >= num_tx_ant):
            raise ValueError("SRS ports.port_tx_ant_map references missing TX antenna")
        covered = set(int(v) for v in mapping.ravel() if int(v) >= 0)
        expected = set(range(num_tx_ant))
        if not expected.issubset(covered):
            raise ValueError("SRS antenna_switching must sound every TX antenna at least once")
    if np.any(mapping < -1):
        raise ValueError("SRS port_tx_ant_map values must be >= -1")
    return mapping


def _build_symbol_prbs(
    config: NRSRSResourceConfig,
    *,
    num_subcarriers: int,
    default_num_prb: int,
) -> tuple[np.ndarray, np.ndarray]:
    carrier_prb = max(1, min(int(default_num_prb), int(ceil(num_subcarriers / 12.0))))
    if carrier_prb < 1:
        raise ValueError("SRS carrier must contain at least one PRB")
    base_count = int(config.bwp_num_prb or carrier_prb)
    starts: list[int] = []
    counts: list[int] = []
    for local_symbol in range(config.num_srs_symbols):
        offset = _symbol_list_value(config.hopping.frequency_offsets_prb, local_symbol, 0)
        count = _symbol_list_value(config.hopping.bandwidth_num_prb, local_symbol, base_count)
        start = int(config.bwp_start_prb) + (int(offset) if config.hopping.enabled else 0)
        count = int(count) if config.hopping.enabled else base_count
        if start < 0 or count < 1:
            raise ValueError("SRS PRB start/count must be valid")
        if start >= carrier_prb or start * 12 >= num_subcarriers:
            raise ValueError("SRS BWP/hopping allocation exceeds carrier PRB/subcarrier count")
        if (start + count > carrier_prb or (start + count) * 12 > num_subcarriers) and (
            num_subcarriers >= 12
        ):
            raise ValueError("SRS BWP/hopping allocation exceeds carrier PRB/subcarrier count")
        starts.append(start)
        counts.append(count)
    return np.asarray(starts, dtype=np.int32), np.asarray(counts, dtype=np.int32)


def _build_flat_resource_indices(
    config: NRSRSResourceConfig,
    *,
    srs_symbols: np.ndarray,
    prb_start_per_symbol: np.ndarray,
    prb_count_per_symbol: np.ndarray,
    num_subcarriers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    resource_mask = np.zeros((config.slot_length_symbols, num_subcarriers), dtype=np.bool_)
    symbols: list[np.ndarray] = []
    subcarriers: list[np.ndarray] = []
    for local_symbol, symbol in enumerate(srs_symbols):
        start = int(prb_start_per_symbol[local_symbol]) * 12
        symbol_prb_end = (
            int(prb_start_per_symbol[local_symbol])
            + int(prb_count_per_symbol[local_symbol])
        ) * 12
        end = min(
            symbol_prb_end,
            num_subcarriers,
        )
        re = np.arange(start + config.comb_offset, end, config.comb_size, dtype=np.int32)
        if re.size == 0:
            raise ValueError("SRS symbol resource mapping produced no RE")
        resource_mask[int(symbol), re] = True
        symbols.append(np.full(re.shape, int(symbol), dtype=np.int32))
        subcarriers.append(re)
    return (
        np.concatenate(symbols).astype(np.int32, copy=False),
        np.concatenate(subcarriers).astype(np.int32, copy=False),
        resource_mask,
    )


def _build_sequence_metadata(
    config: NRSRSResourceConfig,
    srs_symbols: np.ndarray,
    prb_count_per_symbol: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = np.zeros(srs_symbols.shape, dtype=np.int32)
    sequences = np.zeros(srs_symbols.shape, dtype=np.int32)
    roots = np.zeros(srs_symbols.shape, dtype=np.int32)
    for local_symbol, symbol in enumerate(srs_symbols):
        group = (
            config.sequence_id + config.slot_number + int(symbol)
            if config.group_hopping == "enabled"
            else config.sequence_id
        ) % 30
        sequence = (
            config.sequence_id + 7 * config.slot_number + 3 * int(symbol)
            if config.sequence_hopping == "enabled"
            else 0
        ) % 2
        length = max(1, int(prb_count_per_symbol[local_symbol]) * 12 // config.comb_size)
        root_seed = (
            config.sequence_id
            + 7 * group
            + 13 * sequence
            + 5 * int(symbol)
            + 2 * local_symbol
        )
        root = _coprime_root(root_seed, length)
        groups[local_symbol] = int(group)
        sequences[local_symbol] = int(sequence)
        roots[local_symbol] = int(root)
    return groups, sequences, roots


def _build_pilot_symbols(
    config: NRSRSResourceConfig,
    *,
    srs_symbols: np.ndarray,
    re_symbol_indices: np.ndarray,
    re_subcarrier_indices: np.ndarray,
    num_subcarriers: int,
    num_ports: int,
    cyclic_shifts: np.ndarray,
    port_tx_ant_map: np.ndarray,
    sequence_group_indices: np.ndarray,
    sequence_indices: np.ndarray,
    zc_root_indices: np.ndarray,
) -> np.ndarray:
    pilots = np.zeros(
        (num_ports, config.slot_length_symbols, num_subcarriers),
        dtype=np.complex64,
    )
    for local_symbol, symbol in enumerate(srs_symbols):
        symbol_mask = re_symbol_indices == int(symbol)
        re_indices = re_subcarrier_indices[symbol_mask]
        n = np.arange(re_indices.size, dtype=np.float32)
        base = _build_base_sequence(
            config,
            n=n,
            root=int(zc_root_indices[local_symbol]),
            group=int(sequence_group_indices[local_symbol]),
            sequence=int(sequence_indices[local_symbol]),
            length=re_indices.size,
        )
        for port in range(num_ports):
            if int(port_tx_ant_map[port, local_symbol]) < 0:
                continue
            cyclic_phase = np.exp(
                1j * 2.0 * np.pi * np.float32(cyclic_shifts[port]) * n / np.float32(12.0)
            ).astype(np.complex64)
            if config.cyclic_shift_multiplexing == "time":
                time_code = np.complex64(np.exp(
                    1j
                    * 2.0
                    * np.pi
                    * np.float32(port)
                    * np.float32(local_symbol)
                    / np.float32(config.num_srs_symbols)
                ))
            else:
                time_code = np.complex64(1.0 + 0.0j)
            pilots[port, int(symbol), re_indices] = base * cyclic_phase * time_code
    return pilots


def _build_base_sequence(
    config: NRSRSResourceConfig,
    *,
    n: np.ndarray,
    root: int,
    group: int,
    sequence: int,
    length: int,
) -> np.ndarray:
    length_f = np.float32(max(int(length), 1))
    root_f = np.float32(max(int(root), 1))
    if config.sequence_type == "zc_like":
        phase = -np.pi * root_f * n * (n + 1.0) / length_f
    else:
        q = root_f + np.float32(group) / np.float32(31.0) + np.float32(sequence) / np.float32(7.0)
        phase = -np.pi * q * n * (n + 1.0) / length_f
    return np.exp(1j * phase).astype(np.complex64)


def _resolve_cyclic_shifts(config: NRSRSResourceConfig, num_ports: int) -> np.ndarray:
    if config.cyclic_shift_indices is None:
        if config.cyclic_shift_multiplexing == "cyclic_shift":
            values = tuple(int(round(port * 12 / num_ports)) % 12 for port in range(num_ports))
        else:
            values = tuple(port % 12 for port in range(num_ports))
    else:
        values = tuple(int(value) for value in config.cyclic_shift_indices[:num_ports])
    if any(value < 0 or value > 11 for value in values):
        raise ValueError("SRS cyclic_shift_indices must be in [0, 11]")
    if config.cyclic_shift_multiplexing == "cyclic_shift" and len(set(values)) != len(values):
        raise ValueError("SRS cyclic_shift multiplexing requires unique cyclic shifts")
    return np.asarray(values, dtype=np.int32)


def _coprime_root(seed: int, length: int) -> int:
    if length <= 2:
        return 1
    root = 1 + (int(seed) % (length - 1))
    while gcd(root, length) != 1:
        root = 1 + (root % (length - 1))
    return int(root)


def _validate_optional_symbol_list(values: tuple[int, ...], num_symbols: int, name: str) -> None:
    if values and len(values) != num_symbols:
        raise ValueError(f"{name} must be empty or have one value per SRS symbol")


def _symbol_list_value(values: tuple[int, ...], index: int, default: int) -> int:
    if not values:
        return int(default)
    return int(values[index])


def _get(obj: Any, name: str, default: Any) -> Any:
    if obj is None:
        return default
    return getattr(obj, name, default)


def _get_int(obj: Any, name: str, default: int) -> int:
    return int(_get(obj, name, default))


def _get_optional_int(obj: Any, name: str) -> int | None:
    value = _get(obj, name, None)
    return None if value is None else int(value)


def _get_int_tuple(obj: Any, name: str) -> tuple[int, ...]:
    values = _get(obj, name, ())
    if values is None:
        return ()
    return tuple(int(value) for value in values)


def _get_cyclic_shifts(obj: Any) -> tuple[int, ...] | None:
    values = _get(obj, "cyclic_shift_indices", None)
    if values is None:
        return None
    return tuple(int(value) for value in values)


def _get_hopping_config(obj: Any) -> NRSRSHoppingConfig:
    return NRSRSHoppingConfig(
        enabled=bool(_get(obj, "enabled", False)),
        frequency_offsets_prb=_get_int_tuple(obj, "frequency_offsets_prb"),
        bandwidth_num_prb=_get_int_tuple(obj, "bandwidth_num_prb"),
    )


def _get_ports_config(obj: Any) -> NRSRSPortsConfig:
    mapping = _get(obj, "port_tx_ant_map", None)
    map_tuple = None
    if mapping is not None:
        map_tuple = tuple(tuple(int(value) for value in row) for row in mapping)
    num_ports_value = _get(obj, "num_srs_ports", None)
    return NRSRSPortsConfig(
        num_srs_ports=None if num_ports_value is None else int(num_ports_value),
        mapping=str(_get(obj, "mapping", "one_to_one")),
        port_tx_ant_map=map_tuple,
        usage=str(_get(obj, "usage", "non_codebook")),
    )


def _get_power_control_config(obj: Any) -> NRSRSPowerControlConfig:
    return NRSRSPowerControlConfig(
        enabled=bool(_get(obj, "enabled", False)),
        p0_dbm=float(_get(obj, "p0_dbm", 0.0)),
        alpha=float(_get(obj, "alpha", 0.8)),
        min_tx_power_dbm=float(_get(obj, "min_tx_power_dbm", -40.0)),
        max_tx_power_dbm=float(_get(obj, "max_tx_power_dbm", 23.0)),
        serving_rx_policy=str(_get(obj, "serving_rx_policy", "strongest_path")),
    )
