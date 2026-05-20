"""Standards-shaped NR SRS resource helpers.

This module intentionally implements a small, testable SRS subset.  It is not
yet a full 3GPP SRS resource mapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


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
    cyclic_shift_indices: tuple[int, ...] | None = None


@dataclass(frozen=True)
class NRSRSResource:
    config: NRSRSResourceConfig
    srs_symbol_indices: np.ndarray
    re_subcarrier_indices: np.ndarray
    resource_mask: np.ndarray
    pilot_symbols: np.ndarray
    port_index: np.ndarray
    cyclic_shift_indices: np.ndarray


def resolve_srs_resource_config(
    phy_config: Any,
    *,
    num_subcarriers: int,
    num_ports: int,
) -> NRSRSResourceConfig:
    """Resolve SRS config from RTTruthRunConfig/Pydantic config-like objects."""

    srs = getattr(phy_config, "srs_config", None)
    if srs is None:
        srs = getattr(phy_config, "srs", None)
    legacy_num_symbols = int(getattr(phy_config, "num_ofdm_symbols", 2) or 2)
    default_num_srs_symbols = (
        max(legacy_num_symbols, num_ports) if srs is None else legacy_num_symbols
    )
    default_start_symbol = (
        max(0, 14 - default_num_srs_symbols) if srs is None else 12
    )

    cfg = NRSRSResourceConfig(
        slot_length_symbols=_get_int(srs, "slot_length_symbols", 14),
        start_symbol=_get_int(srs, "start_symbol", default_start_symbol),
        num_srs_symbols=_get_int(
            srs,
            "num_srs_symbols",
            default_num_srs_symbols,
        ),
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
        cyclic_shift_indices=_get_cyclic_shifts(srs),
    )
    validate_srs_resource_config(cfg, num_subcarriers=num_subcarriers, num_ports=num_ports)
    return cfg


def validate_srs_resource_config(
    config: NRSRSResourceConfig,
    *,
    num_subcarriers: int,
    num_ports: int,
) -> None:
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
    if config.sequence_type != "zc_like":
        raise ValueError("SRS sequence_type only supports zc_like in v1")
    if config.group_hopping != "disabled":
        raise ValueError("SRS group_hopping must be disabled in v1")
    if config.sequence_hopping != "disabled":
        raise ValueError("SRS sequence_hopping must be disabled in v1")
    if config.num_srs_symbols < num_ports:
        raise ValueError(
            "Stage-1 SRS port separation uses time-symbol orthogonality; "
            "num_srs_symbols must be >= number of SRS ports"
        )
    if config.cyclic_shift_indices is not None and len(config.cyclic_shift_indices) < num_ports:
        raise ValueError("SRS cyclic_shift_indices must provide at least one value per port")
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
    num_ports: int,
    default_num_prb: int,
) -> NRSRSResource:
    """Build full-slot SRS resource mask and port pilot symbols."""

    validate_srs_resource_config(config, num_subcarriers=num_subcarriers, num_ports=num_ports)
    srs_symbols = np.arange(
        config.start_symbol,
        config.start_symbol + config.num_srs_symbols,
        dtype=np.int32,
    )
    re_indices = _build_re_indices(
        config,
        num_subcarriers=num_subcarriers,
        default_num_prb=default_num_prb,
    )
    if re_indices.size == 0:
        raise ValueError("SRS resource mapping produced no resource elements")

    resource_mask = np.zeros((config.slot_length_symbols, num_subcarriers), dtype=np.bool_)
    resource_mask[np.ix_(srs_symbols, re_indices)] = True
    cyclic_shifts = _resolve_cyclic_shifts(config, num_ports)
    port_index = np.arange(num_ports, dtype=np.int32)
    pilot_symbols = _build_pilot_symbols(
        config,
        srs_symbols=srs_symbols,
        re_indices=re_indices,
        num_subcarriers=num_subcarriers,
        num_ports=num_ports,
        cyclic_shifts=cyclic_shifts,
    )
    return NRSRSResource(
        config=config,
        srs_symbol_indices=srs_symbols,
        re_subcarrier_indices=re_indices,
        resource_mask=resource_mask,
        pilot_symbols=pilot_symbols,
        port_index=port_index,
        cyclic_shift_indices=cyclic_shifts,
    )


def _build_re_indices(
    config: NRSRSResourceConfig,
    *,
    num_subcarriers: int,
    default_num_prb: int,
) -> np.ndarray:
    start = config.bwp_start_prb * 12
    if start >= num_subcarriers:
        raise ValueError("SRS BWP start is outside the carrier")
    if config.bwp_num_prb is None:
        requested_prb = max(int(default_num_prb), 1)
        end = min(num_subcarriers, start + requested_prb * 12)
    else:
        end = start + int(config.bwp_num_prb) * 12
        if end > num_subcarriers:
            raise ValueError("SRS BWP exceeds carrier subcarrier count")
    return np.arange(start + config.comb_offset, end, config.comb_size, dtype=np.int32)


def _build_pilot_symbols(
    config: NRSRSResourceConfig,
    *,
    srs_symbols: np.ndarray,
    re_indices: np.ndarray,
    num_subcarriers: int,
    num_ports: int,
    cyclic_shifts: np.ndarray,
) -> np.ndarray:
    pilots = np.zeros(
        (num_ports, config.slot_length_symbols, num_subcarriers),
        dtype=np.complex64,
    )
    n = np.arange(re_indices.size, dtype=np.float32)
    root = np.float32(1 + (config.sequence_id % max(re_indices.size, 1)))
    length = np.float32(max(re_indices.size, 1))
    base = np.exp(-1j * np.pi * root * n * (n + 1.0) / length).astype(np.complex64)
    for port in range(num_ports):
        cyclic_phase = np.exp(
            1j * 2.0 * np.pi * np.float32(cyclic_shifts[port]) * n / np.float32(12.0)
        ).astype(np.complex64)
        for local_symbol, symbol in enumerate(srs_symbols):
            time_code = np.complex64(np.exp(
                1j
                * 2.0
                * np.pi
                * np.float32(port)
                * np.float32(local_symbol)
                / np.float32(config.num_srs_symbols)
            ))
            pilots[port, int(symbol), re_indices] = base * cyclic_phase * time_code
    return pilots


def _resolve_cyclic_shifts(config: NRSRSResourceConfig, num_ports: int) -> np.ndarray:
    if config.cyclic_shift_indices is None:
        values = tuple(range(num_ports))
    else:
        values = config.cyclic_shift_indices[:num_ports]
    return np.asarray(values, dtype=np.int32)


def _get(obj: Any, name: str, default: Any) -> Any:
    if obj is None:
        return default
    return getattr(obj, name, default)


def _get_int(obj: Any, name: str, default: int) -> int:
    return int(_get(obj, name, default))


def _get_optional_int(obj: Any, name: str) -> int | None:
    value = _get(obj, name, None)
    return None if value is None else int(value)


def _get_cyclic_shifts(obj: Any) -> tuple[int, ...] | None:
    values = _get(obj, "cyclic_shift_indices", None)
    if values is None:
        return None
    return tuple(int(value) for value in values)
