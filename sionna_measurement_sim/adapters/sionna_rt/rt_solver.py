"""Minimal Sionna RT truth adapter for Phase 2."""

from __future__ import annotations

import importlib.metadata as metadata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sionna_measurement_sim.adapters.sionna_rt.path_adapter import (
    path_table_to_samples,
    paths_to_table,
)
from sionna_measurement_sim.domain.antenna import AntennaSpec
from sionna_measurement_sim.domain.channel import RTTruthResult
from sionna_measurement_sim.domain.frequency import FrequencyGrid
from sionna_measurement_sim.domain.path import PathSamples, PathTable
from sionna_measurement_sim.domain.topology import Topology


@dataclass(frozen=True)
class SionnaRTConfig:
    """Configuration for the minimal RT truth run."""

    scene_file: Path
    seed: int = 1
    max_depth: int = 0
    los: bool = True
    specular_reflection: bool = False
    diffuse_reflection: bool = False
    refraction: bool = False
    synthetic_array: bool = False
    normalize_cfr: bool = False
    normalize_delays: bool = False


@dataclass(frozen=True)
class SionnaRTTruthAdapterResult:
    """Adapter output without leaking native Sionna objects."""

    truth: RTTruthResult
    path_table: PathTable
    path_samples: PathSamples
    runtime_versions: dict[str, str]
    raw_cfr_shape: tuple[int, ...]
    internal_cfr_shape: tuple[int, ...]


def run_sionna_rt_truth(
    topology: Topology,
    antenna: AntennaSpec,
    frequency: FrequencyGrid,
    config: SionnaRTConfig,
) -> SionnaRTTruthAdapterResult:
    """Run Sionna RT and return TX-first truth CFR."""

    scene, modules = _build_scene(topology, antenna, frequency, config)
    paths = modules["PathSolver"]()(
        scene=scene,
        max_depth=config.max_depth,
        los=config.los,
        specular_reflection=config.specular_reflection,
        diffuse_reflection=config.diffuse_reflection,
        refraction=config.refraction,
        synthetic_array=config.synthetic_array,
        seed=config.seed,
    )

    frequency_offsets_hz = frequency.frequencies_hz - frequency.center_frequency_hz
    raw_cfr = paths.cfr(
        frequencies=frequency_offsets_hz,
        normalize=config.normalize_cfr,
        normalize_delays=config.normalize_delays,
        out_type="numpy",
    )
    cfr = _to_tx_first_cfr(raw_cfr)
    path_power_db = _path_power_db(cfr)
    has_signal = np.any(np.isfinite(cfr) & (np.abs(cfr) > 0.0), axis=(2, 3, 4))
    path_table = paths_to_table(paths)
    path_samples = path_table_to_samples(path_table, topology)

    return SionnaRTTruthAdapterResult(
        truth=RTTruthResult(
            cfr=cfr.astype(np.complex64, copy=False),
            path_power_db=path_power_db,
            has_geometric_signal=has_signal,
        ),
        path_table=path_table,
        path_samples=path_samples,
        runtime_versions=_runtime_versions(),
        raw_cfr_shape=tuple(raw_cfr.shape),
        internal_cfr_shape=tuple(cfr.shape),
    )


def _build_scene(
    topology: Topology,
    antenna: AntennaSpec,
    frequency: FrequencyGrid,
    config: SionnaRTConfig,
) -> tuple[Any, dict[str, Any]]:
    from sionna.rt import PlanarArray, Receiver, Transmitter, load_scene

    scene = load_scene(config.scene_file.resolve().as_posix(), merge_shapes=True)
    scene.frequency = frequency.center_frequency_hz
    scene.tx_array = PlanarArray(
        num_rows=antenna.tx_num_rows,
        num_cols=antenna.tx_num_cols,
        vertical_spacing=float(antenna.tx_spacing_lambda[0]),
        horizontal_spacing=float(antenna.tx_spacing_lambda[1]),
        pattern=antenna.tx_pattern,
        polarization=_sionna_polarization(antenna.tx_polarization),
    )
    scene.rx_array = PlanarArray(
        num_rows=antenna.rx_num_rows,
        num_cols=antenna.rx_num_cols,
        vertical_spacing=float(antenna.rx_spacing_lambda[0]),
        horizontal_spacing=float(antenna.rx_spacing_lambda[1]),
        pattern=antenna.rx_pattern,
        polarization=_sionna_polarization(antenna.rx_polarization),
    )

    receivers = []
    for index, (label, position) in enumerate(
        zip(topology.tx_labels, topology.tx_positions_m, strict=True)
    ):
        scene.add(Transmitter(name=f"tx{index}_{label}", position=position.tolist()))
    for index, (label, position) in enumerate(
        zip(topology.rx_labels, topology.rx_positions_m, strict=True)
    ):
        receiver = Receiver(name=f"rx{index}_{label}", position=position.tolist())
        scene.add(receiver)
        receivers.append(receiver)

    if receivers:
        for tx_name in scene.transmitters:
            scene.get(tx_name).look_at(receivers[0])

    from sionna.rt import PathSolver

    return scene, {"PathSolver": PathSolver}


def _sionna_polarization(polarization: str) -> str:
    if polarization in {"single", "V", "vertical"}:
        return "V"
    return polarization


def _to_tx_first_cfr(raw_cfr: np.ndarray) -> np.ndarray:
    """Convert Sionna CFR [rx, rx_ant, tx, tx_ant, time, subcarrier] to TX-first."""

    cfr = np.asarray(raw_cfr)
    if cfr.ndim != 6:
        msg = f"Sionna cfr must have rank 6, got {cfr.shape}"
        raise ValueError(msg)
    if cfr.shape[4] != 1:
        msg = f"Phase 2 expects one time step, got {cfr.shape[4]}"
        raise ValueError(msg)
    return np.transpose(cfr[:, :, :, :, 0, :], (2, 0, 1, 3, 4))


def _path_power_db(cfr: np.ndarray) -> np.ndarray:
    power = np.mean(np.abs(cfr) ** 2, axis=(2, 3, 4))
    return (10.0 * np.log10(np.maximum(power, 1e-30))).astype(np.float32)


def _runtime_versions() -> dict[str, str]:
    versions = {
        "sionna": "",
        "sionna_rt": _version_or_empty("sionna-rt"),
        "torch": _version_or_empty("torch"),
        "mitsuba": _version_or_empty("mitsuba"),
        "drjit": _version_or_empty("drjit"),
    }
    return versions


def _version_or_empty(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return ""
