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
from sionna_measurement_sim.adapters.sionna_rt.shape_contracts import (
    to_project_cfr,
    to_project_cir,
)
from sionna_measurement_sim.domain.antenna import AntennaSpec
from sionna_measurement_sim.domain.channel import RTTruthResult
from sionna_measurement_sim.domain.cir import CIRTruth
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
    diffraction: bool = False
    synthetic_array: bool = False
    normalize_cfr: bool = False
    normalize_delays: bool = False
    num_time_steps: int = 1
    sampling_frequency_hz: float = 0.0  # for multi-snapshot Doppler synthetic
    tx_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rx_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    merge_shapes: bool = False


@dataclass(frozen=True)
class SionnaRTTruthAdapterResult:
    """Adapter output without leaking native Sionna objects."""

    truth: RTTruthResult
    path_table: PathTable
    path_samples: PathSamples
    runtime_versions: dict[str, str]
    raw_cfr_shape: tuple[int, ...]
    internal_cfr_shape: tuple[int, ...]
    cir_truth: CIRTruth
    tx_orientation_rad: np.ndarray | None = None  # (num_tx, 3) from scene
    rx_orientation_rad: np.ndarray | None = None  # (num_rx, 3) from scene


def run_sionna_rt_truth(
    topology: Topology,
    antenna: AntennaSpec,
    frequency: FrequencyGrid,
    config: SionnaRTConfig,
) -> SionnaRTTruthAdapterResult:
    """Run Sionna RT and return TX-first truth CFR."""

    scene, modules, tx_orientations, rx_orientations = _build_scene(
        topology, antenna, frequency, config,
    )
    paths = modules["PathSolver"]()(
        scene=scene,
        max_depth=config.max_depth,
        los=config.los,
        specular_reflection=config.specular_reflection,
        diffuse_reflection=config.diffuse_reflection,
        refraction=config.refraction,
        diffraction=config.diffraction,
        synthetic_array=config.synthetic_array,
        seed=config.seed,
    )

    frequency_offsets_hz = frequency.frequencies_hz - frequency.center_frequency_hz
    raw_cfr = paths.cfr(
        frequencies=frequency_offsets_hz,
        normalize=config.normalize_cfr,
        normalize_delays=config.normalize_delays,
        num_time_steps=config.num_time_steps,
        sampling_frequency=config.sampling_frequency_hz or None,
        out_type="numpy",
    )
    cfr, cfr_snapshots = _to_tx_first_cfr(raw_cfr, config.num_time_steps)
    has_signal = np.any(np.isfinite(cfr) & (np.abs(cfr) > 0.0), axis=(2, 3, 4))
    power = np.mean(np.abs(cfr) ** 2, axis=(2, 3, 4))
    path_power_db = (10.0 * np.log10(np.maximum(power, 1e-30))).astype(np.float32)
    path_table, geometric_path_count, los_exists, nlos_exists = paths_to_table(paths)
    path_samples = path_table_to_samples(path_table, topology)

    # Extract CIR from Sionna paths
    raw_cir_result = paths.cir(
        sampling_frequency=config.sampling_frequency_hz or None,
        num_time_steps=config.num_time_steps,
        out_type="numpy",
    )
    raw_cir_a, raw_cir_tau = raw_cir_result
    cir_coefficients, cir_delays, cir_valid = to_project_cir(
        raw_cir_a,
        raw_cir_tau,
        path_table.valid,
        config.num_time_steps,
    )

    return SionnaRTTruthAdapterResult(
        truth=RTTruthResult(
            cfr=cfr.astype(np.complex64, copy=False),
            path_power_db=path_power_db,
            has_geometric_signal=has_signal,
            geometric_path_count=geometric_path_count,
            los_exists=los_exists,
            nlos_exists=nlos_exists,
            cfr_snapshots=cfr_snapshots.astype(np.complex64, copy=False)
            if cfr_snapshots is not None else None,
        ),
        path_table=path_table,
        path_samples=path_samples,
        runtime_versions=_runtime_versions(),
        raw_cfr_shape=tuple(raw_cfr.shape),
        internal_cfr_shape=tuple(cfr.shape),
        cir_truth=CIRTruth(
            coefficients=cir_coefficients,
            delays_s=cir_delays,
            valid=cir_valid,
        ),
        tx_orientation_rad=np.array(tx_orientations, dtype=np.float64)
        if tx_orientations else None,
        rx_orientation_rad=np.array(rx_orientations, dtype=np.float64)
        if rx_orientations else None,
    )


def _build_scene(
    topology: Topology,
    antenna: AntennaSpec,
    frequency: FrequencyGrid,
    config: SionnaRTConfig,
) -> tuple[Any, dict[str, Any], list[list[float]], list[list[float]]]:
    from sionna.rt import PlanarArray, Receiver, Transmitter, load_scene

    scene = load_scene(
        config.scene_file.resolve().as_posix(),
        merge_shapes=config.merge_shapes,
    )
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

    transmitters = []
    receivers = []
    for index, (label, position) in enumerate(
        zip(topology.tx_labels, topology.tx_positions_m, strict=True)
    ):
        tx = Transmitter(name=f"tx{index}_{label}", position=position.tolist())
        tx.velocity = list(config.tx_velocity)
        scene.add(tx)
        transmitters.append(tx)
    for index, (label, position) in enumerate(
        zip(topology.rx_labels, topology.rx_positions_m, strict=True)
    ):
        receiver = Receiver(name=f"rx{index}_{label}", position=position.tolist())
        receiver.velocity = list(config.rx_velocity)
        scene.add(receiver)
        receivers.append(receiver)

    # Set TX orientation based on antenna.tx_orientation_mode
    for tx in transmitters:
        if antenna.tx_orientation_mode == "fixed":
            tx.orientation = [float(v) for v in antenna.tx_orientation_rad]
        elif antenna.tx_orientation_mode == "look_at_first_peer":
            if receivers:
                tx.look_at(receivers[0])
        elif antenna.tx_orientation_mode == "look_at_centroid":
            if receivers:
                centroid = np.mean(
                    topology.rx_positions_m, axis=0
                ).tolist()
                tx.look_at(centroid)

    # Read back TX orientations from scene (captures look_at results)
    tx_orientation_list: list[list[float]] = [
        np.asarray(tx.orientation, dtype=float).tolist() for tx in transmitters
    ]

    # Set RX orientation from config (read back for consistency)
    for rx in receivers:
        rx.orientation = [float(v) for v in antenna.rx_orientation_rad]
    rx_orientation_list: list[list[float]] = [
        np.asarray(rx.orientation, dtype=float).tolist() for rx in receivers
    ]

    from sionna.rt import PathSolver

    return scene, {"PathSolver": PathSolver}, tx_orientation_list, rx_orientation_list


def _sionna_polarization(polarization: str) -> str:
    if polarization in {"single", "V", "vertical"}:
        return "V"
    return polarization


def _to_tx_first_cfr(
    raw_cfr: np.ndarray, num_time_steps: int = 1,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Convert Sionna CFR to TX-first 5D, plus optional 6D snapshots.

    Delegates to :func:`shape_contracts.to_project_cfr`.
    """
    return to_project_cfr(raw_cfr, num_time_steps)


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
