"""Standards-shaped NR SRS subset observation path."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from typing import Any

import numpy as np

from sionna_measurement_sim.domain.link import LinkConfig
from sionna_measurement_sim.domain.multiuser import MultiUserSRSResult
from sionna_measurement_sim.domain.observation import (
    EvaluationResult,
    ObservationResult,
    ReceiverSpec,
    WaveformSpec,
)
from sionna_measurement_sim.phy.common_link import (
    ObservationImpairmentChain,
    ResultAssembler,
)
from sionna_measurement_sim.phy.nr_srs_resources import (
    NRSRSResource,
    build_srs_resource,
    resolve_srs_resource_config,
)
from sionna_measurement_sim.phy.power import (
    compute_uplink_power,
    noise_mode_from_config,
    thermal_noise_metadata,
)
from sionna_measurement_sim.phy.spatial_spectrum import project_cfr_to_ul_receiver_samples


def run_nr_srs_observation(
    truth_cfr: np.ndarray,
    link_config: LinkConfig,
    phy_config: Any,
    carrier_config: Any,
    *,
    cfr_snapshots: np.ndarray | None = None,
    has_signal: np.ndarray | None = None,
    path_power_db: np.ndarray | None = None,
    tracer: Any | None = None,
) -> dict[str, Any]:
    """Run standards-shaped NR SRS resource sounding and LS CFR estimation.

    This is a standards-shaped v2 subset rather than a complete 3GPP NR SRS
    implementation. It supports comb/BWP resource mapping, full-slot time
    allocation, deterministic ZC-like or NR-ZC-style pilots, sequence/group
    hopping metadata, cyclic-shift port multiplexing, hopping plans, antenna
    mapping, and optional SRS power scaling.
    """

    _ = link_config
    sc_spacing_hz = float(getattr(phy_config, "subcarrier_spacing_khz", 30)) * 1000.0
    num_subcarriers = int(getattr(carrier_config, "num_subcarriers", truth_cfr.shape[-1]))
    truth = cfr_snapshots if cfr_snapshots is not None else truth_cfr
    with _span(tracer, "nr_srs.project_cfr_to_ul", truth_shape=_shape_tuple(truth)):
        h_ul = project_cfr_to_ul_receiver_samples(truth)
    num_snap, num_ul_tx, num_ul_rx, num_ul_rx_ant, num_ul_tx_ant, num_sc = h_ul.shape
    if num_sc != num_subcarriers:
        num_subcarriers = num_sc
    srs_resource_config = resolve_srs_resource_config(
        phy_config,
        num_subcarriers=num_subcarriers,
        num_tx_ant=num_ul_tx_ant,
    )
    srs_resource = build_srs_resource(
        srs_resource_config,
        num_subcarriers=num_subcarriers,
        num_tx_ant=num_ul_tx_ant,
        default_num_prb=int(getattr(phy_config, "num_prb", max(1, num_subcarriers // 12))),
    )
    num_srs_ports = int(srs_resource.pilot_symbols.shape[0])
    num_symbols = int(srs_resource_config.slot_length_symbols)
    power_config = getattr(phy_config, "power_config", None)
    power_control = compute_uplink_power(
        path_power_db=path_power_db,
        snapshot_count=num_snap,
        tx_count=num_ul_tx,
        rx_count=num_ul_rx,
        port_count=num_srs_ports,
        fixed_tx_power_dbm=float(getattr(phy_config, "tx_power_dbm", 0.0)),
        power_config=power_config,
        legacy_power_control=srs_resource_config.power_control,
    )
    snr_db = float(
        getattr(phy_config, "snr_db", None)
        if getattr(phy_config, "snr_db", None) is not None
        else getattr(phy_config, "observation_snr_db", 30.0)
    )

    with _span(
        tracer,
        "nr_srs.build_tx_grid",
        links=int(num_snap * num_ul_tx * num_ul_rx),
        tx_ant=int(num_ul_tx_ant),
        symbols=int(num_symbols),
        subcarriers=int(num_subcarriers),
    ):
        tx_grid = _build_srs_tx_grid(
            link_shape=(num_snap, num_ul_tx, num_ul_rx),
            num_ul_tx_ant=num_ul_tx_ant,
            srs_resource=srs_resource,
            power_scale_linear=power_control.power_scale_linear,
        )
    _record_array_event(tracer, "nr_srs.array_shape", "h_ul", h_ul)
    _record_array_event(tracer, "nr_srs.array_shape", "tx_grid", tx_grid)
    with _span(tracer, "nr_srs.channel_apply_einsum"):
        y_clean = np.einsum("...rtf,...tsf->...rsf", h_ul, tx_grid, optimize=True)
    _record_array_event(tracer, "nr_srs.array_shape", "y_clean", y_clean)
    with _span(tracer, "nr_srs.common_impairments_and_awgn", snr_db=float(snr_db)):
        noise_mode = noise_mode_from_config(power_config)
        thermal_meta = thermal_noise_metadata(
            power_config=power_config,
            default_bandwidth_hz=sc_spacing_hz * num_subcarriers,
        )
        impairment_chain = ObservationImpairmentChain(
            fft_size=num_subcarriers,
            sample_rate_hz=sc_spacing_hz * num_subcarriers,
            random_seed=int(getattr(phy_config, "observation_seed", 42)),
            impairment_config=getattr(phy_config, "impairment_config", None),
            noise_mode=noise_mode,
            thermal_noise_power_mw=thermal_meta["thermal_noise_mw"],
            thermal_noise_config=thermal_meta,
        )
        impairment_result = impairment_chain.apply(y_clean, snr_db=snr_db)
        rx_grid = (
            impairment_result.rx_grid.detach().cpu().numpy().astype(np.complex64, copy=False)
        )
        chain_scalars = ResultAssembler.chain_scalars_to_numpy(impairment_result)
        impairment_fields = ResultAssembler.sample_to_numpy(
            impairment_result.impairment_sample
        )
    _record_array_event(tracer, "nr_srs.array_shape", "rx_grid", rx_grid)
    with _span(tracer, "nr_srs.resource_extract_and_ls"):
        h_hat_resource = _estimate_srs_resource_cfr(
            rx_grid=rx_grid,
            srs_resource=srs_resource,
            power_scale_linear=power_control.power_scale_linear,
        )
    with _span(tracer, "nr_srs.interpolate_full_band"):
        h_hat_ul = _interpolate_resource_cfr(
            h_hat_resource,
            srs_resource=srs_resource,
            num_subcarriers=num_subcarriers,
            num_tx_ant=num_ul_tx_ant,
        )
    with _span(tracer, "nr_srs.to_link_view"):
        cfr_est = h_hat_ul.astype(np.complex64, copy=False)
        truth_dl = (
            np.asarray(truth, dtype=np.complex64)[np.newaxis, ...]
            if np.asarray(truth).ndim == 5
            else np.asarray(truth, dtype=np.complex64)
        )

    link_shape = cfr_est.shape[:3]
    valid_mask = _build_valid_mask(has_signal, link_shape)
    with _span(tracer, "nr_srs.metrics"):
        nmse_db, amplitude_error_db, phase_error_rad, correlation = _estimate_metrics(
            truth_dl,
            cfr_est,
            valid_mask,
        )
        truth_resource = _truth_resource_for_ports(truth_dl, srs_resource)
        resource_nmse_db, _, _, _ = _estimate_metrics(
            truth_resource,
            h_hat_resource,
            valid_mask,
        )
        rssi_dbm = chain_scalars["rssi_dbm"]
        noise_dbm = chain_scalars["noise_power_dbm"]

    with _span(tracer, "nr_srs.domain_models"):
        waveform = WaveformSpec(
            standard="nr_srs",
            sample_rate_hz=sc_spacing_hz * num_subcarriers,
            fft_size=num_subcarriers,
            cp_length=0,
            num_ofdm_symbols=num_symbols,
            pilot_indices=srs_resource.re_subcarrier_indices.copy(),
            data_subcarrier_indices=np.zeros((0,), dtype=np.int32),
            pilot_symbols=_reference_pilot_symbols(srs_resource),
            tx_power_dbm=float(getattr(phy_config, "tx_power_dbm", 0.0)),
        )
        observation = ObservationResult(
            cfr_est=cfr_est,
            valid_mask=valid_mask,
            detection_success=valid_mask.copy(),
            estimation_success=valid_mask.copy(),
            snr_db=chain_scalars["snr_db"],
            rssi_dbm=rssi_dbm,
            noise_power_dbm=noise_dbm,
            cfo_hz=impairment_fields["cfo_hz"],
            sfo_ppm=impairment_fields["sfo_ppm"],
            timing_offset_samples=impairment_fields["timing_offset_samples"],
            phase_offset_rad=impairment_fields["phase_offset_rad"],
            agc_gain_db=impairment_fields["agc_gain_db"],
            clipping_flag=impairment_fields["clipping_flag"],
        )
        evaluation = EvaluationResult(
            nmse_db=nmse_db,
            nmse_db_total=nmse_db.copy(),
            amplitude_error_db=amplitude_error_db,
            phase_error_rad=phase_error_rad,
            correlation=correlation,
            detection_rate=float(np.mean(valid_mask)) if valid_mask.size else 0.0,
            estimation_failure_rate=float(np.mean(~valid_mask)) if valid_mask.size else 0.0,
        )
        waveform_grids = {
            "tx_grid": tx_grid.astype(np.complex64, copy=False),
            "rx_grid": rx_grid,
            "noise_variance": chain_scalars["noise_variance"],
            "tx_power_dbm_per_port": power_control.tx_power_dbm,
            "tx_power_scale_linear": power_control.power_scale_linear,
            "serving_rx_index": power_control.serving_rx_index,
            "path_loss_db": power_control.path_loss_db,
            "power_clipped_flag": power_control.clipped_flag,
            "srs_resource_mask": srs_resource.resource_mask,
            "srs_pilot_symbols": srs_resource.pilot_symbols,
            "srs_re_symbol_indices": srs_resource.re_symbol_indices,
            "srs_re_subcarrier_indices": srs_resource.re_subcarrier_indices,
            "srs_symbol_indices": srs_resource.srs_symbol_indices,
            "srs_port_tx_ant_map": srs_resource.port_tx_ant_map,
            "srs_prb_start_per_symbol": srs_resource.prb_start_per_symbol,
            "srs_prb_count_per_symbol": srs_resource.prb_count_per_symbol,
            "srs_cyclic_shift_indices": srs_resource.cyclic_shift_indices,
            "srs_sequence_group_indices": srs_resource.sequence_group_indices,
            "srs_sequence_indices": srs_resource.sequence_indices,
            "srs_zc_root_indices": srs_resource.zc_root_indices,
            "srs_tx_power_dbm": power_control.tx_power_dbm,
            "srs_power_scale_linear": power_control.power_scale_linear,
            "srs_serving_rx_index": power_control.serving_rx_index,
            "srs_path_loss_db": power_control.path_loss_db,
            "cfr_est_resource": h_hat_resource.astype(np.complex64, copy=False),
            "srs_resource_nmse_db": resource_nmse_db,
            "srs_interpolation_nmse_db": nmse_db,
            "srs_resource_snr_db": chain_scalars["snr_db"],
            "srs_num_resource_elements": np.int32(
                srs_resource.re_subcarrier_indices.size
            ),
        }
        multiuser_result = _run_multiuser_srs_if_enabled(
            h_ul=h_ul,
            srs_resource_config=srs_resource_config,
            power_scale_linear=power_control.power_scale_linear,
            phy_config=phy_config,
            power_config=power_config,
            snr_db=snr_db,
            sc_spacing_hz=sc_spacing_hz,
            num_subcarriers=num_subcarriers,
            default_num_prb=int(getattr(phy_config, "num_prb", max(1, num_subcarriers // 12))),
            tracer=tracer,
        )
    return {
        "nr_waveform_spec": waveform,
        "waveform_spec": waveform,
        "receiver_spec": ReceiverSpec(
            receiver_type="srs_ls_receiver",
            estimator_type="srs_ls",
            sync_method="ideal",
            mimo_detector="none",
            interpolation_method="linear_frequency",
            failure_policy=getattr(phy_config, "receiver_failure_policy", "mark_invalid"),
        ),
        "evaluation": evaluation,
        "observation": observation,
        "impairments": impairment_result.impairment_spec,
        "waveform_grids": waveform_grids,
        "multiuser": multiuser_result,
        "metadata": {
            "srs_scope": "standards_shaped_subset_v2",
            "num_srs_symbols": int(srs_resource.srs_symbol_indices.size),
            "num_ul_tx_ant": num_ul_tx_ant,
            "num_ul_rx_ant": num_ul_rx_ant,
            "comb_size": int(srs_resource_config.comb_size),
            "comb_offset": int(srs_resource_config.comb_offset),
            "start_symbol": int(srs_resource_config.start_symbol),
            "slot_length_symbols": int(srs_resource_config.slot_length_symbols),
            "sequence_type": srs_resource_config.sequence_type,
            "group_hopping": srs_resource_config.group_hopping,
            "sequence_hopping": srs_resource_config.sequence_hopping,
            "cyclic_shift_multiplexing": srs_resource_config.cyclic_shift_multiplexing,
            "ports_mapping": srs_resource_config.ports.mapping,
            "ports_usage": srs_resource_config.ports.usage,
            "hopping_enabled": bool(srs_resource_config.hopping.enabled),
            "power_control_enabled": bool(
                getattr(getattr(power_config, "uplink_control", None), "enabled", False)
                or srs_resource_config.power_control.enabled
            ),
        },
    }


def _span(tracer: Any | None, name: str, **metadata: Any) -> Any:
    if tracer is None:
        return nullcontext()
    return tracer.span(name, **metadata)


def _shape_tuple(array: np.ndarray) -> tuple[int, ...]:
    return tuple(int(dim) for dim in np.asarray(array).shape)


def _record_array_event(
    tracer: Any | None,
    event: str,
    name: str,
    array: np.ndarray,
) -> None:
    if tracer is None:
        return
    arr = np.asarray(array)
    tracer.record_event(
        event,
        array=name,
        shape=_shape_tuple(arr),
        dtype=str(arr.dtype),
        bytes=int(arr.nbytes),
    )


def _build_srs_tx_grid(
    *,
    link_shape: tuple[int, int, int],
    num_ul_tx_ant: int,
    srs_resource: NRSRSResource,
    power_scale_linear: np.ndarray,
) -> np.ndarray:
    num_symbols, num_subcarriers = srs_resource.resource_mask.shape
    tx_grid = np.zeros(
        (*link_shape, num_ul_tx_ant, num_symbols, num_subcarriers),
        dtype=np.complex64,
    )
    scale = np.asarray(power_scale_linear, dtype=np.float32)
    if scale.shape[:2] != link_shape[:2] or scale.shape[2] != srs_resource.pilot_symbols.shape[0]:
        raise ValueError(
            "power_scale_linear must have shape [snapshot,tx,srs_port], "
            f"got {scale.shape}"
        )
    for port in range(srs_resource.pilot_symbols.shape[0]):
        for local_symbol, symbol in enumerate(srs_resource.srs_symbol_indices):
            tx_ant = int(srs_resource.port_tx_ant_map[port, local_symbol])
            if tx_ant < 0:
                continue
            if tx_ant >= num_ul_tx_ant:
                raise ValueError("SRS port_tx_ant_map references missing TX antenna")
            pilot = srs_resource.pilot_symbols[port, int(symbol), :]
            tx_grid[:, :, :, tx_ant, int(symbol), :] += (
                scale[:, :, port][:, :, np.newaxis, np.newaxis]
                * pilot[np.newaxis, np.newaxis, np.newaxis, :]
            )
    return tx_grid


def _run_multiuser_srs_if_enabled(
    *,
    h_ul: np.ndarray,
    srs_resource_config: Any,
    power_scale_linear: np.ndarray,
    phy_config: Any,
    power_config: Any | None,
    snr_db: float,
    sc_spacing_hz: float,
    num_subcarriers: int,
    default_num_prb: int,
    tracer: Any | None,
) -> MultiUserSRSResult | None:
    multi_cfg = getattr(getattr(phy_config, "srs_config", None), "multiuser", None)
    if multi_cfg is None:
        multi_cfg = getattr(getattr(phy_config, "srs", None), "multiuser", None)
    if multi_cfg is None or not bool(getattr(multi_cfg, "enabled", False)):
        return None
    if bool(getattr(srs_resource_config.hopping, "enabled", False)):
        raise ValueError("multi-UE SRS v1 does not support SRS frequency hopping yet")

    active_count = int(getattr(multi_cfg, "active_ue_count", 2))
    if active_count < 1:
        raise ValueError("phy.srs.multiuser.active_ue_count must be positive")
    strategy = str(getattr(multi_cfg, "resource_strategy", "comb_offset"))
    if strategy not in ("comb_offset", "prb_split"):
        raise ValueError("phy.srs.multiuser.resource_strategy must be comb_offset/prb_split")
    if str(getattr(multi_cfg, "frame_policy", "sequential")) != "sequential":
        raise ValueError("multi-UE SRS v1 supports sequential frame_policy only")
    if h_ul.shape[0] != 1:
        raise ValueError("multi-UE SRS v1 supports one static channel snapshot only")

    with _span(tracer, "nr_srs.multiuser_build", active_ue_count=active_count):
        return _run_multiuser_srs(
            h_ul=h_ul,
            srs_resource_config=srs_resource_config,
            power_scale_linear=power_scale_linear,
            strategy=strategy,
            active_count=active_count,
            phy_config=phy_config,
            power_config=power_config,
            snr_db=snr_db,
            sc_spacing_hz=sc_spacing_hz,
            num_subcarriers=num_subcarriers,
            default_num_prb=default_num_prb,
        )


def _run_multiuser_srs(
    *,
    h_ul: np.ndarray,
    srs_resource_config: Any,
    power_scale_linear: np.ndarray,
    strategy: str,
    active_count: int,
    phy_config: Any,
    power_config: Any | None,
    snr_db: float,
    sc_spacing_hz: float,
    num_subcarriers: int,
    default_num_prb: int,
) -> MultiUserSRSResult:
    num_snap, tx_count, rx_count, rx_ant, tx_ant, _ = h_ul.shape
    frame_count = int(np.ceil(tx_count / active_count))
    num_symbols = int(srs_resource_config.slot_length_symbols)
    y_clean_shared = np.zeros(
        (num_snap, frame_count, rx_count, rx_ant, num_symbols, num_subcarriers),
        dtype=np.complex64,
    )
    active_tx_indices = np.full((frame_count, active_count), -1, dtype=np.int32)
    active_tx_mask = np.zeros((frame_count, active_count), dtype=np.bool_)
    resources: list[list[NRSRSResource | None]] = [
        [None for _ in range(active_count)] for _ in range(frame_count)
    ]
    occupancy = np.zeros((frame_count, num_symbols, num_subcarriers), dtype=np.int32)
    comb_offsets = np.full((frame_count, active_count), -1, dtype=np.int32)

    for frame in range(frame_count):
        for active_slot in range(active_count):
            tx_index = frame * active_count + active_slot
            if tx_index >= tx_count:
                continue
            resource_cfg = _multiuser_resource_config(
                srs_resource_config,
                strategy=strategy,
                active_slot=active_slot,
                active_count=active_count,
                default_num_prb=default_num_prb,
            )
            resource = build_srs_resource(
                resource_cfg,
                num_subcarriers=num_subcarriers,
                num_tx_ant=tx_ant,
                default_num_prb=default_num_prb,
            )
            active_tx_indices[frame, active_slot] = tx_index
            active_tx_mask[frame, active_slot] = True
            resources[frame][active_slot] = resource
            comb_offsets[frame, active_slot] = int(resource_cfg.comb_offset)
            occupancy[frame] += resource.resource_mask.astype(np.int32)

            tx_grid = _build_srs_tx_grid(
                link_shape=(num_snap, 1, rx_count),
                num_ul_tx_ant=tx_ant,
                srs_resource=resource,
                power_scale_linear=power_scale_linear[:, tx_index : tx_index + 1, :],
            )
            contribution = np.einsum(
                "...rtf,...tsf->...rsf",
                h_ul[:, tx_index : tx_index + 1, :, :, :, :],
                tx_grid,
                optimize=True,
            )
            y_clean_shared[:, frame, :, :, :, :] += contribution[:, 0, :, :, :, :]

    noise_mode = noise_mode_from_config(power_config)
    thermal_meta = thermal_noise_metadata(
        power_config=power_config,
        default_bandwidth_hz=sc_spacing_hz * num_subcarriers,
    )
    impairment_chain = ObservationImpairmentChain(
        fft_size=num_subcarriers,
        sample_rate_hz=sc_spacing_hz * num_subcarriers,
        random_seed=int(getattr(phy_config, "observation_seed", 42)) + 10007,
        impairment_config=getattr(phy_config, "impairment_config", None),
        noise_mode=noise_mode,
        thermal_noise_power_mw=thermal_meta["thermal_noise_mw"],
        thermal_noise_config=thermal_meta,
    )
    impairment_result = impairment_chain.apply(y_clean_shared, snr_db=snr_db)
    rx_grid_shared = (
        impairment_result.rx_grid.detach().cpu().numpy().astype(np.complex64, copy=False)
    )
    chain_scalars = ResultAssembler.chain_scalars_to_numpy(impairment_result)

    max_re = _max_resource_re(resources)
    max_alloc_sc = _max_allocated_subcarriers(resources)
    first_resource = next(resource for row in resources for resource in row if resource is not None)
    num_ports = int(first_resource.pilot_symbols.shape[0])
    re_symbols = np.full((frame_count, active_count, max_re), -1, dtype=np.int32)
    re_subcarriers = np.full((frame_count, active_count, max_re), -1, dtype=np.int32)
    re_mask = np.zeros((frame_count, active_count, max_re), dtype=np.bool_)
    allocated_indices = np.full((frame_count, active_count, max_alloc_sc), -1, dtype=np.int32)
    allocated_mask = np.zeros((frame_count, active_count, max_alloc_sc), dtype=np.bool_)
    prb_start = np.full(
        (frame_count, active_count, int(srs_resource_config.num_srs_symbols)),
        -1,
        dtype=np.int32,
    )
    prb_count = np.zeros_like(prb_start)
    cfr_resource = np.zeros(
        (num_snap, frame_count, active_count, rx_count, rx_ant, num_ports, max_re),
        dtype=np.complex64,
    )
    cfr_allocated = np.zeros(
        (num_snap, frame_count, active_count, rx_count, rx_ant, tx_ant, max_alloc_sc),
        dtype=np.complex64,
    )

    for frame in range(frame_count):
        rx_frame = rx_grid_shared[:, frame : frame + 1, :, :, :, :]
        for active_slot, resource in enumerate(resources[frame]):
            if resource is None:
                continue
            tx_index = int(active_tx_indices[frame, active_slot])
            re_count = int(resource.re_subcarrier_indices.size)
            re_symbols[frame, active_slot, :re_count] = resource.re_symbol_indices
            re_subcarriers[frame, active_slot, :re_count] = resource.re_subcarrier_indices
            re_mask[frame, active_slot, :re_count] = True
            symbol_count = int(resource.prb_start_per_symbol.size)
            prb_start[frame, active_slot, :symbol_count] = resource.prb_start_per_symbol
            prb_count[frame, active_slot, :symbol_count] = resource.prb_count_per_symbol
            allocated = _allocated_subcarrier_indices(resource)
            allocated_count = int(allocated.size)
            allocated_indices[frame, active_slot, :allocated_count] = allocated
            allocated_mask[frame, active_slot, :allocated_count] = True

            estimated_resource = _estimate_srs_resource_cfr(
                rx_grid=rx_frame,
                srs_resource=resource,
                power_scale_linear=power_scale_linear[:, tx_index : tx_index + 1, :],
            )
            cfr_resource[:, frame, active_slot, :, :, :, :re_count] = estimated_resource[
                :, 0, :, :, :, :
            ]
            estimated_full = _interpolate_resource_cfr(
                estimated_resource,
                srs_resource=resource,
                num_subcarriers=num_subcarriers,
                num_tx_ant=tx_ant,
            )
            cfr_allocated[:, frame, active_slot, :, :, :, :allocated_count] = np.take(
                estimated_full[:, 0, :, :, :, :],
                allocated,
                axis=-1,
            )

    return MultiUserSRSResult(
        resource_strategy=strategy,
        rx_grid_shared=rx_grid_shared,
        noise_variance=chain_scalars["noise_variance"],
        snr_db=chain_scalars["snr_db"],
        rssi_dbm=chain_scalars["rssi_dbm"],
        noise_power_dbm=chain_scalars["noise_power_dbm"],
        active_tx_indices=active_tx_indices,
        active_tx_mask=active_tx_mask,
        comb_offset=comb_offsets,
        prb_start=prb_start,
        prb_count=prb_count,
        re_symbol_indices=re_symbols,
        re_subcarrier_indices=re_subcarriers,
        re_mask=re_mask,
        allocated_subcarrier_indices=allocated_indices,
        allocated_subcarrier_mask=allocated_mask,
        resource_occupancy_count=occupancy,
        resource_collision_mask=occupancy > 1,
        cfr_est_resource=cfr_resource,
        cfr_est_allocated=cfr_allocated,
    )


def _multiuser_resource_config(
    base: Any,
    *,
    strategy: str,
    active_slot: int,
    active_count: int,
    default_num_prb: int,
) -> Any:
    if strategy == "comb_offset":
        if active_count > int(base.comb_size):
            raise ValueError(
                "comb_offset multi-UE SRS requires active_ue_count <= comb_size"
            )
        comb_offset = (int(base.comb_offset) + active_slot) % int(base.comb_size)
        return replace(base, comb_offset=comb_offset)

    if strategy != "prb_split":
        raise ValueError("Unsupported multi-UE SRS resource strategy")
    base_count = int(base.bwp_num_prb or default_num_prb)
    if active_count > base_count:
        raise ValueError("prb_split multi-UE SRS requires active_ue_count <= BWP PRB count")
    start_offset, count = _split_prb_allocation(base_count, active_slot, active_count)
    return replace(
        base,
        bwp_start_prb=int(base.bwp_start_prb) + start_offset,
        bwp_num_prb=count,
    )


def _split_prb_allocation(total_prb: int, active_slot: int, active_count: int) -> tuple[int, int]:
    base = total_prb // active_count
    remainder = total_prb % active_count
    count = base + (1 if active_slot < remainder else 0)
    start = active_slot * base + min(active_slot, remainder)
    return int(start), int(count)


def _max_resource_re(resources: list[list[NRSRSResource | None]]) -> int:
    return max(
        int(resource.re_subcarrier_indices.size)
        for row in resources
        for resource in row
        if resource is not None
    )


def _max_allocated_subcarriers(resources: list[list[NRSRSResource | None]]) -> int:
    return max(
        int(_allocated_subcarrier_indices(resource).size)
        for row in resources
        for resource in row
        if resource is not None
    )


def _allocated_subcarrier_indices(resource: NRSRSResource) -> np.ndarray:
    values: list[np.ndarray] = []
    for start_prb, count_prb in zip(
        resource.prb_start_per_symbol,
        resource.prb_count_per_symbol,
        strict=True,
    ):
        start = int(start_prb) * 12
        stop = start + int(count_prb) * 12
        values.append(np.arange(start, stop, dtype=np.int32))
    if not values:
        return np.zeros((0,), dtype=np.int32)
    return np.unique(np.concatenate(values)).astype(np.int32, copy=False)


def _estimate_srs_resource_cfr(
    *,
    rx_grid: np.ndarray,
    srs_resource: NRSRSResource,
    power_scale_linear: np.ndarray,
) -> np.ndarray:
    rx_grid = np.asarray(rx_grid, dtype=np.complex64)
    link_shape = rx_grid.shape[:3]
    num_rx_ant = rx_grid.shape[3]
    num_ports = srs_resource.pilot_symbols.shape[0]
    h_hat = np.empty(
        (*link_shape, num_rx_ant, num_ports, srs_resource.re_subcarrier_indices.size),
        dtype=np.complex64,
    )
    if srs_resource.config.cyclic_shift_multiplexing == "time":
        return _estimate_time_multiplexed_resource_cfr(
            rx_grid=rx_grid,
            srs_resource=srs_resource,
            power_scale_linear=power_scale_linear,
            output=h_hat,
        )
    return _estimate_cyclic_shift_resource_cfr(
        rx_grid=rx_grid,
        srs_resource=srs_resource,
        power_scale_linear=power_scale_linear,
        output=h_hat,
    )


def _estimate_time_multiplexed_resource_cfr(
    *,
    rx_grid: np.ndarray,
    srs_resource: NRSRSResource,
    power_scale_linear: np.ndarray,
    output: np.ndarray,
) -> np.ndarray:
    output.fill(0.0)
    scale = np.asarray(power_scale_linear, dtype=np.float32)
    unique_subcarriers = np.unique(srs_resource.re_subcarrier_indices)
    for port in range(srs_resource.pilot_symbols.shape[0]):
        port_scale = np.maximum(scale[:, :, port], np.float32(1e-12))
        for subcarrier in unique_subcarriers:
            flat_positions = np.nonzero(srs_resource.re_subcarrier_indices == subcarrier)[0]
            symbols = srs_resource.re_symbol_indices[flat_positions]
            pilots = srs_resource.pilot_symbols[port, symbols, int(subcarrier)]
            ref = (
                pilots[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
                * port_scale[:, :, np.newaxis, np.newaxis, np.newaxis]
            )
            rx_values = rx_grid[..., :, symbols, int(subcarrier)]
            numerator = np.sum(rx_values * np.conjugate(ref), axis=-1)
            denom = np.sum(np.abs(ref) ** 2, axis=-1).astype(np.float32)
            est = numerator / np.maximum(denom, np.float32(1e-30))
            output[..., port, flat_positions] = est[..., np.newaxis]
    return output


def _estimate_cyclic_shift_resource_cfr(
    *,
    rx_grid: np.ndarray,
    srs_resource: NRSRSResource,
    power_scale_linear: np.ndarray,
    output: np.ndarray,
) -> np.ndarray:
    output.fill(0.0)
    scale = np.asarray(power_scale_linear, dtype=np.float32)
    window_cache: dict[int, np.ndarray] = {}
    for port in range(srs_resource.pilot_symbols.shape[0]):
        port_scale = np.maximum(scale[:, :, port], np.float32(1e-12))
        for local_symbol, symbol in enumerate(srs_resource.srs_symbol_indices):
            if int(srs_resource.port_tx_ant_map[port, local_symbol]) < 0:
                continue
            flat_positions = np.nonzero(srs_resource.re_symbol_indices == int(symbol))[0]
            re_indices = srs_resource.re_subcarrier_indices[flat_positions]
            rx_re = rx_grid[..., :, int(symbol), re_indices]
            unit_pilot = srs_resource.pilot_symbols[port, int(symbol), re_indices]
            mixed = (
                rx_re
                * np.conjugate(unit_pilot)[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
                / port_scale[:, :, np.newaxis, np.newaxis, np.newaxis]
            )
            n_re = int(re_indices.size)
            if n_re not in window_cache:
                window_cache[n_re] = _cyclic_shift_delay_window(
                    srs_resource.cyclic_shift_indices,
                    n_re,
                )
            delay = np.fft.ifft(mixed, axis=-1)
            estimate = np.fft.fft(delay * window_cache[n_re], axis=-1).astype(
                np.complex64,
                copy=False,
            )
            output[..., port, flat_positions] = estimate
    return output


def _interpolate_resource_cfr(
    resource_cfr: np.ndarray,
    *,
    srs_resource: NRSRSResource,
    num_subcarriers: int,
    num_tx_ant: int,
) -> np.ndarray:
    resource_cfr = np.asarray(resource_cfr, dtype=np.complex64)
    freq = np.arange(num_subcarriers, dtype=np.float32)
    subcarriers = np.asarray(srs_resource.re_subcarrier_indices, dtype=np.int32)
    local_symbols = _resource_local_symbol_indices(srs_resource)
    flat = resource_cfr.reshape((-1, resource_cfr.shape[-2], resource_cfr.shape[-1]))
    out = np.zeros((flat.shape[0], num_tx_ant, num_subcarriers), dtype=np.complex64)
    for row_idx, row in enumerate(flat):
        for tx_ant in range(num_tx_ant):
            point_subcarriers: list[np.ndarray] = []
            point_values: list[np.ndarray] = []
            for port in range(row.shape[0]):
                active = srs_resource.port_tx_ant_map[port, local_symbols] == tx_ant
                if not np.any(active):
                    continue
                point_subcarriers.append(subcarriers[active])
                point_values.append(row[port, active])
            if not point_subcarriers:
                continue
            x_all = np.concatenate(point_subcarriers).astype(np.int32, copy=False)
            y_all = np.concatenate(point_values).astype(np.complex64, copy=False)
            order = np.argsort(x_all, kind="stable")
            x_sorted = x_all[order]
            y_sorted = y_all[order]
            unique_x, inverse = np.unique(x_sorted, return_inverse=True)
            y_unique = np.zeros(unique_x.shape, dtype=np.complex64)
            counts = np.zeros(unique_x.shape, dtype=np.float32)
            np.add.at(y_unique, inverse, y_sorted)
            np.add.at(counts, inverse, 1.0)
            y_unique /= np.maximum(counts, np.float32(1.0))
            if unique_x.size == 1:
                out[row_idx, tx_ant, :] = y_unique[0]
            else:
                x_float = unique_x.astype(np.float32)
                real = np.interp(freq, x_float, y_unique.real).astype(np.float32)
                imag = np.interp(freq, x_float, y_unique.imag).astype(np.float32)
                out[row_idx, tx_ant, :] = real + 1j * imag
    return out.reshape((*resource_cfr.shape[:-2], num_tx_ant, num_subcarriers))


def _reference_pilot_symbols(srs_resource: NRSRSResource) -> np.ndarray:
    first_port = srs_resource.pilot_symbols[0]
    return np.asarray(
        first_port[srs_resource.re_symbol_indices, srs_resource.re_subcarrier_indices],
        dtype=np.complex64,
    )


def _truth_resource_for_ports(
    truth: np.ndarray,
    srs_resource: NRSRSResource,
) -> np.ndarray:
    truth = np.asarray(truth, dtype=np.complex64)
    out = np.zeros(
        (
            *truth.shape[:4],
            srs_resource.pilot_symbols.shape[0],
            srs_resource.re_subcarrier_indices.size,
        ),
        dtype=np.complex64,
    )
    local_symbols = _resource_local_symbol_indices(srs_resource)
    for port in range(srs_resource.pilot_symbols.shape[0]):
        for flat_idx, local_symbol in enumerate(local_symbols):
            tx_ant = int(srs_resource.port_tx_ant_map[port, local_symbol])
            if tx_ant < 0:
                continue
            subcarrier = int(srs_resource.re_subcarrier_indices[flat_idx])
            out[..., port, flat_idx] = truth[..., tx_ant, subcarrier]
    return out


def _resource_local_symbol_indices(srs_resource: NRSRSResource) -> np.ndarray:
    local_by_symbol = {
        int(symbol): local_idx for local_idx, symbol in enumerate(srs_resource.srs_symbol_indices)
    }
    return np.asarray(
        [local_by_symbol[int(symbol)] for symbol in srs_resource.re_symbol_indices],
        dtype=np.int32,
    )


def _cyclic_shift_delay_window(cyclic_shifts: np.ndarray, n_re: int) -> np.ndarray:
    if n_re < 1:
        raise ValueError("SRS resource must have at least one RE")
    shifts = np.asarray(cyclic_shifts, dtype=np.int32)
    if shifts.size <= 1:
        width = n_re
    else:
        distances: list[int] = []
        for idx, left in enumerate(shifts):
            for right in shifts[idx + 1:]:
                delta = int(abs(int(left) - int(right))) % 12
                if delta == 0:
                    continue
                bin_delta = int(round(n_re * min(delta, 12 - delta) / 12.0))
                if bin_delta > 0:
                    distances.append(bin_delta)
        width = max(1, min(distances) // 2) if distances else max(1, n_re // 8)
    window = np.zeros((n_re,), dtype=np.complex64)
    window[: min(width, n_re)] = 1.0 + 0.0j
    return window


def _build_valid_mask(
    has_signal: np.ndarray | None,
    link_shape: tuple[int, int, int],
) -> np.ndarray:
    if has_signal is None:
        return np.ones(link_shape, dtype=np.bool_)
    link_mask = np.asarray(has_signal, dtype=np.bool_)
    if link_mask.shape != link_shape[1:]:
        raise ValueError(f"has_signal must have shape {link_shape[1:]}, got {link_mask.shape}")
    return np.broadcast_to(link_mask[np.newaxis, ...], link_shape).copy()


def _estimate_metrics(
    truth: np.ndarray,
    estimate: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    error = estimate - truth
    signal_power = np.sum(np.abs(truth) ** 2, axis=(3, 4, 5))
    error_power = np.sum(np.abs(error) ** 2, axis=(3, 4, 5))
    nmse_linear = error_power / np.maximum(signal_power, 1e-30)
    nmse_db = 10.0 * np.log10(np.maximum(nmse_linear, 1e-30)).astype(np.float32)
    amplitude_error_db = 20.0 * np.log10(
        np.maximum(np.mean(np.abs(error), axis=(3, 4, 5)), 1e-30)
    ).astype(np.float32)
    phase_error_rad = np.mean(np.angle(estimate * np.conjugate(truth)), axis=(3, 4, 5)).astype(
        np.float32
    )
    numerator = np.abs(np.sum(np.conjugate(truth) * estimate, axis=(3, 4, 5)))
    denominator = np.sqrt(signal_power) * np.sqrt(np.sum(np.abs(estimate) ** 2, axis=(3, 4, 5)))
    correlation = (numerator / np.maximum(denominator, 1e-30)).astype(np.float32)
    for array in (nmse_db, amplitude_error_db, phase_error_rad, correlation):
        array[~valid_mask] = 0.0
    return nmse_db, amplitude_error_db, phase_error_rad, correlation
