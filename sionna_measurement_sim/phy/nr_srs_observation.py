"""Standards-shaped NR SRS subset observation path."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np

from sionna_measurement_sim.domain.link import LinkConfig
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
from sionna_measurement_sim.phy.spatial_spectrum import project_cfr_to_ul_receiver_samples


def run_nr_srs_observation(
    truth_cfr: np.ndarray,
    link_config: LinkConfig,
    phy_config: Any,
    carrier_config: Any,
    *,
    cfr_snapshots: np.ndarray | None = None,
    has_signal: np.ndarray | None = None,
    tracer: Any | None = None,
) -> dict[str, Any]:
    """Run standards-shaped NR SRS resource sounding and LS CFR estimation.

    This is a standards-shaped subset rather than a complete 3GPP NR SRS
    implementation. Stage 1 supports comb/BWP resource mapping, full-slot time
    allocation, a deterministic ZC-like pilot, and simplified cyclic-shift
    metadata while port separation still uses time-symbol orthogonality.
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
        num_ports=num_ul_tx_ant,
    )
    srs_resource = build_srs_resource(
        srs_resource_config,
        num_subcarriers=num_subcarriers,
        num_ports=num_ul_tx_ant,
        default_num_prb=int(getattr(phy_config, "num_prb", max(1, num_subcarriers // 12))),
    )
    num_symbols = int(srs_resource_config.slot_length_symbols)
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
        )
    _record_array_event(tracer, "nr_srs.array_shape", "h_ul", h_ul)
    _record_array_event(tracer, "nr_srs.array_shape", "tx_grid", tx_grid)
    with _span(tracer, "nr_srs.channel_apply_einsum"):
        y_clean = np.einsum("...rtf,...tsf->...rsf", h_ul, tx_grid, optimize=True)
    _record_array_event(tracer, "nr_srs.array_shape", "y_clean", y_clean)
    with _span(tracer, "nr_srs.common_impairments_and_awgn", snr_db=float(snr_db)):
        impairment_chain = ObservationImpairmentChain(
            fft_size=num_subcarriers,
            sample_rate_hz=sc_spacing_hz * num_subcarriers,
            random_seed=int(getattr(phy_config, "observation_seed", 42)),
            impairment_config=getattr(phy_config, "impairment_config", None),
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
        )
    with _span(tracer, "nr_srs.interpolate_full_band"):
        h_hat_ul = _interpolate_resource_cfr(
            h_hat_resource,
            resource_indices=srs_resource.re_subcarrier_indices,
            num_subcarriers=num_subcarriers,
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
        truth_resource = truth_dl[..., srs_resource.re_subcarrier_indices]
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
            "srs_resource_mask": srs_resource.resource_mask,
            "srs_pilot_symbols": srs_resource.pilot_symbols,
            "srs_port_index": srs_resource.port_index,
            "srs_re_subcarrier_indices": srs_resource.re_subcarrier_indices,
            "srs_symbol_indices": srs_resource.srs_symbol_indices,
            "srs_cyclic_shift_indices": srs_resource.cyclic_shift_indices,
            "cfr_est_resource": h_hat_resource.astype(np.complex64, copy=False),
            "srs_resource_nmse_db": resource_nmse_db,
            "srs_interpolation_nmse_db": nmse_db,
            "srs_resource_snr_db": chain_scalars["snr_db"],
            "srs_num_resource_elements": np.int32(
                srs_resource.re_subcarrier_indices.size
                * srs_resource.srs_symbol_indices.size
            ),
        }
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
        "metadata": {
            "srs_scope": "standards_shaped_subset_stage1",
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
) -> np.ndarray:
    num_symbols, num_subcarriers = srs_resource.resource_mask.shape
    tx_grid = np.zeros(
        (*link_shape, num_ul_tx_ant, num_symbols, num_subcarriers),
        dtype=np.complex64,
    )
    for tx_ant in range(num_ul_tx_ant):
        port = int(srs_resource.port_index[tx_ant])
        tx_grid[..., tx_ant, :, :] = srs_resource.pilot_symbols[port]
    return tx_grid


def _estimate_srs_resource_cfr(
    *,
    rx_grid: np.ndarray,
    srs_resource: NRSRSResource,
) -> np.ndarray:
    rx_grid = np.asarray(rx_grid, dtype=np.complex64)
    srs_symbols = srs_resource.srs_symbol_indices
    re_indices = srs_resource.re_subcarrier_indices
    rx_srs = rx_grid[..., :, srs_symbols, :]
    rx_re = rx_srs[..., re_indices]
    link_shape = rx_grid.shape[:3]
    num_rx_ant = rx_grid.shape[3]
    num_tx_ant = srs_resource.port_index.size
    h_hat = np.empty(
        (*link_shape, num_rx_ant, num_tx_ant, re_indices.size),
        dtype=np.complex64,
    )
    for tx_ant in range(num_tx_ant):
        port = int(srs_resource.port_index[tx_ant])
        pilots = srs_resource.pilot_symbols[
            port,
            srs_symbols[:, np.newaxis],
            re_indices[np.newaxis, :],
        ]
        denom = np.sum(np.abs(pilots) ** 2, axis=0).astype(np.float32)
        numerator = np.sum(
            rx_re
            * np.conjugate(pilots)[
                np.newaxis,
                np.newaxis,
                np.newaxis,
                np.newaxis,
                :,
                :,
            ],
            axis=-2,
        )
        h_hat[..., tx_ant, :] = numerator / np.maximum(denom, np.float32(1e-30))
    return h_hat


def _interpolate_resource_cfr(
    resource_cfr: np.ndarray,
    *,
    resource_indices: np.ndarray,
    num_subcarriers: int,
) -> np.ndarray:
    resource_cfr = np.asarray(resource_cfr, dtype=np.complex64)
    resource_indices = np.asarray(resource_indices, dtype=np.int32)
    if resource_indices.size == 0:
        raise ValueError("resource_indices must not be empty")
    if resource_indices.size == num_subcarriers and np.array_equal(
        resource_indices,
        np.arange(num_subcarriers, dtype=np.int32),
    ):
        return resource_cfr.copy()

    freq = np.arange(num_subcarriers, dtype=np.float32)
    x = resource_indices.astype(np.float32)
    flat = resource_cfr.reshape((-1, resource_indices.size))
    out = np.empty((flat.shape[0], num_subcarriers), dtype=np.complex64)
    if resource_indices.size == 1:
        out[...] = flat[:, :1]
    else:
        for row_idx, row in enumerate(flat):
            real = np.interp(freq, x, row.real).astype(np.float32)
            imag = np.interp(freq, x, row.imag).astype(np.float32)
            out[row_idx] = real + 1j * imag
    return out.reshape((*resource_cfr.shape[:-1], num_subcarriers))


def _reference_pilot_symbols(srs_resource: NRSRSResource) -> np.ndarray:
    first_port = srs_resource.pilot_symbols[0]
    first_symbol = int(srs_resource.srs_symbol_indices[0])
    return np.asarray(
        first_port[first_symbol, srs_resource.re_subcarrier_indices],
        dtype=np.complex64,
    )


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
