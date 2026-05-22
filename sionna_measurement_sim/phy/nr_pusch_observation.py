"""NR PUSCH observation backend with full MIMO support.

Builds Sionna PUSCHConfig(s) from project config, runs PUSCHTransmitter
and PUSCHReceiver with RT CIR via ApplyOFDMChannel, and computes real
BER/BLER.  Supports SU-MIMO 4x4 and lays groundwork for MU-MIMO.

Per-channel-estimator CFR is written to /observation/cfr_est without
broadcasting from a single SISO estimate.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

from sionna_measurement_sim.domain.array import ArraySpectrumConfig
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
from sionna_measurement_sim.phy.nr_channel_backend import (
    create_channel_backend,
)
from sionna_measurement_sim.phy.nr_mimo_channel import (
    pusch_h_to_cfr_est,
    reverse_reciprocity_cfr,
)
from sionna_measurement_sim.phy.spatial_spectrum import (
    build_angle_grid_rad,
    build_aoa_heatmap_label,
    build_bartlett_spectrum,
    build_rx_snapshot_matrix,
)

# ── PUSCH config helpers ────────────────────────────────────────────────


@contextmanager
def _torch_default_device(device_name: str):
    requested = str(device_name or "cpu").strip()
    if requested in ("", "cpu"):
        yield torch.device("cpu")
        return
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"runtime.device={requested!r} requires CUDA, but torch.cuda.is_available() is false"
        )

    previous = torch.get_default_device()
    torch.set_default_device(requested)
    try:
        yield torch.device(requested)
    finally:
        torch.set_default_device(previous)


def build_multiuser_pusch_configs(
    phy_config,
    carrier_config,
    *,
    num_pusch_tx: int | None = None,
) -> list[Any]:
    """Build a list of Sionna PUSCHConfig objects for multi-UE uplink.

    For SU-MIMO the list contains a single PUSCHConfig.  For MU-MIMO
    the list contains one config per UE with non-overlapping DMRS port sets.

    Parameters
    ----------
    num_pusch_tx : int or None
        Number of PUSCH transmitters (UEs).  If None, defaults to 1 (SU-MIMO).
    """
    from sionna.phy.nr import CarrierConfig as SionnaCarrierConfig
    from sionna.phy.nr import PUSCHConfig as SionnaPUSCHConfig
    from sionna.phy.nr import PUSCHDMRSConfig, TBConfig

    num_tx = num_pusch_tx if num_pusch_tx is not None else 1

    num_layers = phy_config.num_layers
    num_antenna_ports = phy_config.num_antenna_ports

    if num_layers < 1:
        raise ValueError(f"num_layers must be >= 1, got {num_layers}")
    if num_tx < 1:
        raise ValueError(f"num_tx must be >= 1, got {num_tx}")
    if num_antenna_ports < num_layers:
        raise ValueError(
            f"num_antenna_ports ({num_antenna_ports}) must be >= num_layers ({num_layers})"
        )
    total_ports_needed = num_tx * num_layers
    if total_ports_needed > 12:
        raise ValueError(
            f"num_tx * num_layers = {total_ports_needed} exceeds max DMRS ports (12)"
        )

    carrier = SionnaCarrierConfig(
        n_size_grid=phy_config.num_prb,
        subcarrier_spacing=phy_config.subcarrier_spacing_khz or 30,
    )
    tb = TBConfig(
        mcs_index=phy_config.mcs_index,
        mcs_table=phy_config.mcs_table,
    )

    configs: list[Any] = []
    for ue_idx in range(num_tx):
        dmrs = PUSCHDMRSConfig(
            config_type=phy_config.pusch_dmrs_config_type,
            length=phy_config.pusch_dmrs_length,
            additional_position=phy_config.pusch_dmrs_additional_position,
            num_cdm_groups_without_data=phy_config.pusch_num_cdm_groups_without_data,
        )
        dmrs.dmrs_port_set = list(
            range(ue_idx * num_layers, (ue_idx + 1) * num_layers)
        )

        pusch_kwargs: dict[str, Any] = {
            "num_layers": num_layers,
            "num_antenna_ports": num_antenna_ports,
        }
        if num_layers < num_antenna_ports:
            pusch_kwargs["precoding"] = "codebook"

        pc = SionnaPUSCHConfig(
            carrier_config=carrier,
            pusch_dmrs_config=dmrs,
            tb_config=tb,
            **pusch_kwargs,
        )
        configs.append(pc)

    return configs


def build_nr_pusch_config(phy_config, carrier_config) -> Any:
    """Build a single Sionna PUSCHConfig (backward-compatible wrapper)."""
    return build_multiuser_pusch_configs(phy_config, carrier_config)[0]


def pusch_config_to_dict(pusch_config) -> dict[str, Any]:
    """Convert a Sionna PUSCHConfig to a JSON-serialisable dict snapshot."""
    return {
        "num_layers": pusch_config.num_layers,
        "num_antenna_ports": pusch_config.num_antenna_ports,
        "num_resource_blocks": pusch_config.num_resource_blocks,
        "num_subcarriers": int(pusch_config.num_subcarriers),
        "num_coded_bits": pusch_config.num_coded_bits,
        "tb_size": pusch_config.tb_size,
        "mcs_index": pusch_config.tb.mcs_index,
        "mcs_table": pusch_config.tb.mcs_table,
        "target_coderate": float(pusch_config.tb.target_coderate),
        "num_bits_per_symbol": int(pusch_config.tb.num_bits_per_symbol),
        "dmrs_config_type": pusch_config.dmrs.config_type,
        "dmrs_length": pusch_config.dmrs.length,
        "dmrs_additional_position": pusch_config.dmrs.additional_position,
        "num_cdm_groups_without_data": pusch_config.dmrs.num_cdm_groups_without_data,
        "carrier_subcarrier_spacing": pusch_config.carrier.subcarrier_spacing,
        "carrier_n_size_grid": pusch_config.carrier.n_size_grid,
        "carrier_num_symbols_per_slot": pusch_config.carrier.num_symbols_per_slot,
        "transform_precoding": pusch_config.transform_precoding,
        "precoding": pusch_config.precoding,
        "mapping_type": pusch_config.mapping_type,
        "frequency_hopping": pusch_config.frequency_hopping,
    }


# ── Stream management and detector ──────────────────────────────────────


def build_stream_management(num_rx: int, num_tx: int, num_layers: int) -> Any:
    """Build Sionna StreamManagement for MIMO uplink."""
    from sionna.phy.mimo import StreamManagement

    rx_tx_association = np.ones([num_rx, num_tx], dtype=bool)
    return StreamManagement(rx_tx_association, num_layers)


def build_mimo_detector(
    resource_grid,
    stream_management,
    detector_type: str = "lmmse",
    num_bits_per_symbol: int = 4,
) -> Any:
    """Build a Sionna OFDM MIMO detector."""
    if detector_type == "lmmse":
        from sionna.phy.ofdm import LinearDetector

        return LinearDetector(
            "lmmse", "bit", "maxlog",
            resource_grid, stream_management,
            "qam", num_bits_per_symbol,
        )
    elif detector_type == "kbest":
        from sionna.phy.ofdm import KBestDetector

        num_tx = stream_management.num_tx
        num_layers = stream_management.num_streams_per_tx
        return KBestDetector(
            "bit", num_tx * num_layers, 64,  # output, num_streams, k
            resource_grid, stream_management,
            "qam", num_bits_per_symbol,
        )
    else:
        raise ValueError(
            f"Unknown mimo_detector: {detector_type!r}. "
            f"Supported: 'lmmse', 'kbest'"
        )


# ── main entry point ────────────────────────────────────────────────────


def run_nr_pusch_observation(
    cir_coefficients: np.ndarray,
    cir_delays: np.ndarray,
    link_config: LinkConfig,
    phy_config,
    carrier_config,
) -> dict:
    device = getattr(phy_config, "device", "cpu")
    with _torch_default_device(device):
        return _run_nr_pusch_observation_impl(
            cir_coefficients=cir_coefficients,
            cir_delays=cir_delays,
            link_config=link_config,
            phy_config=phy_config,
            carrier_config=carrier_config,
        )


def _run_nr_pusch_observation_impl(
    cir_coefficients: np.ndarray,
    cir_delays: np.ndarray,
    link_config: LinkConfig,
    phy_config,
    carrier_config,
) -> dict:
    """Run NR PUSCH uplink observation with full MIMO channel.

    Parameters
    ----------
    cir_coefficients : np.ndarray
        6-D ``[snap, tx, rx, rx_ant, tx_ant, path]`` complex CIR.
    cir_delays : np.ndarray
        6-D ``[snap, tx, rx, rx_ant, tx_ant, path]`` CIR delays in seconds.
    link_config : LinkConfig
    phy_config : PHYConfig or RTTruthRunConfig
    carrier_config : CarrierConfig

    Returns
    -------
    dict
    """
    torch.manual_seed(int(getattr(phy_config, "seed", 42)))
    # 0. Extract config  ──────────────────────────────────────────────
    sc_spacing_hz = float(phy_config.subcarrier_spacing_khz) * 1000.0
    num_prb = phy_config.num_prb
    num_subcarriers = num_prb * 12
    num_ofdm_symbols = getattr(phy_config, "num_ofdm_symbols", 14)
    perfect_csi = getattr(phy_config, "perfect_csi", False)
    mimo_detector_type = getattr(phy_config, "mimo_detector", "lmmse")
    channel_estimator_type = getattr(phy_config, "channel_estimator", "pusch_ls")
    receiver_failure_policy = getattr(phy_config, "receiver_failure_policy", "fail_fast")
    requested_batch_size = _get_nr_pusch_batch_size(phy_config)
    snr_db = getattr(phy_config, "snr_db", None)
    if snr_db is None:
        snr_db = getattr(phy_config, "observation_snr_db", 30.0)

    # 0a. Validate MIMO mode  ────────────────────────────────────────
    mimo_mode = getattr(phy_config, "mimo_mode", "su_mimo")
    if mimo_mode not in ("su_mimo", "mu_mimo"):
        raise NotImplementedError(
            f"mimo_mode={mimo_mode!r} is not supported. "
            f"Supported: 'su_mimo', 'mu_mimo'."
        )

    # 1. Build channel backend from CIR  ──────────────────────────────
    channel_backend_name = getattr(phy_config, "channel_backend", "apply_ofdm")
    backend = create_channel_backend(
        cir_coefficients, cir_delays, link_config,
        sc_spacing_hz, num_subcarriers,
        backend_name=channel_backend_name,
    )

    # 1a. Reject unsupported mode/backend combinations  ───────────────
    if mimo_mode == "mu_mimo" and channel_backend_name == "cir_dataset_ofdm":
        raise NotImplementedError(
            "mu_mimo + cir_dataset_ofdm is not yet supported. "
            "Use channel_backend='apply_ofdm' for MU-MIMO."
        )

    cfr_clean_ref = backend.cfr.copy()
    num_snap = backend.num_snap
    num_ul_tx = backend.num_ul_tx
    num_ul_rx = backend.num_ul_rx

    # 2. Build PUSCH configs — auto-derive num_pusch_tx for MU-MIMO ──
    _pusch_tx = num_ul_tx if mimo_mode == "mu_mimo" else None
    pusch_configs = build_multiuser_pusch_configs(
        phy_config, carrier_config, num_pusch_tx=_pusch_tx,
    )
    _num_pusch_tx = len(pusch_configs)
    _num_layers = pusch_configs[0].num_layers
    _num_antenna_ports = pusch_configs[0].num_antenna_ports

    from sionna.phy.nr import PUSCHTransmitter

    tx = PUSCHTransmitter(pusch_configs, output_domain="freq", return_bits=True)
    num_ofdm_symbols = int(tx.resource_grid.num_ofdm_symbols)

    # 3. StreamManagement and MIMO detector ──────────────────────────
    if mimo_mode == "mu_mimo":
        stream_mgmt = build_stream_management(
            num_rx=num_ul_rx, num_tx=num_ul_tx, num_layers=_num_layers,
        )
    else:
        stream_mgmt = build_stream_management(
            num_rx=1, num_tx=_num_pusch_tx, num_layers=_num_layers,
        )
    num_bits_per_symbol = int(pusch_configs[0].tb.num_bits_per_symbol)
    mimo_detector = build_mimo_detector(
        tx.resource_grid, stream_mgmt,
        detector_type=mimo_detector_type,
        num_bits_per_symbol=num_bits_per_symbol,
    )

    # 4. Build PUSCH receiver ────────────────────────────────────────
    from sionna.phy.nr import PUSCHReceiver

    rx = PUSCHReceiver(
        pusch_transmitter=tx,
        channel_estimator="perfect" if perfect_csi else None,
        mimo_detector=mimo_detector,
        stream_management=stream_mgmt,
        tb_decoder=None,
        return_tb_crc_status=True,
        input_domain="freq",
    )
    ls_estimator = None if perfect_csi else _build_pusch_ls_estimator(tx)

    # 5. Noise power ─────────────────────────────────────────────────
    ebno_db = getattr(phy_config, "ebno_db", None)
    if ebno_db is not None:
        from sionna.phy.utils import ebnodb2no

        target_coderate = float(pusch_configs[0].tb.target_coderate)
        no_val = ebnodb2no(
            ebno_db, num_bits_per_symbol, target_coderate, tx.resource_grid,
        )
        no = no_val.to(dtype=torch.float32)
    else:
        no_val = 10.0 ** (-snr_db / 10.0)
        no = torch.tensor(no_val, dtype=torch.float32)
    noise_variance_override = no if ebno_db is not None else None
    impairment_chain = ObservationImpairmentChain(
        fft_size=num_subcarriers,
        sample_rate_hz=sc_spacing_hz * num_subcarriers,
        random_seed=int(getattr(phy_config, "observation_seed", getattr(phy_config, "seed", 42))),
        impairment_config=getattr(phy_config, "impairment_config", None),
    )

    # 6. Process links ───────────────────────────────────────────────
    if mimo_mode == "mu_mimo":
        proc_result = _process_mu_mimo(
            backend=backend, tx=tx, rx=rx,
            noise_variance_override=noise_variance_override,
            impairment_chain=impairment_chain,
            snr_db=float(snr_db),
            ls_estimator=ls_estimator,
            perfect_csi=perfect_csi,
            num_ofdm_symbols=num_ofdm_symbols,
            receiver_failure_policy=receiver_failure_policy,
            cfr_clean_ref=cfr_clean_ref,
            num_snap=num_snap, num_ul_tx=num_ul_tx, num_ul_rx=num_ul_rx,
        )
    elif requested_batch_size > 1:
        proc_result = _process_su_mimo_batched(
            backend=backend, tx=tx, rx=rx,
            noise_variance_override=noise_variance_override,
            impairment_chain=impairment_chain,
            snr_db=float(snr_db),
            ls_estimator=ls_estimator,
            perfect_csi=perfect_csi,
            num_ofdm_symbols=num_ofdm_symbols,
            receiver_failure_policy=receiver_failure_policy,
            cfr_clean_ref=cfr_clean_ref,
            num_snap=num_snap, num_ul_tx=num_ul_tx, num_ul_rx=num_ul_rx,
            requested_batch_size=requested_batch_size,
        )
    else:
        proc_result = _process_su_mimo_per_link(
            backend=backend, tx=tx, rx=rx,
            noise_variance_override=noise_variance_override,
            impairment_chain=impairment_chain,
            snr_db=float(snr_db),
            ls_estimator=ls_estimator,
            perfect_csi=perfect_csi,
            num_ofdm_symbols=num_ofdm_symbols,
            receiver_failure_policy=receiver_failure_policy,
            cfr_clean_ref=cfr_clean_ref,
            num_snap=num_snap, num_ul_tx=num_ul_tx, num_ul_rx=num_ul_rx,
        )

    cfr_est_full = proc_result["cfr_est_full"]
    waveform_grids = proc_result["waveform_grids"]
    nmse_db_full = proc_result["nmse_db_full"]
    ber_per_link = proc_result["ber_per_link"]
    bler_per_link = proc_result["bler_per_link"]
    estimation_success = proc_result["estimation_success"]
    total_bit_errors = proc_result["total_bit_errors"]
    total_bits = proc_result["total_bits"]
    total_block_errors = proc_result.get("total_block_errors", 0)
    total_blocks = proc_result.get("total_blocks", 0)
    num_receiver_failures = proc_result["num_receiver_failures"]
    batching_stats = proc_result["batching_stats"]
    link_metadata = proc_result["link_metadata"]

    # 7. Keep HDF5 output in resolved link-view orientation.
    # Legacy transpose-reciprocity call sites still opt into the old reverse
    # step so their output remains aligned with their original RT view.
    if backend.reciprocity_applied:
        cfr_clean_ref = reverse_reciprocity_cfr(cfr_clean_ref)
        cfr_est_full = reverse_reciprocity_cfr(cfr_est_full)
        nmse_db_full = np.transpose(nmse_db_full, (0, 2, 1))
        ber_per_link = np.transpose(ber_per_link, (0, 2, 1))
        bler_per_link = np.transpose(bler_per_link, (0, 2, 1))
        estimation_success = np.transpose(estimation_success, (0, 2, 1))
        link_metadata = _transpose_link_metadata_for_reciprocity(link_metadata)
        link_shape = (num_snap, num_ul_rx, num_ul_tx)
    else:
        link_shape = (num_snap, num_ul_tx, num_ul_rx)

    # 8. Aggregate metrics  ───────────────────────────────────────────
    aggregate_ber = total_bit_errors / max(total_bits, 1)
    aggregate_bler = float(np.mean(bler_per_link)) if bler_per_link.size > 0 else 0.0
    impairment_spec = impairment_chain.build_spec(float(snr_db))

    observation = ObservationResult(
        cfr_est=cfr_est_full,
        valid_mask=np.ones(link_shape, dtype=np.bool_),
        detection_success=np.ones(link_shape, dtype=np.bool_),
        estimation_success=estimation_success,
        snr_db=link_metadata["snr_db"],
        rssi_dbm=link_metadata["rssi_dbm"],
        noise_power_dbm=link_metadata["noise_power_dbm"],
        cfo_hz=link_metadata["cfo_hz"],
        sfo_ppm=link_metadata["sfo_ppm"],
        timing_offset_samples=link_metadata["timing_offset_samples"],
        phase_offset_rad=link_metadata["phase_offset_rad"],
        agc_gain_db=_fit_agc_to_link_shape(link_metadata["agc_gain_db"], link_shape),
        clipping_flag=link_metadata["clipping_flag"],
    )

    evaluation = EvaluationResult(
        nmse_db=nmse_db_full,
        nmse_db_total=nmse_db_full.astype(np.float32),
        amplitude_error_db=np.zeros(link_shape, dtype=np.float32),
        phase_error_rad=np.zeros(link_shape, dtype=np.float32),
        correlation=np.ones(link_shape, dtype=np.float32),
        detection_rate=1.0,
        estimation_failure_rate=float(
            num_receiver_failures / max(np.prod(link_shape), 1)
        ),
        ber=float(aggregate_ber),
        bler=float(aggregate_bler),
        num_bit_errors=total_bit_errors,
        num_bits=total_bits,
        num_block_errors=total_block_errors,
        num_blocks=total_blocks,
    )

    # 9. WaveformSpec  ────────────────────────────────────────────────
    sample_rate_hz = sc_spacing_hz * num_subcarriers
    nr_waveform_spec = WaveformSpec(
        standard="nr_pusch",
        sample_rate_hz=sample_rate_hz,
        fft_size=num_subcarriers,
        cp_length=0,
        num_ofdm_symbols=num_ofdm_symbols,
        pilot_indices=np.array([], dtype=np.int32),
        data_subcarrier_indices=np.arange(num_subcarriers, dtype=np.int32),
        pilot_symbols=np.array([], dtype=np.complex64),
        tx_power_dbm=getattr(phy_config, "tx_power_dbm", 0.0),
    )

    # 10. Assemble result  ────────────────────────────────────────────
    result: dict[str, Any] = {
        "cfr_est": cfr_est_full,
        "cfr_clean_ref": cfr_clean_ref[0:1],
        "ber": evaluation.ber,
        "bler": evaluation.bler,
        "pusch_config": pusch_config_to_dict(pusch_configs[0]),
        "waveform_spec": nr_waveform_spec,
        "nr_waveform_spec": nr_waveform_spec,
        "receiver_spec": ReceiverSpec(
            receiver_type="pusch_receiver",
            estimator_type="perfect" if perfect_csi else channel_estimator_type,
            mimo_detector=mimo_detector_type,
            sync_method="perfect",
            failure_policy=receiver_failure_policy,
        ),
        "evaluation": evaluation,
        "observation": observation,
        "impairments": impairment_spec,
        "reciprocity_applied": backend.reciprocity_applied,
        "num_tx_bits": total_bits,
        "tx_signal_shape": waveform_grids["tx_grid"].shape,
        "waveform_grids": waveform_grids,
        "array_outputs": build_array_outputs_from_waveform(
            waveform_grids["rx_grid"],
        ),
        "batching_stats": batching_stats,
    }
    return result


# ── SU-MIMO per-link processing ────────────────────────────────────────


def _get_nr_pusch_batch_size(phy_config: Any) -> int:
    """Read optional NR PUSCH link-batch size without requiring schema changes."""
    for attr_name in (
        "nr_pusch_batch_size",
        "pusch_batch_size",
        "batch_size",
        "su_mimo_link_batch_size",
    ):
        value = getattr(phy_config, attr_name, None)
        if value is not None:
            return max(int(value), 1)
    return 1


def _build_pusch_ls_estimator(tx: Any) -> Any:
    from sionna.phy.nr import PUSCHLSChannelEstimator

    return PUSCHLSChannelEstimator(
        tx.resource_grid,
        dmrs_length=tx._dmrs_length,
        dmrs_additional_position=tx._dmrs_additional_position,
        num_cdm_groups_without_data=tx._num_cdm_groups_without_data,
        interpolation_type="lin",
    )


def _default_batching_stats(requested_batch_size: int, num_links: int) -> dict[str, int]:
    effective = min(max(int(requested_batch_size), 1), max(int(num_links), 1))
    return {
        "requested_batch_size": int(requested_batch_size),
        "effective_batch_size": effective,
        "num_links": int(num_links),
        "num_batches": int(num_links),
        "num_batch_fallbacks": 0,
        "num_single_link_fallbacks": 0,
        "num_failed_batch_attempts": 0,
    }


def _process_su_mimo_per_link(
    backend: Any,
    tx: Any,
    rx: Any,
    noise_variance_override: torch.Tensor | None,
    impairment_chain: ObservationImpairmentChain,
    snr_db: float,
    ls_estimator: Any | None,
    perfect_csi: bool,
    num_ofdm_symbols: int,
    receiver_failure_policy: str,
    cfr_clean_ref: np.ndarray,
    num_snap: int,
    num_ul_tx: int,
    num_ul_rx: int,
) -> dict:
    """Process each (snap, ul_tx, ul_rx) independently (SU-MIMO)."""
    cfr_est_full = np.zeros(cfr_clean_ref.shape, dtype=np.complex64)
    waveform_grids = _empty_waveform_grids(
        num_snap=num_snap,
        num_ul_tx=num_ul_tx,
        num_ul_rx=num_ul_rx,
        num_ul_tx_ant=backend.num_ul_tx_ant,
        num_ul_rx_ant=backend.num_ul_rx_ant,
        num_ofdm_symbols=num_ofdm_symbols,
        num_subcarriers=backend.num_subcarriers,
    )
    nmse_db_full = np.zeros((num_snap, num_ul_tx, num_ul_rx), dtype=np.float32)
    ber_per_link = np.zeros((num_snap, num_ul_tx, num_ul_rx), dtype=np.float32)
    bler_per_link = np.zeros((num_snap, num_ul_tx, num_ul_rx), dtype=np.float32)
    estimation_success = np.ones((num_snap, num_ul_tx, num_ul_rx), dtype=np.bool_)
    total_bit_errors = 0
    total_bits = 0
    total_block_errors = 0
    total_blocks = 0
    num_receiver_failures = 0
    link_metadata = _empty_link_metadata(num_snap, num_ul_tx, num_ul_rx)

    for snap_idx in range(num_snap):
        for ul_tx_idx in range(num_ul_tx):
            for ul_rx_idx in range(num_ul_rx):
                link_result = _process_one_pusch_link(
                    snap_idx=snap_idx,
                    ul_tx_idx=ul_tx_idx,
                    ul_rx_idx=ul_rx_idx,
                    backend=backend,
                    tx=tx,
                    rx=rx,
                    noise_variance_override=noise_variance_override,
                    impairment_chain=impairment_chain,
                    snr_db=snr_db,
                    ls_estimator=ls_estimator,
                    perfect_csi=perfect_csi,
                    num_ofdm_symbols=num_ofdm_symbols,
                    receiver_failure_policy=receiver_failure_policy,
                )
                cfr_est_full[
                    snap_idx, ul_tx_idx, ul_rx_idx, ...
                ] = link_result["cfr_est_slice"]
                waveform_grids["tx_grid"][
                    snap_idx, ul_tx_idx, ul_rx_idx, ...
                ] = link_result["tx_grid_slice"]
                waveform_grids["rx_grid"][
                    snap_idx, ul_tx_idx, ul_rx_idx, ...
                ] = link_result["rx_grid_slice"]
                waveform_grids["noise_variance"][
                    snap_idx, ul_tx_idx, ul_rx_idx
                ] = link_result["noise_variance"]
                _store_link_metadata(
                    link_metadata,
                    snap_idx,
                    ul_tx_idx,
                    ul_rx_idx,
                    link_result["link_metadata"],
                )
                nmse_db_full[
                    snap_idx, ul_tx_idx, ul_rx_idx
                ] = link_result["nmse_db"]
                ber_per_link[
                    snap_idx, ul_tx_idx, ul_rx_idx
                ] = link_result["ber"]
                bler_per_link[
                    snap_idx, ul_tx_idx, ul_rx_idx
                ] = link_result["bler"]
                estimation_success[
                    snap_idx, ul_tx_idx, ul_rx_idx
                ] = link_result["estimation_success"]
                total_bit_errors += link_result["num_bit_errors"]
                total_bits += link_result["num_bits"]
                total_block_errors += link_result.get("num_block_errors", 0)
                total_blocks += link_result.get("num_blocks", 0)
                if link_result.get("receiver_failed", False):
                    num_receiver_failures += 1

    return {
        "cfr_est_full": cfr_est_full,
        "waveform_grids": waveform_grids,
        "nmse_db_full": nmse_db_full,
        "ber_per_link": ber_per_link,
        "bler_per_link": bler_per_link,
        "estimation_success": estimation_success,
        "total_bit_errors": total_bit_errors,
        "total_bits": total_bits,
        "total_block_errors": total_block_errors,
        "total_blocks": total_blocks,
        "num_receiver_failures": num_receiver_failures,
        "link_metadata": link_metadata,
        "batching_stats": _default_batching_stats(
            requested_batch_size=1,
            num_links=num_snap * num_ul_tx * num_ul_rx,
        ),
    }


def _process_su_mimo_batched(
    backend: Any,
    tx: Any,
    rx: Any,
    noise_variance_override: torch.Tensor | None,
    impairment_chain: ObservationImpairmentChain,
    snr_db: float,
    ls_estimator: Any | None,
    perfect_csi: bool,
    num_ofdm_symbols: int,
    receiver_failure_policy: str,
    cfr_clean_ref: np.ndarray,
    num_snap: int,
    num_ul_tx: int,
    num_ul_rx: int,
    requested_batch_size: int,
) -> dict:
    """Process SU-MIMO links in configurable batches."""
    cfr_est_full = np.zeros(cfr_clean_ref.shape, dtype=np.complex64)
    waveform_grids = _empty_waveform_grids(
        num_snap=num_snap,
        num_ul_tx=num_ul_tx,
        num_ul_rx=num_ul_rx,
        num_ul_tx_ant=backend.num_ul_tx_ant,
        num_ul_rx_ant=backend.num_ul_rx_ant,
        num_ofdm_symbols=num_ofdm_symbols,
        num_subcarriers=backend.num_subcarriers,
    )
    nmse_db_full = np.zeros((num_snap, num_ul_tx, num_ul_rx), dtype=np.float32)
    ber_per_link = np.zeros((num_snap, num_ul_tx, num_ul_rx), dtype=np.float32)
    bler_per_link = np.zeros((num_snap, num_ul_tx, num_ul_rx), dtype=np.float32)
    estimation_success = np.ones((num_snap, num_ul_tx, num_ul_rx), dtype=np.bool_)
    total_bit_errors = 0
    total_bits = 0
    total_block_errors = 0
    total_blocks = 0
    num_receiver_failures = 0
    link_metadata = _empty_link_metadata(num_snap, num_ul_tx, num_ul_rx)

    link_indices = [
        (snap_idx, ul_tx_idx, ul_rx_idx)
        for snap_idx in range(num_snap)
        for ul_tx_idx in range(num_ul_tx)
        for ul_rx_idx in range(num_ul_rx)
    ]
    batch_size = min(max(int(requested_batch_size), 1), max(len(link_indices), 1))
    stats = {
        "requested_batch_size": int(requested_batch_size),
        "effective_batch_size": 1,
        "num_links": len(link_indices),
        "num_batches": 0,
        "num_batch_fallbacks": 0,
        "num_single_link_fallbacks": 0,
        "num_failed_batch_attempts": 0,
    }

    def _run_group(
        group: list[tuple[int, int, int]],
        *,
        from_fallback: bool = False,
    ) -> list[dict[str, Any]]:
        if len(group) == 1:
            if from_fallback:
                stats["num_single_link_fallbacks"] += 1
            result = _process_one_pusch_link(
                snap_idx=group[0][0],
                ul_tx_idx=group[0][1],
                ul_rx_idx=group[0][2],
                backend=backend,
                tx=tx,
                rx=rx,
                noise_variance_override=noise_variance_override,
                impairment_chain=impairment_chain,
                snr_db=snr_db,
                ls_estimator=ls_estimator,
                perfect_csi=perfect_csi,
                num_ofdm_symbols=num_ofdm_symbols,
                receiver_failure_policy=receiver_failure_policy,
            )
            stats["num_batches"] += 1
            return [result]

        try:
            results = _process_pusch_link_batch(
                link_indices=group,
                backend=backend,
                tx=tx,
                rx=rx,
                noise_variance_override=noise_variance_override,
                impairment_chain=impairment_chain,
                snr_db=snr_db,
                ls_estimator=ls_estimator,
                perfect_csi=perfect_csi,
                num_ofdm_symbols=num_ofdm_symbols,
                receiver_failure_policy=receiver_failure_policy,
            )
            stats["num_batches"] += 1
            stats["effective_batch_size"] = max(
                stats["effective_batch_size"], len(group),
            )
            return results
        except Exception:
            stats["num_batch_fallbacks"] += 1
            stats["num_failed_batch_attempts"] += 1
            mid = max(1, len(group) // 2)
            return (
                _run_group(group[:mid], from_fallback=True)
                + _run_group(group[mid:], from_fallback=True)
            )

    for start in range(0, len(link_indices), batch_size):
        for link_idx, link_result in zip(
            link_indices[start:start + batch_size],
            _run_group(link_indices[start:start + batch_size]),
            strict=True,
        ):
            snap_idx, ul_tx_idx, ul_rx_idx = link_idx
            cfr_est_full[snap_idx, ul_tx_idx, ul_rx_idx, ...] = link_result[
                "cfr_est_slice"
            ]
            waveform_grids["tx_grid"][snap_idx, ul_tx_idx, ul_rx_idx, ...] = (
                link_result["tx_grid_slice"]
            )
            waveform_grids["rx_grid"][snap_idx, ul_tx_idx, ul_rx_idx, ...] = (
                link_result["rx_grid_slice"]
            )
            waveform_grids["noise_variance"][snap_idx, ul_tx_idx, ul_rx_idx] = (
                link_result["noise_variance"]
            )
            _store_link_metadata(
                link_metadata,
                snap_idx,
                ul_tx_idx,
                ul_rx_idx,
                link_result["link_metadata"],
            )
            nmse_db_full[snap_idx, ul_tx_idx, ul_rx_idx] = link_result["nmse_db"]
            ber_per_link[snap_idx, ul_tx_idx, ul_rx_idx] = link_result["ber"]
            bler_per_link[snap_idx, ul_tx_idx, ul_rx_idx] = link_result["bler"]
            estimation_success[snap_idx, ul_tx_idx, ul_rx_idx] = link_result[
                "estimation_success"
            ]
            total_bit_errors += link_result["num_bit_errors"]
            total_bits += link_result["num_bits"]
            total_block_errors += link_result.get("num_block_errors", 0)
            total_blocks += link_result.get("num_blocks", 0)
            if link_result.get("receiver_failed", False):
                num_receiver_failures += 1

    return {
        "cfr_est_full": cfr_est_full,
        "waveform_grids": waveform_grids,
        "nmse_db_full": nmse_db_full,
        "ber_per_link": ber_per_link,
        "bler_per_link": bler_per_link,
        "estimation_success": estimation_success,
        "total_bit_errors": total_bit_errors,
        "total_bits": total_bits,
        "total_block_errors": total_block_errors,
        "total_blocks": total_blocks,
        "num_receiver_failures": num_receiver_failures,
        "link_metadata": link_metadata,
        "batching_stats": stats,
    }


def _process_pusch_link_batch(
    *,
    link_indices: list[tuple[int, int, int]],
    backend: Any,
    tx: Any,
    rx: Any,
    noise_variance_override: torch.Tensor | None,
    impairment_chain: ObservationImpairmentChain,
    snr_db: float,
    ls_estimator: Any | None,
    perfect_csi: bool,
    num_ofdm_symbols: int,
    receiver_failure_policy: str,
) -> list[dict[str, Any]]:
    """Run one batched SU-MIMO PUSCH forward pass."""
    batch_size = len(link_indices)
    tx_signal, tx_bits = tx(batch_size)
    channel_result = backend.apply_clean_batch_with_h(
        tx_signal,
        link_indices=link_indices,
        num_ofdm_symbols=num_ofdm_symbols,
        resource_grid=tx.resource_grid,
    )
    impairment_result, y = _apply_common_chain_to_sionna_y(
        impairment_chain,
        channel_result.y,
        snr_db=snr_db,
        noise_variance_override=noise_variance_override,
    )
    receiver_no = _receiver_no_from_chain(impairment_result, y)
    h_perfect = channel_result.h

    receiver_failed = np.zeros(batch_size, dtype=np.bool_)
    tb_crc_ok = None
    if perfect_csi:
        try:
            rx_bits, tb_crc_status = rx(y, receiver_no, h_perfect)
        except Exception:
            if receiver_failure_policy == "fail_fast":
                raise
            rx_bits = torch.zeros_like(tx_bits)
            tb_crc_status = torch.zeros(
                (batch_size,), dtype=torch.bool, device=tx_bits.device,
            )
            receiver_failed[:] = True
        tb_crc_ok = tb_crc_status
        cfr_est_batch = _pusch_h_batch_to_cfr_est(h_perfect)
    else:
        if ls_estimator is None:
            raise RuntimeError("LS estimator is required when perfect_csi=False")
        try:
            h_hat, _err_var = ls_estimator(y, receiver_no)
        except Exception:
            if receiver_failure_policy == "fail_fast":
                raise
            h_hat = h_perfect
            receiver_failed[:] = True

        try:
            rx_bits, tb_crc_status = rx(y, receiver_no)
        except Exception:
            if receiver_failure_policy == "fail_fast":
                raise
            rx_bits = torch.zeros_like(tx_bits)
            tb_crc_status = torch.zeros(
                (batch_size,), dtype=torch.bool, device=tx_bits.device,
            )
            receiver_failed[:] = True
        tb_crc_ok = tb_crc_status

        _num_tx_ant = backend.cfr.shape[4]
        h_hat_dim4 = h_hat.shape[4]
        if h_hat_dim4 != _num_tx_ant:
            raise NotImplementedError(
                "Estimated CSI physical antenna-pair CFR requires "
                f"num_layers == num_antenna_ports ({h_hat_dim4} != {_num_tx_ant}). "
                "Effective-channel export is not yet supported."
            )
        cfr_est_batch = _pusch_h_batch_to_cfr_est(h_hat)

    tx_bits_np = _to_numpy(tx_bits)
    rx_bits_np = _to_numpy(rx_bits)
    tb_crc_np = None if tb_crc_ok is None else _to_numpy(tb_crc_ok).astype(np.bool_)

    results: list[dict[str, Any]] = []
    for batch_idx, (snap_idx, ul_tx_idx, ul_rx_idx) in enumerate(link_indices):
        tx_grid_slice, rx_grid_slice, noise_variance = _extract_waveform_link_slices(
            tx_signal=tx_signal,
            y=y,
            no=impairment_result.noise_variance[:, 0, 0],
            pusch_tx_idx=0,
            pusch_rx_idx=0,
            batch_idx=batch_idx,
        )
        link_metadata = _extract_batch_item_metadata(impairment_result, batch_idx)
        cfr_est_slice = cfr_est_batch[batch_idx]

        bit_errors = int(np.sum(rx_bits_np[batch_idx] != tx_bits_np[batch_idx]))
        num_bits = int(np.asarray(tx_bits_np[batch_idx]).size)
        ber = bit_errors / max(num_bits, 1)
        num_blocks, num_block_errors, bler = _tb_crc_metrics_for_batch_item(
            tb_crc_np, batch_idx, bit_errors,
        )

        truth_slice = backend.cfr[snap_idx, ul_tx_idx, ul_rx_idx, ...]
        error = cfr_est_slice - truth_slice
        signal_power = np.mean(np.abs(truth_slice) ** 2)
        noise_power_est = np.mean(np.abs(error) ** 2)
        nmse_val = noise_power_est / max(signal_power, 1e-30)

        results.append({
            "cfr_est_slice": cfr_est_slice,
            "tx_grid_slice": tx_grid_slice,
            "rx_grid_slice": rx_grid_slice,
            "noise_variance": noise_variance,
            "link_metadata": link_metadata,
            "nmse_db": float(10.0 * np.log10(max(nmse_val, 1e-30))),
            "ber": ber,
            "bler": bler,
            "estimation_success": not bool(receiver_failed[batch_idx]),
            "num_bit_errors": bit_errors,
            "num_bits": num_bits,
            "num_block_errors": num_block_errors,
            "num_blocks": num_blocks,
            "receiver_failed": bool(receiver_failed[batch_idx]),
        })
    return results


def _pusch_h_batch_to_cfr_est(h: torch.Tensor) -> np.ndarray:
    h_np = h.detach().cpu().numpy() if isinstance(h, torch.Tensor) else np.asarray(h)
    return h_np[:, 0, :, 0, :, 0, :].astype(np.complex64, copy=False)


def _tb_crc_metrics_for_batch_item(
    tb_crc_np: np.ndarray | None,
    batch_idx: int,
    bit_errors: int,
) -> tuple[int, int, float]:
    if tb_crc_np is None:
        return 1, 0, 1.0 if bit_errors > 0 else 0.0
    crc_item = np.asarray(tb_crc_np[batch_idx])
    num_blocks = int(crc_item.size)
    num_block_errors = int(np.sum(~crc_item))
    return num_blocks, num_block_errors, num_block_errors / max(num_blocks, 1)


# ── MU-MIMO per-snapshot processing ────────────────────────────────────


def _process_mu_mimo(
    backend: Any,
    tx: Any,
    rx: Any,
    noise_variance_override: torch.Tensor | None,
    impairment_chain: ObservationImpairmentChain,
    snr_db: float,
    ls_estimator: Any | None,
    perfect_csi: bool,
    num_ofdm_symbols: int,
    receiver_failure_policy: str,
    cfr_clean_ref: np.ndarray,
    num_snap: int,
    num_ul_tx: int,
    num_ul_rx: int,
) -> dict:
    """Process all TX/RX jointly per snapshot (MU-MIMO).

    Uses :func:`cfr_to_full_mimo_h` to build a multi-TX/RX perfect-CSI
    tensor and runs a single PUSCHReceiver forward pass for all UEs.
    """
    from sionna_measurement_sim.phy.nr_mimo_channel import (
        PUSCHMIMOChannel,
        cfr_to_full_mimo_h,
    )

    cfr_est_full = np.zeros(cfr_clean_ref.shape, dtype=np.complex64)
    waveform_grids = _empty_waveform_grids(
        num_snap=num_snap,
        num_ul_tx=num_ul_tx,
        num_ul_rx=num_ul_rx,
        num_ul_tx_ant=backend.num_ul_tx_ant,
        num_ul_rx_ant=backend.num_ul_rx_ant,
        num_ofdm_symbols=num_ofdm_symbols,
        num_subcarriers=backend.num_subcarriers,
    )
    nmse_db_full = np.zeros((num_snap, num_ul_tx, num_ul_rx), dtype=np.float32)
    ber_per_link = np.zeros((num_snap, num_ul_tx, num_ul_rx), dtype=np.float32)
    bler_per_link = np.zeros((num_snap, num_ul_tx, num_ul_rx), dtype=np.float32)
    estimation_success = np.ones((num_snap, num_ul_tx, num_ul_rx), dtype=np.bool_)
    total_bit_errors = 0
    total_bits = 0
    total_block_errors = 0
    total_blocks = 0
    num_receiver_failures = 0
    link_metadata = _empty_link_metadata(num_snap, num_ul_tx, num_ul_rx)

    # Wrap CFR in PUSCHMIMOChannel for cfr_to_full_mimo_h
    channel = PUSCHMIMOChannel(
        cfr=cfr_clean_ref,
        num_snap=num_snap,
        num_ul_tx=num_ul_tx,
        num_ul_rx=num_ul_rx,
        num_ul_tx_ant=backend.num_ul_tx_ant,
        num_ul_rx_ant=backend.num_ul_rx_ant,
        num_subcarriers=backend.num_subcarriers,
        reciprocity_applied=backend.reciprocity_applied,
    )

    for snap_idx in range(num_snap):
        # 1. Build full MIMO h for this snapshot
        h_full = cfr_to_full_mimo_h(
            channel, snap_idx=snap_idx, num_ofdm_symbols=num_ofdm_symbols,
        )
        # h_full: [1, num_ul_rx, num_ul_rx_ant, num_ul_tx, num_ul_tx_ant, ...]

        # 2. Generate TX signal for all UEs
        tx_signal, tx_bits = tx(1)

        # 3. Apply MIMO OFDM channel via backend's full-MIMO path
        channel_result = backend.apply_clean_full_with_h(
            tx_signal,
            snap_idx=snap_idx,
            num_ofdm_symbols=num_ofdm_symbols,
            resource_grid=tx.resource_grid,
        )
        impairment_result, y = _apply_common_chain_to_sionna_y(
            impairment_chain,
            channel_result.y,
            snr_db=snr_db,
            noise_variance_override=noise_variance_override,
        )
        receiver_no = _receiver_no_from_chain(impairment_result, y)
        h_full = channel_result.h
        for ul_tx_idx in range(num_ul_tx):
            for ul_rx_idx in range(num_ul_rx):
                tx_slice, rx_slice, no_value = _extract_waveform_link_slices(
                    tx_signal=tx_signal,
                    y=y,
                    no=impairment_result.noise_variance[0, 0, :],
                    pusch_tx_idx=ul_tx_idx,
                    pusch_rx_idx=ul_rx_idx,
                )
                waveform_grids["tx_grid"][
                    snap_idx, ul_tx_idx, ul_rx_idx, ...
                ] = tx_slice
                waveform_grids["rx_grid"][
                    snap_idx, ul_tx_idx, ul_rx_idx, ...
                ] = rx_slice
                waveform_grids["noise_variance"][
                    snap_idx, ul_tx_idx, ul_rx_idx
                ] = no_value
                _store_link_metadata(
                    link_metadata,
                    snap_idx,
                    ul_tx_idx,
                    ul_rx_idx,
                    _extract_mu_link_metadata(impairment_result, ul_rx_idx),
                )

        # 4. Run PUSCHReceiver
        receiver_failed = False
        tb_crc_ok = None
        try:
            if perfect_csi:
                rx_bits, tb_crc_status = rx(y, receiver_no, h_full)
            else:
                rx_bits, tb_crc_status = rx(y, receiver_no)
            tb_crc_ok = tb_crc_status
        except Exception:
            rx_bits = torch.zeros_like(tx_bits)
            receiver_failed = True
            if receiver_failure_policy == "fail_fast":
                raise

        # 5. Extract cfr_est per link
        if perfect_csi:
            # h_full: [1, num_ul_rx, num_ul_rx_ant, num_ul_tx, num_ul_tx_ant, sym, sub]
            h_np = h_full[0, :, :, :, :, 0, :].cpu().numpy()
            # → [num_ul_rx, num_ul_rx_ant, num_ul_tx, num_ul_tx_ant, subcarrier]
            # Permute to project UL convention:
            # [num_ul_tx, num_ul_rx, num_ul_rx_ant, num_ul_tx_ant, subcarrier]
            h_np = np.transpose(h_np, (2, 0, 1, 3, 4))
            cfr_est_full[snap_idx, ...] = h_np.astype(np.complex64, copy=False)
        else:
            # Estimated CSI: extract per-link from estimator output
            if ls_estimator is None:
                raise RuntimeError("LS estimator is required when perfect_csi=False")
            try:
                h_hat, _err_var = ls_estimator(y, receiver_no)
            except Exception:
                receiver_failed = True
                h_hat = h_full
                if receiver_failure_policy == "fail_fast":
                    raise

            h_hat_np = h_hat[0, :, :, :, :, 0, :].cpu().numpy()
            # h_hat: [1, num_ul_rx, num_ul_rx_ant, num_ul_tx, num_streams, sym, sub]
            # Permute to [num_ul_tx, num_ul_rx, num_ul_rx_ant, num_streams, subcarrier]
            h_hat_np = np.transpose(h_hat_np, (2, 0, 1, 3, 4))

            _num_tx_ant = cfr_clean_ref.shape[4]
            if h_hat_np.shape[3] == _num_tx_ant:
                cfr_est_full[snap_idx, ...] = h_hat_np.astype(np.complex64, copy=False)
            else:
                raise NotImplementedError(
                    "MU-MIMO estimated CSI requires num_layers == num_antenna_ports"
                )

        # 6. Compute per-link metrics
        rx_bits_np = rx_bits.cpu().numpy()  # [batch, num_tx, bits]
        tx_bits_np = tx_bits.cpu().numpy()  # [batch, num_tx, bits]

        # TB CRC: real BLER from CRC status
        if tb_crc_ok is not None:
            num_blocks = int(tb_crc_ok.numel())
            num_block_errs = int(torch.sum(~tb_crc_ok).item())
            joint_bler = num_block_errs / max(num_blocks, 1)
        else:
            num_blocks = 1
            num_block_errs = 0
            joint_bler = 1.0 if int(np.sum(rx_bits_np != tx_bits_np)) > 0 else 0.0

        total_block_errors += num_block_errs
        total_blocks += num_blocks

        # Per-UE bit errors: tx_bits/rx_bits has shape [batch, num_tx, bits].
        # num_tx corresponds to UL transmitters (UEs).
        # Accumulate ONCE per UE (not per-link).
        joint_bit_errs = int(np.sum(rx_bits_np != tx_bits_np))
        joint_num_bits = int(tx_bits_np.size)
        total_bit_errors += joint_bit_errs
        total_bits += joint_num_bits
        joint_ber = joint_bit_errs / max(joint_num_bits, 1)

        for ul_tx_idx in range(num_ul_tx):
            for ul_rx_idx in range(num_ul_rx):
                # Per-link BER is joint diagnostic (not per-link independent)
                ber_per_link[snap_idx, ul_tx_idx, ul_rx_idx] = joint_ber
                bler_per_link[snap_idx, ul_tx_idx, ul_rx_idx] = joint_bler

                truth_slice = cfr_clean_ref[
                    snap_idx, ul_tx_idx, ul_rx_idx, ...
                ]
                est_slice = cfr_est_full[
                    snap_idx, ul_tx_idx, ul_rx_idx, ...
                ]
                # Ensure est_slice matches truth_slice in tx_ant dim
                if est_slice.shape[1] != truth_slice.shape[1]:
                    est_slice_padded = np.zeros(truth_slice.shape, dtype=np.complex64)
                    est_slice_padded[:, :est_slice.shape[1], :] = est_slice
                    est_slice = est_slice_padded
                error = est_slice - truth_slice
                signal_power = np.mean(np.abs(truth_slice) ** 2)
                noise_power_est = np.mean(np.abs(error) ** 2)
                nmse_val = noise_power_est / max(signal_power, 1e-30)
                nmse_db_full[snap_idx, ul_tx_idx, ul_rx_idx] = float(
                    10.0 * np.log10(max(nmse_val, 1e-30))
                )
                estimation_success[snap_idx, ul_tx_idx, ul_rx_idx] = (
                    not receiver_failed
                )
                if receiver_failed:
                    num_receiver_failures += 1

    return {
        "cfr_est_full": cfr_est_full,
        "waveform_grids": waveform_grids,
        "nmse_db_full": nmse_db_full,
        "ber_per_link": ber_per_link,
        "bler_per_link": bler_per_link,
        "estimation_success": estimation_success,
        "total_bit_errors": total_bit_errors,
        "total_bits": total_bits,
        "total_block_errors": total_block_errors,
        "total_blocks": total_blocks,
        "num_receiver_failures": num_receiver_failures,
        "link_metadata": link_metadata,
        "batching_stats": _default_batching_stats(
            requested_batch_size=1,
            num_links=num_snap,
        ),
    }


# ── per-link processing ─────────────────────────────────────────────────


def _process_one_pusch_link(
    snap_idx: int,
    ul_tx_idx: int,
    ul_rx_idx: int,
    backend: Any,
    tx: Any,
    rx: Any,
    noise_variance_override: torch.Tensor | None,
    impairment_chain: ObservationImpairmentChain,
    snr_db: float,
    ls_estimator: Any | None,
    perfect_csi: bool,
    num_ofdm_symbols: int,
    receiver_failure_policy: str,
) -> dict:
    """Process a single PUSCH link with full MIMO.

    Returns a dict with cfr_est_slice, nmse_db, ber, bler, etc.
    """
    # 1. Generate TX signal
    tx_signal, tx_bits = tx(1)

    # 2. Apply clean MIMO OFDM channel, then common impairments/AWGN
    channel_result = backend.apply_clean_with_h(
        tx_signal,
        snap_idx=snap_idx,
        ul_tx_idx=ul_tx_idx,
        ul_rx_idx=ul_rx_idx,
        num_ofdm_symbols=num_ofdm_symbols,
        resource_grid=tx.resource_grid,
    )
    impairment_result, y = _apply_common_chain_to_sionna_y(
        impairment_chain,
        channel_result.y,
        snr_db=snr_db,
        noise_variance_override=noise_variance_override,
    )
    receiver_no = _receiver_no_from_chain(impairment_result, y)
    h_perfect = channel_result.h
    tx_grid_slice, rx_grid_slice, noise_variance = _extract_waveform_link_slices(
        tx_signal=tx_signal,
        y=y,
        no=impairment_result.noise_variance[:, 0, 0],
        pusch_tx_idx=0,
        pusch_rx_idx=0,
    )
    link_metadata = _extract_batch_item_metadata(impairment_result, 0)
    # y: [batch, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
    # h_perfect: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, ...]

    # 3. Run PUSCHReceiver and get cfr_est
    receiver_failed = False
    cfr_est_slice: np.ndarray | None = None

    tb_crc_ok = None  # CRC status per transport block

    if perfect_csi:
        try:
            rx_bits, tb_crc_status = rx(y, receiver_no, h_perfect)
        except Exception:
            rx_bits = torch.zeros_like(tx_bits)
            tb_crc_status = torch.zeros(tx_bits.shape[0], dtype=torch.bool)
            receiver_failed = True
            if receiver_failure_policy == "fail_fast":
                raise
        tb_crc_ok = tb_crc_status
        # For perfect CSI, cfr_est = physical MIMO channel
        cfr_est_slice = pusch_h_to_cfr_est(h_perfect)
    else:
        # Estimated CSI: let PUSCHReceiver use its internal estimator.
        # We run a separate PUSCHLSChannelEstimator to get h_hat for cfr_est.
        if ls_estimator is None:
            raise RuntimeError("LS estimator is required when perfect_csi=False")
        try:
            h_hat, _err_var = ls_estimator(y, receiver_no)
            # h_hat: [batch, num_rx, num_rx_ant, num_tx, num_streams_per_tx, ...]
        except Exception:
            receiver_failed = True
            h_hat = h_perfect
            if receiver_failure_policy == "fail_fast":
                raise

        try:
            rx_bits, tb_crc_status = rx(y, receiver_no)
        except Exception:
            rx_bits = torch.zeros_like(tx_bits)
            tb_crc_status = torch.zeros(tx_bits.shape[0], dtype=torch.bool)
            receiver_failed = True
            if receiver_failure_policy == "fail_fast":
                raise
        tb_crc_ok = tb_crc_status

        # h_hat has num_streams_per_tx in dim 4 (not num_tx_ant).
        # When num_layers != num_antenna_ports the estimator returns an
        # effective (stream-level) channel that cannot be written into
        # /observation/cfr_est without the precoding matrix.
        _num_tx_ant = backend.cfr.shape[4]
        h_hat_dim4 = h_hat.shape[4]
        if h_hat_dim4 == _num_tx_ant:
            cfr_est_slice = pusch_h_to_cfr_est(h_hat)
        else:
            raise NotImplementedError(
                "Estimated CSI physical antenna-pair CFR requires "
                f"num_layers == num_antenna_ports ({h_hat_dim4} != {_num_tx_ant}). "
                "Effective-channel export is not yet supported."
            )

    if cfr_est_slice is None:
        cfr_est_slice = pusch_h_to_cfr_est(h_perfect)

    # 5. Compute BER/BLER with TB CRC semantics
    num_bit_errors = int(torch.sum(torch.ne(rx_bits, tx_bits)).item())
    num_total_bits = int(tx_bits.shape[-1])
    ber = num_bit_errors / max(num_total_bits, 1)

    # Real BLER from TB CRC status
    if tb_crc_ok is not None:
        num_blocks = int(tb_crc_ok.numel())
        num_block_errors = int(torch.sum(~tb_crc_ok).item())
        bler = num_block_errors / max(num_blocks, 1)
    else:
        num_blocks = 1
        num_block_errors = 0
        bler = 1.0 if num_bit_errors > 0 else 0.0

    # 6. Compute per-link NMSE
    truth_slice = backend.cfr[snap_idx, ul_tx_idx, ul_rx_idx, ...]
    error = cfr_est_slice - truth_slice
    signal_power = np.mean(np.abs(truth_slice) ** 2)
    noise_power_est = np.mean(np.abs(error) ** 2)
    nmse_val = noise_power_est / max(signal_power, 1e-30)
    nmse_db = float(10.0 * np.log10(max(nmse_val, 1e-30)))

    return {
        "cfr_est_slice": cfr_est_slice,
        "tx_grid_slice": tx_grid_slice,
        "rx_grid_slice": rx_grid_slice,
        "noise_variance": noise_variance,
        "link_metadata": link_metadata,
        "nmse_db": nmse_db,
        "ber": ber,
        "bler": bler,
        "estimation_success": not receiver_failed,
        "num_bit_errors": num_bit_errors,
        "num_bits": num_total_bits,
        "num_block_errors": num_block_errors,
        "num_blocks": num_blocks,
        "receiver_failed": receiver_failed,
    }


# ── waveform grid and array-label helpers ──────────────────────────────


def _empty_waveform_grids(
    *,
    num_snap: int,
    num_ul_tx: int,
    num_ul_rx: int,
    num_ul_tx_ant: int,
    num_ul_rx_ant: int,
    num_ofdm_symbols: int,
    num_subcarriers: int,
) -> dict[str, np.ndarray]:
    """Allocate NR PUSCH frequency-domain waveform export arrays."""
    return {
        "tx_grid": np.zeros(
            (
                num_snap,
                num_ul_tx,
                num_ul_rx,
                num_ul_tx_ant,
                num_ofdm_symbols,
                num_subcarriers,
            ),
            dtype=np.complex64,
        ),
        "rx_grid": np.zeros(
            (
                num_snap,
                num_ul_tx,
                num_ul_rx,
                num_ul_rx_ant,
                num_ofdm_symbols,
                num_subcarriers,
            ),
            dtype=np.complex64,
        ),
        "noise_variance": np.zeros((num_snap, num_ul_tx, num_ul_rx), dtype=np.float32),
    }


def _empty_link_metadata(
    num_snap: int,
    num_ul_tx: int,
    num_ul_rx: int,
) -> dict[str, np.ndarray]:
    link_shape = (num_snap, num_ul_tx, num_ul_rx)
    return {
        "snr_db": np.zeros(link_shape, dtype=np.float32),
        "rssi_dbm": np.zeros(link_shape, dtype=np.float32),
        "noise_power_dbm": np.zeros(link_shape, dtype=np.float32),
        "cfo_hz": np.zeros(link_shape, dtype=np.float32),
        "sfo_ppm": np.zeros(link_shape, dtype=np.float32),
        "timing_offset_samples": np.zeros(link_shape, dtype=np.float32),
        "phase_offset_rad": np.zeros(link_shape, dtype=np.float32),
        "agc_gain_db": np.zeros((num_snap, num_ul_rx), dtype=np.float32),
        "clipping_flag": np.zeros(link_shape, dtype=np.bool_),
    }


def _store_link_metadata(
    metadata: dict[str, np.ndarray],
    snap_idx: int,
    ul_tx_idx: int,
    ul_rx_idx: int,
    link_metadata: dict[str, np.float32 | np.bool_],
) -> None:
    for key in (
        "snr_db",
        "rssi_dbm",
        "noise_power_dbm",
        "cfo_hz",
        "sfo_ppm",
        "timing_offset_samples",
        "phase_offset_rad",
        "clipping_flag",
    ):
        metadata[key][snap_idx, ul_tx_idx, ul_rx_idx] = link_metadata[key]
    metadata["agc_gain_db"][snap_idx, ul_rx_idx] = link_metadata["agc_gain_db"]


def _apply_common_chain_to_sionna_y(
    chain: ObservationImpairmentChain,
    y_clean: torch.Tensor,
    *,
    snr_db: float,
    noise_variance_override: torch.Tensor | None,
) -> tuple[Any, torch.Tensor]:
    """Apply common impairments to Sionna ``y`` while preserving receiver shape.

    Sionna uses ``[batch,num_rx,rx_ant,ofdm_symbol,subcarrier]``.  The common
    chain uses leading ``[snapshot,tx,rx]`` axes, so batch becomes snapshot and
    a singleton logical TX axis is inserted.
    """
    if y_clean.ndim != 5:
        raise ValueError(f"Sionna y must be rank 5, got {tuple(y_clean.shape)}")
    common_y = y_clean.unsqueeze(1)
    result = chain.apply(
        common_y,
        snr_db=snr_db,
        noise_variance_override=(
            None
            if noise_variance_override is None
            else _noise_override_for_common(
                noise_variance_override,
                common_y.shape[:3],
                common_y.device,
            )
        ),
    )
    return result, result.rx_grid[:, 0, ...]


def _noise_override_for_common(
    value: torch.Tensor,
    link_shape: tuple[int, int, int],
    device: torch.device,
) -> torch.Tensor:
    override = torch.as_tensor(value, dtype=torch.float32, device=device)
    snapshot, tx, rx = link_shape
    if override.ndim == 0:
        return override
    if override.shape == link_shape:
        return override
    if override.numel() == snapshot:
        return override.reshape(snapshot, 1, 1)
    if override.numel() == snapshot * rx:
        return override.reshape(snapshot, 1, rx)
    if override.numel() == snapshot * tx * rx:
        return override.reshape(link_shape)
    return override


def _receiver_no_from_chain(result: Any, y: torch.Tensor) -> torch.Tensor:
    noise = result.noise_variance.to(device=y.device, dtype=torch.float32)
    if noise.numel() == 1:
        return noise.reshape(())
    return noise[:, 0, :]


def _extract_batch_item_metadata(result: Any, batch_idx: int) -> dict[str, np.float32 | np.bool_]:
    scalars = ResultAssembler.chain_scalars_to_numpy(result)
    sample = ResultAssembler.sample_to_numpy(result.impairment_sample)
    return {
        "snr_db": np.float32(scalars["snr_db"][batch_idx, 0, 0]),
        "rssi_dbm": np.float32(scalars["rssi_dbm"][batch_idx, 0, 0]),
        "noise_power_dbm": np.float32(scalars["noise_power_dbm"][batch_idx, 0, 0]),
        "cfo_hz": np.float32(sample["cfo_hz"][batch_idx, 0, 0]),
        "sfo_ppm": np.float32(sample["sfo_ppm"][batch_idx, 0, 0]),
        "timing_offset_samples": np.float32(
            sample["timing_offset_samples"][batch_idx, 0, 0]
        ),
        "phase_offset_rad": np.float32(sample["phase_offset_rad"][batch_idx, 0, 0]),
        "agc_gain_db": np.float32(sample["agc_gain_db"][batch_idx, 0]),
        "clipping_flag": np.bool_(sample["clipping_flag"][batch_idx, 0, 0]),
    }


def _extract_mu_link_metadata(result: Any, rx_idx: int) -> dict[str, np.float32 | np.bool_]:
    scalars = ResultAssembler.chain_scalars_to_numpy(result)
    sample = ResultAssembler.sample_to_numpy(result.impairment_sample)
    return {
        "snr_db": np.float32(scalars["snr_db"][0, 0, rx_idx]),
        "rssi_dbm": np.float32(scalars["rssi_dbm"][0, 0, rx_idx]),
        "noise_power_dbm": np.float32(scalars["noise_power_dbm"][0, 0, rx_idx]),
        "cfo_hz": np.float32(sample["cfo_hz"][0, 0, rx_idx]),
        "sfo_ppm": np.float32(sample["sfo_ppm"][0, 0, rx_idx]),
        "timing_offset_samples": np.float32(sample["timing_offset_samples"][0, 0, rx_idx]),
        "phase_offset_rad": np.float32(sample["phase_offset_rad"][0, 0, rx_idx]),
        "agc_gain_db": np.float32(sample["agc_gain_db"][0, rx_idx]),
        "clipping_flag": np.bool_(sample["clipping_flag"][0, 0, rx_idx]),
    }


def _transpose_link_metadata_for_reciprocity(
    metadata: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    transposed = dict(metadata)
    for key in (
        "snr_db",
        "rssi_dbm",
        "noise_power_dbm",
        "cfo_hz",
        "sfo_ppm",
        "timing_offset_samples",
        "phase_offset_rad",
        "clipping_flag",
    ):
        transposed[key] = np.transpose(metadata[key], (0, 2, 1))
    snapshot = metadata["snr_db"].shape[0]
    old_tx = metadata["snr_db"].shape[1]
    mean_agc = np.mean(metadata["agc_gain_db"], axis=1, keepdims=True)
    transposed["agc_gain_db"] = np.broadcast_to(mean_agc, (snapshot, old_tx)).copy()
    return transposed


def _fit_agc_to_link_shape(agc_gain_db: np.ndarray, link_shape: tuple[int, int, int]) -> np.ndarray:
    expected = (link_shape[0], link_shape[2])
    if agc_gain_db.shape == expected:
        return agc_gain_db.astype(np.float32, copy=False)
    if agc_gain_db.size == 0:
        return np.zeros(expected, dtype=np.float32)
    mean_agc = np.mean(agc_gain_db, axis=1, keepdims=True)
    return np.broadcast_to(mean_agc, expected).astype(np.float32, copy=True)


def _extract_waveform_link_slices(
    *,
    tx_signal: Any,
    y: Any,
    no: Any,
    pusch_tx_idx: int,
    pusch_rx_idx: int,
    batch_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.float32]:
    """Extract one link's actual Sionna frequency-domain TX/RX grids."""
    tx_np = _to_numpy(tx_signal)
    y_np = _to_numpy(y)
    if tx_np.ndim != 5:
        raise ValueError(f"tx_signal must be rank 5, got {tx_np.shape}")
    if y_np.ndim != 5:
        raise ValueError(f"y must be rank 5, got {y_np.shape}")

    tx_index = pusch_tx_idx if tx_np.shape[1] > 1 else 0
    rx_index = pusch_rx_idx if y_np.shape[1] > 1 else 0
    tx_slice = tx_np[batch_idx, tx_index, ...].astype(np.complex64, copy=False)
    rx_slice = y_np[batch_idx, rx_index, ...].astype(np.complex64, copy=False)
    return tx_slice, rx_slice, np.float32(
        _noise_variance_scalar(no, batch_idx=batch_idx, rx_idx=rx_index)
    )


def build_array_outputs_from_waveform(
    rx_grid: np.ndarray,
    *,
    aoa_label_rad: np.ndarray | None = None,
    spectrum_config: ArraySpectrumConfig | None = None,
    rx_num_rows: int = 1,
    rx_num_cols: int | None = None,
    rx_spacing_lambda: tuple[float, float] = (0.5, 0.5),
    rx_orientation_rad: np.ndarray | None = None,
    truth_spectrum_samples: np.ndarray | None = None,
    cfr_est_spectrum_samples: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build deterministic first-version `/array` outputs from RX grids.

    `aoa_label_rad` is a forward-compatible hook for derived first-path AoA
    labels with shape [snapshot, tx, rx, 2] = [zenith, azimuth] in the
    scene/global frame. If `rx_orientation_rad` is provided, Bartlett spectra
    are also scanned on that same scene/global angle grid.
    """
    config = spectrum_config or ArraySpectrumConfig()
    rx = np.asarray(rx_grid, dtype=np.complex64)
    if rx.ndim != 6:
        raise ValueError(f"rx_grid must be rank 6, got {rx.shape}")

    link_shape = rx.shape[:3]
    num_rx_ant = rx.shape[3]
    if rx_num_cols is None:
        rx_num_cols = num_rx_ant // int(rx_num_rows)
    snapshot_matrix = build_rx_snapshot_matrix(rx)

    angle_grid = build_angle_grid_rad(config)
    labels, spectrum = build_aoa_heatmap_label(aoa_label_rad, angle_grid, link_shape)

    outputs: dict[str, np.ndarray | str] = {
        "rx_snapshot_matrix": snapshot_matrix,
        "aoa_label_rad": labels,
        "aoa_heatmap_label": spectrum,
        "angle_grid_rad": angle_grid,
        "spectrum_policy": config.policy,
    }

    if config.enabled and "rx_grid" in config.sources:
        outputs["spatial_spectrum_observation"] = build_bartlett_spectrum(
            rx,
            rx_num_rows=rx_num_rows,
            rx_num_cols=rx_num_cols,
            rx_spacing_lambda=rx_spacing_lambda,
            rx_orientation_rad=rx_orientation_rad,
            config=config,
        )
    if config.enabled and "truth_cfr" in config.sources and truth_spectrum_samples is not None:
        outputs["spatial_spectrum_truth"] = build_bartlett_spectrum(
            truth_spectrum_samples,
            rx_num_rows=rx_num_rows,
            rx_num_cols=rx_num_cols,
            rx_spacing_lambda=rx_spacing_lambda,
            rx_orientation_rad=rx_orientation_rad,
            config=config,
        )
    if config.enabled and "cfr_est" in config.sources and cfr_est_spectrum_samples is not None:
        outputs["spatial_spectrum_cfr_est"] = build_bartlett_spectrum(
            cfr_est_spectrum_samples,
            rx_num_rows=rx_num_rows,
            rx_num_cols=rx_num_cols,
            rx_spacing_lambda=rx_spacing_lambda,
            rx_orientation_rad=rx_orientation_rad,
            config=config,
        )
    return outputs


def _fixed_angle_grid_rad() -> np.ndarray:
    return build_angle_grid_rad(ArraySpectrumConfig())


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _noise_variance_scalar(no: Any, *, batch_idx: int = 0, rx_idx: int = 0) -> float:
    arr = _to_numpy(no).astype(np.float32, copy=False)
    if arr.size == 0:
        return 0.0
    if arr.ndim == 1:
        index = batch_idx if batch_idx > 0 else rx_idx
        return float(arr[min(index, arr.shape[0] - 1)])
    if arr.ndim >= 2:
        batch = min(batch_idx, arr.shape[0] - 1)
        rx = min(rx_idx, arr.shape[-1] - 1)
        return float(arr.reshape(arr.shape[0], -1)[batch, rx])
    return float(np.ravel(arr)[0])
