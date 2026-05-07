"""NR PUSCH observation backend with full MIMO support.

Builds Sionna PUSCHConfig(s) from project config, runs PUSCHTransmitter
and PUSCHReceiver with RT CIR via ApplyOFDMChannel, and computes real
BER/BLER.  Supports SU-MIMO 4x4 and lays groundwork for MU-MIMO.

Per-channel-estimator CFR is written to /observation/cfr_est without
broadcasting from a single SISO estimate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from sionna_measurement_sim.domain.link import LinkConfig
from sionna_measurement_sim.domain.observation import (
    EvaluationResult,
    ImpairmentSpec,
    ObservationResult,
    ReceiverSpec,
    WaveformSpec,
)
from sionna_measurement_sim.phy.nr_channel_backend import (
    create_channel_backend,
)
from sionna_measurement_sim.phy.nr_mimo_channel import (
    pusch_h_to_cfr_est,
    reverse_reciprocity_cfr,
)

# ── PUSCH config helpers ────────────────────────────────────────────────


def build_multiuser_pusch_configs(
    phy_config,
    carrier_config,
) -> list[Any]:
    """Build a list of Sionna PUSCHConfig objects for multi-UE uplink.

    For SU-MIMO the list contains a single PUSCHConfig.  For MU-MIMO
    the list contains one config per UE with non-overlapping DMRS port sets.
    """
    from sionna.phy.nr import CarrierConfig as SionnaCarrierConfig
    from sionna.phy.nr import PUSCHConfig as SionnaPUSCHConfig
    from sionna.phy.nr import PUSCHDMRSConfig, TBConfig

    num_tx = getattr(phy_config, "num_pusch_tx", None)
    if num_tx is None:
        num_tx = 1  # SU-MIMO default

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
    # 0. Extract config  ──────────────────────────────────────────────
    sc_spacing_hz = float(phy_config.subcarrier_spacing_khz) * 1000.0
    num_prb = phy_config.num_prb
    num_subcarriers = num_prb * 12
    num_ofdm_symbols = getattr(phy_config, "num_ofdm_symbols", 14)
    perfect_csi = getattr(phy_config, "perfect_csi", False)
    mimo_detector_type = getattr(phy_config, "mimo_detector", "lmmse")
    channel_estimator_type = getattr(phy_config, "channel_estimator", "pusch_ls")
    receiver_failure_policy = getattr(phy_config, "receiver_failure_policy", "fail_fast")
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

    cfr_clean_ref = backend.cfr.copy()
    num_snap = backend.num_snap
    num_ul_tx = backend.num_ul_tx
    num_ul_rx = backend.num_ul_rx

    # 2. Build PUSCH configs — auto-derive num_pusch_tx for MU-MIMO ──
    if mimo_mode == "mu_mimo":
        object.__setattr__(phy_config, "num_pusch_tx", num_ul_tx)
    pusch_configs = build_multiuser_pusch_configs(phy_config, carrier_config)
    _num_pusch_tx = len(pusch_configs)
    _num_layers = pusch_configs[0].num_layers
    _num_antenna_ports = pusch_configs[0].num_antenna_ports

    from sionna.phy.nr import PUSCHTransmitter

    tx = PUSCHTransmitter(pusch_configs, output_domain="freq", return_bits=True)

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

    # 6. Process links ───────────────────────────────────────────────
    if mimo_mode == "mu_mimo":
        proc_result = _process_mu_mimo(
            backend=backend, tx=tx, rx=rx, no=no,
            perfect_csi=perfect_csi,
            num_ofdm_symbols=num_ofdm_symbols,
            receiver_failure_policy=receiver_failure_policy,
            cfr_clean_ref=cfr_clean_ref,
            num_snap=num_snap, num_ul_tx=num_ul_tx, num_ul_rx=num_ul_rx,
        )
    else:
        proc_result = _process_su_mimo_per_link(
            backend=backend, tx=tx, rx=rx, no=no,
            perfect_csi=perfect_csi,
            num_ofdm_symbols=num_ofdm_symbols,
            receiver_failure_policy=receiver_failure_policy,
            cfr_clean_ref=cfr_clean_ref,
            num_snap=num_snap, num_ul_tx=num_ul_tx, num_ul_rx=num_ul_rx,
        )

    cfr_est_full = proc_result["cfr_est_full"]
    nmse_db_full = proc_result["nmse_db_full"]
    ber_per_link = proc_result["ber_per_link"]
    bler_per_link = proc_result["bler_per_link"]
    estimation_success = proc_result["estimation_success"]
    total_bit_errors = proc_result["total_bit_errors"]
    total_bits = proc_result["total_bits"]
    total_block_errors = proc_result.get("total_block_errors", 0)
    total_blocks = proc_result.get("total_blocks", 0)
    num_receiver_failures = proc_result["num_receiver_failures"]

    # 7. Reverse UL→DL for HDF5 contract  ────────────────────────────
    # Internal processing always uses UL convention.
    # HDF5 output must be in DL (project) convention.
    cfr_clean_ref = reverse_reciprocity_cfr(cfr_clean_ref)
    cfr_est_full = reverse_reciprocity_cfr(cfr_est_full)
    # Also transpose per-link arrays to DL convention
    nmse_db_full = np.transpose(nmse_db_full, (0, 2, 1))
    ber_per_link = np.transpose(ber_per_link, (0, 2, 1))
    bler_per_link = np.transpose(bler_per_link, (0, 2, 1))
    estimation_success = np.transpose(estimation_success, (0, 2, 1))

    # 8. Aggregate metrics  ───────────────────────────────────────────
    # DL shape = [snap, tx(=ul_rx), rx(=ul_tx), rx_ant, tx_ant, subcarrier]
    dl_link_shape = (num_snap, num_ul_rx, num_ul_tx)
    link_shape = dl_link_shape
    aggregate_ber = total_bit_errors / max(total_bits, 1)
    aggregate_bler = float(np.mean(bler_per_link)) if bler_per_link.size > 0 else 0.0

    observation = ObservationResult(
        cfr_est=cfr_est_full,
        valid_mask=np.ones(link_shape, dtype=np.bool_),
        detection_success=np.ones(link_shape, dtype=np.bool_),
        estimation_success=estimation_success,
        snr_db=np.full(link_shape, float(snr_db), dtype=np.float32),
        rssi_dbm=np.zeros(link_shape, dtype=np.float32),
        noise_power_dbm=np.zeros(link_shape, dtype=np.float32),
        cfo_hz=np.zeros(link_shape, dtype=np.float32),
        sfo_ppm=np.zeros(link_shape, dtype=np.float32),
        timing_offset_samples=np.zeros(link_shape, dtype=np.float32),
        phase_offset_rad=np.zeros(link_shape, dtype=np.float32),
        agc_gain_db=np.zeros((num_snap, num_ul_tx), dtype=np.float32),
        clipping_flag=np.zeros(link_shape, dtype=np.bool_),
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
        "impairments": ImpairmentSpec(
            model_version="nr_pusch_mimo_v1",
            random_seed=42,
            awgn_config=f'{{"snr_db": {float(snr_db)}}}',
        ),
        "reciprocity_applied": backend.reciprocity_applied,
        "num_tx_bits": total_bits,
        "tx_signal_shape": None,
    }
    return result


# ── SU-MIMO per-link processing ────────────────────────────────────────


def _process_su_mimo_per_link(
    backend: Any,
    tx: Any,
    rx: Any,
    no: torch.Tensor,
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
    nmse_db_full = np.zeros((num_snap, num_ul_tx, num_ul_rx), dtype=np.float32)
    ber_per_link = np.zeros((num_snap, num_ul_tx, num_ul_rx), dtype=np.float32)
    bler_per_link = np.zeros((num_snap, num_ul_tx, num_ul_rx), dtype=np.float32)
    estimation_success = np.ones((num_snap, num_ul_tx, num_ul_rx), dtype=np.bool_)
    total_bit_errors = 0
    total_bits = 0
    total_block_errors = 0
    total_blocks = 0
    num_receiver_failures = 0

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
                    no=no,
                    perfect_csi=perfect_csi,
                    num_ofdm_symbols=num_ofdm_symbols,
                    receiver_failure_policy=receiver_failure_policy,
                )
                cfr_est_full[
                    snap_idx, ul_tx_idx, ul_rx_idx, ...
                ] = link_result["cfr_est_slice"]
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
        "nmse_db_full": nmse_db_full,
        "ber_per_link": ber_per_link,
        "bler_per_link": bler_per_link,
        "estimation_success": estimation_success,
        "total_bit_errors": total_bit_errors,
        "total_bits": total_bits,
        "total_block_errors": total_block_errors,
        "total_blocks": total_blocks,
        "num_receiver_failures": num_receiver_failures,
    }


# ── MU-MIMO per-snapshot processing ────────────────────────────────────


def _process_mu_mimo(
    backend: Any,
    tx: Any,
    rx: Any,
    no: torch.Tensor,
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
    nmse_db_full = np.zeros((num_snap, num_ul_tx, num_ul_rx), dtype=np.float32)
    ber_per_link = np.zeros((num_snap, num_ul_tx, num_ul_rx), dtype=np.float32)
    bler_per_link = np.zeros((num_snap, num_ul_tx, num_ul_rx), dtype=np.float32)
    estimation_success = np.ones((num_snap, num_ul_tx, num_ul_rx), dtype=np.bool_)
    total_bit_errors = 0
    total_bits = 0
    num_receiver_failures = 0

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
        # tx_signal: [1, num_ul_tx, num_streams_per_tx, num_ofdm_symbols, fft_size]

        # 3. Apply MIMO OFDM channel
        y = backend.apply(
            tx_signal, no,
            snap_idx=snap_idx, ul_tx_idx=0, ul_rx_idx=0,
            num_ofdm_symbols=num_ofdm_symbols,
            resource_grid=tx.resource_grid,
        )
        # For MU-MIMO, backend.apply with full h handles multi-TX/RX.
        # If using ApplyOFDMChannelBackend, it uses per-link h.
        # For MU-MIMO we bypass backend.apply and use ApplyOFDMChannel directly.
        from sionna.phy.channel import ApplyOFDMChannel

        apply_ch = ApplyOFDMChannel()
        y = apply_ch(tx_signal, h_full, no)

        # 4. Run PUSCHReceiver
        receiver_failed = False
        tb_crc_ok = None
        try:
            if perfect_csi:
                rx_bits, tb_crc_status = rx(y, no, h_full)
            else:
                rx_bits, tb_crc_status = rx(y, no)
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
            from sionna.phy.nr import PUSCHLSChannelEstimator

            _estimator = PUSCHLSChannelEstimator(
                tx.resource_grid,
                dmrs_length=tx._dmrs_length,
                dmrs_additional_position=tx._dmrs_additional_position,
                num_cdm_groups_without_data=tx._num_cdm_groups_without_data,
                interpolation_type="lin",
            )
            try:
                h_hat, _err_var = _estimator(y, no)
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
        rx_bits_np = rx_bits.cpu().numpy()
        tx_bits_np = tx_bits.cpu().numpy()
        # rx_bits/tx_bits: [batch, num_tx, bits]
        # TB CRC: real BLER from CRC status
        if tb_crc_ok is not None:
            num_blocks = int(tb_crc_ok.numel())
            num_block_errs = int(torch.sum(~tb_crc_ok).item())
            joint_bler = num_block_errs / max(num_blocks, 1)
        else:
            num_blocks = 1
            num_block_errs = 0
            joint_bler = 1.0 if int(np.sum(rx_bits_np != tx_bits_np)) > 0 else 0.0

        for ul_tx_idx in range(num_ul_tx):
            for ul_rx_idx in range(num_ul_rx):
                bit_errs = int(np.sum(rx_bits_np != tx_bits_np))
                num_b = int(tx_bits_np.size)
                total_bit_errors += bit_errs
                total_bits += num_b

                ber = bit_errs / max(num_b, 1)
                ber_per_link[snap_idx, ul_tx_idx, ul_rx_idx] = ber
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
        "nmse_db_full": nmse_db_full,
        "ber_per_link": ber_per_link,
        "bler_per_link": bler_per_link,
        "estimation_success": estimation_success,
        "total_bit_errors": total_bit_errors,
        "total_bits": total_bits,
        "num_receiver_failures": num_receiver_failures,
    }


# ── per-link processing ─────────────────────────────────────────────────


def _process_one_pusch_link(
    snap_idx: int,
    ul_tx_idx: int,
    ul_rx_idx: int,
    backend: Any,
    tx: Any,
    rx: Any,
    no: torch.Tensor,
    perfect_csi: bool,
    num_ofdm_symbols: int,
    receiver_failure_policy: str,
) -> dict:
    """Process a single PUSCH link with full MIMO.

    Returns a dict with cfr_est_slice, nmse_db, ber, bler, etc.
    """
    # 1. Build perfect-CSI h for this (snap, ul_tx, ul_rx) link
    h_perfect = backend.perfect_h(
        snap_idx=snap_idx,
        ul_tx_idx=ul_tx_idx,
        ul_rx_idx=ul_rx_idx,
        num_ofdm_symbols=num_ofdm_symbols,
    )
    # h_perfect: [1, 1, num_rx_ant, 1, num_tx_ant, num_ofdm_symbols, fft_size]

    # 2. Generate TX signal
    tx_signal, tx_bits = tx(1)

    # 3. Apply MIMO OFDM channel via backend
    y = backend.apply(
        tx_signal, no,
        snap_idx=snap_idx,
        ul_tx_idx=ul_tx_idx,
        ul_rx_idx=ul_rx_idx,
        num_ofdm_symbols=num_ofdm_symbols,
        resource_grid=tx.resource_grid,
    )
    # y: [batch, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]

    # 4. Run PUSCHReceiver and get cfr_est
    receiver_failed = False
    cfr_est_slice: np.ndarray | None = None

    tb_crc_ok = None  # CRC status per transport block

    if perfect_csi:
        try:
            rx_bits, tb_crc_status = rx(y, no, h_perfect)
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
        from sionna.phy.nr import PUSCHLSChannelEstimator

        _dmrs_len = tx._dmrs_length
        _dmrs_add_pos = tx._dmrs_additional_position
        _num_cdm = tx._num_cdm_groups_without_data
        estimator = PUSCHLSChannelEstimator(
            tx.resource_grid,
            dmrs_length=_dmrs_len,
            dmrs_additional_position=_dmrs_add_pos,
            num_cdm_groups_without_data=_num_cdm,
            interpolation_type="lin",
        )
        try:
            h_hat, _err_var = estimator(y, no)
            # h_hat: [batch, num_rx, num_rx_ant, num_tx, num_streams_per_tx, ...]
        except Exception:
            receiver_failed = True
            h_hat = h_perfect
            if receiver_failure_policy == "fail_fast":
                raise

        try:
            rx_bits, tb_crc_status = rx(y, no)
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
