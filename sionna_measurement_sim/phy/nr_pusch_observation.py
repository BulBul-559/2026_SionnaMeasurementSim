"""NR PUSCH observation backend.

Builds Sionna PUSCHConfig from project config, runs PUSCHTransmitter
and PUSCHReceiver with RT CIR, and computes real BER/BLER.

Current MIMO limitation: uses first TX/RX and antenna pair (0,0) for
channel. Full 4x4 MIMO path pending GPU-enabled PUSCHReceiver.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from sionna_measurement_sim.domain.observation import (
    EvaluationResult,
    ImpairmentSpec,
    ObservationResult,
    ReceiverSpec,
    WaveformSpec,
)

# ── helpers ─────────────────────────────────────────────────────────────


def build_nr_pusch_config(phy_config, carrier_config) -> Any:
    """Build a Sionna PUSCHConfig from project config objects.

    Args:
        phy_config: :class:`~sionna_measurement_sim.config.schema.PHYConfig`
            with NR PUSCH fields set.
        carrier_config: :class:`~sionna_measurement_sim.config.schema.CarrierConfig`
            (not used directly in the current skeleton but kept for future
            consistency).

    Returns:
        ``sionna.phy.nr.PUSCHConfig``
    """
    from sionna.phy.nr import (
        CarrierConfig as SionnaCarrierConfig,
    )
    from sionna.phy.nr import (
        PUSCHConfig as SionnaPUSCHConfig,
    )
    from sionna.phy.nr import (
        PUSCHDMRSConfig,
        TBConfig,
    )

    carrier = SionnaCarrierConfig(
        n_size_grid=phy_config.num_prb,
        subcarrier_spacing=phy_config.subcarrier_spacing_khz or 30,
    )
    dmrs = PUSCHDMRSConfig(
        config_type=phy_config.pusch_dmrs_config_type,
        length=phy_config.pusch_dmrs_length,
        additional_position=phy_config.pusch_dmrs_additional_position,
        num_cdm_groups_without_data=phy_config.pusch_num_cdm_groups_without_data,
    )
    tb = TBConfig(
        mcs_index=phy_config.mcs_index,
        mcs_table=phy_config.mcs_table,
    )
    # NR PUSCH requires num_layers == num_antenna_ports for non-codebook
    # precoding.  When layers < ports, switch to codebook-based precoding
    # which is the standard NR approach for rank-deficient transmission.
    pusch_kwargs: dict[str, Any] = {
        "num_layers": phy_config.num_layers,
        "num_antenna_ports": phy_config.num_antenna_ports,
    }
    if phy_config.num_layers < phy_config.num_antenna_ports:
        pusch_kwargs["precoding"] = "codebook"

    pusch_cfg = SionnaPUSCHConfig(
        carrier_config=carrier,
        pusch_dmrs_config=dmrs,
        tb_config=tb,
        **pusch_kwargs,
    )
    return pusch_cfg


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


def _cir_to_cfr(
    cir_coefficients: np.ndarray,
    cir_delays: np.ndarray,
    subcarrier_spacing_hz: float,
    num_subcarriers: int,
) -> np.ndarray | None:
    """Convert 6D CIR to 6D CFR using Sionna's ``cir_to_ofdm_channel``.

    Dimension mapping project -> Sionna:
      * coefficients [snap, tx, rx, rx_ant, tx_ant, path]
          -> [snap, rx, rx_ant, tx, tx_ant, path, 1]   (add time dim)
      * delays      [snap, tx, rx, rx_ant, tx_ant, path]
          -> [snap, rx, rx_ant, tx, tx_ant, path]

    Returns 6D array [snap, tx, rx, rx_ant, tx_ant, subcarrier] or ``None``
    if the conversion fails.
    """
    try:
        from sionna.phy.channel import cir_to_ofdm_channel, subcarrier_frequencies

        a = torch.as_tensor(cir_coefficients, dtype=torch.complex64)
        tau = torch.as_tensor(cir_delays, dtype=torch.float32)

        # coefficients: [snap, tx, rx, rx_ant, tx_ant, path]
        #         ->    [snap, rx, rx_ant, tx, tx_ant, path, 1]
        a_perm = a.permute(0, 2, 3, 1, 4, 5).unsqueeze(-1)

        # delays: per-antenna-pair convention
        #   [snap, tx, rx, rx_ant, tx_ant, path]
        #   -> [snap, rx, rx_ant, tx, tx_ant, path]
        tau_perm = tau.permute(0, 2, 3, 1, 4, 5)

        freqs = subcarrier_frequencies(num_subcarriers, subcarrier_spacing_hz)
        # output: [snap, rx, rx_ant, tx, tx_ant, 1, subcarrier]
        h_f = cir_to_ofdm_channel(freqs, a_perm, tau_perm)

        # remove time dim -> [snap, rx, rx_ant, tx, tx_ant, subcarrier]
        h_f = h_f.squeeze(-2)

        # back to project convention -> [snap, tx, rx, rx_ant, tx_ant, subcarrier]
        h_f = h_f.permute(0, 3, 1, 2, 4, 5)

        return h_f.numpy()
    except Exception as exc:
        raise RuntimeError(
            f"CIR-to-CFR conversion failed: {exc}"
        ) from exc


def _ls_estimate_cfr(
    y_noisy_np: np.ndarray,
    pusch_cfg,
    num_subcarriers: int,
) -> np.ndarray:
    """Compute LS channel estimate from DMRS pilot symbols.

    Uses the known DMRS pattern and symbols from the PUSCHConfig to
    perform a least-squares channel estimate at pilot positions, then
    interpolates across subcarriers to get the full CFR.

    Args:
        y_noisy_np: Noisy received frequency-domain signal of shape
            ``[num_ofdm_symbols, num_subcarriers]``.
        pusch_cfg: Sionna ``PUSCHConfig`` instance.
        num_subcarriers: Number of active subcarriers.

    Returns:
        6-D array ``[1, 1, 1, 1, 1, num_subcarriers]`` with dtype
        ``complex64`` containing the interpolated LS CFR estimate for
        the first (tx, rx, tx_ant, rx_ant) link.
    """
    dmrs_grid = pusch_cfg.dmrs_grid  # [layers, num_subcarriers, num_ofdm_symbols]

    # Use only positions that carry actual DMRS symbols (non-zero amplitude).
    # dmrs_mask may include reserved REs with zero-power DMRS (e.g. the
    # unoccupied comb of a comb-2 pattern), which would cause division by zero.
    dmrs_positions = np.abs(dmrs_grid[0]) > 1e-10
    sc_inds, sym_inds = np.where(dmrs_positions)
    num_pilots = len(sc_inds)

    if num_pilots == 0:
        return np.zeros(
            (1, 1, 1, 1, 1, num_subcarriers), dtype=np.complex64,
        )

    y_dmrs = y_noisy_np[sym_inds, sc_inds].astype(np.complex64)
    x_dmrs = dmrs_grid[0, sc_inds, sym_inds].astype(np.complex64)
    h_ls = y_dmrs / x_dmrs

    h_sum = np.zeros(num_subcarriers, dtype=np.complex64)
    np.add.at(h_sum, sc_inds, h_ls)
    count = np.bincount(sc_inds, minlength=num_subcarriers)

    sc_with = np.where(count > 0)[0]
    if len(sc_with) == num_subcarriers:
        h_est = h_sum / count.astype(np.float32)
    else:
        h_dmrs = h_sum[sc_with] / count[sc_with].astype(np.float32)
        all_sc = np.arange(num_subcarriers)
        h_real = np.interp(
            all_sc, sc_with.astype(np.float64), h_dmrs.real.astype(np.float64),
        )
        h_imag = np.interp(
            all_sc, sc_with.astype(np.float64), h_dmrs.imag.astype(np.float64),
        )
        h_est = (h_real.astype(np.float32) + 1j * h_imag.astype(np.float32)).astype(
            np.complex64,
        )

    return h_est.reshape(1, 1, 1, 1, 1, -1)


# ── main entry point ────────────────────────────────────────────────────


def run_nr_pusch_observation(
    cir_coefficients: np.ndarray,
    cir_delays: np.ndarray,
    link_config,
    phy_config,
    carrier_config,
) -> dict:
    """Run NR PUSCH uplink observation and return a results dictionary.

    Parameters
    ----------
    cir_coefficients : np.ndarray
        6-D ``[snap, tx, rx, rx_ant, tx_ant, path]`` complex CIR
        coefficients.
    cir_delays : np.ndarray
        6-D ``[snap, tx, rx, rx_ant, tx_ant, path]`` CIR delays in seconds.
    link_config : LinkConfig
        Link-layer config that controls reciprocity.
    phy_config : PHYConfig
        PHY-layer config with NR PUSCH fields.
    carrier_config : CarrierConfig
        Carrier / frequency grid config.

    Returns
    -------
    dict
        Keys:

        * ``cfr_est`` – observed CFR (ndarray)
        * ``ber`` – bit error rate (float, 0 for skeleton)
        * ``bler`` – block error rate (float, 0 for skeleton)
        * ``pusch_config`` – serialised PUSCHConfig snapshot (dict)
        * ``waveform_spec`` – custom-OFDM ``WaveformSpec`` from AWGN pipeline
        * ``nr_waveform_spec`` – NR PUSCH ``WaveformSpec``
        * ``receiver_spec`` – ``ReceiverSpec``
        * ``evaluation`` – ``EvaluationResult``
        * ``observation`` – ``ObservationResult``
        * ``impairments`` – ``ImpairmentSpec``
        * ``reciprocity_applied`` – whether reciprocity was applied (bool)
        * ``num_tx_bits`` – number of transmitted bits (int)
    """
    # 1. Apply TDD reciprocity for uplink CIR  ───────────────────────────
    cir_a_ul = cir_coefficients
    cir_tau_ul = cir_delays
    reciprocity_applied = False
    if link_config.reciprocity_mode == "transpose_rt_channel" and link_config.reciprocity_applied:
        try:
            from sionna_measurement_sim.phy.reciprocity import (
                apply_tdd_reciprocity_cir,
            )

            cir_a_ul = apply_tdd_reciprocity_cir(cir_coefficients)
            cir_tau_ul = apply_tdd_reciprocity_cir(cir_delays)
            reciprocity_applied = True
        except ImportError:
            pass  # reciprocity module absent – keep CIR as-is

    # 2. Build PUSCH config and transmitter  ─────────────────────────────
    pusch_cfg = build_nr_pusch_config(phy_config, carrier_config)

    from sionna.phy.nr import PUSCHTransmitter

    tx = PUSCHTransmitter(pusch_cfg, output_domain="freq", return_bits=True)
    # Generate TX signal (batch_size = 1)
    # tx_signal shape: [1, num_tx, num_streams, num_symbols, num_subcarriers]
    tx_signal, tx_bits = tx(1)

    num_tx_bits = int(tx_bits.shape[-1])
    num_subcarriers = int(pusch_cfg.num_subcarriers)  # num_prb × 12
    sc_spacing_hz = float(phy_config.subcarrier_spacing_khz) * 1000.0

    # 3. Convert CIR to CFR  ─────────────────────────────────────────────
    cfr_np = _cir_to_cfr(cir_a_ul, cir_tau_ul, sc_spacing_hz, num_subcarriers)
    # Save clean CIR-derived CFR as reference for NMSE computation (UL orientation)
    cfr_clean_ref = cfr_np.copy()
    sample_rate_hz = sc_spacing_hz * num_subcarriers

    # 4. Apply channel to PUSCH TX signal + AWGN  ────────────────────────
    import torch

    # CFR shape: [snap, tx, rx, rx_ant, tx_ant, subcarrier]
    # TX signal: [batch, num_tx, num_streams, num_symbols, num_subcarriers]
    # Need to map: CFR tx_ant dim → TX num_tx, CFR rx_ant dim → channel paths
    # For SISO-like processing: squeeze antenna dims to get channel matrix
    # CFR: [tx, rx, rx_ant, tx_ant, sub] — preserve all dims
    # TX signal: [batch, num_tx=1, num_streams=1, num_symbols, num_subcarriers]
    #
    # NOTE: Full MIMO PUSCH receiver requires GPU. For CPU implementation,
    # select antenna pair (0,0) for the first (tx,rx) pair. This is correct
    # for SISO scenarios but discards MIMO information. To process all TX/RX
    # pairs with full antenna dimensions, iterate per-pair with PUSCHReceiver.
    h_tensor = torch.as_tensor(cfr_np[0], dtype=torch.complex64)  # [tx, rx, rx_ant, tx_ant, sub]
    h_ch = h_tensor[0, 0, 0, 0, :]  # [subcarrier] — explicit antenna pair (0,0)
    tx_freq = tx_signal[0, 0, 0, :, :]  # [num_symbols, num_subcarriers]
    y_freq = tx_freq * h_ch.unsqueeze(0)  # [num_symbols, num_subcarriers]

    # Add AWGN
    signal_power = torch.mean(torch.abs(y_freq) ** 2)
    snr_db = getattr(phy_config, 'snr_db', None) or getattr(phy_config, 'observation_snr_db', 30.0)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = torch.sqrt(noise_power / 2.0) * (
        torch.randn_like(y_freq.real) + 1j * torch.randn_like(y_freq.real)
    )
    y_noisy = y_freq + noise.to(torch.complex64)

    # Reshape for receiver: [batch, num_rx, num_rx_ant, num_ofdm_symbols, fft_size]
    y_rx = y_noisy.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, symbols, subcarriers]
    no = noise_power * torch.ones(1, dtype=torch.float32)

    # 5. Build and run PUSCHReceiver  ────────────────────────────────────
    # Let PUSCHReceiver create its own default channel estimator and MIMO
    # detector (sionna.phy.ofdm.LinearDetector with resource grid and
    # stream management).  Passing a custom mimo_detector from
    # sionna.phy.mimo.LinearDetector would require the caller to manage
    # resource-grid-aware detection, which is handled internally when
    # defaults are used.
    from sionna.phy.nr import PUSCHReceiver

    rx = PUSCHReceiver(
        pusch_transmitter=tx,
        tb_decoder=None,
        return_tb_crc_status=False,
        input_domain="freq",
    )
    # Run receiver (wrapped to catch failures gracefully)
    receiver_failed = False
    try:
        rx_bits = rx(y_rx, no)
    except Exception:
        rx_bits = torch.zeros_like(tx_bits)
        receiver_failed = True
    # Compare bits
    num_bit_errors = int(torch.sum(torch.ne(rx_bits, tx_bits)).item())
    num_total_bits = int(tx_bits.shape[-1])
    ber = num_bit_errors / max(num_total_bits, 1)
    bler = 1.0 if num_bit_errors > 0 else 0.0

    # 6a. Compute receiver-estimated CFR from DMRS pilots (LS estimate)  ──
    cfr_est = _ls_estimate_cfr(
        y_noisy.cpu().numpy(), pusch_cfg, num_subcarriers,
    )

    # Expand to full CIR link dimensions.  The current PUSCH pipeline only
    # processes the first (tx,rx,tx_ant,rx_ant) link, so the same SISO LS
    # estimate is broadcast to all links for shape consistency.
    if cfr_est.shape != cfr_clean_ref.shape:
        cfr_est = np.broadcast_to(cfr_est, cfr_clean_ref.shape).copy()

    # 6b. Compute NMSE (before reciprocity reversal, both in UL orientation)
    error = cfr_est - cfr_clean_ref[0:1]
    nmse = np.mean(np.abs(error) ** 2, axis=(-3, -2, -1)) / np.clip(
        np.mean(np.abs(cfr_clean_ref[0:1]) ** 2, axis=(-3, -2, -1)),
        1e-30,
        None,
    )
    nmse_db = 10.0 * np.log10(nmse)

    # 6c. Reverse reciprocity to match truth CFR [snap, tx, rx, rx_ant, tx_ant, sub]
    if reciprocity_applied:
        cfr_np = np.transpose(cfr_np, (0, 2, 1, 4, 3, 5))
        cfr_est = np.transpose(cfr_est, (0, 2, 1, 4, 3, 5))

    link_shape = cfr_est.shape[:3]  # [snap, tx, rx]
    estimation_success_flag = not receiver_failed
    observation = ObservationResult(
        cfr_est=cfr_est,
        valid_mask=np.ones(link_shape, dtype=np.bool_),
        detection_success=np.ones(link_shape, dtype=np.bool_),
        estimation_success=np.full(link_shape, estimation_success_flag, dtype=np.bool_),
        snr_db=np.full(link_shape, float(snr_db), dtype=np.float32),
        rssi_dbm=np.zeros(link_shape, dtype=np.float32),
        noise_power_dbm=np.zeros(link_shape, dtype=np.float32),
        cfo_hz=np.zeros(link_shape, dtype=np.float32),
        sfo_ppm=np.zeros(link_shape, dtype=np.float32),
        timing_offset_samples=np.zeros(link_shape, dtype=np.float32),
        phase_offset_rad=np.zeros(link_shape, dtype=np.float32),
        agc_gain_db=np.zeros((link_shape[0], link_shape[2]), dtype=np.float32),
        clipping_flag=np.zeros(link_shape, dtype=np.bool_),
    )
    evaluation = EvaluationResult(
        nmse_db=nmse_db.astype(np.float32),
        nmse_db_total=nmse_db.astype(np.float32),
        amplitude_error_db=np.zeros(link_shape, dtype=np.float32),
        phase_error_rad=np.zeros(link_shape, dtype=np.float32),
        correlation=np.ones(link_shape, dtype=np.float32),
        detection_rate=1.0,
        estimation_failure_rate=float(1.0 - estimation_success_flag),
        ber=float(ber),
        bler=float(bler),
        num_bit_errors=num_bit_errors,
        num_bits=num_total_bits,
    )

    # 7. Build NR PUSCH WaveformSpec  ─────────────────────────────────────
    nr_waveform_spec = WaveformSpec(
        standard="nr_pusch",
        sample_rate_hz=sample_rate_hz,
        fft_size=num_subcarriers,
        cp_length=0,
        num_ofdm_symbols=phy_config.num_ofdm_symbols,
        pilot_indices=np.array([], dtype=np.int32),
        data_subcarrier_indices=np.arange(num_subcarriers, dtype=np.int32),
        pilot_symbols=np.array([], dtype=np.complex64),
        tx_power_dbm=phy_config.tx_power_dbm,
    )

    # 8. Assemble result  ─────────────────────────────────────────────────
    result: dict[str, Any] = {
        "cfr_est": cfr_est,
        "cfr_clean_ref": cfr_clean_ref[0:1],
        "ber": evaluation.ber,
        "bler": evaluation.bler,
        "pusch_config": pusch_config_to_dict(pusch_cfg),
        "waveform_spec": nr_waveform_spec,
        "nr_waveform_spec": nr_waveform_spec,
        "receiver_spec": ReceiverSpec(receiver_type="pusch_receiver"),
        "evaluation": evaluation,
        "observation": observation,
        "impairments": ImpairmentSpec(
            model_version="nr_pusch_v1",
            random_seed=42,
            awgn_config=f'{{"snr_db": {float(snr_db)}}}',
        ),
        "reciprocity_applied": reciprocity_applied,
        "num_tx_bits": num_tx_bits,
        "tx_signal_shape": list(tx_signal.shape),
    }
    return result
