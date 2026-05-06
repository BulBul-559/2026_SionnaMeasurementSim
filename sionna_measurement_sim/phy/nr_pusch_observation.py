"""NR PUSCH observation backend (Steps 3+5).

Builds a proper Sionna PUSCHConfig from project configuration and runs a
working observation pipeline.  The actual channel estimation currently
delegates to the existing AWGN+LS pipeline while recording NR PUSCH waveform
parameters in the results.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from sionna_measurement_sim.domain.observation import (
    WaveformSpec,
)
from sionna_measurement_sim.phy.observation_pipeline import (
    AWGNObservationConfig,
    PHYObservationBundle,
    run_awgn_ls_observation,
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
        import warnings

        warnings.warn(f"CIR-to-CFR conversion failed: {exc}", stacklevel=2)
        return None


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

    # Fallback: create unit-magnitude CFR when conversion fails
    if cfr_np is None:
        num_snapshots = cir_coefficients.shape[0]
        num_tx = cir_coefficients.shape[1]
        num_rx = cir_coefficients.shape[2]
        num_rx_ant = cir_coefficients.shape[3]
        num_tx_ant = cir_coefficients.shape[4]
        cfr_np = np.ones(
            (num_snapshots, num_tx, num_rx, num_rx_ant, num_tx_ant, num_subcarriers),
            dtype=np.complex64,
        )

    # 4. Run AWGN + LS observation pipeline  ─────────────────────────────
    # Approximate sample rate for impairment modelling
    sample_rate_hz = sc_spacing_hz * num_subcarriers

    obs_config = AWGNObservationConfig(
        snr_db=phy_config.snr_db,
        random_seed=42,
        sample_rate_hz=sample_rate_hz,
        fft_size=num_subcarriers,
        cp_length=0,
        num_ofdm_symbols=phy_config.num_ofdm_symbols,
        tx_power_dbm=phy_config.tx_power_dbm,
    )

    bundle: PHYObservationBundle = run_awgn_ls_observation(
        h_true=cfr_np[0],  # 5-D [tx, rx, rx_ant, tx_ant, subcarrier]
        config=obs_config,
        cfr_snapshots=cfr_np,  # 6-D full batch
    )

    # 5. Build an NR PUSCH-specific WaveformSpec  ────────────────────────
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

    # 6. BER / BLER (placeholder – no real decoding in skeleton)  ────────
    ber = 0.0
    bler = 0.0

    # 7. Assemble result dictionary  ─────────────────────────────────────
    result: dict[str, Any] = {
        "cfr_est": bundle.observation.cfr_est,
        "ber": ber,
        "bler": bler,
        "pusch_config": pusch_config_to_dict(pusch_cfg),
        "waveform_spec": bundle.waveform,
        "nr_waveform_spec": nr_waveform_spec,
        "receiver_spec": bundle.receiver,
        "evaluation": bundle.evaluation,
        "observation": bundle.observation,
        "impairments": bundle.impairments,
        "reciprocity_applied": reciprocity_applied,
        "num_tx_bits": num_tx_bits,
        "tx_signal_shape": list(tx_signal.shape),
    }
    return result
