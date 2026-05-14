"""Tests for the NR PUSCH observation backend."""

from __future__ import annotations

import numpy as np
import pytest

from sionna_measurement_sim.config.schema import CarrierConfig, PHYConfig
from sionna_measurement_sim.domain.array import ArraySpectrumConfig
from sionna_measurement_sim.domain.link import LinkConfig
from sionna_measurement_sim.phy.nr_pusch_observation import (
    build_array_outputs_from_waveform,
    build_nr_pusch_config,
    pusch_config_to_dict,
    run_nr_pusch_observation,
)


class TestBuildNRPUSCHConfig:
    """Verify PUSCHConfig construction from project config objects."""

    def test_build_returns_object_with_expected_attributes(self):
        phy = PHYConfig()
        carrier = CarrierConfig()
        pc = build_nr_pusch_config(phy, carrier)

        assert pc.num_layers == phy.num_layers
        assert pc.num_antenna_ports == phy.num_antenna_ports
        assert pc.num_resource_blocks == phy.num_prb
        assert pc.tb.mcs_index == phy.mcs_index
        assert pc.tb.mcs_table == phy.mcs_table
        assert pc.dmrs.config_type == phy.pusch_dmrs_config_type
        assert pc.dmrs.length == phy.pusch_dmrs_length
        assert pc.dmrs.additional_position == phy.pusch_dmrs_additional_position
        assert pc.dmrs.num_cdm_groups_without_data == phy.pusch_num_cdm_groups_without_data
        assert pc.carrier.subcarrier_spacing == phy.subcarrier_spacing_khz

    def test_build_with_custom_values(self):
        phy = PHYConfig(
            num_prb=52,
            num_ofdm_symbols=14,
            subcarrier_spacing_khz=15,
            num_layers=2,
            num_antenna_ports=2,
            mcs_index=20,
            mcs_table=2,
            pusch_dmrs_config_type=2,
            pusch_dmrs_length=2,
            pusch_dmrs_additional_position=0,
            pusch_num_cdm_groups_without_data=1,
        )
        carrier = CarrierConfig()
        pc = build_nr_pusch_config(phy, carrier)

        assert pc.num_layers == 2
        assert pc.num_antenna_ports == 2
        assert pc.num_resource_blocks == 52
        assert pc.tb.mcs_index == 20
        assert pc.tb.mcs_table == 2
        assert pc.dmrs.config_type == 2
        assert pc.dmrs.length == 2
        assert pc.dmrs.additional_position == 0
        assert pc.dmrs.num_cdm_groups_without_data == 1
        assert pc.carrier.subcarrier_spacing == 15

    def test_subcarrier_count_matches_prb_times_12(self):
        phy = PHYConfig(num_prb=16)
        carrier = CarrierConfig()
        pc = build_nr_pusch_config(phy, carrier)
        assert int(pc.num_subcarriers) == phy.num_prb * 12

    def test_carrier_num_symbols_per_slot_defaults_to_14(self):
        """NR always has 14 symbols per slot for normal CP."""
        phy = PHYConfig()
        carrier = CarrierConfig()
        pc = build_nr_pusch_config(phy, carrier)
        # num_symbols_per_slot is a read-only property derived from numerology
        assert pc.carrier.num_symbols_per_slot == 14


class TestPUSCHConfigToDict:
    """Verify serialisation to a plain dict."""

    def test_dict_contains_expected_keys(self):
        phy = PHYConfig()
        carrier = CarrierConfig()
        pc = build_nr_pusch_config(phy, carrier)
        d = pusch_config_to_dict(pc)

        assert isinstance(d, dict)
        assert d["num_layers"] == phy.num_layers
        assert d["num_antenna_ports"] == phy.num_antenna_ports
        assert d["num_resource_blocks"] == phy.num_prb
        assert d["num_subcarriers"] == phy.num_prb * 12
        assert d["mcs_index"] == phy.mcs_index
        assert d["mcs_table"] == phy.mcs_table
        assert d["dmrs_config_type"] == phy.pusch_dmrs_config_type
        assert d["carrier_subcarrier_spacing"] == phy.subcarrier_spacing_khz

    def test_dict_is_json_serializable(self):
        import json

        phy = PHYConfig()
        carrier = CarrierConfig()
        pc = build_nr_pusch_config(phy, carrier)
        d = pusch_config_to_dict(pc)
        # Should not raise
        json.dumps(d)


class TestRunNRPUSCHObservation:
    """Integration-level tests for the full observation function."""

    def test_returns_dict_with_expected_keys(self):
        # Build small CIR arrays for a 2×2×2×2 scenario
        num_snapshots = 1
        num_tx = 2
        num_rx = 2
        num_rx_ant = 2
        num_tx_ant = 2
        num_paths = 3

        cir_coeff = np.ones(
            (num_snapshots, num_tx, num_rx, num_rx_ant, num_tx_ant, num_paths),
            dtype=np.complex64,
        )
        cir_delays = np.linspace(
            0, 1e-7, num_paths, dtype=np.float32,
        ).reshape(1, 1, 1, 1, 1, num_paths)
        cir_delays = np.broadcast_to(
            cir_delays, (num_snapshots, num_tx, num_rx, num_rx_ant, num_tx_ant, num_paths),
        ).copy()

        link = LinkConfig()
        phy = PHYConfig(num_prb=4, num_ofdm_symbols=14, snr_db=40.0,
                        num_antenna_ports=2, num_layers=2)
        carrier = CarrierConfig()

        result = run_nr_pusch_observation(cir_coeff, cir_delays, link, phy, carrier)

        expected_keys = {
            "cfr_est", "ber", "bler", "pusch_config", "waveform_spec",
            "nr_waveform_spec", "receiver_spec", "evaluation", "observation",
            "impairments", "reciprocity_applied", "num_tx_bits", "tx_signal_shape",
            "waveform_grids", "array_outputs",
        }
        missing = expected_keys - result.keys()
        assert expected_keys.issubset(result.keys()), f"Missing keys: {missing}"

    def test_cfr_est_shape_matches_truth(self):
        num_snapshots = 1
        num_tx = 2
        num_rx = 3
        num_rx_ant = 1
        num_tx_ant = 1
        num_paths = 3

        cir_coeff = np.ones(
            (num_snapshots, num_tx, num_rx, num_rx_ant, num_tx_ant, num_paths),
            dtype=np.complex64,
        )
        cir_delays = np.broadcast_to(
            np.linspace(0, 1e-7, num_paths, dtype=np.float32).reshape(1, 1, 1, 1, 1, num_paths),
            (num_snapshots, num_tx, num_rx, num_rx_ant, num_tx_ant, num_paths),
        ).copy()

        phy = PHYConfig(num_prb=4, snr_db=40.0, num_antenna_ports=1, num_layers=1)
        # Disable reciprocity to keep tx/rx dimensions unchanged
        link = LinkConfig(reciprocity_applied=False)
        carrier = CarrierConfig()
        result = run_nr_pusch_observation(cir_coeff, cir_delays, link, phy, carrier)

        cfr_est = result["cfr_est"]
        # Shape: [snap, tx, rx, rx_ant, tx_ant, subcarrier]
        assert cfr_est.ndim == 6
        assert cfr_est.shape[0] == num_snapshots
        assert cfr_est.shape[1] == num_tx
        assert cfr_est.shape[2] == num_rx
        assert cfr_est.shape[3] == num_rx_ant
        assert cfr_est.shape[4] == num_tx_ant
        assert cfr_est.shape[5] == phy.num_prb * 12

    def test_nr_waveform_spec_standard(self):
        cir_coeff = np.ones((1, 1, 1, 1, 1, 2), dtype=np.complex64)
        cir_delays = np.ones((1, 1, 1, 1, 1, 2), dtype=np.float32) * 1e-9

        phy = PHYConfig(num_prb=4, snr_db=40.0, num_antenna_ports=1, num_layers=1)
        link = LinkConfig()
        carrier = CarrierConfig()
        result = run_nr_pusch_observation(cir_coeff, cir_delays, link, phy, carrier)

        nr_wv = result["nr_waveform_spec"]
        assert nr_wv.standard == "nr_pusch"
        assert nr_wv.fft_size == phy.num_prb * 12

    def test_waveform_grids_are_actual_frequency_domain_outputs(self):
        cir_coeff = np.ones((1, 1, 1, 1, 1, 2), dtype=np.complex64)
        cir_delays = np.ones((1, 1, 1, 1, 1, 2), dtype=np.float32) * 1e-9

        phy = PHYConfig(num_prb=4, snr_db=40.0, num_antenna_ports=1, num_layers=1)
        link = LinkConfig()
        carrier = CarrierConfig()
        result = run_nr_pusch_observation(cir_coeff, cir_delays, link, phy, carrier)

        grids = result["waveform_grids"]
        assert grids["tx_grid"].shape == (1, 1, 1, 1, 14, 48)
        assert grids["rx_grid"].shape == (1, 1, 1, 1, 14, 48)
        assert grids["noise_variance"].shape == (1, 1, 1)
        assert grids["tx_grid"].dtype == np.complex64
        assert grids["rx_grid"].dtype == np.complex64
        assert np.any(np.abs(grids["tx_grid"]) > 0.0)
        assert np.any(np.abs(grids["rx_grid"]) > 0.0)
        np.testing.assert_allclose(grids["noise_variance"], 1e-4, rtol=1e-5)

    def test_su_mimo_batch_size_1_and_4_schema_shapes_match(self):
        cir_coeff = np.ones((1, 2, 2, 1, 1, 2), dtype=np.complex64)
        cir_delays = np.ones((1, 2, 2, 1, 1, 2), dtype=np.float32) * 1e-9

        link = LinkConfig(reciprocity_applied=False)
        carrier = CarrierConfig()

        def _run(batch_size: int):
            phy = PHYConfig(
                num_prb=4,
                snr_db=40.0,
                num_antenna_ports=1,
                num_layers=1,
                perfect_csi=False,
            )
            object.__setattr__(phy, "nr_pusch_batch_size", batch_size)
            return run_nr_pusch_observation(cir_coeff, cir_delays, link, phy, carrier)

        result_1 = _run(1)
        result_4 = _run(4)

        for key in ("cfr_est", "cfr_clean_ref"):
            assert result_4[key].shape == result_1[key].shape
            assert result_4[key].dtype == result_1[key].dtype

        for key in ("tx_grid", "rx_grid", "noise_variance"):
            assert (
                result_4["waveform_grids"][key].shape
                == result_1["waveform_grids"][key].shape
            )
            assert (
                result_4["waveform_grids"][key].dtype
                == result_1["waveform_grids"][key].dtype
            )

        for key in ("nmse_db", "nmse_db_total", "amplitude_error_db"):
            assert getattr(result_4["evaluation"], key).shape == getattr(
                result_1["evaluation"], key,
            ).shape
        assert result_4["observation"].cfr_est.shape == result_1["observation"].cfr_est.shape
        assert set(result_4["pusch_config"]) == set(result_1["pusch_config"])
        assert set(result_4["array_outputs"]) == set(result_1["array_outputs"])
        assert result_1["batching_stats"]["requested_batch_size"] == 1
        assert result_4["batching_stats"]["requested_batch_size"] == 4
        assert result_4["batching_stats"]["effective_batch_size"] == 4
        assert result_4["batching_stats"]["num_batches"] == 1

    def test_array_outputs_have_deterministic_fallback_shapes(self):
        rx_grid = np.zeros((1, 2, 3, 4, 14, 48), dtype=np.complex64)

        arrays = build_array_outputs_from_waveform(rx_grid)

        assert arrays["rx_snapshot_matrix"].shape == (1, 2, 3, 4, 4)
        assert arrays["aoa_label_rad"].shape == (1, 2, 3, 2)
        assert arrays["aoa_heatmap_label"].shape == (1, 2, 3, 91, 181)
        assert arrays["spatial_spectrum_label"].shape == (1, 2, 3, 91, 181)
        assert arrays["angle_grid_rad"].shape == (91, 181, 2)
        assert "spatial_spectrum_observation" not in arrays
        assert np.all(arrays["spatial_spectrum_label"] == 0.0)
        np.testing.assert_allclose(arrays["angle_grid_rad"][0, 0], [0.0, -np.pi])
        np.testing.assert_allclose(arrays["angle_grid_rad"][-1, -1], [np.pi, np.pi])

    def test_array_outputs_accept_aoa_label_hook(self):
        rx_grid = np.ones((1, 1, 1, 2, 2, 2), dtype=np.complex64)
        aoa = np.array([[[[np.pi / 2.0, 0.0]]]], dtype=np.float32)

        arrays = build_array_outputs_from_waveform(rx_grid, aoa_label_rad=aoa)

        spectrum = arrays["spatial_spectrum_label"]
        assert np.count_nonzero(spectrum) == 1
        assert spectrum[0, 0, 0, 45, 90] == 1.0
        np.testing.assert_allclose(arrays["aoa_label_rad"], aoa)

    def test_array_outputs_respect_spectrum_resolution_config(self):
        rx_grid = np.ones((1, 1, 1, 4, 2, 2), dtype=np.complex64)
        config = ArraySpectrumConfig(
            enabled=True,
            sources=("rx_grid",),
            zenith_bins=7,
            azimuth_bins=9,
        )

        arrays = build_array_outputs_from_waveform(
            rx_grid,
            spectrum_config=config,
            rx_num_rows=2,
            rx_num_cols=2,
        )

        assert arrays["angle_grid_rad"].shape == (7, 9, 2)
        assert arrays["spatial_spectrum_label"].shape == (1, 1, 1, 7, 9)
        assert arrays["spatial_spectrum_observation"].shape == (1, 1, 1, 7, 9)
        assert np.all(np.isfinite(arrays["spatial_spectrum_observation"]))

    def test_array_outputs_export_cfr_est_spectrum_when_samples_are_provided(self):
        rx_grid = np.ones((1, 1, 1, 4, 2, 2), dtype=np.complex64)
        cfr_est_samples = np.ones((1, 1, 1, 4, 2, 2), dtype=np.complex64)
        config = ArraySpectrumConfig(
            enabled=True,
            sources=("cfr_est",),
            zenith_bins=7,
            azimuth_bins=9,
        )

        arrays = build_array_outputs_from_waveform(
            rx_grid,
            spectrum_config=config,
            rx_num_rows=2,
            rx_num_cols=2,
            cfr_est_spectrum_samples=cfr_est_samples,
        )

        assert arrays["spatial_spectrum_cfr_est"].shape == (1, 1, 1, 7, 9)
        assert "spatial_spectrum_observation" not in arrays
        assert np.all(np.isfinite(arrays["spatial_spectrum_cfr_est"]))

    def test_reciprocity_applied_flag(self):
        cir_coeff = np.ones((1, 2, 2, 1, 1, 2), dtype=np.complex64)
        cir_delays = np.ones((1, 2, 2, 1, 1, 2), dtype=np.float32) * 1e-9

        link = LinkConfig(reciprocity_applied=True)
        phy = PHYConfig(num_prb=4, snr_db=40.0, num_antenna_ports=1, num_layers=1)
        carrier = CarrierConfig()
        result = run_nr_pusch_observation(cir_coeff, cir_delays, link, phy, carrier)

        assert result["reciprocity_applied"] is True

    def test_no_reciprocity_when_disabled(self):
        cir_coeff = np.ones((1, 2, 2, 1, 1, 2), dtype=np.complex64)
        cir_delays = np.ones((1, 2, 2, 1, 1, 2), dtype=np.float32) * 1e-9

        link = LinkConfig(reciprocity_applied=False)
        phy = PHYConfig(num_prb=4, snr_db=40.0, num_antenna_ports=1, num_layers=1)
        carrier = CarrierConfig()
        result = run_nr_pusch_observation(cir_coeff, cir_delays, link, phy, carrier)

        assert result["reciprocity_applied"] is False

    def test_evaluation_metrics_are_finite(self):
        cir_coeff = np.ones((1, 1, 1, 1, 1, 2), dtype=np.complex64)
        cir_delays = np.ones((1, 1, 1, 1, 1, 2), dtype=np.float32) * 1e-9

        link = LinkConfig()
        phy = PHYConfig(num_prb=4, snr_db=40.0, num_antenna_ports=1, num_layers=1)
        carrier = CarrierConfig()
        result = run_nr_pusch_observation(cir_coeff, cir_delays, link, phy, carrier)

        ev = result["evaluation"]
        assert np.all(np.isfinite(ev.nmse_db))
        assert np.all(np.isfinite(ev.correlation))
        assert ev.detection_rate >= 0.0

    def test_pusch_config_snapshot_matches_phy_config(self):
        cir_coeff = np.ones((1, 1, 1, 2, 2, 2), dtype=np.complex64)
        cir_delays = np.ones((1, 1, 1, 2, 2, 2), dtype=np.float32) * 1e-9

        phy = PHYConfig(num_prb=8, num_layers=2, num_antenna_ports=2, mcs_index=20)
        link = LinkConfig()
        carrier = CarrierConfig()
        result = run_nr_pusch_observation(cir_coeff, cir_delays, link, phy, carrier)

        cfg = result["pusch_config"]
        assert cfg["num_layers"] == 2
        assert cfg["num_resource_blocks"] == 8
        assert cfg["mcs_index"] == 20

    def test_ber_bler_placeholders(self):
        cir_coeff = np.ones((1, 1, 1, 1, 1, 2), dtype=np.complex64)
        cir_delays = np.ones((1, 1, 1, 1, 1, 2), dtype=np.float32) * 1e-9

        link = LinkConfig()
        phy = PHYConfig(num_prb=4, snr_db=40.0, num_antenna_ports=1, num_layers=1)
        carrier = CarrierConfig()
        result = run_nr_pusch_observation(cir_coeff, cir_delays, link, phy, carrier)

        # Skeleton returns 0 for both
        assert result["ber"] == 0.0
        assert result["bler"] == 0.0

    def test_estimated_csi_rejects_unequal_layers_ports(self):
        """num_layers=1, num_antenna_ports=4 with perfect_csi=False must error.

        CIR has 4 UE antennas (rx_ant=4) and 4 BS antennas (tx_ant=4),
        so UL: ul_tx_ant=4 matches num_antenna_ports=4.
        The estimator returns h_hat with num_streams_per_tx=1
        which must trigger NotImplementedError.
        """
        cir_coeff = np.ones((1, 1, 1, 4, 4, 2), dtype=np.complex64)
        cir_delays = np.ones((1, 1, 1, 4, 4, 2), dtype=np.float32) * 1e-9

        link = LinkConfig(reciprocity_applied=False)
        phy = PHYConfig(num_prb=4, snr_db=40.0, num_layers=1, num_antenna_ports=4,
                        perfect_csi=False)
        carrier = CarrierConfig()
        with pytest.raises(NotImplementedError, match="num_layers == num_antenna_ports"):
            run_nr_pusch_observation(cir_coeff, cir_delays, link, phy, carrier)

    def test_estimated_csi_allowed_when_equal_layers_ports(self):
        """num_layers=4, num_antenna_ports=4 with perfect_csi=False must succeed."""
        cir_coeff = np.ones((1, 1, 1, 4, 4, 2), dtype=np.complex64)
        cir_delays = np.ones((1, 1, 1, 4, 4, 2), dtype=np.float32) * 1e-9

        link = LinkConfig(reciprocity_applied=False)
        phy = PHYConfig(num_prb=4, snr_db=40.0, num_layers=4, num_antenna_ports=4,
                        perfect_csi=False)
        carrier = CarrierConfig()
        result = run_nr_pusch_observation(cir_coeff, cir_delays, link, phy, carrier)
        assert result["cfr_est"].shape[3] == 4
        assert result["cfr_est"].shape[4] == 4
