"""Unit tests for NR PUSCH MIMO configuration and detector builders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sionna_measurement_sim.config.schema import CarrierConfig, PHYConfig


class TestBuildMultiUserPUSCHConfigs:
    """Tests for build_multiuser_pusch_configs."""

    def test_su_mimo_single_config(self):
        from sionna_measurement_sim.phy.nr_pusch_observation import (
            build_multiuser_pusch_configs,
        )

        phy = PHYConfig(
            num_prb=4,
            num_layers=1,
            num_antenna_ports=4,
            mcs_index=14,
            mcs_table=1,
        )
        carrier = CarrierConfig()
        configs = build_multiuser_pusch_configs(phy, carrier)
        assert len(configs) == 1
        pc = configs[0]
        assert pc.num_layers == 1
        assert pc.num_antenna_ports == 4
        assert pc.precoding == "codebook"
        assert pc.dmrs.dmrs_port_set == [0]

    def test_full_rank_no_precoding(self):
        from sionna_measurement_sim.phy.nr_pusch_observation import (
            build_multiuser_pusch_configs,
        )

        phy = PHYConfig(
            num_prb=4,
            num_layers=4,
            num_antenna_ports=4,
            mcs_index=14,
            mcs_table=1,
        )
        carrier = CarrierConfig()
        configs = build_multiuser_pusch_configs(phy, carrier)
        pc = configs[0]
        assert pc.num_layers == 4
        assert pc.num_antenna_ports == 4
        assert pc.precoding != "codebook"  # no codebook when layers == ports

    def test_mu_mimo_two_users(self):
        from sionna_measurement_sim.phy.nr_pusch_observation import (
            build_multiuser_pusch_configs,
        )
        from sionna_measurement_sim.rt.truth_pipeline import RTTruthRunConfig

        # Use RTTruthRunConfig which allows arbitrary attributes
        phy = RTTruthRunConfig(
            label_file=Path("tests/fixtures/scenes/test/test5.json"),
            scene_file=Path("tests/fixtures/scenes/test/scene.xml"),
            output_dir=Path("/tmp"),
            num_prb=4,
            num_layers=1,
            num_antenna_ports=4,
            mcs_index=14,
            mcs_table=1,
        )
        carrier = CarrierConfig()
        configs = build_multiuser_pusch_configs(phy, carrier, num_pusch_tx=2)
        assert len(configs) == 2
        assert configs[0].dmrs.dmrs_port_set == [0]
        assert configs[1].dmrs.dmrs_port_set == [1]

    def test_dmrs_port_sets_dont_overlap_multi_user(self):
        from sionna_measurement_sim.phy.nr_pusch_observation import (
            build_multiuser_pusch_configs,
        )
        from sionna_measurement_sim.rt.truth_pipeline import RTTruthRunConfig

        phy = RTTruthRunConfig(
            label_file=Path("tests/fixtures/scenes/test/test5.json"),
            scene_file=Path("tests/fixtures/scenes/test/scene.xml"),
            output_dir=Path("/tmp"),
            num_prb=4,
            num_layers=1,
            num_antenna_ports=4,
        )
        carrier = CarrierConfig()
        configs = build_multiuser_pusch_configs(phy, carrier, num_pusch_tx=4)
        assert len(configs) == 4
        all_ports = []
        for cfg in configs:
            all_ports.extend(cfg.dmrs.dmrs_port_set)
        assert len(all_ports) == len(set(all_ports)), "DMRS port sets must not overlap"
        assert sorted(all_ports) == list(range(4))  # 4 UEs × 1 layer = 4 ports

    def test_num_layers_less_than_one_raises(self):
        from sionna_measurement_sim.phy.nr_pusch_observation import (
            build_multiuser_pusch_configs,
        )

        phy = PHYConfig(num_layers=0, num_antenna_ports=4)
        carrier = CarrierConfig()
        with pytest.raises(ValueError, match="num_layers"):
            build_multiuser_pusch_configs(phy, carrier)

    def test_antenna_ports_less_than_layers_raises(self):
        from sionna_measurement_sim.phy.nr_pusch_observation import (
            build_multiuser_pusch_configs,
        )

        phy = PHYConfig(num_layers=4, num_antenna_ports=2)
        carrier = CarrierConfig()
        with pytest.raises(ValueError, match="num_antenna_ports"):
            build_multiuser_pusch_configs(phy, carrier)

    def test_too_many_ports_raises(self):
        from sionna_measurement_sim.phy.nr_pusch_observation import (
            build_multiuser_pusch_configs,
        )

        phy = PHYConfig(num_layers=4, num_antenna_ports=4)
        carrier = CarrierConfig()
        with pytest.raises(ValueError, match="DMRS ports"):
            build_multiuser_pusch_configs(phy, carrier, num_pusch_tx=4)


class TestBuildStreamManagement:
    """Tests for build_stream_management."""

    def test_1x1_su_mimo(self):
        from sionna_measurement_sim.phy.nr_pusch_observation import (
            build_stream_management,
        )

        sm = build_stream_management(num_rx=1, num_tx=1, num_layers=1)
        assert sm.num_rx == 1
        assert sm.num_tx == 1
        assert sm.num_streams_per_tx == 1

    def test_1x4_mu_mimo(self):
        from sionna_measurement_sim.phy.nr_pusch_observation import (
            build_stream_management,
        )

        sm = build_stream_management(num_rx=1, num_tx=4, num_layers=1)
        assert sm.num_rx == 1
        assert sm.num_tx == 4
        assert sm.num_streams_per_tx == 1

    def test_rx_tx_association_all_true(self):
        from sionna_measurement_sim.phy.nr_pusch_observation import (
            build_stream_management,
        )

        # Use a configuration that StreamManagement supports:
        # 1 RX, 2 TX, 1 layer each (2 streams per RX)
        sm = build_stream_management(num_rx=1, num_tx=2, num_layers=1)
        expected = np.ones([1, 2], dtype=bool)
        np.testing.assert_array_equal(sm.rx_tx_association, expected)


class TestBuildMIMODetector:
    """Tests for build_mimo_detector."""

    def test_lmmse_detector(self):
        from sionna.phy.nr import PUSCHTransmitter

        from sionna_measurement_sim.phy.nr_pusch_observation import (
            build_mimo_detector,
            build_multiuser_pusch_configs,
            build_stream_management,
        )

        phy = PHYConfig(num_prb=4, num_layers=4, num_antenna_ports=4)
        carrier = CarrierConfig()
        configs = build_multiuser_pusch_configs(phy, carrier)
        tx = PUSCHTransmitter(configs, output_domain="freq")
        sm = build_stream_management(num_rx=1, num_tx=1, num_layers=4)

        det = build_mimo_detector(
            tx.resource_grid, sm, detector_type="lmmse", num_bits_per_symbol=4,
        )
        from sionna.phy.ofdm import LinearDetector

        assert isinstance(det, LinearDetector)

    def test_kbest_detector(self):
        from sionna.phy.nr import PUSCHTransmitter

        from sionna_measurement_sim.phy.nr_pusch_observation import (
            build_mimo_detector,
            build_multiuser_pusch_configs,
            build_stream_management,
        )

        phy = PHYConfig(num_prb=4, num_layers=1, num_antenna_ports=1)
        carrier = CarrierConfig()
        configs = build_multiuser_pusch_configs(phy, carrier)
        tx = PUSCHTransmitter(configs, output_domain="freq")
        sm = build_stream_management(num_rx=1, num_tx=1, num_layers=1)

        det = build_mimo_detector(
            tx.resource_grid, sm, detector_type="kbest", num_bits_per_symbol=4,
        )
        from sionna.phy.ofdm import KBestDetector

        assert isinstance(det, KBestDetector)

    def test_unknown_detector_raises(self):
        from sionna.phy.nr import PUSCHTransmitter

        from sionna_measurement_sim.phy.nr_pusch_observation import (
            build_mimo_detector,
            build_multiuser_pusch_configs,
            build_stream_management,
        )

        phy = PHYConfig(num_prb=4)
        carrier = CarrierConfig()
        configs = build_multiuser_pusch_configs(phy, carrier)
        tx = PUSCHTransmitter(configs, output_domain="freq")
        sm = build_stream_management(num_rx=1, num_tx=1, num_layers=1)

        with pytest.raises(ValueError, match="mimo_detector"):
            build_mimo_detector(
                tx.resource_grid, sm,
                detector_type="unknown_det",
                num_bits_per_symbol=4,
            )
