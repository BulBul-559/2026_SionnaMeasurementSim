"""NR PUSCH 4x4 SU-MIMO integration tests.

Self-generates HDF5 output and validates:
- 4x4 MIMO channel is consumed (not SISO broadcast)
- /observation/cfr_est shape matches /channel/truth/cfr shape
- Different antenna pairs have distinct CFR estimates
- mimo_detector field reflects actual detector
- StreamManagement and MIMO detector are active
"""

from pathlib import Path

import h5py
import numpy as np
import pytest


def _run_4x4_mimo_pipeline(
    tmp_path: Path,
    num_layers: int = 1,
    num_antenna_ports: int = 4,
    perfect_csi: bool = True,
    snr_db: float = 40.0,
    channel_backend: str = "apply_ofdm",
) -> Path:
    """Run NR PUSCH pipeline with 4x4 antenna and return HDF5 path."""
    from sionna_measurement_sim.rt.truth_pipeline import (
        RTTruthRunConfig,
        run_rt_truth_pipeline,
    )

    config = RTTruthRunConfig(
        label_file=Path("data/scenes/test/test5.json"),
        scene_file=Path("data/scenes/test/scene.xml"),
        output_dir=tmp_path / "output_4x4",
        num_subcarriers=48,
        seed=42,
        max_tx=1,
        max_rx=1,
        tx_num_rows=2,
        tx_num_cols=2,
        rx_num_rows=2,
        rx_num_cols=2,
        max_depth=3,
        los=True,
        specular_reflection=True,
        observation_snr_db=snr_db,
        phy_standard="nr_pusch",
        num_prb=4,
        num_layers=num_layers,
        num_antenna_ports=num_antenna_ports,
        mcs_index=14,
        mcs_table=1,
        perfect_csi=perfect_csi,
        receiver_failure_policy="fail_fast",
        mimo_detector="lmmse",
        channel_estimator="perfect" if perfect_csi else "pusch_ls",
        channel_backend=channel_backend,
    )
    return run_rt_truth_pipeline(config)


class TestNRPUSCH4x4MIMO:
    """4x4 SU-MIMO NR PUSCH integration tests."""

    def test_4x4_pipeline_self_generated(self, tmp_path):
        """Self-generated 4x4 MIMO pipeline produces valid HDF5."""
        try:
            path = _run_4x4_mimo_pipeline(tmp_path, perfect_csi=True)
        except ImportError:
            pytest.skip("NR PUSCH receiver not available")
        except Exception as exc:
            pytest.fail(f"4x4 MIMO pipeline failed: {exc}")

        assert path.exists()

        with h5py.File(path, "r") as h5:
            truth_cfr = h5["channel/truth/cfr"][()]
            cfr_est = h5["observation/cfr_est"][()]

            # Shape contracts
            assert truth_cfr.ndim == 5
            assert cfr_est.ndim == 6
            assert cfr_est.shape[1:] == truth_cfr.shape, (
                f"cfr_est.shape[1:]={cfr_est.shape[1:]} != "
                f"truth_cfr.shape={truth_cfr.shape}"
            )

            # Antenna dimensions
            assert truth_cfr.shape[2] == 4, f"Expected 4 rx_ant, got {truth_cfr.shape[2]}"
            assert truth_cfr.shape[3] == 4, f"Expected 4 tx_ant, got {truth_cfr.shape[3]}"

            # mimo_detector matches actual config
            rx_det = h5["receiver/mimo_detector"][()]
            if isinstance(rx_det, bytes):
                rx_det = rx_det.decode()
            assert rx_det == "lmmse", f"mimo_detector={rx_det!r}, expected 'lmmse'"

            # estimator_type matches config
            est_type = h5["receiver/estimator_type"][()]
            if isinstance(est_type, bytes):
                est_type = est_type.decode()
            assert est_type == "perfect", f"estimator_type={est_type!r}"

            # Estimation success
            est_success = h5["observation/estimation_success"][()]
            assert np.all(est_success), "Not all links succeeded"

            # NMSE is finite
            nmse = h5["evaluation/nmse_db"][()]
            assert np.all(np.isfinite(nmse))

    def test_4x4_no_siso_broadcast(self, tmp_path):
        """Verify CFR estimates are NOT a single SISO estimate broadcast to all
        antenna pairs."""
        try:
            path = _run_4x4_mimo_pipeline(tmp_path, perfect_csi=True)
        except ImportError:
            pytest.skip("NR PUSCH receiver not available")
        except Exception as exc:
            pytest.fail(f"4x4 MIMO pipeline failed: {exc}")

        with h5py.File(path, "r") as h5:
            cfr_est = h5["observation/cfr_est"][()]

        # Extract channel for different antenna pairs
        # cfr_est: [snap, tx, rx, rx_ant, tx_ant, subcarrier]
        pair_00 = cfr_est[0, 0, 0, 0, 0, :]  # (rx_ant=0, tx_ant=0)
        pair_11 = cfr_est[0, 0, 0, 1, 1, :]  # (rx_ant=1, tx_ant=1)
        pair_33 = cfr_est[0, 0, 0, 3, 3, :]  # (rx_ant=3, tx_ant=3)

        # Different antenna pairs should have different estimates
        assert not np.allclose(pair_00, pair_11, rtol=1e-3, atol=1e-6), (
            "Antenna pair (0,0) and (1,1) have identical CFR estimates — "
            "likely a SISO broadcast"
        )
        assert not np.allclose(pair_11, pair_33, rtol=1e-3, atol=1e-6), (
            "Antenna pair (1,1) and (3,3) have identical CFR estimates — "
            "likely a SISO broadcast"
        )

    def test_4x4_cfr_est_different_from_truth(self, tmp_path):
        """With perfect_csi=True, cfr_est should closely match truth (NMSE < -20 dB)."""
        try:
            path = _run_4x4_mimo_pipeline(tmp_path, perfect_csi=True, snr_db=60.0)
        except ImportError:
            pytest.skip("NR PUSCH receiver not available")
        except Exception as exc:
            pytest.fail(f"4x4 MIMO pipeline failed: {exc}")

        with h5py.File(path, "r") as h5:
            nmse = h5["evaluation/nmse_db"][()]
        # At very high SNR with perfect CSI, NMSE should be excellent
        # (limited only by numerical precision)
        assert np.all(nmse < -20), f"NMSE too high at 60 dB: {nmse}"

    def test_4x4_ber_bler_finite(self, tmp_path):
        """BER and BLER are finite in 4x4 pipeline."""
        try:
            path = _run_4x4_mimo_pipeline(tmp_path, perfect_csi=True)
        except ImportError:
            pytest.skip("NR PUSCH receiver not available")
        except Exception as exc:
            pytest.fail(f"4x4 MIMO pipeline failed: {exc}")

        with h5py.File(path, "r") as h5:
            ber = float(h5["evaluation/ber"][()])
            bler = float(h5["evaluation/bler"][()])
        assert np.isfinite(ber)
        assert np.isfinite(bler)
        assert 0.0 <= ber <= 1.0
        assert 0.0 <= bler <= 1.0

    def test_4x4_receiver_mimo_detector_present(self, tmp_path):
        """Receiver group contains mimo_detector and it matches pipeline."""
        try:
            path = _run_4x4_mimo_pipeline(tmp_path, perfect_csi=True)
        except ImportError:
            pytest.skip("NR PUSCH receiver not available")
        except Exception as exc:
            pytest.fail(f"4x4 MIMO pipeline failed: {exc}")

        with h5py.File(path, "r") as h5:
            assert "receiver/mimo_detector" in h5
            assert "receiver/receiver_type" in h5
            rx_type = h5["receiver/receiver_type"][()]
            if isinstance(rx_type, bytes):
                rx_type = rx_type.decode()
            assert rx_type == "pusch_receiver"

    def test_4x4_with_estimated_csi_smoke(self, tmp_path):
        """Smoke test for estimated CSI (non-perfect)."""
        try:
            path = _run_4x4_mimo_pipeline(
                tmp_path, num_layers=4, num_antenna_ports=4,
                perfect_csi=False, snr_db=40.0,
            )
        except ImportError:
            pytest.skip("NR PUSCH receiver not available")
        except Exception as exc:
            pytest.fail(f"Estimated CSI pipeline failed: {exc}")

        with h5py.File(path, "r") as h5:
            cfr_est = h5["observation/cfr_est"]
            truth_cfr = h5["channel/truth/cfr"]
            assert cfr_est.shape[1:] == truth_cfr.shape
            est_type = h5["receiver/estimator_type"][()]
            if isinstance(est_type, bytes):
                est_type = est_type.decode()
            assert est_type == "pusch_ls"

    def test_estimated_csi_requires_equal_layers_and_ports(self, tmp_path):
        """estimated CSI with num_layers<num_antenna_ports must fail clearly."""
        import pytest as pt

        try:
            _run_4x4_mimo_pipeline(
                tmp_path, num_layers=1, num_antenna_ports=4,
                perfect_csi=False,
            )
        except ImportError:
            pt.skip("NR PUSCH receiver not available")
        except NotImplementedError:
            return  # expected: cannot zero-pad to pretend full MIMO
        except Exception as exc:
            pt.fail(f"Expected NotImplementedError, got {type(exc).__name__}: {exc}")
        else:
            pt.fail(
                "Expected NotImplementedError for "
                "num_layers=1, num_antenna_ports=4 with estimated CSI"
            )

    def test_cir_dataset_ofdm_h_closes_csi_loop(self, tmp_path):
        """cir_dataset_ofdm backend's returned h must be used for cfr_est."""
        try:
            path = _run_4x4_mimo_pipeline(
                tmp_path, num_layers=4, num_antenna_ports=4,
                perfect_csi=True, snr_db=40.0,
                channel_backend="cir_dataset_ofdm",
            )
        except ImportError:
            pytest.skip("NR PUSCH receiver not available")
        except Exception as exc:
            pytest.fail(f"cir_dataset_ofdm pipeline failed: {exc}")

        with h5py.File(path, "r") as h5:
            cfr_est = h5["observation/cfr_est"][()]
            truth = h5["channel/truth/cfr"][()]
            assert cfr_est.shape[1:] == truth.shape
            assert np.all(np.isfinite(cfr_est))
            # At high SNR with perfect CSI, NMSE should be excellent
            nmse = h5["evaluation/nmse_db"][()]
            assert np.all(nmse < -20), f"NMSE too high: {nmse}"
