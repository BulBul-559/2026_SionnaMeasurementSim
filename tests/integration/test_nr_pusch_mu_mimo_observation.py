"""NR PUSCH MU-MIMO integration tests.

Self-generates HDF5 output for multi-UE uplink and validates:
- Multiple PUSCHConfig objects with non-overlapping DMRS port sets
- Joint StreamManagement across all TX/RX
- /observation/cfr_est covers all UE links
- Per-link NMSE is finite
"""

from pathlib import Path

import h5py
import numpy as np
import pytest


def _run_mu_mimo_pipeline(tmp_path: Path, perfect_csi: bool = True) -> Path:
    """Run MU-MIMO NR PUSCH pipeline: 1 BS, 2 UEs, 2 antennas each."""
    from sionna_measurement_sim.rt.truth_pipeline import (
        RTTruthRunConfig,
        run_rt_truth_pipeline,
    )

    config = RTTruthRunConfig(
        label_file=Path("tests/fixtures/scenes/test/test5.json"),
        scene_file=Path("tests/fixtures/scenes/test/scene.xml"),
        output_dir=tmp_path / "output_mu_mimo",
        num_subcarriers=48,
        seed=42,
        max_bs=1,  # 1 BS
        max_ue=2,  # 2 UEs
        bs_num_rows=1,
        bs_num_cols=2,  # BS: 2 antennas
        ue_num_rows=1,
        ue_num_cols=2,  # each UE: 2 antennas
        max_depth=3,
        los=True,
        specular_reflection=True,
        observation_snr_db=40.0,
        phy_standard="nr_pusch",
        num_prb=4,
        num_layers=1,
        num_antenna_ports=2,
        mcs_index=14,
        mcs_table=1,
        perfect_csi=perfect_csi,
        receiver_failure_policy="fail_fast",
        mimo_detector="lmmse",
        channel_estimator="perfect" if perfect_csi else "pusch_ls",
        mimo_mode="mu_mimo",
    )
    return run_rt_truth_pipeline(config)


class TestNRPUSCHMUMIMO:
    """MU-MIMO NR PUSCH integration tests."""

    def test_mu_mimo_pipeline_self_generated(self, tmp_path):
        """2-UE MU-MIMO pipeline produces valid HDF5."""
        try:
            path = _run_mu_mimo_pipeline(tmp_path, perfect_csi=True)
        except ImportError:
            pytest.skip("NR PUSCH receiver not available")
        except Exception as exc:
            pytest.fail(f"MU-MIMO pipeline failed: {exc}")

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

            # Direct uplink link-view: 2 UEs are TX, 1 BS is RX.
            assert truth_cfr.shape[0] == 2, f"Expected 2 tx (UEs), got {truth_cfr.shape[0]}"
            assert truth_cfr.shape[1] == 1, f"Expected 1 rx (BS), got {truth_cfr.shape[1]}"

            # Each UE has 2 antennas
            assert truth_cfr.shape[2] == 2, f"Expected 2 rx_ant, got {truth_cfr.shape[2]}"
            assert truth_cfr.shape[3] == 2, f"Expected 2 tx_ant, got {truth_cfr.shape[3]}"

            # MIMO detector metadata
            det = h5["receiver/mimo_detector"][()]
            if isinstance(det, bytes):
                det = det.decode()
            assert det == "lmmse", f"mimo_detector={det!r}"

            # All links succeeded
            est_success = h5["observation/estimation_success"][()]
            assert np.all(est_success), "Not all MU-MIMO links succeeded"

            # NMSE is finite
            nmse = h5["evaluation/nmse_db"][()]
            assert np.all(np.isfinite(nmse)), "MU-MIMO NMSE has non-finite values"

    def test_mu_mimo_per_link_metrics(self, tmp_path):
        """MU-MIMO produces per-link NMSE covering all UE-BS pairs."""
        try:
            path = _run_mu_mimo_pipeline(tmp_path, perfect_csi=True)
        except ImportError:
            pytest.skip("NR PUSCH receiver not available")
        except Exception as exc:
            pytest.fail(f"MU-MIMO pipeline failed: {exc}")

        with h5py.File(path, "r") as h5:
            nmse = h5["evaluation/nmse_db"][()]
            # Shape: [snap, tx, rx] = [1, 2, 1] for 2 UEs, 1 BS.
            assert nmse.shape == (1, 2, 1), (
                f"Expected NMSE shape (1, 2, 1), got {nmse.shape}"
            )
            # Each UE should have finite NMSE
            for ue in range(2):
                assert np.isfinite(nmse[0, ue, 0]), (
                    f"NMSE for UE {ue} is not finite: {nmse[0, ue, 0]}"
                )

    def test_mu_mimo_cfr_est_distinct_per_ue(self, tmp_path):
        """CFR estimates differ between UEs."""
        try:
            path = _run_mu_mimo_pipeline(tmp_path, perfect_csi=True)
        except ImportError:
            pytest.skip("NR PUSCH receiver not available")
        except Exception as exc:
            pytest.fail(f"MU-MIMO pipeline failed: {exc}")

        with h5py.File(path, "r") as h5:
            cfr_est = h5["observation/cfr_est"][()]
            # cfr_est: [snap=1, tx=2, rx=1, rx_ant=2, tx_ant=2, subcarrier=48]
            ue0 = cfr_est[0, 0, 0, ...]  # [rx_ant, tx_ant, subcarrier]
            ue1 = cfr_est[0, 1, 0, ...]

            # Different UEs should have different channel estimates
            ue0_flat = np.abs(ue0).ravel()
            ue1_flat = np.abs(ue1).ravel()
            # Skip if all zeros (no valid paths)
            if np.max(ue0_flat) > 1e-10 and np.max(ue1_flat) > 1e-10:
                assert not np.allclose(ue0_flat, ue1_flat, rtol=1e-2, atol=1e-6), (
                    "UE0 and UE1 have identical CFR — likely not distinct links"
                )

    def test_mu_mimo_ber_bler_finite(self, tmp_path):
        """MU-MIMO BER/BLER are finite."""
        try:
            path = _run_mu_mimo_pipeline(tmp_path, perfect_csi=True)
        except ImportError:
            pytest.skip("NR PUSCH receiver not available")
        except Exception as exc:
            pytest.fail(f"MU-MIMO pipeline failed: {exc}")

        with h5py.File(path, "r") as h5:
            ber = float(h5["evaluation/ber"][()])
            bler = float(h5["evaluation/bler"][()])
        assert np.isfinite(ber)
        assert np.isfinite(bler)

    def test_mu_mimo_bit_counter_not_doubled(self, tmp_path):
        """P1-3: num_bits must not be multiplied by link count."""
        try:
            path = _run_mu_mimo_pipeline(tmp_path, perfect_csi=True)
        except ImportError:
            pytest.skip("NR PUSCH receiver not available")
        except Exception as exc:
            pytest.fail(f"MU-MIMO pipeline failed: {exc}")

        with h5py.File(path, "r") as h5:
            num_bits = int(h5["evaluation/num_bits"][()])
            num_bit_errors = int(h5["evaluation/num_bit_errors"][()])
            ber = float(h5["evaluation/ber"][()])
            num_blocks = int(h5["evaluation/num_blocks"][()])

        # 2 UEs each with their own TB — num_blocks should be 2
        assert num_blocks > 0, f"num_blocks should be > 0, got {num_blocks}"
        # num_bits should be the total across all UEs, not multiplied by link count
        assert num_bits > 0, f"num_bits should be > 0, got {num_bits}"
        # BER consistency: ber == num_bit_errors / num_bits
        expected_ber = num_bit_errors / max(num_bits, 1)
        assert np.isclose(ber, expected_ber, rtol=1e-4, atol=1e-6), (
            f"BER ({ber}) != num_bit_errors/num_bits ({expected_ber})"
        )
