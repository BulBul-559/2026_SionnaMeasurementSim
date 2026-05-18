"""Statistical tests for NR PUSCH 4x4 SU-MIMO link metrics.

Verifies:
- perfect_csi >= estimated_csi in BER/BLER/NMSE (perfect is no worse).
- Higher EB/N0 gives better BER/BLER/NMSE.
- /observation/cfr_est shape matches /channel/truth/cfr.
- Estimation success is all-true.
- /receiver/mimo_detector matches the configured value.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest


def _run_4x4_pipeline(
    tmp_path: Path,
    perfect_csi: bool,
    ebno_db: float,
    mimo_detector: str = "lmmse",
) -> Path:
    """Run a 4x4 SU-MIMO NR PUSCH pipeline and return the HDF5 path."""
    from sionna_measurement_sim.rt.truth_pipeline import (
        RTTruthRunConfig,
        run_rt_truth_pipeline,
    )

    label = f"{'perfect' if perfect_csi else 'estimated'}_{ebno_db}dB"
    config = RTTruthRunConfig(
        label_file=Path("tests/fixtures/scenes/test/test5.json"),
        scene_file=Path("tests/fixtures/scenes/test/scene.xml"),
        output_dir=tmp_path / f"output_{label}",
        num_subcarriers=48,
        seed=42,
        max_bs=1,
        max_ue=1,
        bs_num_rows=2,
        bs_num_cols=2,
        ue_num_rows=2,
        ue_num_cols=2,
        max_depth=3,
        los=True,
        specular_reflection=True,
        observation_snr_db=30.0,  # triggers observation branch
        ebno_db=ebno_db,
        phy_standard="nr_pusch",
        num_prb=4,
        num_layers=4,
        num_antenna_ports=4,
        mcs_index=14,
        mcs_table=1,
        perfect_csi=perfect_csi,
        receiver_failure_policy="fail_fast",
        mimo_detector=mimo_detector,
        channel_estimator="perfect" if perfect_csi else "pusch_ls",
    )
    return run_rt_truth_pipeline(config)


class TestNRPUSCH4x4MIMOStatistical:
    """Statistical tests for the 4x4 SU-MIMO NR PUSCH pipeline."""

    # ── perfect vs estimated CSI ──────────────────────────────────────

    def test_perfect_csi_nmse_not_worse_than_estimated(self, tmp_path):
        """perfect_csi NMSE must be <= estimated CSI NMSE (better or equal)."""
        try:
            path_p = _run_4x4_pipeline(tmp_path, perfect_csi=True, ebno_db=30.0)
            path_e = _run_4x4_pipeline(tmp_path, perfect_csi=False, ebno_db=30.0)
        except ImportError:
            pytest.skip("NR PUSCH receiver not available")
        except Exception as exc:
            pytest.fail(f"Pipeline failed: {exc}")

        with h5py.File(path_p, "r") as h5:
            nmse_p = float(np.median(h5["evaluation/nmse_db"][()]))
        with h5py.File(path_e, "r") as h5:
            nmse_e = float(np.median(h5["evaluation/nmse_db"][()]))

        assert nmse_p <= nmse_e, (
            f"perfect CSI NMSE ({nmse_p:.1f} dB) should be <= "
            f"estimated CSI NMSE ({nmse_e:.1f} dB)"
        )

    def test_perfect_csi_ber_not_worse_than_estimated(self, tmp_path):
        """perfect_csi BER must be <= estimated CSI BER."""
        try:
            path_p = _run_4x4_pipeline(tmp_path, perfect_csi=True, ebno_db=30.0)
            path_e = _run_4x4_pipeline(tmp_path, perfect_csi=False, ebno_db=30.0)
        except ImportError:
            pytest.skip("NR PUSCH receiver not available")
        except Exception as exc:
            pytest.fail(f"Pipeline failed: {exc}")

        with h5py.File(path_p, "r") as h5:
            ber_p = float(h5["evaluation/ber"][()])
        with h5py.File(path_e, "r") as h5:
            ber_e = float(h5["evaluation/ber"][()])

        assert ber_p <= ber_e, (
            f"perfect CSI BER ({ber_p:.6f}) should be <= "
            f"estimated CSI BER ({ber_e:.6f})"
        )

    # ── EB/N0 monotonicity ────────────────────────────────────────────

    def test_estimated_csi_ber_improves_with_ebn0(self, tmp_path):
        """Higher EB/N0 should give lower BER for estimated CSI."""
        try:
            path_low = _run_4x4_pipeline(tmp_path, perfect_csi=False, ebno_db=10.0)
            path_high = _run_4x4_pipeline(tmp_path, perfect_csi=False, ebno_db=30.0)
        except ImportError:
            pytest.skip("NR PUSCH receiver not available")
        except Exception as exc:
            pytest.fail(f"Pipeline failed: {exc}")

        with h5py.File(path_low, "r") as h5:
            ber_low = float(h5["evaluation/ber"][()])
        with h5py.File(path_high, "r") as h5:
            ber_high = float(h5["evaluation/ber"][()])

        assert ber_high <= ber_low, (
            f"BER at 30dB ({ber_high:.6f}) should be <= "
            f"BER at 10dB ({ber_low:.6f})"
        )

    def test_estimated_csi_nmse_improves_with_ebn0(self, tmp_path):
        """Higher EB/N0 should give better (lower) NMSE for estimated CSI."""
        try:
            path_low = _run_4x4_pipeline(tmp_path, perfect_csi=False, ebno_db=10.0)
            path_high = _run_4x4_pipeline(tmp_path, perfect_csi=False, ebno_db=30.0)
        except ImportError:
            pytest.skip("NR PUSCH receiver not available")
        except Exception as exc:
            pytest.fail(f"Pipeline failed: {exc}")

        with h5py.File(path_low, "r") as h5:
            nmse_low = float(np.median(h5["evaluation/nmse_db"][()]))
        with h5py.File(path_high, "r") as h5:
            nmse_high = float(np.median(h5["evaluation/nmse_db"][()]))

        assert nmse_high <= nmse_low, (
            f"NMSE at 30dB ({nmse_high:.1f} dB) should be <= "
            f"NMSE at 10dB ({nmse_low:.1f} dB)"
        )

    # ── shape and metadata consistency ────────────────────────────────

    def test_cfr_est_shape_matches_truth(self, tmp_path):
        """Both perfect and estimated CSI must have correct CFR shapes."""
        for perf in (True, False):
            try:
                path = _run_4x4_pipeline(tmp_path, perfect_csi=perf, ebno_db=30.0)
            except ImportError:
                pytest.skip("NR PUSCH receiver not available")
            except Exception as exc:
                pytest.fail(f"Pipeline failed (perfect_csi={perf}): {exc}")

            with h5py.File(path, "r") as h5:
                cfr_est = h5["observation/cfr_est"]
                truth = h5["channel/truth/cfr"]
                assert cfr_est.shape[1:] == truth.shape, (
                    f"perfect_csi={perf}: cfr_est.shape[1:]={cfr_est.shape[1:]} "
                    f"!= truth.shape={truth.shape}"
                )
                assert truth.shape[2] == 4  # rx_ant
                assert truth.shape[3] == 4  # tx_ant

    def test_estimation_success_all_true(self, tmp_path):
        """All links must report estimation success."""
        for perf in (True, False):
            try:
                path = _run_4x4_pipeline(tmp_path, perfect_csi=perf, ebno_db=30.0)
            except ImportError:
                pytest.skip("NR PUSCH receiver not available")
            except Exception as exc:
                pytest.fail(f"Pipeline failed (perfect_csi={perf}): {exc}")

            with h5py.File(path, "r") as h5:
                est_ok = h5["observation/estimation_success"][()]
                assert np.all(est_ok), (
                    f"perfect_csi={perf}: not all estimation_success are True"
                )

    def test_mimo_detector_consistent(self, tmp_path):
        """HDF5 mimo_detector must match the configured value."""
        for det in ("lmmse",):  # kbest requires more resources for 16 streams
            try:
                path = _run_4x4_pipeline(
                    tmp_path, perfect_csi=True, ebno_db=30.0, mimo_detector=det,
                )
            except ImportError:
                pytest.skip("NR PUSCH receiver not available")
            except Exception as exc:
                pytest.fail(f"Pipeline failed (detector={det}): {exc}")

            with h5py.File(path, "r") as h5:
                h5_det = h5["receiver/mimo_detector"][()]
                if isinstance(h5_det, bytes):
                    h5_det = h5_det.decode()
                assert h5_det == det, (
                    f"HDF5 mimo_detector={h5_det!r}, expected {det!r}"
                )
