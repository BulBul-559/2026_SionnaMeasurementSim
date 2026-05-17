"""Phase 4b statistical tests for NR PUSCH link metrics.

Verifies channel-estimation quality improves with higher SNR, HDF5 field
completeness, and CFR shape consistency for the NR PUSCH observation pipeline.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

from sionna_measurement_sim.rt.truth_pipeline import (
    RTTruthRunConfig,
    run_rt_truth_pipeline,
)


def _run_nr_pusch(snr_db: float, tmp_path: Path) -> Path:
    """Run the NR PUSCH pipeline at the given SNR and return the HDF5 path."""
    config = RTTruthRunConfig(
        label_file=Path("tests/fixtures/scenes/test/test5.json"),
        scene_file=Path("tests/fixtures/scenes/test/scene.xml"),
        output_dir=tmp_path / f"output_{snr_db}",
        num_subcarriers=48,
        seed=42,
        max_tx=2,
        max_rx=2,
        tx_num_rows=1,
        tx_num_cols=1,
        rx_num_rows=1,
        rx_num_cols=1,
        max_depth=0,
        specular_reflection=False,
        observation_snr_db=snr_db,
        phy_standard="nr_pusch",
        num_prb=4,
        num_layers=1,
        num_antenna_ports=1,
        mcs_index=14,
        mcs_table=1,
    )
    return run_rt_truth_pipeline(config)


class TestNRPUSCHLinkMetrics:
    """Statistical tests for the NR PUSCH observation pipeline."""

    def test_channel_estimation_improves_with_snr(self, tmp_path):
        """Run NR PUSCH at two SNR levels and verify channel metrics improve."""
        try:
            path_low = _run_nr_pusch(10, tmp_path)
            path_high = _run_nr_pusch(30, tmp_path)
        except ImportError:
            pytest.skip("NR PUSCH receiver not available on this machine")
        except Exception:
            pytest.fail("NR PUSCH receiver failed")

        with h5py.File(path_low, "r") as h5:
            nmse_low = float(np.nanmean(h5["evaluation/nmse_db"][()]))
            noise_low = float(np.nanmean(h5["waveform/noise_variance"][()]))
        with h5py.File(path_high, "r") as h5:
            nmse_high = float(np.nanmean(h5["evaluation/nmse_db"][()]))
            noise_high = float(np.nanmean(h5["waveform/noise_variance"][()]))

        assert noise_high < noise_low, (
            f"Noise variance at 30dB ({noise_high}) should be < 10dB ({noise_low})"
        )
        assert nmse_high <= nmse_low, (
            f"NMSE at 30dB ({nmse_high}) should be <= NMSE at 10dB ({nmse_low})"
        )

    def test_nr_pusch_hdf5_fields(self, tmp_path):
        """NR PUSCH output has all required fields with finite values."""
        try:
            path = _run_nr_pusch(30, tmp_path)
        except ImportError:
            pytest.skip("NR PUSCH receiver not available on this machine")
        except Exception:
            pytest.fail("NR PUSCH receiver failed")

        with h5py.File(path, "r") as h5:
            # Waveform group
            std_val = h5["waveform/standard"][()]
            if isinstance(std_val, bytes):
                std_val = std_val.decode("utf-8")
            assert std_val == "nr_pusch", (
                f"Expected 'nr_pusch', got {std_val!r}"
            )

            # Receiver group
            rx_val = h5["receiver/receiver_type"][()]
            if isinstance(rx_val, bytes):
                rx_val = rx_val.decode("utf-8")
            assert rx_val == "pusch_receiver", (
                f"Expected 'pusch_receiver', got {rx_val!r}"
            )

            # Evaluation metrics are finite
            ber = float(h5["evaluation/ber"][()])
            bler = float(h5["evaluation/bler"][()])
            assert np.isfinite(ber), f"BER ({ber}) is not finite"
            assert np.isfinite(bler), f"BLER ({bler}) is not finite"

            # Link group exists with all 5 fields
            link_group = h5["link"]
            expected_fields = {
                "duplex_mode",
                "phy_link_direction",
                "rt_trace_direction",
                "reciprocity_mode",
                "reciprocity_applied",
            }
            actual_fields = set(link_group.keys())
            assert actual_fields == expected_fields, (
                f"Link group fields mismatch. "
                f"Expected {expected_fields}, got {actual_fields}"
            )

    def test_cfr_shape_consistency(self, tmp_path):
        """Observation CFR shape matches truth CFR shape."""
        try:
            path = _run_nr_pusch(30, tmp_path)
        except ImportError:
            pytest.skip("NR PUSCH receiver not available on this machine")
        except Exception:
            pytest.fail("NR PUSCH receiver failed")

        with h5py.File(path, "r") as h5:
            cfr_est = h5["observation/cfr_est"]
            truth_cfr = h5["channel/truth/cfr"]
            assert cfr_est.shape[1:] == truth_cfr.shape, (
                f"Observation CFR shape {cfr_est.shape} (dropping snapshot dim = "
                f"{cfr_est.shape[1:]}) should match truth CFR shape {truth_cfr.shape}"
            )
