import numpy as np
import torch

from sionna_measurement_sim.phy.impairments import (
    ImpairmentConfig,
    apply_base_impairments,
)
from sionna_measurement_sim.phy.observation_pipeline import (
    AWGNObservationConfig,
    run_awgn_ls_observation,
)


class TestCFOvsPhaseDrift:
    def test_cfo_increases_distortion(self):
        """Higher CFO causes larger NMSE between impaired and clean CFR."""
        rng = np.random.default_rng(42)
        h_np = (rng.normal(size=(1, 1, 1, 1, 64)).astype(np.float32)
                + 1j * rng.normal(size=(1, 1, 1, 1, 64)).astype(np.float32))
        h_tensor = torch.as_tensor(h_np, dtype=torch.complex64)

        # No CFO vs low CFO vs high CFO
        results = {}
        for cfo_val in [None, 100.0, 5000.0]:
            impaired, _ = apply_base_impairments(
                h_tensor, 64, 20e6,
                ImpairmentConfig(random_seed=3, cfo_hz=cfo_val),
            )
            # NMSE between clean (no impairment) and impaired
            noise = impaired - h_tensor
            nmse = (torch.sum(torch.abs(noise)**2) /
                    torch.clamp(torch.sum(torch.abs(h_tensor)**2), min=1e-30))
            results[str(cfo_val)] = float(nmse)

        # no-CFO should have near-zero NMSE (minor rounding from FFT/IFFT)
        assert results["None"] < 1e-5, f"No-CFO NMSE should be ~0, got {results['None']}"
        # Higher CFO should cause larger NMSE
        assert results["5000.0"] > results["100.0"], (
            f"High CFO NMSE ({results['5000.0']}) "
            f"should exceed low CFO NMSE ({results['100.0']})"
        )


class TestClippingStatistical:
    def test_lower_threshold_increases_clipping_ratio(self):
        h_true = np.ones((1, 1, 1, 1, 16), dtype=np.complex64)

        high_thresh = run_awgn_ls_observation(
            h_true,
            AWGNObservationConfig(
                snr_db=40.0,
                random_seed=3,
                sample_rate_hz=20e6,
                fft_size=16,
                impairment=ImpairmentConfig(random_seed=7, clipping_threshold=5.0),
            ),
        )
        low_thresh = run_awgn_ls_observation(
            h_true,
            AWGNObservationConfig(
                snr_db=40.0,
                random_seed=3,
                sample_rate_hz=20e6,
                fft_size=16,
                impairment=ImpairmentConfig(random_seed=7, clipping_threshold=0.5),
            ),
        )
        clip_high = float(np.mean(high_thresh.observation.clipping_flag))
        clip_low = float(np.mean(low_thresh.observation.clipping_flag))
        assert clip_low >= clip_high


class TestImpairmentReproducibility:
    def test_fixed_seed_reproducible_pipeline(self):
        h_true = np.ones((1, 1, 1, 1, 8), dtype=np.complex64)
        config = AWGNObservationConfig(
            snr_db=30.0,
            random_seed=42,
            sample_rate_hz=20e6,
            fft_size=8,
            impairment=ImpairmentConfig(random_seed=42, cfo_hz=100.0, sfo_ppm=5.0),
        )
        b1 = run_awgn_ls_observation(h_true, config)
        b2 = run_awgn_ls_observation(h_true, config)
        assert np.allclose(b1.observation.cfo_hz, b2.observation.cfo_hz)
        assert np.allclose(b1.observation.sfo_ppm, b2.observation.sfo_ppm)
        assert np.allclose(b1.observation.cfr_est, b2.observation.cfr_est)
        assert np.allclose(b1.evaluation.nmse_db, b2.evaluation.nmse_db)


class TestImpairmentHDF5Fields:
    def test_impairment_fields_nonzero_when_configured(self):
        h_true = np.ones((1, 1, 1, 1, 8), dtype=np.complex64)
        config = AWGNObservationConfig(
            snr_db=30.0,
            random_seed=1,
            sample_rate_hz=20e6,
            fft_size=8,
            impairment=ImpairmentConfig(
                random_seed=3,
                cfo_hz=100.0,
                sfo_ppm=5.0,
                phase_offset_rad=1.0,
                timing_offset_samples=2.0,
                clipping_threshold=2.0,
            ),
        )
        bundle = run_awgn_ls_observation(h_true, config)

        assert np.all(bundle.observation.cfo_hz == 100.0)
        assert np.all(bundle.observation.sfo_ppm == 5.0)
        assert np.all(bundle.observation.phase_offset_rad == 1.0)
        assert np.all(bundle.observation.timing_offset_samples == 2.0)
        assert bundle.impairments.model_version == "phase5_base_impairments_v1"
        assert "cfo_hz" in bundle.impairments.cfo_sfo_config
        assert "sfo_ppm" in bundle.impairments.cfo_sfo_config

    def test_impairment_fields_zero_when_not_configured(self):
        h_true = np.ones((1, 1, 1, 1, 8), dtype=np.complex64)
        config = AWGNObservationConfig(
            snr_db=40.0, random_seed=1, sample_rate_hz=20e6, fft_size=8,
        )
        bundle = run_awgn_ls_observation(h_true, config)
        assert np.all(bundle.observation.cfo_hz == 0.0)
        assert np.all(bundle.observation.sfo_ppm == 0.0)
        assert np.all(~bundle.observation.clipping_flag)
        assert bundle.impairments.model_version == "phase4_awgn_v1"


class TestFailureScenario:
    def test_severe_clipping_causes_estimation_error(self):
        h_true = np.ones((1, 1, 1, 1, 64), dtype=np.complex64)

        no_clip = run_awgn_ls_observation(
            h_true,
            AWGNObservationConfig(
                snr_db=40.0, random_seed=5, sample_rate_hz=20e6, fft_size=64,
                impairment=ImpairmentConfig(random_seed=1),
            ),
        )
        heavy_clip = run_awgn_ls_observation(
            h_true,
            AWGNObservationConfig(
                snr_db=40.0, random_seed=5, sample_rate_hz=20e6, fft_size=64,
                impairment=ImpairmentConfig(random_seed=1, clipping_threshold=0.1),
            ),
        )
        nmse_no_clip = float(np.median(no_clip.evaluation.nmse_db))
        nmse_heavy_clip = float(np.median(heavy_clip.evaluation.nmse_db))
        assert nmse_heavy_clip > nmse_no_clip or np.all(heavy_clip.observation.clipping_flag)
