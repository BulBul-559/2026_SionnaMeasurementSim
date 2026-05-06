import numpy as np

from sionna_measurement_sim.phy.observation_pipeline import (
    AWGNObservationConfig,
    run_awgn_ls_observation,
)


def test_awgn_nmse_improves_with_snr():
    h_true = np.ones((1, 1, 1, 1, 64), dtype=np.complex64)

    low = run_awgn_ls_observation(
        h_true,
        AWGNObservationConfig(
            snr_db=5.0,
            random_seed=3,
            sample_rate_hz=20e6,
            fft_size=64,
        ),
    )
    high = run_awgn_ls_observation(
        h_true,
        AWGNObservationConfig(
            snr_db=40.0,
            random_seed=3,
            sample_rate_hz=20e6,
            fft_size=64,
        ),
    )

    assert float(np.median(high.evaluation.nmse_db)) < float(np.median(low.evaluation.nmse_db))
    assert float(np.median(high.evaluation.nmse_db)) < -20.0
