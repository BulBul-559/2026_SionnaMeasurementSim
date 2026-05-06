import numpy as np

from sionna_measurement_sim.phy.observation_pipeline import (
    AWGNObservationConfig,
    run_awgn_ls_observation,
)


def test_awgn_observation_shapes_and_nmse_debug_threshold():
    h_true = np.ones((1, 1, 1, 1, 8), dtype=np.complex64)

    bundle = run_awgn_ls_observation(
        h_true,
        AWGNObservationConfig(
            snr_db=40.0,
            random_seed=7,
            sample_rate_hz=20e6,
            fft_size=8,
        ),
    )

    assert bundle.waveform.standard == "custom_ofdm"
    assert bundle.waveform.pilot_indices.shape == (8,)
    assert bundle.observation.cfr_est.shape == (1, 1, 1, 1, 1, 8)
    assert bundle.observation.valid_mask.shape == (1, 1, 1)
    assert float(np.median(bundle.evaluation.nmse_db)) < -20.0
    assert bundle.receiver.estimator_type == "ls"
