import torch

from sionna_measurement_sim.phy.common_link import ObservationImpairmentChain
from sionna_measurement_sim.phy.impairments import ImpairmentConfig


def test_common_chain_noop_without_awgn_keeps_grid_and_zero_metadata():
    rx_clean = torch.ones((2, 1, 3, 1, 2, 16), dtype=torch.complex64)
    chain = ObservationImpairmentChain(
        fft_size=16,
        sample_rate_hz=30e3 * 16,
        random_seed=7,
        awgn_enabled=False,
    )

    result = chain.apply(rx_clean, snr_db=30.0)

    assert torch.allclose(result.rx_grid, rx_clean, atol=1e-6)
    assert result.noise_variance.shape == (2, 1, 3)
    assert torch.all(result.noise_variance == 0.0)
    assert torch.all(result.impairment_sample.cfo_hz == 0.0)
    assert torch.all(result.impairment_sample.sfo_ppm == 0.0)
    assert torch.all(result.impairment_sample.timing_offset_samples == 0.0)
    assert torch.all(result.impairment_sample.phase_offset_rad == 0.0)
    assert torch.all(result.impairment_sample.agc_gain_db == 0.0)
    assert torch.all(~result.impairment_sample.clipping_flag)


def test_common_chain_awgn_noise_shape_and_snr_monotonicity():
    rx_clean = torch.ones((1, 1, 1, 2, 1, 64), dtype=torch.complex64)
    low = ObservationImpairmentChain(
        fft_size=64,
        sample_rate_hz=20e6,
        random_seed=11,
    ).apply(rx_clean, snr_db=0.0)
    high = ObservationImpairmentChain(
        fft_size=64,
        sample_rate_hz=20e6,
        random_seed=11,
    ).apply(rx_clean, snr_db=40.0)

    low_nmse = torch.mean(torch.abs(low.rx_grid - rx_clean) ** 2)
    high_nmse = torch.mean(torch.abs(high.rx_grid - rx_clean) ** 2)

    assert low.noise_variance.shape == (1, 1, 1)
    assert high.noise_variance.shape == (1, 1, 1)
    assert torch.all(low.noise_variance > high.noise_variance)
    assert high_nmse < low_nmse


def test_common_chain_impairment_metadata_nonzero_and_reproducible():
    rx_clean = torch.randn((1, 1, 1, 1, 1, 64), dtype=torch.complex64)
    config = ImpairmentConfig(
        random_seed=3,
        cfo_hz=250.0,
        sfo_ppm=4.0,
        phase_offset_rad=0.25,
        timing_offset_samples=1.5,
        agc_gain_db=6.0,
        clipping_threshold=0.2,
    )
    chain1 = ObservationImpairmentChain(
        fft_size=64,
        sample_rate_hz=20e6,
        random_seed=13,
        impairment_config=config,
    )
    chain2 = ObservationImpairmentChain(
        fft_size=64,
        sample_rate_hz=20e6,
        random_seed=13,
        impairment_config=config,
    )

    result1 = chain1.apply(rx_clean, snr_db=30.0)
    result2 = chain2.apply(rx_clean, snr_db=30.0)
    sample = result1.impairment_sample

    assert torch.all(sample.cfo_hz == 250.0)
    assert torch.all(sample.sfo_ppm == 4.0)
    assert torch.all(sample.phase_offset_rad == 0.25)
    assert torch.all(sample.timing_offset_samples == 1.5)
    assert torch.all(sample.agc_gain_db == 6.0)
    assert torch.all(sample.clipping_flag)
    assert torch.allclose(result1.rx_grid, result2.rx_grid)
