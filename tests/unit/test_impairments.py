import torch

from sionna_measurement_sim.phy.impairments import (
    ImpairmentConfig,
    _apply_agc_adc,
    _apply_cfo_time_domain,
    _apply_sfo,
    _apply_timing_offset,
    apply_base_impairments,
)


class TestCFO:
    def test_cfo_zero_passthrough(self):
        td = torch.ones((1, 1, 1, 1, 1, 8), dtype=torch.complex64)
        cfo = torch.zeros((1, 1, 1))
        result = _apply_cfo_time_domain(td, cfo, 20e6)
        assert torch.allclose(result, td, atol=1e-6)

    def test_cfo_phase_accumulates_over_time(self):
        td = torch.ones((1, 1, 1, 1, 1, 64), dtype=torch.complex64)
        cfo = torch.full((1, 1, 1), 1000.0)  # 1 kHz offset
        result = _apply_cfo_time_domain(td, cfo, 20e6)
        phase_early = torch.angle(result[0, 0, 0, 0, 0, 0])
        phase_late = torch.angle(result[0, 0, 0, 0, 0, -1])
        assert phase_late != phase_early  # phase rotates over time

    def test_cfo_none_disabled(self):
        cfr = torch.ones((1, 1, 1, 1, 1, 8), dtype=torch.complex64)
        config = ImpairmentConfig(random_seed=1, cfo_hz=None)
        result, sample = apply_base_impairments(cfr, 8, 20e6, config)
        assert torch.all(sample.cfo_hz == 0.0)
        assert torch.allclose(result, cfr, atol=1e-5)

    def test_cfo_nominal_value_applied(self):
        # Non-constant CFR so CFO has a visible effect (DC-only CFR is invariant to CFO)
        rng = torch.Generator().manual_seed(42)
        cfr = torch.randn((1, 1, 1, 1, 1, 64), dtype=torch.complex64, generator=rng)
        config = ImpairmentConfig(random_seed=1, cfo_hz=500.0)
        result, sample = apply_base_impairments(cfr, 64, 20e6, config)
        assert torch.allclose(sample.cfo_hz, torch.tensor(500.0))
        assert not torch.allclose(result, cfr, atol=1e-5)


class TestSFO:
    def test_sfo_zero_passthrough(self):
        cfr = torch.ones((1, 1, 1, 1, 1, 8), dtype=torch.complex64)
        sfo = torch.zeros((1, 1, 1))
        result = _apply_sfo(cfr, sfo)
        assert torch.allclose(result, cfr)

    def test_sfo_phase_ramp_by_subcarrier(self):
        cfr = torch.ones((1, 1, 1, 1, 1, 16), dtype=torch.complex64)
        sfo = torch.full((1, 1, 1), 10.0)
        result = _apply_sfo(cfr, sfo)
        phases = torch.angle(result[0, 0, 0, 0, 0, :])
        diffs = torch.diff(phases)
        assert not torch.allclose(diffs, torch.zeros_like(diffs), atol=1e-7)

    def test_sfo_none_disabled(self):
        cfr = torch.ones((1, 1, 1, 1, 1, 8), dtype=torch.complex64)
        config = ImpairmentConfig(random_seed=1, sfo_ppm=None)
        result, sample = apply_base_impairments(cfr, 8, 20e6, config)
        assert torch.all(sample.sfo_ppm == 0.0)


class TestTimingOffset:
    def test_timing_offset_phase_ramp(self):
        cfr = torch.ones((1, 1, 1, 1, 1, 16), dtype=torch.complex64)
        timing = torch.full((1, 1, 1), 3.0)
        result = _apply_timing_offset(cfr, timing, 64)
        phases = torch.angle(result[0, 0, 0, 0, 0, :])
        assert not torch.allclose(phases, torch.zeros_like(phases), atol=1e-7)

    def test_timing_zero_passthrough(self):
        cfr = torch.ones((1, 1, 1, 1, 1, 8), dtype=torch.complex64)
        timing = torch.zeros((1, 1, 1))
        result = _apply_timing_offset(cfr, timing, 64)
        assert torch.allclose(result, cfr)


class TestAGCADC:
    def test_no_clipping_when_threshold_none(self):
        cfr = torch.ones((1, 1, 1, 1, 1, 8), dtype=torch.complex64)
        agc = torch.zeros((1, 1))
        result, clip_flag = _apply_agc_adc(cfr, agc, None)
        assert not torch.any(clip_flag)
        assert torch.allclose(result, cfr)

    def test_clipping_occurs_below_threshold(self):
        cfr = torch.full((1, 1, 1, 1, 1, 8), 10.0, dtype=torch.complex64)
        agc = torch.zeros((1, 1))
        result, clip_flag = _apply_agc_adc(cfr, agc, 5.0)
        assert torch.all(clip_flag)
        assert torch.all(torch.abs(result) <= 5.0 + 1e-5)

    def test_clipping_flag_false_above_threshold(self):
        cfr = torch.full((1, 1, 1, 1, 1, 8), 1.0, dtype=torch.complex64)
        agc = torch.zeros((1, 1))
        result, clip_flag = _apply_agc_adc(cfr, agc, 5.0)
        assert not torch.any(clip_flag)

    def test_agc_gain_scales_signal(self):
        cfr = torch.ones((1, 1, 1, 1, 1, 8), dtype=torch.complex64)
        agc = torch.full((1, 1), 6.0)  # 6 dB gain, 10^(6/20) ≈ 1.9953x linear
        result, _ = _apply_agc_adc(cfr, agc, None)
        expected = 10.0 ** (6.0 / 20.0)
        assert torch.allclose(torch.abs(result), torch.tensor(expected), atol=1e-4)

    def test_lower_threshold_more_clipping(self):
        cfr = torch.full((2, 1, 1, 1, 1, 16), 10.0, dtype=torch.complex64)
        agc = torch.zeros((2, 1))
        _, flag_high = _apply_agc_adc(cfr, agc, 5.0)
        _, flag_low = _apply_agc_adc(cfr, agc, 2.0)
        assert torch.sum(flag_high.int()) == torch.sum(flag_low.int())  # both all clipped
        # use lower clipping vs much lower
        _, flag_moderate = _apply_agc_adc(cfr, agc, 8.0)
        assert torch.sum(flag_moderate.int()) <= torch.sum(flag_high.int())


class TestReproducibility:
    def test_same_seed_same_impairments(self):
        cfr = torch.ones((1, 1, 1, 1, 1, 8), dtype=torch.complex64)
        config = ImpairmentConfig(
            random_seed=42, cfo_hz=100.0, sfo_ppm=5.0,
            phase_offset_rad=0.5, timing_offset_samples=2.0,
        )
        r1, s1 = apply_base_impairments(cfr, 8, 20e6, config)
        r2, s2 = apply_base_impairments(cfr, 8, 20e6, config)
        assert torch.allclose(r1, r2)
        assert torch.allclose(s1.cfo_hz, s2.cfo_hz)

    def test_different_seed_same_nominal(self):
        cfr = torch.ones((1, 1, 1, 1, 1, 8), dtype=torch.complex64)
        c1 = ImpairmentConfig(random_seed=1, cfo_hz=100.0)
        c2 = ImpairmentConfig(random_seed=2, cfo_hz=100.0)
        _, s1 = apply_base_impairments(cfr, 8, 20e6, c1)
        _, s2 = apply_base_impairments(cfr, 8, 20e6, c2)
        assert torch.allclose(s1.cfo_hz, s2.cfo_hz)  # same nominal value


class TestIntegratedPipeline:
    def test_all_impairments_none_passthrough(self):
        cfr = torch.ones((1, 1, 1, 1, 1, 8), dtype=torch.complex64)
        config = ImpairmentConfig(random_seed=1)
        result, sample = apply_base_impairments(cfr, 8, 20e6, config)
        assert torch.allclose(result, cfr, atol=1e-5)
        assert torch.all(~sample.clipping_flag)

    def test_impairment_result_shape(self):
        cfr = torch.ones((2, 1, 2, 1, 1, 16), dtype=torch.complex64)
        config = ImpairmentConfig(
            random_seed=7, cfo_hz=100.0, sfo_ppm=2.0,
            phase_offset_rad=0.3, timing_offset_samples=1.5,
        )
        result, sample = apply_base_impairments(cfr, 16, 20e6, config)
        assert result.shape == cfr.shape
        assert sample.cfo_hz.shape == (2, 1, 2)
        assert sample.sfo_ppm.shape == (2, 1, 2)
        assert sample.phase_offset_rad.shape == (2, 1, 2)
        assert sample.timing_offset_samples.shape == (2, 1, 2)
        assert sample.agc_gain_db.shape == (2, 2)
        assert sample.clipping_flag.shape == (2, 1, 2)

    def test_outputs_finite(self):
        cfr = torch.ones((1, 1, 1, 1, 1, 8), dtype=torch.complex64)
        config = ImpairmentConfig(
            random_seed=3, cfo_hz=200.0, sfo_ppm=5.0,
            phase_offset_rad=1.0, timing_offset_samples=2.0,
            clipping_threshold=2.0,
        )
        result, sample = apply_base_impairments(cfr, 8, 20e6, config)
        assert torch.all(torch.isfinite(result))
        assert torch.all(torch.isfinite(sample.cfo_hz))
        assert torch.all(torch.isfinite(sample.sfo_ppm))
