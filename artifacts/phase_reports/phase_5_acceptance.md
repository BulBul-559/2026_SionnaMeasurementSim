# Phase 5 Acceptance Report

## Summary

Base impairments implemented: CFO, SFO, phase offset, timing offset, AGC, and ADC/clipping.

## Acceptance Commands

```bash
uv run pytest tests/unit tests/statistical -k "cfo or sfo or clipping or impairment"  # 25 passed
uv run pytest  # 47 passed
uv run ruff check .  # all checks passed
uv run python -m sionna_measurement_sim.app.cli run-observation \
    --output-dir outputs/phase5_impairments --snr-db 40 \
    --cfo-hz 100 --sfo-ppm 5 --timing-offset-samples 2 \
    --clipping-threshold 3 --phase-offset-rad 0.5
```

## Acceptance Items

- [x] Fixed seed → reproducible impairment sampling (test_impairment_reproducibility)
- [x] CFO applied: /observation/cfo_hz = 100 Hz
- [x] SFO applied: /observation/sfo_ppm = 5 ppm
- [x] Phase offset: /observation/phase_offset_rad = 0.5 rad
- [x] Timing offset: /observation/timing_offset_samples = 2 samples
- [x] Clipping flag: /observation/clipping_flag present
- [x] /impairments/model_version = "phase5_base_impairments_v1"
- [x] /impairments/random_seed present
- [x] CFO disabled → cfo_hz = 0
- [x] Clipping threshold lowered → clipping ratio not decreased
- [x] All impairment config JSON written to HDF5
- [x] Schema validator updated for all impairment datasets
- [x] CLI flags: --cfo-hz, --sfo-ppm, --phase-offset-rad, --timing-offset-samples, --clipping-threshold

## Output

- outputs/phase5_impairments/results.h5
- outputs/phase5_impairments/manifest.json
- outputs/phase5_impairments/logs/run.log
