# Phase 7 Acceptance Report

## Summary

Calibration and diagnostics implemented: CalibrationResult domain model, /calibration HDF5 group, DiagnosticsReport with per-link summary statistics, and calibration profile integration.

## Acceptance Commands

```bash
uv run pytest tests/unit tests/integration -k "calibration or diagnostics"  # 9 passed
uv run pytest  # 65 passed
uv run ruff check .  # all checks passed
uv run python -m sionna_measurement_sim.app.cli run-observation \
    --output-dir outputs/phase7_calibration --snr-db 40
```

## Acceptance Items

- [x] `/calibration/profile_id` = "synthetic_default"
- [x] `/calibration/fitted_parameters` contains valid JSON
- [x] `/calibration/validation_metrics` contains valid JSON
- [x] Manifest includes `diagnostics` summary with median NMSE, SNR, detection rate, failure rate, num_links
- [x] `ReceiverSpec.calibration_profile_id` = "synthetic_default" (no longer dead label)
- [x] DiagnosticsReport.from_evaluation() computes per-link aggregate statistics
- [x] CalibrationResult.synthetic_default() produces identity correction
- [x] Schema validator checks /calibration datasets when group present
- [x] Diagnostics written to manifest.json for observation runs

## Output

- outputs/phase7_calibration/results.h5 (contains /calibration group)
- outputs/phase7_calibration/manifest.json (contains diagnostics)
