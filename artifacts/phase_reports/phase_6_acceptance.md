# Phase 6 Acceptance Report

## Summary

Motion and Doppler support implemented: multi-snapshot CFR generation, velocity/orientation, /motion HDF5 group, delay-Doppler visualization.

## Acceptance Commands

```bash
uv run pytest tests/adapter tests/integration tests/statistical -k "doppler or motion"  # 5 passed
uv run pytest  # 56 passed
uv run ruff check .  # all checks passed
uv run python -m sionna_measurement_sim.app.cli run-motion \
    --output-dir outputs/phase6_motion --num-time-steps 3 --sampling-frequency-hz 100
python -c "from sionna_measurement_sim.visualization.path_plots import plot_delay_doppler; \
    plot_delay_doppler('outputs/phase6_motion/results.h5', 'outputs/phase6_motion/delay_doppler.png')"
```

## Acceptance Items

- [x] `/devices/tx_velocity_mps` and `/devices/rx_velocity_mps` exist (already from Phase 1)
- [x] `/devices/tx_orientation_rad` and `/devices/rx_orientation_rad` exist (already from Phase 1)
- [x] `/motion/snapshot_id` shape matches `num_time_steps`
- [x] `/motion/timestamp_s` monotonically increasing
- [x] `/motion/sampling_frequency_hz` = 100.0
- [x] `/motion/num_time_steps` = 3
- [x] `/motion/mobility_mode` = "doppler_synthetic"
- [x] Multi-snapshot CFR shape = `(3, 1, 1, 1, 1, 8)` consistent with `num_time_steps=3`
- [x] Static device state has zero velocity
- [x] Non-zero velocity can be configured
- [x] `/paths/samples/doppler_hz` exists with finite values
- [x] Delay-Doppler scatter plot generated
- [x] Timestamp monotonicity enforced in MotionSpec
- [x] Pipeline supports `--num-time-steps` and `--sampling-frequency-hz`
- [x] Schema validator handles both 5D and 6D truth CFR

## Output

- outputs/phase6_motion/results.h5
- outputs/phase6_motion/manifest.json
- outputs/phase6_motion/logs/run.log
- outputs/phase6_motion/delay_doppler.png
