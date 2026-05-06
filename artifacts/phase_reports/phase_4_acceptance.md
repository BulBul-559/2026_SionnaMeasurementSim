# Phase 4 Acceptance

Status: passed

Scope:

- Added PyTorch dependency for PHY tensor operations.
- Added minimal custom OFDM waveform, AWGN impairment, LS estimator output, observation diagnostics, and NMSE evaluation domain models.
- Wrote `/waveform`, `/observation`, `/impairments`, `/receiver`, and `/evaluation` HDF5 groups.
- Added unit, integration, and statistical tests for AWGN-only observation.

Commands run:

```bash
uv add torch
uv run pytest tests/unit tests/integration tests/statistical -k "awgn or observation or nmse"
uv run python -m sionna_measurement_sim.app.cli run-observation --output-dir outputs/phase4_observation --snr-db 40
uv run ruff check .
uv run pytest
git status --short --ignored
```

Results:

- `uv add torch`: passed; installed `torch==2.11.0`.
- `uv run pytest tests/unit tests/integration tests/statistical -k "awgn or observation or nmse"`: passed, 3 tests.
- CLI observation run: passed; generated `outputs/phase4_observation/results.h5`.
- `uv run ruff check .`: passed.
- `uv run pytest`: passed, 22 tests.

Acceptance items:

- `/observation/cfr_est` shape is `[snapshot, tx, rx, rx_ant, tx_ant, subcarrier]`, observed as `(1, 1, 1, 1, 1, 8)`.
- `/observation/cfr_est.shape[1:] == /channel/truth/cfr.shape`.
- `valid_mask`, `detection_success`, and `estimation_success` shape is `(1, 1, 1)`.
- AWGN-only high SNR median NMSE is below `-20 dB`; generated output observed about `-41.3 dB`.
- Statistical test verifies high SNR median NMSE is lower than low SNR median NMSE.
