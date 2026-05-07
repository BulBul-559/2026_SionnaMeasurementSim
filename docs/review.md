# Review: docs 13 and 14

Review date: 2026-05-07

Scope:

- [13_tdd_reciprocity_nr_pusch_phy_plan.md](13_tdd_reciprocity_nr_pusch_phy_plan.md)
- [14_rt_hardening_before_nr_pusch.md](14_rt_hardening_before_nr_pusch.md)
- Lightweight code/test spot check for the requirements described by these two documents.

## Overall Conclusion

13 and 14 describe the right architecture direction: first harden Sionna RT truth, CIR, shape contracts, antenna configuration, and 4x4 MIMO; then use that RT/CIR foundation to build a TDD uplink NR PUSCH PHY path.

However, the documentation set is currently not clean enough to be used as a single source of truth. The main issues are:

- 14 is a prerequisite for 13, but numerically appears after 13. This creates execution-order ambiguity.
- Both 13 and 14 now contain appended `review` sections that are partially stale relative to the current codebase.
- 13 still mixes a final target design with partially implemented skeleton behavior. The document should separate "must be physically correct" from "temporary smoke-test bridge".
- 03 and 06 have not fully caught up with the NR PUSCH fields required by 13, especially `/waveform`, `/receiver`, `/evaluation`, and config naming.
- The current NR PUSCH implementation is closer than the old review says, but still not physically strong enough to claim full 13 completion.

## Status Summary

### 14 RT hardening

Current status: mostly implemented, but the document should be refreshed.

Observed current positives:

- Dedicated RT shape/CIR tests exist:
  - `tests/adapter/test_rt_shape_contracts.py`
  - `tests/adapter/test_rt_cir_adapter.py`
  - `tests/schema/test_rt_cir_schema.py`
  - `tests/integration/test_rt_mimo_4x4_pipeline.py`
- `to_project_cir(...)` now exists in `sionna_measurement_sim/adapters/sionna_rt/shape_contracts.py`.
- `AntennaSpec` now has separate TX/RX orientation fields.
- HDF5 writer now writes distinct `tx_orientation_mode` and `rx_orientation_mode`.
- The 4x4 MIMO RT/CIR tests passed in this review run.

Remaining documentation problems:

- The `## review` section at the end of 14 is stale. It still says the 4x4/CIR tests are missing and that RX orientation is not split, but those points appear to have been addressed in current code.
- 14 is still written mostly as a future plan. It should be converted into a status-aware acceptance document: completed, pending, blocked, and verified commands.
- The document should explicitly say whether `look_at_centroid` for RX is implemented or only TX supports look-at modes. Current RT code sets RX orientation from fixed config, so 14 should not imply symmetric look-at behavior unless implemented.
- 14 should require fail-fast validation for unsupported Sionna pattern/polarization values and document the exact supported values tested locally.
- 14 should keep the no-label-change decision, but it should also clarify how BS/UE roles are mapped from labels before entering 13.

### 13 TDD reciprocity and NR PUSCH PHY

Current status: partially implemented, not fully complete.

Observed current positives:

- `LinkConfig` exists and rejects unsupported non-TDD / non-uplink modes.
- `apply_tdd_reciprocity(...)` and `apply_tdd_reciprocity_cir(...)` exist and have basic unit tests.
- `run-full --phy-standard nr_pusch` exists.
- `run_rt_truth_pipeline(...)` selects the NR PUSCH branch when `phy_standard == "nr_pusch"`.
- `nr_pusch_observation.py` builds Sionna `PUSCHConfig`, `PUSCHTransmitter`, attempts a `PUSCHReceiver`, and records BER/BLER fields.

Remaining technical risks:

- The NR PUSCH backend still contains simplified channel application: it averages antenna dimensions and applies only the first TX/RX pair in the frequency domain. That does not yet satisfy the 4x4 MIMO PUSCH receiver semantics required by 13.
- The receiver path catches all exceptions and falls back to `ber=0.0`, `bler=0.0`, and zero bit counts. This can make a failed receiver look like a perfect link. For 13 acceptance, receiver failure must fail the test or be marked invalid.
- `EvaluationResult` still uses `nmse_db_total`, and `docs/03_data_contract_hdf5.md` still documents `nmse_db` as AWGN-isolation while 13 requires `/evaluation/nmse_db` to mean `H_obs` vs clean `H_true`.
- `WaveformSpec` and `/waveform` remain custom-OFDM shaped. The HDF5 contract does not yet list the NR PUSCH fields from 13: PRB count, slot, DMRS config, layers, antenna ports, MCS, coderate, modulation.
- `/receiver` in 03 still documents legacy receiver fields and does not clearly define `receiver_type="pusch_receiver"`, `channel_estimator`, `mimo_detector`, and `input_domain`.
- 13 asks for integration and statistical tests:
  - `tests/integration/test_nr_pusch_observation.py`
  - `tests/statistical/test_nr_pusch_link_metrics.py`
  These were not present in the current file listing.
- The current 13 appended review is stale. It says the backend is not connected to the main pipeline and no `PUSCHReceiver` is attempted; current code has moved beyond that. The stale review should be replaced by a fresh acceptance matrix.

## Cross-Document Issues

1. Execution order is confusing.

14 must be completed before 13, but 13 is numbered first. Keep the filenames if needed, but add a clear gate in README and phase progress:

```text
Run order for this pair:
  14 RT hardening
  then 13 TDD NR PUSCH
```

2. The HDF5 contract is behind the plans.

13 and 14 both require schema changes, but `03_data_contract_hdf5.md` is not fully aligned with 13. This should be fixed before declaring 13 complete.

3. Config naming is inconsistent.

13 writes `subcarrier_spacing_hz`, while current code/config uses `subcarrier_spacing_khz`. Pick one public config name and document unit semantics consistently. For 5G NR, `subcarrier_spacing_khz` is natural, but HDF5 may also store derived `subcarrier_spacing_hz`.

4. Link direction semantics need one authoritative rule.

13 says `/channel/truth/cfr` remains RT-direction BS-to-UE truth, while PUSCH uses uplink tensors internally. The docs must explicitly define the direction of `/observation/cfr_est` for `standard="nr_pusch"`. If it is uplink perspective, HDF5 must record that clearly.

5. Reciprocity needs a stronger physical note.

The docs should state that TDD reciprocity here assumes same carrier frequency, calibrated RF chains, and consistent antenna phase references. The transform should explicitly state whether it is transpose-only, conjugate transpose, or another calibrated mapping. Do not leave this implicit.

6. MIMO stream mapping is under-specified.

14 defines physical antenna dimensions. 13 defines `num_layers` and `num_antenna_ports`. The bridge between them must be documented:

```text
RT tx_ant/rx_ant
  -> UE/BS antenna ports
  -> PUSCH layers / streams
  -> detector expected dimensions
```

Without this, a 4x4 RT channel can still be collapsed into a SISO-like PHY path without tests noticing.

7. The embedded reviews should not remain as historical truth.

Both 13 and 14 contain inline `## review` sections. They are useful history, but they are now drifting. Prefer either:

- move reviews into this document only, or
- rename old sections to `Historical review` and add dates/status.

## Recommended Acceptance Gates

### Gate A: Refresh 14

Before touching more NR PHY code:

1. Update 14's stale review section.
2. Run and record:

```bash
uv run pytest tests/adapter/test_rt_shape_contracts.py tests/adapter/test_rt_cir_adapter.py tests/schema/test_rt_cir_schema.py tests/integration/test_rt_mimo_4x4_pipeline.py
uv run ruff check .
```

3. Confirm 14 explicitly answers:

```text
Can RT generate auditable 4x4 CFR, CIR, paths, antenna orientation, pattern,
polarization, merge_shapes config, and config snapshots?
```

### Gate B: Align contracts before 13 completion

Update:

- `docs/03_data_contract_hdf5.md`
- `docs/06_config_and_experiment_schema.md`
- schema validator
- HDF5 writer
- schema tests

Required additions include NR PUSCH waveform fields, receiver fields, link direction fields, and corrected NMSE semantics.

### Gate C: Harden NR PUSCH path

For 13 acceptance:

1. Remove silent receiver-success fallback.
2. Do not average antenna dimensions or use only first TX/RX pair for the accepted path.
3. Use the Sionna receiver/channel path in a way that preserves MIMO and PUSCH semantics.
4. Compute BER/BLER from real decoded or estimated bits.
5. Add high/low Eb/N0 and perfect/imperfect CSI statistical checks.

### Gate D: End-to-end tests

At minimum, add:

```text
tests/integration/test_nr_pusch_observation.py
tests/statistical/test_nr_pusch_link_metrics.py
```

These tests should verify:

- `/waveform/standard == "nr_pusch"`
- `/receiver/receiver_type == "pusch_receiver"`
- `/evaluation/ber` and `/evaluation/bler` are finite and not placeholder-only
- high Eb/N0 is no worse than low Eb/N0
- perfect CSI is no worse than estimated CSI
- `/observation/cfr_est` direction and shape are documented and validated

## Commands Run During This Review

```bash
uv run pytest tests/adapter/test_rt_shape_contracts.py tests/adapter/test_rt_cir_adapter.py tests/schema/test_rt_cir_schema.py tests/integration/test_rt_mimo_4x4_pipeline.py tests/unit/test_reciprocity.py tests/unit/test_nr_pusch_config.py
uv run ruff check .
```

Result:

```text
38 passed, 1 warning
ruff: All checks passed
```

This is a good sign for the implemented skeleton and RT hardening tests. It is not yet sufficient to declare 13 complete, because the missing integration/statistical NR PUSCH acceptance tests are part of 13's own definition of done.
