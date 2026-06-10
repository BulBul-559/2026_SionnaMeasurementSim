# Task Configs

This folder stores project-level task templates for repeated production runs.
Unlike `config/defaults/`, these files are not minimal examples; they are
opinionated run recipes that should be copied or patched with scene-specific
paths before launching a batch.

Current task templates:

| Template | Purpose |
|---|---|
| `nr_srs_64prb_formal.yaml` | Formal NR SRS 64 PRB direct-array simulation recipe. It uses a 100 MHz FR1 context but simulates only the occupied 64 PRB bandwidth (`768` subcarriers at 30 kHz SCS). |

For large local runs, keep resolved configs, logs, summaries, and generated
figures inside each `outputs/<run_name>/` directory.

Task templates should be explicit rather than minimal. In particular, keep
`runtime.require_gpu/precision/torch_deterministic`,
`input.label_schema/coordinate_system`, `output.run_id_format`, `motion.*`, and
`calibration.*` written out even when they match schema defaults. Static formal
positioning tasks should set `motion.num_time_steps: 1` unless the experiment is
explicitly about motion/Doppler; otherwise waveform, CFR, RSS, spectrum, and
storage costs scale with the snapshot count.
