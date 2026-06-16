# Mode / Config Performance Optimization Smoke

Date: 2026-06-16

Branch: `codex/mode-config-perf-optimization`

This note records the smoke measurements used while consolidating output modes.
`data/` and `outputs/` remain local ignored paths; the output directories below
are not part of the repository.

## Change Under Test

Schema `2.3.0` removes historical pseudo profiles `rt_lite` and `custom`.
Only three output contracts remain:

- `full`
- `rt_labels_only`
- `iq_link_library`

Lightweight full-contract outputs are now expressed as:

```yaml
output:
  profile: "full"
  products: [...]
```

`link_labels` is now a full product, and `full + products: ["iq"]` shares the
same clean-IQ fast path as `iq_link_library` when no observed IQ, CFR estimate,
ranging, multiuser, calibration, or observation-based array spectrum is needed.

## Smoke Setup

Input:

- `data/front3d_20/front3d_0002/label/label_panel_0p5.json`
- `data/front3d_20/front3d_0002/scene.xml`

Scale:

- `max_bs: 3`
- `max_ue: 4`
- 64 PRB / 768 subcarriers
- sharding disabled
- array spectrum, ranging, visualization, calibration disabled

Temporary configs were generated under:

```text
outputs/mode_config_perf_configs/
```

## Results

| Output | Contract | HDF5 groups | Directory size | Perf duration |
|---|---|---|---:|---:|
| `full + products: ["derived", "link_labels"]` | full | topology/derived/labels/link | 252 KB | 5.543 s |
| `rt_labels_only` | compact labels | topology/derived/labels/link | 252 KB | 4.688 s |
| `full + products: ["cfr_truth"]` | full | channel/truth/cfr | 1.5 MB | 4.680 s |
| `iq_link_library` | compact IQ | iq/link/time_clean | 1.4 MB | 4.925 s |
| `full + products: ["iq"]` | full | iq/link/time_clean | 1.4 MB | 4.784 s |
| `full` | full | channel/paths/waveform/observation/labels | 30 MB | 6.689 s |

Top write payloads:

- Full SRS: `/waveform/rx_grid` raw 16.515 MB, stored 14.801 MB.
- Full SRS: `/paths/samples/vertices_m` raw 6.276 MB, stored 1.346 MB.
- IQ-only: `/iq/link/time_clean` raw 16.515 MB, stored about 1.255 MB.
- CFR-truth-only: `/channel/truth/cfr` raw 2.359 MB, stored 1.330 MB.

## Interpretation

The product-aware full path now behaves like a real compute/write selector:

- `link_labels` can be emitted in full contract without computing CFR/CIR/path
  samples or PHY observation.
- `cfr_truth` computes and writes CFR truth only.
- `iq` no longer needs to run SRS impairment, LS receiver, full-band CFR
  interpolation, ranging, or array outputs when only clean IQ is requested.
- Compact `iq_link_library` and full-contract `products: ["iq"]` produce the
  same payload shape for this setup:
  `[snapshot, tx, rx, rx_ant, sample] = [1, 4, 3, 16, 10752]`.

The existing full baseline for
`outputs/front3d_20_front3d_0002_panel0p5_srs_64prb_formal_full` remains the
reference for full 0002/0p5 cost:

- 57 HDF5 shards
- 282 UE, 11 BS
- about 8.6 GB
- wall time about 228 s
- dominant stages: HDF5 write, RT solve, array outputs, SRS observation,
  visualization, schema validation

The smoke confirms that lightweight products avoid the main full-output storage
drivers while preserving the same RT/SRS truth semantics where requested.
