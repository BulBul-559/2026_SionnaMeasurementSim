# NR PUSCH 5x5000 Performance Profile

Date: 2026-05-12

Branch used for analysis: `perf/nr-pusch-5x5000-profile`

Output directory: `outputs/perf_nrp5x5000_full_output`

## Run Setup

This run used a single isolated RTX 4090 via `CUDA_VISIBLE_DEVICES=0` and the full-output pressure profile:

- 5 BS x 5000 UE = 25000 SU-MIMO links
- NR PUSCH 4x4 SU-MIMO, `pusch_ls`, `lmmse`
- `array.spectrum.enabled=true`
- spectrum sources: `truth_cfr`, `cfr_est`, `rx_grid`
- visualization enabled
- `save_full_paths=false`

The working CLI requires the global `--config` option before `run-full`:

```bash
CUDA_VISIBLE_DEVICES=0 \
SIONNA_PERF_TRACE=1 \
SIONNA_PERF_HARDWARE_INTERVAL_S=1 \
SIONNA_PERF_LINK_LOG_INTERVAL=250 \
uv run python -m sionna_measurement_sim.app.cli \
  --config config/perf/nr_pusch_5x5000_full_output.yaml \
  run-full \
  --max-tx 5 \
  --max-rx 5000 \
  --output-dir outputs/perf_nrp5x5000_full_output
```

The profiler wrote:

- `logs/perf_events.jsonl`
- `logs/hardware_samples.csv`
- `logs/link_chunks.csv`

## Results

Total wall time was 1275.5 s, about 21 min 16 s.

| Stage | Time | Share | Main signal |
|---|---:|---:|---|
| NR PUSCH observation | 839.3 s | 65.8% | Low GPU util, sequential per-link receiver work |
| HDF5 write | 206.5 s | 16.2% | I/O/compression long tail |
| Sionna RT solve | 114.8 s | 9.0% | GPU compute-bound |
| Visualization | 55.9 s | 4.4% | CPU/read/Matplotlib |
| Array outputs and spectra | 52.8 s | 4.1% | CPU vectorized spectrum generation |
| Schema validation | 4.5 s | 0.4% | Small |

NR PUSCH processed all 25000 links with zero receiver failures.

| PUSCH substage | Total | Per-link mean | Interpretation |
|---|---:|---:|---|
| `PUSCHReceiver` | 625.3 s | 25.0 ms | Dominant PUSCH cost |
| LS estimator | 114.1 s | 4.6 ms | Secondary cost; currently built/called per link |
| TX generation | 45.5 s | 1.8 ms | Small but repeated 25000 times |
| Channel apply total | 19.5 s | 0.8 ms | Not the bottleneck |
| Metrics/CPU conversion/slices | 15.7 s | <1 ms each | Minor individually |

Per 250-link chunk was very stable:

- Mean link time: 33.35 ms
- Fastest chunk mean: 32.92 ms/link
- Slowest chunk mean: 34.07 ms/link
- First chunk had one cold-start max outlier around 282 ms

Hardware observations:

| Stage | GPU util avg/max | GPU mem max | CPU avg | RSS max |
|---|---:|---:|---:|---:|
| RT solve | 82% / 100% | 6.5 GB | 103% | 3.3 GB |
| PUSCH link loop | 10.9% / 11% | 11.6 GB | 101% | 5.6 GB |
| Array spectra | 0% | 11.6 GB | 765% | 11.8 GB |
| HDF5 write | 0% | 11.6 GB | 101% | 12.0 GB |
| Visualization | 0% | 11.6 GB | 100% | 12.4 GB |

Output checks:

- `results.h5`: 4.08 GB
- `/array/spatial_spectrum_truth`: `(1, 5000, 5, 91, 181)`
- `/array/spatial_spectrum_cfr_est`: `(1, 5000, 5, 91, 181)`
- `/array/spatial_spectrum_observation`: `(1, 5000, 5, 91, 181)`
- `/waveform/tx_grid`: `(1, 5000, 5, 4, 14, 48)`
- `/waveform/rx_grid`: `(1, 5000, 5, 4, 14, 48)`
- `/waveform/tx_time` and `/waveform/rx_time`: absent
- `figures/index.json`: present; PNG files are non-empty
- HDF5 schema validation passed during the run

## Bottleneck Diagnosis

The main runtime bottleneck is the SU-MIMO per-link NR PUSCH loop.

The run keeps about 11.6 GB on GPU during PUSCH, but GPU utilization stays near 11%. This means the workload is not memory-capacity limited and not saturating GPU compute. It is dominated by many small sequential calls, especially `PUSCHReceiver`, with one Python process feeding one link at a time.

The second bottleneck is HDF5 output. With full spectra enabled, writing takes 206.5 s and produces a 4.08 GB HDF5 file. During this phase GPU is idle and CPU is about one core, so optimizing PUSCH kernels will not reduce this tail.

The third bottleneck is RT solve for large scene/link counts. Unlike PUSCH, RT solve does saturate GPU: average GPU utilization is 82%, peak 100%, and power reaches about 451 W. This explains the observed case where GPU utilization is high while memory use is relatively modest.

Array spectrum generation is a separate CPU-heavy stage. It uses multiple CPU cores and raises RSS close to 12 GB. It is not the top wall-time contributor in this run, but it materially increases memory pressure before HDF5 write and visualization.

The detailed per-link trace itself is heavy: `perf_events.jsonl` is 117 MB. For future repeated profiling, chunk-level logging or sampled per-link tracing should be used unless substage timing is specifically needed.

## Optimization Recommendations

1. Batch SU-MIMO links before optimizing kernels.

   The current shape is effectively 25000 small independent receiver calls. The highest-impact change is to process multiple `(UE, BS)` links in one call using a larger batch dimension where Sionna permits it. Target first batch sizes like 16, 32, 64, then measure GPU utilization and memory growth.

2. Move reusable receiver-side objects out of the per-link path.

   `PUSCHLSChannelEstimator` is constructed inside each link. Even if construction is not the whole 4.6 ms LS cost, it is repeated 25000 times and should be hoisted or cached per resource-grid/DMRS config.

3. Add UE/BS sharding before multi-GPU scheduling.

   Multi-GPU should split the UE dimension across processes, one process per GPU, writing per-shard HDF5 outputs and shard manifests. Current config can limit `max_rx` but does not yet support `rx_start/rx_indices`; add explicit shard selection or generate shard label files.

4. Make heavy outputs shard-aware and optionally post-process.

   Full spectra and waveform grids make HDF5 write a major cost. For production-scale runs, write per-shard files first and merge or index them later. Consider making spatial spectra an offline post-process when the training pipeline does not need them immediately.

5. Avoid duplicate large spectrum storage where possible.

   `aoa_heatmap_label` and `spatial_spectrum_label` are semantically aliases. At 5x5000 and 91x181, storing both as physical datasets adds avoidable file size and write time. Use an HDF5 hard link or keep only one physical dataset with a compatibility alias strategy.

6. Keep RT and PUSCH optimization separate.

   RT is GPU compute-bound; PUSCH is dispatch/receiver-call bound; HDF5 is I/O-bound. A single optimization will not fix all three. Use separate benchmarks for RT-only, PUSCH-only from cached CIR, and HDF5/visualization-only from cached arrays.

## Next Profiling Runs

Recommended follow-up experiments:

1. Run 5x5000 with `array.spectrum.enabled=false` and visualization disabled to isolate RT + PUSCH.
2. Run PUSCH from cached CIR without RT to isolate receiver/link loop cost.
3. Prototype batched SU-MIMO for batch sizes 8, 16, 32, 64 on one GPU.
4. Prototype 2-GPU UE sharding with two independent output files before implementing merge.
5. Compare HDF5 write with spectra disabled, spectra enabled, and alias-only label storage.

## Notes

This report intentionally keeps the profiling instrumentation on the analysis branch. Only this report should be checked out back to `main`; the temporary tracing code and profiling config should not be merged as production code.
