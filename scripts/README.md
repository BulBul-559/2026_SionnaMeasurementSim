# Scripts

This directory contains small analysis and validation helpers that are useful
for local experiments. Scripts may default to paths under `data/` or
`outputs/`, but those directories remain ignored local data roots and must not
be committed.

## CFR Similarity

The CFR similarity helpers compare `/observation/cfr_est` from two simulation
outputs and write figures/CSV/JSON summaries back under `outputs/`.

| Script | Purpose |
|---|---|
| `plot_cfr_similarity_floorplan_heatmaps.py` | Compare matched shard outputs and overlay magnitude/phase/I/Q similarity heatmaps on a floorplan. |
| `plot_cfr_similarity_by_position.py` | Match UE positions across runs before computing similarity, useful for cross-density comparisons. |
| `plot_cfr_similarity_exclude_region.py` | Recompute statistics and heatmaps after excluding a rectangular region. |
| `plot_normalized_cfr_similarity_heatmaps.py` | Normalize existing similarity CSVs after dropping zero-valued missing samples, then replot heatmaps. |
| `plot_radio_map_heatmaps.py` | Generate one `/observation/rssi_dbm` floorplan radio map per BS from a run directory or HDF5 file. |

Do not run these helpers with recursive scans over `data/` or `outputs/`.
Point them at explicit run directories, files, or named output folders.
