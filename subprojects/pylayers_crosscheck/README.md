# PyLayers cross-check subproject

Purpose: decide, with evidence, whether PyLayers is usable as an **independent
second ray-tracing engine** to cross-check the ideal-CSI (CFR) produced by the
Sionna pipeline for the Front3D test scenes.

Scope guardrails:

* **Isolated subproject.** Nothing here is imported by `sionna_measurement_sim`
  and nothing outside this directory is modified. The Sionna side is driven
  through the existing, unmodified CLI with a generated config copy.
* **Structure only.** Only `empty_scene/architecture` geometry is converted;
  furniture is excluded by contract.
* Single scene (`front3d_0015`) and five links (LoS / 1 wall / 2 walls ×2 /
  3 walls), not the 339-scene batch.

## Status at a glance

| stage | state |
|---|---|
| PyLayers environment (isolated, py3.8) | **works** — parses and builds the upstream `defstr.lay` in 0.4 s |
| mesh → `.lay` conversion | **works** — 132 segments, 13 faces, planar (0 crossings/overlaps/duplicate nodes) |
| `Layout.build()` on the converted layout | **blocked** — `buildGt` rejects 10 segments ("connected to LESS than 2 cycles") |
| link selection (LoS / 1 / 2 / 3 walls) | **works** — 794 / 405 / 94 / 11 candidates, 5 picked |
| PyLayers `DLink` CFR | **blocked** by the layout build (DLink samples rooms from `Gt`) |
| Sionna 1×1 baseline | runner ready (`run_sionna_1x1.py`), not executed |
| metric comparison | implemented (`compare.py`), needs the two CFR sets |

## Why the layout build blocks

PyLayers requires the wall network to be a **closed planar subdivision** in which
every segment borders at least two cycles. Auto-converted Front3D structure is
not that:

* walls exist as **two sheets** (inner/outer face of a 0.24 m wall);
* doorways and passages are **holes in the mesh**, not wall segments;
* the mesh carries mouldings, reveals and façade stubs that end in mid-air.

Each defect produced a distinct PyLayers failure, fixed in this order:

| # | failure | fix already implemented |
|---|---|---|
| 1 | point ids must be **negative** (`add_segment: error not a node`) | `build_lay.dedupe_and_split` emits ids `< 0` |
| 2 | layout must live in `<project>/struc/lay` (`getlong`) | `common.install_layout` |
| 3 | crossings with no shared node → Triangle "topological inconsistency" | `build_lay.planarize` |
| 4 | near-duplicate nodes / thin sliver doubles | 5 mm snap + `merge_close_lines` |
| 5 | Triangle `'pa'` inserts a Steiner vertex, tripping `setvnodes_new` | `refine_layout.py` splits the offending wall |
| 6 | dangling wall stubs → `KeyError` in `buildGt` | `build_lay.prune_dangling` |
| 7 | walls bordering no room | `build_lay.clean_planar` (half-edge face walk) |
| 8 | **remaining**: 10 segments still border < 2 cycles | `fit_pylayers.py` (removal loop) — not converged |

The remaining defect is the **thin cavity between the two wall sheets**: it forms
a face of a few square metres that PyLayers counts as a "room", and near
junctions these cavities degenerate. Collapsing the two sheets into one
centreline removes the cavities but opens gaps at every corner, which cascades
into losing most of the plan (tried: 0.30 m merge → 1 room, 15 segments).

## What is measured/verified so far

`front3d_0015` structure (architecture meshes only, furniture excluded):

* 16 shapes, 4 776 triangles, 904.4 m² structure area
* vertical 311.3 m² (34.4 %) / horizontal 588.3 m² (65.1 %) / non-planar 4.8 m² (0.53 %)
* horizontal levels: −0.10, 0.00, 2.65, 2.80 m → `zfloor=−0.1`, `zceil=2.8`
* 71 grouped wall lines (54 ≥ 0.5 m²) → 132 segments, 13 faces after cleanup
* overlay against the SceneGen floorplan: [artifacts/front3d_0015_layout_overlay.png](artifacts/front3d_0015_layout_overlay.png)

Selected links ([artifacts/front3d_0015_links.json](artifacts/front3d_0015_links.json)),
wall crossings counted on the SceneGen 5 cm floorplan (furniture ignored):

| link | BS | UE | walls | 2D distance |
|---|---|---|---|---|
| 0 | BS_0_0 | UE_0_0 | 0 (LoS) | 3.50 m |
| 1 | BS_0_0 | UE_2_110 | 1 | 5.70 m |
| 2 | BS_0_0 | UE_2_153 | 2 | 6.50 m |
| 3 | BS_0_0 | UE_2_168 | 2 | 6.96 m |
| 4 | BS_2_1 | UE_2_100 | 3 | 6.71 m |

Physics kept identical on both sides: ITU-R P.2040-3 concrete (`a=5.24, b=0,
c=0.0462, d=0.7822` → εr = 5.24, σ = 0.121 S/m at 3.438 GHz), slab thickness
0.1 m (Sionna `DEFAULT_THICKNESS`), 768 subcarriers at 30 kHz, single isotropic
antennas, no scattering, no diffraction, 4 interactions.

## Environment

PyLayers is 2019 code (last meaningful commit `d9bc71f`, 2019-08-01) with a
2019-era dependency set. It runs here in a throwaway env:

```
uv python install 3.8
uv venv --python 3.8 .venv-pylayers
uv pip install numpy==1.19.5 scipy==1.5.4 networkx==2.3 shapely==1.7.1 \
               matplotlib==3.3.4 h5py==3.1.0 pandas==1.3.5 seaborn==0.11.2 \
               tqdm configobj descartes triangle
```

Location: `tmp/pylayers_probe/` (not committed, ~1 GB). Sources are fetched per
file from GitHub because pack transfer to this host fails with HTTP 408.

`shims/sitecustomize.py` stubs GIS/plotting modules that PyLayers imports at
module import time but never uses on an indoor cartesian path
(`basemap`, `shapefile`, `gdal`, `pyproj`, `mayavi`, `rtree`, …). Every stub is
announced on stderr; real modules win when present.

## Commands

```bash
# 1. structure diagnostics / converter
python inspect_geom.py front3d_0015
python build_lay.py --scene front3d_0015 --offset-tol 0.015 --merge-sep 0.05
python check_layout.py --scene front3d_0015            # floorplan overlay

# 2. link selection (SceneGen floorplan, furniture ignored)
python select_links.py --scene front3d_0015

# 3. PyLayers side (isolated env)
tmp/pylayers_probe/run_py.sh subprojects/pylayers_crosscheck/diagnose_lay.py --lay <lay>
tmp/pylayers_probe/run_py.sh subprojects/pylayers_crosscheck/refine_layout.py --lay <lay>
tmp/pylayers_probe/run_py.sh subprojects/pylayers_crosscheck/validate_lay.py --lay <lay> --build
tmp/pylayers_probe/run_py.sh subprojects/pylayers_crosscheck/run_pylayers.py --scene front3d_0015

# 4. Sionna single-antenna baseline (existing CLI, unmodified)
python run_sionna_1x1.py --scene front3d_0015 --gpus 2,3 --execute

# 5. comparison
python compare.py --scene front3d_0015
```

## Metrics contract (once both sides exist)

Computed by the same code from both 768-point CFRs: resolvable path count,
first-path excess delay, PDP shape correlation, CFR amplitude correlation,
complex coherence, RMS delay spread, band-mean path gain (`compare.py`).

Suggested acceptance: first-path delay difference < 1 ns, LoS gain difference
< 1 dB, PDP correlation > 0.8, CFR amplitude correlation > 0.7.

## Recommended next step

Rebuild the plan **from the SceneGen floorplan mask** (5 cm occupancy, already
used for link selection) instead of from the mesh sheets: free-space components
give room polygons, their boundaries give single-line walls, and door gaps are
known. That yields a clean, closed, cavity-free subdivision by construction —
the property PyLayers' `buildGt` actually needs — while staying anchored to the
same scene the Sionna run used. Cost estimate: half a day; it also makes the
result reproducible for other scenes.

Keeping the mesh two-sheet representation would require either patching
PyLayers' `buildGt`/`_triangle` (engine modification, weakens the
"independent implementation" argument) or hand-editing each layout.
