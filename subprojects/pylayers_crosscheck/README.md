# PyLayers cross-check subproject

> **STATUS: PAUSED (2026-10-29).** The owner stopped this line of work: the
> reproduction cost is too high relative to the payoff. Keep this directory as
> the evidence trail; do not extend it without a new decision.
>
> What stands: the semantic wall reconstruction now matches the SceneGen
> floorplan (92.5 % of wall pixels within 15 cm, median deviation ~5 cm) after
> fixing a y-flip in the normalized->scene frame mapping. What blocks: PyLayers'
> own `Layout.build()` still fails on this plan (Mikado cycle merge raises
> `MultiPolygon are not allowed`), and getting past it needs an engine port, not
> a shim. Two engine-side compatibility patches were already needed to reach
> that point (see `patches/pylayers_local_fixes.md`), which by itself weakens the
> "independent implementation" argument for a cross-check.

Purpose: decide, with evidence, whether PyLayers can act as an **independent
second ray-tracing engine** to cross-check the ideal-CSI (CFR) that the Sionna
pipeline produces for the Front3D test scenes.

Guardrails: isolated subproject (nothing imports it, nothing outside it was
modified), structure-only geometry (furniture excluded), one scene
(`front3d_0015`) and five links instead of the 339-scene batch.

## Status

| stage | state |
|---|---|
| PyLayers env (isolated py3.8) | **works** — upstream `defstr.lay` builds in 0.4 s |
| semantic wall reconstruction | **works** — 48 wall panels → 136 segments, 110.8 m of wall, 0 crossings / 0 overlaps / 0 duplicate nodes |
| `Layout.build()` on the converted layout | **blocked** — stops in PyLayers' "Mikado" cycle merge (`MultiPolygon are not allowed`) |
| PyLayers `DLink` CFR | blocked by the build (DLink samples rooms from `Gt`) |
| Sionna 1×1 baseline runner | ready ([run_sionna_1x1.py](run_sionna_1x1.py)), not executed |
| metric comparison | implemented ([compare.py](compare.py)), needs both CFR sets |

## Why the semantic reconstruction was necessary

`front3d_0015`'s structure mesh models every wall as **two sheets 0.24 m apart**.
Feeding both to PyLayers encloses a thin cavity that PyLayers counts as a room,
and its graph build then fails on the resulting degenerate faces. Two earlier
attempts and their measured outcomes:

| attempt | result |
|---|---|
| keep both sheets | plan is clean (0 crossings/overlaps) but `buildGt` rejects 10 segments ("connected to LESS than 2 cycles") |
| pair sheets + extend corners | geometry distorted: only 42 % of floorplan wall pixels covered within 15 cm, visibly skewed walls |

The working path is the **SceneGen normalized architecture**
(`3D-Front/scenegen_architecture_normalized/<uuid>/`), which keeps the semantic
decomposition that the exported mesh loses:

```
WallInner 30  WallOuter 18  WallTop 24  WallBottom 24
Door 2  Window 4  Floor 4  Ceiling 4  SlabSide 35  SlabBottom 1  BayWindow 11 ...
```

Each wall is a single box panel, so its centreline is exact, and `Door`/`Window`
panels give the aperture footprint directly. `build_lay_semantic.py` converts the
panels to `.lay` segments, bridges every opening with a transparent AIR segment
(PyLayers needs a closed subdivision; an AIR segment keeps the hole transparent
like the mesh hole in Sionna), and translates the normalized frame back to the
scene frame used by `scene.xml` and the labels (+4.0196, +6.6345).

Result: `zfloor=0.0`, `zceil=2.8`, 48 wall panels + 17 openings, 105 nodes,
136 segments, and the wall network passes its own planarity check
(0 crossings, 0 collinear overlaps, 0 duplicate nodes). PyLayers parses it
(249 nodes, 140 segments after its AIR/_AIR completion) and the constrained
triangulation succeeds (154 triangles).

## Remaining blocker

With the two documented compatibility fixes
([patches/pylayers_local_fixes.md](patches/pylayers_local_fixes.md)) the build
gets through triangulation and the triangle→node mapping, then fails in the
"Mikado" cycle-merge step:

```
File pylayers/gis/layout.py, line 6363, in buildGt
    P = p0 + p1
File pylayers/util/geomutil.py, line 405, in __init__
    raise AttributeError('MultiPolygon are not allowed')
```

That step merges adjacent cycles across each transparent segment and assumes the
union is always a simple polygon. Finishing it is a real porting task (the
transmission graph has to stay consistent), not a compatibility shim.

Engine-side edits made so far (both in the probe copy only, both documented):
`'pa'` → `'p'` in the constrained triangulation, and a guard that drops
triangles referencing Steiner vertices before the layout mapping. Neither changes
the propagation model, but they do mean the engine under test is no longer
pristine - a caveat that matters for a cross-check.

## Scene facts (structure only)

* 16 architecture shapes, 4 776 triangles, 904.4 m² structure area
* vertical 311.3 m² (34.4 %) / horizontal 588.3 m² (65.1 %) / non-planar 4.8 m² (0.53 %)
* levels: −0.10, 0.00, 2.65, 2.80 m
* floorplan overlay: [artifacts/front3d_0015_layout_overlay.png](artifacts/front3d_0015_layout_overlay.png)

## Selected links

Wall crossings counted on the SceneGen 5 cm floorplan (furniture ignored), so
neither engine decides which link is "through N walls"
([artifacts/front3d_0015_links.json](artifacts/front3d_0015_links.json)):

| link | BS | UE | walls | 2D distance |
|---|---|---|---|---|
| 0 | BS_0_0 | UE_0_0 | 0 (LoS) | 3.50 m |
| 1 | BS_0_0 | UE_2_110 | 1 | 5.70 m |
| 2 | BS_0_0 | UE_2_153 | 2 | 6.50 m |
| 3 | BS_0_0 | UE_2_168 | 2 | 6.96 m |
| 4 | BS_2_1 | UE_2_100 | 3 | 6.71 m |

Shared physics: ITU-R P.2040-3 concrete (`a=5.24, b=0, c=0.0462, d=0.7822` →
εr = 5.24, σ = 0.121 S/m at 3.438 GHz), 0.1 m slab (Sionna `DEFAULT_THICKNESS`),
768 subcarriers at 30 kHz spacing, isotropic antennas, no scattering, no
diffraction, 4 interactions.

## Commands

```bash
# structure diagnostics
python inspect_geom.py front3d_0015

# reconstruction (semantic panels; the working path)
python build_lay_semantic.py --scene front3d_0015
python check_layout.py --scene front3d_0015          # overlay vs floorplan
python diagnose_lay.py --lay <lay>                   # planarity audit

# alternatives kept for the record
python build_lay.py --scene front3d_0015             # mesh two-sheet converter
python build_lay_from_mask.py --scene front3d_0015   # floorplan-raster converter

# links
python select_links.py --scene front3d_0015

# PyLayers side (isolated env)
tmp/pylayers_probe/run_py.sh subprojects/pylayers_crosscheck/validate_lay.py --lay <lay> --build
tmp/pylayers_probe/run_py.sh subprojects/pylayers_crosscheck/run_pylayers.py --scene front3d_0015

# Sionna 1x1 baseline (existing CLI, unmodified)
python run_sionna_1x1.py --scene front3d_0015 --gpus 2,3 --execute

# comparison
python compare.py --scene front3d_0015
```

## Environment

PyLayers is 2019 code (last meaningful commit `d9bc71f`, 2019-08-01) with a
2019-era stack, run in a throwaway env at `tmp/pylayers_probe/` (~350 MB, not
committed):

```
uv python install 3.8
uv venv --python 3.8 .venv-pylayers
uv pip install numpy==1.19.5 scipy==1.5.4 networkx==2.3 shapely==1.7.1 \
               matplotlib==3.3.4 h5py==3.1.0 pandas==1.3.5 seaborn==0.11.2 \
               tqdm configobj descartes triangle==20200424
```

Sources are fetched per file from GitHub (git pack transfer fails here with
HTTP 408). `shims/sitecustomize.py` stubs GIS/plotting modules that PyLayers
imports at module-import time but never uses on an indoor cartesian path
(`basemap`, `shapefile`, `gdal`, `pyproj`, `mayavi`, `rtree`, …); every stub is
announced on stderr and real modules win when present.

## Metrics contract (once both sides exist)

Computed by the same code from both 768-point CFRs: resolvable path count,
first-path excess delay, PDP shape correlation, CFR amplitude correlation,
complex coherence, RMS delay spread, band-mean path gain (`compare.py`).
Suggested acceptance: first-path delay difference < 1 ns, LoS gain difference
< 1 dB, PDP correlation > 0.8, CFR amplitude correlation > 0.7.
