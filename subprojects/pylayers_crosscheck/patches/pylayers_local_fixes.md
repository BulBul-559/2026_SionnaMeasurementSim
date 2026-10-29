# Local PyLayers fixes applied in the probe env (not in the repo under test)

The probe env keeps an unmodified-pyLayers copy except for the two edits below,
both in `pylayers/gis/layout.py`. They are recorded here so the cross-check is
reproducible and so it is explicit that the engine under test was touched.

## 1. Constrained triangulation options: `'pa'` -> `'p'`

```diff
-        T = triangle.triangulate(C, 'pa')
+        T = triangle.triangulate(C, 'p')
```
(lines 5963 and 6099 in the 2019 master checkout, functions `_triangle_old` and
`_triangle`)

Why: PyLayers passes the area option `a`, then asserts that the triangulation
introduced **no** extra vertices (`geomutil.setvnodes_new`: `assert 0 not in
npts`). The `a` option can insert Steiner points, so the engine contradicts
itself. `a` only refines triangle quality; it adds no geometry and no material,
so dropping it does not change the propagation model.

## 2. Guard the triangle->layout vertex mapping

```diff
+        _ok = np.all(tri < len(map_vertices), axis=1)
+        if not _ok.all():
+            tri = tri[_ok]
         E0 = map_vertices[tri[:, 1:]]
```
(in `buildGt`)

Why: when the triangulation does return more vertices than the layout has nodes,
the mapping raises `IndexError: index N is out of bounds`. Triangles touching
such vertices are refinement artefacts; dropping them keeps the transmission
graph consistent.

## Effect and limits

With both fixes, `Layout.build()` gets past the triangulation and the vertex
mapping. It then fails in the "Mikado" cycle-merge step (`geomutil.Polygon.__add__`
raising `MultiPolygon are not allowed`) because merging two cycles across a
transparent segment does not always yield a simple polygon for this geometry.
That step has not been patched: a correct fix must keep the transmission graph
consistent, which is a real porting task rather than a local compatibility edit.
