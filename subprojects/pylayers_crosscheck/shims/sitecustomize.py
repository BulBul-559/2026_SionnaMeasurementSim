"""PyLayers import shims for the cross-check probe environment.

The PyLayers master branch imports a long tail of GIS/optional dependencies at
*module import* time even though an indoor cartesian layout + DLink evaluation
never touches them. This probe host has very slow PyPI throughput, so instead of
installing the whole 2019-era stack we register permissive stand-ins for the
modules that are provably unused by this subproject's code path.

Rules:
* The real module always wins if it is importable.
* Only whitelisted names are stubbed (GIS/OSM/optional tooling).
* Every stub is announced on stderr so a run is never silently degraded.
* Accessing a stubbed attribute returns a placeholder that raises when called,
  so an accidental real use fails loudly instead of returning nonsense.

Loaded via `sitecustomize.py` on PYTHONPATH (see run_py.sh).
"""

import importlib
import sys
import types

WHITELIST = (
    'mpl_toolkits.basemap',
    'shapefile',
    'osmapi',
    'simplekml',
    'geocoder',
    'ratelim',
    'dominate',
    'graphviz',
    'gdal',
    'osgeo',
    'osgeo.gdal',
    'pyproj',
    'rtree',
    'imposm',
    'graph_tool',
    'pygsp',
    'mayavi',
    'mayavi.mlab',
    'tvtk',
    'tvtk.api',
    'pyface',
    'traits',
    'traits.api',
    'apptools',
    'envisage',
)


class _StubMeta(type):
    pass


def _make_stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__doc__ = f'PyLayers cross-check stub for {name} (never used on this path)'
    mod.is_stub = True

    def _attr(attr):
        if attr.startswith('__'):
            raise AttributeError(attr)

        class _Unavailable:
            is_stub = True

            def __init__(self, *a, **k):
                raise RuntimeError(
                    f'{name}.{attr} is stubbed in the PyLayers cross-check env; '
                    'this code path is not supported here')

        _Unavailable.__name__ = f'{name}.{attr}'
        return _Unavailable

    mod.__getattr__ = _attr
    return mod


def _install():
    for name in WHITELIST:
        if name in sys.modules:
            continue
        try:
            importlib.import_module(name)
            continue
        except Exception:  # noqa: BLE001
            pass
        parts = name.split('.')
        for i in range(1, len(parts)):
            parent = '.'.join(parts[:i])
            if parent not in sys.modules:
                pmod = _make_stub(parent)
                sys.modules[parent] = pmod
                print(f'[pylayers-shim] stub {parent}', file=sys.stderr)
        stub = _make_stub(name)
        sys.modules[name] = stub
        # make `from x.y import Z` work on the parent package too
        if len(parts) > 1:
            setattr(sys.modules['.'.join(parts[:-1])], parts[-1], stub)
        print(f'[pylayers-shim] stub {name}', file=sys.stderr)


_install()
