"""DiffeRT based independent ray-tracing cross-check for the Front3D 339 CFR set.

This package is deliberately standalone: it does not import anything from
``sionna_measurement_sim`` and is not wired into the existing simulation system.
It only *reads* Sionna artefacts (scene XML, labels, dataset HDF5) for comparison.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
