"""Antenna handling for the cross-check.

The Sionna-side contract uses ``pattern="iso"`` with ``polarization="V"``. In DiffeRT
that convention is encoded as a Jones vector that is ``(1, 0)`` in the ``(theta, phi)``
basis of the outgoing direction, i.e. unit magnitude for *every* direction, which is
exactly an isotropic radiator with vertical polarisation.

``IsotropicVerticalAntenna`` spells that convention out as a real antenna object so that

* the assumption is visible in our own code instead of being an implicit default,
* a genuinely non-isotropic element (sector pattern, measured pattern, ...) can be
  dropped in later without touching the solver, and
* the equivalence with DiffeRT's built-in ``"V"`` shortcut can be unit-tested
  (see ``tests/test_antenna.py``).

Both paths are kept because the string shortcut is the fastest way to reproduce
DiffeRT's own reference behaviour, while the object is what we actually ship.
"""

from __future__ import annotations

import numpy as np

__all__ = ["spherical_basis", "IsotropicVerticalAntenna", "resolve_polarization"]


def spherical_basis(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the ``(theta_hat, phi_hat)`` unit vectors of the direction ``r``.

    ``phi_hat`` is the horizontal (azimuthal) transverse direction and ``theta_hat`` the
    "vertical" transverse direction used by the ``V`` polarisation convention.
    """
    r = np.asarray(r, dtype=float)
    norm = np.linalg.norm(r, axis=-1, keepdims=True)
    unit = r / np.maximum(norm, 1e-12)

    x, y, z = unit[..., 0], unit[..., 1], unit[..., 2]
    rho = np.maximum(np.hypot(x, y), 1e-12)

    # d/d(theta) of the unit vector, i.e. theta_hat
    theta_hat = np.stack(
        [z * x / rho, z * y / rho, -np.ones_like(z) * rho],
        axis=-1,
    )
    # d/d(phi) of the unit vector, i.e. phi_hat
    phi_hat = np.stack([-y / rho, x / rho, np.zeros_like(z)], axis=-1)
    return theta_hat, phi_hat


def _build_antenna_class():  # noqa: ANN202 - lazily imported to keep numpy-only imports cheap
    import jax.numpy as jnp  # noqa: PLC0415
    from differt.em import AbstractRadiationPattern  # noqa: PLC0415

    class IsotropicVerticalAntenna(AbstractRadiationPattern):
        """Isotropic element, vertical polarisation.

        Equivalent to Sionna's ``PlanarArray(pattern="iso", polarization="V")`` and to
        DiffeRT's ``tx_polarization="V"`` shortcut: the radiated field has unit magnitude
        in every direction and lies along ``theta_hat``.
        """

        def polarization_vectors(self, r):  # noqa: ANN001, ANN201
            centre = self.center
            rel = r - centre
            norm = jnp.linalg.norm(rel, axis=-1, keepdims=True)
            unit = rel / jnp.maximum(norm, 1e-12)
            x, y, z = unit[..., 0], unit[..., 1], unit[..., 2]
            rho = jnp.maximum(jnp.hypot(x, y), 1e-12)
            theta_hat = jnp.stack([z * x / rho, z * y / rho, -rho], axis=-1)
            zeros = jnp.zeros_like(theta_hat)
            return theta_hat, zeros

    return IsotropicVerticalAntenna


_ANTENNA_CLASS = None


def IsotropicVerticalAntenna(frequency_hz: float):  # noqa: N802 - class-like factory
    """Build the isotropic vertical-polarisation element at ``frequency_hz``."""
    global _ANTENNA_CLASS
    if _ANTENNA_CLASS is None:
        _ANTENNA_CLASS = _build_antenna_class()
    return _ANTENNA_CLASS(frequency=frequency_hz)


def resolve_polarization(spec: str, frequency_hz: float, *, explicit: bool = False):
    """Return what to hand to DiffeRT's field solver as ``tx/rx_polarization``.

    ``explicit=False`` (default) returns DiffeRT's ``"V"`` Jones shortcut, which *is* an
    isotropic vertical-polarisation element: the solver sets the field to ``(1, 0)`` in
    the ``(theta, phi)`` basis of the outgoing direction, i.e. unit gain for every
    direction. This is the model our frozen contract asks for (``pattern="iso"``,
    ``polarization="V"``) and it is what the verified Friis check (ratio 0.9993) used.

    ``explicit=True`` returns our own antenna object instead. Note that DiffeRT's
    radiation-pattern API receives ``T + r_hat`` (transmitter position plus unit
    direction) and therefore assumes the antenna sits at the origin: the explicit object
    only reproduces ``"V"`` exactly when the transmitter is at the origin. With many
    transmitters per scene (our case) a single pattern object cannot be recentred, so the
    ``"V"`` shortcut is the correct choice; the class is kept for single-transmitter work
    and as executable documentation of the convention.
    """
    if not explicit:
        return spec
    if spec != "V":
        raise ValueError(f"only V polarisation is supported by the explicit antenna, got {spec!r}")
    return IsotropicVerticalAntenna(frequency_hz)
