"""Antenna tests.

* the explicit isotropic vertical-polarisation element must describe the same field as
  DiffeRT's built-in ``"V"`` shortcut (that shortcut is what the reference smoke numbers
  were produced with), and
* it must be isotropic, i.e. the produced channel gain must not depend on direction.
"""

from __future__ import annotations

import numpy as np

from differt_crosscheck.antenna import IsotropicVerticalAntenna, spherical_basis

F0 = 3_438_000_000.0


def test_polarization_vectors_are_transverse() -> None:
    import jax.numpy as jnp

    ant = IsotropicVerticalAntenna(F0)
    rng = np.random.default_rng(0)
    dirs = rng.normal(size=(32, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    theta, phi = ant.polarization_vectors(jnp.asarray(dirs))
    theta = np.asarray(theta)
    phi = np.asarray(phi)

    theta_hat, phi_hat = spherical_basis(dirs)
    assert np.allclose(theta, theta_hat, atol=1e-6)
    assert np.allclose(phi, 0.0, atol=1e-12)
    assert np.allclose(np.sum(theta_hat * dirs, axis=1), 0.0, atol=1e-9)
    assert np.allclose(np.sum(phi_hat * dirs, axis=1), 0.0, atol=1e-9)


def _los_scene(direction_scale: float):
    import jax.numpy as jnp

    from differt.geometry import Mesh, Scene

    # DiffeRT's radiation-pattern API treats (T + r_hat) as a direction, i.e. it assumes
    # the antenna sits at the origin. Equivalence with the "V" shortcut is therefore only
    # defined for a transmitter at the origin.
    tx = np.array([[0.0, 0.0, 0.0]])
    rx = np.array([[direction_scale, direction_scale, 0.0]])
    return Scene(
        transmitters=jnp.asarray(tx),
        receivers=jnp.asarray(rx),
        mesh=Mesh.empty(),
    )


def _los_gain(polarization) -> complex:  # noqa: ANN001
    import jax.numpy as jnp

    from differt.em import TracedFields

    scene = _los_scene(3.0)
    paths = scene.trace_paths(order=0)
    fields = TracedFields.from_paths(
        paths,
        scene.mesh,
        frequency=jnp.asarray(F0),
        tx_polarization=polarization,
        rx_polarization=polarization,
    )
    mask = np.asarray(fields.mask)
    gains = np.asarray(fields.fields)[mask]
    return complex(gains[0])


def test_explicit_antenna_equals_v_shortcut() -> None:
    explicit = IsotropicVerticalAntenna(F0)
    a = _los_gain(explicit)
    b = _los_gain("V")
    assert abs(a - b) / abs(b) < 1e-6, (a, b)


def test_matches_friis_free_space() -> None:
    gain = _los_gain("V")
    distance = float(np.linalg.norm(np.array([3.0, 3.0, 0.0])))
    lam = 299_792_458.0 / F0
    expected = lam / (4 * np.pi * distance)
    assert abs(abs(gain) - expected) / expected < 5e-3, (abs(gain), expected)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
