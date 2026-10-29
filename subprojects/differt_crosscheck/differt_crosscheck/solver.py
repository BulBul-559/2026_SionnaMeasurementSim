"""Path tracing and CFR synthesis on top of DiffeRT.

DiffeRT does not expose a CFR helper: per path it returns a complex channel gain and a
propagation delay. The CFR in our contract is therefore synthesised as

    H(f) = sum_k  a_k(f) * exp(-j 2 pi f tau_k)

which is exactly how the Sionna side builds its transfer function, so both engines can be
compared link for link. Because the CFR is linear in the path set, contributions of
different path orders are simply accumulated.

Hard-won implementation notes (each one cost a debugging round):

* ``frequency`` may not be batched against ``[tx, rx, path]`` field arrays, so the field
  solver is ``vmap``-ped over frequency instead.
* the mesh must be closed over (static): DiffeRT's diffraction/wedge helpers require
  concrete arrays and crash on traced ones.
* the traced paths must be *arguments*: capturing them as constants produced 7.2 GB of
  JAX constants and unusable compile times.
* requesting several orders in one call (``order=(0, 1, 2, ...)``) pads candidates to
  ``max_num_candidates`` and blew past 1 TB of allocation; one scalar order per call and
  accumulation is both cheap and predictable.
* ``max_num_candidates`` must stay generous: a small cap triggers an internal
  "Invalid indexing in slice" FFI error in the candidate merge.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .config import SolverSpec
from .scene import LinkScene

__all__ = ["CfrResult", "trace_paths", "compute_cfr"]


@dataclass(frozen=True)
class CfrResult:
    """Synthesised channels plus the diagnostics needed to judge them."""

    cfr: np.ndarray  # (num_tx, num_elements, num_freq) complex64
    paths_per_order: dict  # order -> valid path count
    diagnostics: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.cfr.shape)


def _interaction_set(spec: SolverSpec):
    from differt.em import InteractionType  # noqa: PLC0415

    mapping = {
        "reflection": InteractionType.REFLECTION,
        "diffraction": InteractionType.DIFFRACTION,
        "transmission": InteractionType.TRANSMISSION,
        "scattering": InteractionType.SCATTERING,
    }
    return frozenset(mapping[name] for name in spec.interactions())


def trace_paths(link_scene: LinkScene, spec: SolverSpec, order: int):
    """Trace one scalar path order for every (UE, port) pair."""
    allowed = _interaction_set(spec)
    kwargs: dict[str, object] = {
        "order": int(order),
        "solver": spec.tracer,
        "allowed_interactions": allowed,
    }
    if spec.tracer in ("sbr", "hybrid"):
        kwargs["num_rays"] = spec.num_rays
    if spec.tracer == "sbr":
        kwargs["max_num_candidates"] = spec.max_num_candidates

    t0 = time.time()
    paths = link_scene.scene.trace_paths(**kwargs)
    return paths, time.time() - t0


def _fields_vmap(traced_paths, mesh, tx_pol, rx_pol):
    """Frequency-batched field evaluation for one traced path set."""
    import jax  # noqa: PLC0415

    from differt.em import TracedFields  # noqa: PLC0415

    def fields_at(paths_arg, frequency):  # noqa: ANN001
        return TracedFields.from_paths(
            paths_arg,
            mesh,
            frequency=frequency,
            tx_polarization=tx_pol,
            rx_polarization=rx_pol,
        )

    return jax.jit(jax.vmap(fields_at, in_axes=(None, 0)))


def _slice_paths(traced_paths, index):
    """Take a sub-batch (e.g. a single transmitter) of a traced path set."""
    import equinox as eqx  # noqa: PLC0415

    return eqx.tree_at(
        lambda p: (p.vertices, p.objects, p.mask, p.interaction_types),
        traced_paths,
        (
            traced_paths.vertices[index],
            traced_paths.objects[index],
            traced_paths.mask[index],
            traced_paths.interaction_types[index],
        ),
    )


def _prune_paths(traced_paths):
    """Drop path slots that are invalid for *every* (transmitter, port) pair.

    SBR keeps a generous candidate budget (``max_num_candidates``), and the invalid
    slots are still carried through the field solve. Keeping only the slots that are
    valid somewhere typically shrinks the candidate axis by one to two orders of
    magnitude, which is the difference between a two-minute compile and a few seconds.
    """
    import equinox as eqx  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    mask = np.asarray(traced_paths.mask)
    flat = mask.reshape(-1, mask.shape[-1])
    keep = np.flatnonzero(flat.any(axis=0))
    if keep.size == mask.shape[-1]:
        return traced_paths, keep.size
    if keep.size == 0:
        return traced_paths, 0

    idx = jnp.asarray(keep)
    return (
        eqx.tree_at(
            lambda p: (p.vertices, p.objects, p.mask, p.interaction_types),
            traced_paths,
            (
                jnp.take(traced_paths.vertices, idx, axis=traced_paths.vertices.ndim - 3),
                jnp.take(traced_paths.objects, idx, axis=traced_paths.objects.ndim - 2),
                jnp.take(traced_paths.mask, idx, axis=traced_paths.mask.ndim - 1),
                jnp.take(
                    traced_paths.interaction_types,
                    idx,
                    axis=traced_paths.interaction_types.ndim - 2,
                ),
            ),
        ),
        int(keep.size),
    )


def _cfr_for_paths(
    traced_paths,
    mesh,
    frequencies_hz: np.ndarray,
    tx_pol,
    rx_pol,
    *,
    freq_chunk: int,
    num_tx: int,
    num_el: int,
    tx_chunk: int = 1,
) -> tuple[np.ndarray, int]:
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415

    mask_np = np.asarray(traced_paths.mask)
    num_valid = int(mask_np.sum())
    cfr = np.zeros((num_tx, num_el, frequencies_hz.size), dtype=np.complex64)
    if num_valid == 0:
        return cfr, 0

    traced_paths, kept_slots = _prune_paths(traced_paths)
    batched = _fields_vmap(traced_paths, mesh, tx_pol, rx_pol)

    # Evaluate transmitter-by-transmitter: the XLA graph of a 10-UE x 32-port x 4-bounce
    # unrolled field solve reached ~86 GB RSS and had to be killed. Slicing the traced
    # paths keeps the compiled program small and identical across transmitters, so it is
    # compiled once and reused.
    for t_start in range(0, num_tx, tx_chunk):
        t_stop = min(t_start + tx_chunk, num_tx)
        sub_paths = _slice_paths(traced_paths, slice(t_start, t_stop))

        for start in range(0, frequencies_hz.size, freq_chunk):
            stop = min(start + freq_chunk, frequencies_hz.size)
            f_chunk = jnp.asarray(frequencies_hz[start:stop])
            out = batched(sub_paths, f_chunk)
            jax.block_until_ready(out.fields)

            mask = np.asarray(out.mask)
            gain = np.asarray(out.fields)
            delay = np.asarray(out.delay)
            omega = -2j * np.pi * frequencies_hz[start:stop][:, None, None, None]
            cfr[t_start:t_stop, :, start:stop] = np.sum(
                np.where(mask, gain * np.exp(omega * delay), 0.0),
                axis=-1,
            ).transpose(1, 2, 0).astype(np.complex64)
    return cfr, num_valid


def compute_cfr(
    link_scene: LinkScene,
    frequencies_hz: np.ndarray,
    spec: SolverSpec,
    antenna_spec,
    *,
    tx_polarization=None,
    rx_polarization=None,
    freq_chunk: int = 128,
    tx_chunk: int = 2,
) -> CfrResult:
    """Trace every requested order, then accumulate the CFR."""
    frequencies_hz = np.asarray(frequencies_hz, dtype=float)
    mesh = link_scene.scene.mesh
    tx_pol = tx_polarization if tx_polarization is not None else antenna_spec[0]
    rx_pol = rx_polarization if rx_polarization is not None else antenna_spec[1]

    num_tx = link_scene.num_tx
    num_el = link_scene.num_elements
    cfr = np.zeros((num_tx, num_el, frequencies_hz.size), dtype=np.complex64)

    paths_per_order: dict[int, int] = {}
    trace_seconds: dict[int, float] = {}
    fields_seconds = 0.0
    candidates: dict[int, int] = {}

    for order in spec.orders():
        traced, t_trace = trace_paths(link_scene, spec, order)
        candidates[order] = int(np.asarray(traced.mask).shape[-1])
        trace_seconds[order] = t_trace

        t0 = time.time()
        part, num_valid = _cfr_for_paths(
            traced,
            mesh,
            frequencies_hz,
            tx_pol,
            rx_pol,
            freq_chunk=freq_chunk,
            num_tx=num_tx,
            num_el=num_el,
            tx_chunk=tx_chunk,
        )
        fields_seconds += time.time() - t0
        paths_per_order[order] = num_valid
        cfr += part
        print(
            f"    [order {order}] candidates={candidates[order]} valid={num_valid} "
            f"trace={t_trace:.1f}s fields={time.time() - t0:.1f}s",
            flush=True,
        )

    diagnostics = {
        "tracer": spec.tracer,
        "interactions": sorted(spec.interactions()),
        "orders": list(spec.orders()),
        "paths_per_order": paths_per_order,
        "candidates_per_order": candidates,
        "trace_seconds_per_order": trace_seconds,
        "trace_seconds_total": float(sum(trace_seconds.values())),
        "fields_seconds_total": fields_seconds,
        "num_valid_paths_total": int(sum(paths_per_order.values())),
        "freq_chunk": freq_chunk,
        "tx_chunk": tx_chunk,
        "num_rays": spec.num_rays,
        "max_num_candidates": spec.max_num_candidates,
    }
    return CfrResult(cfr=cfr, paths_per_order=paths_per_order, diagnostics=diagnostics)
