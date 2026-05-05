"""Gaussian dimple pattern foundation.

Samples dimple centers from a :class:`RiskMap` probability
distribution and evaluates a Gaussian dimple depth field
``eta(u, v)`` on a normalized ``(u, v)`` grid. ``u`` is the
circumferential coordinate (circular, distance wraps around
``u = 0``); ``v`` is the height coordinate (linear).

Per CLAUDE.md the physical depth field is

    delta(u, v) = dmax * clip( sum_i A_i * exp(-||r - r_i||^2 / (2 sigma_i^2)), 0, 1 )

This module computes the unit-less ``eta = clip(sum, 0, 1)`` part
and stores ``dmax`` as ``max_depth`` for later physical scaling.
**Mesh displacement is not performed here.**

Out of scope
------------
Poisson disk strict min-distance enforcement, Sobol / Halton
quasi-random sampling, mesh displacement, patterned STL or CAD
export, pattern optimization, residual hotspot updates, thermal
modeling, visualization, physical manufacturing constraints,
config-file wiring, actual PET STL handling, and Open3D backends
are intentionally not implemented here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from optics_simulation.contribution.risk_map import RiskMap


class PatternError(Exception):
    """Raised for invalid pattern-generation inputs."""


@dataclass(frozen=True)
class GaussianDimple:
    center_u: float  # in [0, 1)
    center_v: float  # in [0, 1]
    amplitude: float  # > 0
    sigma_u: float  # > 0
    sigma_v: float  # > 0


@dataclass(frozen=True)
class GaussianDimplePattern:
    dimples: tuple[GaussianDimple, ...]
    depth_field: np.ndarray         # (nv, nu) float64, eta in [0, 1]
    resolution: tuple[int, int]     # (nv, nu)
    max_depth: float                # physical dmax (stored, not applied)
    seed: int | None


def _validate_resolution(resolution: tuple[int, int]) -> tuple[int, int]:
    res = tuple(resolution)
    if len(res) != 2:
        raise PatternError(
            f"resolution must have exactly 2 entries (nv, nu); got {res}"
        )
    nv = int(res[0])
    nu = int(res[1])
    if nv <= 0 or nu <= 0:
        raise PatternError(
            f"resolution entries must be positive integers; "
            f"got (nv, nu)=({nv}, {nu})"
        )
    return nv, nu


def _validate_dimple(d: GaussianDimple, *, index: int) -> None:
    if not (0.0 <= float(d.center_u) < 1.0):
        raise PatternError(
            f"dimples[{index}].center_u must be in [0, 1); "
            f"got {d.center_u}"
        )
    if not (0.0 <= float(d.center_v) <= 1.0):
        raise PatternError(
            f"dimples[{index}].center_v must be in [0, 1]; "
            f"got {d.center_v}"
        )
    if float(d.amplitude) <= 0.0:
        raise PatternError(
            f"dimples[{index}].amplitude must be > 0; got {d.amplitude}"
        )
    if float(d.sigma_u) <= 0.0:
        raise PatternError(
            f"dimples[{index}].sigma_u must be > 0; got {d.sigma_u}"
        )
    if float(d.sigma_v) <= 0.0:
        raise PatternError(
            f"dimples[{index}].sigma_v must be > 0; got {d.sigma_v}"
        )


def _validate_risk_map_shapes(risk_map: RiskMap) -> tuple[int, int]:
    rs = risk_map.risk_map.shape
    ps = risk_map.probability_map.shape
    ms = risk_map.active_mask.shape
    if not (rs == ps == ms):
        raise PatternError(
            f"RiskMap arrays have inconsistent shapes: "
            f"risk_map={rs}, probability_map={ps}, active_mask={ms}"
        )
    if len(rs) != 2:
        raise PatternError(
            f"RiskMap arrays must be 2D; got shape {rs}"
        )
    return int(rs[0]), int(rs[1])


def sample_dimple_centers_from_risk(
    risk_map: RiskMap,
    *,
    count: int,
    seed: int | None = None,
) -> np.ndarray:
    """Sample ``count`` dimple centers from ``risk_map.probability_map``.

    Returns a ``(count, 2)`` float array whose columns are
    ``(center_u, center_v)``. ``count == 0`` returns an empty
    ``(0, 2)`` array without consulting the probability map. For
    ``count > 0`` the probability map must be finite, non-negative,
    and have positive sum; otherwise :class:`PatternError` is
    raised.

    Sampling is with replacement and uses
    :func:`numpy.random.default_rng`. ``seed`` is forwarded as-is
    (``None`` uses OS entropy; a fixed ``int`` is deterministic).
    """
    if not isinstance(risk_map, RiskMap):
        raise PatternError(
            f"risk_map must be a RiskMap; got {type(risk_map).__name__}"
        )

    count_i = int(count)
    if count_i < 0:
        raise PatternError(f"count must be >= 0; got {count_i}")

    if count_i == 0:
        return np.zeros((0, 2), dtype=float)

    nv, nu = _validate_risk_map_shapes(risk_map)

    prob = np.asarray(risk_map.probability_map, dtype=float)
    if not np.isfinite(prob).all():
        raise PatternError("probability_map contains NaN or inf values")
    if (prob < 0.0).any():
        raise PatternError("probability_map contains negative values")

    total = float(prob.sum())
    if total <= 0.0:
        raise PatternError(
            "probability_map has zero total mass; cannot sample with count > 0"
        )

    flat = (prob / total).ravel()
    rng = np.random.default_rng(seed)
    flat_indices = rng.choice(flat.size, size=count_i, replace=True, p=flat)

    rows = (flat_indices // nu).astype(np.int64)
    cols = (flat_indices % nu).astype(np.int64)
    centers_u = (cols.astype(float) + 0.5) / float(nu)
    centers_v = (rows.astype(float) + 0.5) / float(nv)
    return np.stack([centers_u, centers_v], axis=-1)


def evaluate_gaussian_dimple_field(
    dimples: Sequence[GaussianDimple],
    *,
    resolution: tuple[int, int],
    clip: bool = True,
) -> np.ndarray:
    """Evaluate the summed Gaussian dimple field on a pixel-center grid.

    ``u`` distance is circular (wraps around ``u = 0``); ``v``
    distance is linear. The field is sampled at pixel centers
    ``u_j = (j + 0.5)/nu``, ``v_i = (i + 0.5)/nv`` and returned as
    a ``(nv, nu)`` float64 array. ``clip=True`` clips the result
    to ``[0, 1]``; ``clip=False`` returns the raw sum (which can
    exceed 1 when dimples overlap). An empty ``dimples`` sequence
    returns a zero field.
    """
    nv, nu = _validate_resolution(resolution)
    dimples_tuple = tuple(dimples)

    if len(dimples_tuple) == 0:
        return np.zeros((nv, nu), dtype=float)

    for i, d in enumerate(dimples_tuple):
        if not isinstance(d, GaussianDimple):
            raise PatternError(
                f"dimples[{i}] must be a GaussianDimple; "
                f"got {type(d).__name__}"
            )
        _validate_dimple(d, index=i)

    centers_u = np.array(
        [float(d.center_u) for d in dimples_tuple], dtype=float
    )
    centers_v = np.array(
        [float(d.center_v) for d in dimples_tuple], dtype=float
    )
    amps = np.array([float(d.amplitude) for d in dimples_tuple], dtype=float)
    sus = np.array([float(d.sigma_u) for d in dimples_tuple], dtype=float)
    svs = np.array([float(d.sigma_v) for d in dimples_tuple], dtype=float)

    u_axis = (np.arange(nu, dtype=float) + 0.5) / float(nu)  # (nu,)
    v_axis = (np.arange(nv, dtype=float) + 0.5) / float(nv)  # (nv,)

    du = np.abs(u_axis[None, :] - centers_u[:, None])         # (D, nu)
    du = np.minimum(du, 1.0 - du)
    dv = np.abs(v_axis[None, :] - centers_v[:, None])         # (D, nv)

    u_term = (du * du) / (2.0 * (sus[:, None] ** 2))           # (D, nu)
    v_term = (dv * dv) / (2.0 * (svs[:, None] ** 2))           # (D, nv)

    # field[i, j] = sum_d amp[d] * exp(-(u_term[d, j] + v_term[d, i]))
    # Build (D, nv, nu) tensor and sum over axis 0.
    exponent = v_term[:, :, None] + u_term[:, None, :]         # (D, nv, nu)
    contributions = amps[:, None, None] * np.exp(-exponent)
    field = contributions.sum(axis=0)

    if clip:
        field = np.clip(field, 0.0, 1.0)
    return field


def create_gaussian_dimple_pattern(
    risk_map: RiskMap,
    *,
    count: int,
    amplitude: float = 1.0,
    sigma_u: float = 0.02,
    sigma_v: float = 0.02,
    max_depth: float = 0.25,
    seed: int | None = None,
) -> GaussianDimplePattern:
    """Sample dimple centers from ``risk_map`` and build the depth field.

    Validates ``count >= 0`` and ``amplitude``, ``sigma_u``,
    ``sigma_v``, ``max_depth`` all strictly positive. ``count == 0``
    returns an empty pattern with a zero depth field. The depth
    field is always clipped to ``[0, 1]``; ``max_depth`` is stored
    on the result for later physical scaling but is **not** applied
    to any mesh in this task.
    """
    if not isinstance(risk_map, RiskMap):
        raise PatternError(
            f"risk_map must be a RiskMap; got {type(risk_map).__name__}"
        )

    count_i = int(count)
    if count_i < 0:
        raise PatternError(f"count must be >= 0; got {count_i}")

    if float(amplitude) <= 0.0:
        raise PatternError(f"amplitude must be > 0; got {amplitude}")
    if float(sigma_u) <= 0.0:
        raise PatternError(f"sigma_u must be > 0; got {sigma_u}")
    if float(sigma_v) <= 0.0:
        raise PatternError(f"sigma_v must be > 0; got {sigma_v}")
    if float(max_depth) <= 0.0:
        raise PatternError(f"max_depth must be > 0; got {max_depth}")

    nv, nu = _validate_risk_map_shapes(risk_map)

    if count_i == 0:
        return GaussianDimplePattern(
            dimples=(),
            depth_field=np.zeros((nv, nu), dtype=float),
            resolution=(nv, nu),
            max_depth=float(max_depth),
            seed=None if seed is None else int(seed),
        )

    centers = sample_dimple_centers_from_risk(
        risk_map, count=count_i, seed=seed
    )
    dimples = tuple(
        GaussianDimple(
            center_u=float(centers[i, 0]),
            center_v=float(centers[i, 1]),
            amplitude=float(amplitude),
            sigma_u=float(sigma_u),
            sigma_v=float(sigma_v),
        )
        for i in range(centers.shape[0])
    )
    depth_field = evaluate_gaussian_dimple_field(
        dimples, resolution=(nv, nu), clip=True
    )
    return GaussianDimplePattern(
        dimples=dimples,
        depth_field=depth_field,
        resolution=(nv, nu),
        max_depth=float(max_depth),
        seed=None if seed is None else int(seed),
    )
