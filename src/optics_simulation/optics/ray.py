"""Ray sources and ray bundle representation.

A :class:`RayBundle` is a pair of dense ``(N, 3)`` arrays — origins and
unit-norm directions. Construction goes through
:func:`make_ray_bundle`, which validates shape and normalizes
directions; downstream intersection / optics code assumes directions
are already unit length.

Limitations
-----------
- Only parallel ray grids are provided here. Cone, fan, and spectrum-
  weighted sources are intentionally out of scope for the baseline
  intersection scaffold.
- Zero-length direction vectors raise :class:`OpticsError`. There is
  no silent fallback because a zero direction is meaningless for ray
  tracing and would propagate ``nan`` through every later stage.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


class OpticsError(Exception):
    """Base class for optics package errors (ray construction, intersection)."""


@dataclass(frozen=True)
class RayBundle:
    origins: np.ndarray
    directions: np.ndarray
    ray_count: int


def _validate_n3(arr: np.ndarray, name: str) -> np.ndarray:
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise OpticsError(f"{name} must have shape (N, 3); got {arr.shape}")
    return arr


def make_ray_bundle(
    origins: np.ndarray | Sequence[Sequence[float]],
    directions: np.ndarray | Sequence[Sequence[float]],
) -> RayBundle:
    """Build a RayBundle. Directions are normalized to unit length.

    Raises :class:`OpticsError` if shapes mismatch or any direction
    has zero length.
    """
    o = np.ascontiguousarray(np.asarray(origins, dtype=float))
    d = np.ascontiguousarray(np.asarray(directions, dtype=float))
    o = _validate_n3(o, "origins")
    d = _validate_n3(d, "directions")
    if o.shape != d.shape:
        raise OpticsError(
            f"origins {o.shape} and directions {d.shape} must have the same shape"
        )

    norms = np.linalg.norm(d, axis=1)
    if np.any(norms == 0.0):
        bad = int(np.argmax(norms == 0.0))
        raise OpticsError(
            f"direction at index {bad} has zero length; cannot normalize"
        )
    d_unit = d / norms[:, None]

    return RayBundle(origins=o, directions=d_unit, ray_count=int(o.shape[0]))


def parallel_ray_grid(
    origin_plane_z: float,
    direction: tuple[float, float, float],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    nx: int,
    ny: int,
) -> RayBundle:
    """Regular grid of parallel rays starting on the plane ``z = origin_plane_z``.

    Useful as a baseline solar-like source. ``direction`` is a single
    3-vector applied to every ray and is normalized internally.
    """
    if nx < 1 or ny < 1:
        raise OpticsError(f"nx and ny must be >= 1; got nx={nx}, ny={ny}")

    xs = np.linspace(x_range[0], x_range[1], nx)
    ys = np.linspace(y_range[0], y_range[1], ny)
    xv, yv = np.meshgrid(xs, ys, indexing="xy")
    origins = np.column_stack(
        [xv.ravel(), yv.ravel(), np.full(xv.size, float(origin_plane_z))]
    )

    d = np.asarray(direction, dtype=float).reshape(-1)
    if d.size != 3:
        raise OpticsError(f"direction must have 3 components; got {d.size}")
    n = float(np.linalg.norm(d))
    if n == 0.0:
        raise OpticsError("direction has zero length; cannot normalize")
    d_unit = d / n
    directions = np.broadcast_to(d_unit, origins.shape).copy()

    return make_ray_bundle(origins, directions)
