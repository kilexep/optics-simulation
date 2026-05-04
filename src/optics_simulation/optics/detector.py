"""Detector plane and ray-plane intersection.

Defines a finite rectangular detector in 3D and computes which rays
in a :class:`RayBundle` intersect it within the in-plane bounds.
Output is a dense, input-aligned :class:`DetectorHitResult` whose
miss rows are filled with sentinels (``inf`` / ``NaN``) so callers
can index by the original input ray index.

Convention
----------
- ``normal`` is the plane normal. Caller chooses which side counts
  as "front"; the function only requires ``t > t_min`` along the
  ray to register a hit.
- ``up`` is the caller's preferred in-plane up direction. Any
  component along ``normal`` is projected out and the remainder is
  renormalized to give the detector's in-plane up axis. ``up``
  parallel to ``normal`` raises :class:`OpticsError`.
- ``right = cross(up, normal)`` — yields a right-handed
  ``(right, up, normal)`` basis. With ``normal=+z`` and ``up=+y``
  this gives ``right=+x``, the standard screen orientation.
- Local 2D coordinates: ``local_x = (P - center) · right`` and
  ``local_y = (P - center) · up``. A point counts as a hit when
  ``|local_x| <= width/2`` and ``|local_y| <= height/2``
  (boundaries inclusive).

Out of scope
------------
Pixel binning, irradiance accumulation, C99 / Eexceed / Ahot,
contribution map, ray-history logging, thermal modeling, and
visualization are intentionally not implemented here. This module
is the geometric foundation those layers will compose on top of.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optics_simulation.optics.ray import OpticsError, RayBundle
from optics_simulation.optics.refraction import normalize_vector


_EPS_PARALLEL = 1e-12
_EPS_UP_PROJ = 1e-9


@dataclass(frozen=True)
class DetectorPlane:
    center: np.ndarray
    normal: np.ndarray
    up: np.ndarray
    right: np.ndarray
    width: float
    height: float


@dataclass(frozen=True)
class DetectorHitResult:
    t_hit: np.ndarray         # shape (N,), float, np.inf where miss
    hit_mask: np.ndarray      # shape (N,), bool
    hit_points: np.ndarray    # shape (N, 3), float, NaN where miss
    local_xy: np.ndarray      # shape (N, 2), float, NaN where miss
    ray_count: int


def _as_3vec(v, name: str) -> np.ndarray:
    arr = np.asarray(v, dtype=float).reshape(-1)
    if arr.size != 3:
        raise OpticsError(f"{name} must have 3 components; got size {arr.size}")
    return arr


def create_detector_plane(
    center: tuple[float, float, float],
    normal: tuple[float, float, float],
    up: tuple[float, float, float],
    width: float,
    height: float,
) -> DetectorPlane:
    """Build a :class:`DetectorPlane` from world-space parameters.

    See module docstring for the basis convention. ``up`` is
    projected onto the detector plane and renormalized; passing
    ``up`` parallel to ``normal`` raises :class:`OpticsError`.
    """
    if width <= 0.0 or height <= 0.0:
        raise OpticsError(
            f"detector width and height must be positive; got "
            f"width={width}, height={height}"
        )

    n = normalize_vector(normal)
    up_in = _as_3vec(up, "up")

    up_proj = up_in - float(np.dot(up_in, n)) * n
    up_proj_norm = float(np.linalg.norm(up_proj))
    if up_proj_norm < _EPS_UP_PROJ:
        raise OpticsError("up must not be parallel to normal")
    u = up_proj / up_proj_norm

    r = np.cross(u, n)
    c = _as_3vec(center, "center")

    return DetectorPlane(
        center=c,
        normal=n,
        up=u,
        right=r,
        width=float(width),
        height=float(height),
    )


def intersect_detector_plane(
    rays: RayBundle,
    detector: DetectorPlane,
    *,
    t_min: float = 0.0,
) -> DetectorHitResult:
    """Intersect every ray in ``rays`` with ``detector``.

    Returns a dense, input-aligned result. Rays that are parallel to
    the plane, that meet it at ``t <= t_min``, or whose intersection
    point falls outside the detector's ``width`` x ``height`` bounds
    are reported as misses with sentinel values
    (``t_hit = np.inf``, ``hit_points = NaN``, ``local_xy = NaN``).
    """
    if not isinstance(rays, RayBundle):
        raise OpticsError(
            f"rays must be a RayBundle; got {type(rays).__name__}"
        )
    if not isinstance(detector, DetectorPlane):
        raise OpticsError(
            f"detector must be a DetectorPlane; got {type(detector).__name__}"
        )
    if t_min < 0.0:
        raise OpticsError(f"t_min must be >= 0; got {t_min}")

    n = rays.ray_count
    t_hit = np.full(n, np.inf, dtype=float)
    hit_mask = np.zeros(n, dtype=bool)
    hit_points = np.full((n, 3), np.nan, dtype=float)
    local_xy = np.full((n, 2), np.nan, dtype=float)

    if n == 0:
        return DetectorHitResult(
            t_hit=t_hit,
            hit_mask=hit_mask,
            hit_points=hit_points,
            local_xy=local_xy,
            ray_count=0,
        )

    denom = rays.directions @ detector.normal
    not_parallel = np.abs(denom) >= _EPS_PARALLEL

    t = np.full(n, np.inf, dtype=float)
    if not_parallel.any():
        numer = (detector.center - rays.origins) @ detector.normal
        t[not_parallel] = numer[not_parallel] / denom[not_parallel]

    candidates = not_parallel & (t > t_min)
    if not candidates.any():
        return DetectorHitResult(
            t_hit=t_hit,
            hit_mask=hit_mask,
            hit_points=hit_points,
            local_xy=local_xy,
            ray_count=n,
        )

    cand_idx = np.flatnonzero(candidates)
    cand_origins = rays.origins[cand_idx]
    cand_dirs = rays.directions[cand_idx]
    cand_t = t[cand_idx]
    hp = cand_origins + cand_t[:, None] * cand_dirs

    rel = hp - detector.center
    lx = rel @ detector.right
    ly = rel @ detector.up

    half_w = detector.width / 2.0
    half_h = detector.height / 2.0
    within = (np.abs(lx) <= half_w) & (np.abs(ly) <= half_h)

    if within.any():
        good = cand_idx[within]
        hit_mask[good] = True
        t_hit[good] = cand_t[within]
        hit_points[good] = hp[within]
        local_xy[good, 0] = lx[within]
        local_xy[good, 1] = ly[within]

    return DetectorHitResult(
        t_hit=t_hit,
        hit_mask=hit_mask,
        hit_points=hit_points,
        local_xy=local_xy,
        ray_count=n,
    )
