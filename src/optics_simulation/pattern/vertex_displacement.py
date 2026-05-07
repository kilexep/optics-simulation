"""Pattern-field-to-vertex displacement amount foundation.

Computes per-vertex Gaussian-dimple field values and the
corresponding ``normalized_depth`` (eta in [0, 1]) and
``physical_depth`` (eta * max_depth) **amounts** for a mesh's
surface coordinates ``(u, v)``. **No vertex is moved.** This module
is the amount-only foundation between Gaussian pattern generation
and a future mesh-displacement step (which will require additional
geometric machinery: normal direction policy, self-intersection
checks, wall-thickness checks, manufacturability constraints, and
patterned STL / CAD export).

Research framing
----------------
This project's patterned PET-bottle research targets local focusing
and heat-risk metric reduction (caustic peak C99, Eexceed, Tmax
surrogate). It does not directly claim fire prevention. The
physical mechanism of the dimple pattern is **not** an increase in
reflected light but a change in surface normals on transmission,
which redistributes the transmitted angular spread (BTDF entropy)
and reduces local focusing. The amounts computed here are the
input to that mechanism — the surface normals will only change
once a downstream step actually displaces vertices using these
amounts. **This task does not produce a manufacturable real
PET-bottle pattern.**

Out of scope
------------
Actual vertex displacement, mesh copies, inward/outward normal
direction policy, STL / OBJ / CAD / STEP export, mesh repair,
self-intersection / wall-thickness / manufacturability checks,
real PET STL handling, shell / water volume, automatic medium
tracking, visualization, pattern optimization, residual hotspot
updates, thermal modeling, and config-file wiring are
intentionally not implemented here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from optics_simulation.geometry.surface_coordinates import SurfaceCoordinateMap
from optics_simulation.pattern.gaussian import (
    GaussianDimple,
    GaussianDimplePattern,
    PatternError,
    _validate_dimple,
)


_TOL = 1e-12


@dataclass(frozen=True)
class VertexPatternDisplacement:
    normalized_depth: np.ndarray   # (N,) float64 in [0, 1]
    physical_depth: np.ndarray     # (N,) float64 = normalized_depth * max_depth
    active_mask: np.ndarray        # (N,) bool
    vertex_count: int
    max_depth: float
    coordinate_system: str = "normalized_cylindrical_uv"
    application_mode: str = "amount_only"


def _check_uv_arrays(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if u.ndim != 1 or v.ndim != 1:
        raise PatternError(
            f"u and v must be 1D; got u.ndim={u.ndim}, v.ndim={v.ndim}"
        )
    if u.shape != v.shape:
        raise PatternError(
            f"u and v must have the same shape; got u.shape={u.shape}, "
            f"v.shape={v.shape}"
        )
    if not np.isfinite(u).all():
        raise PatternError("u contains NaN or inf values")
    if not np.isfinite(v).all():
        raise PatternError("v contains NaN or inf values")

    if (u < -_TOL).any() or (u >= 1.0 + _TOL).any():
        bad = int(np.argmax((u < -_TOL) | (u >= 1.0 + _TOL)))
        raise PatternError(
            f"u must be in [0, 1) within tol={_TOL}; "
            f"first violation at index {bad} (u={float(u[bad])})"
        )
    if (v < -_TOL).any() or (v > 1.0 + _TOL).any():
        bad = int(np.argmax((v < -_TOL) | (v > 1.0 + _TOL)))
        raise PatternError(
            f"v must be in [0, 1] within tol={_TOL}; "
            f"first violation at index {bad} (v={float(v[bad])})"
        )

    u_clipped = np.clip(u, 0.0, np.nextafter(1.0, 0.0))
    v_clipped = np.clip(v, 0.0, 1.0)
    return u_clipped, v_clipped


def evaluate_gaussian_pattern_at_points(
    pattern: GaussianDimplePattern,
    u: np.ndarray,
    v: np.ndarray,
    *,
    clip: bool = True,
) -> np.ndarray:
    """Evaluate the summed Gaussian dimple field at scattered (u, v).

    ``u`` distance is circular (wraps around ``u = 0``); ``v``
    distance is linear, matching
    :func:`evaluate_gaussian_dimple_field`. ``clip=True`` clips the
    result to ``[0, 1]``; ``clip=False`` returns the raw sum
    (which can exceed 1 when dimples overlap). Empty
    ``pattern.dimples`` returns an all-zero array of shape ``(N,)``.

    ``u`` and ``v`` must be 1D arrays of equal shape, finite, and
    inside ``[0, 1)`` and ``[0, 1]`` respectively (a tolerance of
    ``1e-12`` is allowed and clipped before evaluation).
    """
    if not isinstance(pattern, GaussianDimplePattern):
        raise PatternError(
            f"pattern must be a GaussianDimplePattern; "
            f"got {type(pattern).__name__}"
        )

    u_arr = np.asarray(u, dtype=float)
    v_arr = np.asarray(v, dtype=float)
    u_clipped, v_clipped = _check_uv_arrays(u_arr, v_arr)
    n = int(u_clipped.shape[0])

    dimples_tuple = tuple(pattern.dimples)
    if len(dimples_tuple) == 0:
        return np.zeros((n,), dtype=float)

    for i, d in enumerate(dimples_tuple):
        if not isinstance(d, GaussianDimple):
            raise PatternError(
                f"pattern.dimples[{i}] must be a GaussianDimple; "
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

    du = np.abs(u_clipped[None, :] - centers_u[:, None])      # (D, N)
    du = np.minimum(du, 1.0 - du)
    dv = np.abs(v_clipped[None, :] - centers_v[:, None])      # (D, N)

    exponent = (du * du) / (2.0 * (sus[:, None] ** 2)) + \
               (dv * dv) / (2.0 * (svs[:, None] ** 2))
    contributions = amps[:, None] * np.exp(-exponent)         # (D, N)
    field = contributions.sum(axis=0)                         # (N,)

    if clip:
        field = np.clip(field, 0.0, 1.0)
    return field


def compute_vertex_displacement_amounts(
    mesh: trimesh.Trimesh,
    surface_map: SurfaceCoordinateMap,
    pattern: GaussianDimplePattern,
    *,
    active_threshold: float = 0.0,
    exclude_v_boundary_epsilon: float = 0.0,
    include_mask: np.ndarray | None = None,
) -> VertexPatternDisplacement:
    """Compute per-vertex Gaussian-pattern depth amounts (no mesh edit).

    For every vertex of ``mesh`` (consumed via ``surface_map``),
    evaluate ``pattern`` to get ``normalized_depth`` (eta in
    ``[0, 1]``) and compute ``physical_depth = normalized_depth *
    pattern.max_depth``. **The mesh is not modified.** ``max_depth``
    is used only as the amount scale; this module never moves a
    vertex.

    Parameters
    ----------
    active_threshold
        Pixels with ``normalized_depth > active_threshold`` are
        marked active. Must be ``>= 0``.
    exclude_v_boundary_epsilon
        Conservative cap / boundary exclusion for the synthetic
        cylinder fixture: when ``> 0``, vertices with
        ``v <= eps`` or ``v >= 1 - eps`` have their
        ``normalized_depth`` zeroed (and are therefore inactive).
        Must satisfy ``0 <= eps < 0.5``. Real STL geometries
        should usually drive cap exclusion via face-group masks
        instead; this parameter is intentionally minimal.
    include_mask
        Optional per-vertex boolean mask of shape
        ``(surface_map.point_count,)``. When supplied, vertices
        with ``include_mask == False`` have their
        ``normalized_depth`` (and therefore ``physical_depth``)
        zeroed out before ``active_mask`` is computed, so
        ``active_mask`` is also ``False`` on those vertices.
        Boundary exclusion via ``exclude_v_boundary_epsilon`` is
        applied independently. The intended synthetic-shell use is
        ``include_mask =
        SyntheticShellVertexMasks.outer_lateral_interior_mask`` so
        that pattern displacement applies only to the outer
        lateral interior vertices of a shell fixture. ``None``
        (default) preserves the prior whole-mesh behavior.

    Raises
    ------
    PatternError
        On invalid inputs (wrong types, empty mesh, point-count
        mismatch, surface_map shape inconsistencies, negative
        ``active_threshold``, ``exclude_v_boundary_epsilon`` out of
        ``[0, 0.5)``, non-finite / out-of-range ``surface_map``
        coordinates, ``include_mask`` shape mismatch, NaN / inf in
        an ``include_mask``, or an ``include_mask`` whose dtype is
        neither bool nor numeric-castable-to-bool).
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise PatternError(
            f"mesh must be a trimesh.Trimesh; got {type(mesh).__name__}"
        )
    if not isinstance(surface_map, SurfaceCoordinateMap):
        raise PatternError(
            f"surface_map must be a SurfaceCoordinateMap; "
            f"got {type(surface_map).__name__}"
        )
    if not isinstance(pattern, GaussianDimplePattern):
        raise PatternError(
            f"pattern must be a GaussianDimplePattern; "
            f"got {type(pattern).__name__}"
        )

    vertex_count = int(len(mesh.vertices))
    if vertex_count == 0:
        raise PatternError("mesh has no vertices")

    if int(surface_map.point_count) != vertex_count:
        raise PatternError(
            f"surface_map.point_count ({surface_map.point_count}) does not "
            f"match len(mesh.vertices) ({vertex_count})"
        )
    if surface_map.u.shape != (vertex_count,) or surface_map.v.shape != (vertex_count,):
        raise PatternError(
            f"surface_map.u and surface_map.v must have shape "
            f"({vertex_count},); got u={surface_map.u.shape}, "
            f"v={surface_map.v.shape}"
        )

    if float(active_threshold) < 0.0:
        raise PatternError(
            f"active_threshold must be >= 0; got {active_threshold}"
        )
    eps = float(exclude_v_boundary_epsilon)
    if not (0.0 <= eps < 0.5):
        raise PatternError(
            f"exclude_v_boundary_epsilon must be in [0, 0.5); got {eps}"
        )

    include_bool: np.ndarray | None = None
    if include_mask is not None:
        try:
            include_arr = np.asarray(include_mask)
        except (TypeError, ValueError) as exc:
            raise PatternError(
                f"include_mask must be convertible to a 1-D array; "
                f"got {type(include_mask).__name__}"
            ) from exc
        if include_arr.shape != (vertex_count,):
            raise PatternError(
                f"include_mask must have shape ({vertex_count},); "
                f"got {include_arr.shape}"
            )
        if include_arr.dtype == bool:
            include_bool = include_arr.astype(bool, copy=True)
        elif np.issubdtype(include_arr.dtype, np.number):
            if not np.isfinite(include_arr).all():
                raise PatternError(
                    "include_mask contains NaN or inf values"
                )
            include_bool = include_arr.astype(bool, copy=True)
        else:
            raise PatternError(
                f"include_mask must be bool or numeric castable to "
                f"bool; got dtype={include_arr.dtype}"
            )

    u_arr = np.asarray(surface_map.u, dtype=float)
    v_arr = np.asarray(surface_map.v, dtype=float)

    normalized = evaluate_gaussian_pattern_at_points(
        pattern, u_arr, v_arr, clip=True
    )

    if eps > 0.0:
        boundary_mask = (v_arr <= eps) | (v_arr >= 1.0 - eps)
        normalized = np.where(boundary_mask, 0.0, normalized)

    if include_bool is not None:
        normalized = np.where(include_bool, normalized, 0.0)

    max_depth = float(pattern.max_depth)
    physical = normalized * max_depth

    active_mask = normalized > float(active_threshold)

    return VertexPatternDisplacement(
        normalized_depth=normalized.astype(float, copy=True),
        physical_depth=physical.astype(float, copy=True),
        active_mask=active_mask.astype(bool, copy=True),
        vertex_count=vertex_count,
        max_depth=max_depth,
    )
