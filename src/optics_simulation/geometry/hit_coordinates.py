"""Ray-mesh hit point to normalized surface (u, v) mapping.

Bridges :class:`IntersectionResult` (per-ray first-hit data) and
:class:`SurfaceCoordinateMap` (per-vertex normalized cylindrical
coordinates) by computing barycentric weights of each hit point in
its hit triangle and interpolating the triangle's vertex (u, v)
values.

Output is a dense, ``IntersectionResult``-aligned
:class:`HitSurfaceCoordinates`. Miss rows carry ``NaN`` sentinels in
``u``, ``v``, and ``barycentric_weights`` so callers can index by
the original ray index.

u wrap convention
-----------------
``u`` is a circumferential coordinate in ``[0, 1)``. A face whose
three vertices straddle ``u = 0`` (for example
``u = (0.99, 0.01, 0.02)``) cannot be linearly interpolated as-is.
This module uses the following wrap-aware policy:

- Compute the per-face ``u`` range ``u_max - u_min``.
- If the range is greater than ``0.5``, treat the face as crossing
  the wrap boundary. Add ``1.0`` to every vertex ``u < 0.5`` and
  interpolate in the shifted interval, then take ``result % 1.0``.
- Otherwise interpolate ``u`` linearly with the barycentric weights.

This heuristic is correct as long as no face spans more than half
of the cylinder circumference. The synthetic bottle fixture uses
``sections >= 8`` (per-face range <= 0.125), so the policy is safe
for the project's current geometries.

Limitations
-----------
- This module follows :mod:`surface_coordinates` and assumes the
  mesh is z-axis aligned. For top/bottom cap center vertices that
  lie exactly on the axis (``x == cx`` and ``y == cy``), ``u`` is
  mathematically undefined; the convention is ``u = 0`` (numpy
  ``arctan2``). Cap-interior hits inherit this ambiguity through
  barycentric interpolation. Auto-axis alignment, ray-history
  logging, contribution maps, and hotspot selection are out of
  scope for this task.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from optics_simulation.geometry.mesh_io import GeometryError
from optics_simulation.geometry.surface_coordinates import SurfaceCoordinateMap
from optics_simulation.optics.intersection import IntersectionResult


_DEGENERATE_DENOM_EPS = 1e-15


@dataclass(frozen=True)
class HitSurfaceCoordinates:
    u: np.ndarray                    # (N,) float, NaN for miss
    v: np.ndarray                    # (N,) float, NaN for miss
    barycentric_weights: np.ndarray  # (N, 3) float, NaN row for miss
    hit_mask: np.ndarray             # (N,) bool, copied from intersection
    primitive_ids: np.ndarray        # (N,) int64, copied from intersection
    ray_count: int


def _compute_barycentric(
    tri_verts: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """Vectorized barycentric weights of ``points`` in ``tri_verts``.

    ``tri_verts`` has shape ``(H, 3, 3)`` (H triangles, 3 vertices,
    3 coordinates); ``points`` has shape ``(H, 3)``. Returns weights
    of shape ``(H, 3)`` aligned with the columns of ``tri_verts``.
    Raises :class:`GeometryError` if any triangle is degenerate
    (``|denom| < 1e-15``).
    """
    a = tri_verts[:, 0, :]
    b = tri_verts[:, 1, :]
    c = tri_verts[:, 2, :]

    v0 = b - a
    v1 = c - a
    v2 = points - a

    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)

    denom = d00 * d11 - d01 * d01
    if np.any(np.abs(denom) < _DEGENERATE_DENOM_EPS):
        bad = int(np.argmin(np.abs(denom)))
        raise GeometryError(
            f"degenerate triangle encountered at hit index {bad} "
            f"(|denom|={float(np.abs(denom[bad]))})"
        )

    bw = (d11 * d20 - d01 * d21) / denom
    cw = (d00 * d21 - d01 * d20) / denom
    aw = 1.0 - bw - cw
    return np.stack([aw, bw, cw], axis=-1)


def _interpolate_u_with_wrap(
    vu: np.ndarray,
    bary: np.ndarray,
) -> np.ndarray:
    """Wrap-aware barycentric interpolation of vertex u values.

    See module docstring for the policy. ``vu`` and ``bary`` both
    have shape ``(H, 3)``; returns ``(H,)`` in ``[0, 1)``.
    """
    u_max = vu.max(axis=1)
    u_min = vu.min(axis=1)
    wrap_mask = (u_max - u_min) > 0.5

    vu_shifted = np.where(
        wrap_mask[:, None] & (vu < 0.5),
        vu + 1.0,
        vu,
    )
    u_hit = (bary * vu_shifted).sum(axis=1)
    return np.mod(u_hit, 1.0)


def hit_to_surface_coordinates(
    mesh: trimesh.Trimesh,
    intersection: IntersectionResult,
    surface_map: SurfaceCoordinateMap,
) -> HitSurfaceCoordinates:
    """Map each ray-mesh hit to its (u, v) on the mesh surface.

    For every hit row in ``intersection``, looks up the hit face's
    three vertices, computes the barycentric weights of the hit
    point inside that triangle, and barycentric-interpolates the
    per-vertex ``u`` / ``v`` from ``surface_map``. ``u`` is
    interpolated with the wrap-aware policy described in the module
    docstring; ``v`` is plain linear barycentric interpolation. Miss
    rows are filled with ``NaN`` sentinels. Output arrays align with
    ``intersection.ray_count``.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise GeometryError(
            f"mesh must be a trimesh.Trimesh; got {type(mesh).__name__}"
        )
    if not isinstance(intersection, IntersectionResult):
        raise GeometryError(
            f"intersection must be an IntersectionResult; "
            f"got {type(intersection).__name__}"
        )
    if not isinstance(surface_map, SurfaceCoordinateMap):
        raise GeometryError(
            f"surface_map must be a SurfaceCoordinateMap; "
            f"got {type(surface_map).__name__}"
        )

    vertex_count = int(len(mesh.vertices))
    if int(surface_map.point_count) != vertex_count:
        raise GeometryError(
            f"surface_map.point_count ({surface_map.point_count}) does not "
            f"match len(mesh.vertices) ({vertex_count})"
        )

    n = int(intersection.ray_count)
    u_out = np.full(n, np.nan, dtype=float)
    v_out = np.full(n, np.nan, dtype=float)
    bary_out = np.full((n, 3), np.nan, dtype=float)
    hit_mask_copy = np.array(intersection.hit_mask, dtype=bool, copy=True)
    primitive_ids_copy = np.array(
        intersection.primitive_ids, dtype=np.int64, copy=True
    )

    if n == 0:
        return HitSurfaceCoordinates(
            u=u_out,
            v=v_out,
            barycentric_weights=bary_out,
            hit_mask=hit_mask_copy,
            primitive_ids=primitive_ids_copy,
            ray_count=0,
        )

    hit_idx = np.flatnonzero(intersection.hit_mask)
    if hit_idx.size == 0:
        return HitSurfaceCoordinates(
            u=u_out,
            v=v_out,
            barycentric_weights=bary_out,
            hit_mask=hit_mask_copy,
            primitive_ids=primitive_ids_copy,
            ray_count=n,
        )

    face_ids = intersection.primitive_ids[hit_idx]
    face_count = int(len(mesh.faces))
    if face_ids.min() < 0 or face_ids.max() >= face_count:
        raise GeometryError(
            f"primitive_ids contain out-of-range face index "
            f"(min={int(face_ids.min())}, max={int(face_ids.max())}, "
            f"face_count={face_count})"
        )

    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices, dtype=float)

    face_vertex_idx = faces[face_ids]               # (H, 3)
    tri_verts = vertices[face_vertex_idx]           # (H, 3, 3)
    hit_points = np.asarray(intersection.hit_points, dtype=float)[hit_idx]

    bary = _compute_barycentric(tri_verts, hit_points)

    vu = np.asarray(surface_map.u, dtype=float)[face_vertex_idx]   # (H, 3)
    vv = np.asarray(surface_map.v, dtype=float)[face_vertex_idx]   # (H, 3)

    u_hit = _interpolate_u_with_wrap(vu, bary)
    v_hit = (bary * vv).sum(axis=1)

    bary_out[hit_idx] = bary
    u_out[hit_idx] = u_hit
    v_out[hit_idx] = v_hit

    return HitSurfaceCoordinates(
        u=u_out,
        v=v_out,
        barycentric_weights=bary_out,
        hit_mask=hit_mask_copy,
        primitive_ids=primitive_ids_copy,
        ray_count=n,
    )
