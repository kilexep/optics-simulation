"""First-hit ray-mesh intersection scaffold.

Computes the first intersection of each ray in a :class:`RayBundle`
against a ``trimesh.Trimesh`` and returns dense per-ray arrays. Misses
are marked with ``inf`` / ``nan`` / ``-1`` so result rows always align
with the input ray index.

Backend
-------
This implementation uses ``trimesh.ray.RayMeshIntersector``. Result
field names (``t_hit``, ``primitive_ids``, ``hit_points``,
``hit_normals``) follow Open3D ``RaycastingScene.cast_rays`` so the
backend can be swapped later without changing call sites.

Out of scope
------------
Snell refraction, Fresnel reflection, multiple-bounce tracing,
per-ray energy / power, and detector accumulation are all deferred.
This module only answers "where does ray i first hit the mesh, and
what is the surface normal there".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from optics_simulation.optics.ray import OpticsError, RayBundle


@dataclass(frozen=True)
class IntersectionResult:
    t_hit: np.ndarray         # shape (N,), float, np.inf where miss
    hit_mask: np.ndarray      # shape (N,), bool
    hit_points: np.ndarray    # shape (N, 3), np.nan where miss
    hit_normals: np.ndarray   # shape (N, 3), np.nan where miss
    primitive_ids: np.ndarray  # shape (N,), int64, -1 where miss
    ray_count: int


def intersect_rays(
    mesh: trimesh.Trimesh,
    rays: RayBundle,
) -> IntersectionResult:
    """Return first-hit intersection arrays aligned with ``rays`` index.

    The output arrays all have leading dimension ``rays.ray_count``.
    Rows corresponding to rays that miss the mesh are filled with
    sentinels (``inf`` / ``nan`` / ``-1``).
    """
    if not isinstance(rays, RayBundle):
        raise OpticsError(
            f"rays must be a RayBundle; got {type(rays).__name__}"
        )

    n = rays.ray_count
    t_hit = np.full(n, np.inf, dtype=float)
    hit_mask = np.zeros(n, dtype=bool)
    hit_points = np.full((n, 3), np.nan, dtype=float)
    hit_normals = np.full((n, 3), np.nan, dtype=float)
    primitive_ids = np.full(n, -1, dtype=np.int64)

    if n == 0:
        return IntersectionResult(
            t_hit=t_hit,
            hit_mask=hit_mask,
            hit_points=hit_points,
            hit_normals=hit_normals,
            primitive_ids=primitive_ids,
            ray_count=0,
        )

    locations, ray_indices, face_indices = mesh.ray.intersects_location(
        ray_origins=rays.origins,
        ray_directions=rays.directions,
        multiple_hits=False,
    )

    if len(ray_indices) > 0:
        ray_indices = np.asarray(ray_indices, dtype=np.int64)
        face_indices = np.asarray(face_indices, dtype=np.int64)
        locations = np.asarray(locations, dtype=float)

        deltas = locations - rays.origins[ray_indices]
        t_values = np.einsum("ij,ij->i", deltas, rays.directions[ray_indices])

        hit_mask[ray_indices] = True
        t_hit[ray_indices] = t_values
        hit_points[ray_indices] = locations
        primitive_ids[ray_indices] = face_indices

        face_normals = np.asarray(mesh.face_normals, dtype=float)
        hit_normals[ray_indices] = face_normals[face_indices]

    return IntersectionResult(
        t_hit=t_hit,
        hit_mask=hit_mask,
        hit_points=hit_points,
        hit_normals=hit_normals,
        primitive_ids=primitive_ids,
        ray_count=n,
    )
