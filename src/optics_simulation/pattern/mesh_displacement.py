"""Synthetic mesh-copy displacement foundation.

Research framing
----------------
This module is the synthetic mesh-copy displacement foundation for
the project's local focusing / heat-risk metric reduction research
(caustic peak ``C99``, ``Eexceed``, ``Tmax`` surrogate). The
caustic-reducing pattern mechanism here is **not** an increase in
reflected light: it is a change in the surface normals that
redistributes the *transmitted* angular spread (BTDF entropy) and
thereby reduces local focusing on a downstream absorber surface.
This module supplies the geometric step where per-vertex amounts
produced by :func:`compute_vertex_displacement_amounts` are applied
to a *copy* of the mesh, so that the original mesh stays untouched.

The default direction is ``mode="inward"`` (along the negative
vertex normal), matching the indentation / dimple model used in the
research. Outward displacement is provided for symmetry but is not
the caustic-reducing direction. The result is **not a manufacturable
real PET-bottle pattern**: this is a synthetic fixture-level
computation. STL / OBJ / CAD / STEP export, mesh repair,
self-intersection solving, shell or wall-thickness modeling,
medium tracking, and manufacturability validation are intentionally
not implemented here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from optics_simulation.pattern.gaussian import PatternError
from optics_simulation.pattern.vertex_displacement import (
    VertexPatternDisplacement,
)


_VALID_MODES = ("inward", "outward")
_NORMAL_NORM_TOL = 1e-12


@dataclass(frozen=True)
class DisplacedMeshResult:
    """Output of :func:`create_displaced_mesh_copy`.

    Attributes
    ----------
    original_vertex_count
        ``len(mesh.vertices)`` of the input mesh.
    displaced_mesh
        A new :class:`trimesh.Trimesh` with displaced vertices and
        the original faces preserved.
    displacement_vectors
        Per-vertex displacement vectors with shape ``(N, 3)``.
        Inactive vertices have a zero row.
    displacement_magnitudes
        Per-vertex Euclidean norms of ``displacement_vectors`` with
        shape ``(N,)``. Inactive vertices have value ``0.0``.
    moved_mask
        Boolean mask of vertices with strictly positive effective
        depth, shape ``(N,)``. Equivalent to
        ``effective_depth > 0`` where ``effective_depth = physical_depth``
        on active vertices and ``0`` on inactive vertices.
    mode
        ``"inward"`` (vertices moved opposite the vertex normal) or
        ``"outward"`` (vertices moved along the vertex normal).
    max_displacement
        ``displacement_magnitudes.max()``.
    mean_displacement
        Mean of ``displacement_magnitudes`` over **all** vertices,
        including inactive vertices whose magnitude is ``0``.
    """
    original_vertex_count: int
    displaced_mesh: trimesh.Trimesh
    displacement_vectors: np.ndarray
    displacement_magnitudes: np.ndarray
    moved_mask: np.ndarray
    mode: str
    max_displacement: float
    mean_displacement: float


def create_displaced_mesh_copy(
    mesh: trimesh.Trimesh,
    displacement: VertexPatternDisplacement,
    *,
    mode: str = "inward",
    process: bool = False,
) -> DisplacedMeshResult:
    """Build a displaced copy of ``mesh`` driven by per-vertex amounts.

    The original ``mesh`` is **not** mutated: its ``vertices`` and
    ``faces`` arrays are read but never written. A new
    :class:`trimesh.Trimesh` is returned with vertices shifted along
    (``mode="outward"``) or against (``mode="inward"``) the original
    mesh's vertex normals by ``displacement.physical_depth``. Inactive
    vertices (``displacement.active_mask == False``) are not moved.

    Convention
    ----------
    ``mesh.vertex_normals`` are assumed to be **outward**-pointing
    (the convention used by :mod:`trimesh` for closed meshes with
    consistent winding, including the synthetic subdivided cylinder
    fixtures in :mod:`optics_simulation.geometry.synthetic_bottle`).
    Therefore:

    - ``mode="inward"``:
      ``new_vertices = vertices - normals * effective_depth[:, None]``
      (default; indentation / dimple direction).
    - ``mode="outward"``:
      ``new_vertices = vertices + normals * effective_depth[:, None]``
      (provided for symmetry; not the caustic-reducing direction).

    Parameters
    ----------
    mesh
        Input :class:`trimesh.Trimesh`. Must have at least one
        vertex.
    displacement
        :class:`VertexPatternDisplacement` whose ``vertex_count`` and
        per-vertex array shapes match ``mesh``.
    mode
        ``"inward"`` (default) or ``"outward"``.
    process
        Forwarded to the new :class:`trimesh.Trimesh`. Default
        ``False`` so trimesh does not silently merge vertices, drop
        degenerate faces, or otherwise alter topology.

    Raises
    ------
    PatternError
        On invalid types, invalid ``mode``, empty mesh, vertex-count
        mismatch, malformed ``physical_depth`` / ``active_mask``
        arrays (wrong shape, NaN / inf, negative depth, non-bool
        mask), malformed ``mesh.vertex_normals`` (wrong shape, non
        finite, or any near-zero norm).
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise PatternError(
            f"mesh must be a trimesh.Trimesh; got {type(mesh).__name__}"
        )
    if not isinstance(displacement, VertexPatternDisplacement):
        raise PatternError(
            f"displacement must be a VertexPatternDisplacement; "
            f"got {type(displacement).__name__}"
        )
    if mode not in _VALID_MODES:
        raise PatternError(
            f"mode must be one of {_VALID_MODES}; got {mode!r}"
        )

    n = int(len(mesh.vertices))
    if n == 0:
        raise PatternError("mesh has no vertices")
    if int(displacement.vertex_count) != n:
        raise PatternError(
            f"displacement.vertex_count ({int(displacement.vertex_count)}) "
            f"does not match len(mesh.vertices) ({n})"
        )

    physical_depth = np.asarray(displacement.physical_depth, dtype=float)
    if physical_depth.shape != (n,):
        raise PatternError(
            f"displacement.physical_depth must have shape ({n},); "
            f"got {physical_depth.shape}"
        )
    if not np.isfinite(physical_depth).all():
        raise PatternError(
            "displacement.physical_depth contains NaN or inf values"
        )
    if (physical_depth < 0.0).any():
        bad = int(np.argmin(physical_depth))
        raise PatternError(
            f"displacement.physical_depth must be non-negative; "
            f"first violation at index {bad} "
            f"(value={float(physical_depth[bad])})"
        )

    active_mask_raw = np.asarray(displacement.active_mask)
    if active_mask_raw.dtype != bool:
        raise PatternError(
            f"displacement.active_mask must be bool-typed; "
            f"got dtype={active_mask_raw.dtype}"
        )
    if active_mask_raw.shape != (n,):
        raise PatternError(
            f"displacement.active_mask must have shape ({n},); "
            f"got {active_mask_raw.shape}"
        )
    active_mask = active_mask_raw.astype(bool, copy=True)

    vertices = np.asarray(mesh.vertices, dtype=float).copy()
    normals = np.asarray(mesh.vertex_normals, dtype=float).copy()
    if normals.shape != (n, 3):
        raise PatternError(
            f"mesh.vertex_normals must have shape ({n}, 3); "
            f"got {normals.shape}"
        )
    if not np.isfinite(normals).all():
        raise PatternError(
            "mesh.vertex_normals contain NaN or inf values"
        )
    norms = np.linalg.norm(normals, axis=1)
    if (norms < _NORMAL_NORM_TOL).any():
        bad = int(np.argmin(norms))
        raise PatternError(
            f"mesh.vertex_normals has near-zero norm at index {bad} "
            f"(norm={float(norms[bad])}); cannot determine "
            f"displacement direction"
        )
    unit_normals = normals / norms[:, None]

    effective_depth = np.where(active_mask, physical_depth, 0.0)
    sign = -1.0 if mode == "inward" else +1.0
    displacement_vectors = (sign * unit_normals) * effective_depth[:, None]
    new_vertices = vertices + displacement_vectors

    displacement_magnitudes = np.linalg.norm(displacement_vectors, axis=1)
    moved_mask = effective_depth > 0.0

    faces = np.asarray(mesh.faces).copy()
    displaced_mesh = trimesh.Trimesh(
        vertices=new_vertices,
        faces=faces,
        process=bool(process),
    )

    return DisplacedMeshResult(
        original_vertex_count=n,
        displaced_mesh=displaced_mesh,
        displacement_vectors=displacement_vectors.astype(float, copy=True),
        displacement_magnitudes=displacement_magnitudes.astype(
            float, copy=True
        ),
        moved_mask=moved_mask.astype(bool, copy=True),
        mode=mode,
        max_displacement=float(displacement_magnitudes.max()),
        mean_displacement=float(displacement_magnitudes.mean()),
    )
