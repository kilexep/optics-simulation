"""Actual-STL normal-orientation-aware patterned mesh copy.

Actual STL patterned optical-to-thermal risk smoke check; **not a
physical PET-bottle validation**. Sibling of
:func:`create_displaced_mesh_copy` for **actual STL meshes** whose
vertex normals may point inward instead of outward. Picks the sign
that moves the **moved** vertices radially toward the bounding-box
xy axis (so the indentation is geometrically inward regardless of
the STL's normal convention) and writes the result to a fresh
:class:`trimesh.Trimesh`. The original mesh is read but never
mutated.

Limitations
-----------
- Normal-direction handling is geometric robustness, not physical
  validation.
- The choice between ``-normals`` and ``+normals`` is driven by a
  median radial distance over the moved vertices; ties prefer
  ``-normals``.
- ``create_displaced_mesh_copy`` is intentionally **not modified**;
  it remains the synthetic outward-normal convention.
- This module does not repair meshes, infer material regions, or
  validate manufacturability.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from optics_simulation.pattern.gaussian import PatternError
from optics_simulation.pattern.vertex_displacement import (
    VertexPatternDisplacement,
)


_NORMAL_NORM_TOL = 1e-12


@dataclass(frozen=True)
class ActualSTLPatternedMeshResult:
    patterned_mesh: trimesh.Trimesh
    displacement_vectors: np.ndarray
    displacement_magnitudes: np.ndarray
    moved_mask: np.ndarray
    original_vertex_count: int
    moved_vertex_count: int
    max_displacement: float
    mean_displacement: float
    selected_offset_sign: float
    original_radial_stat: float
    candidate_minus_radial_stat: float
    candidate_plus_radial_stat: float
    selected_radial_stat: float
    inward_displacement_detected: bool
    mode: str = "actual_stl_inward_auto"


def create_actual_stl_patterned_mesh_copy(
    mesh: trimesh.Trimesh,
    displacement: VertexPatternDisplacement,
    *,
    center_xy: tuple[float, float] | None = None,
    process: bool = False,
) -> ActualSTLPatternedMeshResult:
    """Build a patterned copy of an actual STL with auto-direction selection.

    Actual STL patterned optical-to-thermal risk smoke check;
    **not a physical PET-bottle validation**. Selects the
    displacement sign (``-normals`` or ``+normals``) whose median
    radial distance over the **moved vertices** is smaller, then
    applies that sign to ``displacement.physical_depth`` on the
    ``displacement.active_mask`` subset. Vertices outside the
    active mask or with zero physical depth are unchanged. Vertex
    normals are read from ``mesh.vertex_normals`` and renormalized
    to unit length internally.

    Parameters
    ----------
    mesh
        Input :class:`trimesh.Trimesh`.
    displacement
        :class:`VertexPatternDisplacement` whose ``vertex_count``
        matches ``mesh``.
    center_xy
        Optional ``(cx, cy)`` axis center for the radial
        statistic. Defaults to ``mesh.bounds`` midpoint.
    process
        Forwarded to the new :class:`trimesh.Trimesh`. Default
        ``False`` so trimesh does not silently merge or alter.

    Raises
    ------
    PatternError
        On non-Trimesh / empty input, vertex-count mismatch,
        non-finite or shape-mismatched ``physical_depth`` /
        ``active_mask`` / ``mesh.vertex_normals``, or near-zero
        normal length.
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

    n = int(len(mesh.vertices))
    if n == 0:
        raise PatternError("mesh has no vertices")
    if int(displacement.vertex_count) != n:
        raise PatternError(
            f"displacement.vertex_count "
            f"({int(displacement.vertex_count)}) does not match "
            f"len(mesh.vertices) ({n})"
        )

    physical_depth = np.asarray(
        displacement.physical_depth, dtype=float,
    )
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

    vertices = np.asarray(mesh.vertices, dtype=float)
    if not np.isfinite(vertices).all():
        raise PatternError(
            "mesh.vertices contain NaN or inf values"
        )
    normals = np.asarray(mesh.vertex_normals, dtype=float)
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
            f"mesh.vertex_normals has near-zero norm at index "
            f"{bad} (norm={float(norms[bad])})"
        )
    unit_normals = normals / norms[:, None]

    effective_depth = np.where(active_mask, physical_depth, 0.0)
    moves = effective_depth > 0.0

    if center_xy is None:
        bounds = np.asarray(mesh.bounds, dtype=float)
        cx = float(0.5 * (bounds[0, 0] + bounds[1, 0]))
        cy = float(0.5 * (bounds[0, 1] + bounds[1, 1]))
    else:
        cx = float(center_xy[0])
        cy = float(center_xy[1])

    if bool(moves.any()):
        moved_idx = np.flatnonzero(moves)
        moved_v = vertices[moved_idx]
        moved_n = unit_normals[moved_idx]
        moved_d = effective_depth[moved_idx]
        candidate_minus = moved_v - moved_n * moved_d[:, None]
        candidate_plus = moved_v + moved_n * moved_d[:, None]

        def _radial_median(arr: np.ndarray) -> float:
            r = np.sqrt(
                (arr[:, 0] - cx) ** 2 + (arr[:, 1] - cy) ** 2
            )
            return float(np.median(r))

        original_stat = _radial_median(moved_v)
        minus_stat = _radial_median(candidate_minus)
        plus_stat = _radial_median(candidate_plus)

        if minus_stat <= plus_stat:
            selected_sign = -1.0
            selected_stat = minus_stat
        else:
            selected_sign = +1.0
            selected_stat = plus_stat
    else:
        original_stat = 0.0
        minus_stat = 0.0
        plus_stat = 0.0
        selected_sign = -1.0
        selected_stat = 0.0

    inward_detected = bool(selected_stat < original_stat)

    displacement_vectors = (
        (selected_sign * unit_normals) * effective_depth[:, None]
    )
    new_vertices = vertices + displacement_vectors
    displacement_magnitudes = np.linalg.norm(
        displacement_vectors, axis=1,
    )

    faces = np.asarray(mesh.faces).copy()
    patterned_mesh = trimesh.Trimesh(
        vertices=new_vertices,
        faces=faces,
        process=bool(process),
    )

    return ActualSTLPatternedMeshResult(
        patterned_mesh=patterned_mesh,
        displacement_vectors=displacement_vectors.astype(
            float, copy=True,
        ),
        displacement_magnitudes=displacement_magnitudes.astype(
            float, copy=True,
        ),
        moved_mask=moves.astype(bool, copy=True),
        original_vertex_count=n,
        moved_vertex_count=int(moves.sum()),
        max_displacement=float(displacement_magnitudes.max()),
        mean_displacement=float(displacement_magnitudes.mean()),
        selected_offset_sign=float(selected_sign),
        original_radial_stat=float(original_stat),
        candidate_minus_radial_stat=float(minus_stat),
        candidate_plus_radial_stat=float(plus_stat),
        selected_radial_stat=float(selected_stat),
        inward_displacement_detected=inward_detected,
    )
