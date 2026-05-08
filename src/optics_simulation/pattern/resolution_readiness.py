"""Pattern-resolution readiness diagnostic for actual STL meshes.

Actual STL pattern-resolution readiness diagnostic; **not a
physical PET-bottle validation**. Earlier risk-guided pattern
sweeps showed that on a low-resolution PET STL (~580 vertices,
~1156 faces, body mask ~80 vertices) every Gaussian dimple
candidate produced a "mixed" tradeoff under default guardrails.
That outcome can be caused either by genuinely unfavorable
pattern parameters or by the underlying mesh being too coarse to
represent a smooth indentation in the requested ``sigma_u`` /
``sigma_v`` / ``max_depth`` range.

This module provides a *diagnostic* answer to "is this mesh dense
enough for vertex-displacement patterning at these parameters?"
without running optical or thermal simulation, without modifying
the mesh, and without claiming manufacturability.

Two pieces:

- :func:`compute_pattern_resolution_readiness` reads a
  :class:`trimesh.Trimesh` and a vertex-level boolean
  ``include_mask`` (typically the actual-STL body-region mask) and
  computes edge-length statistics over selected edges, optional
  ``(u, v)`` nearest-neighbor spacing among selected vertices, and
  a coarse ``readiness_label`` over the requested pattern
  parameters.
- :func:`create_subdivided_mesh_copy_for_patterning` returns a
  subdivided copy via ``trimesh.Trimesh.subdivide`` for diagnostic
  re-evaluation. Subdivision is a synthetic numerical refinement,
  not a validated manufacturing surface.

Limitations
-----------
- The ``readiness_label`` is a coarse rule-of-thumb classifier
  using fixed thresholds; it is **not** a manufacturability score.
- ``estimated_uv_spacing_median`` uses an O(N^2) all-pairs nearest
  neighbor over selected vertices in the unit ``(u, v)`` plane;
  this is acceptable for the diagnostic body-mask sizes we
  expect (tens to a few thousand vertices).
- Subdivision uses ``trimesh.Trimesh.subdivide``; it does not
  smooth, repair, or otherwise reshape the mesh beyond the
  subdivided face split.
- This module does **not** repair meshes, infer material regions,
  or validate manufacturability.
- This is **not** fire-prevention validation, **not** PET-bottle
  safety, and **not** a calibrated physical model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import trimesh

from optics_simulation.geometry.surface_coordinates import (
    SurfaceCoordinateMap,
)
from optics_simulation.pattern.gaussian import PatternError


_VALID_READINESS_LABELS = (
    "pattern_resolution_adequate",
    "pattern_resolution_marginal",
    "pattern_resolution_insufficient",
)


@dataclass(frozen=True)
class PatternResolutionReadinessReport:
    vertex_count: int
    selected_vertex_count: int
    selected_fraction: float
    face_count: int
    selected_edge_count: int
    edge_length_min: float
    edge_length_median: float
    edge_length_mean: float
    edge_length_p90: float
    edge_length_max: float
    estimated_uv_spacing_median: float | None
    sigma_u: float
    sigma_v: float
    max_depth: float
    sigma_u_to_uv_spacing_ratio: float | None
    sigma_v_to_uv_spacing_ratio: float | None
    max_depth_to_edge_median_ratio: float
    min_vertices_required: int
    readiness_label: str
    warnings: tuple[str, ...]
    report_type: str = "pattern_resolution_readiness_diagnostic"


@dataclass(frozen=True)
class MeshSubdivisionReport:
    original_vertex_count: int
    original_face_count: int
    subdivided_vertex_count: int
    subdivided_face_count: int
    iterations: int
    process: bool
    report_type: str = "mesh_subdivision_diagnostic"


def _check_positive_float(value: float, *, name: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise PatternError(
            f"{name} must be a finite float > 0; got {value!r}"
        ) from exc
    if not math.isfinite(v) or v <= 0.0:
        raise PatternError(
            f"{name} must be a finite float > 0; got {v}"
        )
    return v


def _check_positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PatternError(
            f"{name} must be a positive int; got {value!r}"
        )
    if value <= 0:
        raise PatternError(
            f"{name} must be a positive int; got {value}"
        )
    return int(value)


def _validate_include_mask(
    mesh: trimesh.Trimesh, mask,
) -> np.ndarray:
    n = int(len(mesh.vertices))
    arr = np.asarray(mask)
    if arr.shape != (n,):
        raise PatternError(
            f"include_mask must have shape ({n},); got {arr.shape}"
        )
    if arr.dtype == bool:
        return arr.astype(bool, copy=True)
    if not np.issubdtype(arr.dtype, np.number):
        raise PatternError(
            "include_mask must be bool or numeric castable to "
            f"bool; got dtype={arr.dtype}"
        )
    if not np.isfinite(arr).all():
        raise PatternError(
            "include_mask contains NaN or inf values"
        )
    return arr.astype(bool, copy=True)


def _circular_u_diff(
    u_a: np.ndarray, u_b: np.ndarray,
) -> np.ndarray:
    raw = np.abs(u_a - u_b)
    return np.minimum(raw, 1.0 - raw)


def _estimate_uv_spacing_median(
    surface_map: SurfaceCoordinateMap,
    selected_indices: np.ndarray,
) -> float | None:
    """Median nearest-neighbor (u, v) spacing over selected vertices.

    O(N^2) all-pairs in the unit (u, v) plane with circular ``u``
    wraparound. Returns ``None`` when fewer than two selected
    vertices exist.
    """
    n_sel = int(selected_indices.size)
    if n_sel < 2:
        return None

    u_all = np.asarray(surface_map.u, dtype=float)
    v_all = np.asarray(surface_map.v, dtype=float)
    u_sel = u_all[selected_indices]
    v_sel = v_all[selected_indices]

    nearest = np.empty(n_sel, dtype=float)
    for i in range(n_sel):
        du = _circular_u_diff(u_sel[i], u_sel)
        dv = v_sel[i] - v_sel
        d = np.sqrt(du * du + dv * dv)
        d[i] = np.inf
        nearest[i] = float(d.min())

    finite = nearest[np.isfinite(nearest)]
    if finite.size == 0:
        return None
    return float(np.median(finite))


def _classify_readiness(
    *,
    selected_vertex_count: int,
    selected_edge_count: int,
    sigma_to_spacing_min: float | None,
    max_depth_to_edge_ratio_value: float,
    min_vertices_required: int,
    min_sigma_to_spacing_ratio: float,
    max_depth_to_edge_ratio: float,
) -> tuple[str, list[str]]:
    warnings: list[str] = []

    if selected_edge_count == 0:
        warnings.append(
            "no selected edges (mask selects fewer than two "
            "connected vertices); pattern displacement cannot be "
            "represented"
        )
        return "pattern_resolution_insufficient", warnings

    if selected_vertex_count < int(min_vertices_required):
        warnings.append(
            f"selected_vertex_count {selected_vertex_count} "
            f"< min_vertices_required {int(min_vertices_required)}"
        )

    if sigma_to_spacing_min is not None:
        if sigma_to_spacing_min < float(min_sigma_to_spacing_ratio):
            warnings.append(
                f"min(sigma_u, sigma_v) / uv_spacing_median "
                f"{sigma_to_spacing_min:.4f} < required ratio "
                f"{float(min_sigma_to_spacing_ratio):.4f}"
            )

    if (
        float(max_depth_to_edge_ratio_value)
        > float(max_depth_to_edge_ratio)
    ):
        warnings.append(
            f"max_depth / edge_length_median "
            f"{float(max_depth_to_edge_ratio_value):.4f} > limit "
            f"{float(max_depth_to_edge_ratio):.4f}"
        )

    fail_count = 0
    if selected_vertex_count < int(min_vertices_required):
        fail_count += 1
    if (
        sigma_to_spacing_min is not None
        and sigma_to_spacing_min < float(min_sigma_to_spacing_ratio)
    ):
        fail_count += 1
    if (
        float(max_depth_to_edge_ratio_value)
        > float(max_depth_to_edge_ratio)
    ):
        fail_count += 1

    if fail_count >= 2:
        return "pattern_resolution_insufficient", warnings
    if fail_count == 1:
        return "pattern_resolution_marginal", warnings
    return "pattern_resolution_adequate", warnings


def compute_pattern_resolution_readiness(
    mesh: trimesh.Trimesh,
    *,
    include_mask: np.ndarray,
    surface_map: SurfaceCoordinateMap | None = None,
    sigma_u: float = 0.03,
    sigma_v: float = 0.03,
    max_depth: float = 0.05,
    min_vertices_required: int = 200,
    min_sigma_to_spacing_ratio: float = 2.0,
    max_depth_to_edge_ratio: float = 0.25,
) -> PatternResolutionReadinessReport:
    """Compute a pattern-resolution readiness diagnostic for a mesh.

    Actual STL pattern-resolution readiness diagnostic; **not a
    physical PET-bottle validation**. Reads ``mesh`` and the
    vertex-level boolean ``include_mask`` (typically a body-region
    mask) and computes:

    - Edge-length statistics over edges whose **both** endpoints
      are in the include mask. Edges with only one endpoint inside
      the mask are excluded so the statistics describe the local
      surface that would actually carry pattern displacement.
    - Optional ``(u, v)`` nearest-neighbor spacing median over
      selected vertices when ``surface_map`` is provided. ``u`` is
      treated as circular in ``[0, 1)``.
    - ``sigma_u_to_uv_spacing_ratio`` /
      ``sigma_v_to_uv_spacing_ratio`` against the spacing median.
    - ``max_depth_to_edge_median_ratio = max_depth /
      edge_length_median``.

    The ``readiness_label`` is a coarse rule-of-thumb classifier:

    - ``"pattern_resolution_insufficient"`` when no selected
      edges exist, or when two or more of the warning rules fire.
    - ``"pattern_resolution_marginal"`` when exactly one warning
      rule fires.
    - ``"pattern_resolution_adequate"`` otherwise.

    Warning rules:

    - ``selected_vertex_count < min_vertices_required``.
    - ``min(sigma_u, sigma_v) / uv_spacing_median <
      min_sigma_to_spacing_ratio`` (only when ``surface_map`` is
      provided).
    - ``max_depth / edge_length_median > max_depth_to_edge_ratio``.

    The classifier is a diagnostic; ``"pattern_resolution_adequate"``
    is **not** a manufacturability claim. The function does **not**
    mutate the mesh and does **not** write files.

    Raises
    ------
    PatternError
        On invalid mesh type, mismatched ``include_mask`` shape,
        non-finite mask values, non-positive ``sigma_u`` /
        ``sigma_v`` / ``max_depth``, or non-positive thresholds.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise PatternError(
            f"mesh must be a trimesh.Trimesh; "
            f"got {type(mesh).__name__}"
        )

    mask = _validate_include_mask(mesh, include_mask)

    sigma_u_v = _check_positive_float(sigma_u, name="sigma_u")
    sigma_v_v = _check_positive_float(sigma_v, name="sigma_v")
    max_depth_v = _check_positive_float(max_depth, name="max_depth")
    _check_positive_int(
        int(min_vertices_required),
        name="min_vertices_required",
    )
    _check_positive_float(
        float(min_sigma_to_spacing_ratio),
        name="min_sigma_to_spacing_ratio",
    )
    _check_positive_float(
        float(max_depth_to_edge_ratio),
        name="max_depth_to_edge_ratio",
    )

    vertex_count = int(len(mesh.vertices))
    face_count = int(len(mesh.faces))
    selected_vertex_count = int(mask.sum())
    selected_fraction = (
        float(selected_vertex_count) / float(vertex_count)
        if vertex_count > 0 else 0.0
    )

    vertices = np.asarray(mesh.vertices, dtype=float)
    edges = np.asarray(mesh.edges_unique, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[1] != 2 or edges.shape[0] == 0:
        edge_lengths_sel = np.empty(0, dtype=float)
    else:
        both_in = mask[edges[:, 0]] & mask[edges[:, 1]]
        sel_edges = edges[both_in]
        if sel_edges.shape[0] == 0:
            edge_lengths_sel = np.empty(0, dtype=float)
        else:
            edge_lengths_sel = np.linalg.norm(
                vertices[sel_edges[:, 0]]
                - vertices[sel_edges[:, 1]],
                axis=1,
            )

    selected_edge_count = int(edge_lengths_sel.size)
    if selected_edge_count > 0:
        edge_min = float(edge_lengths_sel.min())
        edge_median = float(np.median(edge_lengths_sel))
        edge_mean = float(edge_lengths_sel.mean())
        edge_p90 = float(np.percentile(edge_lengths_sel, 90.0))
        edge_max = float(edge_lengths_sel.max())
    else:
        edge_min = 0.0
        edge_median = 0.0
        edge_mean = 0.0
        edge_p90 = 0.0
        edge_max = 0.0

    selected_indices = np.flatnonzero(mask).astype(np.int64, copy=True)
    if surface_map is None:
        uv_spacing_median: float | None = None
    else:
        if not isinstance(surface_map, SurfaceCoordinateMap):
            raise PatternError(
                "surface_map must be a SurfaceCoordinateMap or None; "
                f"got {type(surface_map).__name__}"
            )
        if int(surface_map.point_count) != vertex_count:
            raise PatternError(
                f"surface_map.point_count "
                f"({int(surface_map.point_count)}) does not match "
                f"mesh vertex count ({vertex_count})"
            )
        uv_spacing_median = _estimate_uv_spacing_median(
            surface_map, selected_indices,
        )

    if uv_spacing_median is None or uv_spacing_median <= 0.0:
        sigma_u_ratio: float | None = None
        sigma_v_ratio: float | None = None
        sigma_to_spacing_min: float | None = None
    else:
        sigma_u_ratio = float(sigma_u_v) / float(uv_spacing_median)
        sigma_v_ratio = float(sigma_v_v) / float(uv_spacing_median)
        sigma_to_spacing_min = float(
            min(sigma_u_ratio, sigma_v_ratio)
        )

    if edge_median > 0.0:
        max_depth_to_edge = float(max_depth_v) / float(edge_median)
    else:
        max_depth_to_edge = float("inf")

    label, warnings = _classify_readiness(
        selected_vertex_count=selected_vertex_count,
        selected_edge_count=selected_edge_count,
        sigma_to_spacing_min=sigma_to_spacing_min,
        max_depth_to_edge_ratio_value=max_depth_to_edge,
        min_vertices_required=int(min_vertices_required),
        min_sigma_to_spacing_ratio=float(
            min_sigma_to_spacing_ratio
        ),
        max_depth_to_edge_ratio=float(max_depth_to_edge_ratio),
    )

    return PatternResolutionReadinessReport(
        vertex_count=vertex_count,
        selected_vertex_count=selected_vertex_count,
        selected_fraction=selected_fraction,
        face_count=face_count,
        selected_edge_count=selected_edge_count,
        edge_length_min=edge_min,
        edge_length_median=edge_median,
        edge_length_mean=edge_mean,
        edge_length_p90=edge_p90,
        edge_length_max=edge_max,
        estimated_uv_spacing_median=uv_spacing_median,
        sigma_u=float(sigma_u_v),
        sigma_v=float(sigma_v_v),
        max_depth=float(max_depth_v),
        sigma_u_to_uv_spacing_ratio=sigma_u_ratio,
        sigma_v_to_uv_spacing_ratio=sigma_v_ratio,
        max_depth_to_edge_median_ratio=float(max_depth_to_edge),
        min_vertices_required=int(min_vertices_required),
        readiness_label=label,
        warnings=tuple(warnings),
    )


def create_subdivided_mesh_copy_for_patterning(
    mesh: trimesh.Trimesh,
    *,
    iterations: int = 1,
    process: bool = False,
) -> tuple[trimesh.Trimesh, MeshSubdivisionReport]:
    """Subdivide a mesh ``iterations`` times and return a fresh copy.

    Actual STL pattern-resolution readiness diagnostic; **not a
    physical PET-bottle validation**. Each iteration calls
    :meth:`trimesh.Trimesh.subdivide` (loop subdivision-style face
    split). Subdivision is a synthetic numerical refinement for
    diagnostics, **not** mesh repair, smoothing, optimization, or
    a validated manufacturing surface. The original ``mesh`` is
    read but never mutated; the returned copy is independent.

    Raises
    ------
    PatternError
        On invalid mesh type or non-positive ``iterations``.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise PatternError(
            f"mesh must be a trimesh.Trimesh; "
            f"got {type(mesh).__name__}"
        )
    iters = _check_positive_int(int(iterations), name="iterations")

    original_vertex_count = int(len(mesh.vertices))
    original_face_count = int(len(mesh.faces))

    work = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices, dtype=float).copy(),
        faces=np.asarray(mesh.faces, dtype=np.int64).copy(),
        process=bool(process),
    )
    for _ in range(iters):
        work = work.subdivide()

    subdivided = trimesh.Trimesh(
        vertices=np.asarray(work.vertices, dtype=float).copy(),
        faces=np.asarray(work.faces, dtype=np.int64).copy(),
        process=bool(process),
    )

    report = MeshSubdivisionReport(
        original_vertex_count=original_vertex_count,
        original_face_count=original_face_count,
        subdivided_vertex_count=int(len(subdivided.vertices)),
        subdivided_face_count=int(len(subdivided.faces)),
        iterations=iters,
        process=bool(process),
    )
    return subdivided, report
