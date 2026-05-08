"""Legacy-style target-dimension scaling and inner-offset mesh helpers.

Legacy-style STL optical smoke check; **not a physical PET-bottle
validation**. Reproduces the geometric / scaling structure used by
the old interactive PET-bottle experiment inside the current
testable optics framework. Two pure helpers are exposed:

- :func:`compute_target_dimension_scale` and
  :func:`create_target_scaled_mesh_copy` apply the legacy
  anisotropic ``(scale_xy, scale_xy, scale_z)`` scaling so that the
  resulting bounding box matches a caller-supplied
  ``target_diameter`` (max of x and y extents) and
  ``target_height`` (z extent).
- :func:`create_inner_offset_mesh_from_vertex_normals` builds a
  legacy-style inner offset mesh by displacing each vertex along
  ``-thickness * vertex_normal`` and sharing the original face
  table; with ``invert=True`` (default) the resulting mesh's
  face winding is flipped via ``mesh.invert()`` so it can act as
  the water boundary.

Limitations
-----------
- The anisotropic scaling reproduces the legacy experiment's
  normalization only; physical STL scaling is a separate
  calibration problem and is **not** addressed here.
- The inner offset is a legacy-style generated inner surface; it
  is **not** guaranteed to represent the true wall thickness of
  any real PET bottle, neck, shoulder, base petaloid, label, cap,
  seam, or manufacturing defect.
- These helpers do **not** repair meshes, do **not** infer
  material regions, do **not** perform optical or thermal
  simulation, and do **not** prove fire prevention or PET-bottle
  safety.
- Input meshes are read but never mutated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from optics_simulation.geometry.mesh_io import GeometryError


@dataclass(frozen=True)
class BottleTargetScaleReport:
    original_bounds_min: np.ndarray
    original_bounds_max: np.ndarray
    scaled_bounds_min: np.ndarray
    scaled_bounds_max: np.ndarray
    target_height: float
    target_diameter: float
    original_height: float
    original_xy_extent: float
    scale_xyz: tuple[float, float, float]
    scaled_height: float
    scaled_xy_extent: float
    report_type: str = "bottle_target_dimension_scale_report"


@dataclass(frozen=True)
class InnerOffsetMeshReport:
    thickness: float
    vertex_count: int
    face_count: int
    source_mesh_watertight: bool
    inner_mesh_watertight: bool
    inverted: bool
    report_type: str = "inner_offset_mesh_report"


def _validate_mesh(mesh: trimesh.Trimesh) -> None:
    if not isinstance(mesh, trimesh.Trimesh):
        raise GeometryError(
            f"mesh must be a trimesh.Trimesh; got {type(mesh).__name__}"
        )
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        raise GeometryError("mesh is empty; cannot scale or offset")


def _validate_finite_positive(value: float, name: str) -> float:
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        raise GeometryError(
            f"{name} must be a finite float > 0; got {v}"
        )
    return v


def compute_target_dimension_scale(
    mesh: trimesh.Trimesh,
    *,
    target_height: float,
    target_diameter: float,
) -> tuple[float, float, float]:
    """Compute the legacy anisotropic scale ``(scale_xy, scale_xy, scale_z)``.

    Legacy-style STL optical smoke check; **not a physical
    PET-bottle validation**. Returns the scale factors used by the
    old experiment so a caller-supplied ``trimesh.Trimesh``'s
    bounding box hits ``target_diameter`` along the larger of the
    x / y extents and ``target_height`` along the z extent::

        x_extent = bounds_max[0] - bounds_min[0]
        y_extent = bounds_max[1] - bounds_min[1]
        z_extent = bounds_max[2] - bounds_min[2]
        scale_xy = target_diameter / max(x_extent, y_extent)
        scale_z  = target_height / z_extent

    Parameters
    ----------
    mesh
        Input :class:`trimesh.Trimesh`. Must be non-empty and have
        finite, non-zero extents along all three axes.
    target_height, target_diameter
        Caller-supplied target dimensions in the same length unit
        as the post-scaling mesh. Must be finite floats ``> 0``.

    Raises
    ------
    GeometryError
        On invalid mesh, zero / non-finite extent, or non-finite /
        non-positive target value.
    """
    _validate_mesh(mesh)
    th = _validate_finite_positive(target_height, "target_height")
    td = _validate_finite_positive(target_diameter, "target_diameter")

    bounds = np.asarray(mesh.bounds, dtype=float)
    if not np.all(np.isfinite(bounds)):
        raise GeometryError("mesh bounds contain non-finite values")

    extents = bounds[1] - bounds[0]
    x_extent = float(extents[0])
    y_extent = float(extents[1])
    z_extent = float(extents[2])
    xy_extent = max(x_extent, y_extent)

    if xy_extent <= 0.0 or not np.isfinite(xy_extent):
        raise GeometryError(
            f"mesh xy extent must be > 0; got max(x,y)={xy_extent}"
        )
    if z_extent <= 0.0 or not np.isfinite(z_extent):
        raise GeometryError(
            f"mesh z extent must be > 0; got {z_extent}"
        )

    scale_xy = td / xy_extent
    scale_z = th / z_extent
    return (float(scale_xy), float(scale_xy), float(scale_z))


def create_target_scaled_mesh_copy(
    mesh: trimesh.Trimesh,
    *,
    target_height: float,
    target_diameter: float,
    process: bool = False,
) -> tuple[trimesh.Trimesh, BottleTargetScaleReport]:
    """Build a legacy target-dimension-scaled copy of ``mesh``.

    Legacy-style STL optical smoke check; **not a physical
    PET-bottle validation**. The input mesh is read but **never
    mutated**. Returns ``(scaled_mesh, report)`` where
    ``scaled_mesh`` is a fresh :class:`trimesh.Trimesh` whose
    vertices have been multiplied component-wise by the
    ``(scale_xy, scale_xy, scale_z)`` factors from
    :func:`compute_target_dimension_scale`. The report records
    both the original and post-scaling bounds so callers can
    confirm the resulting target dimensions.

    The resulting scaling is anisotropic and reproduces the legacy
    experiment's normalization only; it is not a physical-units
    calibration. ``process=False`` is the default so that
    ``trimesh`` does not silently merge vertices, drop degenerate
    faces, or alter winding.

    Raises
    ------
    GeometryError
        On invalid mesh, zero / non-finite extent, or invalid
        target values.
    """
    _validate_mesh(mesh)
    scale_xyz = compute_target_dimension_scale(
        mesh,
        target_height=target_height,
        target_diameter=target_diameter,
    )

    original_vertices = np.asarray(mesh.vertices, dtype=float)
    original_bounds = np.asarray(mesh.bounds, dtype=float)
    scale_arr = np.asarray(scale_xyz, dtype=float).reshape(1, 3)
    scaled_vertices = original_vertices * scale_arr

    scaled_mesh = trimesh.Trimesh(
        vertices=scaled_vertices,
        faces=np.asarray(mesh.faces).copy(),
        process=bool(process),
    )

    scaled_bounds = np.asarray(scaled_mesh.bounds, dtype=float)
    scaled_extents = scaled_bounds[1] - scaled_bounds[0]
    scaled_xy_extent = float(max(scaled_extents[0], scaled_extents[1]))
    scaled_height = float(scaled_extents[2])

    original_extents = original_bounds[1] - original_bounds[0]
    original_xy_extent = float(
        max(original_extents[0], original_extents[1])
    )
    original_height = float(original_extents[2])

    report = BottleTargetScaleReport(
        original_bounds_min=original_bounds[0].astype(
            float, copy=True,
        ),
        original_bounds_max=original_bounds[1].astype(
            float, copy=True,
        ),
        scaled_bounds_min=scaled_bounds[0].astype(float, copy=True),
        scaled_bounds_max=scaled_bounds[1].astype(float, copy=True),
        target_height=float(target_height),
        target_diameter=float(target_diameter),
        original_height=original_height,
        original_xy_extent=original_xy_extent,
        scale_xyz=scale_xyz,
        scaled_height=scaled_height,
        scaled_xy_extent=scaled_xy_extent,
    )
    return scaled_mesh, report


def create_inner_offset_mesh_from_vertex_normals(
    mesh: trimesh.Trimesh,
    *,
    thickness: float,
    invert: bool = True,
    process: bool = False,
) -> tuple[trimesh.Trimesh, InnerOffsetMeshReport]:
    """Generate a legacy-style inner offset mesh from vertex normals.

    Legacy-style STL optical smoke check; **not a physical
    PET-bottle validation**. Reproduces the old experiment's
    "water boundary" mesh by computing::

        vertices_inner = mesh.vertices - thickness * mesh.vertex_normals

    and emitting a fresh :class:`trimesh.Trimesh` that **shares the
    original face table** (a copy is held internally so the input
    mesh is never mutated). When ``invert=True`` (the default)
    ``inner_mesh.invert()`` is called so the face winding is
    flipped and the resulting surface acts as the water-side
    boundary.

    This is a legacy-style generated inner surface; it is **not
    guaranteed to represent true wall thickness** and does **not**
    perform mesh repair, hole filling, normal fixing, or material-
    region inference.

    Parameters
    ----------
    mesh
        Input :class:`trimesh.Trimesh`. Must be non-empty and have
        finite vertex normals.
    thickness
        Strict positive offset distance applied along
        ``-mesh.vertex_normals``. Must be a finite float ``> 0``.
    invert
        If ``True`` (default) the resulting mesh's face winding is
        flipped via ``mesh.invert()`` so the surface points into
        the cavity.
    process
        Forwarded to :class:`trimesh.Trimesh`. Default ``False``
        so that ``trimesh`` does not silently merge vertices,
        drop degenerate faces, or alter winding.

    Raises
    ------
    GeometryError
        On non-Trimesh / empty input, non-finite vertex normals,
        non-finite or non-positive ``thickness``.
    """
    _validate_mesh(mesh)
    th = _validate_finite_positive(thickness, "thickness")

    vertex_normals = np.asarray(mesh.vertex_normals, dtype=float)
    vertices = np.asarray(mesh.vertices, dtype=float)

    if vertex_normals.shape != vertices.shape:
        raise GeometryError(
            f"vertex_normals shape {vertex_normals.shape} does not "
            f"match vertices shape {vertices.shape}"
        )
    if not np.all(np.isfinite(vertex_normals)):
        raise GeometryError(
            "mesh.vertex_normals contain non-finite values; cannot "
            "build inner offset mesh"
        )

    inner_vertices = vertices - th * vertex_normals
    inner_faces = np.asarray(mesh.faces).copy()

    inner_mesh = trimesh.Trimesh(
        vertices=inner_vertices,
        faces=inner_faces,
        process=bool(process),
    )
    if invert:
        inner_mesh.invert()

    report = InnerOffsetMeshReport(
        thickness=float(th),
        vertex_count=int(len(inner_mesh.vertices)),
        face_count=int(len(inner_mesh.faces)),
        source_mesh_watertight=bool(mesh.is_watertight),
        inner_mesh_watertight=bool(inner_mesh.is_watertight),
        inverted=bool(invert),
    )
    return inner_mesh, report
