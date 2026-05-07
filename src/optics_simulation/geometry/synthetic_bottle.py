"""Synthetic PET-like bottle mesh fixtures.

Three synthetic fixtures for development and pipeline
compatibility smoke checks:

- :func:`create_synthetic_bottle_body` — minimal
  :func:`trimesh.creation.cylinder` wrapper. Lateral vertices land
  only on the top and bottom rings, so ``surface_map.v`` is always
  exactly 0 or 1. Suitable for ray-trace / angle-scan smoke checks
  where vertex distribution along z does not matter.
- :func:`create_subdivided_synthetic_bottle_body` — same outer
  geometry, but with ``height_segments + 1`` z-rings on the side
  surface so that ``surface_map.v`` takes interior values
  ``j / height_segments`` for ``j = 0..height_segments``. Suitable
  for vertex-displacement-amount smoke checks that need patterns
  to actually intersect non-boundary vertices.
- :func:`create_subdivided_synthetic_bottle_shell` — closed hollow
  cylindrical shell with finite ``wall_thickness`` (outer lateral
  surface, inner lateral surface, top annular rim, bottom annular
  rim). Suitable for synthetic empty-shell / water-filled-shell
  side-incidence smoke checks where the optical path is a 4
  interface ``air -> PET -> cavity -> PET -> air`` sequence.

Synthetic shell/fill-state optical-to-thermal smoke check; **not
a physical PET-bottle validation**. The shell is a synthetic
cylindrical shell fixture, **not a real PET bottle**. It
approximates wall thickness and fill-state interfaces only, and
does **not** model neck, shoulder, base petaloid geometry,
labels, caps, seams, or manufacturing defects. None of these
fixtures load STL files, run boolean operations, depend on an
external solid-modeling backend, or read config files.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from optics_simulation.geometry.mesh_io import GeometryError


_MIN_SECTIONS = 8
_MIN_HEIGHT_SEGMENTS = 1


@dataclass(frozen=True)
class SyntheticShellVertexMasks:
    outer_wall_mask: np.ndarray
    inner_wall_mask: np.ndarray
    z_boundary_mask: np.ndarray
    outer_lateral_interior_mask: np.ndarray
    inner_lateral_interior_mask: np.ndarray
    vertex_count: int
    outer_radius: float
    inner_radius: float
    height: float
    boundary_epsilon: float


def create_synthetic_bottle_body(
    *,
    radius: float = 30.0,
    height: float = 120.0,
    sections: int = 96,
) -> trimesh.Trimesh:
    """Build a simple z-axis aligned solid cylinder bottle-like fixture.

    Returns a watertight :class:`trimesh.Trimesh` centered near the
    origin with z bounds ``[-height/2, +height/2]`` and a circular
    cross-section of radius ``radius`` approximated by ``sections``
    flat panels.

    This is a synthetic bottle-like fixture, **not a real PET bottle
    STL**. It does not model wall thickness, neck, shoulder, base
    curvature, or water volume. Units are caller-defined; the
    function is compatible with the project's ``config/geometry.yaml``
    ``units: mm`` convention but does not enforce or read it.

    Parameters
    ----------
    radius
        Cylinder radius in caller-defined units. Must be > 0.
    height
        Cylinder height along z. Must be > 0; the resulting mesh
        spans ``[-height/2, +height/2]`` along z.
    sections
        Number of flat panels approximating the circular
        cross-section. Must be >= 8.

    Raises
    ------
    GeometryError
        If ``radius <= 0``, ``height <= 0``, or ``sections < 8``.
    """
    radius_f = float(radius)
    height_f = float(height)
    sections_i = int(sections)

    if radius_f <= 0.0:
        raise GeometryError(f"radius must be positive; got {radius_f}")
    if height_f <= 0.0:
        raise GeometryError(f"height must be positive; got {height_f}")
    if sections_i < _MIN_SECTIONS:
        raise GeometryError(
            f"sections must be >= {_MIN_SECTIONS}; got {sections_i}"
        )

    return trimesh.creation.cylinder(
        radius=radius_f,
        height=height_f,
        sections=sections_i,
    )


def create_subdivided_synthetic_bottle_body(
    *,
    radius: float = 30.0,
    height: float = 120.0,
    sections: int = 96,
    height_segments: int = 24,
) -> trimesh.Trimesh:
    """Build a vertically subdivided solid cylinder bottle-like fixture.

    Returns a watertight :class:`trimesh.Trimesh` centered near the
    origin with z bounds ``[-height/2, +height/2]`` and a circular
    cross-section of radius ``radius`` approximated by ``sections``
    flat panels. Unlike :func:`create_synthetic_bottle_body`, the
    side surface is split into ``height_segments`` quad rows
    (``height_segments + 1`` z-rings), so vertices land at
    intermediate ``v = j / height_segments`` for
    ``j = 0..height_segments``. This makes the fixture suitable
    for vertex-displacement-amount smoke checks that need patterns
    to intersect non-boundary vertices.

    This is a simple vertically subdivided solid cylinder synthetic
    bottle-like fixture, **not a real PET bottle STL**. It does not
    model wall thickness, an inner surface, neck, shoulder, base
    curvature, or water volume. Units are caller-defined; the
    function is compatible with the project's
    ``config/geometry.yaml`` ``units: mm`` convention but does not
    enforce or read it.

    Parameters
    ----------
    radius
        Cylinder radius in caller-defined units. Must be ``> 0``.
    height
        Cylinder height along z. Must be ``> 0``; the resulting
        mesh spans ``[-height/2, +height/2]`` along z.
    sections
        Number of flat panels approximating the circular
        cross-section. Must be ``>= 8``.
    height_segments
        Number of quad rows along z on the side surface. Must be
        ``>= 1``. ``height_segments + 1`` z-rings are emitted.

    Raises
    ------
    GeometryError
        If ``radius <= 0``, ``height <= 0``, ``sections < 8``, or
        ``height_segments < 1``.
    """
    radius_f = float(radius)
    height_f = float(height)
    sections_i = int(sections)
    height_segments_i = int(height_segments)

    if radius_f <= 0.0:
        raise GeometryError(f"radius must be positive; got {radius_f}")
    if height_f <= 0.0:
        raise GeometryError(f"height must be positive; got {height_f}")
    if sections_i < _MIN_SECTIONS:
        raise GeometryError(
            f"sections must be >= {_MIN_SECTIONS}; got {sections_i}"
        )
    if height_segments_i < _MIN_HEIGHT_SEGMENTS:
        raise GeometryError(
            f"height_segments must be >= {_MIN_HEIGHT_SEGMENTS}; "
            f"got {height_segments_i}"
        )

    s = sections_i
    nz = height_segments_i + 1
    half_h = 0.5 * height_f

    theta = np.linspace(0.0, 2.0 * np.pi, s, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    z_vals = np.linspace(-half_h, +half_h, nz)

    side_xy = np.stack(
        [radius_f * cos_t, radius_f * sin_t], axis=-1
    )                                                   # (s, 2)
    side_vertices = np.empty((nz * s, 3), dtype=float)
    for j in range(nz):
        side_vertices[j * s:(j + 1) * s, 0:2] = side_xy
        side_vertices[j * s:(j + 1) * s, 2] = z_vals[j]

    bottom_center_idx = nz * s
    top_center_idx = bottom_center_idx + 1
    centers = np.array(
        [[0.0, 0.0, -half_h], [0.0, 0.0, +half_h]], dtype=float
    )
    vertices = np.vstack([side_vertices, centers])

    faces_list: list[list[int]] = []

    # Lateral surface — winding chosen so face normals point radial outward.
    # Cross-product sanity: T_theta x T_z = radial_outward, so for each
    # triangle we order vertices so the first edge is along theta+ and
    # the second edge has a +z component (or vice versa with both signs
    # consistent), giving a positive radial normal.
    for j in range(height_segments_i):
        for k in range(s):
            kn = (k + 1) % s
            a = j * s + k
            b = (j + 1) * s + k
            c = j * s + kn
            d = (j + 1) * s + kn
            faces_list.append([a, c, d])
            faces_list.append([a, d, b])

    # Bottom cap (j = 0) — outward normal = -z.
    for k in range(s):
        kn = (k + 1) % s
        faces_list.append([bottom_center_idx, kn, k])

    # Top cap (j = height_segments) — outward normal = +z.
    top_ring_off = height_segments_i * s
    for k in range(s):
        kn = (k + 1) % s
        faces_list.append(
            [top_center_idx, top_ring_off + k, top_ring_off + kn]
        )

    faces = np.asarray(faces_list, dtype=np.int64)
    return trimesh.Trimesh(
        vertices=vertices, faces=faces, process=False
    )


def create_subdivided_synthetic_bottle_shell(
    *,
    outer_radius: float = 30.0,
    wall_thickness: float = 1.0,
    height: float = 120.0,
    sections: int = 96,
    height_segments: int = 24,
) -> trimesh.Trimesh:
    """Build a closed hollow cylindrical shell fixture.

    Synthetic shell/fill-state optical-to-thermal smoke fixture;
    **not a physical PET-bottle validation**. The shell is a
    z-axis aligned closed hollow cylindrical tube approximating a
    PET wall as four panels:

    - outer lateral surface at radius ``outer_radius`` (face
      normals point radially outward),
    - inner lateral surface at radius
      ``outer_radius - wall_thickness`` (face normals point
      radially inward, toward the cavity),
    - top annular rim at ``z = +height/2`` (face normals point
      ``+z``),
    - bottom annular rim at ``z = -height/2`` (face normals point
      ``-z``).

    The central cavity remains empty space; no cavity-fill mesh,
    no cap, no neck, no shoulder, no base petaloid geometry, no
    label, no seam, and no manufacturing defect is modeled. The
    fill state (empty / water-filled) is supplied to the optical
    pipeline through caller-provided interface-sequence presets,
    not through automatic medium tracking.

    The mesh is built from raw vertex / face arrays with
    ``process=False`` so that ``trimesh`` does not silently merge
    vertices, drop degenerate faces, or alter winding. The
    construction is hand-tuned to be watertight; no boolean
    backend, no external solid-modeling dependency, and no STL I/O
    are used.

    Parameters
    ----------
    outer_radius
        Outer lateral surface radius in caller-defined units.
        Must be ``> 0``.
    wall_thickness
        PET wall thickness. Must be ``> 0`` and strictly less than
        ``outer_radius`` so the inner radius stays positive.
    height
        Cylinder height along z. Must be ``> 0``; the resulting
        mesh spans ``[-height/2, +height/2]`` along z.
    sections
        Number of flat panels approximating both circular
        cross-sections. Must be ``>= 8``.
    height_segments
        Number of quad rows along z on each lateral surface.
        Must be ``>= 1``. ``height_segments + 1`` z-rings are
        emitted on each lateral surface.

    Raises
    ------
    GeometryError
        If ``outer_radius <= 0``, ``wall_thickness <= 0``,
        ``wall_thickness >= outer_radius``, ``height <= 0``,
        ``sections < 8``, or ``height_segments < 1``.
    """
    outer_r = float(outer_radius)
    wall_t = float(wall_thickness)
    height_f = float(height)
    sections_i = int(sections)
    height_segments_i = int(height_segments)

    if outer_r <= 0.0:
        raise GeometryError(
            f"outer_radius must be positive; got {outer_r}"
        )
    if wall_t <= 0.0:
        raise GeometryError(
            f"wall_thickness must be positive; got {wall_t}"
        )
    if wall_t >= outer_r:
        raise GeometryError(
            f"wall_thickness ({wall_t}) must be strictly less than "
            f"outer_radius ({outer_r}) so the inner radius is positive"
        )
    if height_f <= 0.0:
        raise GeometryError(f"height must be positive; got {height_f}")
    if sections_i < _MIN_SECTIONS:
        raise GeometryError(
            f"sections must be >= {_MIN_SECTIONS}; got {sections_i}"
        )
    if height_segments_i < _MIN_HEIGHT_SEGMENTS:
        raise GeometryError(
            f"height_segments must be >= {_MIN_HEIGHT_SEGMENTS}; "
            f"got {height_segments_i}"
        )

    inner_r = outer_r - wall_t
    s = sections_i
    nz = height_segments_i + 1
    half_h = 0.5 * height_f

    theta = np.linspace(0.0, 2.0 * np.pi, s, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    z_vals = np.linspace(-half_h, +half_h, nz)

    outer_xy = np.stack(
        [outer_r * cos_t, outer_r * sin_t], axis=-1
    )
    inner_xy = np.stack(
        [inner_r * cos_t, inner_r * sin_t], axis=-1
    )

    outer_vertices = np.empty((nz * s, 3), dtype=float)
    inner_vertices = np.empty((nz * s, 3), dtype=float)
    for j in range(nz):
        outer_vertices[j * s:(j + 1) * s, 0:2] = outer_xy
        outer_vertices[j * s:(j + 1) * s, 2] = z_vals[j]
        inner_vertices[j * s:(j + 1) * s, 0:2] = inner_xy
        inner_vertices[j * s:(j + 1) * s, 2] = z_vals[j]

    vertices = np.vstack([outer_vertices, inner_vertices])
    inner_off = nz * s

    faces_list: list[list[int]] = []

    # Outer lateral surface — face normals point radially outward.
    # Same winding as solid bottle body lateral surface.
    for j in range(height_segments_i):
        for k in range(s):
            kn = (k + 1) % s
            a = j * s + k
            b = (j + 1) * s + k
            c = j * s + kn
            d = (j + 1) * s + kn
            faces_list.append([a, c, d])
            faces_list.append([a, d, b])

    # Inner lateral surface — face normals point radially inward
    # toward the cavity. Reverse winding compared to outer.
    for j in range(height_segments_i):
        for k in range(s):
            kn = (k + 1) % s
            a = inner_off + j * s + k
            b = inner_off + (j + 1) * s + k
            c = inner_off + j * s + kn
            d = inner_off + (j + 1) * s + kn
            faces_list.append([a, d, c])
            faces_list.append([a, b, d])

    # Top annular rim (z = +half_h) — face normals point +z.
    top_outer_off = height_segments_i * s
    top_inner_off = inner_off + height_segments_i * s
    for k in range(s):
        kn = (k + 1) % s
        oc = top_outer_off + k
        on = top_outer_off + kn
        ic = top_inner_off + k
        in_ = top_inner_off + kn
        faces_list.append([oc, on, in_])
        faces_list.append([oc, in_, ic])

    # Bottom annular rim (z = -half_h) — face normals point -z.
    bot_outer_off = 0
    bot_inner_off = inner_off
    for k in range(s):
        kn = (k + 1) % s
        oc = bot_outer_off + k
        on = bot_outer_off + kn
        ic = bot_inner_off + k
        in_ = bot_inner_off + kn
        faces_list.append([oc, in_, on])
        faces_list.append([oc, ic, in_])

    faces = np.asarray(faces_list, dtype=np.int64)
    return trimesh.Trimesh(
        vertices=vertices, faces=faces, process=False
    )


def classify_synthetic_shell_vertices(
    mesh: trimesh.Trimesh,
    *,
    outer_radius: float,
    wall_thickness: float,
    height: float,
    radial_tolerance: float = 1e-6,
    boundary_epsilon: float = 0.0,
) -> SyntheticShellVertexMasks:
    """Classify synthetic shell vertices into outer / inner / boundary groups.

    Synthetic outer-surface patterned shell smoke check; **not a
    physical PET-bottle validation**. Reduces a synthetic
    cylindrical-shell mesh (typically the output of
    :func:`create_subdivided_synthetic_bottle_shell`) to per-vertex
    boolean masks driven by radial distance from the z-axis and
    distance from the top/bottom z-boundary. The intended
    downstream use is to drive
    :func:`compute_vertex_displacement_amounts` via its
    ``include_mask`` argument so that pattern displacement applies
    only to outer lateral interior vertices.

    This classifier is **only** for the synthetic shell fixture.
    It is **not** a general STL surface segmentation algorithm; it
    does **not** classify neck, shoulder, base petaloid geometry,
    labels, caps, seams, or manufacturing defects. ``mesh`` is
    read but never mutated.

    Mask definitions
    ----------------
    Let ``r = sqrt(x^2 + y^2)``, ``inner_radius = outer_radius -
    wall_thickness``, and ``half_h = height / 2``::

        outer_wall_mask              = abs(r - outer_radius) <= radial_tolerance
        inner_wall_mask              = abs(r - inner_radius) <= radial_tolerance
        z_boundary_mask              = (z <= -half_h + boundary_epsilon * height)
                                       | (z >= +half_h - boundary_epsilon * height)
                                       (all-False when boundary_epsilon == 0)
        outer_lateral_interior_mask  = outer_wall_mask & ~z_boundary_mask
        inner_lateral_interior_mask  = inner_wall_mask & ~z_boundary_mask

    ``boundary_epsilon`` is in normalized v space, mirroring
    :func:`compute_vertex_displacement_amounts`. With
    ``boundary_epsilon = 0`` (default) no vertex is flagged as a
    z-boundary vertex even if it lies exactly on ``z = +-half_h``.

    Parameters
    ----------
    mesh
        Input :class:`trimesh.Trimesh`. Read but not mutated.
    outer_radius, wall_thickness, height
        Same geometric parameters used to build the shell. Must be
        ``> 0``; ``wall_thickness`` must be strictly less than
        ``outer_radius``.
    radial_tolerance
        Strict positive radial tolerance used for both outer and
        inner wall membership. Must be ``< wall_thickness / 2``
        for outer/inner masks to be disjoint; this constraint is
        not enforced here so callers can choose tolerances
        appropriate to their input mesh.
    boundary_epsilon
        Normalized-v boundary fraction in ``[0, 0.5)``. Mirrors
        ``compute_vertex_displacement_amounts``'s
        ``exclude_v_boundary_epsilon``.

    Raises
    ------
    GeometryError
        On invalid mesh type, non-positive ``outer_radius`` /
        ``wall_thickness`` / ``height``, ``wall_thickness >=
        outer_radius``, non-positive or non-finite
        ``radial_tolerance``, or ``boundary_epsilon`` outside
        ``[0, 0.5)``.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise GeometryError(
            f"mesh must be a trimesh.Trimesh; got {type(mesh).__name__}"
        )

    outer_r = float(outer_radius)
    wall_t = float(wall_thickness)
    height_f = float(height)
    radial_tol = float(radial_tolerance)
    eps = float(boundary_epsilon)

    if not np.isfinite(outer_r) or outer_r <= 0.0:
        raise GeometryError(
            f"outer_radius must be a finite float > 0; got {outer_r}"
        )
    if not np.isfinite(wall_t) or wall_t <= 0.0:
        raise GeometryError(
            f"wall_thickness must be a finite float > 0; got {wall_t}"
        )
    if wall_t >= outer_r:
        raise GeometryError(
            f"wall_thickness ({wall_t}) must be strictly less than "
            f"outer_radius ({outer_r})"
        )
    if not np.isfinite(height_f) or height_f <= 0.0:
        raise GeometryError(
            f"height must be a finite float > 0; got {height_f}"
        )
    if not np.isfinite(radial_tol) or radial_tol <= 0.0:
        raise GeometryError(
            f"radial_tolerance must be a finite float > 0; "
            f"got {radial_tol}"
        )
    if not np.isfinite(eps) or eps < 0.0 or eps >= 0.5:
        raise GeometryError(
            f"boundary_epsilon must be a finite float in [0, 0.5); "
            f"got {eps}"
        )

    inner_r = outer_r - wall_t

    vertices = np.asarray(mesh.vertices, dtype=float)
    n = int(vertices.shape[0])
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise GeometryError(
            f"mesh.vertices must have shape (N, 3); "
            f"got {vertices.shape}"
        )

    r = np.sqrt(vertices[:, 0] ** 2 + vertices[:, 1] ** 2)
    z = vertices[:, 2]
    half_h = 0.5 * height_f

    outer_wall = np.abs(r - outer_r) <= radial_tol
    inner_wall = np.abs(r - inner_r) <= radial_tol

    if eps > 0.0:
        eps_z = eps * height_f
        z_boundary = (z <= -half_h + eps_z) | (z >= half_h - eps_z)
    else:
        z_boundary = np.zeros(n, dtype=bool)

    outer_interior = outer_wall & ~z_boundary
    inner_interior = inner_wall & ~z_boundary

    return SyntheticShellVertexMasks(
        outer_wall_mask=outer_wall.astype(bool, copy=True),
        inner_wall_mask=inner_wall.astype(bool, copy=True),
        z_boundary_mask=z_boundary.astype(bool, copy=True),
        outer_lateral_interior_mask=outer_interior.astype(
            bool, copy=True
        ),
        inner_lateral_interior_mask=inner_interior.astype(
            bool, copy=True
        ),
        vertex_count=n,
        outer_radius=float(outer_r),
        inner_radius=float(inner_r),
        height=float(height_f),
        boundary_epsilon=float(eps),
    )
