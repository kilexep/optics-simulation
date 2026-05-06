"""Synthetic PET-like bottle mesh fixtures.

Two solid-cylinder fixtures for development and pipeline
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

Both fixtures return :class:`trimesh.Trimesh` solid cylinder
approximations and are **not real PET bottle STLs**: neither
models wall thickness, an inner surface, neck, shoulder, base
curvature, or water volume. No file I/O, no boolean operations,
no external solid-modeling backend, and no config-file reads
happen here.
"""
from __future__ import annotations

import numpy as np
import trimesh

from optics_simulation.geometry.mesh_io import GeometryError


_MIN_SECTIONS = 8
_MIN_HEIGHT_SEGMENTS = 1


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
