"""Synthetic PET-like bottle mesh fixture.

Generates a simple z-axis aligned solid cylinder as a synthetic
bottle-like fixture for development and pipeline-compatibility
smoke checks. The returned mesh is **not a real PET bottle STL**:
it does not model wall thickness, neck, shoulder, base curvature,
or water volume. It is a placeholder until a real STL ingestion
flow is wired up, and lets the angle scan / detector pipelines run
against a curved (rather than flat-slab) geometry.

The mesh is produced by :func:`trimesh.creation.cylinder` with
caller-supplied ``radius`` / ``height`` / ``sections``. No file I/O,
no boolean operations, no external solid-modeling backend, and no
config-file reads happen here.
"""
from __future__ import annotations

import trimesh

from optics_simulation.geometry.mesh_io import GeometryError


_MIN_SECTIONS = 8


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
