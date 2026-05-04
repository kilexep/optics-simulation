"""STL/mesh file ingestion.

Loads STL (and other trimesh-supported) mesh files into
``trimesh.Trimesh`` objects, with explicit error handling for
missing, empty, or invalid inputs. This module is the only place in
the geometry package that performs file I/O. Quality reporting is
delegated to :mod:`optics_simulation.geometry.quality`.
"""
from __future__ import annotations

from pathlib import Path

import trimesh

from optics_simulation.geometry.quality import (
    MeshQualityReport,
    create_mesh_quality_report,
)


class GeometryError(Exception):
    """Base class for geometry package errors."""


class MeshLoadError(GeometryError):
    """Raised when a mesh file cannot be loaded or is empty/invalid."""


def load_mesh(path: str | Path) -> trimesh.Trimesh:
    p = Path(path)
    if not p.is_file():
        raise MeshLoadError(f"Mesh file not found: {p}")

    try:
        loaded = trimesh.load(p, force="mesh")
    except Exception as exc:
        raise MeshLoadError(f"Failed to load mesh from {p}: {exc}") from exc

    if not isinstance(loaded, trimesh.Trimesh):
        raise MeshLoadError(
            f"Loaded object from {p} is not a Trimesh "
            f"(got {type(loaded).__name__})"
        )

    if len(loaded.vertices) == 0 or len(loaded.faces) == 0:
        raise MeshLoadError(f"Loaded mesh from {p} is empty")

    return loaded


def load_mesh_with_report(
    path: str | Path,
    units: str | None = None,
) -> tuple[trimesh.Trimesh, MeshQualityReport]:
    mesh = load_mesh(path)
    report = create_mesh_quality_report(mesh, source_path=path, units=units)
    return mesh, report
