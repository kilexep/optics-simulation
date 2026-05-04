"""Geometry package: STL/mesh I/O and quality reporting."""
from optics_simulation.geometry.mesh_io import (
    GeometryError,
    MeshLoadError,
    load_mesh,
    load_mesh_with_report,
)
from optics_simulation.geometry.quality import (
    MeshQualityReport,
    create_mesh_quality_report,
)

__all__ = [
    "GeometryError",
    "MeshLoadError",
    "MeshQualityReport",
    "create_mesh_quality_report",
    "load_mesh",
    "load_mesh_with_report",
]
