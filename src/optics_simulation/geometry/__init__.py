"""Geometry package: STL/mesh I/O, quality reporting, and surface coordinates."""
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
from optics_simulation.geometry.surface_coordinates import (
    SurfaceCoordinateMap,
    create_vertex_surface_coordinates,
    normalize_cylindrical_coordinates,
    point_to_normalized_cylindrical,
)
from optics_simulation.geometry.synthetic_bottle import (
    create_subdivided_synthetic_bottle_body,
    create_synthetic_bottle_body,
)
from optics_simulation.geometry.hit_coordinates import (
    HitSurfaceCoordinates,
    hit_to_surface_coordinates,
)

__all__ = [
    "GeometryError",
    "HitSurfaceCoordinates",
    "MeshLoadError",
    "MeshQualityReport",
    "SurfaceCoordinateMap",
    "create_mesh_quality_report",
    "create_subdivided_synthetic_bottle_body",
    "create_synthetic_bottle_body",
    "create_vertex_surface_coordinates",
    "hit_to_surface_coordinates",
    "load_mesh",
    "load_mesh_with_report",
    "normalize_cylindrical_coordinates",
    "point_to_normalized_cylindrical",
]
