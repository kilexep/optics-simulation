"""Geometry package: STL/mesh I/O, quality reporting, and surface coordinates."""
from optics_simulation.geometry.bottle_readiness import (
    BottleMeshReadinessReport,
    CylindricalFitReport,
    compute_bottle_mesh_readiness_report,
    compute_cylindrical_fit_report,
)
from optics_simulation.geometry.bottle_readiness_io import (
    BottleMeshReadinessFileReport,
    load_bottle_mesh_readiness_report,
)
from optics_simulation.geometry.bottle_scaling import (
    BottleTargetScaleReport,
    InnerOffsetMeshReport,
    compute_target_dimension_scale,
    create_inner_offset_mesh_from_vertex_normals,
    create_target_scaled_mesh_copy,
)
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
    SyntheticShellVertexMasks,
    classify_synthetic_shell_vertices,
    create_subdivided_synthetic_bottle_body,
    create_subdivided_synthetic_bottle_shell,
    create_synthetic_bottle_body,
)
from optics_simulation.geometry.hit_coordinates import (
    HitSurfaceCoordinates,
    hit_to_surface_coordinates,
)

__all__ = [
    "BottleMeshReadinessFileReport",
    "BottleMeshReadinessReport",
    "BottleTargetScaleReport",
    "CylindricalFitReport",
    "GeometryError",
    "HitSurfaceCoordinates",
    "InnerOffsetMeshReport",
    "MeshLoadError",
    "MeshQualityReport",
    "SurfaceCoordinateMap",
    "SyntheticShellVertexMasks",
    "classify_synthetic_shell_vertices",
    "compute_bottle_mesh_readiness_report",
    "compute_cylindrical_fit_report",
    "compute_target_dimension_scale",
    "create_inner_offset_mesh_from_vertex_normals",
    "create_mesh_quality_report",
    "create_subdivided_synthetic_bottle_body",
    "create_subdivided_synthetic_bottle_shell",
    "create_synthetic_bottle_body",
    "create_target_scaled_mesh_copy",
    "create_vertex_surface_coordinates",
    "hit_to_surface_coordinates",
    "load_bottle_mesh_readiness_report",
    "load_mesh",
    "load_mesh_with_report",
    "normalize_cylindrical_coordinates",
    "point_to_normalized_cylindrical",
]
