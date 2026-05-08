"""Pattern package: Gaussian dimple sampling and depth-field evaluation."""
from optics_simulation.pattern.actual_stl_displacement import (
    ActualSTLPatternedMeshResult,
    create_actual_stl_patterned_mesh_copy,
)
from optics_simulation.pattern.descriptor import (
    gaussian_pattern_from_descriptor,
    gaussian_pattern_to_descriptor,
)
from optics_simulation.pattern.gaussian import (
    GaussianDimple,
    GaussianDimplePattern,
    PatternError,
    create_gaussian_dimple_pattern,
    evaluate_gaussian_dimple_field,
    sample_dimple_centers_from_risk,
)
from optics_simulation.pattern.mesh_displacement import (
    DisplacedMeshResult,
    create_displaced_mesh_copy,
)
from optics_simulation.pattern.vertex_displacement import (
    VertexPatternDisplacement,
    compute_vertex_displacement_amounts,
    evaluate_gaussian_pattern_at_points,
)

__all__ = [
    "ActualSTLPatternedMeshResult",
    "DisplacedMeshResult",
    "GaussianDimple",
    "GaussianDimplePattern",
    "PatternError",
    "VertexPatternDisplacement",
    "compute_vertex_displacement_amounts",
    "create_actual_stl_patterned_mesh_copy",
    "create_displaced_mesh_copy",
    "create_gaussian_dimple_pattern",
    "evaluate_gaussian_dimple_field",
    "evaluate_gaussian_pattern_at_points",
    "gaussian_pattern_from_descriptor",
    "gaussian_pattern_to_descriptor",
    "sample_dimple_centers_from_risk",
]
