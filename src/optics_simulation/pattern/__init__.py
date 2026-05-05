"""Pattern package: Gaussian dimple sampling and depth-field evaluation."""
from optics_simulation.pattern.gaussian import (
    GaussianDimple,
    GaussianDimplePattern,
    PatternError,
    create_gaussian_dimple_pattern,
    evaluate_gaussian_dimple_field,
    sample_dimple_centers_from_risk,
)

__all__ = [
    "GaussianDimple",
    "GaussianDimplePattern",
    "PatternError",
    "create_gaussian_dimple_pattern",
    "evaluate_gaussian_dimple_field",
    "sample_dimple_centers_from_risk",
]
