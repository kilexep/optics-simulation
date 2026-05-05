"""Pattern package: Gaussian dimple sampling and depth-field evaluation."""
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

__all__ = [
    "GaussianDimple",
    "GaussianDimplePattern",
    "PatternError",
    "create_gaussian_dimple_pattern",
    "evaluate_gaussian_dimple_field",
    "gaussian_pattern_from_descriptor",
    "gaussian_pattern_to_descriptor",
    "sample_dimple_centers_from_risk",
]
