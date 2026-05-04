"""Optics package: ray sources, intersection, Snell/Fresnel, propagation, pipeline."""
from optics_simulation.optics.intersection import (
    IntersectionResult,
    intersect_rays,
)
from optics_simulation.optics.pipeline import (
    SingleInterfacePipelineResult,
    run_single_interface_pipeline,
)
from optics_simulation.optics.propagation import (
    PropagationResult,
    propagate_through_interface,
)
from optics_simulation.optics.ray import (
    OpticsError,
    RayBundle,
    make_ray_bundle,
    parallel_ray_grid,
)
from optics_simulation.optics.refraction import (
    RefractionResult,
    fresnel_unpolarized,
    normalize_vector,
    refract_direction,
)

__all__ = [
    "IntersectionResult",
    "OpticsError",
    "PropagationResult",
    "RayBundle",
    "RefractionResult",
    "SingleInterfacePipelineResult",
    "fresnel_unpolarized",
    "intersect_rays",
    "make_ray_bundle",
    "normalize_vector",
    "parallel_ray_grid",
    "propagate_through_interface",
    "refract_direction",
    "run_single_interface_pipeline",
]
