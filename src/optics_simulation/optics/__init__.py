"""Optics package: ray sources, ray-mesh intersection, and Snell/Fresnel."""
from optics_simulation.optics.intersection import (
    IntersectionResult,
    intersect_rays,
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
    "RayBundle",
    "RefractionResult",
    "fresnel_unpolarized",
    "intersect_rays",
    "make_ray_bundle",
    "normalize_vector",
    "parallel_ray_grid",
    "refract_direction",
]
