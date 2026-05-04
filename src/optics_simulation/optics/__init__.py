"""Optics package: ray sources and ray-mesh intersection scaffold."""
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

__all__ = [
    "IntersectionResult",
    "OpticsError",
    "RayBundle",
    "intersect_rays",
    "make_ray_bundle",
    "parallel_ray_grid",
]
