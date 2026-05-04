"""Single-interface smoke pipeline.

Thin wrapper that wires :func:`intersect_rays` and
:func:`propagate_through_interface` into a single entry point and
collects per-bundle counts. No new physics is introduced here:
multi-bounce, reflected-ray generation, Fresnel power weighting,
medium-state tracking, detector accumulation, irradiance / contribution
maps, and thermal modeling are intentionally out of scope.

Counts
------
The four counters partition the input bundle on two axes:

- ``ray_count == hit_count + miss_count``
- ``hit_count == transmitted_count + tir_count``
- ``ray_count == transmitted_count + tir_count + miss_count``

A miss ray is one that did not intersect the mesh. A TIR ray hit the
mesh but underwent total internal reflection (so no transmitted ray
was emitted). A transmitted ray hit the mesh and produced one entry
in ``propagation.next_rays``.
"""
from __future__ import annotations

from dataclasses import dataclass

import trimesh

from optics_simulation.optics.intersection import (
    IntersectionResult,
    intersect_rays,
)
from optics_simulation.optics.propagation import (
    PropagationResult,
    propagate_through_interface,
)
from optics_simulation.optics.ray import OpticsError, RayBundle


@dataclass(frozen=True)
class SingleInterfacePipelineResult:
    rays: RayBundle
    intersection: IntersectionResult
    propagation: PropagationResult
    ray_count: int
    hit_count: int
    miss_count: int
    transmitted_count: int
    tir_count: int


def run_single_interface_pipeline(
    mesh: trimesh.Trimesh,
    rays: RayBundle,
    eta_i: float,
    eta_t: float,
    epsilon: float = 1e-6,
) -> SingleInterfacePipelineResult:
    """Run one ray-mesh intersection plus refractive propagation step.

    ``eta_i`` is the medium the rays are currently in; ``eta_t`` is the
    medium they are entering. Both must be positive. ``epsilon`` is the
    next-ray origin offset along the refracted direction; must be >= 0.

    See module docstring for the count invariants this function
    guarantees.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise OpticsError(
            f"mesh must be a trimesh.Trimesh; got {type(mesh).__name__}"
        )
    if not isinstance(rays, RayBundle):
        raise OpticsError(
            f"rays must be a RayBundle; got {type(rays).__name__}"
        )
    if eta_i <= 0.0 or eta_t <= 0.0:
        raise OpticsError(
            f"refractive indices must be positive; got eta_i={eta_i}, eta_t={eta_t}"
        )
    if epsilon < 0.0:
        raise OpticsError(f"epsilon must be >= 0; got {epsilon}")

    intersection = intersect_rays(mesh, rays)
    propagation = propagate_through_interface(
        rays, intersection, eta_i, eta_t, epsilon=epsilon
    )

    ray_count = int(rays.ray_count)
    hit_count = int(intersection.hit_mask.sum())
    miss_count = ray_count - hit_count
    transmitted_count = int(propagation.active_mask.sum())
    tir_count = int(propagation.tir_mask.sum())

    return SingleInterfacePipelineResult(
        rays=rays,
        intersection=intersection,
        propagation=propagation,
        ray_count=ray_count,
        hit_count=hit_count,
        miss_count=miss_count,
        transmitted_count=transmitted_count,
        tir_count=tir_count,
    )
