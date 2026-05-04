"""Single-interface ray propagation.

Connects :class:`RayBundle`, :class:`IntersectionResult`, and
:func:`refract_direction` into one step: for every input ray that hit
the mesh and was *not* total-internally-reflected, build the next
ray (origin = hit point + epsilon * refracted direction, direction =
refracted direction) and emit a compact ``next_rays`` bundle. Total
internal reflection rays and miss rays do not produce a next ray in
this module — reflected-ray generation belongs to a multi-bounce
tracer that is intentionally out of scope here.

Layout
------
The result mixes dense and compact arrays:

- ``active_mask`` and ``tir_mask`` and ``reflectance`` and
  ``transmittance`` are dense, length ``N`` (the input ray count).
  This matches the convention used by :class:`IntersectionResult`,
  so callers can index them directly with the original input index.
- ``next_rays`` is compact, length ``M = active_mask.sum()``.
  ``source_ray_indices`` is the bridge: ``source_ray_indices[k]``
  gives the input index that produced ``next_rays[k]``.

Sentinels
---------
- Miss rays (``hit_mask == False``): ``reflectance`` and
  ``transmittance`` are ``NaN``; both masks are ``False``.
- TIR rays: ``reflectance == 1.0``, ``transmittance == 0.0``;
  ``tir_mask == True``, ``active_mask == False``; no entry in
  ``next_rays`` and no entry in ``source_ray_indices``.
- Active (transmitted) rays: actual Fresnel R / T (with
  ``R + T == 1`` up to clamping); ``active_mask == True``;
  one corresponding entry in ``next_rays`` and ``source_ray_indices``.

Limitations
-----------
- **Scalar inner loop.** ``refract_direction`` is called once per
  hit ray in a Python loop. Vectorization is deferred.
- **Single-pair eta.** ``eta_i`` and ``eta_t`` are scalars applied
  to every ray. Medium-state tracking (PET / water / air auto
  routing) is the caller's responsibility and out of scope.
- **No reflected-ray output.** TIR rays terminate here. Multi-bounce
  tracing is a separate module.
- **Origin offset along refracted direction.** The next ray's
  origin is moved ``epsilon`` units along the refracted direction
  (not along the incident direction or normal). This is the most
  robust default for avoiding immediate self-intersection at the
  same face under non-grazing transmission.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optics_simulation.optics.intersection import IntersectionResult
from optics_simulation.optics.ray import OpticsError, RayBundle, make_ray_bundle
from optics_simulation.optics.refraction import refract_direction


@dataclass(frozen=True)
class PropagationResult:
    next_rays: RayBundle
    active_mask: np.ndarray         # shape (N,), bool — hit AND not TIR
    tir_mask: np.ndarray            # shape (N,), bool — hit AND TIR
    source_ray_indices: np.ndarray  # shape (M,), int64 — input index per next_rays entry
    reflectance: np.ndarray         # shape (N,), float — NaN on miss, 1.0 on TIR
    transmittance: np.ndarray       # shape (N,), float — NaN on miss, 0.0 on TIR


def propagate_through_interface(
    rays: RayBundle,
    intersection: IntersectionResult,
    eta_i: float,
    eta_t: float,
    epsilon: float = 1e-6,
) -> PropagationResult:
    """Build the next-bounce :class:`RayBundle` for one dielectric interface.

    ``eta_i`` is the index of the medium the rays are currently in;
    ``eta_t`` is the medium they are entering. Both must be positive.

    ``epsilon`` is the distance the next-ray origin is offset along
    its own refracted direction past the hit point. ``epsilon == 0``
    is allowed for tests, but positive ``epsilon`` is recommended to
    avoid self-intersection at the same face on the next cast.
    """
    if not isinstance(rays, RayBundle):
        raise OpticsError(
            f"rays must be a RayBundle; got {type(rays).__name__}"
        )
    if not isinstance(intersection, IntersectionResult):
        raise OpticsError(
            f"intersection must be an IntersectionResult; "
            f"got {type(intersection).__name__}"
        )
    if intersection.ray_count != rays.ray_count:
        raise OpticsError(
            f"intersection.ray_count ({intersection.ray_count}) does not match "
            f"rays.ray_count ({rays.ray_count})"
        )
    if eta_i <= 0.0 or eta_t <= 0.0:
        raise OpticsError(
            f"refractive indices must be positive; got eta_i={eta_i}, eta_t={eta_t}"
        )
    if epsilon < 0.0:
        raise OpticsError(f"epsilon must be >= 0; got {epsilon}")

    n = rays.ray_count
    active_mask = np.zeros(n, dtype=bool)
    tir_mask = np.zeros(n, dtype=bool)
    reflectance = np.full(n, np.nan, dtype=float)
    transmittance = np.full(n, np.nan, dtype=float)

    next_origins_list: list[np.ndarray] = []
    next_directions_list: list[np.ndarray] = []
    source_indices_list: list[int] = []

    hit_indices = np.flatnonzero(intersection.hit_mask)
    for i in hit_indices:
        idx = int(i)
        result = refract_direction(
            wi=rays.directions[idx],
            normal=intersection.hit_normals[idx],
            eta_i=eta_i,
            eta_t=eta_t,
        )
        if result.total_internal_reflection:
            tir_mask[idx] = True
            reflectance[idx] = 1.0
            transmittance[idx] = 0.0
            continue

        active_mask[idx] = True
        reflectance[idx] = result.reflectance
        transmittance[idx] = result.transmittance
        direction = np.asarray(result.direction, dtype=float)
        origin = np.asarray(intersection.hit_points[idx], dtype=float) + epsilon * direction
        next_origins_list.append(origin)
        next_directions_list.append(direction)
        source_indices_list.append(idx)

    if next_origins_list:
        next_origins = np.stack(next_origins_list, axis=0)
        next_directions = np.stack(next_directions_list, axis=0)
    else:
        next_origins = np.zeros((0, 3), dtype=float)
        next_directions = np.zeros((0, 3), dtype=float)

    next_rays = make_ray_bundle(next_origins, next_directions)
    source_ray_indices = np.asarray(source_indices_list, dtype=np.int64)

    return PropagationResult(
        next_rays=next_rays,
        active_mask=active_mask,
        tir_mask=tir_mask,
        source_ray_indices=source_ray_indices,
        reflectance=reflectance,
        transmittance=transmittance,
    )
