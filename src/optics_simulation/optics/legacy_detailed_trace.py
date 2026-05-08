"""Detailed legacy PET-water trace at one angle / detector distance.

Actual STL hotspot-backtracked risk-guided pattern smoke check;
**not a physical PET-bottle validation**. Wraps a single
``(angle_degrees, detector_distance)`` evaluation of the legacy
PET-water four-step trace into a single
:class:`LegacyDetailedTraceResult` so downstream code can pull out
the rays, the per-step intersection results, the detector hits
and accumulation, the relative irradiance surrogate, the optical
metrics, and the per-ray first-shell-hit ``(u, v)`` coordinates
without re-running the trace. Source / detector geometry follows
the same world-y rotation convention as
:func:`run_legacy_pet_water_angle_distance_scan`.

Limitations
-----------
- This is a single-condition evaluation; it does not iterate
  angles or distances.
- The contribution / risk-map generation step is intentionally
  separated; this module only produces the trace-level inputs.
- This is **not** fire-prevention validation, **not** PET-bottle
  safety, and **not** manufacturing-ready.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from optics_simulation.geometry.hit_coordinates import (
    HitSurfaceCoordinates,
    hit_to_surface_coordinates,
)
from optics_simulation.geometry.surface_coordinates import (
    SurfaceCoordinateMap,
    create_vertex_surface_coordinates,
)
from optics_simulation.metrics.irradiance import (
    DetectorIrradianceSurrogate,
    compute_relative_irradiance_surrogate,
)
from optics_simulation.metrics.optical import (
    OpticalMetrics,
    compute_optical_metrics,
)
from optics_simulation.optics.detector import (
    DetectorHitResult,
    create_detector_plane,
    intersect_detector_plane,
)
from optics_simulation.optics.detector_accumulation import (
    DetectorAccumulationResult,
    accumulate_detector_hits,
    create_detector_grid,
)
from optics_simulation.optics.legacy_pet_water import (
    LegacyPetWaterTraceSetup,
)
from optics_simulation.optics.legacy_scan import (
    create_oriented_parallel_ray_grid,
)
from optics_simulation.optics.multi_mesh_trace import (
    MultiMeshTraceResult,
    run_multi_mesh_trace,
)
from optics_simulation.optics.ray import OpticsError, RayBundle


_DEFAULT_THRESHOLDS = (2.0, 5.0, 10.0)
_DEFAULT_TOP_PERCENT = 1.0


@dataclass(frozen=True)
class LegacyDetailedTraceResult:
    angle_degrees: float
    detector_distance: float
    rays: RayBundle
    trace_result: MultiMeshTraceResult
    detector_hits: DetectorHitResult
    accumulation: DetectorAccumulationResult
    irradiance_surrogate: DetectorIrradianceSurrogate
    optical_metrics: OpticalMetrics
    first_shell_hit_coordinates: HitSurfaceCoordinates
    shell_surface_map: SurfaceCoordinateMap
    source_area: float


def _rotation_matrix_around_y(angle_rad: float) -> np.ndarray:
    ca = math.cos(angle_rad)
    sa = math.sin(angle_rad)
    return np.array(
        [
            [ca, 0.0, sa],
            [0.0, 1.0, 0.0],
            [-sa, 0.0, ca],
        ],
        dtype=float,
    )


def run_legacy_pet_water_detailed_trace(
    *,
    setup: LegacyPetWaterTraceSetup,
    angle_degrees: float,
    detector_distance: float,
    source_width: float,
    source_height: float,
    source_radius: float,
    sample_count_y: int,
    sample_count_z: int,
    detector_size: float,
    detector_resolution: tuple[int, int],
    epsilon: float = 0.1,
    incident_irradiance: float = 1.0,
    thresholds: tuple[float, ...] = _DEFAULT_THRESHOLDS,
    top_percent: float = _DEFAULT_TOP_PERCENT,
) -> LegacyDetailedTraceResult:
    """Run the legacy PET-water trace at a single (angle, distance).

    Actual STL hotspot-backtracked risk-guided pattern smoke
    check; **not a physical PET-bottle validation**. See module
    docstring for what this returns. Geometry follows
    :func:`run_legacy_pet_water_angle_distance_scan`'s convention.
    """
    if not isinstance(setup, LegacyPetWaterTraceSetup):
        raise OpticsError(
            "setup must be a LegacyPetWaterTraceSetup; got "
            f"{type(setup).__name__}"
        )
    if not (
        math.isfinite(float(detector_distance))
        and float(detector_distance) > 0.0
    ):
        raise OpticsError(
            f"detector_distance must be a finite float > 0; got "
            f"{detector_distance}"
        )
    if not (
        math.isfinite(float(source_radius))
        and float(source_radius) > 0.0
    ):
        raise OpticsError(
            f"source_radius must be a finite float > 0; got "
            f"{source_radius}"
        )
    if not (
        math.isfinite(float(source_width))
        and float(source_width) > 0.0
        and math.isfinite(float(source_height))
        and float(source_height) > 0.0
    ):
        raise OpticsError(
            f"source_width and source_height must be finite > 0; "
            f"got source_width={source_width}, "
            f"source_height={source_height}"
        )
    if (
        int(sample_count_y) < 1
        or int(sample_count_z) < 1
    ):
        raise OpticsError(
            f"sample_count_y and sample_count_z must be >= 1; got "
            f"sample_count_y={sample_count_y}, "
            f"sample_count_z={sample_count_z}"
        )
    if not (
        math.isfinite(float(detector_size))
        and float(detector_size) > 0.0
    ):
        raise OpticsError(
            f"detector_size must be a finite float > 0; got "
            f"{detector_size}"
        )

    bounds = np.asarray(setup.shell_mesh.bounds, dtype=float)
    ref_point = 0.5 * (bounds[0] + bounds[1])
    rotation = _rotation_matrix_around_y(
        math.radians(float(angle_degrees))
    )
    local_source_center = np.array(
        [float(source_radius), 0.0, 0.0],
    )
    local_direction = np.array([-1.0, 0.0, 0.0])
    local_up = np.array([0.0, 0.0, 1.0])
    source_center = ref_point + rotation @ local_source_center
    ray_direction = rotation @ local_direction
    plane_up = rotation @ local_up

    rays = create_oriented_parallel_ray_grid(
        center=tuple(float(x) for x in source_center.tolist()),
        direction=tuple(float(x) for x in ray_direction.tolist()),
        up=tuple(float(x) for x in plane_up.tolist()),
        width=float(source_width),
        height=float(source_height),
        sample_count_y=int(sample_count_y),
        sample_count_z=int(sample_count_z),
    )
    trace = run_multi_mesh_trace(
        rays, setup.step_specs, epsilon=float(epsilon),
    )

    local_detector_center = np.array(
        [-float(detector_distance), 0.0, 0.0],
    )
    local_detector_normal = np.array([1.0, 0.0, 0.0])
    detector_center = ref_point + rotation @ local_detector_center
    detector_normal = rotation @ local_detector_normal
    detector_up = rotation @ local_up

    res = tuple(detector_resolution)
    if len(res) != 2:
        raise OpticsError(
            f"detector_resolution must be (ny, nx); got {res}"
        )

    detector = create_detector_plane(
        center=tuple(float(x) for x in detector_center.tolist()),
        normal=tuple(float(x) for x in detector_normal.tolist()),
        up=tuple(float(x) for x in detector_up.tolist()),
        width=float(detector_size),
        height=float(detector_size),
    )
    detector_grid = create_detector_grid(
        width=float(detector_size),
        height=float(detector_size),
        resolution=(int(res[0]), int(res[1])),
    )
    hits = intersect_detector_plane(trace.final_rays, detector)
    accumulation = accumulate_detector_hits(
        hits, detector_grid, weights=trace.final_ray_weights,
    )
    source_area = float(source_width) * float(source_height)
    surrogate = compute_relative_irradiance_surrogate(
        accumulation,
        detector_grid,
        ray_count=int(rays.ray_count),
        source_area=source_area,
        incident_irradiance=float(incident_irradiance),
    )
    metrics = compute_optical_metrics(
        surrogate.relative_irradiance_map,
        incident_reference=1.0,
        thresholds=thresholds,
        top_percent=float(top_percent),
    )

    shell_surface_map = create_vertex_surface_coordinates(
        setup.shell_mesh,
    )
    if not trace.steps:
        raise OpticsError(
            "trace produced no steps; cannot compute first shell "
            "hit coordinates"
        )
    step0 = trace.steps[0]
    first_shell_hit_coordinates = hit_to_surface_coordinates(
        setup.shell_mesh,
        step0.intersection,
        shell_surface_map,
    )

    return LegacyDetailedTraceResult(
        angle_degrees=float(angle_degrees),
        detector_distance=float(detector_distance),
        rays=rays,
        trace_result=trace,
        detector_hits=hits,
        accumulation=accumulation,
        irradiance_surrogate=surrogate,
        optical_metrics=metrics,
        first_shell_hit_coordinates=first_shell_hit_coordinates,
        shell_surface_map=shell_surface_map,
        source_area=source_area,
    )
