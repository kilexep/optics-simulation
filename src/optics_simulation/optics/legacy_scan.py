"""Legacy actual-STL angle / detector-distance optical scan.

Actual STL legacy-style optical scan smoke check; **not a physical
PET-bottle validation**. Exposes two pieces:

- :func:`create_oriented_parallel_ray_grid` builds an arbitrarily
  oriented rectangular grid of parallel rays from
  ``(center, direction, up, width, height, sample_count_y,
  sample_count_z)``. Internally ``up`` is projected to be
  perpendicular to ``direction`` and renormalized; ``right`` is
  ``cross(up_perp, direction)`` (same handedness convention as
  :func:`create_detector_plane`).
- :func:`run_legacy_pet_water_angle_distance_scan` reuses an
  already-built :class:`LegacyPetWaterTraceSetup` and sweeps over
  caller-supplied ``angles_degrees`` and ``detector_distances``.
  For each angle the source plane and propagation direction are
  rotated around the world y-axis, the four-step
  ``run_multi_mesh_trace`` is run once, and every requested
  detector distance is evaluated against the surviving rays.

Scope (also enforced at runtime)
--------------------------------
- This scan uses the loaded STL as a geometry input.
- It uses target-dimension scaling and a generated inner offset
  mesh.
- It does not prove physical accuracy.
- It does not repair meshes.
- It does not infer material regions automatically.
- It does not perform thermal simulation.
- It does not prove fire prevention or PET-bottle safety.
- Detector intensity metrics are relative optical surrogates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from optics_simulation.metrics.irradiance import (
    compute_relative_irradiance_surrogate,
)
from optics_simulation.metrics.optical import (
    compute_optical_metrics,
)
from optics_simulation.optics.detector import (
    create_detector_plane,
    intersect_detector_plane,
)
from optics_simulation.optics.detector_accumulation import (
    accumulate_detector_hits,
    create_detector_grid,
)
from optics_simulation.optics.legacy_pet_water import (
    LegacyPetWaterTraceSetup,
)
from optics_simulation.optics.multi_mesh_trace import (
    run_multi_mesh_trace,
)
from optics_simulation.optics.ray import (
    OpticsError,
    RayBundle,
    make_ray_bundle,
)


_DEFAULT_SOURCE_RADIUS = 200.0
_UP_PROJECTION_EPS = 1e-9
_DEFAULT_THRESHOLDS = (2.0, 5.0, 10.0)
_DEFAULT_TOP_PERCENT = 1.0


@dataclass(frozen=True)
class LegacySourcePlaneConfig:
    center: np.ndarray
    direction: np.ndarray
    up: np.ndarray
    width: float
    height: float
    sample_count_y: int
    sample_count_z: int


@dataclass(frozen=True)
class LegacyOpticalScanEntry:
    angle_degrees: float
    detector_distance: float
    ray_count: int
    final_ray_count: int
    detector_hits: int
    total_weight: float
    max_relative_irradiance: float
    cmax: float
    c99: float
    termination_reason: str


@dataclass(frozen=True)
class LegacyOpticalScanResult:
    entries: tuple[LegacyOpticalScanEntry, ...]
    angle_count: int
    detector_count: int
    max_relative_irradiance_angle: float | None
    max_relative_irradiance_detector_distance: float | None
    max_relative_irradiance: float | None
    max_c99_angle: float | None
    max_c99_detector_distance: float | None
    max_c99: float | None
    result_type: str = "legacy_actual_stl_optical_scan"


def _as_3vec(v, name: str) -> np.ndarray:
    arr = np.asarray(v, dtype=float).reshape(-1)
    if arr.size != 3:
        raise OpticsError(
            f"{name} must have 3 components; got size {arr.size}"
        )
    if not np.all(np.isfinite(arr)):
        raise OpticsError(f"{name} must be finite; got {arr}")
    return arr


def create_oriented_parallel_ray_grid(
    *,
    center: tuple[float, float, float],
    direction: tuple[float, float, float],
    up: tuple[float, float, float],
    width: float,
    height: float,
    sample_count_y: int,
    sample_count_z: int,
) -> RayBundle:
    """Build a rectangular grid of parallel rays on an oriented source plane.

    Actual STL legacy-style optical scan smoke check; **not a
    physical PET-bottle validation**. ``direction`` is normalized.
    ``up`` is projected onto the plane perpendicular to
    ``direction`` and renormalized; passing ``up`` parallel to
    ``direction`` raises :class:`OpticsError`. The in-plane
    horizontal axis is ``right = cross(up_perp, direction)`` (same
    handedness convention as :func:`create_detector_plane`). The
    source plane spans ``width`` along ``right`` and ``height``
    along ``up_perp``; ``sample_count_y`` samples ``right`` and
    ``sample_count_z`` samples ``up_perp``.

    The total ray count is ``sample_count_y * sample_count_z``.
    The function does **no** I/O and **no** randomization.
    """
    c = _as_3vec(center, "center")
    d = _as_3vec(direction, "direction")
    u_in = _as_3vec(up, "up")

    if not (math.isfinite(width) and math.isfinite(height)):
        raise OpticsError(
            f"width and height must be finite; got "
            f"width={width}, height={height}"
        )
    if width <= 0.0 or height <= 0.0:
        raise OpticsError(
            f"width and height must be > 0; got "
            f"width={width}, height={height}"
        )

    if int(sample_count_y) < 1 or int(sample_count_z) < 1:
        raise OpticsError(
            f"sample_count_y and sample_count_z must be >= 1; got "
            f"sample_count_y={sample_count_y}, "
            f"sample_count_z={sample_count_z}"
        )

    d_norm = float(np.linalg.norm(d))
    if d_norm == 0.0:
        raise OpticsError("direction has zero length; cannot normalize")
    d_unit = d / d_norm

    up_proj = u_in - float(np.dot(u_in, d_unit)) * d_unit
    up_proj_norm = float(np.linalg.norm(up_proj))
    if up_proj_norm < _UP_PROJECTION_EPS:
        raise OpticsError("up must not be parallel to direction")
    u = up_proj / up_proj_norm

    r = np.cross(u, d_unit)

    ny = int(sample_count_y)
    nz = int(sample_count_z)

    if ny == 1:
        ys = np.array([0.0], dtype=float)
    else:
        ys = np.linspace(-0.5 * width, 0.5 * width, ny)
    if nz == 1:
        zs = np.array([0.0], dtype=float)
    else:
        zs = np.linspace(-0.5 * height, 0.5 * height, nz)

    yv, zv = np.meshgrid(ys, zs, indexing="xy")
    flat_y = yv.ravel()
    flat_z = zv.ravel()

    origins = (
        c.reshape(1, 3)
        + flat_y.reshape(-1, 1) * r.reshape(1, 3)
        + flat_z.reshape(-1, 1) * u.reshape(1, 3)
    )
    directions = np.broadcast_to(
        d_unit.reshape(1, 3), origins.shape,
    ).copy()

    return make_ray_bundle(origins, directions)


def _validate_angles_distances(
    angles: tuple[float, ...],
    distances: tuple[float, ...],
) -> None:
    for i, a in enumerate(angles):
        if not math.isfinite(float(a)):
            raise OpticsError(
                f"angles_degrees[{i}] must be finite; got {a}"
            )
    for i, d in enumerate(distances):
        if not math.isfinite(float(d)):
            raise OpticsError(
                f"detector_distances[{i}] must be finite; got {d}"
            )
        if float(d) <= 0.0:
            raise OpticsError(
                f"detector_distances[{i}] must be > 0; got {d}"
            )


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


def run_legacy_pet_water_angle_distance_scan(
    *,
    setup: LegacyPetWaterTraceSetup,
    angles_degrees: tuple[float, ...],
    detector_distances: tuple[float, ...],
    source_width: float,
    source_height: float,
    sample_count_y: int,
    sample_count_z: int,
    detector_size: float,
    detector_resolution: tuple[int, int],
    epsilon: float = 0.1,
    incident_irradiance: float = 1.0,
    thresholds: tuple[float, ...] = _DEFAULT_THRESHOLDS,
    top_percent: float = _DEFAULT_TOP_PERCENT,
    source_radius: float = _DEFAULT_SOURCE_RADIUS,
) -> LegacyOpticalScanResult:
    """Run a legacy-style angle / detector-distance optical scan.

    Actual STL legacy-style optical scan smoke check; **not a
    physical PET-bottle validation**. For each angle in
    ``angles_degrees``:

    1. Rotate the source position, propagation direction, and
       in-plane "up" around the world y-axis. The unrotated
       source center is ``ref_point + (source_radius, 0, 0)``
       with direction ``(-1, 0, 0)`` and up ``(0, 0, 1)``;
       ``ref_point`` is the bounding-box center of the scaled
       shell mesh.
    2. Build the source ray grid via
       :func:`create_oriented_parallel_ray_grid`.
    3. Run :func:`run_multi_mesh_trace` once on
       ``setup.step_specs``.
    4. For each detector distance, place a detector plane at
       ``ref_point + R(-distance, 0, 0)`` with normal pointing
       back toward the source, accumulate hits with cumulative
       Fresnel transmission weights, normalize to the relative
       irradiance surrogate, and reduce to ``Cmax`` / ``C99``.

    The trace is **run once per angle** — detectors are cheap
    plane intersections of the surviving rays. Entries are
    emitted in angle-major order (all detectors of angle[0],
    then all of angle[1], etc.). On a tie the aggregate maxima
    select the first occurrence.

    Empty ``angles_degrees`` or empty ``detector_distances``
    produce an empty result with ``angle_count`` and
    ``detector_count`` reflecting the original input lengths and
    every aggregate field set to ``None``.

    Raises
    ------
    OpticsError
        On invalid setup type, invalid sample counts, invalid
        source / detector dimensions, non-finite angles, or
        non-finite / non-positive detector distances.
    """
    if not isinstance(setup, LegacyPetWaterTraceSetup):
        raise OpticsError(
            "setup must be a LegacyPetWaterTraceSetup; got "
            f"{type(setup).__name__}"
        )

    angles = tuple(float(a) for a in angles_degrees)
    distances = tuple(float(d) for d in detector_distances)
    _validate_angles_distances(angles, distances)

    if not (
        math.isfinite(float(source_width))
        and math.isfinite(float(source_height))
    ):
        raise OpticsError(
            f"source_width and source_height must be finite; got "
            f"source_width={source_width}, "
            f"source_height={source_height}"
        )
    if float(source_width) <= 0.0 or float(source_height) <= 0.0:
        raise OpticsError(
            f"source_width and source_height must be > 0; got "
            f"source_width={source_width}, "
            f"source_height={source_height}"
        )
    if not math.isfinite(float(detector_size)) or float(detector_size) <= 0.0:
        raise OpticsError(
            f"detector_size must be a finite float > 0; got "
            f"{detector_size}"
        )
    if (
        not math.isfinite(float(source_radius))
        or float(source_radius) <= 0.0
    ):
        raise OpticsError(
            f"source_radius must be a finite float > 0; got "
            f"{source_radius}"
        )

    if (
        not math.isfinite(float(epsilon)) or float(epsilon) < 0.0
    ):
        raise OpticsError(
            f"epsilon must be a finite float >= 0; got {epsilon}"
        )

    res = tuple(detector_resolution)
    if len(res) != 2:
        raise OpticsError(
            f"detector_resolution must be (ny, nx); got {res}"
        )

    bounds = np.asarray(setup.shell_mesh.bounds, dtype=float)
    ref_point = 0.5 * (bounds[0] + bounds[1])
    source_area = float(source_width) * float(source_height)

    entries: list[LegacyOpticalScanEntry] = []

    if len(angles) == 0 or len(distances) == 0:
        return LegacyOpticalScanResult(
            entries=(),
            angle_count=len(angles),
            detector_count=len(distances),
            max_relative_irradiance_angle=None,
            max_relative_irradiance_detector_distance=None,
            max_relative_irradiance=None,
            max_c99_angle=None,
            max_c99_detector_distance=None,
            max_c99=None,
        )

    for angle_deg in angles:
        rotation = _rotation_matrix_around_y(math.radians(angle_deg))
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

        for distance in distances:
            local_detector_center = np.array(
                [-float(distance), 0.0, 0.0],
            )
            local_detector_normal = np.array([1.0, 0.0, 0.0])
            detector_center = ref_point + rotation @ local_detector_center
            detector_normal = rotation @ local_detector_normal
            detector_up = rotation @ local_up

            detector = create_detector_plane(
                center=tuple(
                    float(x) for x in detector_center.tolist()
                ),
                normal=tuple(
                    float(x) for x in detector_normal.tolist()
                ),
                up=tuple(float(x) for x in detector_up.tolist()),
                width=float(detector_size),
                height=float(detector_size),
            )
            detector_grid = create_detector_grid(
                width=float(detector_size),
                height=float(detector_size),
                resolution=(int(res[0]), int(res[1])),
            )
            hits = intersect_detector_plane(
                trace.final_rays, detector,
            )
            accumulation = accumulate_detector_hits(
                hits, detector_grid,
                weights=trace.final_ray_weights,
            )
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

            entries.append(
                LegacyOpticalScanEntry(
                    angle_degrees=float(angle_deg),
                    detector_distance=float(distance),
                    ray_count=int(rays.ray_count),
                    final_ray_count=int(trace.final_rays.ray_count),
                    detector_hits=int(accumulation.total_hits),
                    total_weight=float(accumulation.total_weight),
                    max_relative_irradiance=float(
                        surrogate.relative_irradiance_map.max()
                    ),
                    cmax=float(metrics.cmax),
                    c99=float(metrics.c99),
                    termination_reason=str(
                        trace.termination_reason
                    ),
                )
            )

    max_irr_idx = max(
        range(len(entries)),
        key=lambda i: entries[i].max_relative_irradiance,
    )
    max_c99_idx = max(
        range(len(entries)),
        key=lambda i: entries[i].c99,
    )
    e_irr = entries[max_irr_idx]
    e_c99 = entries[max_c99_idx]

    return LegacyOpticalScanResult(
        entries=tuple(entries),
        angle_count=len(angles),
        detector_count=len(distances),
        max_relative_irradiance_angle=float(e_irr.angle_degrees),
        max_relative_irradiance_detector_distance=float(
            e_irr.detector_distance,
        ),
        max_relative_irradiance=float(e_irr.max_relative_irradiance),
        max_c99_angle=float(e_c99.angle_degrees),
        max_c99_detector_distance=float(e_c99.detector_distance),
        max_c99=float(e_c99.c99),
    )
