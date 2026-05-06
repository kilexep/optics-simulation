"""Detector distance sweep foundation.

Synthetic angle-distance optical sweep foundation. Extends the
single-detector :func:`run_baseline_angle_scan` to a 2D sweep over
incident angles and detector-plane z positions, returning per
(angle, detector_z) :class:`OpticalMetrics`. Caustic hotspots are
sensitive to the detector distance, so a single detector plane is
not enough to localize the worst-case ``C99`` for a given setup;
this module supplies the orchestration layer for a synthetic
``Risk(theta, d)`` foundation.

Research framing
----------------
This is a synthetic angle-distance optical sweep foundation for
local focusing / heat-risk metric comparison. It is **not** a real
PET-bottle validation, and it does not validate any fire-prevention
claim. Results from this foundation describe a synthetic geometry
under a synthetic ray grid; downstream tasks (visualization,
contribution maps, optimization, thermal modeling) are out of
scope here.

Per-angle trace reuse
---------------------
For each incident angle the multi-step trace is run **once**, and
the resulting ``trace.final_rays`` are intersected against every
``detector_z`` in turn. The detector pixel grid is also built once
(its dimensions are constant across the sweep). This avoids
re-tracing rays for every detector position.

Out of scope
------------
Original-vs-displaced comparison, visualization, file export,
CSV / Parquet logging, detector heatmap saving, real PET STL
ingestion, shell / wall-thickness modeling, water volume,
automatic medium tracking, Open3D backend, physical irradiance
unit conversion, pixel-area normalization, thermal modeling,
optimization, pattern generation, mesh displacement, and
config-file reads are all intentionally not implemented here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import trimesh

from optics_simulation.angle_scan.baseline import (
    AngleScanError,
    direction_from_incident_angle,
)
from optics_simulation.metrics import OpticalMetrics, compute_optical_metrics
from optics_simulation.optics import (
    accumulate_detector_hits,
    create_detector_grid,
    create_detector_plane,
    intersect_detector_plane,
    parallel_ray_grid,
    run_multi_step_trace,
)


@dataclass(frozen=True)
class PerAngleDistanceResult:
    angle_degrees: float
    detector_z: float
    direction: np.ndarray
    ray_count: int
    final_ray_count: int
    detector_hits: int
    metrics: OpticalMetrics
    termination_reason: str


@dataclass(frozen=True)
class AngleDistanceScanResult:
    per_result: tuple[PerAngleDistanceResult, ...]
    angle_count: int
    detector_count: int
    max_c99_angle: float | None
    max_c99_detector_z: float | None
    max_c99: float | None


def _empty_result(angle_count: int, detector_count: int) -> AngleDistanceScanResult:
    return AngleDistanceScanResult(
        per_result=(),
        angle_count=int(angle_count),
        detector_count=int(detector_count),
        max_c99_angle=None,
        max_c99_detector_z=None,
        max_c99=None,
    )


def _validated_finite_floats(
    values: Sequence[float],
    *,
    name: str,
) -> list[float]:
    out: list[float] = []
    for i, raw in enumerate(values):
        try:
            v = float(raw)
        except (TypeError, ValueError) as exc:
            raise AngleScanError(
                f"{name}[{i}] must be a finite float; got {raw!r}"
            ) from exc
        if not np.isfinite(v):
            raise AngleScanError(
                f"{name}[{i}] must be finite; got {v}"
            )
        out.append(v)
    return out


def _validated_center_xy(
    detector_center_xy: Sequence[float],
) -> tuple[float, float]:
    try:
        xy = tuple(detector_center_xy)
    except TypeError as exc:
        raise AngleScanError(
            f"detector_center_xy must be a length-2 sequence of finite "
            f"floats; got {detector_center_xy!r}"
        ) from exc
    if len(xy) != 2:
        raise AngleScanError(
            f"detector_center_xy must have exactly 2 entries (x, y); "
            f"got length {len(xy)}"
        )
    try:
        cx = float(xy[0])
        cy = float(xy[1])
    except (TypeError, ValueError) as exc:
        raise AngleScanError(
            f"detector_center_xy entries must be finite floats; "
            f"got {detector_center_xy!r}"
        ) from exc
    if not (np.isfinite(cx) and np.isfinite(cy)):
        raise AngleScanError(
            f"detector_center_xy entries must be finite; "
            f"got ({cx}, {cy})"
        )
    return cx, cy


def run_angle_distance_sweep(
    mesh: trimesh.Trimesh,
    angles_degrees: Sequence[float],
    detector_z_values: Sequence[float],
    *,
    ray_grid_config: dict,
    interface_sequence: Sequence[tuple[float, float]],
    detector_width: float,
    detector_height: float,
    detector_resolution: tuple[int, int],
    detector_center_xy: tuple[float, float] = (0.0, 0.0),
    detector_normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
    detector_up: tuple[float, float, float] = (0.0, 1.0, 0.0),
    incident_reference: float = 1.0,
    thresholds: tuple[float, ...] = (2.0, 5.0, 10.0),
    top_percent: float = 1.0,
) -> AngleDistanceScanResult:
    """Run the synthetic-mesh optical pipeline over angles x detector z.

    For each incident angle, a ray grid is generated and a
    multi-step trace is performed **once**. The resulting
    ``trace.final_rays`` are then intersected against every
    detector position in ``detector_z_values`` and converted to
    :class:`OpticalMetrics` via the pixel grid + accumulation +
    metrics chain. Results are returned in **angle-major** order:
    for ``angles=[a0, a1]`` and ``detector_z_values=[d0, d1, d2]``,
    ``per_result`` is::

        (a0, d0), (a0, d1), (a0, d2),
        (a1, d0), (a1, d1), (a1, d2)

    so ``len(per_result) == len(angles) * len(detector_z_values)``.

    ``ray_grid_config`` is forwarded to :func:`parallel_ray_grid`
    after the ``"direction"`` key is overridden by the per-angle
    direction. The caller's dict is **not** mutated; a shallow
    copy is taken internally.

    Empty-axis policy
    -----------------
    If either ``angles_degrees`` or ``detector_z_values`` is empty,
    ``per_result`` is ``()`` and ``max_c99_angle`` /
    ``max_c99_detector_z`` / ``max_c99`` are all ``None``.
    ``angle_count`` and ``detector_count`` echo the actual input
    lengths so callers can still distinguish the two empty cases.

    Tie-break for ``max_c99``
    -------------------------
    ``max C99`` ties are resolved by **first occurrence in
    ``per_result`` order**. Because the result order is
    angle-major, this is equivalent to "angle input order first,
    then detector_z input order". This is **not** a "smallest
    distance" rule; it depends on the input order rather than on
    the value of the distance.

    Detector geometry
    -----------------
    For each ``z`` in ``detector_z_values`` a detector plane with
    center ``(detector_center_xy[0], detector_center_xy[1], z)``
    is constructed. ``detector_normal`` and ``detector_up`` apply
    to every plane in the sweep. The pixel grid
    (``detector_width`` x ``detector_height`` at
    ``detector_resolution``) is built once and reused. Invalid
    detector geometry (non-positive width / height, ``up`` parallel
    to ``normal``, malformed resolution) propagates as
    :class:`OpticsError` from
    :func:`create_detector_plane` / :func:`create_detector_grid`.

    Validation done in this module
    ------------------------------
    - Every ``angles_degrees[i]`` must be a finite float.
    - Every ``detector_z_values[i]`` must be a finite float.
    - ``detector_center_xy`` must be a length-2 sequence of finite
      floats.

    Any of these raises :class:`AngleScanError`. Duplicate
    ``detector_z`` values are accepted and preserved in input
    order.
    """
    angle_list = _validated_finite_floats(
        angles_degrees, name="angles_degrees"
    )
    distance_list = _validated_finite_floats(
        detector_z_values, name="detector_z_values"
    )
    center_xy = _validated_center_xy(detector_center_xy)

    if not angle_list or not distance_list:
        return _empty_result(len(angle_list), len(distance_list))

    detector_grid = create_detector_grid(
        width=detector_width,
        height=detector_height,
        resolution=detector_resolution,
    )

    grid_kwargs = dict(ray_grid_config)
    grid_kwargs.pop("direction", None)

    per_list: list[PerAngleDistanceResult] = []
    cx, cy = center_xy

    for angle in angle_list:
        direction = direction_from_incident_angle(angle, plane="xz")
        rays = parallel_ray_grid(
            direction=tuple(direction),
            **grid_kwargs,
        )
        trace = run_multi_step_trace(mesh, rays, interface_sequence)

        for z in distance_list:
            detector = create_detector_plane(
                center=(cx, cy, z),
                normal=detector_normal,
                up=detector_up,
                width=detector_width,
                height=detector_height,
            )
            hits = intersect_detector_plane(trace.final_rays, detector)
            accum = accumulate_detector_hits(hits, detector_grid)
            metrics = compute_optical_metrics(
                accum.weight_map,
                incident_reference=incident_reference,
                thresholds=thresholds,
                top_percent=top_percent,
            )
            per_list.append(
                PerAngleDistanceResult(
                    angle_degrees=float(angle),
                    detector_z=float(z),
                    direction=direction.copy(),
                    ray_count=int(rays.ray_count),
                    final_ray_count=int(trace.final_rays.ray_count),
                    detector_hits=int(accum.total_hits),
                    metrics=metrics,
                    termination_reason=trace.termination_reason,
                )
            )

    c99_values = np.array(
        [p.metrics.c99 for p in per_list], dtype=float
    )
    idx = int(np.argmax(c99_values))
    chosen = per_list[idx]
    return AngleDistanceScanResult(
        per_result=tuple(per_list),
        angle_count=len(angle_list),
        detector_count=len(distance_list),
        max_c99_angle=float(chosen.angle_degrees),
        max_c99_detector_z=float(chosen.detector_z),
        max_c99=float(chosen.metrics.c99),
    )
