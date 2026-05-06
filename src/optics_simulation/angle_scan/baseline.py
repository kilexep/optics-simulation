"""Baseline angle scan foundation.

Orchestration-only layer that runs the existing synthetic-mesh
optical pipeline (parallel ray grid -> multi-step trace -> detector
plane intersection -> detector accumulation -> optical metrics) once
per incident angle and collects per-angle :class:`OpticalMetrics`.

Convention
----------
- ``angle_degrees = 0`` corresponds to ``direction = (0, 0, -1)``
  (normal incidence, downward along -z).
- A positive angle in the ``xz`` plane gives
  ``direction = (sin(theta), 0, -cos(theta))``.
- Returned directions are unit norm. No clamping or wrapping of the
  input angle is performed; callers decide whether very large angles
  are physically meaningful.

Out of scope
------------
Real PET STL ingestion, Open3D backend, contribution maps, hotspot
backtracking, pattern generation, optimization, thermal modeling,
visualization, CSV/Parquet export, config-file reads, automatic
medium tracking, physical irradiance unit conversion, pixel-area
normalization, bottle rotation scan, and target distance sweep are
all intentionally not implemented here. This module is a foundation
those layers will compose on top of.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import trimesh

from optics_simulation.metrics import OpticalMetrics, compute_optical_metrics
from optics_simulation.optics import (
    DetectorGrid,
    DetectorPlane,
    accumulate_detector_hits,
    intersect_detector_plane,
    parallel_ray_grid,
    run_multi_step_trace,
)


class AngleScanError(Exception):
    """Raised for invalid angle scan inputs (unsupported plane, etc.)."""


@dataclass(frozen=True)
class PerAngleResult:
    angle_degrees: float
    direction: np.ndarray
    ray_count: int
    final_ray_count: int
    detector_hits: int
    metrics: OpticalMetrics
    termination_reason: str


@dataclass(frozen=True)
class AngleScanResult:
    per_angle: tuple[PerAngleResult, ...]
    angle_count: int
    max_c99_angle: float | None
    max_c99: float | None


def direction_from_incident_angle(
    angle_degrees: float,
    *,
    plane: str = "xz",
) -> np.ndarray:
    """Convert an incident angle (degrees) into a unit ray direction.

    Only ``plane="xz"`` is supported in this task: the returned
    direction is ``(sin(theta), 0, -cos(theta))`` where
    ``theta = radians(angle_degrees)``. ``angle_degrees = 0`` yields
    ``(0, 0, -1)`` (normal incidence, downward along -z). The input
    angle is not clamped or wrapped. Any other ``plane`` value
    raises :class:`AngleScanError`; ``yz`` / azimuth extensions are
    deferred to the bottle rotation / 3D angle scan tasks.
    """
    if plane != "xz":
        raise AngleScanError(
            f"plane must be 'xz' in the baseline angle scan; got {plane!r}"
        )

    theta = np.deg2rad(float(angle_degrees))
    direction = np.array(
        [np.sin(theta), 0.0, -np.cos(theta)],
        dtype=float,
    )
    return direction


def _empty_result() -> AngleScanResult:
    return AngleScanResult(
        per_angle=(),
        angle_count=0,
        max_c99_angle=None,
        max_c99=None,
    )


def run_baseline_angle_scan(
    mesh: trimesh.Trimesh,
    angles_degrees: Sequence[float],
    *,
    ray_grid_config: dict,
    interface_sequence: Sequence[tuple[float, float]],
    detector: DetectorPlane,
    detector_grid: DetectorGrid,
    incident_reference: float = 1.0,
    thresholds: tuple[float, ...] = (2.0, 5.0, 10.0),
    top_percent: float = 1.0,
    use_power_weights: bool = False,
) -> AngleScanResult:
    """Run the synthetic-mesh optical pipeline once per incident angle.

    For each entry in ``angles_degrees``:

        direction_from_incident_angle
        -> parallel_ray_grid
        -> run_multi_step_trace
        -> intersect_detector_plane
        -> accumulate_detector_hits
        -> compute_optical_metrics

    No new physics is introduced here; this is purely an orchestration
    layer over the existing modules.

    ``ray_grid_config`` is a caller-supplied dict of keyword arguments
    forwarded to :func:`parallel_ray_grid` (for example
    ``{"origin_plane_z": 20.0, "x_range": (-7.5, 7.5),
    "y_range": (-7.5, 7.5), "nx": 11, "ny": 11}``); the ``direction``
    key, if present, is overridden by the per-angle direction. No
    config files are read by this function.

    An empty ``angles_degrees`` returns an :class:`AngleScanResult`
    with ``per_angle == ()``, ``angle_count == 0``, and both
    ``max_c99_angle`` and ``max_c99`` set to ``None``. Otherwise
    ``max_c99_angle`` is the angle whose :class:`OpticalMetrics`
    has the largest ``c99``; ties are broken by first occurrence
    (``np.argmax`` semantics). Note that a higher ``c99`` means a
    more concentrated caustic, i.e. a higher-risk angle, not a
    "better" angle.

    ``use_power_weights`` (default ``False``) controls how detector
    hits are accumulated. When ``False``, every detector hit
    contributes a unit weight (the historical ray-count surrogate).
    When ``True``, the cumulative Fresnel transmission weight from
    :class:`MultiStepTraceResult.final_ray_weights` is forwarded to
    :func:`accumulate_detector_hits` as the ``weights`` argument,
    making ``c99`` / ``cmax`` a transmitted-power surrogate. Adds
    cumulative Fresnel transmission weighting to detector
    accumulation; **still not a full physical irradiance calibration**
    (no pixel-area normalization, no spectral integration, no
    polarization, no dispersion, no reflected branches, no source
    intensity calibration).
    """
    angle_list = [float(a) for a in angles_degrees]
    if not angle_list:
        return _empty_result()

    grid_kwargs = dict(ray_grid_config)
    grid_kwargs.pop("direction", None)

    per_angle_list: list[PerAngleResult] = []
    for angle in angle_list:
        direction = direction_from_incident_angle(angle, plane="xz")
        rays = parallel_ray_grid(
            direction=tuple(direction),
            **grid_kwargs,
        )
        trace = run_multi_step_trace(mesh, rays, interface_sequence)
        hits = intersect_detector_plane(trace.final_rays, detector)
        weights_for_accum = (
            trace.final_ray_weights if use_power_weights else None
        )
        accum = accumulate_detector_hits(
            hits, detector_grid, weights=weights_for_accum
        )
        metrics = compute_optical_metrics(
            accum.weight_map,
            incident_reference=incident_reference,
            thresholds=thresholds,
            top_percent=top_percent,
        )
        per_angle_list.append(
            PerAngleResult(
                angle_degrees=angle,
                direction=direction.copy(),
                ray_count=int(rays.ray_count),
                final_ray_count=int(trace.final_rays.ray_count),
                detector_hits=int(accum.total_hits),
                metrics=metrics,
                termination_reason=trace.termination_reason,
            )
        )

    c99_values = np.array(
        [p.metrics.c99 for p in per_angle_list], dtype=float
    )
    idx = int(np.argmax(c99_values))
    max_c99_angle = per_angle_list[idx].angle_degrees
    max_c99 = float(c99_values[idx])

    return AngleScanResult(
        per_angle=tuple(per_angle_list),
        angle_count=len(per_angle_list),
        max_c99_angle=max_c99_angle,
        max_c99=max_c99,
    )
