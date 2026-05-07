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

from optics_simulation.metrics import (
    DetectorIrradianceSurrogate,
    OpticalMetrics,
    compute_optical_metrics,
    compute_relative_irradiance_surrogate,
)
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
    irradiance_surrogate: DetectorIrradianceSurrogate | None = None


def _infer_source_area_from_ray_grid_config(
    ray_grid_config: dict,
) -> float:
    """Infer the source-plane area from ``ray_grid_config``.

    Reads ``x_range`` and ``y_range`` from ``ray_grid_config`` and
    returns ``(x_max - x_min) * (y_max - y_min)``. Both ranges must
    be length-2 finite-float sequences with strictly positive width.
    Raises :class:`AngleScanError` on any malformation. Only called
    when ``use_relative_irradiance=True`` and ``source_area`` is not
    explicitly provided.
    """
    if "x_range" not in ray_grid_config or "y_range" not in ray_grid_config:
        raise AngleScanError(
            "ray_grid_config must contain 'x_range' and 'y_range' to "
            "infer source_area for use_relative_irradiance=True"
        )
    x_range = ray_grid_config["x_range"]
    y_range = ray_grid_config["y_range"]
    try:
        x_min = float(x_range[0])
        x_max = float(x_range[1])
        y_min = float(y_range[0])
        y_max = float(y_range[1])
    except (TypeError, ValueError, IndexError) as exc:
        raise AngleScanError(
            f"ray_grid_config x_range and y_range must be length-2 "
            f"finite-float sequences; got x_range={x_range!r}, "
            f"y_range={y_range!r}"
        ) from exc
    if not (
        np.isfinite(x_min) and np.isfinite(x_max)
        and np.isfinite(y_min) and np.isfinite(y_max)
    ):
        raise AngleScanError(
            f"ray_grid_config x_range and y_range must be finite; got "
            f"x_range=({x_min}, {x_max}), y_range=({y_min}, {y_max})"
        )
    if x_max <= x_min or y_max <= y_min:
        raise AngleScanError(
            f"ray_grid_config x_range and y_range must satisfy max > "
            f"min; got x_range=({x_min}, {x_max}), "
            f"y_range=({y_min}, {y_max})"
        )
    return (x_max - x_min) * (y_max - y_min)


def _validate_source_area_value(value: float) -> float:
    """Validate caller-supplied ``source_area`` for relative-irradiance mode."""
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise AngleScanError(
            f"source_area must be a finite float > 0; got {value!r}"
        ) from exc
    if not np.isfinite(v) or v <= 0.0:
        raise AngleScanError(
            f"source_area must be finite and > 0; got {v}"
        )
    return v


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
    use_relative_irradiance: bool = False,
    source_area: float | None = None,
    incident_irradiance: float = 1.0,
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

    ``use_relative_irradiance`` (default ``False``) controls whether
    the per-angle ``OpticalMetrics`` is computed on the raw
    ``accumulation.weight_map`` (``False``, historical behavior) or
    on the relative-irradiance map produced by
    :func:`compute_relative_irradiance_surrogate` (``True``). Uses
    source-plane area, ray count, and detector pixel area to compute
    a relative irradiance surrogate; **still not a calibrated
    W/m^2 measurement**.

    When ``use_relative_irradiance=True``, ``source_area`` is read
    from the caller (must be finite and ``> 0``) or, if ``None``,
    inferred from ``ray_grid_config["x_range"] / ["y_range"]`` as
    ``(x_max - x_min) * (y_max - y_min)``. ``incident_irradiance``
    defaults to ``1.0``; the relative-irradiance map cancels it
    out, but ``DetectorIrradianceSurrogate.detector_power_map`` and
    ``total_incident_power`` still scale by it. The optical-metric
    call uses ``incident_reference=1.0`` because the relative map
    is already normalized; the ``incident_reference`` argument
    therefore applies only to the ``use_relative_irradiance=False``
    path.

    When ``use_relative_irradiance=False``, ``source_area`` is not
    validated (it is ignored). ``PerAngleResult.irradiance_surrogate``
    is ``None`` in that path and a populated
    :class:`DetectorIrradianceSurrogate` in the relative path.
    """
    angle_list = [float(a) for a in angles_degrees]
    if not angle_list:
        return _empty_result()

    grid_kwargs = dict(ray_grid_config)
    grid_kwargs.pop("direction", None)

    source_area_value: float | None = None
    if use_relative_irradiance:
        if source_area is None:
            source_area_value = _infer_source_area_from_ray_grid_config(
                ray_grid_config
            )
        else:
            source_area_value = _validate_source_area_value(source_area)

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
        if use_relative_irradiance:
            surrogate = compute_relative_irradiance_surrogate(
                accum,
                detector_grid,
                ray_count=int(rays.ray_count),
                source_area=source_area_value,
                incident_irradiance=incident_irradiance,
            )
            metrics = compute_optical_metrics(
                surrogate.relative_irradiance_map,
                incident_reference=1.0,
                thresholds=thresholds,
                top_percent=top_percent,
            )
        else:
            surrogate = None
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
                irradiance_surrogate=surrogate,
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
