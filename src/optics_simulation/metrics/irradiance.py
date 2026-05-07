"""Relative irradiance surrogate normalization foundation.

Converts detector accumulation weights into a relative irradiance
surrogate using source-plane area, ray count, and detector pixel
area; **still not a calibrated W/m^2 measurement**. The output is
a unit-less density that captures the spatial structure of the
caustic concentration relative to the incident irradiance, but
does not anchor to any physical W/m^2 calibration: no spectral
integration, no polarization tracking, no pixel spectral
response, no material absorptivity, no off-axis foreshortening
correction is applied.

Formulation
-----------
Given a :class:`DetectorAccumulationResult` (with its
``weight_map``), the corresponding :class:`DetectorGrid`, the
source-plane ``ray_count``, ``source_area``, and a normalization
``incident_irradiance``:

    total_incident_power   = incident_irradiance * source_area
    incident_power_per_ray = total_incident_power / ray_count
    detector_power_map     = accumulation.weight_map
                             * incident_power_per_ray
    pixel_area             = detector_grid.pixel_width
                             * detector_grid.pixel_height
    irradiance_map         = detector_power_map / pixel_area
    relative_irradiance_map = irradiance_map / incident_irradiance
    total_detector_power   = detector_power_map.sum()

Algebraic note
--------------
The relative irradiance map collapses to a unit-less density:

    relative_irradiance_map
        = weight_map * source_area / (ray_count * pixel_area)

so ``incident_irradiance`` cancels out in
``relative_irradiance_map`` but is preserved on
``detector_power_map`` and ``total_incident_power`` for callers
that want the scale.

Out of scope
------------
This module does not modify ``compute_optical_metrics`` /
``OpticalMetrics`` / ``MetricsError``, and does not auto-pipeline
into ``run_baseline_angle_scan`` / ``run_angle_distance_sweep``.
It does not perform calibrated W/m^2 conversion, solar spectrum
integration, pixel spectral absorption, material absorptivity
weighting, or off-axis geometric foreshortening correction.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optics_simulation.metrics.optical import MetricsError
from optics_simulation.optics.detector_accumulation import (
    DetectorAccumulationResult,
    DetectorGrid,
)


@dataclass(frozen=True)
class DetectorIrradianceSurrogate:
    detector_power_map: np.ndarray         # (ny, nx) float, >= 0
    relative_irradiance_map: np.ndarray    # (ny, nx) float, >= 0; unit-less
    pixel_area: float
    source_area: float
    incident_irradiance: float
    total_incident_power: float
    incident_power_per_ray: float
    total_detector_power: float
    ray_count: int
    detector_resolution: tuple[int, int]
    normalization_mode: str = "source_area_per_ray_over_pixel_area"


def _check_positive_finite_float(
    value: float, *, name: str
) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise MetricsError(
            f"{name} must be a finite float > 0; got {value!r}"
        ) from exc
    if not np.isfinite(v):
        raise MetricsError(
            f"{name} must be finite; got {v}"
        )
    if v <= 0.0:
        raise MetricsError(
            f"{name} must be > 0; got {v}"
        )
    return v


def compute_relative_irradiance_surrogate(
    accumulation: DetectorAccumulationResult,
    detector_grid: DetectorGrid,
    *,
    ray_count: int,
    source_area: float,
    incident_irradiance: float = 1.0,
) -> DetectorIrradianceSurrogate:
    """Compute the relative irradiance surrogate for a detector map.

    Converts detector accumulation weights into a relative
    irradiance surrogate using source-plane area, ray count, and
    detector pixel area; **still not a calibrated W/m^2
    measurement**.

    Formulation::

        total_incident_power   = incident_irradiance * source_area
        incident_power_per_ray = total_incident_power / ray_count
        detector_power_map     = accumulation.weight_map
                                 * incident_power_per_ray
        pixel_area             = detector_grid.pixel_width
                                 * detector_grid.pixel_height
        irradiance_map         = detector_power_map / pixel_area
        relative_irradiance_map = irradiance_map / incident_irradiance
        total_detector_power   = detector_power_map.sum()

    Equivalent algebraic form (the relative map is unit-less)::

        relative_irradiance_map
            = weight_map * source_area / (ray_count * pixel_area)

    so ``incident_irradiance`` cancels out in
    ``relative_irradiance_map`` but is preserved on
    ``detector_power_map`` and ``total_incident_power``.

    Parameters
    ----------
    accumulation
        :class:`DetectorAccumulationResult` carrying the detector
        ``weight_map`` to be normalized.
    detector_grid
        :class:`DetectorGrid` whose ``pixel_width`` and
        ``pixel_height`` define the per-pixel area used in the
        normalization, and whose ``resolution`` must match
        ``accumulation.weight_map.shape``.
    ray_count
        Strict positive ``int`` count of source-plane rays whose
        accumulated weights are stored in ``weight_map``.
        ``bool`` values are rejected.
    source_area
        Finite positive ``float`` source-plane area. Caller
        authoritative; this function does not infer it from ray
        geometry.
    incident_irradiance
        Finite positive ``float`` normalization scale. Defaults to
        ``1.0``. Cancels out in ``relative_irradiance_map`` but
        is preserved on ``detector_power_map`` and
        ``total_incident_power``.

    Raises
    ------
    MetricsError
        On invalid types, non-positive or non-finite scalars, a
        non-finite or negative ``weight_map``, a zero or negative
        pixel area, or a ``weight_map`` shape that does not match
        ``detector_grid.resolution``.
    """
    if not isinstance(accumulation, DetectorAccumulationResult):
        raise MetricsError(
            f"accumulation must be a DetectorAccumulationResult; "
            f"got {type(accumulation).__name__}"
        )
    if not isinstance(detector_grid, DetectorGrid):
        raise MetricsError(
            f"detector_grid must be a DetectorGrid; "
            f"got {type(detector_grid).__name__}"
        )

    if isinstance(ray_count, bool) or not isinstance(ray_count, int):
        raise MetricsError(
            f"ray_count must be a positive int; got {ray_count!r}"
        )
    if ray_count <= 0:
        raise MetricsError(
            f"ray_count must be a positive int; got {ray_count}"
        )

    source_area_f = _check_positive_finite_float(
        source_area, name="source_area"
    )
    incident_irradiance_f = _check_positive_finite_float(
        incident_irradiance, name="incident_irradiance"
    )

    pixel_width = float(detector_grid.pixel_width)
    pixel_height = float(detector_grid.pixel_height)
    if not (np.isfinite(pixel_width) and np.isfinite(pixel_height)):
        raise MetricsError(
            f"detector pixel area must be finite and > 0; got "
            f"pixel_width={pixel_width}, pixel_height={pixel_height}"
        )
    pixel_area = pixel_width * pixel_height
    if not np.isfinite(pixel_area) or pixel_area <= 0.0:
        raise MetricsError(
            f"detector pixel area must be finite and > 0; got "
            f"pixel_width={pixel_width}, pixel_height={pixel_height}"
        )

    weight_map = np.asarray(accumulation.weight_map, dtype=float)
    expected_shape = tuple(detector_grid.resolution)
    if weight_map.shape != expected_shape:
        raise MetricsError(
            f"accumulation.weight_map shape {weight_map.shape} does not "
            f"match detector_grid.resolution {expected_shape}"
        )
    if not np.isfinite(weight_map).all():
        raise MetricsError(
            "accumulation.weight_map contains NaN or inf values"
        )
    if (weight_map < 0.0).any():
        bad = int(np.argmin(weight_map))
        raise MetricsError(
            f"accumulation.weight_map must be non-negative; first "
            f"violation at flat index {bad} "
            f"(value={float(weight_map.flat[bad])})"
        )

    total_incident_power = incident_irradiance_f * source_area_f
    incident_power_per_ray = total_incident_power / float(ray_count)
    detector_power_map = (weight_map * incident_power_per_ray).astype(
        float, copy=True
    )
    relative_irradiance_map = (
        detector_power_map / pixel_area / incident_irradiance_f
    ).astype(float, copy=True)
    total_detector_power = float(detector_power_map.sum())

    return DetectorIrradianceSurrogate(
        detector_power_map=detector_power_map,
        relative_irradiance_map=relative_irradiance_map,
        pixel_area=float(pixel_area),
        source_area=float(source_area_f),
        incident_irradiance=float(incident_irradiance_f),
        total_incident_power=float(total_incident_power),
        incident_power_per_ray=float(incident_power_per_ray),
        total_detector_power=total_detector_power,
        ray_count=int(ray_count),
        detector_resolution=expected_shape,
    )
