"""Optical angle-distance scan -> thermal target-heating coupling.

Synthetic normalized optical-to-thermal comparison foundation;
**not a physical PET-bottle fire-prevention validation**.
Consumes an :class:`AngleDistanceScanResult` whose entries
already carry a :class:`DetectorIrradianceSurrogate`
(``use_relative_irradiance=True`` path) and produces either:

- a scalar lumped target-heating result per entry via
  :func:`run_lumped_heating_over_angle_distance_scan`
  (``scalar_mode='max_relative_irradiance'``), or
- a per-pixel thermal-map result per entry via
  :func:`run_lumped_heating_maps_over_angle_distance_scan`
  (``map_mode='relative_irradiance_map'``).

Per-entry scalar chain (for ``scalar_mode='max_relative_irradiance'``)::

    s = entry.irradiance_surrogate
    max_rel = float(s.relative_irradiance_map.max())
    incident_flux_w_m2 = nominal_incident_irradiance_w_m2 * max_rel
    heating = simulate_lumped_target_heating(
        incident_flux_w_m2=incident_flux_w_m2, ...
    )

Per-entry map chain (for ``map_mode='relative_irradiance_map'``)::

    s = entry.irradiance_surrogate
    incident_flux_map = (
        nominal_incident_irradiance_w_m2 * s.relative_irradiance_map
    )
    heating_map = simulate_lumped_target_heating_map(
        incident_flux_map, ...
    )

Interpretation
--------------
- The optical input is a relative irradiance surrogate, **not
  calibrated W/m^2** unless the caller supplies a calibrated
  ``nominal_incident_irradiance_w_m2``.
- The thermal model is a simplified lumped target-heating
  surrogate; it is **not pyrolysis**, **not CFD**, **not
  combustion chemistry**, **not spatial conduction**, and **not
  ignition validation**. In the per-pixel map path each pixel is
  treated as an independent lumped target.
- ``max_temperature_*`` aggregate fields point at the entry whose
  per-target heating curve / heating map reached the highest
  temperature *under this synthetic setup only*; they do **not**
  identify a real PET-bottle risk angle or risk distance.
- Delta values across pattern variants are reported for comparison
  only; a negative delta is **not required**.

Out of scope
------------
Time-varying flux, ignition criterion, material database,
manufacturability validation, fire-prevention proof, calibrated
W/m^2 anchoring, spatial heat conduction, multi-layer / multi
material targets.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optics_simulation.angle_scan.distance_sweep import (
    AngleDistanceScanResult,
)
from optics_simulation.thermal.lumped_target import (
    LumpedTargetHeatingMapResult,
    LumpedTargetHeatingResult,
    ThermalError,
    simulate_lumped_target_heating,
    simulate_lumped_target_heating_map,
)


_VALID_SCALAR_MODES = ("max_relative_irradiance",)
_VALID_MAP_MODES = ("relative_irradiance_map",)


@dataclass(frozen=True)
class PerAngleDistanceHeatingResult:
    angle_degrees: float
    detector_z: float
    optical_c99: float
    optical_cmax: float
    max_relative_irradiance: float
    detector_hits: int
    final_ray_count: int
    termination_reason: str
    incident_flux_w_m2: float
    heating_result: LumpedTargetHeatingResult


@dataclass(frozen=True)
class AngleDistanceHeatingScanResult:
    per_result: tuple[PerAngleDistanceHeatingResult, ...]
    result_count: int
    max_temperature_angle: float | None
    max_temperature_detector_z: float | None
    max_temperature_k: float | None
    max_temperature_rise_k: float | None
    max_incident_flux_w_m2: float | None
    nominal_incident_irradiance_w_m2: float
    scalar_mode: str


@dataclass(frozen=True)
class PerAngleDistanceHeatingMapResult:
    angle_degrees: float
    detector_z: float
    optical_c99: float
    optical_cmax: float
    max_relative_irradiance: float
    detector_hits: int
    final_ray_count: int
    termination_reason: str
    incident_flux_map_max_w_m2: float
    heating_map_result: LumpedTargetHeatingMapResult


@dataclass(frozen=True)
class AngleDistanceHeatingMapScanResult:
    per_result: tuple[PerAngleDistanceHeatingMapResult, ...]
    result_count: int
    max_temperature_angle: float | None
    max_temperature_detector_z: float | None
    max_temperature_k: float | None
    max_temperature_rise_k: float | None
    max_top_percent_temperature_rise_k: float | None
    nominal_incident_irradiance_w_m2: float
    map_mode: str


def _check_finite_nonneg(value: float, *, name: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ThermalError(
            f"{name} must be a finite float >= 0; got {value!r}"
        ) from exc
    if not np.isfinite(v):
        raise ThermalError(f"{name} must be finite; got {v}")
    if v < 0.0:
        raise ThermalError(f"{name} must be >= 0; got {v}")
    return v


def _empty_result(
    nominal: float, scalar_mode: str,
) -> AngleDistanceHeatingScanResult:
    return AngleDistanceHeatingScanResult(
        per_result=(),
        result_count=0,
        max_temperature_angle=None,
        max_temperature_detector_z=None,
        max_temperature_k=None,
        max_temperature_rise_k=None,
        max_incident_flux_w_m2=None,
        nominal_incident_irradiance_w_m2=float(nominal),
        scalar_mode=str(scalar_mode),
    )


def _empty_map_result(
    nominal: float, map_mode: str,
) -> AngleDistanceHeatingMapScanResult:
    return AngleDistanceHeatingMapScanResult(
        per_result=(),
        result_count=0,
        max_temperature_angle=None,
        max_temperature_detector_z=None,
        max_temperature_k=None,
        max_temperature_rise_k=None,
        max_top_percent_temperature_rise_k=None,
        nominal_incident_irradiance_w_m2=float(nominal),
        map_mode=str(map_mode),
    )


def run_lumped_heating_over_angle_distance_scan(
    scan_result: AngleDistanceScanResult,
    *,
    nominal_incident_irradiance_w_m2: float,
    duration_s: float,
    dt_s: float,
    areal_heat_capacity_j_m2k: float,
    absorptivity: float = 1.0,
    h_conv_w_m2k: float = 0.0,
    emissivity: float = 0.0,
    ambient_temp_k: float = 293.15,
    initial_temp_k: float | None = None,
    threshold_temp_k: float | None = None,
    scalar_mode: str = "max_relative_irradiance",
) -> AngleDistanceHeatingScanResult:
    """Run lumped-target heating once per angle-distance entry.

    Synthetic normalized optical-to-thermal coupling; **not a
    physical PET-bottle fire-prevention validation**. Each
    :class:`PerAngleDistanceResult` in ``scan_result`` must carry a
    :class:`DetectorIrradianceSurrogate` (i.e. produced by
    ``run_angle_distance_sweep(..., use_relative_irradiance=True)``).
    The reduction from the relative-irradiance map to a scalar
    flux input is controlled by ``scalar_mode``; only
    ``"max_relative_irradiance"`` is supported in this foundation.

    Parameters
    ----------
    scan_result
        :class:`AngleDistanceScanResult` from the angle-distance
        sweep. Empty ``per_result`` returns an empty heating
        scan with all ``max_*`` fields ``None``.
    nominal_incident_irradiance_w_m2
        Scale multiplier applied to ``max_relative_irradiance`` to
        produce the thermal ``incident_flux_w_m2``. Must be finite
        and ``>= 0``. **Not calibrated W/m^2** unless the caller
        anchors it to a calibrated solar reference.
    duration_s, dt_s, areal_heat_capacity_j_m2k, absorptivity,
    h_conv_w_m2k, emissivity, ambient_temp_k, initial_temp_k,
    threshold_temp_k
        Forwarded verbatim to
        :func:`simulate_lumped_target_heating`. All thermal-side
        validation is delegated there.
    scalar_mode
        Reduction from the per-entry relative-irradiance map to a
        scalar flux input. ``"max_relative_irradiance"``
        (the only supported mode in this task) takes
        ``float(s.relative_irradiance_map.max())``.

    Raises
    ------
    ThermalError
        On invalid ``scan_result`` type, missing
        ``irradiance_surrogate`` on a non-empty entry, invalid
        ``scalar_mode``, invalid
        ``nominal_incident_irradiance_w_m2``, or any
        downstream :class:`ThermalError` from
        :func:`simulate_lumped_target_heating`.
    """
    if not isinstance(scan_result, AngleDistanceScanResult):
        raise ThermalError(
            f"scan_result must be an AngleDistanceScanResult; "
            f"got {type(scan_result).__name__}"
        )
    nominal = _check_finite_nonneg(
        nominal_incident_irradiance_w_m2,
        name="nominal_incident_irradiance_w_m2",
    )
    if scalar_mode not in _VALID_SCALAR_MODES:
        raise ThermalError(
            f"scalar_mode must be one of {_VALID_SCALAR_MODES}; "
            f"got {scalar_mode!r}"
        )

    if not scan_result.per_result:
        return _empty_result(nominal, scalar_mode)

    for i, entry in enumerate(scan_result.per_result):
        if entry.irradiance_surrogate is None:
            raise ThermalError(
                f"scan_result.per_result[{i}].irradiance_surrogate "
                f"is None; this coupling layer requires the angle "
                f"scan to be run with use_relative_irradiance=True"
            )

    per_list: list[PerAngleDistanceHeatingResult] = []
    for entry in scan_result.per_result:
        s = entry.irradiance_surrogate
        max_rel = float(s.relative_irradiance_map.max())
        incident_flux = float(nominal) * max_rel
        heating = simulate_lumped_target_heating(
            incident_flux_w_m2=incident_flux,
            duration_s=duration_s,
            dt_s=dt_s,
            areal_heat_capacity_j_m2k=areal_heat_capacity_j_m2k,
            absorptivity=absorptivity,
            h_conv_w_m2k=h_conv_w_m2k,
            emissivity=emissivity,
            ambient_temp_k=ambient_temp_k,
            initial_temp_k=initial_temp_k,
            threshold_temp_k=threshold_temp_k,
        )
        per_list.append(
            PerAngleDistanceHeatingResult(
                angle_degrees=float(entry.angle_degrees),
                detector_z=float(entry.detector_z),
                optical_c99=float(entry.metrics.c99),
                optical_cmax=float(entry.metrics.cmax),
                max_relative_irradiance=float(max_rel),
                detector_hits=int(entry.detector_hits),
                final_ray_count=int(entry.final_ray_count),
                termination_reason=str(entry.termination_reason),
                incident_flux_w_m2=float(incident_flux),
                heating_result=heating,
            )
        )

    max_temps = np.array(
        [p.heating_result.max_temperature_k for p in per_list],
        dtype=float,
    )
    idx = int(np.argmax(max_temps))
    chosen = per_list[idx]
    return AngleDistanceHeatingScanResult(
        per_result=tuple(per_list),
        result_count=len(per_list),
        max_temperature_angle=float(chosen.angle_degrees),
        max_temperature_detector_z=float(chosen.detector_z),
        max_temperature_k=float(chosen.heating_result.max_temperature_k),
        max_temperature_rise_k=float(
            chosen.heating_result.max_temperature_rise_k
        ),
        max_incident_flux_w_m2=float(chosen.incident_flux_w_m2),
        nominal_incident_irradiance_w_m2=float(nominal),
        scalar_mode=str(scalar_mode),
    )


def run_lumped_heating_maps_over_angle_distance_scan(
    scan_result: AngleDistanceScanResult,
    *,
    nominal_incident_irradiance_w_m2: float,
    duration_s: float,
    dt_s: float,
    areal_heat_capacity_j_m2k: float,
    absorptivity: float = 1.0,
    h_conv_w_m2k: float = 0.0,
    emissivity: float = 0.0,
    ambient_temp_k: float = 293.15,
    initial_temp_k: float | None = None,
    threshold_temp_k: float | None = None,
    top_percent: float = 1.0,
    map_mode: str = "relative_irradiance_map",
) -> AngleDistanceHeatingMapScanResult:
    """Run per-pixel lumped target-heating once per angle-distance entry.

    Synthetic per-pixel thermal-map surrogate; **not an ignition
    validation model**. Each :class:`PerAngleDistanceResult` in
    ``scan_result`` must carry a
    :class:`DetectorIrradianceSurrogate` (i.e. produced by
    ``run_angle_distance_sweep(..., use_relative_irradiance=True)``).
    The per-entry chain is::

        s = entry.irradiance_surrogate
        incident_flux_map = (
            nominal_incident_irradiance_w_m2 * s.relative_irradiance_map
        )
        heating_map = simulate_lumped_target_heating_map(
            incident_flux_map, ...
        )

    Each pixel is treated as an independent 0-D lumped target. The
    optical input is a relative irradiance map — **not calibrated
    W/m^2** unless the caller provides a calibrated
    ``nominal_incident_irradiance_w_m2``. This is **not CFD**,
    **not pyrolysis**, **not spatial conduction**, and **not**
    ignition validation. Delta values across pattern variants are
    reported for comparison only; a negative delta is **not
    required**.

    Parameters
    ----------
    scan_result
        :class:`AngleDistanceScanResult` from the angle-distance
        sweep. Empty ``per_result`` returns an empty heating-map
        scan with all ``max_*`` fields ``None``.
    nominal_incident_irradiance_w_m2
        Scale multiplier applied to the relative irradiance map to
        produce the per-pixel ``incident_flux_map``. Must be finite
        and ``>= 0``. **Not calibrated W/m^2** unless the caller
        anchors it to a calibrated solar reference.
    duration_s, dt_s, areal_heat_capacity_j_m2k, absorptivity,
    h_conv_w_m2k, emissivity, ambient_temp_k, initial_temp_k,
    threshold_temp_k, top_percent
        Forwarded verbatim to
        :func:`simulate_lumped_target_heating_map`. All thermal-side
        validation is delegated there.
    map_mode
        Reduction from the surrogate to the per-entry flux map.
        ``"relative_irradiance_map"`` (the only supported mode in
        this task) takes
        ``s.relative_irradiance_map`` directly.

    Raises
    ------
    ThermalError
        On invalid ``scan_result`` type, missing
        ``irradiance_surrogate`` on a non-empty entry, invalid
        ``map_mode``, invalid
        ``nominal_incident_irradiance_w_m2``, or any downstream
        :class:`ThermalError` from
        :func:`simulate_lumped_target_heating_map`.
    """
    if not isinstance(scan_result, AngleDistanceScanResult):
        raise ThermalError(
            f"scan_result must be an AngleDistanceScanResult; "
            f"got {type(scan_result).__name__}"
        )
    nominal = _check_finite_nonneg(
        nominal_incident_irradiance_w_m2,
        name="nominal_incident_irradiance_w_m2",
    )
    if map_mode not in _VALID_MAP_MODES:
        raise ThermalError(
            f"map_mode must be one of {_VALID_MAP_MODES}; "
            f"got {map_mode!r}"
        )

    if not scan_result.per_result:
        return _empty_map_result(nominal, map_mode)

    for i, entry in enumerate(scan_result.per_result):
        if entry.irradiance_surrogate is None:
            raise ThermalError(
                f"scan_result.per_result[{i}].irradiance_surrogate "
                f"is None; this coupling layer requires the angle "
                f"scan to be run with use_relative_irradiance=True"
            )

    per_list: list[PerAngleDistanceHeatingMapResult] = []
    for entry in scan_result.per_result:
        s = entry.irradiance_surrogate
        rel_map = np.asarray(s.relative_irradiance_map, dtype=float)
        incident_flux_map = float(nominal) * rel_map
        heating_map = simulate_lumped_target_heating_map(
            incident_flux_map,
            duration_s=duration_s,
            dt_s=dt_s,
            areal_heat_capacity_j_m2k=areal_heat_capacity_j_m2k,
            absorptivity=absorptivity,
            h_conv_w_m2k=h_conv_w_m2k,
            emissivity=emissivity,
            ambient_temp_k=ambient_temp_k,
            initial_temp_k=initial_temp_k,
            threshold_temp_k=threshold_temp_k,
            top_percent=top_percent,
        )
        per_list.append(
            PerAngleDistanceHeatingMapResult(
                angle_degrees=float(entry.angle_degrees),
                detector_z=float(entry.detector_z),
                optical_c99=float(entry.metrics.c99),
                optical_cmax=float(entry.metrics.cmax),
                max_relative_irradiance=float(rel_map.max()),
                detector_hits=int(entry.detector_hits),
                final_ray_count=int(entry.final_ray_count),
                termination_reason=str(entry.termination_reason),
                incident_flux_map_max_w_m2=float(
                    incident_flux_map.max()
                ),
                heating_map_result=heating_map,
            )
        )

    max_temps = np.array(
        [p.heating_map_result.max_temperature_k for p in per_list],
        dtype=float,
    )
    idx = int(np.argmax(max_temps))
    chosen = per_list[idx]
    top_rises = np.array(
        [
            p.heating_map_result.top_percent_max_temperature_rise_k
            for p in per_list
        ],
        dtype=float,
    )
    return AngleDistanceHeatingMapScanResult(
        per_result=tuple(per_list),
        result_count=len(per_list),
        max_temperature_angle=float(chosen.angle_degrees),
        max_temperature_detector_z=float(chosen.detector_z),
        max_temperature_k=float(
            chosen.heating_map_result.max_temperature_k
        ),
        max_temperature_rise_k=float(
            chosen.heating_map_result.max_temperature_rise_k
        ),
        max_top_percent_temperature_rise_k=float(top_rises.max()),
        nominal_incident_irradiance_w_m2=float(nominal),
        map_mode=str(map_mode),
    )
