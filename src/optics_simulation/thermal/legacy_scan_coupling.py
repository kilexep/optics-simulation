"""Optical-to-thermal-risk coupling over a legacy STL optical scan.

Actual STL optical-to-thermal risk surrogate scan; **not a
physical PET-bottle validation**. Consumes a
:class:`LegacyOpticalScanResult` whose entries carry the per-entry
:class:`DetectorIrradianceSurrogate` (run the optical scan with
``store_irradiance_surrogate=True``), converts each entry's
relative irradiance map to a per-pixel incident-flux map using a
caller-supplied ``nominal_incident_irradiance_w_m2``, runs the
per-pixel lumped-target heating surrogate, and reduces the
resulting heating map to :class:`ThermalRiskMetrics`. All
``angle_degrees`` / ``detector_distance`` ordering from the
optical scan is preserved.

Scope (also enforced at runtime)
--------------------------------
- This uses the actual loaded STL as geometry input.
- The STL is target-dimension normalized and uses a generated
  inner water mesh.
- The optical map is a relative irradiance surrogate.
- The thermal map is a per-pixel lumped target-heating surrogate.
- Threshold values are illustrative unless calibrated by
  experiment.
- This is not pyrolysis.
- This is not CFD.
- This is not fire-prevention validation.
- Do not use old H.max as a primary research metric.
- Do not claim PET-bottle safety.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optics_simulation.optics.legacy_scan import (
    LegacyOpticalScanResult,
)
from optics_simulation.thermal.lumped_target import (
    ThermalError,
    simulate_lumped_target_heating_map,
)
from optics_simulation.thermal.risk_metrics import (
    ThermalRiskMetrics,
    compute_thermal_risk_metrics,
)


@dataclass(frozen=True)
class LegacyThermalRiskEntry:
    angle_degrees: float
    detector_distance: float
    optical_c99: float
    optical_cmax: float
    max_relative_irradiance: float
    detector_hits: int
    final_ray_count: int
    incident_flux_map_max_w_m2: float
    thermal_metrics: ThermalRiskMetrics


@dataclass(frozen=True)
class LegacyThermalRiskScanResult:
    entries: tuple[LegacyThermalRiskEntry, ...]
    entry_count: int
    max_temperature_k: float | None
    max_temperature_angle: float | None
    max_temperature_distance: float | None
    max_top_percent_temperature_rise_k: float | None
    max_top_percent_temperature_rise_angle: float | None
    max_top_percent_temperature_rise_distance: float | None
    max_threshold_exceeded_count: int | None
    max_threshold_exceeded_count_angle: float | None
    max_threshold_exceeded_count_distance: float | None
    nominal_incident_irradiance_w_m2: float
    result_type: str = "legacy_stl_thermal_risk_scan"


def _check_nominal_irradiance(value: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ThermalError(
            f"nominal_incident_irradiance_w_m2 must be a finite "
            f"float >= 0; got {value!r}"
        ) from exc
    if not np.isfinite(v):
        raise ThermalError(
            f"nominal_incident_irradiance_w_m2 must be finite; "
            f"got {v}"
        )
    if v < 0.0:
        raise ThermalError(
            f"nominal_incident_irradiance_w_m2 must be >= 0; "
            f"got {v}"
        )
    return v


def compute_thermal_risk_over_legacy_optical_scan(
    optical_result: LegacyOpticalScanResult,
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
) -> LegacyThermalRiskScanResult:
    """Compute thermal-risk metrics over a legacy optical scan result.

    Actual STL optical-to-thermal risk surrogate scan; **not a
    physical PET-bottle validation**. For every entry of
    ``optical_result`` (which must have been produced by
    :func:`run_legacy_pet_water_angle_distance_scan` with
    ``store_irradiance_surrogate=True``), convert the entry's
    relative irradiance map into a per-pixel incident-flux map by
    multiplying by ``nominal_incident_irradiance_w_m2``, run the
    per-pixel lumped-target heating surrogate, and reduce to
    :class:`ThermalRiskMetrics`. Threshold values are illustrative
    unless calibrated by experiment. This is **not** pyrolysis,
    **not** CFD, and does **not** prove fire prevention or
    PET-bottle safety. Old H.max is **not** used as a primary
    metric.

    Parameters
    ----------
    optical_result
        :class:`LegacyOpticalScanResult` with
        ``irradiance_surrogate is not None`` on every entry.
    nominal_incident_irradiance_w_m2
        Finite scalar ``>= 0`` used as the per-source-area
        absolute scale; multiplied through the relative irradiance
        map to produce the per-pixel incident-flux surrogate.
    duration_s, dt_s, areal_heat_capacity_j_m2k, absorptivity,
    h_conv_w_m2k, emissivity, ambient_temp_k, initial_temp_k,
    threshold_temp_k, top_percent
        Forwarded verbatim to
        :func:`simulate_lumped_target_heating_map`. ``top_percent``
        is also forwarded to
        :func:`compute_thermal_risk_metrics`.

    Raises
    ------
    ThermalError
        On invalid ``optical_result`` type, non-finite or negative
        ``nominal_incident_irradiance_w_m2``, missing
        ``irradiance_surrogate`` on any entry, or any downstream
        thermal-validation failure.
    """
    if not isinstance(optical_result, LegacyOpticalScanResult):
        raise ThermalError(
            "optical_result must be a LegacyOpticalScanResult; "
            f"got {type(optical_result).__name__}"
        )

    nominal = _check_nominal_irradiance(
        nominal_incident_irradiance_w_m2,
    )

    if len(optical_result.entries) == 0:
        return LegacyThermalRiskScanResult(
            entries=(),
            entry_count=0,
            max_temperature_k=None,
            max_temperature_angle=None,
            max_temperature_distance=None,
            max_top_percent_temperature_rise_k=None,
            max_top_percent_temperature_rise_angle=None,
            max_top_percent_temperature_rise_distance=None,
            max_threshold_exceeded_count=None,
            max_threshold_exceeded_count_angle=None,
            max_threshold_exceeded_count_distance=None,
            nominal_incident_irradiance_w_m2=nominal,
        )

    thermal_entries: list[LegacyThermalRiskEntry] = []
    for index, optical_entry in enumerate(optical_result.entries):
        surrogate = optical_entry.irradiance_surrogate
        if surrogate is None:
            raise ThermalError(
                f"optical_result.entries[{index}].irradiance_surrogate "
                "is None; rerun the optical scan with "
                "store_irradiance_surrogate=True"
            )

        incident_flux_map = (
            nominal
            * np.asarray(
                surrogate.relative_irradiance_map, dtype=float,
            )
        )
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
        thermal_metrics = compute_thermal_risk_metrics(
            heating_map,
            pixel_area=float(surrogate.pixel_area),
            top_percent=top_percent,
        )

        thermal_entries.append(
            LegacyThermalRiskEntry(
                angle_degrees=float(optical_entry.angle_degrees),
                detector_distance=float(
                    optical_entry.detector_distance,
                ),
                optical_c99=float(optical_entry.c99),
                optical_cmax=float(optical_entry.cmax),
                max_relative_irradiance=float(
                    optical_entry.max_relative_irradiance,
                ),
                detector_hits=int(optical_entry.detector_hits),
                final_ray_count=int(optical_entry.final_ray_count),
                incident_flux_map_max_w_m2=float(
                    nominal
                    * float(optical_entry.max_relative_irradiance)
                ),
                thermal_metrics=thermal_metrics,
            )
        )

    max_temp_idx = max(
        range(len(thermal_entries)),
        key=lambda i: float(
            thermal_entries[i].thermal_metrics.max_temperature_k,
        ),
    )
    max_top_idx = max(
        range(len(thermal_entries)),
        key=lambda i: float(
            thermal_entries[i].thermal_metrics
            .top_percent_max_temperature_rise_k,
        ),
    )
    max_thresh_idx = max(
        range(len(thermal_entries)),
        key=lambda i: int(
            thermal_entries[i].thermal_metrics.threshold_exceeded_count,
        ),
    )
    e_temp = thermal_entries[max_temp_idx]
    e_top = thermal_entries[max_top_idx]
    e_thresh = thermal_entries[max_thresh_idx]

    return LegacyThermalRiskScanResult(
        entries=tuple(thermal_entries),
        entry_count=len(thermal_entries),
        max_temperature_k=float(
            e_temp.thermal_metrics.max_temperature_k,
        ),
        max_temperature_angle=float(e_temp.angle_degrees),
        max_temperature_distance=float(e_temp.detector_distance),
        max_top_percent_temperature_rise_k=float(
            e_top.thermal_metrics
            .top_percent_max_temperature_rise_k,
        ),
        max_top_percent_temperature_rise_angle=float(
            e_top.angle_degrees,
        ),
        max_top_percent_temperature_rise_distance=float(
            e_top.detector_distance,
        ),
        max_threshold_exceeded_count=int(
            e_thresh.thermal_metrics.threshold_exceeded_count,
        ),
        max_threshold_exceeded_count_angle=float(
            e_thresh.angle_degrees,
        ),
        max_threshold_exceeded_count_distance=float(
            e_thresh.detector_distance,
        ),
        nominal_incident_irradiance_w_m2=nominal,
    )
