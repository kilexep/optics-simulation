"""Synthetic thermal-risk surrogate metrics.

Synthetic thermal-risk surrogate metrics; **not an ignition
validation model**. Reduces a per-pixel
:class:`LumpedTargetHeatingMapResult` (the output of
:func:`simulate_lumped_target_heating_map`) to a single reusable
:class:`ThermalRiskMetrics` value object that summarizes the
distribution-sensitive thermal response of the synthetic optical
caustic.

Why this exists
---------------
The scalar lumped-target coupling collapses each (angle,
detector_z) entry to a single ``max_relative_irradiance`` and
loses any distribution-sensitive information when the peak pixel
does not move. The per-pixel thermal-map foundation keeps the
2D map but exposes many fields. This module provides one
canonical reduction so that downstream comparisons (original vs
displaced, ablation studies, optimization scoring) can be driven
by a stable, named set of distribution-sensitive thermal-risk
metrics rather than ad-hoc per-call reductions.

Research framing (also enforced in the demo)
--------------------------------------------
- Thermal risk metrics are computed from a per-pixel lumped
  thermal-map surrogate.
- Threshold values are illustrative unless calibrated by
  experiment.
- This is **not** pyrolysis.
- This is **not** CFD.
- This is **not** spatial conduction.
- This does **not** prove fire prevention or PET-bottle safety.
- Delta values across pattern variants are comparison
  diagnostics only; a negative delta is **not required**.

Definitions
-----------
For a :class:`LumpedTargetHeatingMapResult` ``hm`` with pixel
count ``N``:

- ``max_temperature_k = hm.max_temperature_map_k.max()``
- ``max_temperature_rise_k = hm.max_temperature_rise_map_k.max()``
- ``final_temperature_k = hm.final_temperature_map_k.max()``
  (hottest final pixel)
- ``final_temperature_rise_k = hm.final_temperature_rise_map_k.max()``
- ``mean_max_temperature_k = hm.max_temperature_map_k.mean()``
- ``mean_max_temperature_rise_k = hm.max_temperature_rise_map_k.mean()``
- ``mean_final_temperature_k = hm.final_temperature_map_k.mean()``
- ``top_percent_max_temperature_rise_k =
   mean(top k pixels of hm.max_temperature_rise_map_k)``
  with ``k = max(1, ceil(N * top_percent / 100))``.
- ``threshold_exceeded_count = hm.threshold_exceeded_count``
- ``threshold_exceeded_fraction = threshold_exceeded_count / N``
- ``threshold_exceeded_area = threshold_exceeded_count * pixel_area``
  (or ``None`` when ``pixel_area is None``).
- The threshold time stats (``min``, ``mean``, ``max``) are
  computed over finite entries of
  ``hm.time_to_threshold_map_s``; all three are ``None`` when no
  pixel ever reached the threshold (or no threshold was set).

Limitations
-----------
``exceedance_duration_sum_s`` and
``exceedance_degree_seconds_sum_ks`` would require the full
per-pixel temperature trajectory, which the current
:func:`simulate_lumped_target_heating_map` deliberately does not
materialize (memory bound). They are exposed as ``None`` here
and documented as such; a future task may add an opt-in trajectory
storage path. This module **does not** modify
:func:`simulate_lumped_target_heating_map` or
:class:`LumpedTargetHeatingMapResult` to add that path.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from optics_simulation.thermal.lumped_target import (
    LumpedTargetHeatingMapResult,
    ThermalError,
)


@dataclass(frozen=True)
class ThermalRiskMetrics:
    max_temperature_k: float
    max_temperature_rise_k: float
    final_temperature_k: float
    final_temperature_rise_k: float
    mean_max_temperature_k: float
    mean_max_temperature_rise_k: float
    mean_final_temperature_k: float
    top_percent_max_temperature_rise_k: float
    threshold_temp_k: float | None
    threshold_exceeded_count: int
    threshold_exceeded_fraction: float
    threshold_exceeded_area: float | None
    mean_time_to_threshold_s: float | None
    min_time_to_threshold_s: float | None
    max_time_to_threshold_s: float | None
    exceedance_duration_sum_s: float | None
    exceedance_degree_seconds_sum_ks: float | None
    pixel_count: int
    pixel_area: float | None
    top_percent: float
    metric_type: str = "thermal_risk_surrogate_metrics"


def _check_top_percent(value: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ThermalError(
            f"top_percent must be a finite float in (0, 100]; "
            f"got {value!r}"
        ) from exc
    if not np.isfinite(v):
        raise ThermalError(f"top_percent must be finite; got {v}")
    if v <= 0.0 or v > 100.0:
        raise ThermalError(f"top_percent must be in (0, 100]; got {v}")
    return v


def _check_pixel_area(value: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ThermalError(
            f"pixel_area must be a finite float > 0; got {value!r}"
        ) from exc
    if not np.isfinite(v):
        raise ThermalError(f"pixel_area must be finite; got {v}")
    if v <= 0.0:
        raise ThermalError(f"pixel_area must be > 0; got {v}")
    return v


def compute_thermal_risk_metrics(
    heating_map: LumpedTargetHeatingMapResult,
    *,
    pixel_area: float | None = None,
    top_percent: float = 1.0,
) -> ThermalRiskMetrics:
    """Reduce a per-pixel heating map to thermal-risk surrogate metrics.

    Synthetic thermal-risk surrogate metrics; **not an ignition
    validation model**. Threshold values are illustrative unless
    calibrated by experiment. This is **not** pyrolysis, **not**
    CFD, **not** spatial conduction, and does **not** prove fire
    prevention or PET-bottle safety. Delta values across pattern
    variants are comparison diagnostics only; a negative delta is
    **not required**.

    Parameters
    ----------
    heating_map
        :class:`LumpedTargetHeatingMapResult` produced by
        :func:`simulate_lumped_target_heating_map`. The
        per-pixel maps are read directly; this function does not
        re-run any thermal integration.
    pixel_area
        Optional finite positive ``float`` (in the same area unit
        as the detector grid pixel area). When provided,
        ``threshold_exceeded_area = threshold_exceeded_count *
        pixel_area``; when ``None`` the area is reported as
        ``None``.
    top_percent
        Top-percent thermal rise: the mean of the top ``k`` pixels
        of ``max_temperature_rise_map_k``, with
        ``k = max(1, ceil(pixel_count * top_percent / 100))``.
        Must satisfy ``0 < top_percent <= 100``.

    Raises
    ------
    ThermalError
        On invalid ``heating_map`` type, invalid ``top_percent``,
        invalid ``pixel_area``, NaN / inf in any temperature map,
        or inconsistent map shapes inside ``heating_map``.
    """
    if not isinstance(heating_map, LumpedTargetHeatingMapResult):
        raise ThermalError(
            f"heating_map must be a LumpedTargetHeatingMapResult; "
            f"got {type(heating_map).__name__}"
        )

    top_pct = _check_top_percent(top_percent)

    pixel_area_val: float | None
    if pixel_area is None:
        pixel_area_val = None
    else:
        pixel_area_val = _check_pixel_area(pixel_area)

    final_map = np.asarray(
        heating_map.final_temperature_map_k, dtype=float
    )
    max_map = np.asarray(
        heating_map.max_temperature_map_k, dtype=float
    )
    final_rise = np.asarray(
        heating_map.final_temperature_rise_map_k, dtype=float
    )
    max_rise = np.asarray(
        heating_map.max_temperature_rise_map_k, dtype=float
    )
    ttt = np.asarray(
        heating_map.time_to_threshold_map_s, dtype=float
    )
    threshold_mask = np.asarray(
        heating_map.threshold_exceeded_mask, dtype=bool
    )

    reference_shape = final_map.shape
    if final_map.size == 0:
        raise ThermalError(
            "heating_map maps must have a non-empty shape; "
            f"got {reference_shape}"
        )
    for arr_name, arr in (
        ("max_temperature_map_k", max_map),
        ("final_temperature_rise_map_k", final_rise),
        ("max_temperature_rise_map_k", max_rise),
        ("time_to_threshold_map_s", ttt),
        ("threshold_exceeded_mask", threshold_mask),
    ):
        if arr.shape != reference_shape:
            raise ThermalError(
                f"heating_map.{arr_name} shape {arr.shape} "
                f"inconsistent with heating_map."
                f"final_temperature_map_k shape {reference_shape}"
            )

    for arr_name, arr in (
        ("final_temperature_map_k", final_map),
        ("max_temperature_map_k", max_map),
        ("final_temperature_rise_map_k", final_rise),
        ("max_temperature_rise_map_k", max_rise),
    ):
        if not np.isfinite(arr).all():
            raise ThermalError(
                f"heating_map.{arr_name} contains NaN or inf values"
            )
    if np.isinf(ttt).any():
        raise ThermalError(
            "heating_map.time_to_threshold_map_s contains inf values"
        )

    pixel_count = int(max_map.size)

    max_t = float(max_map.max())
    max_t_rise = float(max_rise.max())
    final_t = float(final_map.max())
    final_t_rise = float(final_rise.max())
    mean_max_t = float(max_map.mean())
    mean_max_t_rise = float(max_rise.mean())
    mean_final_t = float(final_map.mean())

    k = max(1, int(np.ceil(pixel_count * top_pct / 100.0)))
    sorted_rise = np.sort(max_rise.ravel())
    top_k_values = sorted_rise[-k:]
    top_pct_rise = float(top_k_values.mean())

    threshold_count = int(heating_map.threshold_exceeded_count)
    threshold_fraction = float(threshold_count) / float(pixel_count)

    threshold_area: float | None
    if pixel_area_val is None:
        threshold_area = None
    else:
        threshold_area = float(threshold_count) * float(pixel_area_val)

    finite_ttt = ttt[np.isfinite(ttt)]
    if finite_ttt.size > 0:
        mean_ttt: float | None = float(finite_ttt.mean())
        min_ttt: float | None = float(finite_ttt.min())
        max_ttt: float | None = float(finite_ttt.max())
    else:
        mean_ttt = None
        min_ttt = None
        max_ttt = None

    threshold_value: float | None
    if heating_map.threshold_temp_k is None:
        threshold_value = None
    else:
        threshold_value = float(heating_map.threshold_temp_k)

    return ThermalRiskMetrics(
        max_temperature_k=max_t,
        max_temperature_rise_k=max_t_rise,
        final_temperature_k=final_t,
        final_temperature_rise_k=final_t_rise,
        mean_max_temperature_k=mean_max_t,
        mean_max_temperature_rise_k=mean_max_t_rise,
        mean_final_temperature_k=mean_final_t,
        top_percent_max_temperature_rise_k=top_pct_rise,
        threshold_temp_k=threshold_value,
        threshold_exceeded_count=threshold_count,
        threshold_exceeded_fraction=threshold_fraction,
        threshold_exceeded_area=threshold_area,
        mean_time_to_threshold_s=mean_ttt,
        min_time_to_threshold_s=min_ttt,
        max_time_to_threshold_s=max_ttt,
        exceedance_duration_sum_s=None,
        exceedance_degree_seconds_sum_ks=None,
        pixel_count=pixel_count,
        pixel_area=pixel_area_val,
        top_percent=float(top_pct),
    )
