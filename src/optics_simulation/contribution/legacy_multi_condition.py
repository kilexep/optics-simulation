"""Multi-condition hotspot contribution aggregation for actual STL.

Actual STL multi-condition hotspot-backtracked risk map smoke
check; **not a physical PET-bottle validation**. Single-condition
contribution maps cover one ``(angle_degrees, detector_distance)``
combination at a time, but the legacy parity scan and the actual
STL thermal-risk scan both expose surrogate worst conditions at
many different ``(angle, distance)`` pairs. This module:

- Selects multiple worst conditions from a paired
  :class:`LegacyOpticalScanResult` /
  :class:`LegacyThermalRiskScanResult` using requested reason
  labels (e.g. ``max_temperature``, ``max_c99``,
  ``max_threshold_count``).
- For each selected condition, runs the same single-condition
  detailed trace + hotspot backtracking pipeline as
  :func:`build_legacy_hotspot_contribution_map`.
- Sums the per-condition contribution count and weight maps into
  an aggregate :class:`ContributionMap`, normalizes the aggregate
  by its own max, and post-processes it into an aggregate
  :class:`RiskMap` with the same threshold / epsilon policy.

Limitations
-----------
- The aggregate contribution map is a **diagnostic** map, not a
  measured physical risk field.
- Selected conditions are surrogate worst cases; they are not
  required to be globally optimal under every metric.
- Per-condition rays are kept independent — no silent
  deduplication across conditions. Per-condition
  ``selected_ray_indices`` are concatenated only for bookkeeping
  and are not globally meaningful.
- This is **not** fire-prevention validation, **not** PET-bottle
  safety, and **not** manufacturing-ready.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from optics_simulation.contribution.legacy_hotspot_backtracking import (
    LegacyHotspotContributionResult,
    build_legacy_hotspot_contribution_map,
)
from optics_simulation.contribution.map import (
    ContributionError,
    ContributionMap,
)
from optics_simulation.contribution.risk_map import (
    RiskMap,
    build_risk_map,
)
from optics_simulation.optics.legacy_detailed_trace import (
    run_legacy_pet_water_detailed_trace,
)
from optics_simulation.optics.legacy_pet_water import (
    LegacyPetWaterTraceSetup,
)
from optics_simulation.optics.legacy_scan import (
    LegacyOpticalScanResult,
)
from optics_simulation.optics.ray import OpticsError
from optics_simulation.thermal.legacy_scan_coupling import (
    LegacyThermalRiskScanResult,
)


_VALID_REASONS = (
    "max_temperature",
    "max_c99",
    "max_threshold_count",
)


@dataclass(frozen=True)
class LegacyConditionSelection:
    angle_degrees: float
    detector_distance: float
    reason: str
    score: float


@dataclass(frozen=True)
class LegacyMultiConditionContributionResult:
    selected_conditions: tuple[LegacyConditionSelection, ...]
    per_condition_results: tuple[LegacyHotspotContributionResult, ...]
    aggregate_contribution_map: ContributionMap
    aggregate_risk_map: RiskMap
    condition_count: int
    total_selected_rays: int
    aggregate_nonzero_bins: int
    contribution_resolution: tuple[int, int]
    result_type: str = "legacy_multi_condition_hotspot_contribution"


def _validate_reasons(reasons: tuple[str, ...]) -> tuple[str, ...]:
    rs = tuple(str(r) for r in reasons)
    if len(rs) == 0:
        raise OpticsError(
            "include_reasons must contain at least one entry"
        )
    for r in rs:
        if r not in _VALID_REASONS:
            raise OpticsError(
                f"include_reasons entry {r!r} is not one of "
                f"{_VALID_REASONS}"
            )
    return rs


def select_legacy_worst_conditions(
    optical_result: LegacyOpticalScanResult,
    thermal_result: LegacyThermalRiskScanResult,
    *,
    max_conditions: int = 3,
    include_reasons: tuple[str, ...] = (
        "max_temperature",
        "max_c99",
        "max_threshold_count",
    ),
) -> tuple[LegacyConditionSelection, ...]:
    """Select unique ``(angle, distance)`` pairs as surrogate worst conditions.

    Actual STL multi-condition hotspot-backtracked risk map smoke
    check; **not a physical PET-bottle validation**. Walks
    ``include_reasons`` in order, reads the corresponding aggregate
    field from ``optical_result`` or ``thermal_result``, and emits
    a :class:`LegacyConditionSelection` for it. Pairs are
    deduplicated by ``(angle_degrees, detector_distance)`` while
    preserving the ``include_reasons`` order. Output is truncated
    to at most ``max_conditions`` entries.

    Empty ``optical_result.entries`` and empty
    ``thermal_result.entries`` produce an empty tuple. Reasons
    whose underlying aggregate fields are ``None`` are silently
    skipped.

    Selected conditions are surrogate worst cases under the
    requested reasons; they are **not** required to be globally
    optimal under every metric.

    Raises
    ------
    OpticsError
        On invalid result types, non-positive ``max_conditions``,
        or unknown reason labels.
    """
    if not isinstance(optical_result, LegacyOpticalScanResult):
        raise OpticsError(
            "optical_result must be a LegacyOpticalScanResult; "
            f"got {type(optical_result).__name__}"
        )
    if not isinstance(thermal_result, LegacyThermalRiskScanResult):
        raise OpticsError(
            "thermal_result must be a LegacyThermalRiskScanResult; "
            f"got {type(thermal_result).__name__}"
        )

    if isinstance(max_conditions, bool) or not isinstance(
        max_conditions, int
    ):
        raise OpticsError(
            f"max_conditions must be a positive int; "
            f"got {max_conditions!r}"
        )
    if int(max_conditions) <= 0:
        raise OpticsError(
            f"max_conditions must be a positive int; "
            f"got {max_conditions}"
        )

    reasons = _validate_reasons(tuple(include_reasons))

    if (
        len(optical_result.entries) == 0
        or len(thermal_result.entries) == 0
    ):
        return ()

    selections: list[LegacyConditionSelection] = []
    seen: set[tuple[float, float]] = set()

    for reason in reasons:
        if reason == "max_temperature":
            angle = thermal_result.max_temperature_angle
            distance = thermal_result.max_temperature_distance
            score_value = thermal_result.max_temperature_k
        elif reason == "max_c99":
            angle = optical_result.max_c99_angle
            distance = optical_result.max_c99_detector_distance
            score_value = optical_result.max_c99
        elif reason == "max_threshold_count":
            angle = thermal_result.max_threshold_exceeded_count_angle
            distance = (
                thermal_result.max_threshold_exceeded_count_distance
            )
            tcount = thermal_result.max_threshold_exceeded_count
            score_value = (
                None if tcount is None else float(int(tcount))
            )
        else:  # pragma: no cover - guarded by _validate_reasons
            continue

        if angle is None or distance is None or score_value is None:
            continue

        key = (float(angle), float(distance))
        if key in seen:
            continue
        seen.add(key)

        selections.append(
            LegacyConditionSelection(
                angle_degrees=float(angle),
                detector_distance=float(distance),
                reason=reason,
                score=float(score_value),
            )
        )

        if len(selections) >= int(max_conditions):
            break

    return tuple(selections)


def _empty_contribution_map(
    nv: int, nu: int,
) -> ContributionMap:
    return ContributionMap(
        count_map=np.zeros((int(nv), int(nu)), dtype=np.int64),
        weight_map=np.zeros((int(nv), int(nu)), dtype=float),
        normalized_map=np.zeros((int(nv), int(nu)), dtype=float),
        resolution=(int(nv), int(nu)),
        total_selected=0,
        total_weight=0.0,
        selected_ray_indices=np.empty(0, dtype=np.int64),
        u_bin_indices=np.empty(0, dtype=np.int64),
        v_bin_indices=np.empty(0, dtype=np.int64),
    )


def build_multi_condition_legacy_hotspot_contribution_map(
    *,
    setup: LegacyPetWaterTraceSetup,
    conditions: Sequence[LegacyConditionSelection],
    source_width: float,
    source_height: float,
    source_radius: float,
    sample_count_y: int,
    sample_count_z: int,
    detector_size: float,
    detector_resolution: tuple[int, int],
    epsilon: float = 0.1,
    hotspot_top_percent: float = 10.0,
    contribution_resolution: tuple[int, int] = (32, 64),
    risk_threshold: float | None = None,
    risk_epsilon: float = 0.01,
) -> LegacyMultiConditionContributionResult:
    """Aggregate hotspot contribution maps across multiple worst conditions.

    Actual STL multi-condition hotspot-backtracked risk map smoke
    check; **not a physical PET-bottle validation**. For each
    :class:`LegacyConditionSelection` in ``conditions``:

    1. Run :func:`run_legacy_pet_water_detailed_trace` at that
       ``(angle_degrees, detector_distance)``.
    2. Run :func:`build_legacy_hotspot_contribution_map` to map
       the detector hotspot rays back to first-shell-hit
       ``(u, v)`` and produce a per-condition contribution map at
       ``contribution_resolution``.

    The per-condition count and weight maps are then summed
    element-wise. The aggregate ``normalized_map`` is the
    aggregate weight map divided by its own positive max
    (zero map otherwise). The aggregate map is post-processed
    into a :class:`RiskMap` via :func:`build_risk_map` with
    ``threshold=risk_threshold`` and ``epsilon=risk_epsilon``.

    Per-condition rays are kept independent — no silent
    deduplication across conditions. The aggregate
    ``selected_ray_indices`` / ``u_bin_indices`` / ``v_bin_indices``
    are concatenations of the per-condition arrays and are not
    globally meaningful. Empty ``conditions`` produces a result
    with empty per-condition tuples and zero-valued aggregate
    maps at ``contribution_resolution``.

    Raises
    ------
    OpticsError
        On invalid setup type, invalid ``conditions`` entries,
        invalid source / detector dimensions, non-positive
        ``hotspot_top_percent``, or downstream optics validation
        failures.
    ContributionError
        On invalid contribution resolution or downstream
        contribution validation failures.
    """
    if not isinstance(setup, LegacyPetWaterTraceSetup):
        raise OpticsError(
            "setup must be a LegacyPetWaterTraceSetup; got "
            f"{type(setup).__name__}"
        )

    try:
        cond_list = list(conditions)
    except TypeError as exc:
        raise OpticsError(
            "conditions must be a Sequence of "
            "LegacyConditionSelection"
        ) from exc

    for i, c in enumerate(cond_list):
        if not isinstance(c, LegacyConditionSelection):
            raise OpticsError(
                f"conditions[{i}] must be a "
                f"LegacyConditionSelection; got {type(c).__name__}"
            )

    res = tuple(contribution_resolution)
    if len(res) != 2:
        raise ContributionError(
            f"contribution_resolution must be (nv, nu); got {res}"
        )
    nv = int(res[0])
    nu = int(res[1])
    if nv <= 0 or nu <= 0:
        raise ContributionError(
            f"contribution_resolution entries must be positive "
            f"integers; got (nv, nu)=({nv}, {nu})"
        )

    tp = float(hotspot_top_percent)
    if not (math.isfinite(tp) and 0.0 < tp <= 100.0):
        raise OpticsError(
            f"hotspot_top_percent must be a finite float in "
            f"(0, 100]; got {hotspot_top_percent}"
        )

    if len(cond_list) == 0:
        empty_contribution = _empty_contribution_map(nv, nu)
        empty_risk = build_risk_map(
            empty_contribution,
            threshold=risk_threshold,
            epsilon=float(risk_epsilon),
        )
        return LegacyMultiConditionContributionResult(
            selected_conditions=(),
            per_condition_results=(),
            aggregate_contribution_map=empty_contribution,
            aggregate_risk_map=empty_risk,
            condition_count=0,
            total_selected_rays=0,
            aggregate_nonzero_bins=0,
            contribution_resolution=(nv, nu),
        )

    per_condition: list[LegacyHotspotContributionResult] = []
    agg_count = np.zeros((nv, nu), dtype=np.int64)
    agg_weight = np.zeros((nv, nu), dtype=float)
    sel_idx_chunks: list[np.ndarray] = []
    u_bin_chunks: list[np.ndarray] = []
    v_bin_chunks: list[np.ndarray] = []
    total_selected = 0

    for c in cond_list:
        detailed = run_legacy_pet_water_detailed_trace(
            setup=setup,
            angle_degrees=float(c.angle_degrees),
            detector_distance=float(c.detector_distance),
            source_width=float(source_width),
            source_height=float(source_height),
            source_radius=float(source_radius),
            sample_count_y=int(sample_count_y),
            sample_count_z=int(sample_count_z),
            detector_size=float(detector_size),
            detector_resolution=(
                int(detector_resolution[0]),
                int(detector_resolution[1]),
            ),
            epsilon=float(epsilon),
        )
        hotspot = build_legacy_hotspot_contribution_map(
            detailed,
            top_percent=tp,
            contribution_resolution=(nv, nu),
            risk_threshold=risk_threshold,
            risk_epsilon=float(risk_epsilon),
        )
        per_condition.append(hotspot)
        agg_count = agg_count + np.asarray(
            hotspot.contribution_map.count_map, dtype=np.int64,
        )
        agg_weight = agg_weight + np.asarray(
            hotspot.contribution_map.weight_map, dtype=float,
        )
        sel_idx_chunks.append(
            np.asarray(
                hotspot.contribution_map.selected_ray_indices,
                dtype=np.int64,
            ).copy()
        )
        u_bin_chunks.append(
            np.asarray(
                hotspot.contribution_map.u_bin_indices,
                dtype=np.int64,
            ).copy()
        )
        v_bin_chunks.append(
            np.asarray(
                hotspot.contribution_map.v_bin_indices,
                dtype=np.int64,
            ).copy()
        )
        total_selected += int(
            hotspot.contribution_map.total_selected
        )

    agg_max = float(agg_weight.max()) if agg_weight.size > 0 else 0.0
    if agg_max > 0.0:
        normalized_map = agg_weight / agg_max
    else:
        normalized_map = np.zeros_like(agg_weight, dtype=float)

    if sel_idx_chunks:
        agg_sel_indices = np.concatenate(sel_idx_chunks).astype(
            np.int64, copy=True,
        )
        agg_u_bins = np.concatenate(u_bin_chunks).astype(
            np.int64, copy=True,
        )
        agg_v_bins = np.concatenate(v_bin_chunks).astype(
            np.int64, copy=True,
        )
    else:
        agg_sel_indices = np.empty(0, dtype=np.int64)
        agg_u_bins = np.empty(0, dtype=np.int64)
        agg_v_bins = np.empty(0, dtype=np.int64)

    aggregate_contribution = ContributionMap(
        count_map=agg_count,
        weight_map=agg_weight,
        normalized_map=normalized_map,
        resolution=(nv, nu),
        total_selected=int(total_selected),
        total_weight=float(agg_weight.sum()),
        selected_ray_indices=agg_sel_indices,
        u_bin_indices=agg_u_bins,
        v_bin_indices=agg_v_bins,
    )
    aggregate_risk = build_risk_map(
        aggregate_contribution,
        threshold=risk_threshold,
        epsilon=float(risk_epsilon),
    )

    nonzero_bins = int((agg_weight > 0.0).sum())

    return LegacyMultiConditionContributionResult(
        selected_conditions=tuple(cond_list),
        per_condition_results=tuple(per_condition),
        aggregate_contribution_map=aggregate_contribution,
        aggregate_risk_map=aggregate_risk,
        condition_count=int(len(cond_list)),
        total_selected_rays=int(total_selected),
        aggregate_nonzero_bins=nonzero_bins,
        contribution_resolution=(nv, nu),
    )
