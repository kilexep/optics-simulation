"""Ring-offset risk-guided pattern parameter sweep diagnostic.

Actual STL ring-offset risk-guided pattern parameter sweep
diagnostic; **not a physical PET-bottle validation**. The pipeline
is structured into five conceptually distinct stages so research
logic and implementation detail do not bleed into each other:

1. **Selection** — pick a small number of "worst" baseline
   conditions via :func:`select_legacy_worst_conditions` and
   build a multi-condition aggregate hotspot risk map via
   :func:`build_multi_condition_legacy_hotspot_contribution_map`.
   This stage is performed by the **caller** before invoking the
   sweep and is summarized in
   :class:`RingOffsetSweepResult.selected_condition_count`.
2. **Ring transform** — for each
   :class:`RingOffsetSweepCandidateSpec`, transform the
   selection-stage risk map via
   :func:`create_ring_offset_risk_map`. ``inner_radius_px`` and
   ``outer_radius_px`` are in **risk-map bin units** (see
   :data:`RING_RADIUS_UNITS_DESCRIPTION`).
3. **Candidate pattern generation** — drive
   :func:`create_risk_guided_legacy_patterned_pet_water_setup`
   with the transformed ring risk map and a fixed pattern shape
   (``pattern_count`` / sigmas / ``max_depth`` / ``seed`` are
   shared across candidates so the sweep isolates the ring
   transform's effect).
4. **Evaluation** — run the legacy four-step PET-water
   angle / detector-distance scan, derive per-pixel thermal-risk
   metrics, and compare against the caller's baseline thermal
   scan. The evaluation grid contains every
   ``(angle_degrees, detector_distance)`` produced by the
   schedule, **including** the angles/distances that drove the
   selection stage.
5. **Reporting / ranking** — split each candidate's evaluation
   results into a **selection-condition (in-sample)** subset and a
   **holdout** subset (see
   :func:`split_selected_vs_holdout_conditions`), summarize each
   subset, run a Pareto non-dominance filter on
   ``(worst_delta_max_temperature_k, best_delta_threshold_count,
   -pass_ratio)`` (defaulting to the holdout subset, falling back
   to the full evaluation grid when holdout is empty), drop
   non-discriminative axes (range zero across candidates), and
   compute a min-max-normalized composite score on the surviving
   axes as a tie-break.

Why selection vs holdout matter
-------------------------------
The selection stage chose a small set of conditions specifically
because they were worst-case in the baseline. Re-evaluating the
patterned candidates only on those same conditions gives an
in-sample diagnostic that can over-credit a candidate that simply
moved one specific caustic without changing global behavior. The
holdout subset is the rest of the evaluation grid and is the
preferred ranking basis for the same reason a held-out test set
is preferred in supervised learning. This module reports both.

Limitations
-----------
- Ranking is a **diagnostic** ordering, **not** safety
  certification. The phrase "best by ..." here means "the
  candidate that minimizes the corresponding diagnostic axis on
  the chosen basis"; if every candidate worsens the worst-case
  metric the result is reported as
  ``no_improving_candidate_under_current_sweep == True`` and the
  same candidate is **least-bad**, not "best".
- ``new_bin_fraction`` is reported per entry but **excluded from
  the Pareto and composite ranking axes**, because the same
  numerical value can mean either "the pattern correctly stayed
  in the existing hotspot region" (good) or "the pattern failed
  to move the hotspot at all" (bad). Treat it as a *qualitative*
  signal, paired with ``source_overlap_fraction`` and
  ``retained_source_bin_fraction``.
- A higher guardrail pass count does **not** prove safety.
- A ring-offset pattern can still be mixed.
- Negative results from this sweep are limited to the tested ring
  family, the fixed pattern-shape settings, and the selected
  conditions used to drive the risk map. They are **not** a
  blanket negative judgment on the ring-offset family in general.
- This is **not** fire-prevention validation, **not** PET-bottle
  safety, and **not** manufacturing-ready.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from optics_simulation.contribution.legacy_contribution_diagnostic import (
    build_pattern_induced_hotspot_diagnostic,
)
from optics_simulation.contribution.legacy_multi_condition import (
    LegacyConditionSelection,
)
from optics_simulation.contribution.risk_map import RiskMap
from optics_simulation.contribution.risk_transform import (
    create_ring_offset_risk_map,
)
from optics_simulation.optics.legacy_patterned_pet_water import (
    LegacyPatternedPetWaterSetup,
    create_risk_guided_legacy_patterned_pet_water_setup,
)
from optics_simulation.optics.legacy_pet_water import (
    LegacyPetWaterTraceSetup,
)
from optics_simulation.optics.legacy_scan import (
    run_legacy_pet_water_angle_distance_scan,
)
from optics_simulation.optics.ray import OpticsError
from optics_simulation.thermal.legacy_risk_comparison import (
    LegacyThermalRiskComparisonEntry,
    LegacyThermalRiskComparisonResult,
    compare_legacy_thermal_risk_scans,
)
from optics_simulation.thermal.legacy_scan_coupling import (
    LegacyThermalRiskScanResult,
    compute_thermal_risk_over_legacy_optical_scan,
)
from optics_simulation.thermal.risk_comparison import (
    ThermalRiskGuardrailConfig,
)


RING_RADIUS_UNITS_DESCRIPTION = (
    "inner_radius_px and outer_radius_px are in risk-map bin "
    "units. The aggregate risk map has shape "
    "(nv, nu) = contribution_resolution, where nv counts height "
    "bins (axis 0) and nu counts circumferential bins (axis 1, "
    "wrap-around). They are NOT mesh-space vertex hops, NOT "
    "detector-grid pixels, and NOT world-space millimeters."
)


# ---------------------------------------------------------------------------
# Stage 2: ring transform candidate spec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RingOffsetSweepCandidateSpec:
    """Spec for one ring-offset candidate. See module docstring stage 2."""

    name: str
    inner_radius_px: int
    outer_radius_px: int
    risk_epsilon: float = 0.01


# ---------------------------------------------------------------------------
# Stage 5 helpers: selection-vs-holdout split + subset summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConditionSubsetDiagnostics:
    """Aggregate diagnostics over a subset of comparison entries.

    Used for the full evaluation grid, the selection-condition
    in-sample subset, and the holdout subset.
    """

    entry_count: int
    pass_count: int
    fail_count: int
    pass_ratio: float
    worst_delta_max_temperature_k: float | None
    best_delta_max_temperature_k: float | None
    best_delta_threshold_count: int | None
    improved_count: int
    worsened_count: int
    mixed_count: int
    unchanged_count: int
    dominant_label: str
    diagnostics_type: str = (
        "thermal_risk_condition_subset_diagnostics"
    )


def _empty_subset_diag() -> ConditionSubsetDiagnostics:
    return ConditionSubsetDiagnostics(
        entry_count=0,
        pass_count=0,
        fail_count=0,
        pass_ratio=0.0,
        worst_delta_max_temperature_k=None,
        best_delta_max_temperature_k=None,
        best_delta_threshold_count=None,
        improved_count=0,
        worsened_count=0,
        mixed_count=0,
        unchanged_count=0,
        dominant_label="unchanged",
    )


def summarize_thermal_risk_comparison_subset(
    entries: Sequence[LegacyThermalRiskComparisonEntry],
) -> ConditionSubsetDiagnostics:
    """Aggregate a sequence of thermal-risk comparison entries.

    Returns a :class:`ConditionSubsetDiagnostics` with pass/fail
    counts, worst / best delta-max-temperature, best (most
    negative) delta-threshold-count, tradeoff-label histogram, and
    the dominant tradeoff label. Empty input returns an empty
    summary with ``entry_count == 0`` and ``None`` aggregates.

    The function is shared between the full evaluation grid, the
    selection-condition in-sample subset, and the holdout subset.
    """
    n = int(len(entries))
    if n == 0:
        return _empty_subset_diag()

    pass_count = 0
    fail_count = 0
    improved_count = 0
    worsened_count = 0
    mixed_count = 0
    unchanged_count = 0
    delta_max_t: list[float] = []
    delta_count: list[int] = []
    for e in entries:
        c = e.comparison
        delta_max_t.append(float(c.delta_max_temperature_k))
        delta_count.append(int(c.delta_threshold_exceeded_count))
        if bool(c.passes_guardrails):
            pass_count += 1
        else:
            fail_count += 1
        label = str(c.tradeoff_label)
        if label == "improved":
            improved_count += 1
        elif label == "worsened":
            worsened_count += 1
        elif label == "mixed":
            mixed_count += 1
        else:
            unchanged_count += 1

    arr_t = np.asarray(delta_max_t, dtype=float)
    arr_c = np.asarray(delta_count, dtype=np.int64)
    label_counts = {
        "improved": improved_count,
        "worsened": worsened_count,
        "mixed": mixed_count,
        "unchanged": unchanged_count,
    }
    dominant = max(label_counts.items(), key=lambda kv: kv[1])[0]
    return ConditionSubsetDiagnostics(
        entry_count=n,
        pass_count=int(pass_count),
        fail_count=int(fail_count),
        pass_ratio=float(pass_count) / float(n),
        worst_delta_max_temperature_k=float(arr_t.max()),
        best_delta_max_temperature_k=float(arr_t.min()),
        best_delta_threshold_count=int(arr_c.min()),
        improved_count=int(improved_count),
        worsened_count=int(worsened_count),
        mixed_count=int(mixed_count),
        unchanged_count=int(unchanged_count),
        dominant_label=str(dominant),
    )


def split_selected_vs_holdout_conditions(
    comparison: LegacyThermalRiskComparisonResult,
    selections: Sequence[LegacyConditionSelection],
) -> tuple[
    tuple[LegacyThermalRiskComparisonEntry, ...],
    tuple[LegacyThermalRiskComparisonEntry, ...],
]:
    """Split comparison entries into in-sample and holdout subsets.

    A comparison entry is **in-sample** when its
    ``(angle_degrees, detector_distance)`` matches any
    :class:`LegacyConditionSelection` that drove the selection
    stage; it is **holdout** otherwise. Selection entries with
    duplicate ``(angle, distance)`` keys collapse into the same
    membership test (no double counting). The returned subsets
    preserve the input ``comparison.entries`` order.

    Raises
    ------
    OpticsError
        On invalid ``comparison`` type or invalid ``selections``
        entries.
    """
    if not isinstance(
        comparison, LegacyThermalRiskComparisonResult,
    ):
        raise OpticsError(
            "comparison must be a LegacyThermalRiskComparisonResult; "
            f"got {type(comparison).__name__}"
        )
    selected_keys: set[tuple[float, float]] = set()
    for s in selections:
        if not isinstance(s, LegacyConditionSelection):
            raise OpticsError(
                "selections entries must be "
                "LegacyConditionSelection instances; got "
                f"{type(s).__name__}"
            )
        selected_keys.add(
            (float(s.angle_degrees), float(s.detector_distance))
        )

    in_sample: list[LegacyThermalRiskComparisonEntry] = []
    holdout: list[LegacyThermalRiskComparisonEntry] = []
    for e in comparison.entries:
        key = (
            float(e.angle_degrees),
            float(e.detector_distance),
        )
        if key in selected_keys:
            in_sample.append(e)
        else:
            holdout.append(e)
    return tuple(in_sample), tuple(holdout)


def compute_holdout_diagnostics(
    comparison: LegacyThermalRiskComparisonResult,
    selections: Sequence[LegacyConditionSelection],
) -> ConditionSubsetDiagnostics:
    """Convenience: split and summarize the holdout subset only."""
    _, holdout = split_selected_vs_holdout_conditions(
        comparison, selections,
    )
    return summarize_thermal_risk_comparison_subset(holdout)


# ---------------------------------------------------------------------------
# Stage 5 entry / result dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RingOffsetSweepEntry:
    """One ring-offset candidate's full diagnostic record.

    Coverage / complexity diagnostics
    ---------------------------------
    - ``risk_support_expansion_ratio`` is
      ``transformed_risk_active_count / source_risk_active_count``
      and reflects how aggressively the ring transform spreads
      the source risk support.
    - ``moved_vertex_fraction`` is
      ``moved_vertex_count / body_mask_vertex_count`` (or zero
      when the body mask is empty), reflecting how much of the
      patterned region was actually displaced.

    Hotspot overlap diagnostics
    ---------------------------
    - ``baseline_nonzero_bins`` and ``candidate_nonzero_bins``
      are bin counts on the worst-worsened comparison entry's
      contribution map (computed via
      :func:`build_pattern_induced_hotspot_diagnostic`).
    - ``source_overlap_bin_count`` (= shared bins) and
      ``source_overlap_fraction`` (= shared / baseline) report
      how much of the baseline hotspot footprint the candidate
      retains. ``retained_source_bin_fraction`` is the same
      fraction; both names are kept so consumers can choose the
      one that reads best in their context.
    - ``candidate_only_bin_count`` and ``new_bin_fraction``
      report how much *new* hotspot footprint the candidate
      created. A zero ``new_bin_fraction`` paired with a high
      ``source_overlap_fraction`` is **not** a "good
      containment"; it is "the pattern did not move the hotspot
      and may have strengthened it".
    - ``jaccard_overlap`` = shared / (baseline + candidate -
      shared) summarizes the symmetric overlap.

    Comparison summaries
    --------------------
    ``full_diagnostics``, ``in_sample_diagnostics``, and
    ``holdout_diagnostics`` are
    :class:`ConditionSubsetDiagnostics` instances summarizing the
    full evaluation grid, the selection-condition in-sample
    subset, and the holdout subset respectively. Ranking on the
    sweep result defaults to the holdout subset.
    """

    candidate: RingOffsetSweepCandidateSpec
    units_description: str
    source_risk_active_count: int
    transformed_risk_active_count: int
    risk_support_expansion_ratio: float
    moved_vertex_count: int
    moved_vertex_fraction: float
    body_mask_vertex_count: int
    max_displacement: float
    full_diagnostics: ConditionSubsetDiagnostics
    in_sample_diagnostics: ConditionSubsetDiagnostics
    holdout_diagnostics: ConditionSubsetDiagnostics
    hotspot_diagnostic_angle_degrees: float
    hotspot_diagnostic_detector_distance: float
    baseline_nonzero_bins: int
    candidate_nonzero_bins: int
    source_overlap_bin_count: int
    source_overlap_fraction: float
    retained_source_bin_fraction: float
    candidate_only_bin_count: int
    new_bin_fraction: float
    jaccard_overlap: float


@dataclass(frozen=True)
class RingOffsetSweepResult:
    """Sweep diagnostic result with explicit selection / holdout split.

    Ranking summary
    ---------------
    - ``pareto_basis`` is one of ``"holdout"`` (preferred) or
      ``"full"`` (fallback when the holdout subset is empty).
    - ``pareto_axes`` is the tuple of axis names that actually
      participated in the Pareto filter; non-discriminative axes
      (range zero across candidates on the chosen basis) are
      reported in ``non_discriminative_metrics`` and dropped from
      the filter so they do not give the false impression of a
      multi-axis decision.
    - ``best_by_metric_names`` and ``best_by_metric_values`` map
      each axis to the candidate that minimizes it on the chosen
      basis. When
      ``no_improving_candidate_under_current_sweep`` is ``True``,
      the same entries are *least-bad* on that axis, not "best"
      in any safety sense.
    - ``composite_ranking_names`` is the candidate names sorted
      ascending by a min-max-normalized composite score on the
      Pareto axes. Composite is intended as a *tie-break / hint*
      inside the Pareto set, not as a primary ranking.

    Negative-result flags
    ---------------------
    - ``no_improving_candidate_under_current_sweep`` is ``True``
      when **every** candidate's full-grid
      ``worst_delta_max_temperature_k`` is strictly positive
      (i.e. every candidate worsens the worst-case max
      temperature).
    - ``no_improving_candidate_in_holdout`` is the same predicate
      but evaluated on each candidate's holdout subset.
    """

    entries: tuple[RingOffsetSweepEntry, ...]
    candidate_count: int
    units_description: str
    selected_condition_count: int
    evaluation_condition_count: int
    holdout_condition_count: int
    pareto_basis: str
    pareto_axes: tuple[str, ...]
    pareto_non_dominated_names: tuple[str, ...]
    non_discriminative_metrics: tuple[str, ...]
    best_by_metric_names: dict[str, str | None]
    best_by_metric_values: dict[str, float]
    composite_ranking_names: tuple[str, ...]
    composite_best_name: str | None
    composite_best_score: float | None
    composite_basis: str
    no_improving_candidate_under_current_sweep: bool
    no_improving_candidate_in_holdout: bool
    result_type: str = "ring_offset_sweep_diagnostic"


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _check_positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OpticsError(
            f"{name} must be a positive int; got {value!r}"
        )
    if int(value) <= 0:
        raise OpticsError(
            f"{name} must be a positive int; got {value}"
        )
    return int(value)


def _check_positive_float(value: float, *, name: str) -> float:
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        raise OpticsError(
            f"{name} must be a finite float > 0; got {v}"
        )
    return v


def _validate_candidate(
    cand: RingOffsetSweepCandidateSpec, *, index: int,
) -> None:
    if not isinstance(cand, RingOffsetSweepCandidateSpec):
        raise OpticsError(
            f"candidates[{index}] must be a "
            f"RingOffsetSweepCandidateSpec; got "
            f"{type(cand).__name__}"
        )
    if not isinstance(cand.name, str) or not cand.name:
        raise OpticsError(
            f"candidates[{index}].name must be a non-empty str; "
            f"got {cand.name!r}"
        )
    if (
        isinstance(cand.inner_radius_px, bool)
        or not isinstance(cand.inner_radius_px, int)
        or int(cand.inner_radius_px) < 0
    ):
        raise OpticsError(
            f"candidates[{index}].inner_radius_px must be a "
            f"non-negative int; got {cand.inner_radius_px!r}"
        )
    if (
        isinstance(cand.outer_radius_px, bool)
        or not isinstance(cand.outer_radius_px, int)
        or int(cand.outer_radius_px) <= int(cand.inner_radius_px)
    ):
        raise OpticsError(
            f"candidates[{index}].outer_radius_px must be a int "
            f"strictly greater than inner_radius_px; got "
            f"inner={cand.inner_radius_px!r}, "
            f"outer={cand.outer_radius_px!r}"
        )
    eps = float(cand.risk_epsilon)
    if not math.isfinite(eps) or eps < 0.0:
        raise OpticsError(
            f"candidates[{index}].risk_epsilon must be a finite "
            f"float >= 0; got {cand.risk_epsilon}"
        )


def _patterned_setup_to_legacy(
    setup: LegacyPatternedPetWaterSetup,
) -> LegacyPetWaterTraceSetup:
    original = setup.original_setup
    return LegacyPetWaterTraceSetup(
        shell_mesh=setup.patterned_shell_mesh,
        water_mesh=setup.patterned_water_mesh,
        step_specs=setup.patterned_step_specs,
        scale_report=original.scale_report,
        inner_offset_report=setup.patterned_inner_offset_report,
        target_height=original.target_height,
        target_diameter=original.target_diameter,
        wall_thickness=original.wall_thickness,
        ior_air=original.ior_air,
        ior_pet=original.ior_pet,
        ior_water=original.ior_water,
    )


# ---------------------------------------------------------------------------
# Ranking helpers
# ---------------------------------------------------------------------------


_PARETO_AXIS_NAMES = (
    "delta_max_temperature_k",
    "delta_threshold_count",
    "pass_ratio_neg",
)


def _axis_value(
    entry: RingOffsetSweepEntry, axis: str, *, basis: str,
) -> float:
    """Extract the chosen axis value from the chosen subset.

    All axes are oriented "lower is better". ``pass_ratio_neg``
    is ``-pass_ratio`` so that higher pass ratios produce smaller
    (better) values on the same axis convention.
    """
    if basis == "full":
        diag = entry.full_diagnostics
    elif basis == "in_sample":
        diag = entry.in_sample_diagnostics
    elif basis == "holdout":
        diag = entry.holdout_diagnostics
    else:
        raise OpticsError(
            f"basis must be one of full/in_sample/holdout; "
            f"got {basis!r}"
        )
    if axis == "delta_max_temperature_k":
        v = diag.worst_delta_max_temperature_k
        return float("inf") if v is None else float(v)
    if axis == "delta_threshold_count":
        v = diag.best_delta_threshold_count
        return float("inf") if v is None else float(int(v))
    if axis == "pass_ratio_neg":
        return -float(diag.pass_ratio)
    raise OpticsError(f"unknown axis {axis!r}")


def _is_basis_populated(
    entries: Sequence[RingOffsetSweepEntry], basis: str,
) -> bool:
    for e in entries:
        if basis == "full":
            n = int(e.full_diagnostics.entry_count)
        elif basis == "in_sample":
            n = int(e.in_sample_diagnostics.entry_count)
        elif basis == "holdout":
            n = int(e.holdout_diagnostics.entry_count)
        else:
            return False
        if n > 0:
            return True
    return False


def _resolve_basis(
    entries: Sequence[RingOffsetSweepEntry],
    requested: str,
) -> str:
    if requested in ("full", "in_sample", "holdout"):
        if _is_basis_populated(entries, requested):
            return requested
        # Holdout fall-through to full.
        if requested == "holdout" and _is_basis_populated(
            entries, "full"
        ):
            return "full"
        return requested
    raise OpticsError(
        f"basis must be one of full/in_sample/holdout; "
        f"got {requested!r}"
    )


def _axis_is_discriminative(
    entries: Sequence[RingOffsetSweepEntry], axis: str, *,
    basis: str, tol: float = 1e-12,
) -> bool:
    if len(entries) == 0:
        return False
    values = np.array(
        [_axis_value(e, axis, basis=basis) for e in entries],
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return False
        return float(finite.max() - finite.min()) > tol
    return float(values.max() - values.min()) > tol


def rank_ring_offset_sweep_pareto_indices(
    entries: Sequence[RingOffsetSweepEntry], *,
    basis: str = "holdout",
) -> tuple[int, ...]:
    """Pareto non-dominated filter on the chosen subset basis.

    Diagnostic Pareto filter on
    ``(delta_max_temperature_k, delta_threshold_count,
    -pass_ratio)``; all three are lower-is-better in that
    coordinate. The basis selects which
    :class:`ConditionSubsetDiagnostics` field to read; it
    defaults to ``"holdout"`` and falls back to ``"full"`` when
    the holdout subset is empty for every candidate.

    Non-discriminative axes (range zero across candidates) are
    automatically excluded so a metric that happens to be
    constant in this sweep does not give the false impression of
    a multi-axis decision. The returned indices preserve the
    input order.
    """
    if not entries:
        return ()
    use_basis = _resolve_basis(entries, basis)
    discriminative_axes = tuple(
        axis for axis in _PARETO_AXIS_NAMES
        if _axis_is_discriminative(entries, axis, basis=use_basis)
    )
    if len(discriminative_axes) == 0:
        # No metric distinguishes candidates; treat them all as
        # non-dominated.
        return tuple(range(len(entries)))

    n = len(entries)
    coords = np.empty((n, len(discriminative_axes)), dtype=float)
    for i, e in enumerate(entries):
        for j, axis in enumerate(discriminative_axes):
            coords[i, j] = _axis_value(e, axis, basis=use_basis)

    nondominated: list[int] = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            le_all = bool(np.all(coords[j] <= coords[i]))
            lt_any = bool(np.any(coords[j] < coords[i]))
            if le_all and lt_any:
                dominated = True
                break
        if not dominated:
            nondominated.append(i)
    return tuple(nondominated)


def _normalize_lower_better(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr, dtype=float)
    lo = float(finite.min())
    hi = float(finite.max())
    if hi - lo <= 0.0:
        return np.zeros_like(arr, dtype=float)
    norm = np.zeros_like(arr, dtype=float)
    mask = np.isfinite(arr)
    norm[mask] = (arr[mask] - lo) / (hi - lo)
    norm[~mask] = 1.0  # treat non-finite as worst
    return norm


def _composite_scores(
    entries: Sequence[RingOffsetSweepEntry], *,
    basis: str,
    axes: Sequence[str],
    weights: Sequence[float],
) -> np.ndarray:
    n = len(entries)
    if n == 0:
        return np.empty(0, dtype=float)
    score = np.zeros(n, dtype=float)
    for axis, weight in zip(axes, weights):
        col = np.array(
            [_axis_value(e, axis, basis=basis) for e in entries],
            dtype=float,
        )
        score = score + float(weight) * _normalize_lower_better(col)
    return score


def rank_ring_offset_sweep_by_composite_score(
    entries: Sequence[RingOffsetSweepEntry], *,
    basis: str = "holdout",
    weight_max_temperature: float = 1.0,
    weight_threshold_count: float = 1.0,
    weight_pass_ratio: float = 1.0,
) -> tuple[int, ...]:
    """Rank candidates ascending by min-max-normalized composite score.

    Diagnostic ranking only — **not** safety certification. The
    composite is a weighted sum of min-max-normalized
    discriminative axes from
    :func:`rank_ring_offset_sweep_pareto_indices`. Non-finite
    values are treated as worst-case (1.0 after normalization).
    Constant axes contribute zero to every score and are
    effectively ignored. Indices are returned in ascending score
    order.
    """
    if not entries:
        return ()
    if (
        not math.isfinite(float(weight_max_temperature))
        or float(weight_max_temperature) < 0.0
        or not math.isfinite(float(weight_threshold_count))
        or float(weight_threshold_count) < 0.0
        or not math.isfinite(float(weight_pass_ratio))
        or float(weight_pass_ratio) < 0.0
    ):
        raise OpticsError(
            "composite weights must be finite floats >= 0"
        )
    use_basis = _resolve_basis(entries, basis)
    axes = tuple(
        axis for axis in _PARETO_AXIS_NAMES
        if _axis_is_discriminative(entries, axis, basis=use_basis)
    )
    weight_lookup = {
        "delta_max_temperature_k": float(weight_max_temperature),
        "delta_threshold_count": float(weight_threshold_count),
        "pass_ratio_neg": float(weight_pass_ratio),
    }
    weights = tuple(weight_lookup[a] for a in axes)
    scores = _composite_scores(
        entries, basis=use_basis, axes=axes, weights=weights,
    )
    order = np.argsort(scores, kind="stable")
    return tuple(int(i) for i in order.tolist())


# ---------------------------------------------------------------------------
# Stage 4 + 5: end-to-end sweep
# ---------------------------------------------------------------------------


def _no_improving_predicate(
    entries: Sequence[RingOffsetSweepEntry], *, basis: str,
) -> bool:
    if len(entries) == 0:
        return False
    for e in entries:
        if basis == "full":
            v = e.full_diagnostics.worst_delta_max_temperature_k
        elif basis == "in_sample":
            v = e.in_sample_diagnostics.worst_delta_max_temperature_k
        elif basis == "holdout":
            v = e.holdout_diagnostics.worst_delta_max_temperature_k
        else:
            return False
        if v is None:
            # Treat empty subset as "no information" — do not flag.
            return False
        if float(v) <= 0.0:
            return False
    return True


def _best_by_axis(
    entries: Sequence[RingOffsetSweepEntry], axis: str, *,
    basis: str,
) -> tuple[str | None, float]:
    if len(entries) == 0:
        return None, float("inf")
    values = np.array(
        [_axis_value(e, axis, basis=basis) for e in entries],
        dtype=float,
    )
    idx = int(np.argmin(values))
    return str(entries[idx].candidate.name), float(values[idx])


def run_actual_stl_ring_offset_sweep(
    *,
    original_setup: LegacyPetWaterTraceSetup,
    baseline_thermal_scan: LegacyThermalRiskScanResult,
    aggregate_risk_map: RiskMap,
    body_include_mask: np.ndarray,
    candidates: Sequence[RingOffsetSweepCandidateSpec],
    selections: Sequence[LegacyConditionSelection],
    angles_degrees: tuple[float, ...],
    detector_distances: tuple[float, ...],
    source_width: float,
    source_height: float,
    source_radius: float,
    sample_count_y: int,
    sample_count_z: int,
    detector_size: float,
    detector_resolution: tuple[int, int],
    duration_s: float,
    dt_s: float,
    pattern_count: int = 20,
    pattern_amplitude: float = 1.0,
    pattern_sigma_u: float = 0.03,
    pattern_sigma_v: float = 0.03,
    pattern_max_depth: float = 0.05,
    pattern_seed: int = 42,
    active_threshold: float = 0.01,
    nominal_incident_irradiance_w_m2: float = 1000.0,
    areal_heat_capacity_j_m2k: float = 1200.0,
    absorptivity: float = 0.8,
    h_conv_w_m2k: float = 10.0,
    emissivity: float = 0.9,
    ambient_temp_k: float = 293.15,
    threshold_temp_k: float = 373.15,
    wall_thickness: float = 0.3,
    inner_offset_mode: str = "auto",
    hotspot_top_percent: float = 10.0,
    contribution_resolution: tuple[int, int] = (32, 64),
    guardrails: ThermalRiskGuardrailConfig | None = None,
    ranking_basis: str = "holdout",
    composite_weight_max_temperature: float = 1.0,
    composite_weight_threshold_count: float = 1.0,
    composite_weight_pass_ratio: float = 1.0,
) -> RingOffsetSweepResult:
    """Run a ring-offset diagnostic sweep with selection / holdout split.

    Actual STL ring-offset risk-guided pattern parameter sweep
    diagnostic; **not a physical PET-bottle validation**.
    Pattern shape parameters (``pattern_count``, sigmas,
    ``max_depth``, ``seed``) are held fixed across all
    candidates so the sweep isolates the ring transform's effect
    on hotspot redistribution.

    The ``selections`` argument **must** be the same selection
    list that was used to build ``aggregate_risk_map``; the sweep
    uses it to mark each comparison entry as in-sample (the
    selection condition itself) or holdout (every other condition
    on the evaluation grid). Ranking defaults to ``"holdout"``.

    Raises
    ------
    OpticsError
        On invalid setup type, invalid candidate spec, empty
        candidate list, non-positive sample counts, invalid
        detector resolution, or invalid ranking basis.
    PatternError, GeometryError, ThermalError, ContributionError
        From downstream pipeline calls.
    """
    if not isinstance(original_setup, LegacyPetWaterTraceSetup):
        raise OpticsError(
            "original_setup must be a LegacyPetWaterTraceSetup; "
            f"got {type(original_setup).__name__}"
        )
    if not isinstance(
        baseline_thermal_scan, LegacyThermalRiskScanResult,
    ):
        raise OpticsError(
            "baseline_thermal_scan must be a "
            "LegacyThermalRiskScanResult; got "
            f"{type(baseline_thermal_scan).__name__}"
        )
    if not isinstance(aggregate_risk_map, RiskMap):
        raise OpticsError(
            "aggregate_risk_map must be a RiskMap; got "
            f"{type(aggregate_risk_map).__name__}"
        )
    if ranking_basis not in ("holdout", "full", "in_sample"):
        raise OpticsError(
            "ranking_basis must be one of "
            "holdout/full/in_sample; got "
            f"{ranking_basis!r}"
        )

    try:
        cand_list = list(candidates)
    except TypeError as exc:
        raise OpticsError(
            "candidates must be a Sequence of "
            "RingOffsetSweepCandidateSpec"
        ) from exc
    if len(cand_list) == 0:
        raise OpticsError(
            "candidates must contain at least one "
            "RingOffsetSweepCandidateSpec"
        )
    seen_names: set[str] = set()
    for i, c in enumerate(cand_list):
        _validate_candidate(c, index=i)
        if c.name in seen_names:
            raise OpticsError(
                f"candidates[{i}].name {c.name!r} is duplicated"
            )
        seen_names.add(c.name)

    selections_list = list(selections)
    for i, s in enumerate(selections_list):
        if not isinstance(s, LegacyConditionSelection):
            raise OpticsError(
                f"selections[{i}] must be a "
                "LegacyConditionSelection; got "
                f"{type(s).__name__}"
            )

    angles = tuple(float(a) for a in angles_degrees)
    distances = tuple(float(d) for d in detector_distances)
    if len(angles) == 0 or len(distances) == 0:
        raise OpticsError(
            "angles_degrees and detector_distances must both be "
            "non-empty"
        )
    _check_positive_int(int(sample_count_y), name="sample_count_y")
    _check_positive_int(int(sample_count_z), name="sample_count_z")
    _check_positive_float(
        float(source_width), name="source_width",
    )
    _check_positive_float(
        float(source_height), name="source_height",
    )
    _check_positive_float(
        float(source_radius), name="source_radius",
    )
    _check_positive_float(
        float(detector_size), name="detector_size",
    )
    _check_positive_float(float(duration_s), name="duration_s")
    _check_positive_float(float(dt_s), name="dt_s")

    res = tuple(detector_resolution)
    if len(res) != 2:
        raise OpticsError(
            f"detector_resolution must be (ny, nx); got {res}"
        )
    _check_positive_int(int(res[0]), name="detector_resolution[0]")
    _check_positive_int(int(res[1]), name="detector_resolution[1]")

    body_mask_arr = np.asarray(body_include_mask)
    body_mask_vertex_count = int(body_mask_arr.sum())

    scan_kwargs = dict(
        angles_degrees=angles,
        detector_distances=distances,
        source_width=float(source_width),
        source_height=float(source_height),
        sample_count_y=int(sample_count_y),
        sample_count_z=int(sample_count_z),
        detector_size=float(detector_size),
        detector_resolution=(int(res[0]), int(res[1])),
        source_radius=float(source_radius),
        epsilon=0.1,
        store_irradiance_surrogate=True,
    )
    thermal_kwargs = dict(
        nominal_incident_irradiance_w_m2=float(
            nominal_incident_irradiance_w_m2,
        ),
        duration_s=float(duration_s),
        dt_s=float(dt_s),
        areal_heat_capacity_j_m2k=float(areal_heat_capacity_j_m2k),
        absorptivity=float(absorptivity),
        h_conv_w_m2k=float(h_conv_w_m2k),
        emissivity=float(emissivity),
        ambient_temp_k=float(ambient_temp_k),
        threshold_temp_k=float(threshold_temp_k),
    )

    sweep_entries: list[RingOffsetSweepEntry] = []
    for c in cand_list:
        ring_result = create_ring_offset_risk_map(
            aggregate_risk_map,
            inner_radius_px=int(c.inner_radius_px),
            outer_radius_px=int(c.outer_radius_px),
            wrap_u=True,
            epsilon=float(c.risk_epsilon),
        )
        ring_risk = ring_result.transformed_risk_map
        if int(ring_risk.active_count) == 0:
            raise OpticsError(
                f"candidate {c.name!r} produced an empty ring "
                "risk map; (inner, outer) too aggressive for the "
                "input risk map"
            )

        patterned_setup = (
            create_risk_guided_legacy_patterned_pet_water_setup(
                original_setup=original_setup,
                risk_map=ring_risk,
                body_include_mask=body_include_mask,
                pattern_count=int(pattern_count),
                pattern_amplitude=float(pattern_amplitude),
                pattern_sigma_u=float(pattern_sigma_u),
                pattern_sigma_v=float(pattern_sigma_v),
                pattern_max_depth=float(pattern_max_depth),
                pattern_seed=int(pattern_seed),
                active_threshold=float(active_threshold),
                wall_thickness=float(wall_thickness),
                inner_offset_mode=str(inner_offset_mode),
            )
        )
        cand_legacy = _patterned_setup_to_legacy(patterned_setup)
        cand_optical = run_legacy_pet_water_angle_distance_scan(
            setup=cand_legacy, **scan_kwargs,
        )
        cand_thermal = compute_thermal_risk_over_legacy_optical_scan(
            cand_optical, **thermal_kwargs,
        )
        comparison = compare_legacy_thermal_risk_scans(
            baseline_thermal_scan, cand_thermal,
            guardrails=guardrails,
        )
        diagnostic = build_pattern_induced_hotspot_diagnostic(
            baseline_setup=original_setup,
            candidate_setup=cand_legacy,
            comparison_result=comparison,
            source_width=float(source_width),
            source_height=float(source_height),
            source_radius=float(source_radius),
            sample_count_y=int(sample_count_y),
            sample_count_z=int(sample_count_z),
            detector_size=float(detector_size),
            detector_resolution=(int(res[0]), int(res[1])),
            epsilon=0.1,
            hotspot_top_percent=float(hotspot_top_percent),
            contribution_resolution=(
                int(contribution_resolution[0]),
                int(contribution_resolution[1]),
            ),
        )

        in_sample_entries, holdout_entries = (
            split_selected_vs_holdout_conditions(
                comparison, selections_list,
            )
        )
        full_diag = summarize_thermal_risk_comparison_subset(
            comparison.entries,
        )
        in_sample_diag = (
            summarize_thermal_risk_comparison_subset(
                in_sample_entries,
            )
        )
        holdout_diag = summarize_thermal_risk_comparison_subset(
            holdout_entries,
        )

        moved = patterned_setup.patterned_mesh_result
        if body_mask_vertex_count > 0:
            moved_fraction = (
                float(moved.moved_vertex_count)
                / float(body_mask_vertex_count)
            )
        else:
            moved_fraction = 0.0
        if int(ring_result.source_active_count) > 0:
            expansion = (
                float(ring_result.transformed_active_count)
                / float(ring_result.source_active_count)
            )
        else:
            expansion = 0.0

        overlap = diagnostic.overlap
        union_size = (
            int(overlap.baseline_nonzero_bins)
            + int(overlap.candidate_nonzero_bins)
            - int(overlap.shared_nonzero_bins)
        )
        if union_size > 0:
            jaccard = (
                float(overlap.shared_nonzero_bins)
                / float(union_size)
            )
        else:
            jaccard = 0.0

        sweep_entries.append(
            RingOffsetSweepEntry(
                candidate=c,
                units_description=RING_RADIUS_UNITS_DESCRIPTION,
                source_risk_active_count=int(
                    ring_result.source_active_count,
                ),
                transformed_risk_active_count=int(
                    ring_result.transformed_active_count,
                ),
                risk_support_expansion_ratio=float(expansion),
                moved_vertex_count=int(moved.moved_vertex_count),
                moved_vertex_fraction=float(moved_fraction),
                body_mask_vertex_count=int(body_mask_vertex_count),
                max_displacement=float(moved.max_displacement),
                full_diagnostics=full_diag,
                in_sample_diagnostics=in_sample_diag,
                holdout_diagnostics=holdout_diag,
                hotspot_diagnostic_angle_degrees=float(
                    diagnostic.worsened_angle_degrees,
                ),
                hotspot_diagnostic_detector_distance=float(
                    diagnostic.worsened_detector_distance,
                ),
                baseline_nonzero_bins=int(
                    overlap.baseline_nonzero_bins,
                ),
                candidate_nonzero_bins=int(
                    overlap.candidate_nonzero_bins,
                ),
                source_overlap_bin_count=int(
                    overlap.shared_nonzero_bins,
                ),
                source_overlap_fraction=float(
                    overlap.overlap_fraction_of_baseline,
                ),
                retained_source_bin_fraction=float(
                    overlap.overlap_fraction_of_baseline,
                ),
                candidate_only_bin_count=int(
                    overlap.candidate_only_bins,
                ),
                new_bin_fraction=float(
                    overlap.candidate_new_bin_fraction,
                ),
                jaccard_overlap=float(jaccard),
            )
        )

    selected_keys = {
        (float(s.angle_degrees), float(s.detector_distance))
        for s in selections_list
    }
    selected_condition_count = int(len(selected_keys))
    full_count = (
        int(sweep_entries[0].full_diagnostics.entry_count)
        if sweep_entries else 0
    )
    holdout_count = (
        int(sweep_entries[0].holdout_diagnostics.entry_count)
        if sweep_entries else 0
    )

    use_basis = _resolve_basis(sweep_entries, ranking_basis)
    discriminative_axes = tuple(
        axis for axis in _PARETO_AXIS_NAMES
        if _axis_is_discriminative(
            sweep_entries, axis, basis=use_basis,
        )
    )
    non_discriminative = tuple(
        axis for axis in _PARETO_AXIS_NAMES
        if axis not in discriminative_axes
    )
    pareto_indices = rank_ring_offset_sweep_pareto_indices(
        sweep_entries, basis=use_basis,
    )
    pareto_names = tuple(
        sweep_entries[i].candidate.name for i in pareto_indices
    )

    composite_order = rank_ring_offset_sweep_by_composite_score(
        sweep_entries,
        basis=use_basis,
        weight_max_temperature=float(
            composite_weight_max_temperature,
        ),
        weight_threshold_count=float(
            composite_weight_threshold_count,
        ),
        weight_pass_ratio=float(composite_weight_pass_ratio),
    )
    composite_names = tuple(
        sweep_entries[i].candidate.name for i in composite_order
    )
    if composite_order:
        scores = _composite_scores(
            sweep_entries,
            basis=use_basis,
            axes=discriminative_axes,
            weights=tuple(
                {
                    "delta_max_temperature_k": float(
                        composite_weight_max_temperature
                    ),
                    "delta_threshold_count": float(
                        composite_weight_threshold_count
                    ),
                    "pass_ratio_neg": float(
                        composite_weight_pass_ratio
                    ),
                }[a]
                for a in discriminative_axes
            ),
        )
        idx_min_score = int(composite_order[0])
        composite_best_name: str | None = str(
            sweep_entries[idx_min_score].candidate.name,
        )
        composite_best_score: float | None = float(
            scores[idx_min_score],
        )
    else:
        composite_best_name = None
        composite_best_score = None

    best_names: dict[str, str | None] = {}
    best_values: dict[str, float] = {}
    for axis in _PARETO_AXIS_NAMES:
        nm, val = _best_by_axis(
            sweep_entries, axis, basis=use_basis,
        )
        best_names[axis] = nm
        best_values[axis] = float(val)

    no_improving_full = _no_improving_predicate(
        sweep_entries, basis="full",
    )
    no_improving_holdout = _no_improving_predicate(
        sweep_entries, basis="holdout",
    )

    return RingOffsetSweepResult(
        entries=tuple(sweep_entries),
        candidate_count=int(len(sweep_entries)),
        units_description=RING_RADIUS_UNITS_DESCRIPTION,
        selected_condition_count=selected_condition_count,
        evaluation_condition_count=full_count,
        holdout_condition_count=holdout_count,
        pareto_basis=str(use_basis),
        pareto_axes=tuple(discriminative_axes),
        pareto_non_dominated_names=pareto_names,
        non_discriminative_metrics=non_discriminative,
        best_by_metric_names=best_names,
        best_by_metric_values=best_values,
        composite_ranking_names=composite_names,
        composite_best_name=composite_best_name,
        composite_best_score=composite_best_score,
        composite_basis=str(use_basis),
        no_improving_candidate_under_current_sweep=bool(
            no_improving_full,
        ),
        no_improving_candidate_in_holdout=bool(
            no_improving_holdout,
        ),
    )


# ---------------------------------------------------------------------------
# Result-invariant validation helper
# ---------------------------------------------------------------------------


def validate_ring_offset_sweep_result(
    result: RingOffsetSweepResult,
) -> tuple[str, ...]:
    """Validate research-design invariants on a sweep result.

    Returns a tuple of human-readable problem descriptions; an
    empty tuple means every checked invariant holds. The helper
    is intended as a *diagnostic* gate, not a runtime proof:
    callers that want a hard failure can ``assert not
    validate_ring_offset_sweep_result(result)`` after the sweep
    runs.

    Checked invariants (each labelled in the returned strings)
    -------------------------------------------------------
    - ``selection_count`` + ``holdout_count`` ==
      ``evaluation_condition_count``.
    - Per-entry ``in_sample.entry_count`` +
      ``holdout.entry_count`` == ``full.entry_count``.
    - Each entry's ``holdout.entry_count`` ==
      ``result.holdout_condition_count``.
    - For every ``ConditionSubsetDiagnostics`` reachable through
      the result, ``pass_count + fail_count == entry_count`` and
      ``0 <= pass_ratio <= 1``.
    - All bin counts (``baseline_nonzero_bins``,
      ``candidate_nonzero_bins``, ``source_overlap_bin_count``,
      ``candidate_only_bin_count``) are non-negative.
    - All overlap fractions (``source_overlap_fraction``,
      ``retained_source_bin_fraction``, ``new_bin_fraction``,
      ``jaccard_overlap``, ``moved_vertex_fraction``) are within
      ``[0, 1]``.
    - ``risk_support_expansion_ratio`` is finite and >= 0.
    - Per candidate, ``outer_radius_px > inner_radius_px >= 0``.
    - ``no_improving_candidate_in_holdout`` matches the actual
      holdout ``worst_delta_max_temperature_k`` values for all
      entries.
    - Every ``best_by_metric_names[axis]`` (when not ``None``)
      identifies a real entry in ``result.entries``.
    - Every name in ``pareto_non_dominated_names`` and
      ``composite_ranking_names`` identifies a real entry.
    - ``non_discriminative_metrics`` and ``pareto_axes`` are
      disjoint and their union is a subset of the known axis
      names.
    - When all axes are non-discriminative, no
      ``best_by_metric_names`` value is silently promoted to a
      "winner" without callers being able to detect it.
    """
    problems: list[str] = []

    if not isinstance(result, RingOffsetSweepResult):
        return (
            f"result must be a RingOffsetSweepResult; got "
            f"{type(result).__name__}",
        )

    sel = int(result.selected_condition_count)
    holdout = int(result.holdout_condition_count)
    eval_count = int(result.evaluation_condition_count)
    if sel + holdout != eval_count:
        problems.append(
            f"selection_count + holdout_count "
            f"({sel} + {holdout}) != "
            f"evaluation_condition_count ({eval_count})"
        )

    known_axes = {
        "delta_max_temperature_k",
        "delta_threshold_count",
        "pass_ratio_neg",
    }
    pareto_axis_set = set(result.pareto_axes)
    nd_set = set(result.non_discriminative_metrics)
    if not pareto_axis_set.isdisjoint(nd_set):
        problems.append(
            "pareto_axes and non_discriminative_metrics overlap: "
            f"{sorted(pareto_axis_set & nd_set)}"
        )
    if not (pareto_axis_set | nd_set).issubset(known_axes):
        problems.append(
            "pareto_axes / non_discriminative_metrics contain "
            f"unknown axis names: "
            f"{sorted((pareto_axis_set | nd_set) - known_axes)}"
        )

    entry_names = {e.candidate.name for e in result.entries}
    for axis, nm in result.best_by_metric_names.items():
        if nm is None:
            continue
        if nm not in entry_names:
            problems.append(
                f"best_by_metric_names[{axis!r}] = {nm!r} not "
                "in result.entries"
            )
    for nm in result.pareto_non_dominated_names:
        if nm not in entry_names:
            problems.append(
                f"pareto_non_dominated_names contains {nm!r} "
                "not in result.entries"
            )
    for nm in result.composite_ranking_names:
        if nm not in entry_names:
            problems.append(
                f"composite_ranking_names contains {nm!r} not "
                "in result.entries"
            )

    holdout_worst_values: list[float] = []
    for i, entry in enumerate(result.entries):
        c = entry.candidate
        if int(c.outer_radius_px) <= int(c.inner_radius_px):
            problems.append(
                f"entries[{i}].candidate.outer_radius_px "
                f"({c.outer_radius_px}) must be > "
                f"inner_radius_px ({c.inner_radius_px})"
            )
        if int(c.inner_radius_px) < 0:
            problems.append(
                f"entries[{i}].candidate.inner_radius_px "
                f"({c.inner_radius_px}) must be >= 0"
            )

        for label, diag in (
            ("full", entry.full_diagnostics),
            ("in_sample", entry.in_sample_diagnostics),
            ("holdout", entry.holdout_diagnostics),
        ):
            n = int(diag.entry_count)
            if int(diag.pass_count) + int(diag.fail_count) != n:
                problems.append(
                    f"entries[{i}].{label}_diagnostics: "
                    f"pass_count + fail_count "
                    f"({diag.pass_count} + {diag.fail_count}) "
                    f"!= entry_count ({n})"
                )
            if n > 0:
                expected = (
                    float(diag.pass_count) / float(n)
                )
                if abs(float(diag.pass_ratio) - expected) > 1e-9:
                    problems.append(
                        f"entries[{i}].{label}_diagnostics."
                        f"pass_ratio ({diag.pass_ratio}) does not "
                        f"match pass_count/entry_count ({expected})"
                    )
            if not (0.0 <= float(diag.pass_ratio) <= 1.0):
                problems.append(
                    f"entries[{i}].{label}_diagnostics."
                    f"pass_ratio ({diag.pass_ratio}) outside "
                    "[0, 1]"
                )

        n_full = int(entry.full_diagnostics.entry_count)
        n_in = int(entry.in_sample_diagnostics.entry_count)
        n_out = int(entry.holdout_diagnostics.entry_count)
        if n_in + n_out != n_full:
            problems.append(
                f"entries[{i}]: in_sample.entry_count + "
                f"holdout.entry_count ({n_in} + {n_out}) != "
                f"full.entry_count ({n_full})"
            )
        if n_out != holdout:
            problems.append(
                f"entries[{i}].holdout_diagnostics.entry_count "
                f"({n_out}) != "
                f"result.holdout_condition_count ({holdout})"
            )

        for fld_name, count in (
            ("baseline_nonzero_bins", entry.baseline_nonzero_bins),
            ("candidate_nonzero_bins", entry.candidate_nonzero_bins),
            ("source_overlap_bin_count",
             entry.source_overlap_bin_count),
            ("candidate_only_bin_count",
             entry.candidate_only_bin_count),
            ("source_risk_active_count",
             entry.source_risk_active_count),
            ("transformed_risk_active_count",
             entry.transformed_risk_active_count),
            ("moved_vertex_count", entry.moved_vertex_count),
            ("body_mask_vertex_count",
             entry.body_mask_vertex_count),
        ):
            if int(count) < 0:
                problems.append(
                    f"entries[{i}].{fld_name} ({count}) must be "
                    ">= 0"
                )

        for fld_name, frac in (
            ("source_overlap_fraction",
             entry.source_overlap_fraction),
            ("retained_source_bin_fraction",
             entry.retained_source_bin_fraction),
            ("new_bin_fraction", entry.new_bin_fraction),
            ("jaccard_overlap", entry.jaccard_overlap),
            ("moved_vertex_fraction",
             entry.moved_vertex_fraction),
        ):
            if not (0.0 <= float(frac) <= 1.0):
                problems.append(
                    f"entries[{i}].{fld_name} ({frac}) outside "
                    "[0, 1]"
                )

        ratio = float(entry.risk_support_expansion_ratio)
        if not (ratio == ratio and ratio != float("inf")):
            problems.append(
                f"entries[{i}].risk_support_expansion_ratio "
                f"({ratio}) is not finite"
            )
        if int(entry.source_risk_active_count) > 0 and ratio < 0.0:
            problems.append(
                f"entries[{i}].risk_support_expansion_ratio "
                f"({ratio}) must be >= 0 when source active "
                "count > 0"
            )

        v = entry.holdout_diagnostics.worst_delta_max_temperature_k
        if v is not None:
            holdout_worst_values.append(float(v))

    if holdout_worst_values:
        all_positive = all(v > 0.0 for v in holdout_worst_values)
        if (
            bool(result.no_improving_candidate_in_holdout)
            != all_positive
        ):
            problems.append(
                "no_improving_candidate_in_holdout "
                f"({result.no_improving_candidate_in_holdout}) "
                f"does not match holdout worst Δmax temperatures "
                f"(all_positive={all_positive})"
            )

    return tuple(problems)
