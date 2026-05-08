"""Pattern-induced hotspot contribution diagnostics for actual STL.

Actual STL pattern-induced hotspot diagnostic; **not a physical
PET-bottle validation**. Earlier risk-guided pattern sweeps showed
"mixed" tradeoff outcomes where a candidate reduced one
thermal-risk metric but worsened another at some
``(angle_degrees, detector_distance)`` entries. This module asks a
narrower diagnostic question:

    "When the patterned shell is worse than the baseline at a
    specific entry, are the patterned detector hotspots backtracked
    to the same shell-surface bins as the baseline hotspots, or to
    new bins?"

Two pieces:

- :func:`compare_legacy_contribution_maps` reads a baseline and a
  candidate :class:`ContributionMap` of identical shape and
  reports per-bin overlap counts and weight sums (shared,
  baseline-only, candidate-only).
- :func:`build_pattern_induced_hotspot_diagnostic` selects the
  worsened ``(angle_degrees, detector_distance)`` entry from a
  :class:`LegacyThermalRiskComparisonResult`, runs detailed traces
  for ``baseline_setup`` and ``candidate_setup`` at that entry,
  builds a hotspot contribution map for each via
  :func:`build_legacy_hotspot_contribution_map`, and forwards them
  through :func:`compare_legacy_contribution_maps`.

Limitations
-----------
- The contribution map is a **diagnostic** map of where surviving
  hotspot rays last touched the outer shell. It is **not** a
  measured physical risk field.
- "Candidate-only bins" are surrogate ray-backtracking
  diagnostics; they do **not** prove physical causality between
  the pattern and the new caustic location.
- A pattern can reduce one metric while inducing new hotspots
  elsewhere. This module exposes that explicitly; it does not
  optimize the pattern.
- This is **not** fire-prevention validation, **not** PET-bottle
  safety, and **not** manufacturing-ready.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from optics_simulation.contribution.legacy_hotspot_backtracking import (
    LegacyHotspotContributionResult,
    build_legacy_hotspot_contribution_map,
)
from optics_simulation.contribution.map import (
    ContributionError,
    ContributionMap,
)
from optics_simulation.optics.legacy_detailed_trace import (
    run_legacy_pet_water_detailed_trace,
)
from optics_simulation.optics.legacy_pet_water import (
    LegacyPetWaterTraceSetup,
)
from optics_simulation.thermal.legacy_risk_comparison import (
    LegacyThermalRiskComparisonResult,
)
from optics_simulation.thermal.lumped_target import ThermalError


@dataclass(frozen=True)
class ContributionMapOverlapDiagnostic:
    baseline_nonzero_bins: int
    candidate_nonzero_bins: int
    shared_nonzero_bins: int
    baseline_only_bins: int
    candidate_only_bins: int
    overlap_fraction_of_baseline: float
    overlap_fraction_of_candidate: float
    candidate_new_bin_fraction: float
    baseline_total_weight: float
    candidate_total_weight: float
    shared_candidate_weight: float
    candidate_only_weight: float
    result_type: str = "legacy_contribution_map_overlap_diagnostic"


@dataclass(frozen=True)
class PatternInducedHotspotDiagnostic:
    worsened_angle_degrees: float
    worsened_detector_distance: float
    baseline_contribution: LegacyHotspotContributionResult
    candidate_contribution: LegacyHotspotContributionResult
    overlap: ContributionMapOverlapDiagnostic
    baseline_selected_rays: int
    candidate_selected_rays: int
    induced_new_bins: int
    induced_new_bin_fraction: float
    tradeoff_label: str
    delta_max_temperature_k: float
    delta_threshold_exceeded_count: int
    result_type: str = "pattern_induced_hotspot_diagnostic"


def compare_legacy_contribution_maps(
    baseline: ContributionMap,
    candidate: ContributionMap,
) -> ContributionMapOverlapDiagnostic:
    """Compare two contribution maps' nonzero bin sets and weights.

    Actual STL pattern-induced hotspot diagnostic; **not a physical
    PET-bottle validation**. Both maps must be
    :class:`ContributionMap` instances of identical shape. The
    nonzero mask for each side is ``weight_map > 0``. The result
    reports per-bin counts (shared / baseline-only /
    candidate-only) plus the candidate's weight summed over the
    shared and candidate-only bins.

    ``candidate_new_bin_fraction`` is ``candidate_only_bins /
    candidate_nonzero_bins`` when the candidate has any nonzero
    bin, otherwise ``0.0``. Symmetrically,
    ``overlap_fraction_of_baseline`` is ``shared_nonzero_bins /
    baseline_nonzero_bins`` (or ``0.0`` when the baseline has no
    nonzero bin).

    Raises
    ------
    ContributionError
        On non-:class:`ContributionMap` inputs or shape mismatch.
    """
    if not isinstance(baseline, ContributionMap):
        raise ContributionError(
            "baseline must be a ContributionMap; got "
            f"{type(baseline).__name__}"
        )
    if not isinstance(candidate, ContributionMap):
        raise ContributionError(
            "candidate must be a ContributionMap; got "
            f"{type(candidate).__name__}"
        )
    b_w = np.asarray(baseline.weight_map, dtype=float)
    c_w = np.asarray(candidate.weight_map, dtype=float)
    if b_w.shape != c_w.shape:
        raise ContributionError(
            f"baseline.weight_map shape {b_w.shape} does not match "
            f"candidate.weight_map shape {c_w.shape}"
        )

    b_mask = b_w > 0.0
    c_mask = c_w > 0.0
    shared = b_mask & c_mask
    baseline_only = b_mask & ~c_mask
    candidate_only = c_mask & ~b_mask

    baseline_nonzero = int(b_mask.sum())
    candidate_nonzero = int(c_mask.sum())
    shared_count = int(shared.sum())
    baseline_only_count = int(baseline_only.sum())
    candidate_only_count = int(candidate_only.sum())

    overlap_fraction_of_baseline = (
        float(shared_count) / float(baseline_nonzero)
        if baseline_nonzero > 0 else 0.0
    )
    overlap_fraction_of_candidate = (
        float(shared_count) / float(candidate_nonzero)
        if candidate_nonzero > 0 else 0.0
    )
    candidate_new_bin_fraction = (
        float(candidate_only_count) / float(candidate_nonzero)
        if candidate_nonzero > 0 else 0.0
    )

    baseline_total_weight = float(b_w.sum())
    candidate_total_weight = float(c_w.sum())
    shared_candidate_weight = float(c_w[shared].sum())
    candidate_only_weight = float(c_w[candidate_only].sum())

    return ContributionMapOverlapDiagnostic(
        baseline_nonzero_bins=baseline_nonzero,
        candidate_nonzero_bins=candidate_nonzero,
        shared_nonzero_bins=shared_count,
        baseline_only_bins=baseline_only_count,
        candidate_only_bins=candidate_only_count,
        overlap_fraction_of_baseline=overlap_fraction_of_baseline,
        overlap_fraction_of_candidate=overlap_fraction_of_candidate,
        candidate_new_bin_fraction=candidate_new_bin_fraction,
        baseline_total_weight=baseline_total_weight,
        candidate_total_weight=candidate_total_weight,
        shared_candidate_weight=shared_candidate_weight,
        candidate_only_weight=candidate_only_weight,
    )


def _select_worsened_entry(
    comparison_result: LegacyThermalRiskComparisonResult,
):
    """Select the worsened entry from a comparison result.

    Order of preference:

    1. Largest positive ``delta_max_temperature_k``.
    2. If all ``delta_max_temperature_k`` are non-positive,
       largest ``delta_top_percent_max_temperature_rise_k``.
    3. Otherwise, the first entry.
    """
    entries = comparison_result.entries
    if len(entries) == 0:
        raise ThermalError(
            "comparison_result.entries is empty; cannot select a "
            "worsened entry"
        )

    delta_max_t = np.array(
        [
            float(e.comparison.delta_max_temperature_k)
            for e in entries
        ],
        dtype=float,
    )
    if np.any(delta_max_t > 0.0):
        idx = int(np.argmax(delta_max_t))
        return entries[idx]

    delta_top = np.array(
        [
            float(
                e.comparison
                .delta_top_percent_max_temperature_rise_k
            )
            for e in entries
        ],
        dtype=float,
    )
    if np.any(delta_top > 0.0):
        idx = int(np.argmax(delta_top))
        return entries[idx]

    return entries[0]


def build_pattern_induced_hotspot_diagnostic(
    *,
    baseline_setup: LegacyPetWaterTraceSetup,
    candidate_setup: LegacyPetWaterTraceSetup,
    comparison_result: LegacyThermalRiskComparisonResult,
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
) -> PatternInducedHotspotDiagnostic:
    """Backtrack a worsened patterned hotspot to shell-surface bins.

    Actual STL pattern-induced hotspot diagnostic; **not a physical
    PET-bottle validation**. Picks the worst worsened entry from
    ``comparison_result`` (largest positive
    ``delta_max_temperature_k``, falling back to
    ``delta_top_percent_max_temperature_rise_k``, then the first
    entry), runs
    :func:`run_legacy_pet_water_detailed_trace` once for
    ``baseline_setup`` and once for ``candidate_setup`` at that
    entry's ``(angle_degrees, detector_distance)``, builds a
    hotspot contribution map for each via
    :func:`build_legacy_hotspot_contribution_map`, and reports the
    overlap diagnostic.

    Both ``baseline_setup`` and ``candidate_setup`` must be
    :class:`LegacyPetWaterTraceSetup` instances. Patterned setups
    can be wrapped to this shape via the same adapter the demos
    use (re-emit ``shell_mesh`` / ``water_mesh`` / ``step_specs``
    on a fresh :class:`LegacyPetWaterTraceSetup`).

    The returned diagnostic does **not** prove physical causality.
    Candidate-only bins are surrogate ray-backtracking signals
    that may indicate a pattern-induced new hotspot location, but
    they do **not** establish manufacturing-ready behavior or
    fire-prevention safety.

    Raises
    ------
    ContributionError
        On invalid setup types, non-:class:`ContributionMap`
        downstream artifacts, or shape mismatches between the two
        contribution maps.
    ThermalError
        On invalid ``comparison_result`` type or empty entries.
    OpticsError, MetricsError
        From downstream optics / metrics validation.
    """
    if not isinstance(baseline_setup, LegacyPetWaterTraceSetup):
        raise ContributionError(
            "baseline_setup must be a LegacyPetWaterTraceSetup; "
            f"got {type(baseline_setup).__name__}"
        )
    if not isinstance(candidate_setup, LegacyPetWaterTraceSetup):
        raise ContributionError(
            "candidate_setup must be a LegacyPetWaterTraceSetup; "
            f"got {type(candidate_setup).__name__}"
        )
    if not isinstance(
        comparison_result, LegacyThermalRiskComparisonResult,
    ):
        raise ThermalError(
            "comparison_result must be a "
            "LegacyThermalRiskComparisonResult; got "
            f"{type(comparison_result).__name__}"
        )
    tp = float(hotspot_top_percent)
    if not (math.isfinite(tp) and 0.0 < tp <= 100.0):
        raise ContributionError(
            f"hotspot_top_percent must be a finite float in "
            f"(0, 100]; got {hotspot_top_percent}"
        )

    res = tuple(contribution_resolution)
    if len(res) != 2:
        raise ContributionError(
            f"contribution_resolution must be (nv, nu); got {res}"
        )

    entry = _select_worsened_entry(comparison_result)
    angle = float(entry.angle_degrees)
    distance = float(entry.detector_distance)

    common_kwargs = dict(
        angle_degrees=angle,
        detector_distance=distance,
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

    baseline_detailed = run_legacy_pet_water_detailed_trace(
        setup=baseline_setup, **common_kwargs,
    )
    candidate_detailed = run_legacy_pet_water_detailed_trace(
        setup=candidate_setup, **common_kwargs,
    )

    baseline_hotspot = build_legacy_hotspot_contribution_map(
        baseline_detailed,
        top_percent=tp,
        contribution_resolution=(int(res[0]), int(res[1])),
    )
    candidate_hotspot = build_legacy_hotspot_contribution_map(
        candidate_detailed,
        top_percent=tp,
        contribution_resolution=(int(res[0]), int(res[1])),
    )

    overlap = compare_legacy_contribution_maps(
        baseline_hotspot.contribution_map,
        candidate_hotspot.contribution_map,
    )

    return PatternInducedHotspotDiagnostic(
        worsened_angle_degrees=angle,
        worsened_detector_distance=distance,
        baseline_contribution=baseline_hotspot,
        candidate_contribution=candidate_hotspot,
        overlap=overlap,
        baseline_selected_rays=int(
            baseline_hotspot.mapped_initial_ray_count
        ),
        candidate_selected_rays=int(
            candidate_hotspot.mapped_initial_ray_count
        ),
        induced_new_bins=int(overlap.candidate_only_bins),
        induced_new_bin_fraction=float(
            overlap.candidate_new_bin_fraction
        ),
        tradeoff_label=str(entry.comparison.tradeoff_label),
        delta_max_temperature_k=float(
            entry.comparison.delta_max_temperature_k
        ),
        delta_threshold_exceeded_count=int(
            entry.comparison.delta_threshold_exceeded_count
        ),
    )
