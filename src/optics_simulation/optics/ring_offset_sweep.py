"""Ring-offset risk-guided pattern parameter sweep + Pareto ranking.

Actual STL ring-offset risk-guided pattern parameter sweep
diagnostic; **not a physical PET-bottle validation**. Earlier
single-pair (centered vs one ring-offset) diagnostics showed
ring-offset patterns can shift hotspot contribution overlap
(``candidate_only_bins`` rises from ``0`` to a small positive
fraction) while sometimes increasing ``worst_delta_max_temperature_k``
and sometimes increasing the guardrail pass count. Whether any
specific ``(inner_radius_px, outer_radius_px)`` combination
**reduces** the worst-case hotspot or merely redistributes it
cannot be answered by a single comparison.

This module sweeps a small list of ring-offset candidates and
ranks them by three diagnostic axes:

- ``worst_delta_max_temperature_k`` (lower better)
- ``best_delta_threshold_count`` (lower better)
- ``pass_ratio = pass_count / entry_count`` (higher better)

Two ranking helpers are exposed:

- :func:`rank_ring_offset_sweep_pareto_indices` returns the
  indices of candidates that are not strictly dominated by any
  other candidate on the three-axis tuple above.
- :func:`rank_ring_offset_sweep_by_composite_score` returns the
  indices sorted ascending by a min-max-normalized weighted sum
  of the same three axes (lower score better).

Limitations
-----------
- Ranking is a **diagnostic** ordering, **not** safety
  certification. A "best" candidate is the candidate that
  minimizes the chosen diagnostic objective in this synthetic
  surrogate sweep, **not** a manufacturing-ready pattern.
- ``new_bin_fraction`` is reported per entry but **excluded**
  from the Pareto / composite ranking, because a low value can
  mean either "the pattern correctly stayed in the existing
  hotspot region" or "the pattern failed to move the hotspot at
  all". Treat it as a *qualitative* signal.
- A higher guardrail pass count does **not** prove safety.
- A ring-offset pattern can still be mixed.
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
    compare_legacy_thermal_risk_scans,
)
from optics_simulation.thermal.legacy_scan_coupling import (
    LegacyThermalRiskScanResult,
    compute_thermal_risk_over_legacy_optical_scan,
)
from optics_simulation.thermal.risk_comparison import (
    ThermalRiskGuardrailConfig,
)


@dataclass(frozen=True)
class RingOffsetSweepCandidateSpec:
    name: str
    inner_radius_px: int
    outer_radius_px: int
    risk_epsilon: float = 0.01


@dataclass(frozen=True)
class RingOffsetSweepEntry:
    candidate: RingOffsetSweepCandidateSpec
    source_risk_active_count: int
    ring_risk_active_count: int
    moved_vertex_count: int
    max_displacement: float
    delta_max_temperature_k: float
    delta_threshold_count: int
    candidate_only_bins: int
    new_bin_fraction: float
    guardrail_pass_count: int
    guardrail_fail_count: int
    total_condition_count: int
    pass_ratio: float
    tradeoff_label: str


@dataclass(frozen=True)
class RingOffsetSweepResult:
    entries: tuple[RingOffsetSweepEntry, ...]
    candidate_count: int
    pareto_candidate_names: tuple[str, ...]
    best_by_max_temperature_delta: str | None
    best_delta_max_temperature_k: float | None
    best_by_threshold_count_delta: str | None
    best_delta_threshold_count: int | None
    best_by_pass_ratio: str | None
    best_pass_ratio: float | None
    best_by_composite_score: str | None
    best_composite_score: float | None
    result_type: str = "actual_stl_ring_offset_sweep"


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


def _dominant_label(comparison) -> str:
    counts = {
        "improved": int(comparison.improved_count),
        "worsened": int(comparison.worsened_count),
        "mixed": int(comparison.mixed_count),
        "unchanged": int(comparison.unchanged_count),
    }
    return max(counts.items(), key=lambda kv: kv[1])[0]


def run_actual_stl_ring_offset_sweep(
    *,
    original_setup: LegacyPetWaterTraceSetup,
    baseline_thermal_scan: LegacyThermalRiskScanResult,
    aggregate_risk_map: RiskMap,
    body_include_mask: np.ndarray,
    candidates: Sequence[RingOffsetSweepCandidateSpec],
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
    composite_weight_max_temperature: float = 1.0,
    composite_weight_threshold_count: float = 1.0,
    composite_weight_pass_ratio: float = 1.0,
) -> RingOffsetSweepResult:
    """Sweep ring-offset risk-guided pattern candidates and rank diagnostics.

    Actual STL ring-offset risk-guided pattern parameter sweep
    diagnostic; **not a physical PET-bottle validation**. For each
    :class:`RingOffsetSweepCandidateSpec` in ``candidates``:

    1. Transform ``aggregate_risk_map`` via
       :func:`create_ring_offset_risk_map` using the candidate's
       ``inner_radius_px`` / ``outer_radius_px``.
    2. Build a risk-guided patterned PET-water setup against the
       transformed ring risk map and the supplied
       ``body_include_mask``.
    3. Run the legacy four-step PET-water angle / detector-distance
       scan with ``store_irradiance_surrogate=True``.
    4. Convert to thermal-risk metrics and compare against
       ``baseline_thermal_scan``.
    5. Run :func:`build_pattern_induced_hotspot_diagnostic` on the
       worst worsened entry to compute the contribution-overlap
       diagnostic (``candidate_only_bins`` /
       ``new_bin_fraction``).

    The returned result includes the per-candidate entries and
    four diagnostic "best by ..." pointers plus the indices of
    Pareto-non-dominated candidates on
    ``(worst_delta_max_temperature_k, best_delta_threshold_count,
    -pass_ratio)``. The ``new_bin_fraction`` is **not** part of
    the dominance test — see module docstring.

    Pattern shape parameters (``pattern_count``, sigmas,
    ``max_depth``, ``seed``) are held **fixed** across all
    candidates so the sweep isolates the ring-transform's effect
    on hotspot redistribution; pattern-shape sweeping belongs in
    a separate study that can re-use this module's result type.

    Raises
    ------
    OpticsError
        On invalid setup type, invalid candidate spec, empty
        candidate list, non-positive sample counts, or invalid
        detector resolution.
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

    if (
        not math.isfinite(float(composite_weight_max_temperature))
        or float(composite_weight_max_temperature) < 0.0
        or not math.isfinite(float(composite_weight_threshold_count))
        or float(composite_weight_threshold_count) < 0.0
        or not math.isfinite(float(composite_weight_pass_ratio))
        or float(composite_weight_pass_ratio) < 0.0
    ):
        raise OpticsError(
            "composite weights must be finite floats >= 0"
        )

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

        moved = patterned_setup.patterned_mesh_result
        total_count = int(comparison.entry_count)
        pass_count = int(comparison.pass_count)
        fail_count = int(comparison.fail_count)
        pass_ratio = (
            float(pass_count) / float(total_count)
            if total_count > 0 else 0.0
        )
        worst_delta_max_t = (
            float(comparison.worst_delta_max_temperature_k)
            if comparison.worst_delta_max_temperature_k is not None
            else float("inf")
        )
        best_delta_count = (
            int(comparison.best_delta_threshold_count)
            if comparison.best_delta_threshold_count is not None
            else 0
        )

        sweep_entries.append(
            RingOffsetSweepEntry(
                candidate=c,
                source_risk_active_count=int(
                    ring_result.source_active_count,
                ),
                ring_risk_active_count=int(
                    ring_result.transformed_active_count,
                ),
                moved_vertex_count=int(moved.moved_vertex_count),
                max_displacement=float(moved.max_displacement),
                delta_max_temperature_k=float(worst_delta_max_t),
                delta_threshold_count=int(best_delta_count),
                candidate_only_bins=int(
                    diagnostic.overlap.candidate_only_bins,
                ),
                new_bin_fraction=float(
                    diagnostic.induced_new_bin_fraction,
                ),
                guardrail_pass_count=pass_count,
                guardrail_fail_count=fail_count,
                total_condition_count=total_count,
                pass_ratio=float(pass_ratio),
                tradeoff_label=str(_dominant_label(comparison)),
            )
        )

    pareto_indices = rank_ring_offset_sweep_pareto_indices(
        sweep_entries,
    )
    pareto_names = tuple(
        sweep_entries[i].candidate.name for i in pareto_indices
    )

    composite_order = rank_ring_offset_sweep_by_composite_score(
        sweep_entries,
        weight_max_temperature=float(
            composite_weight_max_temperature,
        ),
        weight_threshold_count=float(
            composite_weight_threshold_count,
        ),
        weight_pass_ratio=float(composite_weight_pass_ratio),
    )

    deltas_t = np.array(
        [e.delta_max_temperature_k for e in sweep_entries],
        dtype=float,
    )
    deltas_count = np.array(
        [e.delta_threshold_count for e in sweep_entries],
        dtype=np.int64,
    )
    pass_ratios = np.array(
        [e.pass_ratio for e in sweep_entries], dtype=float,
    )
    idx_min_t = int(np.argmin(deltas_t))
    idx_min_count = int(np.argmin(deltas_count))
    idx_max_pass = int(np.argmax(pass_ratios))

    composite_scores = _composite_scores(
        sweep_entries,
        weight_max_temperature=float(
            composite_weight_max_temperature,
        ),
        weight_threshold_count=float(
            composite_weight_threshold_count,
        ),
        weight_pass_ratio=float(composite_weight_pass_ratio),
    )
    if composite_order:
        idx_min_score = int(composite_order[0])
        best_score = float(composite_scores[idx_min_score])
    else:
        idx_min_score = idx_min_t
        best_score = float(composite_scores[idx_min_score])

    return RingOffsetSweepResult(
        entries=tuple(sweep_entries),
        candidate_count=int(len(sweep_entries)),
        pareto_candidate_names=pareto_names,
        best_by_max_temperature_delta=str(
            sweep_entries[idx_min_t].candidate.name,
        ),
        best_delta_max_temperature_k=float(deltas_t[idx_min_t]),
        best_by_threshold_count_delta=str(
            sweep_entries[idx_min_count].candidate.name,
        ),
        best_delta_threshold_count=int(deltas_count[idx_min_count]),
        best_by_pass_ratio=str(
            sweep_entries[idx_max_pass].candidate.name,
        ),
        best_pass_ratio=float(pass_ratios[idx_max_pass]),
        best_by_composite_score=str(
            sweep_entries[idx_min_score].candidate.name,
        ),
        best_composite_score=float(best_score),
    )


def rank_ring_offset_sweep_pareto_indices(
    entries: Sequence[RingOffsetSweepEntry],
) -> tuple[int, ...]:
    """Return indices of candidates not strictly dominated on three axes.

    Diagnostic Pareto filter on
    ``(delta_max_temperature_k, delta_threshold_count, -pass_ratio)``;
    all three are lower-is-better in that coordinate. Candidate
    ``A`` strictly dominates ``B`` iff every coordinate of ``A``
    is ``<=`` the corresponding coordinate of ``B`` and at least
    one is strictly less. Returned indices preserve the input
    order.

    ``new_bin_fraction`` is **excluded** from the dominance test
    because the same numerical value can mean either "the pattern
    correctly stayed in the existing hotspot region" (good) or
    "the pattern failed to move the hotspot" (bad). It is
    reported in :class:`RingOffsetSweepEntry` for inspection but
    not used as a ranking axis.
    """
    n = len(entries)
    if n == 0:
        return ()
    coords = np.empty((n, 3), dtype=float)
    for i, e in enumerate(entries):
        coords[i, 0] = float(e.delta_max_temperature_k)
        coords[i, 1] = float(e.delta_threshold_count)
        coords[i, 2] = -float(e.pass_ratio)

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
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo <= 0.0:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def _composite_scores(
    entries: Sequence[RingOffsetSweepEntry],
    *,
    weight_max_temperature: float,
    weight_threshold_count: float,
    weight_pass_ratio: float,
) -> np.ndarray:
    n = len(entries)
    if n == 0:
        return np.empty(0, dtype=float)
    deltas_t = np.array(
        [e.delta_max_temperature_k for e in entries], dtype=float,
    )
    deltas_count = np.array(
        [float(e.delta_threshold_count) for e in entries],
        dtype=float,
    )
    pass_ratios = np.array(
        [e.pass_ratio for e in entries], dtype=float,
    )
    n_t = _normalize_lower_better(deltas_t)
    n_c = _normalize_lower_better(deltas_count)
    n_p = _normalize_lower_better(-pass_ratios)
    return (
        float(weight_max_temperature) * n_t
        + float(weight_threshold_count) * n_c
        + float(weight_pass_ratio) * n_p
    )


def rank_ring_offset_sweep_by_composite_score(
    entries: Sequence[RingOffsetSweepEntry],
    *,
    weight_max_temperature: float = 1.0,
    weight_threshold_count: float = 1.0,
    weight_pass_ratio: float = 1.0,
) -> tuple[int, ...]:
    """Rank candidates ascending by a min-max-normalized composite score.

    Diagnostic ranking only — **not** safety certification. The
    composite is a min-max-normalized weighted sum on the same
    three axes as :func:`rank_ring_offset_sweep_pareto_indices`,
    where each axis is rescaled to ``[0, 1]`` (with all-equal
    columns mapped to all-zero so they do not affect ordering).
    Indices are returned in ascending score order; ties preserve
    the input order via :func:`numpy.argsort`'s stable kind.
    """
    n = len(entries)
    if n == 0:
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

    scores = _composite_scores(
        entries,
        weight_max_temperature=weight_max_temperature,
        weight_threshold_count=weight_threshold_count,
        weight_pass_ratio=weight_pass_ratio,
    )
    order = np.argsort(scores, kind="stable")
    return tuple(int(i) for i in order.tolist())
