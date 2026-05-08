"""Risk-guided patterned PET-water parameter sweep for actual STL.

Actual STL risk-guided pattern parameter sweep smoke check;
**not a physical PET-bottle validation**. Reuses a caller-supplied
multi-condition aggregate :class:`RiskMap` and a body-region
include mask, then for each entry of a small list of
:class:`RiskGuidedPatternCandidateSpec` instances:

1. Build a risk-guided patterned PET-water setup via
   :func:`create_risk_guided_legacy_patterned_pet_water_setup`
   using the candidate's pattern parameters.
2. Run the legacy four-step PET-water scan over the same
   ``angles_degrees`` x ``detector_distances`` schedule that was
   used for the baseline scan, with
   ``store_irradiance_surrogate=True``.
3. Convert the candidate's optical scan to thermal-risk metrics
   via :func:`compute_thermal_risk_over_legacy_optical_scan`.
4. Compare the candidate scan to the supplied
   ``baseline_thermal_scan`` via
   :func:`compare_legacy_thermal_risk_scans`.

The sweep returns a :class:`RiskGuidedPatternSweepResult` whose
``entries`` preserve the input candidate order. Aggregate
"best by ..." pointers are exposed only as the candidate name
that minimizes the corresponding delta diagnostic in this
synthetic sweep; they are **not** safety recommendations.

Limitations
-----------
- The sweep is **diagnostic**, not optimization.
- A candidate can reduce one metric while increasing another.
- ``passes_guardrails`` is **not** safety certification.
- A negative delta is **not required**.
- This is **not** fire-prevention validation, **not** PET-bottle
  safety, and **not** manufacturing-ready.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from optics_simulation.contribution.risk_map import RiskMap
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


@dataclass(frozen=True)
class RiskGuidedPatternCandidateSpec:
    name: str
    pattern_count: int
    sigma_u: float
    sigma_v: float
    max_depth: float
    seed: int
    amplitude: float = 1.0


@dataclass(frozen=True)
class RiskGuidedPatternSweepEntry:
    candidate: RiskGuidedPatternCandidateSpec
    moved_vertex_count: int
    max_displacement: float
    comparison: LegacyThermalRiskComparisonResult
    worst_delta_max_temperature_k: float | None
    best_delta_max_temperature_k: float | None
    best_delta_threshold_count: int | None
    pass_count: int
    fail_count: int
    mixed_count: int
    improved_count: int
    worsened_count: int
    unchanged_count: int


@dataclass(frozen=True)
class RiskGuidedPatternSweepResult:
    entries: tuple[RiskGuidedPatternSweepEntry, ...]
    candidate_count: int
    pass_candidate_count: int
    mixed_candidate_count: int
    best_candidate_by_max_temperature_delta: str | None
    best_delta_max_temperature_k: float | None
    best_candidate_by_threshold_count_delta: str | None
    best_delta_threshold_count: int | None
    result_type: str = "actual_stl_risk_guided_pattern_parameter_sweep"


def _check_positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise OpticsError(
            f"{name} must be a positive int; got {value!r}"
        )
    if value <= 0:
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
    candidate: RiskGuidedPatternCandidateSpec, *, index: int,
) -> None:
    if not isinstance(candidate, RiskGuidedPatternCandidateSpec):
        raise OpticsError(
            f"candidates[{index}] must be a "
            f"RiskGuidedPatternCandidateSpec; got "
            f"{type(candidate).__name__}"
        )
    if not isinstance(candidate.name, str) or not candidate.name:
        raise OpticsError(
            f"candidates[{index}].name must be a non-empty str; "
            f"got {candidate.name!r}"
        )
    if (
        isinstance(candidate.pattern_count, bool)
        or not isinstance(candidate.pattern_count, int)
        or int(candidate.pattern_count) <= 0
    ):
        raise OpticsError(
            f"candidates[{index}].pattern_count must be a positive "
            f"int; got {candidate.pattern_count!r}"
        )
    for fname, fvalue in (
        ("sigma_u", candidate.sigma_u),
        ("sigma_v", candidate.sigma_v),
        ("max_depth", candidate.max_depth),
        ("amplitude", candidate.amplitude),
    ):
        fv = float(fvalue)
        if not math.isfinite(fv) or fv <= 0.0:
            raise OpticsError(
                f"candidates[{index}].{fname} must be a finite "
                f"float > 0; got {fvalue}"
            )
    if (
        isinstance(candidate.seed, bool)
        or not isinstance(candidate.seed, int)
    ):
        raise OpticsError(
            f"candidates[{index}].seed must be an int; "
            f"got {candidate.seed!r}"
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


def run_actual_stl_risk_guided_pattern_parameter_sweep(
    *,
    original_setup: LegacyPetWaterTraceSetup,
    baseline_thermal_scan: LegacyThermalRiskScanResult,
    aggregate_risk_map: RiskMap,
    body_include_mask: np.ndarray,
    candidates: Sequence[RiskGuidedPatternCandidateSpec],
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
    nominal_incident_irradiance_w_m2: float = 1000.0,
    areal_heat_capacity_j_m2k: float = 1200.0,
    absorptivity: float = 0.8,
    h_conv_w_m2k: float = 10.0,
    emissivity: float = 0.9,
    ambient_temp_k: float = 293.15,
    threshold_temp_k: float = 373.15,
    wall_thickness: float = 0.3,
    inner_offset_mode: str = "auto",
    guardrails: ThermalRiskGuardrailConfig | None = None,
) -> RiskGuidedPatternSweepResult:
    """Sweep risk-guided pattern parameters and compare to a thermal baseline.

    Actual STL risk-guided pattern parameter sweep smoke check;
    **not a physical PET-bottle validation**. The sweep is
    diagnostic, not optimization. A candidate can reduce one
    metric while increasing another. ``passes_guardrails`` is not
    safety certification. Aggregate "best by ..." pointers are
    candidates that minimize the corresponding delta diagnostic in
    this synthetic sweep; they are **not** safety recommendations.

    Raises
    ------
    OpticsError
        On invalid setup type, invalid candidate spec, empty
        candidate list, non-positive durations, non-positive
        sample counts, or invalid detector resolution.
    PatternError, GeometryError, ThermalError
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
            "RiskGuidedPatternCandidateSpec"
        ) from exc
    if len(cand_list) == 0:
        raise OpticsError(
            "candidates must contain at least one "
            "RiskGuidedPatternCandidateSpec"
        )
    seen_names: set[str] = set()
    for i, c in enumerate(cand_list):
        _validate_candidate(c, index=i)
        if c.name in seen_names:
            raise OpticsError(
                f"candidates[{i}].name {c.name!r} is duplicated; "
                "candidate names must be unique"
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

    sweep_entries: list[RiskGuidedPatternSweepEntry] = []
    pass_candidate_count = 0
    mixed_candidate_count = 0

    for c in cand_list:
        patterned_setup = (
            create_risk_guided_legacy_patterned_pet_water_setup(
                original_setup=original_setup,
                risk_map=aggregate_risk_map,
                body_include_mask=body_include_mask,
                pattern_count=int(c.pattern_count),
                pattern_amplitude=float(c.amplitude),
                pattern_sigma_u=float(c.sigma_u),
                pattern_sigma_v=float(c.sigma_v),
                pattern_max_depth=float(c.max_depth),
                pattern_seed=int(c.seed),
                wall_thickness=float(wall_thickness),
                inner_offset_mode=str(inner_offset_mode),
            )
        )
        patterned_legacy = _patterned_setup_to_legacy(patterned_setup)
        cand_optical = run_legacy_pet_water_angle_distance_scan(
            setup=patterned_legacy, **scan_kwargs,
        )
        cand_thermal = compute_thermal_risk_over_legacy_optical_scan(
            cand_optical, **thermal_kwargs,
        )
        comparison = compare_legacy_thermal_risk_scans(
            baseline_thermal_scan, cand_thermal,
            guardrails=guardrails,
        )

        moved = patterned_setup.patterned_mesh_result
        entry = RiskGuidedPatternSweepEntry(
            candidate=c,
            moved_vertex_count=int(moved.moved_vertex_count),
            max_displacement=float(moved.max_displacement),
            comparison=comparison,
            worst_delta_max_temperature_k=(
                comparison.worst_delta_max_temperature_k
            ),
            best_delta_max_temperature_k=(
                comparison.best_delta_max_temperature_k
            ),
            best_delta_threshold_count=(
                comparison.best_delta_threshold_count
            ),
            pass_count=int(comparison.pass_count),
            fail_count=int(comparison.fail_count),
            mixed_count=int(comparison.mixed_count),
            improved_count=int(comparison.improved_count),
            worsened_count=int(comparison.worsened_count),
            unchanged_count=int(comparison.unchanged_count),
        )
        sweep_entries.append(entry)
        if int(comparison.fail_count) == 0:
            pass_candidate_count += 1
        if int(comparison.mixed_count) > 0:
            mixed_candidate_count += 1

    best_max_idx: int | None = None
    best_max_value: float | None = None
    for i, e in enumerate(sweep_entries):
        v = e.worst_delta_max_temperature_k
        if v is None:
            continue
        if best_max_value is None or float(v) < float(best_max_value):
            best_max_value = float(v)
            best_max_idx = i

    best_count_idx: int | None = None
    best_count_value: int | None = None
    for i, e in enumerate(sweep_entries):
        v = e.best_delta_threshold_count
        if v is None:
            continue
        if best_count_value is None or int(v) < int(best_count_value):
            best_count_value = int(v)
            best_count_idx = i

    return RiskGuidedPatternSweepResult(
        entries=tuple(sweep_entries),
        candidate_count=int(len(sweep_entries)),
        pass_candidate_count=int(pass_candidate_count),
        mixed_candidate_count=int(mixed_candidate_count),
        best_candidate_by_max_temperature_delta=(
            None if best_max_idx is None
            else str(sweep_entries[best_max_idx].candidate.name)
        ),
        best_delta_max_temperature_k=best_max_value,
        best_candidate_by_threshold_count_delta=(
            None if best_count_idx is None
            else str(sweep_entries[best_count_idx].candidate.name)
        ),
        best_delta_threshold_count=best_count_value,
    )
