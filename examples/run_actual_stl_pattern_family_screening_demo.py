"""Actual STL pattern-family thermal-risk screening demo.

Actual STL pattern-family thermal-risk screening smoke check;
not a physical PET-bottle validation. Loads the supplied STL,
runs an original baseline legacy PET-water optical-to-thermal
risk scan, and then for each member of a small fixed
pattern-candidate family runs the same scan against a
patterned-shell variant of the STL. Each candidate is compared to
the baseline under three :class:`ThermalRiskGuardrailConfig`
scenarios (strict / measurement-tolerance / relaxed). Results are
printed as a diagnostic screening table only; no candidate is
declared safe, best, or fire-prevention validated.

Old H.max raw detector-bin count is **NOT** used as a primary
metric. Canonical reductions are the relative irradiance
surrogate, ``C99``, and the per-pixel thermal-risk metrics.

Scope (also enforced at runtime)
--------------------------------
- This uses an actual STL as a geometry input.
- The STL is target-dimension normalized.
- Patterning uses a heuristic body-region mask.
- Pattern displacement is normal-orientation aware, but not
  manufacturing validation.
- The generated inner mesh is a legacy-style water boundary, not
  measured wall thickness.
- Thermal-risk metrics are surrogate metrics.
- Guardrails are diagnostic filters, not safety certification.
- A candidate can improve one metric and worsen another.
- Delta values are comparison diagnostics only.
- A negative delta is not required.
- Do not claim fire prevention or PET-bottle safety.

Usage::

    python examples/run_actual_stl_pattern_family_screening_demo.py \\
        --mesh data/raw/stl/pet_bottle.stl \\
        --target-height 225.6 --target-diameter 72.1 \\
        --wall-thickness 0.3 --inner-offset-mode auto \\
        --angle-count 18 --angle-step 20 \\
        --detector-count 8 --detector-start 100 --detector-spacing 40 \\
        --sample-count-y 11 --sample-count-z 21 \\
        --detector-resolution 40 \\
        --duration 60 --dt 0.5

The script exits 0 when every invariant holds and 1 otherwise.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from optics_simulation.geometry import load_mesh
from optics_simulation.optics import (
    LegacyOpticalScanResult,
    LegacyPatternedPetWaterSetup,
    LegacyPetWaterTraceSetup,
    create_legacy_experiment_schedule,
    create_legacy_patterned_pet_water_setup,
    create_legacy_pet_water_trace_setup,
    run_legacy_pet_water_angle_distance_scan,
)
from optics_simulation.thermal import (
    LegacyThermalRiskComparisonResult,
    LegacyThermalRiskScanResult,
    ThermalRiskGuardrailConfig,
    compare_legacy_thermal_risk_scans,
    compute_thermal_risk_over_legacy_optical_scan,
)


_DEFAULT_MESH_PATH = "data/raw/stl/pet_bottle.stl"


@dataclass(frozen=True)
class PatternCandidateSpec:
    name: str
    count: int
    sigma_u: float
    sigma_v: float
    max_depth: float
    seed: int
    amplitude: float = 1.0


CANDIDATES: tuple[PatternCandidateSpec, ...] = (
    PatternCandidateSpec(
        name="shallow_broad_01",
        count=20, sigma_u=0.08, sigma_v=0.08,
        max_depth=0.02, seed=1,
    ),
    PatternCandidateSpec(
        name="shallow_broad_02",
        count=20, sigma_u=0.06, sigma_v=0.06,
        max_depth=0.05, seed=2,
    ),
    PatternCandidateSpec(
        name="moderate_medium",
        count=20, sigma_u=0.04, sigma_v=0.04,
        max_depth=0.10, seed=5,
    ),
    PatternCandidateSpec(
        name="moderate_narrow",
        count=20, sigma_u=0.02, sigma_v=0.02,
        max_depth=0.10, seed=6,
    ),
    PatternCandidateSpec(
        name="sparse_deep",
        count=8, sigma_u=0.03, sigma_v=0.03,
        max_depth=0.25, seed=8,
    ),
    PatternCandidateSpec(
        name="current_default",
        count=20, sigma_u=0.03, sigma_v=0.03,
        max_depth=0.05, seed=42,
    ),
)


@dataclass(frozen=True)
class GuardrailScenarioSpec:
    name: str
    config: ThermalRiskGuardrailConfig


GUARDRAIL_SCENARIOS: tuple[GuardrailScenarioSpec, ...] = (
    GuardrailScenarioSpec(
        name="strict",
        config=ThermalRiskGuardrailConfig(
            max_temperature_increase_limit_k=0.0,
            top_percent_rise_increase_limit_k=0.0,
            threshold_count_increase_limit=0,
            threshold_area_increase_limit=0.0,
            tolerance=1e-12,
        ),
    ),
    GuardrailScenarioSpec(
        name="measurement_0p1K",
        config=ThermalRiskGuardrailConfig(
            max_temperature_increase_limit_k=0.1,
            top_percent_rise_increase_limit_k=0.1,
            threshold_count_increase_limit=0,
            threshold_area_increase_limit=0.0,
            tolerance=1e-9,
        ),
    ),
    GuardrailScenarioSpec(
        name="relaxed_1K",
        config=ThermalRiskGuardrailConfig(
            max_temperature_increase_limit_k=1.0,
            top_percent_rise_increase_limit_k=1.0,
            threshold_count_increase_limit=0,
            threshold_area_increase_limit=0.0,
            tolerance=1e-9,
        ),
    ),
)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Actual STL pattern-family thermal-risk screening "
            "smoke check; not a physical PET-bottle validation."
        ),
    )
    parser.add_argument(
        "--mesh", type=str, default=_DEFAULT_MESH_PATH,
    )
    parser.add_argument("--target-height", type=float, default=225.6)
    parser.add_argument("--target-diameter", type=float, default=72.1)
    parser.add_argument("--wall-thickness", type=float, default=0.3)
    parser.add_argument(
        "--inner-offset-mode",
        choices=("auto", "minus_normals", "plus_normals"),
        default="auto",
    )
    parser.add_argument("--angle-count", type=int, default=18)
    parser.add_argument("--angle-step", type=float, default=20.0)
    parser.add_argument("--detector-count", type=int, default=8)
    parser.add_argument("--detector-start", type=float, default=100.0)
    parser.add_argument(
        "--detector-spacing", type=float, default=40.0,
    )
    parser.add_argument("--source-width", type=float, default=100.0)
    parser.add_argument("--source-height", type=float, default=250.0)
    parser.add_argument("--source-radius", type=float, default=200.0)
    parser.add_argument("--sample-count-y", type=int, default=11)
    parser.add_argument("--sample-count-z", type=int, default=21)
    parser.add_argument("--detector-size", type=float, default=400.0)
    parser.add_argument(
        "--detector-resolution", type=int, default=40,
    )
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument(
        "--areal-heat-capacity", type=float, default=1200.0,
    )
    parser.add_argument("--absorptivity", type=float, default=0.8)
    parser.add_argument("--h-conv", type=float, default=10.0)
    parser.add_argument("--emissivity", type=float, default=0.9)
    parser.add_argument(
        "--nominal-incident-irradiance",
        type=float, default=1000.0,
    )
    parser.add_argument(
        "--threshold-temp", type=float, default=373.15,
    )
    return parser


def _print_static_header() -> None:
    print("Actual STL pattern-family screening demo")
    print(
        "Actual STL pattern-family thermal-risk screening smoke "
        "check; not a physical PET-bottle validation."
    )
    print("Note: this uses an actual STL as a geometry input.")
    print("Note: the STL is target-dimension normalized.")
    print("Note: patterning uses a heuristic body-region mask.")
    print(
        "Note: pattern displacement is normal-orientation aware, "
        "but not manufacturing validation."
    )
    print(
        "Note: the generated inner mesh is a legacy-style water "
        "boundary, not measured wall thickness."
    )
    print("Note: thermal-risk metrics are surrogate metrics.")
    print(
        "Note: guardrails are diagnostic filters, not safety "
        "certification."
    )
    print(
        "Note: a candidate can improve one metric and worsen "
        "another."
    )
    print(
        "Note: delta values are comparison diagnostics only; a "
        "negative delta is not required."
    )
    print(
        "Note: do not claim fire prevention or PET-bottle safety."
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


def _print_screening_table_row(
    *,
    candidate: PatternCandidateSpec,
    setup: LegacyPatternedPetWaterSetup,
    optical_scan: LegacyOpticalScanResult,
    comparisons: dict[str, LegacyThermalRiskComparisonResult],
) -> None:
    body_mask = setup.body_mask_report
    moved = setup.patterned_mesh_result
    print(f"Candidate: {candidate.name}")
    print(
        f"  Body mask selected vertices: "
        f"{int(body_mask.selected_count)} / "
        f"{int(body_mask.vertex_count)}"
    )
    print(
        f"  Pattern moved vertices: "
        f"{int(moved.moved_vertex_count)}"
    )
    print(
        f"  Candidate optical max C99: "
        f"{0.0 if optical_scan.max_c99 is None else float(optical_scan.max_c99):.6f}"
    )
    for scenario_name, comparison in comparisons.items():
        # Use first-occurrence aggregate label for printing.
        if int(comparison.entry_count) > 0:
            label_counts = {
                "improved": comparison.improved_count,
                "worsened": comparison.worsened_count,
                "mixed": comparison.mixed_count,
                "unchanged": comparison.unchanged_count,
            }
            dominant_label = max(
                label_counts.items(), key=lambda kv: kv[1]
            )[0]
        else:
            dominant_label = "unchanged"
        worst_dt = comparison.worst_delta_max_temperature_k
        best_dthresh = comparison.best_delta_threshold_count
        print(f"  Scenario: {scenario_name}")
        print(f"    Tradeoff label: {dominant_label}")
        print(
            f"    Passes guardrails: "
            f"{int(comparison.pass_count)} / "
            f"{int(comparison.entry_count)}"
        )
        if worst_dt is None:
            print("    Delta max temperature: None")
        else:
            print(
                f"    Delta max temperature: "
                f"{float(worst_dt):+.6f} (worst)"
            )
        if best_dthresh is None:
            print("    Delta threshold count: None")
        else:
            print(
                f"    Delta threshold count: "
                f"{int(best_dthresh):+d} (best)"
            )


def _all_entries_pass(
    comparison: LegacyThermalRiskComparisonResult,
) -> bool:
    return (
        int(comparison.entry_count) > 0
        and int(comparison.pass_count) == int(comparison.entry_count)
    )


def _has_mixed_entries(
    comparison: LegacyThermalRiskComparisonResult,
) -> bool:
    return int(comparison.mixed_count) > 0


def _check_invariants(
    *,
    candidate_count: int,
    baseline_thermal: LegacyThermalRiskScanResult,
    candidate_results: list[
        tuple[
            PatternCandidateSpec,
            LegacyPatternedPetWaterSetup,
            LegacyOpticalScanResult,
            LegacyThermalRiskScanResult,
            dict[str, LegacyThermalRiskComparisonResult],
        ]
    ],
) -> bool:
    checks: list[bool] = []
    checks.append(int(candidate_count) == len(candidate_results))
    checks.append(int(baseline_thermal.entry_count) > 0)
    for cand_spec, setup, optical, thermal, comps in candidate_results:
        checks.append(
            int(thermal.entry_count) == int(baseline_thermal.entry_count)
        )
        checks.append(
            int(setup.body_mask_report.selected_count) > 0
        )
        checks.append(
            int(setup.patterned_mesh_result.moved_vertex_count) > 0
        )
        for scenario_name, comparison in comps.items():
            checks.append(
                int(comparison.entry_count)
                == int(baseline_thermal.entry_count)
            )
            checks.append(
                int(comparison.pass_count)
                + int(comparison.fail_count)
                == int(comparison.entry_count)
            )
    return all(checks)


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    _print_static_header()

    print(f"Mesh path: {Path(args.mesh)}")
    print(f"Candidate count: {len(CANDIDATES)}")
    print(
        f"Guardrail scenarios: "
        f"{[s.name for s in GUARDRAIL_SCENARIOS]}"
    )

    mesh = load_mesh(Path(args.mesh))
    baseline_setup = create_legacy_pet_water_trace_setup(
        mesh,
        target_height=float(args.target_height),
        target_diameter=float(args.target_diameter),
        wall_thickness=float(args.wall_thickness),
        inner_offset_mode=str(args.inner_offset_mode),
    )

    schedule = create_legacy_experiment_schedule(
        angle_count=int(args.angle_count),
        angle_step_degrees=float(args.angle_step),
        detector_count=int(args.detector_count),
        detector_start=float(args.detector_start),
        detector_spacing=float(args.detector_spacing),
        detector_size=float(args.detector_size),
        source_width=float(args.source_width),
        source_height=float(args.source_height),
        source_radius=float(args.source_radius),
        sample_count_y=int(args.sample_count_y),
        sample_count_z=int(args.sample_count_z),
    )

    scan_kwargs = dict(
        angles_degrees=schedule.angles_degrees,
        detector_distances=schedule.detector_distances,
        source_width=schedule.source_width,
        source_height=schedule.source_height,
        sample_count_y=schedule.sample_count_y,
        sample_count_z=schedule.sample_count_z,
        detector_size=schedule.detector_size,
        detector_resolution=(
            int(args.detector_resolution),
            int(args.detector_resolution),
        ),
        source_radius=schedule.source_radius,
        epsilon=0.1,
        store_irradiance_surrogate=True,
    )

    baseline_optical = run_legacy_pet_water_angle_distance_scan(
        setup=baseline_setup, **scan_kwargs,
    )
    thermal_kwargs = dict(
        nominal_incident_irradiance_w_m2=float(
            args.nominal_incident_irradiance
        ),
        duration_s=float(args.duration),
        dt_s=float(args.dt),
        areal_heat_capacity_j_m2k=float(args.areal_heat_capacity),
        absorptivity=float(args.absorptivity),
        h_conv_w_m2k=float(args.h_conv),
        emissivity=float(args.emissivity),
        threshold_temp_k=float(args.threshold_temp),
    )
    baseline_thermal = compute_thermal_risk_over_legacy_optical_scan(
        baseline_optical, **thermal_kwargs,
    )
    print(
        f"Baseline thermal entries: "
        f"{int(baseline_thermal.entry_count)}"
    )

    candidate_results: list[
        tuple[
            PatternCandidateSpec,
            LegacyPatternedPetWaterSetup,
            LegacyOpticalScanResult,
            LegacyThermalRiskScanResult,
            dict[str, LegacyThermalRiskComparisonResult],
        ]
    ] = []

    for cand_spec in CANDIDATES:
        patterned_setup = create_legacy_patterned_pet_water_setup(
            mesh,
            target_height=float(args.target_height),
            target_diameter=float(args.target_diameter),
            wall_thickness=float(args.wall_thickness),
            inner_offset_mode=str(args.inner_offset_mode),
            pattern_count=int(cand_spec.count),
            pattern_sigma_u=float(cand_spec.sigma_u),
            pattern_sigma_v=float(cand_spec.sigma_v),
            pattern_max_depth=float(cand_spec.max_depth),
            pattern_seed=int(cand_spec.seed),
        )
        patterned_legacy = _patterned_setup_to_legacy(patterned_setup)
        cand_optical = run_legacy_pet_water_angle_distance_scan(
            setup=patterned_legacy, **scan_kwargs,
        )
        cand_thermal = compute_thermal_risk_over_legacy_optical_scan(
            cand_optical, **thermal_kwargs,
        )
        comparisons: dict[str, LegacyThermalRiskComparisonResult] = {}
        for scenario in GUARDRAIL_SCENARIOS:
            comparisons[scenario.name] = (
                compare_legacy_thermal_risk_scans(
                    baseline_thermal, cand_thermal,
                    guardrails=scenario.config,
                )
            )
        candidate_results.append(
            (
                cand_spec,
                patterned_setup,
                cand_optical,
                cand_thermal,
                comparisons,
            )
        )
        _print_screening_table_row(
            candidate=cand_spec,
            setup=patterned_setup,
            optical_scan=cand_optical,
            comparisons=comparisons,
        )

    print("Screening summary")
    strict_passes = sum(
        1 for _, _, _, _, comps in candidate_results
        if _all_entries_pass(comps["strict"])
    )
    measurement_passes = sum(
        1 for _, _, _, _, comps in candidate_results
        if _all_entries_pass(comps["measurement_0p1K"])
    )
    relaxed_passes = sum(
        1 for _, _, _, _, comps in candidate_results
        if _all_entries_pass(comps["relaxed_1K"])
    )
    mixed_candidates = sum(
        1 for _, _, _, _, comps in candidate_results
        if _has_mixed_entries(comps["strict"])
    )
    print(f"Strict passes: {strict_passes} / {len(candidate_results)}")
    print(
        f"Measurement tolerance passes: "
        f"{measurement_passes} / {len(candidate_results)}"
    )
    print(
        f"Relaxed passes: {relaxed_passes} / {len(candidate_results)}"
    )
    print(
        f"Mixed candidates: "
        f"{mixed_candidates} / {len(candidate_results)}"
    )
    print(
        "Note: pass count is a diagnostic filter, not safety "
        "certification."
    )

    ok = _check_invariants(
        candidate_count=len(CANDIDATES),
        baseline_thermal=baseline_thermal,
        candidate_results=candidate_results,
    )
    print(f"Invariants: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
