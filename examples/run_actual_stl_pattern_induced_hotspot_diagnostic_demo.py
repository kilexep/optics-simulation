"""Actual STL pattern-induced hotspot diagnostic demo.

Actual STL pattern-induced hotspot diagnostic; not a physical
PET-bottle validation. Loads the supplied STL, runs the
self-consistent subdivided baseline / aggregate-risk-map / single
risk-guided candidate flow at one ``--candidate-name``, then
compares the candidate to the baseline thermal-risk scan, picks
the worst worsened ``(angle_degrees, detector_distance)`` entry,
and runs detector-hotspot ray backtracking on **both** the
baseline and the candidate at that entry. The two contribution
maps are then compared bin-by-bin so the user can see whether
the patterned hotspot rays land on the **same** shell-surface
bins as the baseline (overlap) or on **new** bins
(candidate-only).

The diagnostic answers a narrow question:

    "When the patterned shell is worse than the baseline at this
    entry, are the patterned hotspot rays backtracked to the same
    surface bins as the baseline, or to new bins?"

Old H.max raw detector-bin count is **NOT** used as a primary
metric. Canonical reductions remain the relative irradiance
surrogate, ``C99``, and the per-pixel thermal-risk metrics; this
demo's contribution-map overlap is a separate diagnostic and is
not promoted to a primary metric.

Scope (also enforced at runtime)
--------------------------------
- This diagnostic backtracks worsened patterned hotspots to
  shell-surface coordinates.
- It compares baseline contribution bins and pattern-induced
  contribution bins.
- New induced hotspot bins are diagnostic only.
- The contribution map is not a measured physical risk field.
- A pattern can reduce one metric while inducing new hotspots
  elsewhere.
- This is not optimization.
- This does not prove fire prevention or PET-bottle safety.

Usage::

    python examples/run_actual_stl_pattern_induced_hotspot_diagnostic_demo.py \\
        --mesh data/raw/stl/pet_bottle.stl \\
        --target-height 225.6 --target-diameter 72.1 \\
        --wall-thickness 0.3 --inner-offset-mode auto \\
        --subdivision-iterations 1 \\
        --candidate-name rg_current_default \\
        --pattern-count 20 --pattern-sigma 0.03 \\
        --pattern-max-depth 0.05 --pattern-seed 42 \\
        --max-conditions 3 --hotspot-top-percent 10 \\
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
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optics_simulation.contribution import (
    build_multi_condition_legacy_hotspot_contribution_map,
    build_pattern_induced_hotspot_diagnostic,
    select_legacy_worst_conditions,
)
from optics_simulation.geometry import (
    create_actual_bottle_body_vertex_mask,
    create_target_scaled_mesh_copy,
    load_mesh,
)
from optics_simulation.optics import (
    LegacyPatternedPetWaterSetup,
    LegacyPetWaterTraceSetup,
    create_legacy_experiment_schedule,
    create_legacy_pet_water_trace_setup_from_shell_mesh,
    create_risk_guided_legacy_patterned_pet_water_setup,
    run_legacy_pet_water_angle_distance_scan,
)
from optics_simulation.pattern import (
    create_subdivided_mesh_copy_for_patterning,
)
from optics_simulation.thermal import (
    compare_legacy_thermal_risk_scans,
    compute_thermal_risk_over_legacy_optical_scan,
)


_DEFAULT_MESH_PATH = "data/raw/stl/pet_bottle.stl"


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Actual STL pattern-induced hotspot diagnostic; not a "
            "physical PET-bottle validation."
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
    parser.add_argument(
        "--subdivision-iterations", type=int, default=1,
    )
    parser.add_argument(
        "--candidate-name", type=str, default="rg_current_default",
    )
    parser.add_argument("--pattern-count", type=int, default=20)
    parser.add_argument("--pattern-sigma", type=float, default=0.03)
    parser.add_argument(
        "--pattern-max-depth", type=float, default=0.05,
    )
    parser.add_argument("--pattern-seed", type=int, default=42)
    parser.add_argument("--max-conditions", type=int, default=3)
    parser.add_argument(
        "--hotspot-top-percent", type=float, default=10.0,
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
    print("Actual STL pattern-induced hotspot diagnostic demo")
    print(
        "Actual STL pattern-induced hotspot diagnostic; not a "
        "physical PET-bottle validation."
    )
    print(
        "Note: this diagnostic backtracks worsened patterned "
        "hotspots to shell-surface coordinates."
    )
    print(
        "Note: it compares baseline contribution bins and "
        "pattern-induced contribution bins."
    )
    print(
        "Note: new induced hotspot bins are diagnostic only."
    )
    print(
        "Note: the contribution map is not a measured physical "
        "risk field."
    )
    print(
        "Note: a pattern can reduce one metric while inducing "
        "new hotspots elsewhere."
    )
    print("Note: this is not optimization.")
    print(
        "Note: this does not prove fire prevention or PET-bottle "
        "safety."
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


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    _print_static_header()

    print(f"Mesh path: {Path(args.mesh)}")
    print(f"Candidate: {str(args.candidate_name)}")
    print(
        f"Subdivision iterations: "
        f"{int(args.subdivision_iterations)}"
    )

    mesh = load_mesh(Path(args.mesh))

    scaled_shell, _ = create_target_scaled_mesh_copy(
        mesh,
        target_height=float(args.target_height),
        target_diameter=float(args.target_diameter),
    )
    if int(args.subdivision_iterations) > 0:
        subdivided_shell, _ = (
            create_subdivided_mesh_copy_for_patterning(
                scaled_shell,
                iterations=int(args.subdivision_iterations),
            )
        )
    else:
        subdivided_shell = scaled_shell

    baseline_setup = (
        create_legacy_pet_water_trace_setup_from_shell_mesh(
            subdivided_shell,
            wall_thickness=float(args.wall_thickness),
            inner_offset_mode=str(args.inner_offset_mode),
        )
    )
    body_mask = create_actual_bottle_body_vertex_mask(
        subdivided_shell,
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

    baseline_optical = run_legacy_pet_water_angle_distance_scan(
        setup=baseline_setup, **scan_kwargs,
    )
    baseline_thermal = compute_thermal_risk_over_legacy_optical_scan(
        baseline_optical, **thermal_kwargs,
    )

    selections = select_legacy_worst_conditions(
        baseline_optical, baseline_thermal,
        max_conditions=int(args.max_conditions),
    )
    if len(selections) == 0:
        print("Selected condition count: 0")
        print("Aggregate risk active count: 0")
        print("Invariants: FAIL")
        return 1

    multi = build_multi_condition_legacy_hotspot_contribution_map(
        setup=baseline_setup,
        conditions=selections,
        source_width=float(args.source_width),
        source_height=float(args.source_height),
        source_radius=float(args.source_radius),
        sample_count_y=int(args.sample_count_y),
        sample_count_z=int(args.sample_count_z),
        detector_size=float(args.detector_size),
        detector_resolution=(
            int(args.detector_resolution),
            int(args.detector_resolution),
        ),
        epsilon=0.1,
        hotspot_top_percent=float(args.hotspot_top_percent),
    )
    if int(multi.aggregate_risk_map.active_count) == 0:
        print("Aggregate risk active count: 0")
        print(
            "Aggregate risk active count is 0; pattern-induced "
            "hotspot diagnostic skipped."
        )
        print("Invariants: FAIL")
        return 1

    patterned_setup = (
        create_risk_guided_legacy_patterned_pet_water_setup(
            original_setup=baseline_setup,
            risk_map=multi.aggregate_risk_map,
            body_include_mask=body_mask.include_mask,
            pattern_count=int(args.pattern_count),
            pattern_sigma_u=float(args.pattern_sigma),
            pattern_sigma_v=float(args.pattern_sigma),
            pattern_max_depth=float(args.pattern_max_depth),
            pattern_seed=int(args.pattern_seed),
            wall_thickness=float(args.wall_thickness),
            inner_offset_mode=str(args.inner_offset_mode),
        )
    )
    candidate_legacy = _patterned_setup_to_legacy(patterned_setup)

    candidate_optical = run_legacy_pet_water_angle_distance_scan(
        setup=candidate_legacy, **scan_kwargs,
    )
    candidate_thermal = (
        compute_thermal_risk_over_legacy_optical_scan(
            candidate_optical, **thermal_kwargs,
        )
    )
    comparison = compare_legacy_thermal_risk_scans(
        baseline_thermal, candidate_thermal,
    )

    diagnostic = build_pattern_induced_hotspot_diagnostic(
        baseline_setup=baseline_setup,
        candidate_setup=candidate_legacy,
        comparison_result=comparison,
        source_width=float(args.source_width),
        source_height=float(args.source_height),
        source_radius=float(args.source_radius),
        sample_count_y=int(args.sample_count_y),
        sample_count_z=int(args.sample_count_z),
        detector_size=float(args.detector_size),
        detector_resolution=(
            int(args.detector_resolution),
            int(args.detector_resolution),
        ),
        epsilon=0.1,
        hotspot_top_percent=float(args.hotspot_top_percent),
        contribution_resolution=(32, 64),
    )

    print(
        f"Worsened angle: "
        f"{float(diagnostic.worsened_angle_degrees):.4f}"
    )
    print(
        f"Worsened detector distance: "
        f"{float(diagnostic.worsened_detector_distance):.4f}"
    )
    print(
        f"Delta max temperature: "
        f"{float(diagnostic.delta_max_temperature_k):+.6f}"
    )
    print(
        f"Delta threshold exceeded count: "
        f"{int(diagnostic.delta_threshold_exceeded_count):+d}"
    )
    print(
        f"Baseline selected rays: "
        f"{int(diagnostic.baseline_selected_rays)}"
    )
    print(
        f"Candidate selected rays: "
        f"{int(diagnostic.candidate_selected_rays)}"
    )
    print(
        f"Baseline contribution bins: "
        f"{int(diagnostic.overlap.baseline_nonzero_bins)}"
    )
    print(
        f"Candidate contribution bins: "
        f"{int(diagnostic.overlap.candidate_nonzero_bins)}"
    )
    print(
        f"Shared contribution bins: "
        f"{int(diagnostic.overlap.shared_nonzero_bins)}"
    )
    print(
        f"Candidate-only contribution bins: "
        f"{int(diagnostic.overlap.candidate_only_bins)}"
    )
    print(
        f"Candidate new bin fraction: "
        f"{float(diagnostic.induced_new_bin_fraction):.6f}"
    )
    print(
        f"Candidate-only contribution weight: "
        f"{float(diagnostic.overlap.candidate_only_weight):.6f}"
    )
    print(f"Tradeoff label: {diagnostic.tradeoff_label}")

    checks: list[bool] = []
    res_b = (
        diagnostic.baseline_contribution
        .contribution_map.weight_map.shape
    )
    res_c = (
        diagnostic.candidate_contribution
        .contribution_map.weight_map.shape
    )
    checks.append(res_b == res_c)
    frac = float(diagnostic.induced_new_bin_fraction)
    checks.append(0.0 <= frac <= 1.0)
    checks.append(int(diagnostic.baseline_selected_rays) >= 0)
    checks.append(int(diagnostic.candidate_selected_rays) >= 0)
    ok = all(checks)
    print(f"Invariants: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
