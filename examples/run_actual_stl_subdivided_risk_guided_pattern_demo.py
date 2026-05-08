"""Subdivided actual STL risk-guided pattern smoke demo.

Subdivided actual STL risk-guided pattern smoke check; not a
physical PET-bottle validation. Loads the supplied STL, runs the
multi-condition aggregate hotspot risk map on the coarse scaled
shell (for efficiency), then builds a synthetic subdivided copy
of the same scaled shell and applies the risk-guided Gaussian
dimple pattern on the subdivided shell. The subdivided original
shell and the subdivided risk-guided patterned shell are then
compared head-to-head under the same legacy PET-water
angle / detector-distance scan schedule. The purpose of this
demo is to disentangle "is the mixed tradeoff caused by pattern
parameters or by mesh resolution?"; it is not a physical
PET-bottle validation, not manufacturing-ready, and not a
fire-prevention proof.

Old H.max raw detector-bin count is **NOT** used as a primary
metric. Canonical reductions are the relative irradiance
surrogate, ``C99``, and the per-pixel thermal-risk metrics.

Scope (also enforced at runtime)
--------------------------------
- Subdivision is a synthetic numerical refinement for
  pattern-resolution diagnostics.
- Subdivision is not mesh repair.
- Subdivision is not manufacturing validation.
- The aggregate risk map is selected from the coarse baseline
  (multi-condition hotspot backtracking on the coarse shell);
  subdivided original and subdivided patterned shells are then
  compared head-to-head.
- The generated pattern is risk-guided but not optimized.
- Thermal-risk metrics are surrogate metrics.
- Delta values are diagnostic only.
- A negative delta is not required.
- Do not claim fire prevention or PET-bottle safety.

Usage::

    python examples/run_actual_stl_subdivided_risk_guided_pattern_demo.py \\
        --mesh data/raw/stl/pet_bottle.stl \\
        --target-height 225.6 --target-diameter 72.1 \\
        --wall-thickness 0.3 --inner-offset-mode auto \\
        --subdivision-iterations 1 \\
        --max-conditions 3 --hotspot-top-percent 10 \\
        --angle-count 18 --angle-step 20 \\
        --detector-count 8 --detector-start 100 --detector-spacing 40 \\
        --sample-count-y 11 --sample-count-z 21 \\
        --detector-resolution 40 \\
        --pattern-count 20 --pattern-sigma 0.03 \\
        --pattern-max-depth 0.05 --pattern-seed 42 \\
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
    select_legacy_worst_conditions,
)
from optics_simulation.geometry import load_mesh
from optics_simulation.optics import (
    LegacyOpticalScanResult,
    LegacyPatternedPetWaterSetup,
    LegacyPetWaterTraceSetup,
    create_legacy_experiment_schedule,
    create_legacy_pet_water_trace_setup,
    create_subdivided_risk_guided_legacy_pattern_setup,
    run_legacy_pet_water_angle_distance_scan,
)
from optics_simulation.thermal import (
    LegacyThermalRiskScanResult,
    compare_legacy_thermal_risk_scans,
    compute_thermal_risk_over_legacy_optical_scan,
)


_DEFAULT_MESH_PATH = "data/raw/stl/pet_bottle.stl"


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Subdivided actual STL risk-guided pattern smoke "
            "check; not a physical PET-bottle validation."
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
    parser.add_argument("--max-conditions", type=int, default=3)
    parser.add_argument(
        "--hotspot-top-percent", type=float, default=10.0,
    )
    parser.add_argument("--pattern-count", type=int, default=20)
    parser.add_argument("--pattern-sigma", type=float, default=0.03)
    parser.add_argument(
        "--pattern-max-depth", type=float, default=0.05,
    )
    parser.add_argument("--pattern-seed", type=int, default=42)
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
    print("Subdivided actual STL risk-guided pattern demo")
    print(
        "Subdivided actual STL risk-guided pattern smoke check; "
        "not a physical PET-bottle validation."
    )
    print(
        "Note: subdivision is a synthetic numerical refinement "
        "for pattern-resolution diagnostics."
    )
    print("Note: subdivision is not mesh repair.")
    print("Note: subdivision is not manufacturing validation.")
    print(
        "Note: the aggregate risk map is selected from the "
        "coarse baseline; subdivided original and subdivided "
        "patterned shells are then compared head-to-head."
    )
    print(
        "Note: the generated pattern is risk-guided but not "
        "optimized."
    )
    print("Note: thermal-risk metrics are surrogate metrics.")
    print(
        "Note: delta values are diagnostic only; a negative "
        "delta is not required."
    )
    print(
        "Note: do not claim fire prevention or PET-bottle "
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


def _format_optional_float(value, fmt: str = "{:.6f}") -> str:
    if value is None:
        return "None"
    return fmt.format(float(value))


def _check_invariants(
    *,
    base_vertex_count: int,
    subdivided_vertex_count: int,
    original_selected: int,
    subdivided_selected: int,
    moved_vertex_count: int,
    original_optical: LegacyOpticalScanResult,
    patterned_optical: LegacyOpticalScanResult,
    original_thermal: LegacyThermalRiskScanResult,
    patterned_thermal: LegacyThermalRiskScanResult,
    comparison,
) -> bool:
    checks: list[bool] = []
    checks.append(
        int(subdivided_vertex_count) > int(base_vertex_count)
    )
    checks.append(int(subdivided_selected) >= int(original_selected))
    checks.append(int(moved_vertex_count) > 0)
    checks.append(
        int(len(original_optical.entries))
        == int(len(patterned_optical.entries))
    )
    checks.append(
        int(original_thermal.entry_count)
        == int(patterned_thermal.entry_count)
    )
    checks.append(int(comparison.entry_count) > 0)
    return all(checks)


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    _print_static_header()

    print(f"Mesh path: {Path(args.mesh)}")
    print(
        f"Subdivision iterations: "
        f"{int(args.subdivision_iterations)}"
    )

    mesh = load_mesh(Path(args.mesh))

    coarse_setup = create_legacy_pet_water_trace_setup(
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

    coarse_optical = run_legacy_pet_water_angle_distance_scan(
        setup=coarse_setup, **scan_kwargs,
    )
    coarse_thermal = compute_thermal_risk_over_legacy_optical_scan(
        coarse_optical, **thermal_kwargs,
    )

    selections = select_legacy_worst_conditions(
        coarse_optical, coarse_thermal,
        max_conditions=int(args.max_conditions),
    )
    print(f"Selected condition count: {len(selections)}")
    for s in selections:
        print(
            f"Selected condition: angle="
            f"{float(s.angle_degrees):.4f}, "
            f"distance={float(s.detector_distance):.4f}, "
            f"reason={s.reason}, score={float(s.score):.6f}"
        )

    if len(selections) == 0:
        print("Aggregate risk active count: 0")
        print("Invariants: FAIL")
        return 1

    multi = build_multi_condition_legacy_hotspot_contribution_map(
        setup=coarse_setup,
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
    print(
        f"Aggregate risk active count: "
        f"{int(multi.aggregate_risk_map.active_count)}"
    )

    if int(multi.aggregate_risk_map.active_count) == 0:
        print(
            "Aggregate risk active count is 0; subdivided "
            "risk-guided pattern skipped."
        )
        print("Invariants: FAIL")
        return 1

    sub_setup = create_subdivided_risk_guided_legacy_pattern_setup(
        mesh,
        target_height=float(args.target_height),
        target_diameter=float(args.target_diameter),
        wall_thickness=float(args.wall_thickness),
        inner_offset_mode=str(args.inner_offset_mode),
        subdivision_iterations=int(args.subdivision_iterations),
        risk_map=multi.aggregate_risk_map,
        pattern_count=int(args.pattern_count),
        pattern_sigma_u=float(args.pattern_sigma),
        pattern_sigma_v=float(args.pattern_sigma),
        pattern_max_depth=float(args.pattern_max_depth),
        pattern_seed=int(args.pattern_seed),
    )

    print(
        f"Original readiness: "
        f"{sub_setup.original_readiness.readiness_label}"
    )
    print(
        f"Subdivided readiness: "
        f"{sub_setup.subdivided_readiness.readiness_label}"
    )
    print(
        f"Original selected vertices: "
        f"{int(sub_setup.original_readiness.selected_vertex_count)} / "
        f"{int(sub_setup.original_readiness.vertex_count)}"
    )
    print(
        f"Subdivided selected vertices: "
        f"{int(sub_setup.subdivided_readiness.selected_vertex_count)} / "
        f"{int(sub_setup.subdivided_readiness.vertex_count)}"
    )
    print(
        f"Pattern moved vertices: "
        f"{int(sub_setup.moved_vertex_count)}"
    )
    print(
        f"Max displacement: "
        f"{float(sub_setup.max_displacement):.6f}"
    )

    original_subdiv_optical = run_legacy_pet_water_angle_distance_scan(
        setup=sub_setup.original_subdivided_setup,
        **scan_kwargs,
    )
    patterned_subdiv_legacy = _patterned_setup_to_legacy(
        sub_setup.patterned_subdivided_setup,
    )
    patterned_subdiv_optical = (
        run_legacy_pet_water_angle_distance_scan(
            setup=patterned_subdiv_legacy, **scan_kwargs,
        )
    )

    original_subdiv_thermal = (
        compute_thermal_risk_over_legacy_optical_scan(
            original_subdiv_optical, **thermal_kwargs,
        )
    )
    patterned_subdiv_thermal = (
        compute_thermal_risk_over_legacy_optical_scan(
            patterned_subdiv_optical, **thermal_kwargs,
        )
    )

    comparison = compare_legacy_thermal_risk_scans(
        original_subdiv_thermal, patterned_subdiv_thermal,
    )

    o_max_temp = original_subdiv_thermal.max_temperature_k
    p_max_temp = patterned_subdiv_thermal.max_temperature_k
    if o_max_temp is not None and p_max_temp is not None:
        delta_max_temp = float(p_max_temp) - float(o_max_temp)
    else:
        delta_max_temp = None
    o_thresh = original_subdiv_thermal.max_threshold_exceeded_count
    p_thresh = patterned_subdiv_thermal.max_threshold_exceeded_count
    if o_thresh is not None and p_thresh is not None:
        delta_thresh = int(p_thresh) - int(o_thresh)
    else:
        delta_thresh = None

    print(
        f"Original subdivided max temperature: "
        f"{_format_optional_float(o_max_temp)}"
    )
    print(
        f"Patterned subdivided max temperature: "
        f"{_format_optional_float(p_max_temp)}"
    )
    if delta_max_temp is None:
        print("Delta max temperature: None")
    else:
        print(f"Delta max temperature: {delta_max_temp:+.6f}")

    print(
        f"Original threshold exceeded count: "
        f"{0 if o_thresh is None else int(o_thresh)}"
    )
    print(
        f"Patterned threshold exceeded count: "
        f"{0 if p_thresh is None else int(p_thresh)}"
    )
    if delta_thresh is None:
        print("Delta threshold exceeded count: None")
    else:
        print(f"Delta threshold exceeded count: {delta_thresh:+d}")

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
    print(f"Tradeoff label: {dominant_label}")
    print(
        f"Passes guardrails: {int(comparison.pass_count)} / "
        f"{int(comparison.entry_count)}"
    )
    if int(comparison.entry_count) > 0:
        print(
            f"Guardrail violations: "
            f"{int(comparison.fail_count)} / "
            f"{int(comparison.entry_count)}"
        )
    else:
        print("Guardrail violations: 0 / 0")

    ok = _check_invariants(
        base_vertex_count=int(
            len(sub_setup.base_scaled_shell.vertices)
        ),
        subdivided_vertex_count=int(
            len(sub_setup.subdivided_shell.vertices)
        ),
        original_selected=int(
            sub_setup.original_readiness.selected_vertex_count
        ),
        subdivided_selected=int(
            sub_setup.subdivided_readiness.selected_vertex_count
        ),
        moved_vertex_count=int(sub_setup.moved_vertex_count),
        original_optical=original_subdiv_optical,
        patterned_optical=patterned_subdiv_optical,
        original_thermal=original_subdiv_thermal,
        patterned_thermal=patterned_subdiv_thermal,
        comparison=comparison,
    )
    print(f"Invariants: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
