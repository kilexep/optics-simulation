"""Actual STL centered vs ring-offset risk-guided pattern diagnostic.

Actual STL offset-ring risk-guided pattern diagnostic; not a
physical PET-bottle validation. Loads the supplied STL, builds the
self-consistent subdivided baseline + multi-condition aggregate
hotspot risk map, and then evaluates two pattern-placement
strategies head-to-head against the same subdivided baseline:

- **Centered**: a risk-guided Gaussian dimple pattern sampled
  directly from the multi-condition aggregate risk map. Dimples
  fall on top of baseline hotspot contribution bins.
- **Ring-offset**: the same aggregate risk map transformed via
  :func:`create_ring_offset_risk_map` so the active pixels lie in
  an annular ring around the original hotspot bins. A second
  Gaussian dimple pattern is sampled from this transformed map.

Both candidates use the **same subdivided body include mask** and
the **same scan schedule**. Each is compared to the subdivided
baseline thermal-risk scan via
:func:`compare_legacy_thermal_risk_scans`, and a
:func:`build_pattern_induced_hotspot_diagnostic` is run for each
on its own worst worsened entry to compare baseline vs. patterned
contribution overlap.

Old H.max raw detector-bin count is **NOT** used as a primary
metric. Canonical reductions remain the relative irradiance
surrogate, ``C99``, and the per-pixel thermal-risk metrics; the
ring offset is only a pattern-placement strategy diagnostic.

Scope (also enforced at runtime)
--------------------------------
- Offset-ring transformation is a diagnostic pattern-placement
  strategy.
- It tests whether avoiding direct dimple placement on hotspot
  contribution centers changes hotspot behavior.
- It is not optimization.
- It is not manufacturing validation.
- Contribution overlap is a surrogate ray-backtracking diagnostic.
- Delta values are diagnostic only.
- A negative delta is not required.
- Do not claim fire prevention or PET-bottle safety.
- A lower new-bin fraction is not automatically better.
- A ring-offset pattern can still be mixed.

Usage::

    python examples/run_actual_stl_ring_offset_pattern_diagnostic_demo.py \\
        --mesh data/raw/stl/pet_bottle.stl \\
        --target-height 225.6 --target-diameter 72.1 \\
        --wall-thickness 0.3 --inner-offset-mode auto \\
        --subdivision-iterations 1 \\
        --max-conditions 3 --hotspot-top-percent 10 \\
        --ring-inner-radius 1 --ring-outer-radius 3 \\
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
    build_pattern_induced_hotspot_diagnostic,
    create_ring_offset_risk_map,
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
            "Actual STL offset-ring risk-guided pattern diagnostic; "
            "not a physical PET-bottle validation."
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
    parser.add_argument("--ring-inner-radius", type=int, default=1)
    parser.add_argument("--ring-outer-radius", type=int, default=3)
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
    print("Actual STL ring-offset pattern diagnostic demo")
    print(
        "Actual STL offset-ring risk-guided pattern diagnostic; "
        "not a physical PET-bottle validation."
    )
    print(
        "Note: offset-ring transformation is a diagnostic "
        "pattern-placement strategy."
    )
    print(
        "Note: it tests whether avoiding direct dimple placement "
        "on hotspot contribution centers changes hotspot "
        "behavior."
    )
    print("Note: it is not optimization.")
    print("Note: it is not manufacturing validation.")
    print(
        "Note: contribution overlap is a surrogate ray-"
        "backtracking diagnostic."
    )
    print(
        "Note: delta values are diagnostic only; a negative "
        "delta is not required."
    )
    print(
        "Note: a lower new-bin fraction is not automatically "
        "better."
    )
    print(
        "Note: a ring-offset pattern can still be mixed."
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


def _format_optional_float(value, fmt: str = "{:+.6f}") -> str:
    if value is None:
        return "None"
    return fmt.format(float(value))


def _format_optional_int(value, fmt: str = "{:+d}") -> str:
    if value is None:
        return "None"
    return fmt.format(int(value))


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    _print_static_header()

    print(f"Mesh path: {Path(args.mesh)}")
    print("Centered risk-guided pattern: enabled")
    print("Ring-offset risk-guided pattern: enabled")
    print(f"Ring inner radius: {int(args.ring_inner_radius)}")
    print(f"Ring outer radius: {int(args.ring_outer_radius)}")
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
    print(f"Baseline scan entries: {int(len(baseline_optical.entries))}")

    selections = select_legacy_worst_conditions(
        baseline_optical, baseline_thermal,
        max_conditions=int(args.max_conditions),
    )
    print(f"Selected condition count: {len(selections)}")

    if len(selections) == 0:
        print("Source risk active count: 0")
        print("Ring risk active count: 0")
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
    source_risk_map = multi.aggregate_risk_map
    print(
        f"Source risk active count: "
        f"{int(source_risk_map.active_count)}"
    )

    ring_result = create_ring_offset_risk_map(
        source_risk_map,
        inner_radius_px=int(args.ring_inner_radius),
        outer_radius_px=int(args.ring_outer_radius),
        wrap_u=True,
        epsilon=0.01,
    )
    ring_risk_map = ring_result.transformed_risk_map
    print(
        f"Ring risk active count: "
        f"{int(ring_risk_map.active_count)}"
    )

    if (
        int(source_risk_map.active_count) == 0
        or int(ring_risk_map.active_count) == 0
    ):
        print(
            "Source or ring risk active count is 0; diagnostic "
            "skipped."
        )
        print("Invariants: FAIL")
        return 1

    centered_setup = (
        create_risk_guided_legacy_patterned_pet_water_setup(
            original_setup=baseline_setup,
            risk_map=source_risk_map,
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
    ring_setup = (
        create_risk_guided_legacy_patterned_pet_water_setup(
            original_setup=baseline_setup,
            risk_map=ring_risk_map,
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
    centered_legacy = _patterned_setup_to_legacy(centered_setup)
    ring_legacy = _patterned_setup_to_legacy(ring_setup)

    print(
        f"Centered moved vertices: "
        f"{int(centered_setup.patterned_mesh_result.moved_vertex_count)}"
    )
    print(
        f"Ring moved vertices: "
        f"{int(ring_setup.patterned_mesh_result.moved_vertex_count)}"
    )

    centered_optical = run_legacy_pet_water_angle_distance_scan(
        setup=centered_legacy, **scan_kwargs,
    )
    ring_optical = run_legacy_pet_water_angle_distance_scan(
        setup=ring_legacy, **scan_kwargs,
    )
    centered_thermal = compute_thermal_risk_over_legacy_optical_scan(
        centered_optical, **thermal_kwargs,
    )
    ring_thermal = compute_thermal_risk_over_legacy_optical_scan(
        ring_optical, **thermal_kwargs,
    )

    centered_comparison = compare_legacy_thermal_risk_scans(
        baseline_thermal, centered_thermal,
    )
    ring_comparison = compare_legacy_thermal_risk_scans(
        baseline_thermal, ring_thermal,
    )

    centered_diag = build_pattern_induced_hotspot_diagnostic(
        baseline_setup=baseline_setup,
        candidate_setup=centered_legacy,
        comparison_result=centered_comparison,
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
    ring_diag = build_pattern_induced_hotspot_diagnostic(
        baseline_setup=baseline_setup,
        candidate_setup=ring_legacy,
        comparison_result=ring_comparison,
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
        f"Centered delta max temperature: "
        f"{_format_optional_float(centered_comparison.worst_delta_max_temperature_k)}"
    )
    print(
        f"Ring delta max temperature: "
        f"{_format_optional_float(ring_comparison.worst_delta_max_temperature_k)}"
    )
    print(
        f"Centered delta threshold count: "
        f"{_format_optional_int(centered_comparison.best_delta_threshold_count)}"
    )
    print(
        f"Ring delta threshold count: "
        f"{_format_optional_int(ring_comparison.best_delta_threshold_count)}"
    )
    print(
        f"Centered candidate-only bins: "
        f"{int(centered_diag.overlap.candidate_only_bins)}"
    )
    print(
        f"Ring candidate-only bins: "
        f"{int(ring_diag.overlap.candidate_only_bins)}"
    )
    print(
        f"Centered new bin fraction: "
        f"{float(centered_diag.induced_new_bin_fraction):.6f}"
    )
    print(
        f"Ring new bin fraction: "
        f"{float(ring_diag.induced_new_bin_fraction):.6f}"
    )

    centered_label = (
        "improved" if centered_comparison.improved_count
        >= max(
            centered_comparison.worsened_count,
            centered_comparison.mixed_count,
            centered_comparison.unchanged_count,
        )
        else "worsened" if centered_comparison.worsened_count
        >= max(
            centered_comparison.mixed_count,
            centered_comparison.unchanged_count,
        )
        else "mixed" if centered_comparison.mixed_count
        >= centered_comparison.unchanged_count
        else "unchanged"
    )
    ring_label = (
        "improved" if ring_comparison.improved_count
        >= max(
            ring_comparison.worsened_count,
            ring_comparison.mixed_count,
            ring_comparison.unchanged_count,
        )
        else "worsened" if ring_comparison.worsened_count
        >= max(
            ring_comparison.mixed_count,
            ring_comparison.unchanged_count,
        )
        else "mixed" if ring_comparison.mixed_count
        >= ring_comparison.unchanged_count
        else "unchanged"
    )
    print(f"Centered tradeoff label: {centered_label}")
    print(f"Ring tradeoff label: {ring_label}")
    print(
        f"Centered passes guardrails: "
        f"{int(centered_comparison.pass_count)} / "
        f"{int(centered_comparison.entry_count)}"
    )
    print(
        f"Ring passes guardrails: "
        f"{int(ring_comparison.pass_count)} / "
        f"{int(ring_comparison.entry_count)}"
    )

    checks: list[bool] = []
    checks.append(int(len(baseline_optical.entries)) > 0)
    checks.append(
        int(len(centered_optical.entries))
        == int(len(baseline_optical.entries))
    )
    checks.append(
        int(len(ring_optical.entries))
        == int(len(baseline_optical.entries))
    )
    checks.append(int(source_risk_map.active_count) > 0)
    checks.append(int(ring_risk_map.active_count) > 0)
    checks.append(
        ring_risk_map.risk_map.shape == source_risk_map.risk_map.shape
    )
    checks.append(
        int(centered_setup.patterned_mesh_result.moved_vertex_count)
        > 0
    )
    checks.append(
        int(ring_setup.patterned_mesh_result.moved_vertex_count)
        > 0
    )
    ok = all(checks)
    print(f"Invariants: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
