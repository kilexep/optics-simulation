"""Actual STL ring-offset risk-guided pattern parameter sweep demo.

Actual STL ring-offset risk-guided pattern parameter sweep
diagnostic; not a physical PET-bottle validation. Loads the
supplied STL, builds the self-consistent subdivided baseline +
multi-condition aggregate hotspot risk map, and then sweeps a
small grid of ``(inner_radius_px, outer_radius_px)`` ring-offset
candidates against the same subdivided baseline. For each
candidate the demo runs the full risk-guided pattern pipeline
(ring transform -> Gaussian dimple sampling -> patterned
PET-water trace -> thermal-risk metrics) and computes the
``compare_legacy_thermal_risk_scans`` /
``build_pattern_induced_hotspot_diagnostic`` per-candidate
diagnostics. Finally, candidates are ranked by Pareto
non-dominance on
``(worst_delta_max_temperature_k, best_delta_threshold_count,
-pass_ratio)`` and by a min-max-normalized composite score on the
same axes.

Old H.max raw detector-bin count is **NOT** used as a primary
metric. Canonical reductions remain the relative irradiance
surrogate, ``C99``, and the per-pixel thermal-risk metrics; the
ring-offset sweep is a pattern-placement strategy diagnostic
only.

Scope (also enforced at runtime)
--------------------------------
- This is a surrogate diagnostic, not a fire-prevention proof.
- A higher guardrail pass count does not prove safety.
- A lower new-bin fraction is not automatically better, because
  it may mean the pattern failed to move/redistribute the
  hotspot.
- Ring-offset can reduce overlap with existing hotspots but may
  create new caustics.
- Ranking is diagnostic ordering; no candidate is declared
  manufacturing-ready.

Usage::

    python examples/run_actual_stl_ring_offset_sweep_demo.py \\
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
from optics_simulation.geometry import (
    create_actual_bottle_body_vertex_mask,
    create_target_scaled_mesh_copy,
    load_mesh,
)
from optics_simulation.optics import (
    RingOffsetSweepCandidateSpec,
    create_legacy_experiment_schedule,
    create_legacy_pet_water_trace_setup_from_shell_mesh,
    run_actual_stl_ring_offset_sweep,
    run_legacy_pet_water_angle_distance_scan,
)
from optics_simulation.pattern import (
    create_subdivided_mesh_copy_for_patterning,
)
from optics_simulation.thermal import (
    compute_thermal_risk_over_legacy_optical_scan,
)


_DEFAULT_MESH_PATH = "data/raw/stl/pet_bottle.stl"


# Default (inner, outer) sweep grid. inner < outer is enforced by
# the sweep runner; combinations where outer <= inner are simply
# dropped here so the runner only sees valid candidates.
_DEFAULT_INNERS: tuple[int, ...] = (1, 2, 3)
_DEFAULT_OUTERS: tuple[int, ...] = (3, 5, 7)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Actual STL ring-offset risk-guided pattern parameter "
            "sweep diagnostic; not a physical PET-bottle validation."
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
    print("Actual STL ring-offset sweep demo")
    print(
        "Actual STL ring-offset risk-guided pattern parameter "
        "sweep diagnostic; not a physical PET-bottle validation."
    )
    print(
        "Note: this is a surrogate diagnostic, not a "
        "fire-prevention proof."
    )
    print(
        "Note: a higher guardrail pass count does not prove "
        "safety."
    )
    print(
        "Note: a lower new-bin fraction is not automatically "
        "better; it may mean the pattern failed to move the "
        "hotspot."
    )
    print(
        "Note: ring-offset can reduce overlap with existing "
        "hotspots but may create new caustics."
    )
    print(
        "Note: ranking is diagnostic ordering; no candidate is "
        "declared manufacturing-ready."
    )


def _build_candidates() -> tuple[RingOffsetSweepCandidateSpec, ...]:
    out: list[RingOffsetSweepCandidateSpec] = []
    for inner in _DEFAULT_INNERS:
        for outer in _DEFAULT_OUTERS:
            if int(outer) <= int(inner):
                # Skip invalid (outer <= inner).
                continue
            out.append(
                RingOffsetSweepCandidateSpec(
                    name=f"ring_in{int(inner)}_out{int(outer)}",
                    inner_radius_px=int(inner),
                    outer_radius_px=int(outer),
                )
            )
    return tuple(out)


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    _print_static_header()

    candidates = _build_candidates()
    print(f"Mesh path: {Path(args.mesh)}")
    print(f"Candidate count: {len(candidates)}")
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
    print(f"Selected condition count: {len(selections)}")
    if len(selections) == 0:
        print("Source risk active count: 0")
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
    print(
        f"Source risk active count: "
        f"{int(multi.aggregate_risk_map.active_count)}"
    )
    if int(multi.aggregate_risk_map.active_count) == 0:
        print("Aggregate risk is empty; sweep skipped.")
        print("Invariants: FAIL")
        return 1

    sweep = run_actual_stl_ring_offset_sweep(
        original_setup=baseline_setup,
        baseline_thermal_scan=baseline_thermal,
        aggregate_risk_map=multi.aggregate_risk_map,
        body_include_mask=body_mask.include_mask,
        candidates=candidates,
        angles_degrees=schedule.angles_degrees,
        detector_distances=schedule.detector_distances,
        source_width=schedule.source_width,
        source_height=schedule.source_height,
        source_radius=schedule.source_radius,
        sample_count_y=schedule.sample_count_y,
        sample_count_z=schedule.sample_count_z,
        detector_size=schedule.detector_size,
        detector_resolution=(
            int(args.detector_resolution),
            int(args.detector_resolution),
        ),
        duration_s=float(args.duration),
        dt_s=float(args.dt),
        pattern_count=int(args.pattern_count),
        pattern_sigma_u=float(args.pattern_sigma),
        pattern_sigma_v=float(args.pattern_sigma),
        pattern_max_depth=float(args.pattern_max_depth),
        pattern_seed=int(args.pattern_seed),
        nominal_incident_irradiance_w_m2=float(
            args.nominal_incident_irradiance
        ),
        areal_heat_capacity_j_m2k=float(args.areal_heat_capacity),
        absorptivity=float(args.absorptivity),
        h_conv_w_m2k=float(args.h_conv),
        emissivity=float(args.emissivity),
        threshold_temp_k=float(args.threshold_temp),
        wall_thickness=float(args.wall_thickness),
        inner_offset_mode=str(args.inner_offset_mode),
        hotspot_top_percent=float(args.hotspot_top_percent),
    )

    for entry in sweep.entries:
        c = entry.candidate
        print(f"Candidate: {c.name}")
        print(
            f"  Inner radius: {int(c.inner_radius_px)}, "
            f"outer radius: {int(c.outer_radius_px)}"
        )
        print(
            f"  Source risk active: "
            f"{int(entry.source_risk_active_count)}, "
            f"ring risk active: "
            f"{int(entry.ring_risk_active_count)}"
        )
        print(
            f"  Moved vertices: {int(entry.moved_vertex_count)}"
        )
        print(
            f"  Worst delta max temperature: "
            f"{float(entry.delta_max_temperature_k):+.6f}"
        )
        print(
            f"  Best delta threshold count: "
            f"{int(entry.delta_threshold_count):+d}"
        )
        print(
            f"  Candidate-only bins: "
            f"{int(entry.candidate_only_bins)}"
        )
        print(
            f"  New bin fraction: "
            f"{float(entry.new_bin_fraction):.6f}"
        )
        print(
            f"  Guardrail pass: "
            f"{int(entry.guardrail_pass_count)} / "
            f"{int(entry.total_condition_count)} "
            f"(ratio={float(entry.pass_ratio):.4f})"
        )
        print(f"  Tradeoff label: {entry.tradeoff_label}")

    print("Pareto non-dominated candidates:")
    if sweep.pareto_candidate_names:
        for nm in sweep.pareto_candidate_names:
            print(f"  - {nm}")
    else:
        print("  (none)")

    print(
        f"Best by max-temperature delta: "
        f"{sweep.best_by_max_temperature_delta} "
        f"(delta={float(sweep.best_delta_max_temperature_k):+.6f})"
    )
    print(
        f"Best by threshold-count delta: "
        f"{sweep.best_by_threshold_count_delta} "
        f"(delta={int(sweep.best_delta_threshold_count):+d})"
    )
    print(
        f"Best by pass ratio: "
        f"{sweep.best_by_pass_ratio} "
        f"(ratio={float(sweep.best_pass_ratio):.4f})"
    )
    print(
        f"Best by composite score: "
        f"{sweep.best_by_composite_score} "
        f"(score={float(sweep.best_composite_score):.6f})"
    )
    print(
        "Note: \"best by ...\" reports the candidate that "
        "minimizes the corresponding diagnostic in this surrogate "
        "sweep, not a safety recommendation."
    )

    checks: list[bool] = []
    checks.append(int(sweep.candidate_count) == len(candidates))
    checks.append(len(selections) > 0)
    checks.append(int(multi.aggregate_risk_map.active_count) > 0)
    checks.append(
        all(
            int(e.guardrail_pass_count)
            + int(e.guardrail_fail_count)
            == int(e.total_condition_count)
            for e in sweep.entries
        )
    )
    checks.append(
        all(
            int(e.moved_vertex_count) > 0
            for e in sweep.entries
        )
    )
    checks.append(len(sweep.pareto_candidate_names) > 0)
    ok = all(checks)
    print(f"Invariants: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
