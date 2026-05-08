"""Actual STL ring-offset risk-guided pattern parameter sweep demo.

Actual STL ring-offset risk-guided pattern parameter sweep
diagnostic; not a physical PET-bottle validation. Loads the
supplied STL, builds a self-consistent subdivided baseline +
multi-condition aggregate hotspot risk map, and sweeps an
``(inner_radius_px, outer_radius_px)`` grid of ring-offset
candidates. For each candidate the demo runs the full risk-guided
pattern pipeline and computes:

- selection-condition (in-sample) vs holdout vs full-grid
  thermal-risk diagnostics,
- baseline-vs-candidate hotspot-contribution overlap metrics
  (shared bins, retained source fraction, candidate-only bins,
  jaccard overlap),
- ring transform coverage metrics
  (``risk_support_expansion_ratio``, ``moved_vertex_fraction``).

Candidates are then Pareto-filtered on the **holdout** subset's
``(worst_delta_max_temperature_k, best_delta_threshold_count,
-pass_ratio)`` tuple, with non-discriminative axes auto-dropped.
A min-max-normalized composite score is reported as a tie-break
*hint inside the Pareto set*, not a primary ranking. A flag
records whether any candidate strictly improved worst-case
behavior.

Old H.max raw detector-bin count is **NOT** used as a primary
metric. Canonical reductions remain the relative irradiance
surrogate, ``C99``, and the per-pixel thermal-risk metrics; the
ring-offset sweep is a pattern-placement strategy diagnostic
only.

Scope (also enforced at runtime)
--------------------------------
- This is a surrogate diagnostic, not a fire-prevention proof.
- A higher guardrail pass ratio does not prove safety.
- A lower new-bin fraction is not automatically better; it may
  mean the pattern failed to move the hotspot.
- A candidate with zero new-bin fraction may still substantially
  overlap the original hotspot.
- The current negative result is limited to the tested ring
  family and fixed pattern-shape settings; it is not a blanket
  judgment on ring-offset patterns in general.

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
    RING_RADIUS_UNITS_DESCRIPTION,
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
    parser.add_argument(
        "--ranking-basis",
        choices=("holdout", "full", "in_sample"),
        default="holdout",
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
        "Note: a higher guardrail pass ratio does not prove "
        "safety."
    )
    print(
        "Note: a lower new-bin fraction is not automatically "
        "better; it may mean the pattern failed to move the "
        "hotspot."
    )
    print(
        "Note: a candidate with zero new-bin fraction may still "
        "substantially overlap the original hotspot."
    )
    print(
        "Note: the current negative result is limited to the "
        "tested ring family and fixed pattern-shape settings."
    )
    print(f"Note: units -- {RING_RADIUS_UNITS_DESCRIPTION}")


def _build_candidates() -> tuple[RingOffsetSweepCandidateSpec, ...]:
    out: list[RingOffsetSweepCandidateSpec] = []
    for inner in _DEFAULT_INNERS:
        for outer in _DEFAULT_OUTERS:
            if int(outer) <= int(inner):
                continue
            out.append(
                RingOffsetSweepCandidateSpec(
                    name=f"ring_in{int(inner)}_out{int(outer)}",
                    inner_radius_px=int(inner),
                    outer_radius_px=int(outer),
                )
            )
    return tuple(out)


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

    candidates = _build_candidates()
    print(f"Mesh path: {Path(args.mesh)}")
    print(f"Candidate count: {len(candidates)}")
    print(
        f"Subdivision iterations: "
        f"{int(args.subdivision_iterations)}"
    )
    print(f"Ranking basis: {str(args.ranking_basis)}")

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
        selections=selections,
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
        ranking_basis=str(args.ranking_basis),
    )

    print(
        f"Selection condition count: "
        f"{int(sweep.selected_condition_count)}"
    )
    print(
        f"Evaluation condition count: "
        f"{int(sweep.evaluation_condition_count)}"
    )
    print(
        f"Holdout condition count: "
        f"{int(sweep.holdout_condition_count)}"
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
            f"{int(entry.source_risk_active_count)} -> "
            f"ring active: "
            f"{int(entry.transformed_risk_active_count)} "
            f"(expansion ratio "
            f"{float(entry.risk_support_expansion_ratio):.4f})"
        )
        print(
            f"  Moved vertices: "
            f"{int(entry.moved_vertex_count)} / "
            f"{int(entry.body_mask_vertex_count)} "
            f"(fraction "
            f"{float(entry.moved_vertex_fraction):.4f})"
        )
        # In-sample vs holdout vs full
        full = entry.full_diagnostics
        ins = entry.in_sample_diagnostics
        hold = entry.holdout_diagnostics
        print(
            f"  Full worst delta max temperature: "
            f"{_format_optional_float(full.worst_delta_max_temperature_k)}"
        )
        print(
            f"  In-sample worst delta max temperature: "
            f"{_format_optional_float(ins.worst_delta_max_temperature_k)}"
        )
        print(
            f"  Holdout worst delta max temperature: "
            f"{_format_optional_float(hold.worst_delta_max_temperature_k)}"
        )
        print(
            f"  Full best delta threshold count: "
            f"{_format_optional_int(full.best_delta_threshold_count)}"
        )
        print(
            f"  Holdout best delta threshold count: "
            f"{_format_optional_int(hold.best_delta_threshold_count)}"
        )
        print(
            f"  Full pass ratio: "
            f"{int(full.pass_count)}/{int(full.entry_count)} "
            f"= {float(full.pass_ratio):.4f}"
        )
        print(
            f"  Holdout pass ratio: "
            f"{int(hold.pass_count)}/{int(hold.entry_count)} "
            f"= {float(hold.pass_ratio):.4f}"
        )
        # Overlap on the worst-worsened entry
        print(
            f"  Hotspot overlap (at angle="
            f"{float(entry.hotspot_diagnostic_angle_degrees):.4f}, "
            f"distance="
            f"{float(entry.hotspot_diagnostic_detector_distance):.4f}):"
        )
        print(
            f"    Baseline bins: "
            f"{int(entry.baseline_nonzero_bins)}, "
            f"candidate bins: "
            f"{int(entry.candidate_nonzero_bins)}"
        )
        print(
            f"    Source overlap bins: "
            f"{int(entry.source_overlap_bin_count)} "
            f"(retained source fraction "
            f"{float(entry.retained_source_bin_fraction):.4f})"
        )
        print(
            f"    Candidate-only bins: "
            f"{int(entry.candidate_only_bin_count)} "
            f"(new bin fraction "
            f"{float(entry.new_bin_fraction):.4f})"
        )
        print(
            f"    Jaccard overlap: "
            f"{float(entry.jaccard_overlap):.4f}"
        )
        print(
            f"  Tradeoff label (holdout): {hold.dominant_label}"
        )

    print("Pareto non-dominated candidates "
          f"(basis={sweep.pareto_basis}, "
          f"axes={list(sweep.pareto_axes)}):")
    if sweep.pareto_non_dominated_names:
        for nm in sweep.pareto_non_dominated_names:
            print(f"  - {nm}")
    else:
        print("  (none)")
    if sweep.non_discriminative_metrics:
        print(
            f"Non-discriminative metrics (range zero across "
            f"candidates on basis={sweep.pareto_basis}): "
            f"{list(sweep.non_discriminative_metrics)}"
        )
    else:
        print("Non-discriminative metrics: (none)")

    if bool(sweep.no_improving_candidate_under_current_sweep):
        prefix = "Least-bad by"
        print(
            "no_improving_candidate_under_current_sweep: True "
            "(every candidate's full-grid worst delta max "
            "temperature is strictly positive)"
        )
    else:
        prefix = "Best by"
        print(
            "no_improving_candidate_under_current_sweep: False"
        )
    if bool(sweep.no_improving_candidate_in_holdout):
        print(
            "no_improving_candidate_in_holdout: True "
            "(holdout subset shows no improving candidate)"
        )
    else:
        print("no_improving_candidate_in_holdout: False")

    nm_t = sweep.best_by_metric_names.get("delta_max_temperature_k")
    val_t = sweep.best_by_metric_values.get(
        "delta_max_temperature_k",
    )
    nm_c = sweep.best_by_metric_names.get("delta_threshold_count")
    val_c = sweep.best_by_metric_values.get(
        "delta_threshold_count",
    )
    nm_p = sweep.best_by_metric_names.get("pass_ratio_neg")
    val_p = sweep.best_by_metric_values.get("pass_ratio_neg")
    print(
        f"{prefix} max-temperature delta: {nm_t} "
        f"(value={_format_optional_float(val_t)})"
    )
    print(
        f"{prefix} threshold-count delta: {nm_c} "
        f"(value={_format_optional_float(val_c)})"
    )
    if nm_p is None:
        print(f"{prefix} pass ratio: None")
    else:
        # pass_ratio_neg is -pass_ratio; display as positive.
        print(
            f"{prefix} pass ratio: {nm_p} "
            f"(ratio={(-float(val_p)):.4f})"
        )
    if sweep.composite_best_name is not None:
        print(
            f"Composite tie-break (Pareto-internal hint): "
            f"{sweep.composite_best_name} "
            f"(score={float(sweep.composite_best_score):.6f})"
        )
        print(
            f"Composite ranking (top-down): "
            f"{list(sweep.composite_ranking_names)}"
        )
    else:
        print("Composite tie-break: None")
    print(
        "Note: \"best/least-bad by ...\" reports the candidate "
        "that minimizes the corresponding diagnostic in this "
        "surrogate sweep, not a safety recommendation."
    )

    # In-sample vs holdout top candidate divergence diagnostic.
    in_sample_top = None
    in_sample_best = None
    holdout_top = None
    holdout_best = None
    for e in sweep.entries:
        v_in = e.in_sample_diagnostics.worst_delta_max_temperature_k
        v_out = e.holdout_diagnostics.worst_delta_max_temperature_k
        if v_in is not None:
            if in_sample_best is None or float(v_in) < float(in_sample_best):
                in_sample_best = float(v_in)
                in_sample_top = e.candidate.name
        if v_out is not None:
            if holdout_best is None or float(v_out) < float(holdout_best):
                holdout_best = float(v_out)
                holdout_top = e.candidate.name
    print(
        f"In-sample top by max-T delta: {in_sample_top} "
        f"({_format_optional_float(in_sample_best)})"
    )
    print(
        f"Holdout top by max-T delta: {holdout_top} "
        f"({_format_optional_float(holdout_best)})"
    )
    if in_sample_top != holdout_top:
        print(
            "In-sample vs holdout top diverge: rankings depend "
            "on which condition subset is used."
        )
    else:
        print(
            "In-sample vs holdout top agree."
        )

    # Pass-ratio top vs worst-Δmax-T top divergence
    pass_top = nm_p
    delta_top = nm_t
    if pass_top is not None and delta_top is not None and pass_top != delta_top:
        print(
            "Pass-ratio top vs worst-delta-T top diverge: "
            f"pass_ratio_top={pass_top}, delta_max_T_top={delta_top}"
        )
    else:
        print(
            "Pass-ratio top vs worst-delta-T top agree."
        )

    checks: list[bool] = []
    checks.append(int(sweep.candidate_count) == len(candidates))
    checks.append(len(selections) > 0)
    checks.append(int(multi.aggregate_risk_map.active_count) > 0)
    checks.append(
        all(
            int(e.full_diagnostics.entry_count)
            == int(e.in_sample_diagnostics.entry_count)
            + int(e.holdout_diagnostics.entry_count)
            for e in sweep.entries
        )
    )
    checks.append(
        all(
            int(e.moved_vertex_count) > 0
            for e in sweep.entries
        )
    )
    ok = all(checks)
    print(f"Invariants: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
