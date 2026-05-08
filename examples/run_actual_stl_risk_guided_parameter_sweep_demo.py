"""Actual STL risk-guided pattern parameter sweep demo.

Actual STL risk-guided pattern parameter sweep smoke check;
not a physical PET-bottle validation. Loads the supplied STL,
builds a multi-condition aggregate hotspot risk map (the same
diagnostic field used in
``run_actual_stl_multi_condition_risk_guided_pattern_demo.py``),
then sweeps a small list of risk-guided pattern parameter
candidates against the baseline thermal-risk scan. Each candidate
shares the aggregate risk map and the body include mask but has
its own ``pattern_count`` / ``sigma_u`` / ``sigma_v`` /
``max_depth`` / ``seed``. Per-candidate results are printed as a
diagnostic table; the aggregate "best by ..." pointer reports the
candidate that minimizes the corresponding delta diagnostic in
this synthetic sweep.

Old H.max raw detector-bin count is **NOT** used as a primary
metric. Canonical reductions are the relative irradiance
surrogate, ``C99``, and the per-pixel thermal-risk metrics.

Scope (also enforced at runtime)
--------------------------------
- The sweep uses a multi-condition hotspot-backtracked aggregate
  risk map.
- The sweep is diagnostic, not optimization.
- A candidate can reduce one metric while increasing another.
- Passing a guardrail is not safety certification.
- A negative delta is not required.
- Do not claim fire prevention or PET-bottle safety.
- Do not rank candidates as manufacturing-ready patterns.

Usage::

    python examples/run_actual_stl_risk_guided_parameter_sweep_demo.py \\
        --mesh data/raw/stl/pet_bottle.stl \\
        --target-height 225.6 --target-diameter 72.1 \\
        --wall-thickness 0.3 --inner-offset-mode auto \\
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
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optics_simulation.contribution import (
    build_multi_condition_legacy_hotspot_contribution_map,
    select_legacy_worst_conditions,
)
from optics_simulation.geometry import (
    create_actual_bottle_body_vertex_mask,
    load_mesh,
)
from optics_simulation.optics import (
    RiskGuidedPatternCandidateSpec,
    create_legacy_experiment_schedule,
    create_legacy_pet_water_trace_setup,
    run_actual_stl_risk_guided_pattern_parameter_sweep,
    run_legacy_pet_water_angle_distance_scan,
)
from optics_simulation.thermal import (
    compute_thermal_risk_over_legacy_optical_scan,
)


_DEFAULT_MESH_PATH = "data/raw/stl/pet_bottle.stl"


@dataclass(frozen=True)
class _LocalCandidate:
    name: str
    count: int
    sigma: float
    depth: float
    seed: int


_CANDIDATES: tuple[_LocalCandidate, ...] = (
    _LocalCandidate("rg_shallow_broad_01",
                    count=20, sigma=0.08, depth=0.02, seed=1),
    _LocalCandidate("rg_shallow_broad_02",
                    count=20, sigma=0.06, depth=0.05, seed=2),
    _LocalCandidate("rg_shallow_medium",
                    count=20, sigma=0.04, depth=0.05, seed=3),
    _LocalCandidate("rg_moderate_broad",
                    count=20, sigma=0.06, depth=0.10, seed=4),
    _LocalCandidate("rg_moderate_medium",
                    count=20, sigma=0.04, depth=0.10, seed=5),
    _LocalCandidate("rg_moderate_narrow",
                    count=20, sigma=0.02, depth=0.10, seed=6),
    _LocalCandidate("rg_sparse_shallow",
                    count=8, sigma=0.06, depth=0.05, seed=7),
    _LocalCandidate("rg_current_default",
                    count=20, sigma=0.03, depth=0.05, seed=42),
)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Actual STL risk-guided pattern parameter sweep smoke "
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
    print("Actual STL risk-guided parameter sweep demo")
    print(
        "Actual STL risk-guided pattern parameter sweep smoke "
        "check; not a physical PET-bottle validation."
    )
    print(
        "Note: the sweep uses a multi-condition hotspot-backtracked "
        "aggregate risk map."
    )
    print(
        "Note: the sweep is diagnostic, not optimization."
    )
    print(
        "Note: a candidate can reduce one metric while increasing "
        "another."
    )
    print(
        "Note: passing a guardrail is not safety certification."
    )
    print(
        "Note: delta values are comparison diagnostics only; a "
        "negative delta is not required."
    )
    print(
        "Note: do not claim fire prevention or PET-bottle safety."
    )
    print(
        "Note: do not rank candidates as manufacturing-ready "
        "patterns."
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
    print(f"Candidate count: {len(_CANDIDATES)}")
    print(f"Max conditions: {int(args.max_conditions)}")

    mesh = load_mesh(Path(args.mesh))
    original_setup = create_legacy_pet_water_trace_setup(
        mesh,
        target_height=float(args.target_height),
        target_diameter=float(args.target_diameter),
        wall_thickness=float(args.wall_thickness),
        inner_offset_mode=str(args.inner_offset_mode),
    )

    body_mask = create_actual_bottle_body_vertex_mask(
        original_setup.shell_mesh,
    )
    print(
        f"Body mask selected vertices: "
        f"{int(body_mask.selected_count)} / "
        f"{int(body_mask.vertex_count)}"
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
        setup=original_setup, **scan_kwargs,
    )
    baseline_thermal = compute_thermal_risk_over_legacy_optical_scan(
        baseline_optical, **thermal_kwargs,
    )

    selections = select_legacy_worst_conditions(
        baseline_optical, baseline_thermal,
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
        setup=original_setup,
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
            "Aggregate risk active count is 0; risk-guided sweep "
            "skipped."
        )
        print("Invariants: FAIL")
        return 1

    cand_specs = tuple(
        RiskGuidedPatternCandidateSpec(
            name=str(c.name),
            pattern_count=int(c.count),
            sigma_u=float(c.sigma),
            sigma_v=float(c.sigma),
            max_depth=float(c.depth),
            seed=int(c.seed),
        )
        for c in _CANDIDATES
    )

    sweep = run_actual_stl_risk_guided_pattern_parameter_sweep(
        original_setup=original_setup,
        baseline_thermal_scan=baseline_thermal,
        aggregate_risk_map=multi.aggregate_risk_map,
        body_include_mask=body_mask.include_mask,
        candidates=cand_specs,
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
    )

    for entry in sweep.entries:
        print(f"Candidate: {entry.candidate.name}")
        print(
            f"  Moved vertices: {int(entry.moved_vertex_count)}"
        )
        print(
            f"  Worst delta max temperature: "
            f"{_format_optional_float(entry.worst_delta_max_temperature_k)}"
        )
        print(
            f"  Best delta threshold count: "
            f"{_format_optional_int(entry.best_delta_threshold_count)}"
        )
        print(f"  Pass count: {int(entry.pass_count)}")
        print(f"  Fail count: {int(entry.fail_count)}")
        print(f"  Mixed count: {int(entry.mixed_count)}")

    print(f"Pass candidate count: {int(sweep.pass_candidate_count)}")
    print(
        f"Mixed candidate count: "
        f"{int(sweep.mixed_candidate_count)}"
    )
    print(
        f"Best by max-temperature delta: "
        f"{sweep.best_candidate_by_max_temperature_delta} "
        f"(delta={_format_optional_float(sweep.best_delta_max_temperature_k)})"
    )
    print(
        f"Best by threshold-count delta: "
        f"{sweep.best_candidate_by_threshold_count_delta} "
        f"(delta={_format_optional_int(sweep.best_delta_threshold_count)})"
    )
    print(
        "Note: \"best by ...\" reports the candidate that "
        "minimizes the corresponding delta diagnostic in this "
        "synthetic sweep, not a safety recommendation."
    )

    checks: list[bool] = []
    checks.append(int(sweep.candidate_count) == len(_CANDIDATES))
    checks.append(len(selections) > 0)
    checks.append(
        int(multi.aggregate_risk_map.active_count) >= 0
    )
    checks.append(
        all(
            int(e.moved_vertex_count) > 0 for e in sweep.entries
        )
    )
    ok = all(checks)
    print(f"Invariants: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
