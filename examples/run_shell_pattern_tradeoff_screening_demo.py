"""Synthetic shell pattern tradeoff screening demo.

Synthetic thermal-risk comparison and guardrail smoke check; not
a physical PET-bottle validation. Evaluates a small library of
outer-surface Gaussian dimple pattern candidates on the synthetic
cylindrical-shell fixture under empty-shell and water-filled-shell
4-interface presets, then reports thermal-risk deltas and
guardrail pass/fail / tradeoff label per candidate per fill state.

This is **not** an optimizer.
This is a coarse screening smoke check.
A candidate that reduces threshold-exceeded count but increases
peak / top-percent temperature is reported as a **mixed tradeoff**,
not as an improvement, and is excluded from the "passed" count
even when its threshold-area delta is favorable.

Why this exists
---------------
The previous outer-surface patterned-shell smoke run produced
distribution-sensitive but conflicting deltas (threshold-exceeded
count down, max temperature up). Reporting only one of those
deltas would be a research-logic error. This demo runs the
named multi-metric ``compare_angle_distance_thermal_risk_scans``
reduction with a conservative default
:class:`ThermalRiskGuardrailConfig` so that every candidate is
scored against multiple metrics and explicitly bucketed as
``passed`` / ``failed`` / ``mixed``.

Research framing (also enforced at runtime)
-------------------------------------------
- Guardrails are conservative diagnostic filters, **not** safety
  certification.
- ``passes_guardrails == True`` is **not** safety certification.
- Threshold values are illustrative unless calibrated by
  experiment.
- Delta values are comparison diagnostics only.
- A candidate that lowers threshold count but increases peak
  temperature is a mixed tradeoff, not an improvement.
- This demo does **not** select a manufacturing-ready pattern.
- This demo does **not** prove fire prevention or PET-bottle
  safety.

Limitations
-----------
- Synthetic cylindrical shell fixture; not a real PET bottle STL.
- Synthetic uniform RiskMap; not detector-derived caustic risk.
- Side-incidence ray grid; not solar geometry.
- 0-D lumped-capacitance per pixel; no spatial conduction, no
  spectral absorptivity, no view factor.
- Single-angle-single-distance side-incidence "scan" wrapped as
  an :class:`AngleDistanceThermalRiskScanResult` of length 1 per
  fill state. The angle/detector-z labels are synthetic
  placeholders chosen to match the side-incidence detector setup;
  they are **not** the angle-distance sweep used by the
  parallel-incidence demos.
- ``NOMINAL_INCIDENT_IRRADIANCE_W_M2 = 1000.0`` is illustrative
  only.
- No STL / OBJ / CAD / STEP export, no file write, no
  visualization, no CSV / Parquet logging.

Run from the repository root::

    python examples/run_shell_pattern_tradeoff_screening_demo.py

The script exits 0 when every invariant holds and 1 otherwise.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from optics_simulation.contribution.risk_map import RiskMap
from optics_simulation.geometry import (
    SyntheticShellVertexMasks,
    classify_synthetic_shell_vertices,
    create_subdivided_synthetic_bottle_shell,
    create_vertex_surface_coordinates,
)
from optics_simulation.metrics import (
    compute_relative_irradiance_surrogate,
)
from optics_simulation.optics import (
    RayBundle,
    accumulate_detector_hits,
    create_detector_grid,
    create_detector_plane,
    create_shell_medium_preset,
    intersect_detector_plane,
    make_ray_bundle,
    run_multi_step_trace,
)
from optics_simulation.pattern import (
    compute_vertex_displacement_amounts,
    create_displaced_mesh_copy,
    create_gaussian_dimple_pattern,
)
from optics_simulation.thermal import (
    AngleDistanceThermalRiskComparisonResult,
    AngleDistanceThermalRiskScanResult,
    PerAngleDistanceThermalRiskResult,
    ThermalRiskGuardrailConfig,
    compare_angle_distance_thermal_risk_scans,
    compute_thermal_risk_metrics,
    simulate_lumped_target_heating_map,
)


IOR_AIR = 1.00028
IOR_PET = 1.575
IOR_WATER = 1.333

OUTER_RADIUS = 30.0
WALL_THICKNESS = 1.0
HEIGHT = 120.0
SECTIONS = 96
HEIGHT_SEGMENTS = 24

RISK_RESOLUTION = (32, 64)  # (nv, nu)
ACTIVE_THRESHOLD = 0.01
BOUNDARY_EPSILON = 0.02
DISPLACEMENT_MODE = "inward"

ORIGIN_X = 100.0
Y_RANGE = (-50.0, 50.0)
Z_RANGE = (-70.0, 70.0)
RAY_NY = 11
RAY_NZ = 11

DETECTOR_CENTER = (-100.0, 0.0, 0.0)
DETECTOR_NORMAL = (1.0, 0.0, 0.0)
DETECTOR_UP = (0.0, 0.0, 1.0)
DETECTOR_WIDTH = 200.0
DETECTOR_HEIGHT = 200.0
DETECTOR_RESOLUTION = (40, 40)

NOMINAL_INCIDENT_IRRADIANCE_W_M2 = 1000.0
DURATION_S = 60.0
DT_S = 0.5
AREAL_HEAT_CAPACITY_J_M2K = 1200.0
ABSORPTIVITY = 0.8
H_CONV_W_M2K = 10.0
EMISSIVITY = 0.9
AMBIENT_TEMP_K = 293.15
THRESHOLD_TEMP_K = 373.15  # illustrative only; not a material ignition point
TOP_PERCENT_THERMAL = 1.0

VALID_TERMINATIONS = frozenset(
    {
        "completed_interfaces",
        "no_active_rays",
        "no_interfaces",
        "max_steps_reached",
    }
)

FILL_MEDIA = ("air", "water")
SCREENING_ANGLE_PLACEHOLDER = 0.0
SCREENING_DETECTOR_Z_PLACEHOLDER = float(DETECTOR_CENTER[0])


@dataclass(frozen=True)
class PatternSpec:
    name: str
    count: int
    amplitude: float
    sigma_u: float
    sigma_v: float
    max_depth: float
    seed: int


PATTERN_CANDIDATES: tuple[PatternSpec, ...] = (
    PatternSpec(
        name="shallow_broad",
        count=20, amplitude=1.0,
        sigma_u=0.06, sigma_v=0.06,
        max_depth=0.05, seed=11,
    ),
    PatternSpec(
        name="moderate_broad",
        count=20, amplitude=1.0,
        sigma_u=0.05, sigma_v=0.05,
        max_depth=0.10, seed=22,
    ),
    PatternSpec(
        name="current_default",
        count=20, amplitude=1.0,
        sigma_u=0.03, sigma_v=0.03,
        max_depth=0.25, seed=42,
    ),
    PatternSpec(
        name="sparse_deep",
        count=8, amplitude=1.0,
        sigma_u=0.03, sigma_v=0.03,
        max_depth=0.25, seed=7,
    ),
)


def _build_simple_risk_map(resolution: tuple[int, int]) -> RiskMap:
    nv, nu = int(resolution[0]), int(resolution[1])
    risk = np.ones((nv, nu), dtype=float)
    total = float(risk.sum())
    return RiskMap(
        risk_map=risk,
        probability_map=risk / total,
        active_mask=np.ones((nv, nu), dtype=bool),
        total_risk=total,
        active_count=nv * nu,
        epsilon=0.0,
        threshold=None,
    )


def _build_side_incidence_rays(
    *,
    origin_x: float,
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    ny: int,
    nz: int,
    direction: tuple[float, float, float] = (-1.0, 0.0, 0.0),
) -> RayBundle:
    ys = np.linspace(y_range[0], y_range[1], ny)
    zs = np.linspace(z_range[0], z_range[1], nz)
    yv, zv = np.meshgrid(ys, zs, indexing="xy")
    origins = np.column_stack([
        np.full(yv.size, float(origin_x)),
        yv.ravel(),
        zv.ravel(),
    ])
    directions = np.broadcast_to(
        np.asarray(direction, dtype=float).reshape(1, 3),
        origins.shape,
    ).copy()
    return make_ray_bundle(origins, directions)


def _wrap_single_entry_scan(
    *,
    metrics,
    detector_hits: int,
    final_ray_count: int,
    termination_reason: str,
) -> AngleDistanceThermalRiskScanResult:
    entry = PerAngleDistanceThermalRiskResult(
        angle_degrees=SCREENING_ANGLE_PLACEHOLDER,
        detector_z=SCREENING_DETECTOR_Z_PLACEHOLDER,
        optical_c99=0.0,
        optical_cmax=0.0,
        detector_hits=int(detector_hits),
        final_ray_count=int(final_ray_count),
        termination_reason=str(termination_reason),
        thermal_metrics=metrics,
    )
    return AngleDistanceThermalRiskScanResult(
        per_result=(entry,),
        result_count=1,
        max_temperature_angle=SCREENING_ANGLE_PLACEHOLDER,
        max_temperature_detector_z=SCREENING_DETECTOR_Z_PLACEHOLDER,
        max_temperature_k=float(metrics.max_temperature_k),
        max_temperature_rise_k=float(metrics.max_temperature_rise_k),
        max_top_percent_temperature_rise_angle=(
            SCREENING_ANGLE_PLACEHOLDER
        ),
        max_top_percent_temperature_rise_detector_z=(
            SCREENING_DETECTOR_Z_PLACEHOLDER
        ),
        max_top_percent_temperature_rise_k=float(
            metrics.top_percent_max_temperature_rise_k
        ),
        max_threshold_exceeded_count_angle=(
            SCREENING_ANGLE_PLACEHOLDER
        ),
        max_threshold_exceeded_count_detector_z=(
            SCREENING_DETECTOR_Z_PLACEHOLDER
        ),
        max_threshold_exceeded_count=int(
            metrics.threshold_exceeded_count
        ),
        top_percent=TOP_PERCENT_THERMAL,
    )


def _run_thermal_risk_scan(
    mesh,
    rays: RayBundle,
    interface_sequence: tuple[tuple[float, float], ...],
    source_area: float,
) -> tuple[AngleDistanceThermalRiskScanResult, int, int, str, float]:
    trace = run_multi_step_trace(
        mesh, rays, list(interface_sequence),
    )
    detector = create_detector_plane(
        center=DETECTOR_CENTER,
        normal=DETECTOR_NORMAL,
        up=DETECTOR_UP,
        width=DETECTOR_WIDTH,
        height=DETECTOR_HEIGHT,
    )
    detector_grid = create_detector_grid(
        width=DETECTOR_WIDTH,
        height=DETECTOR_HEIGHT,
        resolution=DETECTOR_RESOLUTION,
    )
    hits = intersect_detector_plane(trace.final_rays, detector)
    accum = accumulate_detector_hits(
        hits, detector_grid, weights=trace.final_ray_weights,
    )
    surrogate = compute_relative_irradiance_surrogate(
        accum,
        detector_grid,
        ray_count=int(rays.ray_count),
        source_area=float(source_area),
        incident_irradiance=1.0,
    )
    incident_flux_map = (
        float(NOMINAL_INCIDENT_IRRADIANCE_W_M2)
        * surrogate.relative_irradiance_map
    )
    heating_map = simulate_lumped_target_heating_map(
        incident_flux_map,
        duration_s=DURATION_S,
        dt_s=DT_S,
        areal_heat_capacity_j_m2k=AREAL_HEAT_CAPACITY_J_M2K,
        absorptivity=ABSORPTIVITY,
        h_conv_w_m2k=H_CONV_W_M2K,
        emissivity=EMISSIVITY,
        ambient_temp_k=AMBIENT_TEMP_K,
        threshold_temp_k=THRESHOLD_TEMP_K,
        top_percent=TOP_PERCENT_THERMAL,
    )
    metrics = compute_thermal_risk_metrics(
        heating_map,
        pixel_area=float(surrogate.pixel_area),
        top_percent=TOP_PERCENT_THERMAL,
    )
    scan = _wrap_single_entry_scan(
        metrics=metrics,
        detector_hits=int(accum.total_hits),
        final_ray_count=int(trace.final_rays.ray_count),
        termination_reason=str(trace.termination_reason),
    )
    return (
        scan,
        int(accum.total_hits),
        int(trace.final_rays.ray_count),
        str(trace.termination_reason),
        float(surrogate.pixel_area),
    )


def _build_displaced_mesh(
    shell_mesh,
    surface_map,
    shell_masks: SyntheticShellVertexMasks,
    spec: PatternSpec,
):
    risk_map = _build_simple_risk_map(RISK_RESOLUTION)
    pattern = create_gaussian_dimple_pattern(
        risk_map,
        count=int(spec.count),
        amplitude=float(spec.amplitude),
        sigma_u=float(spec.sigma_u),
        sigma_v=float(spec.sigma_v),
        max_depth=float(spec.max_depth),
        seed=int(spec.seed),
    )
    displacement = compute_vertex_displacement_amounts(
        shell_mesh,
        surface_map,
        pattern,
        active_threshold=ACTIVE_THRESHOLD,
        exclude_v_boundary_epsilon=BOUNDARY_EPSILON,
        include_mask=shell_masks.outer_lateral_interior_mask,
    )
    displaced_result = create_displaced_mesh_copy(
        shell_mesh,
        displacement,
        mode=DISPLACEMENT_MODE,
        process=False,
    )
    moved = int(displaced_result.moved_mask.sum())
    inner_displaced = int(
        (
            displaced_result.moved_mask & shell_masks.inner_wall_mask
        ).sum()
    )
    rim_displaced = int(
        (
            displaced_result.moved_mask & shell_masks.z_boundary_mask
        ).sum()
    )
    return displaced_result, moved, inner_displaced, rim_displaced


def _classify_overall_label(
    comparisons_by_fill: dict[
        str, AngleDistanceThermalRiskComparisonResult,
    ],
) -> str:
    has_mixed = any(
        cmp.mixed_count > 0
        for cmp in comparisons_by_fill.values()
    )
    if has_mixed:
        return "mixed"
    all_pass = all(
        cmp.fail_count == 0 and cmp.result_count > 0
        for cmp in comparisons_by_fill.values()
    )
    if all_pass:
        return "passed"
    return "failed"


def _format_optional_float(value, fmt: str = "+.4f") -> str:
    if value is None:
        return "None"
    return format(float(value), fmt)


def _print_static_header(
    rays: RayBundle,
    source_area: float,
    pixel_area: float,
    shell_masks: SyntheticShellVertexMasks,
    shell_watertight: bool,
) -> None:
    print("Shell pattern tradeoff screening demo")
    print(
        "Synthetic thermal-risk comparison and guardrail smoke "
        "check; not a physical PET-bottle validation."
    )
    print(
        "Note: guardrails are conservative diagnostic filters, not "
        "safety certification."
    )
    print(
        "Note: a candidate that lowers threshold count but "
        "increases peak temperature is a mixed tradeoff, not an "
        "improvement."
    )
    print(
        "Note: this is not optimization. This does not select a "
        "manufacturing-ready pattern."
    )
    print(
        "Note: threshold values are illustrative unless calibrated "
        "by experiment."
    )
    print(
        "Note: this does not prove fire prevention or PET-bottle "
        "safety."
    )
    print(
        "Note: no STL / OBJ / CAD / STEP export, no file write, no "
        "visualization is performed."
    )
    print(f"Mode: {DISPLACEMENT_MODE}")
    print(f"Candidate count: {len(PATTERN_CANDIDATES)}")
    print(
        f"Side-incidence ray grid: origin_x={ORIGIN_X}, "
        f"y_range={Y_RANGE}, z_range={Z_RANGE}, "
        f"ny={RAY_NY}, nz={RAY_NZ}"
    )
    print(f"Ray count: {int(rays.ray_count)}")
    print(f"Source area: {float(source_area):.4f}")
    print(
        f"Detector resolution: {DETECTOR_RESOLUTION[0]}"
        f"x{DETECTOR_RESOLUTION[1]}"
    )
    print(f"Detector pixel area: {float(pixel_area):.6f}")
    print(
        f"Outer interior eligible vertex count: "
        f"{int(shell_masks.outer_lateral_interior_mask.sum())}"
    )
    print(f"Shell watertight: {bool(shell_watertight)}")
    print(
        f"Nominal incident irradiance: "
        f"{float(NOMINAL_INCIDENT_IRRADIANCE_W_M2):.4f}"
    )
    print(f"Threshold temperature: {float(THRESHOLD_TEMP_K):.4f}")


def _print_candidate_block(
    spec: PatternSpec,
    moved: int,
    inner_displaced: int,
    rim_displaced: int,
    max_displacement: float,
    comparisons_by_fill: dict[
        str, AngleDistanceThermalRiskComparisonResult,
    ],
    overall_label: str,
) -> None:
    print(f"Candidate: {spec.name}")
    print(
        f"Pattern parameters: count={spec.count}, "
        f"amplitude={spec.amplitude}, sigma_u={spec.sigma_u}, "
        f"sigma_v={spec.sigma_v}, max_depth={spec.max_depth}, "
        f"seed={spec.seed}"
    )
    print(f"Moved vertex count: {int(moved)}")
    print(f"Inner wall displaced vertices: {int(inner_displaced)}")
    print(f"Rim displaced vertices: {int(rim_displaced)}")
    print(f"Max displacement: {float(max_displacement):.4f}")
    print(f"Overall screening label: {overall_label}")

    for medium in FILL_MEDIA:
        cmp = comparisons_by_fill[medium]
        per_entry = cmp.per_result[0].comparison
        print(f"Fill medium: {medium}")
        print(f"Tradeoff label: {per_entry.tradeoff_label}")
        print(f"Passes guardrails: {bool(per_entry.passes_guardrails)}")
        violations = (
            ", ".join(per_entry.guardrail_violations)
            if per_entry.guardrail_violations
            else "(none)"
        )
        print(f"Guardrail violations: {violations}")
        print(
            f"Delta max temperature: "
            f"{float(per_entry.delta_max_temperature_k):+.4f}"
        )
        print(
            f"Delta top-percent rise: "
            f"{float(per_entry.delta_top_percent_max_temperature_rise_k):+.4f}"
        )
        print(
            f"Delta threshold count: "
            f"{int(per_entry.delta_threshold_exceeded_count):+d}"
        )
        print(
            f"Delta threshold area: "
            f"{_format_optional_float(per_entry.delta_threshold_exceeded_area)}"
        )


def _check_invariants(
    rays: RayBundle,
    candidate_records: list[dict],
) -> bool:
    checks: list[bool] = []

    expected_ray_count = RAY_NY * RAY_NZ
    checks.append(int(rays.ray_count) == expected_ray_count)
    checks.append(len(candidate_records) == len(PATTERN_CANDIDATES))

    for record in candidate_records:
        checks.append(int(record["moved"]) > 0)
        checks.append(int(record["inner_displaced"]) == 0)
        checks.append(int(record["rim_displaced"]) == 0)
        comparisons = record["comparisons"]
        for medium in FILL_MEDIA:
            checks.append(medium in comparisons)
            cmp = comparisons[medium]
            checks.append(int(cmp.result_count) > 0)
            checks.append(
                int(cmp.pass_count) + int(cmp.fail_count)
                == int(cmp.result_count)
            )
            for p in cmp.per_result:
                comparison = p.comparison
                checks.append(np.isfinite(
                    float(comparison.delta_max_temperature_k)
                ))
                checks.append(np.isfinite(
                    float(
                        comparison.delta_top_percent_max_temperature_rise_k
                    )
                ))
                if (
                    comparison.delta_threshold_exceeded_area is not None
                ):
                    checks.append(np.isfinite(
                        float(
                            comparison.delta_threshold_exceeded_area
                        )
                    ))
        for medium in FILL_MEDIA:
            for kind in (
                "baseline_termination", "candidate_termination",
            ):
                checks.append(record[medium][kind] in VALID_TERMINATIONS)

    return all(checks)


def main() -> int:
    shell_mesh = create_subdivided_synthetic_bottle_shell(
        outer_radius=OUTER_RADIUS,
        wall_thickness=WALL_THICKNESS,
        height=HEIGHT,
        sections=SECTIONS,
        height_segments=HEIGHT_SEGMENTS,
    )
    surface_map = create_vertex_surface_coordinates(shell_mesh)
    shell_masks = classify_synthetic_shell_vertices(
        shell_mesh,
        outer_radius=OUTER_RADIUS,
        wall_thickness=WALL_THICKNESS,
        height=HEIGHT,
        boundary_epsilon=BOUNDARY_EPSILON,
    )

    rays = _build_side_incidence_rays(
        origin_x=ORIGIN_X,
        y_range=Y_RANGE,
        z_range=Z_RANGE,
        ny=RAY_NY,
        nz=RAY_NZ,
    )
    y_span = float(Y_RANGE[1] - Y_RANGE[0])
    z_span = float(Z_RANGE[1] - Z_RANGE[0])
    source_area = float(y_span * z_span)

    presets = {
        medium: create_shell_medium_preset(fill_medium=medium)
        for medium in FILL_MEDIA
    }

    baseline_scans: dict[str, dict] = {}
    for medium in FILL_MEDIA:
        scan, hits, final, termination, pixel_area = (
            _run_thermal_risk_scan(
                shell_mesh,
                rays,
                presets[medium].interface_sequence,
                source_area,
            )
        )
        baseline_scans[medium] = {
            "scan": scan,
            "hits": hits,
            "final": final,
            "termination": termination,
            "pixel_area": pixel_area,
        }

    pixel_area = float(baseline_scans[FILL_MEDIA[0]]["pixel_area"])

    _print_static_header(
        rays, source_area, pixel_area,
        shell_masks,
        shell_watertight=bool(shell_mesh.is_watertight),
    )

    guardrails = ThermalRiskGuardrailConfig()

    candidate_records: list[dict] = []
    for spec in PATTERN_CANDIDATES:
        displaced_result, moved, inner_displaced, rim_displaced = (
            _build_displaced_mesh(
                shell_mesh, surface_map, shell_masks, spec,
            )
        )
        comparisons: dict[
            str, AngleDistanceThermalRiskComparisonResult,
        ] = {}
        per_medium: dict[str, dict] = {}
        for medium in FILL_MEDIA:
            (
                candidate_scan, candidate_hits,
                candidate_final, candidate_termination, _,
            ) = _run_thermal_risk_scan(
                displaced_result.displaced_mesh,
                rays,
                presets[medium].interface_sequence,
                source_area,
            )
            comparison = compare_angle_distance_thermal_risk_scans(
                baseline_scans[medium]["scan"],
                candidate_scan,
                guardrails=guardrails,
            )
            comparisons[medium] = comparison
            per_medium[medium] = {
                "baseline_termination": str(
                    baseline_scans[medium]["termination"]
                ),
                "candidate_termination": str(candidate_termination),
            }
        overall_label = _classify_overall_label(comparisons)
        record = {
            "spec": spec,
            "moved": moved,
            "inner_displaced": inner_displaced,
            "rim_displaced": rim_displaced,
            "max_displacement": float(
                displaced_result.max_displacement
            ),
            "comparisons": comparisons,
            "overall_label": overall_label,
        }
        record.update(per_medium)
        candidate_records.append(record)

        _print_candidate_block(
            spec,
            moved, inner_displaced, rim_displaced,
            float(displaced_result.max_displacement),
            comparisons,
            overall_label,
        )

    passed = sum(
        1 for r in candidate_records if r["overall_label"] == "passed"
    )
    failed = sum(
        1 for r in candidate_records if r["overall_label"] == "failed"
    )
    mixed = sum(
        1 for r in candidate_records if r["overall_label"] == "mixed"
    )

    print("Screening summary")
    print(f"Passed candidates: {int(passed)}")
    print(f"Failed candidates: {int(failed)}")
    print(f"Mixed candidates: {int(mixed)}")
    print(
        "Note: Passed / Failed / Mixed counts are diagnostic "
        "buckets, not safety verdicts."
    )

    ok = _check_invariants(rays, candidate_records)
    print(f"Invariants: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
