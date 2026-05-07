"""Outer-surface patterned shell fill-state thermal-risk demo.

Synthetic outer-surface patterned shell smoke check; not a
physical PET-bottle validation. Wires together: synthetic
cylindrical shell fixture -> per-vertex shell vertex masks via
:func:`classify_synthetic_shell_vertices` -> per-vertex
normalized cylindrical (u, v) -> synthetic uniform RiskMap ->
Gaussian dimple pattern -> per-vertex displacement amounts with
``include_mask = outer_lateral_interior_mask`` (so only outer
lateral interior vertices are eligible for displacement) ->
displaced shell mesh copy (mode=inward) -> for each fill medium
in {empty, water-filled} run the same side-incidence
optical-to-thermal-risk chain as the previous shell demos on
both the original shell and the patterned shell, then print the
side-by-side thermal-risk comparison.

Why this exists
---------------
The previous shell fill-state demo compared solid / empty /
water-filled fixtures but did **not** apply any pattern to the
shell. The previous patterned demos pattern the entire mesh of a
solid cylinder. This demo restricts pattern displacement to the
synthetic shell's outer lateral interior vertices, so the
thermal-risk comparison reflects an outer-surface-only dimple
pattern with the inner wall and rims preserved exactly. It is
**still** synthetic: not a real PET STL, not a manufacturable
pattern, not automatic medium tracking, and not fire-prevention
validation.

Pattern displacement is applied only to synthetic outer shell
vertices selected by a mask. Inner wall and rim vertices are
intentionally not displaced. This is still a synthetic
cylindrical shell, not a real PET bottle STL. Interface sequences
are caller-provided presets, not automatic medium tracking.
Thermal metrics are surrogate metrics, not ignition validation.
Delta values are comparison diagnostics only; a negative delta is
**not required**, and a smaller patterned value is not required
either. Threshold values are illustrative unless calibrated by
experiment.

Limitations
-----------
- Synthetic cylindrical shell fixture; not a real PET bottle STL.
- Synthetic uniform RiskMap; not detector-derived caustic risk.
- Side-incidence ray grid; not solar geometry.
- 0-D lumped-capacitance per pixel; no spatial conduction, no
  spectral absorptivity, no view factor, no time-varying source.
- ``NOMINAL_INCIDENT_IRRADIANCE_W_M2 = 1000.0`` is an
  illustrative scale only. It does not anchor the result to any
  calibrated solar flux.
- Threshold ``THRESHOLD_TEMP_K = 373.15`` is illustrative, **not**
  a material ignition threshold.
- Interface sequences are caller-provided presets for the
  synthetic shell; this is **not** automatic medium tracking.
- No STL / OBJ / CAD / STEP export, no file write, no
  visualization, no CSV / Parquet logging.

Run from the repository root::

    python examples/run_patterned_shell_fill_state_thermal_risk_demo.py

The script exits 0 when every invariant holds and 1 otherwise.
"""
from __future__ import annotations

import sys
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
    DetectorIrradianceSurrogate,
    compute_relative_irradiance_surrogate,
)
from optics_simulation.optics import (
    RayBundle,
    ShellMediumTrackingResult,
    accumulate_detector_hits,
    create_detector_grid,
    create_detector_plane,
    create_shell_medium_preset,
    intersect_detector_plane,
    make_ray_bundle,
    run_multi_step_trace,
    run_surface_classified_shell_trace,
)
from optics_simulation.pattern import (
    compute_vertex_displacement_amounts,
    create_displaced_mesh_copy,
    create_gaussian_dimple_pattern,
)
from optics_simulation.thermal import (
    LumpedTargetHeatingMapResult,
    ThermalRiskMetrics,
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
PATTERN_COUNT = 20
PATTERN_AMPLITUDE = 1.0
PATTERN_SIGMA_U = 0.03
PATTERN_SIGMA_V = 0.03
PATTERN_MAX_DEPTH = 0.25
PATTERN_SEED = 42
ACTIVE_THRESHOLD = 0.01
BOUNDARY_EPSILON = 0.02
DISPLACEMENT_MODE = "inward"

ORIGIN_X = 100.0
SIDE_Y_RANGE = (-25.0, 25.0)
SIDE_Z_RANGE = (-50.0, 50.0)
Y_RANGE = SIDE_Y_RANGE
Z_RANGE = SIDE_Z_RANGE
RAY_NY = 11
RAY_NZ = 11
TRACE_EPSILON = 0.1
TRACE_RADIAL_TOLERANCE = 0.1
TRACE_Z_TOLERANCE = 1e-6

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


def _run_case(
    mesh,
    rays: RayBundle,
    interface_sequence: tuple[tuple[float, float], ...],
    source_area: float,
) -> tuple[
    int, int, str,
    DetectorIrradianceSurrogate,
    LumpedTargetHeatingMapResult,
    ThermalRiskMetrics,
]:
    trace = run_multi_step_trace(
        mesh, rays, list(interface_sequence),
        epsilon=TRACE_EPSILON,
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
    return (
        int(trace.final_rays.ray_count),
        int(accum.total_hits),
        str(trace.termination_reason),
        surrogate,
        heating_map,
        metrics,
    )


def _print_static_header(
    rays: RayBundle,
    source_area: float,
    pixel_area: float,
    masks: SyntheticShellVertexMasks,
    moved_count: int,
    inner_displaced: int,
    rim_displaced: int,
    max_displacement: float,
    shell_watertight: bool,
    vertices_mutated: bool,
    faces_mutated: bool,
) -> None:
    print("Patterned shell fill-state thermal risk demo")
    print(
        "Synthetic outer-surface patterned shell smoke check; "
        "not a physical PET-bottle validation."
    )
    print(
        "Note: pattern applied to outer shell surface only. Inner "
        "wall and rim vertices are intentionally not displaced."
    )
    print("Pattern applied to outer shell surface only.")
    print(
        "Note: this is still a synthetic cylindrical shell, not a "
        "real PET bottle STL."
    )
    print(
        "Note: interface sequences are caller-provided presets, "
        "not automatic medium tracking."
    )
    print(
        "Note: thermal metrics are surrogate metrics, not ignition "
        "validation. Threshold values are illustrative unless "
        "calibrated by experiment."
    )
    print(
        "Note: Delta values are comparison diagnostics only; a "
        "negative delta is not required, and a smaller patterned "
        "value is not required either."
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
        f"Outer wall vertex count: "
        f"{int(masks.outer_wall_mask.sum())}"
    )
    print(
        f"Inner wall vertex count: "
        f"{int(masks.inner_wall_mask.sum())}"
    )
    print(
        f"Z-boundary vertex count: "
        f"{int(masks.z_boundary_mask.sum())}"
    )
    print(
        f"Outer interior eligible vertex count: "
        f"{int(masks.outer_lateral_interior_mask.sum())}"
    )
    print(f"Moved vertex count: {int(moved_count)}")
    print(f"Max displacement: {float(max_displacement):.4f}")
    print(f"Inner wall displaced vertices: {int(inner_displaced)}")
    print(f"Rim displaced vertices: {int(rim_displaced)}")
    print(f"Shell watertight: {bool(shell_watertight)}")
    print(f"Mesh vertices mutated: {bool(vertices_mutated)}")
    print(f"Mesh faces mutated: {bool(faces_mutated)}")
    print(
        f"Nominal incident irradiance: "
        f"{float(NOMINAL_INCIDENT_IRRADIANCE_W_M2):.4f}"
    )


def _print_fill_block(
    fill_medium: str,
    original_metrics: ThermalRiskMetrics,
    patterned_metrics: ThermalRiskMetrics,
    original_hits: int,
    patterned_hits: int,
    original_final: int,
    patterned_final: int,
    original_termination: str,
    patterned_termination: str,
) -> None:
    print(f"Fill medium: {fill_medium}")
    print(
        f"Original shell termination: {original_termination}"
    )
    print(
        f"Patterned shell termination: {patterned_termination}"
    )
    print(f"Original final ray count: {int(original_final)}")
    print(f"Patterned final ray count: {int(patterned_final)}")
    print(f"Original detector hits: {int(original_hits)}")
    print(f"Patterned detector hits: {int(patterned_hits)}")
    print(
        f"Original shell max temperature: "
        f"{float(original_metrics.max_temperature_k):.4f}"
    )
    print(
        f"Patterned shell max temperature: "
        f"{float(patterned_metrics.max_temperature_k):.4f}"
    )
    delta_max_t = (
        float(patterned_metrics.max_temperature_k)
        - float(original_metrics.max_temperature_k)
    )
    print(f"Delta max temperature: {delta_max_t:+.4f}")
    print(
        f"Original shell max temperature rise: "
        f"{float(original_metrics.max_temperature_rise_k):+.4f}"
    )
    print(
        f"Patterned shell max temperature rise: "
        f"{float(patterned_metrics.max_temperature_rise_k):+.4f}"
    )
    delta_rise = (
        float(patterned_metrics.max_temperature_rise_k)
        - float(original_metrics.max_temperature_rise_k)
    )
    print(f"Delta max temperature rise: {delta_rise:+.4f}")
    print(
        f"Original shell top-percent temperature rise: "
        f"{float(original_metrics.top_percent_max_temperature_rise_k):+.4f}"
    )
    print(
        f"Patterned shell top-percent temperature rise: "
        f"{float(patterned_metrics.top_percent_max_temperature_rise_k):+.4f}"
    )
    delta_top = (
        float(patterned_metrics.top_percent_max_temperature_rise_k)
        - float(original_metrics.top_percent_max_temperature_rise_k)
    )
    print(f"Delta top-percent temperature rise: {delta_top:+.4f}")
    print(
        f"Original threshold exceeded count: "
        f"{int(original_metrics.threshold_exceeded_count)}"
    )
    print(
        f"Patterned threshold exceeded count: "
        f"{int(patterned_metrics.threshold_exceeded_count)}"
    )
    delta_count = (
        int(patterned_metrics.threshold_exceeded_count)
        - int(original_metrics.threshold_exceeded_count)
    )
    print(f"Delta threshold exceeded count: {delta_count:+d}")
    orig_area = original_metrics.threshold_exceeded_area
    patt_area = patterned_metrics.threshold_exceeded_area
    print(
        f"Original threshold exceeded area: "
        f"{float(orig_area):.6f}"
        if orig_area is not None else
        "Original threshold exceeded area: None"
    )
    print(
        f"Patterned threshold exceeded area: "
        f"{float(patt_area):.6f}"
        if patt_area is not None else
        "Patterned threshold exceeded area: None"
    )
    if orig_area is not None and patt_area is not None:
        delta_area = float(patt_area) - float(orig_area)
        print(f"Delta threshold exceeded area: {delta_area:+.6f}")
    else:
        print("Delta threshold exceeded area: None")


def _print_medium_tracking_block(
    *,
    air_tracking: ShellMediumTrackingResult,
    water_tracking: ShellMediumTrackingResult,
) -> None:
    print(
        "Note: shell medium preset validation is performed only "
        "within the paraxial synthetic side-incidence envelope."
    )
    print(f"Trace epsilon: {float(TRACE_EPSILON):.4f}")
    print(f"Validated side y range: {SIDE_Y_RANGE}")
    print(f"Validated side z range: {SIDE_Z_RANGE}")
    for fill_label, tracking in (
        ("air", air_tracking),
        ("water", water_tracking),
    ):
        print(f"Fill medium ({fill_label}) medium path: "
              + " -> ".join(tracking.medium_path))
        print(
            f"Fill medium ({fill_label}) medium tracking validation passed: "
            f"{bool(tracking.validation_passed)}"
        )
        print(
            f"Fill medium ({fill_label}) unexpected surface count: "
            f"{int(tracking.unexpected_surface_total)}"
        )
    print(
        f"Medium tracking validation passed: "
        f"{bool(air_tracking.validation_passed and water_tracking.validation_passed)}"
    )
    print(
        f"Unexpected surface count: "
        f"{int(air_tracking.unexpected_surface_total + water_tracking.unexpected_surface_total)}"
    )


def _check_invariants(
    rays: RayBundle,
    moved_count: int,
    inner_displaced: int,
    rim_displaced: int,
    vertices_mutated: bool,
    faces_mutated: bool,
    displaced_mesh_obj,
    original_mesh_obj,
    cases: list[tuple[
        int, int, str,
        int, int, str,
        ThermalRiskMetrics, ThermalRiskMetrics,
    ]],
    air_tracking: ShellMediumTrackingResult,
    water_tracking: ShellMediumTrackingResult,
) -> bool:
    checks: list[bool] = []

    expected_ray_count = RAY_NY * RAY_NZ
    checks.append(int(rays.ray_count) == expected_ray_count)

    checks.append(int(moved_count) > 0)
    checks.append(int(inner_displaced) == 0)
    checks.append(int(rim_displaced) == 0)

    checks.append(not bool(vertices_mutated))
    checks.append(not bool(faces_mutated))
    checks.append(displaced_mesh_obj is not original_mesh_obj)

    for (
        original_final, original_hits, original_termination,
        patterned_final, patterned_hits, patterned_termination,
        original_metrics, patterned_metrics,
    ) in cases:
        for final, hits, termination in (
            (original_final, original_hits, original_termination),
            (patterned_final, patterned_hits, patterned_termination),
        ):
            checks.append(0 <= int(final) <= int(rays.ray_count))
            checks.append(int(hits) >= 0)
            checks.append(int(hits) <= int(final))
            checks.append(termination in VALID_TERMINATIONS)

        for metrics in (original_metrics, patterned_metrics):
            checks.append(
                np.isfinite(float(metrics.max_temperature_k))
            )
            checks.append(
                np.isfinite(float(metrics.max_temperature_rise_k))
            )
            checks.append(
                np.isfinite(
                    float(metrics.top_percent_max_temperature_rise_k)
                )
            )
            checks.append(
                np.isfinite(float(metrics.final_temperature_k))
            )
            checks.append(int(metrics.threshold_exceeded_count) >= 0)
            checks.append(
                0.0 <= float(metrics.threshold_exceeded_fraction) <= 1.0
            )
            if metrics.threshold_exceeded_area is not None:
                checks.append(
                    float(metrics.threshold_exceeded_area) >= 0.0
                )

    for tracking in (air_tracking, water_tracking):
        checks.append(bool(tracking.validation_passed))
        checks.append(int(tracking.unexpected_surface_total) == 0)

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

    risk_map = _build_simple_risk_map(RISK_RESOLUTION)
    pattern = create_gaussian_dimple_pattern(
        risk_map,
        count=PATTERN_COUNT,
        amplitude=PATTERN_AMPLITUDE,
        sigma_u=PATTERN_SIGMA_U,
        sigma_v=PATTERN_SIGMA_V,
        max_depth=PATTERN_MAX_DEPTH,
        seed=PATTERN_SEED,
    )

    displacement = compute_vertex_displacement_amounts(
        shell_mesh,
        surface_map,
        pattern,
        active_threshold=ACTIVE_THRESHOLD,
        exclude_v_boundary_epsilon=BOUNDARY_EPSILON,
        include_mask=shell_masks.outer_lateral_interior_mask,
    )

    vertices_before = np.array(shell_mesh.vertices, copy=True)
    faces_before = np.array(shell_mesh.faces, copy=True)

    displaced_result = create_displaced_mesh_copy(
        shell_mesh,
        displacement,
        mode=DISPLACEMENT_MODE,
        process=False,
    )

    moved_count = int(displaced_result.moved_mask.sum())
    max_displacement = float(displaced_result.max_displacement)
    inner_displaced = int(
        (displaced_result.moved_mask & shell_masks.inner_wall_mask).sum()
    )
    rim_displaced = int(
        (displaced_result.moved_mask & shell_masks.z_boundary_mask).sum()
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

    cases: dict[str, tuple] = {}
    for medium in FILL_MEDIA:
        original_case = _run_case(
            shell_mesh,
            rays,
            presets[medium].interface_sequence,
            source_area,
        )
        patterned_case = _run_case(
            displaced_result.displaced_mesh,
            rays,
            presets[medium].interface_sequence,
            source_area,
        )
        cases[medium] = (original_case, patterned_case)

    pixel_area = float(cases[FILL_MEDIA[0]][0][3].pixel_area)

    vertices_mutated = not np.array_equal(
        np.asarray(shell_mesh.vertices), vertices_before
    )
    faces_mutated = not np.array_equal(
        np.asarray(shell_mesh.faces), faces_before
    )

    _print_static_header(
        rays, source_area, pixel_area,
        shell_masks,
        moved_count, inner_displaced, rim_displaced,
        max_displacement,
        shell_watertight=bool(shell_mesh.is_watertight),
        vertices_mutated=vertices_mutated,
        faces_mutated=faces_mutated,
    )
    for medium in FILL_MEDIA:
        original_case, patterned_case = cases[medium]
        (
            original_final, original_hits, original_termination,
            _, _, original_metrics,
        ) = original_case
        (
            patterned_final, patterned_hits, patterned_termination,
            _, _, patterned_metrics,
        ) = patterned_case
        _print_fill_block(
            medium,
            original_metrics, patterned_metrics,
            original_hits, patterned_hits,
            original_final, patterned_final,
            original_termination, patterned_termination,
        )

    air_tracking = run_surface_classified_shell_trace(
        shell_mesh, rays,
        fill_medium="air",
        outer_radius=OUTER_RADIUS,
        wall_thickness=WALL_THICKNESS,
        height=HEIGHT,
        radial_tolerance=TRACE_RADIAL_TOLERANCE,
        z_tolerance=TRACE_Z_TOLERANCE,
        epsilon=TRACE_EPSILON,
    )
    water_tracking = run_surface_classified_shell_trace(
        shell_mesh, rays,
        fill_medium="water",
        outer_radius=OUTER_RADIUS,
        wall_thickness=WALL_THICKNESS,
        height=HEIGHT,
        radial_tolerance=TRACE_RADIAL_TOLERANCE,
        z_tolerance=TRACE_Z_TOLERANCE,
        epsilon=TRACE_EPSILON,
    )
    _print_medium_tracking_block(
        air_tracking=air_tracking,
        water_tracking=water_tracking,
    )

    invariant_cases = []
    for medium in FILL_MEDIA:
        original_case, patterned_case = cases[medium]
        invariant_cases.append((
            int(original_case[0]), int(original_case[1]), str(original_case[2]),
            int(patterned_case[0]), int(patterned_case[1]), str(patterned_case[2]),
            original_case[5], patterned_case[5],
        ))

    ok = _check_invariants(
        rays,
        moved_count, inner_displaced, rim_displaced,
        vertices_mutated, faces_mutated,
        displaced_result.displaced_mesh,
        shell_mesh,
        invariant_cases,
        air_tracking=air_tracking,
        water_tracking=water_tracking,
    )
    print(f"Invariants: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
