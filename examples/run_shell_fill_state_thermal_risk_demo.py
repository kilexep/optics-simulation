"""Synthetic shell fill-state thermal-risk smoke demo.

Synthetic shell/fill-state optical-to-thermal smoke check; not a
physical PET-bottle validation. Wires together: synthetic
solid-cylinder body and synthetic cylindrical-shell fixture ->
caller-provided side-incidence ray grid (-x direction) ->
multi-step trace with caller-provided 2- or 4-interface
sequences -> detector plane intersection on the -x side ->
detector pixel-grid accumulation with cumulative Fresnel
transmission weighting -> source-area / ray-count /
detector-pixel-area normalization to a relative irradiance
surrogate -> per-pixel lumped target-heating map ->
:class:`ThermalRiskMetrics` reduction -> side-by-side comparison
of three cases:

- ``Solid cylinder``: ``[(air, PET), (PET, air)]``
- ``Empty shell``: ``create_shell_medium_preset(fill_medium="air")``
- ``Water-filled shell``:
  ``create_shell_medium_preset(fill_medium="water")``

Why this exists
---------------
The previous original-vs-displaced demos all use a solid-cylinder
surrogate and a 2-interface ``air -> PET -> air`` path. This demo
plugs the closed hollow shell fixture and the 4-interface fill-
state presets into the same optical-to-thermal-risk chain so that
caustic and thermal-risk metrics can be compared between solid /
empty / water-filled cases. It is a side-incidence synthetic
smoke check; it is **not** automatic medium tracking, **not** a
real PET-bottle STL, and **not** a fire-prevention validation.

Interpretation notes (also printed at runtime)
----------------------------------------------
- The shell is a synthetic cylindrical shell fixture, not a real
  PET bottle. It approximates wall thickness and fill-state
  interfaces only.
- It does not model neck, shoulder, base petaloid geometry,
  labels, caps, seams, or manufacturing defects.
- Interface sequences are caller-provided presets for this
  synthetic fixture; this is not automatic medium tracking.
- Thermal metrics are surrogate metrics, not ignition validation.
- Threshold values are illustrative unless calibrated by
  experiment.
- Delta values are comparison diagnostics only; do not claim
  water-filled shell is safer or riskier unless calibrated.
- This does not prove fire prevention or PET-bottle safety.

Limitations
-----------
- Synthetic solid cylinder and synthetic shell fixtures only;
  not real PET bottle STLs.
- 0-D lumped-capacitance per pixel; no spatial conduction, no
  spectral absorptivity, no view factor, no time-varying source.
- ``NOMINAL_INCIDENT_IRRADIANCE_W_M2 = 1000.0`` is an
  illustrative scale only. It does not anchor the result to any
  calibrated solar flux.
- No STL / OBJ / CAD / STEP export, no file write, no
  visualization, no CSV / Parquet logging.

Run from the repository root::

    python examples/run_shell_fill_state_thermal_risk_demo.py

The script exits 0 when every invariant holds and 1 otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from optics_simulation.geometry import (
    create_subdivided_synthetic_bottle_body,
    create_subdivided_synthetic_bottle_shell,
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

SOLID_INTERFACES = ((IOR_AIR, IOR_PET), (IOR_PET, IOR_AIR))

VALID_TERMINATIONS = frozenset(
    {
        "completed_interfaces",
        "no_active_rays",
        "no_interfaces",
        "max_steps_reached",
    }
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
    # Headline labels (no fill prefix) for summary parsers.
    print(
        f"Medium tracking validation passed: "
        f"{bool(air_tracking.validation_passed and water_tracking.validation_passed)}"
    )
    print(
        f"Unexpected surface count: "
        f"{int(air_tracking.unexpected_surface_total + water_tracking.unexpected_surface_total)}"
    )


def _print_summary(
    rays: RayBundle,
    source_area: float,
    pixel_area: float,
    solid_metrics: ThermalRiskMetrics,
    empty_metrics: ThermalRiskMetrics,
    water_metrics: ThermalRiskMetrics,
    solid_hits: int,
    empty_hits: int,
    water_hits: int,
    solid_final: int,
    empty_final: int,
    water_final: int,
    solid_termination: str,
    empty_termination: str,
    water_termination: str,
    shell_watertight: bool,
) -> None:
    print("Shell fill-state thermal risk demo")
    print(
        "Synthetic shell/fill-state optical-to-thermal smoke "
        "check; not a physical PET-bottle validation."
    )
    print(
        "Note: the shell is a synthetic cylindrical shell fixture, "
        "not a real PET bottle. It approximates wall thickness and "
        "fill-state interfaces only."
    )
    print(
        "Note: it does not model neck, shoulder, base petaloid "
        "geometry, labels, caps, seams, or manufacturing defects."
    )
    print(
        "Note: interface sequences are caller-provided presets for "
        "this synthetic fixture; this is not automatic medium "
        "tracking."
    )
    print(
        "Note: thermal metrics are surrogate metrics, not ignition "
        "validation. Threshold values are illustrative unless "
        "calibrated by experiment."
    )
    print(
        "Note: Delta values are comparison diagnostics only; do "
        "not claim water-filled shell is safer or riskier unless "
        "calibrated."
    )
    print(
        "Note: this does not prove fire prevention or PET-bottle "
        "safety."
    )
    print(
        "Note: no STL / OBJ / CAD / STEP export, no file write, no "
        "visualization is performed."
    )
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
        f"Nominal incident irradiance: "
        f"{float(NOMINAL_INCIDENT_IRRADIANCE_W_M2):.4f}"
    )
    print(f"Shell watertight: {bool(shell_watertight)}")

    print(f"Solid cylinder termination: {solid_termination}")
    print(f"Empty shell termination: {empty_termination}")
    print(f"Water-filled shell termination: {water_termination}")

    print(f"Solid final ray count: {int(solid_final)}")
    print(f"Empty shell final ray count: {int(empty_final)}")
    print(f"Water shell final ray count: {int(water_final)}")

    print(f"Solid detector hits: {int(solid_hits)}")
    print(f"Empty shell detector hits: {int(empty_hits)}")
    print(f"Water shell detector hits: {int(water_hits)}")

    print(
        f"Solid max temperature: "
        f"{float(solid_metrics.max_temperature_k):.4f}"
    )
    print(
        f"Empty shell max temperature: "
        f"{float(empty_metrics.max_temperature_k):.4f}"
    )
    print(
        f"Water shell max temperature: "
        f"{float(water_metrics.max_temperature_k):.4f}"
    )

    print(
        f"Solid max temperature rise: "
        f"{float(solid_metrics.max_temperature_rise_k):+.4f}"
    )
    print(
        f"Empty shell max temperature rise: "
        f"{float(empty_metrics.max_temperature_rise_k):+.4f}"
    )
    print(
        f"Water shell max temperature rise: "
        f"{float(water_metrics.max_temperature_rise_k):+.4f}"
    )

    print(
        f"Solid top-percent temperature rise: "
        f"{float(solid_metrics.top_percent_max_temperature_rise_k):+.4f}"
    )
    print(
        f"Empty shell top-percent temperature rise: "
        f"{float(empty_metrics.top_percent_max_temperature_rise_k):+.4f}"
    )
    print(
        f"Water shell top-percent temperature rise: "
        f"{float(water_metrics.top_percent_max_temperature_rise_k):+.4f}"
    )

    print(
        f"Solid threshold exceeded count: "
        f"{int(solid_metrics.threshold_exceeded_count)}"
    )
    print(
        f"Empty shell threshold exceeded count: "
        f"{int(empty_metrics.threshold_exceeded_count)}"
    )
    print(
        f"Water shell threshold exceeded count: "
        f"{int(water_metrics.threshold_exceeded_count)}"
    )


def _check_invariants(
    rays: RayBundle,
    solid_metrics: ThermalRiskMetrics,
    empty_metrics: ThermalRiskMetrics,
    water_metrics: ThermalRiskMetrics,
    solid_hits: int,
    empty_hits: int,
    water_hits: int,
    solid_final: int,
    empty_final: int,
    water_final: int,
    solid_termination: str,
    empty_termination: str,
    water_termination: str,
    shell_watertight: bool,
    air_tracking: ShellMediumTrackingResult,
    water_tracking: ShellMediumTrackingResult,
) -> bool:
    checks: list[bool] = []

    expected_ray_count = RAY_NY * RAY_NZ
    checks.append(int(rays.ray_count) == expected_ray_count)

    for final, hits, termination in (
        (solid_final, solid_hits, solid_termination),
        (empty_final, empty_hits, empty_termination),
        (water_final, water_hits, water_termination),
    ):
        checks.append(0 <= int(final) <= int(rays.ray_count))
        checks.append(int(hits) >= 0)
        checks.append(int(hits) <= int(final))
        checks.append(termination in VALID_TERMINATIONS)

    for metrics in (solid_metrics, empty_metrics, water_metrics):
        checks.append(np.isfinite(float(metrics.max_temperature_k)))
        checks.append(
            np.isfinite(float(metrics.max_temperature_rise_k))
        )
        checks.append(
            np.isfinite(
                float(metrics.top_percent_max_temperature_rise_k)
            )
        )
        checks.append(np.isfinite(float(metrics.final_temperature_k)))
        checks.append(int(metrics.threshold_exceeded_count) >= 0)
        checks.append(
            0.0 <= float(metrics.threshold_exceeded_fraction) <= 1.0
        )
        if metrics.threshold_exceeded_area is not None:
            checks.append(float(metrics.threshold_exceeded_area) >= 0.0)

    checks.append(bool(shell_watertight))

    for tracking in (air_tracking, water_tracking):
        checks.append(bool(tracking.validation_passed))
        checks.append(int(tracking.unexpected_surface_total) == 0)

    return all(checks)


def main() -> int:
    solid_mesh = create_subdivided_synthetic_bottle_body(
        radius=OUTER_RADIUS,
        height=HEIGHT,
        sections=SECTIONS,
        height_segments=HEIGHT_SEGMENTS,
    )
    shell_mesh = create_subdivided_synthetic_bottle_shell(
        outer_radius=OUTER_RADIUS,
        wall_thickness=WALL_THICKNESS,
        height=HEIGHT,
        sections=SECTIONS,
        height_segments=HEIGHT_SEGMENTS,
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

    empty_preset = create_shell_medium_preset(fill_medium="air")
    water_preset = create_shell_medium_preset(fill_medium="water")

    (
        solid_final, solid_hits, solid_termination,
        solid_surrogate, _, solid_metrics,
    ) = _run_case(solid_mesh, rays, SOLID_INTERFACES, source_area)
    (
        empty_final, empty_hits, empty_termination,
        empty_surrogate, _, empty_metrics,
    ) = _run_case(
        shell_mesh, rays, empty_preset.interface_sequence, source_area,
    )
    (
        water_final, water_hits, water_termination,
        water_surrogate, _, water_metrics,
    ) = _run_case(
        shell_mesh, rays, water_preset.interface_sequence, source_area,
    )

    pixel_area = float(solid_surrogate.pixel_area)

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

    _print_summary(
        rays, source_area, pixel_area,
        solid_metrics, empty_metrics, water_metrics,
        solid_hits, empty_hits, water_hits,
        solid_final, empty_final, water_final,
        solid_termination, empty_termination, water_termination,
        shell_watertight=bool(shell_mesh.is_watertight),
    )
    _print_medium_tracking_block(
        air_tracking=air_tracking,
        water_tracking=water_tracking,
    )
    ok = _check_invariants(
        rays,
        solid_metrics, empty_metrics, water_metrics,
        solid_hits, empty_hits, water_hits,
        solid_final, empty_final, water_final,
        solid_termination, empty_termination, water_termination,
        shell_watertight=bool(shell_mesh.is_watertight),
        air_tracking=air_tracking,
        water_tracking=water_tracking,
    )
    print(f"Invariants: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
