"""Per-pixel thermal-map original-vs-displaced demo.

Synthetic per-pixel thermal-map surrogate; not an ignition
validation model. Wires together: subdivided synthetic bottle
mesh -> per-vertex normalized cylindrical (u, v) -> synthetic
uniform RiskMap -> Gaussian dimple pattern -> per-vertex
displacement amounts -> displaced mesh copy (mode=inward) ->
per-mesh angle-distance sweep with use_power_weights=True and
use_relative_irradiance=True -> per-pixel lumped target-heating
map per (angle, detector_z) entry on each mesh ->
distribution-sensitive Original vs Displaced comparison of the
thermal-map surrogates.

Why a thermal-map surrogate
---------------------------
The previous scalar coupling reduced each entry's relative
irradiance map to a single ``max_relative_irradiance`` scalar
before feeding the lumped-target heater. That collapsed the
spatial distribution of the optical caustic into one number, so
patterns that redistribute the caustic without changing its peak
showed identical thermal output. The per-pixel thermal-map
surrogate keeps the full 2D relative irradiance map and runs an
independent 0-D lumped-target heater **per pixel**, so the
original vs displaced comparison is sensitive to caustic
redistribution as well as to peak shifts.

Interpretation notes (also printed at runtime)
----------------------------------------------
- The thermal-map model consumes an optical relative irradiance
  map. The input is **not calibrated W/m^2** unless the caller
  supplies a calibrated incident irradiance.
- This is **not CFD**, **not pyrolysis chemistry**, **not
  spatial conduction**. Each pixel is treated as an independent
  lumped target.
- This does **not** prove fire prevention or PET-bottle safety.
- Delta values are reported for comparison only; a negative
  delta is **not required**. A smaller displaced value is
  **not** required.
- The threshold is illustrative, **not** a material ignition
  threshold.

Limitations
-----------
- Synthetic subdivided solid-cylinder fixture; not a real PET
  bottle STL.
- Synthetic uniform RiskMap; not detector-derived caustic risk.
- Single PET interface sequence ``[(AIR, PET), (PET, AIR)]``;
  no shell, no wall thickness, no fluid medium.
- 0-D lumped-capacitance per pixel; no spatial conduction, no
  spectral absorptivity, no view factor, no time-varying source.
- ``NOMINAL_INCIDENT_IRRADIANCE_W_M2 = 1000.0`` is an
  illustrative scale only. It does not anchor the result to any
  calibrated solar flux.
- No STL / OBJ / CAD / STEP export, no file write, no
  visualization, no CSV / Parquet logging.

Run from the repository root:

    python examples/run_thermal_map_original_vs_displaced_demo.py

The script exits 0 when every invariant holds and 1 otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from optics_simulation.angle_scan import (
    AngleDistanceScanResult,
    run_angle_distance_sweep,
)
from optics_simulation.contribution.risk_map import RiskMap
from optics_simulation.geometry import (
    create_subdivided_synthetic_bottle_body,
    create_vertex_surface_coordinates,
)
from optics_simulation.pattern import (
    compute_vertex_displacement_amounts,
    create_displaced_mesh_copy,
    create_gaussian_dimple_pattern,
)
from optics_simulation.thermal import (
    AngleDistanceHeatingMapScanResult,
    run_lumped_heating_maps_over_angle_distance_scan,
)


IOR_AIR = 1.00028
IOR_PET = 1.575
HOTSPOT_THRESHOLDS = (2.0, 5.0, 10.0)
TOP_PERCENT_FOR_C99 = 1.0
INCIDENT_REFERENCE = 1.0
INCIDENT_IRRADIANCE_NORMALIZATION = 1.0  # surrogate, unit-less

BOTTLE_RADIUS = 30.0
BOTTLE_HEIGHT = 120.0
BOTTLE_SECTIONS = 96
BOTTLE_HEIGHT_SEGMENTS = 24

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

ANGLES_DEGREES = (0.0, 15.0, 30.0)
DETECTOR_Z_VALUES = (-80.0, -100.0, -120.0)

RAY_GRID_NX = 11
RAY_GRID_NY = 11
DETECTOR_WIDTH = 200.0
DETECTOR_HEIGHT = 200.0
DETECTOR_RESOLUTION = (40, 40)
INTERFACE_SEQUENCE = [(IOR_AIR, IOR_PET), (IOR_PET, IOR_AIR)]
RAY_GRID_CONFIG = {
    "origin_plane_z": 200.0,
    "x_range": (-50.0, 50.0),
    "y_range": (-70.0, 70.0),
    "nx": RAY_GRID_NX,
    "ny": RAY_GRID_NY,
}

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
MAP_MODE = "relative_irradiance_map"

VALID_TERMINATIONS = frozenset(
    {
        "completed_interfaces",
        "no_active_rays",
        "no_interfaces",
        "max_steps_reached",
    }
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


def _run_optical_sweep(mesh) -> AngleDistanceScanResult:
    return run_angle_distance_sweep(
        mesh=mesh,
        angles_degrees=ANGLES_DEGREES,
        detector_z_values=DETECTOR_Z_VALUES,
        ray_grid_config=RAY_GRID_CONFIG,
        interface_sequence=INTERFACE_SEQUENCE,
        detector_width=DETECTOR_WIDTH,
        detector_height=DETECTOR_HEIGHT,
        detector_resolution=DETECTOR_RESOLUTION,
        incident_reference=INCIDENT_REFERENCE,
        thresholds=HOTSPOT_THRESHOLDS,
        top_percent=TOP_PERCENT_FOR_C99,
        use_power_weights=True,
        use_relative_irradiance=True,
        incident_irradiance=INCIDENT_IRRADIANCE_NORMALIZATION,
    )


def _run_thermal_maps(
    scan: AngleDistanceScanResult,
) -> AngleDistanceHeatingMapScanResult:
    return run_lumped_heating_maps_over_angle_distance_scan(
        scan,
        nominal_incident_irradiance_w_m2=NOMINAL_INCIDENT_IRRADIANCE_W_M2,
        duration_s=DURATION_S,
        dt_s=DT_S,
        areal_heat_capacity_j_m2k=AREAL_HEAT_CAPACITY_J_M2K,
        absorptivity=ABSORPTIVITY,
        h_conv_w_m2k=H_CONV_W_M2K,
        emissivity=EMISSIVITY,
        ambient_temp_k=AMBIENT_TEMP_K,
        threshold_temp_k=THRESHOLD_TEMP_K,
        top_percent=TOP_PERCENT_THERMAL,
        map_mode=MAP_MODE,
    )


def _fmt_or_none(value, fmt: str = "") -> str:
    if value is None:
        return "None"
    return format(float(value), fmt)


def _total_threshold_count(
    thermal: AngleDistanceHeatingMapScanResult,
) -> int:
    return int(
        sum(
            int(p.heating_map_result.threshold_exceeded_count)
            for p in thermal.per_result
        )
    )


def _print_summary(
    original_thermal: AngleDistanceHeatingMapScanResult,
    displaced_thermal: AngleDistanceHeatingMapScanResult,
    moved_count: int,
    max_displacement: float,
    vertices_mutated: bool,
    faces_mutated: bool,
) -> None:
    print("Thermal map original-vs-displaced demo")
    print(
        "Synthetic per-pixel thermal-map surrogate; not an "
        "ignition validation model."
    )
    print(
        "Note: the thermal-map model consumes an optical relative "
        "irradiance map. The input is not calibrated W/m^2 unless "
        "a calibrated incident irradiance is supplied."
    )
    print(
        "Note: this is not CFD, not pyrolysis chemistry, not "
        "spatial conduction. Each pixel is treated as an "
        "independent lumped target."
    )
    print(
        "Note: this does not prove fire prevention or PET-bottle "
        "safety."
    )
    print(
        "Note: Delta values are reported for comparison only; a "
        "negative delta is not required, and a smaller displaced "
        "value is not required either."
    )
    print(
        "Note: the threshold is illustrative, not a material "
        "ignition threshold."
    )
    print(
        "Note: no STL / OBJ / CAD / STEP export, no file write, no "
        "visualization is performed."
    )
    print(f"Mode: {DISPLACEMENT_MODE}")
    print(f"Angles: {', '.join(str(a) for a in ANGLES_DEGREES)}")
    print(
        "Detector z values: "
        + ", ".join(str(z) for z in DETECTOR_Z_VALUES)
    )
    print(f"Moved vertex count: {int(moved_count)}")
    print(f"Max displacement: {float(max_displacement):.4f}")
    print(
        f"Nominal incident irradiance: "
        f"{float(NOMINAL_INCIDENT_IRRADIANCE_W_M2):.4f}"
    )

    print(
        f"Original max thermal-map temperature: "
        f"{_fmt_or_none(original_thermal.max_temperature_k, '.4f')}"
    )
    print(
        f"Displaced max thermal-map temperature: "
        f"{_fmt_or_none(displaced_thermal.max_temperature_k, '.4f')}"
    )
    if (
        original_thermal.max_temperature_k is not None
        and displaced_thermal.max_temperature_k is not None
    ):
        d_max_t = (
            float(displaced_thermal.max_temperature_k)
            - float(original_thermal.max_temperature_k)
        )
        print(f"Delta max thermal-map temperature: {d_max_t:+.4f}")
    else:
        print("Delta max thermal-map temperature: None")

    print(
        f"Original max temperature rise: "
        f"{_fmt_or_none(original_thermal.max_temperature_rise_k, '+.4f')}"
    )
    print(
        f"Displaced max temperature rise: "
        f"{_fmt_or_none(displaced_thermal.max_temperature_rise_k, '+.4f')}"
    )
    if (
        original_thermal.max_temperature_rise_k is not None
        and displaced_thermal.max_temperature_rise_k is not None
    ):
        d_rise = (
            float(displaced_thermal.max_temperature_rise_k)
            - float(original_thermal.max_temperature_rise_k)
        )
        print(f"Delta max temperature rise: {d_rise:+.4f}")
    else:
        print("Delta max temperature rise: None")

    print(
        f"Original top-percent temperature rise: "
        f"{_fmt_or_none(original_thermal.max_top_percent_temperature_rise_k, '+.4f')}"
    )
    print(
        f"Displaced top-percent temperature rise: "
        f"{_fmt_or_none(displaced_thermal.max_top_percent_temperature_rise_k, '+.4f')}"
    )
    if (
        original_thermal.max_top_percent_temperature_rise_k is not None
        and displaced_thermal.max_top_percent_temperature_rise_k is not None
    ):
        d_top = (
            float(
                displaced_thermal.max_top_percent_temperature_rise_k
            )
            - float(
                original_thermal.max_top_percent_temperature_rise_k
            )
        )
        print(f"Delta top-percent temperature rise: {d_top:+.4f}")
    else:
        print("Delta top-percent temperature rise: None")

    orig_thr_count = _total_threshold_count(original_thermal)
    disp_thr_count = _total_threshold_count(displaced_thermal)
    print(f"Original threshold exceeded count: {orig_thr_count}")
    print(f"Displaced threshold exceeded count: {disp_thr_count}")
    print(
        f"Delta threshold exceeded count: "
        f"{(disp_thr_count - orig_thr_count):+d}"
    )

    print(
        f"Original max temperature angle: "
        f"{_fmt_or_none(original_thermal.max_temperature_angle, '')}"
    )
    print(
        f"Original max temperature detector z: "
        f"{_fmt_or_none(original_thermal.max_temperature_detector_z, '')}"
    )
    print(
        f"Displaced max temperature angle: "
        f"{_fmt_or_none(displaced_thermal.max_temperature_angle, '')}"
    )
    print(
        f"Displaced max temperature detector z: "
        f"{_fmt_or_none(displaced_thermal.max_temperature_detector_z, '')}"
    )

    print(
        "Per-result legend -> Original map max: per-entry max "
        "pixel of the lumped-target max temperature map on the "
        "original mesh. Displaced map max: same on the displaced "
        "mesh. Delta map max: Displaced minus Original."
    )
    for op, dp in zip(
        original_thermal.per_result, displaced_thermal.per_result,
    ):
        orig_map_max = float(
            op.heating_map_result.max_temperature_map_k.max()
        )
        disp_map_max = float(
            dp.heating_map_result.max_temperature_map_k.max()
        )
        delta_map_max = disp_map_max - orig_map_max
        print(
            f"Angle {op.angle_degrees} z={op.detector_z}: "
            f"Original map max={orig_map_max:.4f} "
            f"Displaced map max={disp_map_max:.4f} "
            f"Delta map max={delta_map_max:+.4f} "
            f"Original detector hits={int(op.detector_hits)} "
            f"Displaced detector hits={int(dp.detector_hits)} "
            f"termination={op.termination_reason}"
        )

    print(f"Mesh vertices mutated: {bool(vertices_mutated)}")
    print(f"Mesh faces mutated: {bool(faces_mutated)}")


def _check_invariants(
    original_scan: AngleDistanceScanResult,
    displaced_scan: AngleDistanceScanResult,
    original_thermal: AngleDistanceHeatingMapScanResult,
    displaced_thermal: AngleDistanceHeatingMapScanResult,
    moved_count: int,
    displaced_mesh_obj,
    original_mesh_obj,
    vertices_mutated: bool,
    faces_mutated: bool,
) -> bool:
    checks: list[bool] = []

    n_angles = len(ANGLES_DEGREES)
    n_dist = len(DETECTOR_Z_VALUES)
    expected_total = n_angles * n_dist

    checks.append(int(original_scan.angle_count) == n_angles)
    checks.append(int(displaced_scan.angle_count) == n_angles)
    checks.append(int(original_scan.detector_count) == n_dist)
    checks.append(int(displaced_scan.detector_count) == n_dist)
    checks.append(len(original_scan.per_result) == expected_total)
    checks.append(len(displaced_scan.per_result) == expected_total)

    checks.append(original_thermal.result_count == expected_total)
    checks.append(displaced_thermal.result_count == expected_total)

    for op, dp in zip(
        original_thermal.per_result, displaced_thermal.per_result,
    ):
        checks.append(op.angle_degrees == dp.angle_degrees)
        checks.append(op.detector_z == dp.detector_z)
        checks.append(op.termination_reason in VALID_TERMINATIONS)
        checks.append(dp.termination_reason in VALID_TERMINATIONS)

    for entry in original_scan.per_result:
        checks.append(entry.irradiance_surrogate is not None)
    for entry in displaced_scan.per_result:
        checks.append(entry.irradiance_surrogate is not None)

    for thermal in (original_thermal, displaced_thermal):
        for p in thermal.per_result:
            hm = p.heating_map_result
            checks.append(np.isfinite(hm.final_temperature_map_k).all())
            checks.append(np.isfinite(hm.max_temperature_map_k).all())
            checks.append(
                np.isfinite(hm.final_temperature_rise_map_k).all()
            )
            checks.append(
                np.isfinite(hm.max_temperature_rise_map_k).all()
            )
            checks.append(np.isfinite(hm.absorbed_flux_map_w_m2).all())
            checks.append(np.isfinite(hm.incident_flux_map_w_m2).all())
            checks.append((hm.max_temperature_map_k >= 0.0).all())
            checks.append(int(hm.threshold_exceeded_count) >= 0)
            ttt = hm.time_to_threshold_map_s
            checks.append(
                bool(np.all(np.isfinite(ttt) | np.isnan(ttt)))
            )

    for thermal in (original_thermal, displaced_thermal):
        if thermal.max_temperature_k is not None:
            checks.append(np.isfinite(float(thermal.max_temperature_k)))
            checks.append(float(thermal.max_temperature_k) >= 0.0)
        if thermal.max_top_percent_temperature_rise_k is not None:
            checks.append(
                np.isfinite(
                    float(thermal.max_top_percent_temperature_rise_k)
                )
            )

    checks.append(int(moved_count) > 0)
    checks.append(displaced_mesh_obj is not original_mesh_obj)
    checks.append(not bool(vertices_mutated))
    checks.append(not bool(faces_mutated))

    return all(checks)


def main() -> int:
    original_mesh = create_subdivided_synthetic_bottle_body(
        radius=BOTTLE_RADIUS,
        height=BOTTLE_HEIGHT,
        sections=BOTTLE_SECTIONS,
        height_segments=BOTTLE_HEIGHT_SEGMENTS,
    )
    surface_map = create_vertex_surface_coordinates(original_mesh)

    risk = _build_simple_risk_map(RISK_RESOLUTION)
    pattern = create_gaussian_dimple_pattern(
        risk,
        count=PATTERN_COUNT,
        amplitude=PATTERN_AMPLITUDE,
        sigma_u=PATTERN_SIGMA_U,
        sigma_v=PATTERN_SIGMA_V,
        max_depth=PATTERN_MAX_DEPTH,
        seed=PATTERN_SEED,
    )

    displacement = compute_vertex_displacement_amounts(
        original_mesh,
        surface_map,
        pattern,
        active_threshold=ACTIVE_THRESHOLD,
        exclude_v_boundary_epsilon=BOUNDARY_EPSILON,
    )

    vertices_before = np.array(original_mesh.vertices, copy=True)
    faces_before = np.array(original_mesh.faces, copy=True)

    displaced_result = create_displaced_mesh_copy(
        original_mesh,
        displacement,
        mode=DISPLACEMENT_MODE,
        process=False,
    )

    original_scan = _run_optical_sweep(original_mesh)
    displaced_scan = _run_optical_sweep(displaced_result.displaced_mesh)

    original_thermal = _run_thermal_maps(original_scan)
    displaced_thermal = _run_thermal_maps(displaced_scan)

    vertices_mutated = not np.array_equal(
        np.asarray(original_mesh.vertices), vertices_before
    )
    faces_mutated = not np.array_equal(
        np.asarray(original_mesh.faces), faces_before
    )

    moved_count = int(displaced_result.moved_mask.sum())
    max_displacement = float(displaced_result.max_displacement)

    _print_summary(
        original_thermal, displaced_thermal,
        moved_count, max_displacement,
        vertices_mutated, faces_mutated,
    )
    ok = _check_invariants(
        original_scan, displaced_scan,
        original_thermal, displaced_thermal,
        moved_count,
        displaced_result.displaced_mesh,
        original_mesh,
        vertices_mutated, faces_mutated,
    )
    print(f"Invariants: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
