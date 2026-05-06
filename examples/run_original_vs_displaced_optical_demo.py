"""Original vs displaced optical demo.

Synthetic original-vs-displaced optical comparison smoke check;
not a physical PET-bottle validation. Wires together: subdivided
synthetic bottle mesh -> per-vertex normalized cylindrical (u, v)
-> synthetic uniform RiskMap -> Gaussian dimple pattern -> per
vertex displacement amounts -> displaced mesh copy
(mode=inward) -> baseline angle scan on the original mesh ->
baseline angle scan on the displaced mesh under **identical**
ray, detector, interface, and metric settings -> per-angle
console comparison + max-C99 angle for each mesh + invariant
check. **Original mesh vertices and faces are not modified.**

Original and displaced meshes are scanned with identical ray
grid, interface sequence, detector plane, detector grid,
incident reference, threshold tuple, and ``top_percent``
configurations. The only varied input across the two scans is
the mesh itself.

``Delta C99 = displaced.c99 - original.c99`` is reported per
angle for comparison only. **Negative values are not required.**
This demo does not claim that the displaced mesh improves
caustic metrics; it is a comparison-pipeline smoke check, not
an improvement proof. Whether the displaced mesh shows lower or
higher C99 depends on the synthetic setup chosen here.

Limitations
-----------
- The mesh is a vertically subdivided solid cylinder approximation
  (``create_subdivided_synthetic_bottle_body``). It does not model
  wall thickness, an inner surface, neck, shoulder, base curvature,
  or water volume.
- Risk is uniform-active (every pixel equally likely); this is
  not a physical caustic risk distribution.
- The interface sequence treats the mesh as a single PET body
  (one entry interface, one exit interface); multi-wall and
  fluid volumes are out of scope.
- No STL / OBJ / CAD / STEP export, no file write, no
  visualization, no thermal model, no optimization, no residual
  hotspot update, no manufacturability validation, no
  self-intersection solver, no mesh repair, no medium tracking,
  no physical irradiance unit conversion, no pixel-area
  normalization.
- Watertightness of the displaced mesh is neither asserted nor
  printed by this demo; that is a downstream mesh quality
  concern.

Run from the repository root:

    python examples/run_original_vs_displaced_optical_demo.py

The script exits 0 when every invariant holds and 1 otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from optics_simulation.angle_scan import (
    AngleScanResult,
    run_baseline_angle_scan,
)
from optics_simulation.contribution.risk_map import RiskMap
from optics_simulation.geometry import (
    create_subdivided_synthetic_bottle_body,
    create_vertex_surface_coordinates,
)
from optics_simulation.optics import (
    create_detector_grid,
    create_detector_plane,
)
from optics_simulation.pattern import (
    DisplacedMeshResult,
    compute_vertex_displacement_amounts,
    create_displaced_mesh_copy,
    create_gaussian_dimple_pattern,
)


IOR_AIR = 1.00028
IOR_PET = 1.575
HOTSPOT_THRESHOLDS = (2.0, 5.0, 10.0)
TOP_PERCENT_FOR_C99 = 1.0
INCIDENT_REFERENCE = 1.0  # unit-less relative irradiance surrogate

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
RAY_GRID_NX = 11
RAY_GRID_NY = 11
EXPECTED_RAY_COUNT = RAY_GRID_NX * RAY_GRID_NY
VALID_TERMINATIONS = frozenset(
    {
        "completed_interfaces",
        "no_active_rays",
        "no_interfaces",
        "max_steps_reached",
    }
)


def _build_simple_risk_map(resolution: tuple[int, int]) -> RiskMap:
    """Synthetic uniform-active RiskMap for the comparison smoke check.

    Real research would use ``build_risk_map(contribution, ...)``
    on top of a detector-linked contribution map. This demo
    bypasses that chain because it focuses on the
    original-vs-displaced comparison-pipeline smoke check, not on
    hotspot localization.
    """
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


def _print_summary(
    original_mesh,
    displaced_result: DisplacedMeshResult,
    original_scan: AngleScanResult,
    displaced_scan: AngleScanResult,
    vertices_mutated: bool,
    faces_mutated: bool,
) -> None:
    print("Original vs displaced optical demo")
    print(
        "Synthetic original-vs-displaced optical comparison smoke check; "
        "not a physical PET-bottle validation."
    )
    print(
        "Note: original and displaced meshes are scanned with identical "
        "ray, detector, interface, and metric settings."
    )
    print(
        "Note: Delta C99 is reported for comparison only; negative "
        "values are not required."
    )
    print("Note: using synthetic uniform RiskMap; not detector-derived risk.")
    print("Note: no STL / OBJ / CAD / STEP export is performed.")
    print(
        "Note: this demo is a comparison smoke check only; it does not "
        "claim the displaced mesh improves caustic metrics."
    )
    print(f"Mode: {displaced_result.mode}")
    angles_str = ", ".join(str(a) for a in ANGLES_DEGREES)
    print(f"Angles: {angles_str}")
    print(f"Moved vertex count: {int(displaced_result.moved_mask.sum())}")
    print(f"Max displacement: {float(displaced_result.max_displacement):.4f}")
    print(f"Mean displacement: {float(displaced_result.mean_displacement):.4f}")

    if original_scan.max_c99_angle is None:
        print("Original max C99 angle: None")
    else:
        print(f"Original max C99 angle: {float(original_scan.max_c99_angle)}")
    if original_scan.max_c99 is None:
        print("Original max C99: None")
    else:
        print(f"Original max C99: {float(original_scan.max_c99):.4f}")
    if displaced_scan.max_c99_angle is None:
        print("Displaced max C99 angle: None")
    else:
        print(
            f"Displaced max C99 angle: {float(displaced_scan.max_c99_angle)}"
        )
    if displaced_scan.max_c99 is None:
        print("Displaced max C99: None")
    else:
        print(f"Displaced max C99: {float(displaced_scan.max_c99):.4f}")

    for orig, disp in zip(original_scan.per_angle, displaced_scan.per_angle):
        delta = float(disp.metrics.c99) - float(orig.metrics.c99)
        print(
            f"Angle {orig.angle_degrees}: "
            f"Original C99: {float(orig.metrics.c99):.4f} "
            f"Displaced C99: {float(disp.metrics.c99):.4f} "
            f"Delta C99: {delta:.4f} "
            f"Original detector hits: {int(orig.detector_hits)} "
            f"Displaced detector hits: {int(disp.detector_hits)} "
            f"Original termination: {orig.termination_reason} "
            f"Displaced termination: {disp.termination_reason}"
        )

    print(f"Mesh vertices mutated: {bool(vertices_mutated)}")
    print(f"Mesh faces mutated: {bool(faces_mutated)}")


def _check_invariants(
    original_mesh,
    displaced_result: DisplacedMeshResult,
    original_scan: AngleScanResult,
    displaced_scan: AngleScanResult,
    vertices_mutated: bool,
    faces_mutated: bool,
) -> bool:
    checks: list[bool] = []

    n_angles = len(ANGLES_DEGREES)
    checks.append(int(original_scan.angle_count) == n_angles)
    checks.append(int(displaced_scan.angle_count) == n_angles)
    checks.append(len(original_scan.per_angle) == n_angles)
    checks.append(len(displaced_scan.per_angle) == n_angles)

    orig_angles = tuple(p.angle_degrees for p in original_scan.per_angle)
    disp_angles = tuple(p.angle_degrees for p in displaced_scan.per_angle)
    checks.append(orig_angles == ANGLES_DEGREES)
    checks.append(disp_angles == ANGLES_DEGREES)
    checks.append(orig_angles == disp_angles)

    for entry in original_scan.per_angle:
        checks.append(int(entry.ray_count) == EXPECTED_RAY_COUNT)
        checks.append(int(entry.final_ray_count) >= 0)
        checks.append(int(entry.final_ray_count) <= int(entry.ray_count))
        checks.append(int(entry.detector_hits) >= 0)
        checks.append(float(entry.metrics.c99) >= 0.0)
        checks.append(entry.termination_reason in VALID_TERMINATIONS)

    for entry in displaced_scan.per_angle:
        checks.append(int(entry.ray_count) == EXPECTED_RAY_COUNT)
        checks.append(int(entry.final_ray_count) >= 0)
        checks.append(int(entry.final_ray_count) <= int(entry.ray_count))
        checks.append(int(entry.detector_hits) >= 0)
        checks.append(float(entry.metrics.c99) >= 0.0)
        checks.append(entry.termination_reason in VALID_TERMINATIONS)

    checks.append(original_scan.max_c99_angle is not None)
    checks.append(original_scan.max_c99 is not None)
    checks.append(displaced_scan.max_c99_angle is not None)
    checks.append(displaced_scan.max_c99 is not None)

    checks.append(int(displaced_result.moved_mask.sum()) > 0)
    checks.append(float(displaced_result.max_displacement) > 0.0)
    checks.append(float(displaced_result.mean_displacement) >= 0.0)
    checks.append(displaced_result.mode == "inward")
    checks.append(displaced_result.displaced_mesh is not original_mesh)
    checks.append(
        int(displaced_result.original_vertex_count)
        == int(len(original_mesh.vertices))
    )

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

    vertices_mutated = not np.array_equal(
        np.asarray(original_mesh.vertices), vertices_before
    )
    faces_mutated = not np.array_equal(
        np.asarray(original_mesh.faces), faces_before
    )

    ray_grid_config = {
        "origin_plane_z": 200.0,
        "x_range": (-50.0, 50.0),
        "y_range": (-70.0, 70.0),
        "nx": RAY_GRID_NX,
        "ny": RAY_GRID_NY,
    }
    interface_sequence = [(IOR_AIR, IOR_PET), (IOR_PET, IOR_AIR)]
    detector = create_detector_plane(
        center=(0.0, 0.0, -100.0),
        normal=(0.0, 0.0, 1.0),
        up=(0.0, 1.0, 0.0),
        width=200.0,
        height=200.0,
    )
    detector_grid = create_detector_grid(
        width=200.0, height=200.0, resolution=(40, 40)
    )

    original_scan = run_baseline_angle_scan(
        mesh=original_mesh,
        angles_degrees=ANGLES_DEGREES,
        ray_grid_config=ray_grid_config,
        interface_sequence=interface_sequence,
        detector=detector,
        detector_grid=detector_grid,
        incident_reference=INCIDENT_REFERENCE,
        thresholds=HOTSPOT_THRESHOLDS,
        top_percent=TOP_PERCENT_FOR_C99,
    )
    displaced_scan = run_baseline_angle_scan(
        mesh=displaced_result.displaced_mesh,
        angles_degrees=ANGLES_DEGREES,
        ray_grid_config=ray_grid_config,
        interface_sequence=interface_sequence,
        detector=detector,
        detector_grid=detector_grid,
        incident_reference=INCIDENT_REFERENCE,
        thresholds=HOTSPOT_THRESHOLDS,
        top_percent=TOP_PERCENT_FOR_C99,
    )

    _print_summary(
        original_mesh, displaced_result, original_scan, displaced_scan,
        vertices_mutated, faces_mutated,
    )
    ok = _check_invariants(
        original_mesh, displaced_result, original_scan, displaced_scan,
        vertices_mutated, faces_mutated,
    )
    print(f"Invariants: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
