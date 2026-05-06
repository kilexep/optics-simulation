"""Displaced mesh quality demo.

Synthetic displaced-mesh quality smoke check; not a physical
PET-bottle pattern. Wires together: subdivided synthetic bottle
mesh -> per-vertex normalized cylindrical (u, v) -> synthetic
uniform RiskMap -> Gaussian dimple pattern -> per-vertex
displacement amounts -> displaced mesh copy (mode=inward) ->
mesh quality report comparison (original vs displaced) and
displacement summary. **Original mesh vertices and faces are not
modified.**

This demo uses a synthetic uniform RiskMap to focus only on the
mesh-copy displacement quality smoke check. It does not represent
detector-derived contribution risk; the detector / contribution
chain is exercised by ``run_detector_linked_contribution_map_demo``
and ``run_gaussian_pattern_demo``. Here we deliberately bypass it
to keep the message of this demo (mesh-copy displacement quality)
clear.

Limitations
-----------
- The mesh is a vertically subdivided solid cylinder approximation
  (``create_subdivided_synthetic_bottle_body``). It does not model
  wall thickness, an inner surface, neck, shoulder, base curvature,
  or water volume.
- Risk is uniform-active (every pixel equally likely); this is
  not a physical caustic risk distribution.
- ``create_displaced_mesh_copy`` returns a displaced copy only; no
  STL / OBJ / CAD / STEP export, no file write, no optical
  comparison, no angle scan, no detector pipeline, no
  manufacturability validation, no self-intersection solver, no
  mesh repair, no shell / wall-thickness modeling, no medium
  tracking, no optimization, no residual-update, and no thermal
  model are performed.
- Watertightness of the displaced mesh is reported but not
  required to remain True after displacement. Watertightness
  preservation is a downstream mesh quality / refinement concern
  and is intentionally out of scope here.

Run from the repository root:

    python examples/run_displaced_mesh_quality_demo.py

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
    create_mesh_quality_report,
    create_subdivided_synthetic_bottle_body,
    create_vertex_surface_coordinates,
)
from optics_simulation.geometry.quality import MeshQualityReport
from optics_simulation.pattern import (
    DisplacedMeshResult,
    compute_vertex_displacement_amounts,
    create_displaced_mesh_copy,
    create_gaussian_dimple_pattern,
)


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


def _build_simple_risk_map(resolution: tuple[int, int]) -> RiskMap:
    """Synthetic uniform-active RiskMap for mesh-copy quality demo.

    Real research would use ``build_risk_map(contribution, ...)``
    on top of a detector-linked contribution map. This demo
    bypasses that chain because it focuses on the mesh-copy
    displacement quality step, not on hotspot localization.
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
    mesh,
    result: DisplacedMeshResult,
    original_report: MeshQualityReport,
    displaced_report: MeshQualityReport,
    vertices_mutated: bool,
    faces_mutated: bool,
) -> None:
    print("Displaced mesh quality demo")
    print(
        "Synthetic displaced-mesh quality smoke check; "
        "not a physical PET-bottle pattern."
    )
    print("Note: using synthetic uniform RiskMap; not detector-derived risk.")
    print("Note: no STL / OBJ / CAD / STEP export is performed.")
    print(f"Mode: {result.mode}")
    print(f"Original vertex count: {int(original_report.vertex_count)}")
    print(f"Displaced vertex count: {int(displaced_report.vertex_count)}")
    print(f"Original face count: {int(original_report.face_count)}")
    print(f"Displaced face count: {int(displaced_report.face_count)}")
    print(f"Moved vertex count: {int(result.moved_mask.sum())}")
    print(f"Max displacement: {float(result.max_displacement):.4f}")
    print(f"Mean displacement: {float(result.mean_displacement):.4f}")
    print(f"Original watertight: {bool(original_report.is_watertight)}")
    print(f"Displaced watertight: {bool(displaced_report.is_watertight)}")
    print(
        f"Original bounds z: "
        f"[{float(original_report.bounds_min[2]):.4f}, "
        f"{float(original_report.bounds_max[2]):.4f}]"
    )
    print(
        f"Displaced bounds z: "
        f"[{float(displaced_report.bounds_min[2]):.4f}, "
        f"{float(displaced_report.bounds_max[2]):.4f}]"
    )
    print(f"Mesh vertices mutated: {bool(vertices_mutated)}")
    print(f"Mesh faces mutated: {bool(faces_mutated)}")


def _check_invariants(
    mesh,
    result: DisplacedMeshResult,
    original_report: MeshQualityReport,
    displaced_report: MeshQualityReport,
    vertices_mutated: bool,
    faces_mutated: bool,
) -> bool:
    checks: list[bool] = []

    n = int(len(mesh.vertices))

    checks.append(result.displaced_mesh is not mesh)
    checks.append(int(result.original_vertex_count) == n)
    checks.append(
        int(original_report.vertex_count) == int(displaced_report.vertex_count)
    )
    checks.append(
        int(original_report.face_count) == int(displaced_report.face_count)
    )

    checks.append(result.displacement_vectors.shape == (n, 3))
    checks.append(result.displacement_magnitudes.shape == (n,))
    checks.append(result.moved_mask.shape == (n,))

    checks.append(int(result.moved_mask.sum()) > 0)
    checks.append(float(result.max_displacement) > 0.0)
    checks.append(float(result.mean_displacement) >= 0.0)
    checks.append(result.mode == "inward")

    checks.append(not bool(vertices_mutated))
    checks.append(not bool(faces_mutated))

    checks.append(int(original_report.vertex_count) > 0)
    checks.append(int(original_report.face_count) > 0)
    checks.append(int(displaced_report.vertex_count) > 0)
    checks.append(int(displaced_report.face_count) > 0)

    return all(checks)


def main() -> int:
    mesh = create_subdivided_synthetic_bottle_body(
        radius=BOTTLE_RADIUS,
        height=BOTTLE_HEIGHT,
        sections=BOTTLE_SECTIONS,
        height_segments=BOTTLE_HEIGHT_SEGMENTS,
    )
    surface_map = create_vertex_surface_coordinates(mesh)

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
        mesh,
        surface_map,
        pattern,
        active_threshold=ACTIVE_THRESHOLD,
        exclude_v_boundary_epsilon=BOUNDARY_EPSILON,
    )

    vertices_before = np.array(mesh.vertices, copy=True)
    faces_before = np.array(mesh.faces, copy=True)

    result = create_displaced_mesh_copy(
        mesh,
        displacement,
        mode=DISPLACEMENT_MODE,
        process=False,
    )

    vertices_mutated = not np.array_equal(
        np.asarray(mesh.vertices), vertices_before
    )
    faces_mutated = not np.array_equal(
        np.asarray(mesh.faces), faces_before
    )

    original_report = create_mesh_quality_report(mesh)
    displaced_report = create_mesh_quality_report(result.displaced_mesh)

    _print_summary(
        mesh, result, original_report, displaced_report,
        vertices_mutated, faces_mutated,
    )
    ok = _check_invariants(
        mesh, result, original_report, displaced_report,
        vertices_mutated, faces_mutated,
    )
    print(f"Invariants: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
