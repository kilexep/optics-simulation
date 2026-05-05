"""Vertex displacement amount demo.

Synthetic vertex-displacement-amount smoke check; not a physical
PET-bottle pattern. Wires together: synthetic z-axis aligned
bottle-like mesh -> per-vertex normalized cylindrical (u, v) ->
synthetic uniform RiskMap -> Gaussian dimple pattern ->
descriptor JSON round-trip -> per-vertex normalized / physical
depth amounts. **Mesh vertices and faces are not modified.**

This demo uses a synthetic uniform RiskMap to focus only on
pattern-to-vertex amount evaluation. It does not represent
detector-derived contribution risk; the detector / contribution
chain is exercised by ``run_detector_linked_contribution_map_demo``
and ``run_gaussian_pattern_demo``. Here we deliberately bypass it
to keep the message of this demo (amount-only computation) clear.

Limitations
-----------
- The mesh is a simple solid cylinder approximation. It does not
  model wall thickness, neck, shoulder, base curvature, or water
  volume.
- Risk is uniform-active (every pixel equally likely); this is
  not a physical caustic risk distribution.
- The descriptor round-trip uses ``json.dumps`` / ``json.loads``
  in memory only — no file is written.
- ``compute_vertex_displacement_amounts`` returns amounts only;
  vertex movement, normal direction policy, self-intersection
  and wall-thickness checks, and STL / CAD export are out of
  scope.

Run from the repository root:

    python examples/run_vertex_displacement_amount_demo.py

The script exits 0 when every invariant holds and 1 otherwise.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from optics_simulation.contribution.risk_map import RiskMap
from optics_simulation.geometry import (
    create_synthetic_bottle_body,
    create_vertex_surface_coordinates,
)
from optics_simulation.pattern import (
    GaussianDimplePattern,
    VertexPatternDisplacement,
    compute_vertex_displacement_amounts,
    create_gaussian_dimple_pattern,
    gaussian_pattern_from_descriptor,
    gaussian_pattern_to_descriptor,
)


BOTTLE_RADIUS = 30.0
BOTTLE_HEIGHT = 120.0
BOTTLE_SECTIONS = 96
RISK_RESOLUTION = (32, 64)  # (nv, nu)
PATTERN_COUNT = 20
PATTERN_AMPLITUDE = 1.0
PATTERN_SIGMA_U = 0.03
PATTERN_SIGMA_V = 0.03
PATTERN_MAX_DEPTH = 0.25
PATTERN_SEED = 42
ACTIVE_THRESHOLD = 0.01
BOUNDARY_EPSILON = 0.02


def _build_simple_risk_map(resolution: tuple[int, int]) -> RiskMap:
    """Synthetic uniform-active RiskMap for amount-only demo.

    Real research would use ``build_risk_map(contribution, ...)``
    on top of a detector-linked contribution map. This demo
    bypasses that chain because it focuses on the pattern -> vertex
    amount step, not on hotspot localization.
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
    surface_map,
    pattern: GaussianDimplePattern,
    displacement: VertexPatternDisplacement,
    vertices_mutated: bool,
    faces_mutated: bool,
) -> None:
    print("Vertex displacement amount demo")
    print(
        "Synthetic vertex-displacement-amount smoke check; "
        "not a physical PET-bottle pattern."
    )
    print("Note: using synthetic uniform RiskMap; not detector-derived risk.")
    print(
        "Note: this demo computes vertex displacement amounts only; "
        "mesh vertices are not moved."
    )
    print(f"Vertex count: {int(displacement.vertex_count)}")
    print(f"Active vertex count: {int(displacement.active_mask.sum())}")
    print(f"Normalized depth min: {float(displacement.normalized_depth.min()):.4f}")
    print(f"Normalized depth max: {float(displacement.normalized_depth.max()):.4f}")
    print(f"Physical depth min: {float(displacement.physical_depth.min()):.4f}")
    print(f"Physical depth max: {float(displacement.physical_depth.max()):.4f}")
    print(f"Max depth: {float(pattern.max_depth):.4f}")
    print(f"Boundary exclusion epsilon: {float(BOUNDARY_EPSILON):.4f}")
    print(f"Mesh vertices mutated: {bool(vertices_mutated)}")
    print(f"Mesh faces mutated: {bool(faces_mutated)}")


def _check_invariants(
    mesh,
    surface_map,
    pattern: GaussianDimplePattern,
    displacement: VertexPatternDisplacement,
    vertices_mutated: bool,
    faces_mutated: bool,
) -> bool:
    checks: list[bool] = []

    n = int(len(mesh.vertices))
    checks.append(int(displacement.vertex_count) == n)
    checks.append(displacement.normalized_depth.shape == (n,))
    checks.append(displacement.physical_depth.shape == (n,))
    checks.append(displacement.active_mask.shape == (n,))

    checks.append(bool(np.isfinite(displacement.normalized_depth).all()))
    checks.append(bool(np.isfinite(displacement.physical_depth).all()))
    checks.append(float(displacement.normalized_depth.min()) >= 0.0)
    checks.append(float(displacement.normalized_depth.max()) <= 1.0 + 1e-12)

    checks.append(
        bool(
            np.allclose(
                displacement.physical_depth,
                displacement.normalized_depth * float(pattern.max_depth),
                atol=1e-15,
            )
        )
    )
    checks.append(
        float(displacement.physical_depth.max()) <= float(PATTERN_MAX_DEPTH) + 1e-12
    )

    active_count = int(displacement.active_mask.sum())
    checks.append(0 <= active_count <= n)

    v = np.asarray(surface_map.v, dtype=float)
    boundary = (v <= BOUNDARY_EPSILON) | (v >= 1.0 - BOUNDARY_EPSILON)
    if boundary.any():
        checks.append(
            float(np.abs(displacement.normalized_depth[boundary]).max()) == 0.0
        )

    checks.append(not bool(vertices_mutated))
    checks.append(not bool(faces_mutated))

    return all(checks)


def main() -> int:
    mesh = create_synthetic_bottle_body(
        radius=BOTTLE_RADIUS,
        height=BOTTLE_HEIGHT,
        sections=BOTTLE_SECTIONS,
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

    descriptor = gaussian_pattern_to_descriptor(pattern)
    json_text = json.dumps(descriptor)
    descriptor_reloaded = json.loads(json_text)
    restored_pattern = gaussian_pattern_from_descriptor(descriptor_reloaded)

    vertices_before = np.array(mesh.vertices, copy=True)
    faces_before = np.array(mesh.faces, copy=True)

    displacement = compute_vertex_displacement_amounts(
        mesh,
        surface_map,
        restored_pattern,
        active_threshold=ACTIVE_THRESHOLD,
        exclude_v_boundary_epsilon=BOUNDARY_EPSILON,
    )

    vertices_mutated = not np.array_equal(
        np.asarray(mesh.vertices), vertices_before
    )
    faces_mutated = not np.array_equal(
        np.asarray(mesh.faces), faces_before
    )

    _print_summary(
        mesh, surface_map, restored_pattern, displacement,
        vertices_mutated, faces_mutated,
    )
    ok = _check_invariants(
        mesh, surface_map, restored_pattern, displacement,
        vertices_mutated, faces_mutated,
    )
    print(f"Invariants: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
