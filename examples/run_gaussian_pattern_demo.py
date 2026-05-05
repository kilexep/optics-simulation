"""Gaussian pattern demo.

Synthetic Gaussian-pattern smoke check; not a physical PET-bottle
pattern. End-to-end wires the foundations built so far: synthetic
z-axis aligned bottle-like mesh -> parallel ray grid -> mesh
first-hit intersection -> per-vertex surface (u, v) -> hit-to-
surface (u, v) -> multi-step trace (with lineage tracking) ->
detector plane intersection -> detector grid accumulation ->
detector hotspot ray selection -> remap detector-space hotspot
ray indices to initial-ray indices via
``trace.final_source_ray_indices`` -> initial-ray-aligned
:class:`HotspotSelection` -> contribution map binning -> risk-map
post-processing -> Gaussian dimple sampling -> dimple depth field
``eta(u, v)``.

This demo confirms the ContributionMap -> RiskMap ->
GaussianDimplePattern chain runs end-to-end against a synthetic
geometry. **Mesh displacement is not performed** and no patterned
STL is exported.

Limitations
-----------
- The mesh is a simple solid cylinder approximation. It does not
  model wall thickness, neck, shoulder, base curvature, or water
  volume.
- Selected weights are currently 1.0 per ray; true per-ray optical
  power weighting is not implemented yet (``DetectorAccumulationResult``
  does not preserve per-ray weights).
- No Poisson-disk minimum-distance enforcement, no Sobol / Halton
  quasi-random sampling, no manufacturing constraints. Sampling is
  weighted-with-replacement only.
- ``risk.total_risk == 0`` is handled by generating an empty
  pattern (``count=0``) instead of raising; this keeps the demo
  robust under synthetic-geometry edge cases.

Run from the repository root:

    python examples/run_gaussian_pattern_demo.py

The script exits 0 when every invariant holds and 1 otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from optics_simulation.contribution import (
    ContributionMap,
    build_risk_map,
    create_contribution_map,
)
from optics_simulation.contribution.risk_map import RiskMap
from optics_simulation.geometry import (
    create_synthetic_bottle_body,
    create_vertex_surface_coordinates,
    hit_to_surface_coordinates,
)
from optics_simulation.metrics import select_hotspot_rays
from optics_simulation.metrics.hotspot_selection import HotspotSelection
from optics_simulation.optics import (
    accumulate_detector_hits,
    create_detector_grid,
    create_detector_plane,
    intersect_detector_plane,
    intersect_rays,
    parallel_ray_grid,
    run_multi_step_trace,
)
from optics_simulation.pattern import (
    GaussianDimplePattern,
    create_gaussian_dimple_pattern,
)


IOR_AIR = 1.00028
IOR_PET = 1.575
BOTTLE_RADIUS = 30.0
BOTTLE_HEIGHT = 120.0
BOTTLE_SECTIONS = 96
RAY_GRID_NX = 11
RAY_GRID_NY = 11
EXPECTED_RAY_COUNT = RAY_GRID_NX * RAY_GRID_NY
RAY_GRID_ORIGIN_Z = 200.0
RAY_GRID_X_RANGE = (-50.0, 50.0)
RAY_GRID_Y_RANGE = (-70.0, 70.0)
RAY_DIRECTION = (0.0, 0.0, -1.0)
DETECTOR_CENTER = (0.0, 0.0, -100.0)
DETECTOR_NORMAL = (0.0, 0.0, 1.0)
DETECTOR_UP = (0.0, 1.0, 0.0)
DETECTOR_WIDTH = 200.0
DETECTOR_HEIGHT = 200.0
DETECTOR_RESOLUTION = (40, 40)
TOP_PERCENT = 10.0
CONTRIBUTION_RESOLUTION = (32, 64)  # (nv, nu)
RISK_THRESHOLD: float | None = None
RISK_EPSILON = 0.01
PATTERN_COUNT = 20
PATTERN_AMPLITUDE = 1.0
PATTERN_SIGMA_U = 0.03
PATTERN_SIGMA_V = 0.03
PATTERN_MAX_DEPTH = 0.25
PATTERN_SEED = 42


def _build_initial_aligned_selection(
    rays_count: int,
    initial_indices: np.ndarray,
    detector_selection: HotspotSelection,
) -> HotspotSelection:
    """Initial-ray-aligned HotspotSelection from detector-space selection.

    See run_detector_linked_contribution_map_demo for the lineage
    rationale; the contribution map is binned on surface (u, v),
    not on detector pixels, so the forwarded pixel indices are
    detector-space bookkeeping only.
    """
    selected_mask = np.zeros(int(rays_count), dtype=bool)
    if initial_indices.size > 0:
        selected_mask[initial_indices] = True

    return HotspotSelection(
        selected_mask=selected_mask,
        selected_ray_indices=initial_indices.astype(np.int64, copy=True),
        selected_pixel_indices=np.array(
            detector_selection.selected_pixel_indices, dtype=np.int64, copy=True
        ),
        selected_weights=np.array(
            detector_selection.selected_weights, dtype=float, copy=True
        ),
        pixel_mask=np.array(
            detector_selection.pixel_mask, dtype=bool, copy=True
        ),
        ray_count=int(rays_count),
        selected_count=int(initial_indices.size),
        selection_mode=detector_selection.selection_mode,
    )


def _print_summary(
    rays,
    contribution: ContributionMap,
    risk: RiskMap,
    pattern: GaussianDimplePattern,
    zero_risk_skipped: bool,
) -> None:
    print("Gaussian pattern demo")
    print(
        "Synthetic Gaussian-pattern smoke check; "
        "not a physical PET-bottle pattern."
    )
    print(
        "Selected weights are currently 1.0 per ray; "
        "true per-ray optical power weighting is not implemented yet."
    )
    print(f"Initial ray count: {int(rays.ray_count)}")
    print(f"Contribution total selected: {int(contribution.total_selected)}")
    print(f"Risk active count: {int(risk.active_count)}")
    print(f"Risk total: {float(risk.total_risk):.4f}")
    print(
        f"Probability map sum: {float(risk.probability_map.sum()):.6f}"
    )
    print(f"Dimple count: {len(pattern.dimples)}")
    print(
        f"Depth field shape: "
        f"({pattern.depth_field.shape[0]}, {pattern.depth_field.shape[1]})"
    )
    print(f"Depth field min: {float(pattern.depth_field.min()):.4f}")
    print(f"Depth field max: {float(pattern.depth_field.max()):.4f}")
    print(f"Depth field mean: {float(pattern.depth_field.mean()):.4f}")
    print(f"Max depth: {float(pattern.max_depth):.4f}")
    print(f"Seed: {pattern.seed}")
    if zero_risk_skipped:
        print(
            "Note: risk.total_risk == 0; pattern was generated with "
            "count=0 to avoid PatternError."
        )


def _check_invariants(
    rays,
    contribution: ContributionMap,
    risk: RiskMap,
    pattern: GaussianDimplePattern,
    zero_risk_skipped: bool,
) -> bool:
    checks: list[bool] = []

    checks.append(int(rays.ray_count) == EXPECTED_RAY_COUNT)
    checks.append(contribution.count_map.shape == CONTRIBUTION_RESOLUTION)

    if zero_risk_skipped:
        checks.append(len(pattern.dimples) == 0)
        checks.append(
            bool(
                np.isclose(
                    float(risk.probability_map.sum()), 0.0, atol=1e-12
                )
            )
        )
    else:
        checks.append(len(pattern.dimples) == PATTERN_COUNT)
        checks.append(
            bool(
                np.isclose(
                    float(risk.probability_map.sum()), 1.0, atol=1e-9
                )
            )
        )

    checks.append(pattern.depth_field.shape == risk.risk_map.shape)
    checks.append(bool(np.isfinite(pattern.depth_field).all()))
    checks.append(float(pattern.depth_field.min()) >= 0.0)
    checks.append(float(pattern.depth_field.max()) <= 1.0 + 1e-12)
    checks.append(float(pattern.max_depth) == float(PATTERN_MAX_DEPTH))
    checks.append(pattern.seed == PATTERN_SEED)
    checks.append(pattern.resolution == risk.risk_map.shape)

    return all(checks)


def main() -> int:
    mesh = create_synthetic_bottle_body(
        radius=BOTTLE_RADIUS,
        height=BOTTLE_HEIGHT,
        sections=BOTTLE_SECTIONS,
    )

    rays = parallel_ray_grid(
        origin_plane_z=RAY_GRID_ORIGIN_Z,
        direction=RAY_DIRECTION,
        x_range=RAY_GRID_X_RANGE,
        y_range=RAY_GRID_Y_RANGE,
        nx=RAY_GRID_NX,
        ny=RAY_GRID_NY,
    )

    first_hits = intersect_rays(mesh, rays)
    surface_map = create_vertex_surface_coordinates(mesh)
    hit_uv = hit_to_surface_coordinates(mesh, first_hits, surface_map)

    interface_sequence = [(IOR_AIR, IOR_PET), (IOR_PET, IOR_AIR)]
    trace = run_multi_step_trace(mesh, rays, interface_sequence)

    detector = create_detector_plane(
        center=DETECTOR_CENTER,
        normal=DETECTOR_NORMAL,
        up=DETECTOR_UP,
        width=DETECTOR_WIDTH,
        height=DETECTOR_HEIGHT,
    )
    detector_hits = intersect_detector_plane(trace.final_rays, detector)
    detector_grid = create_detector_grid(
        width=DETECTOR_WIDTH,
        height=DETECTOR_HEIGHT,
        resolution=DETECTOR_RESOLUTION,
    )
    accumulation = accumulate_detector_hits(detector_hits, detector_grid)

    hotspot_selection = select_hotspot_rays(
        detector_hits,
        accumulation,
        top_percent=TOP_PERCENT,
    )

    final_indices = hotspot_selection.selected_ray_indices
    initial_indices = trace.final_source_ray_indices[final_indices]

    initial_aligned_selection = _build_initial_aligned_selection(
        int(rays.ray_count),
        initial_indices,
        hotspot_selection,
    )

    contribution = create_contribution_map(
        initial_aligned_selection,
        hit_uv,
        resolution=CONTRIBUTION_RESOLUTION,
    )

    risk = build_risk_map(
        contribution,
        threshold=RISK_THRESHOLD,
        epsilon=RISK_EPSILON,
    )

    zero_risk_skipped = float(risk.total_risk) <= 0.0
    if zero_risk_skipped:
        pattern = create_gaussian_dimple_pattern(
            risk,
            count=0,
            amplitude=PATTERN_AMPLITUDE,
            sigma_u=PATTERN_SIGMA_U,
            sigma_v=PATTERN_SIGMA_V,
            max_depth=PATTERN_MAX_DEPTH,
            seed=PATTERN_SEED,
        )
    else:
        pattern = create_gaussian_dimple_pattern(
            risk,
            count=PATTERN_COUNT,
            amplitude=PATTERN_AMPLITUDE,
            sigma_u=PATTERN_SIGMA_U,
            sigma_v=PATTERN_SIGMA_V,
            max_depth=PATTERN_MAX_DEPTH,
            seed=PATTERN_SEED,
        )

    _print_summary(rays, contribution, risk, pattern, zero_risk_skipped)
    ok = _check_invariants(
        rays, contribution, risk, pattern, zero_risk_skipped
    )
    print(f"Invariants: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
