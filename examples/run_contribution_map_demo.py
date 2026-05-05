"""Contribution map demo.

Synthetic contribution-map smoke check that wires together the
foundations built so far: synthetic z-axis aligned bottle-like mesh
-> parallel ray grid -> mesh first-hit intersection -> per-vertex
surface (u, v) -> hit-to-surface (u, v) -> multi-step trace ->
detector plane intersection -> detector grid accumulation ->
hotspot ray selection -> contribution map binning.

This is **not** a physical PET-bottle contribution map. The mesh is
a simple solid cylinder approximation (no wall thickness, neck,
shoulder, base, or water volume). The detector hotspot rays
reported in the console are **not linked** to the contribution
map: ``trace.final_rays`` is a compact ray bundle whose indices do
not point back to the initial ray bundle, so detector hotspot ray
indices cannot currently be matched to mesh first-hit ``(u, v)``.
Linking the two will require ray-history / source-index tracking
through ``run_multi_step_trace`` and is a follow-up task.

For this demo the contribution map is built from a **synthetic**
:class:`HotspotSelection` whose selected mask is the mesh
first-hit mask. The contribution map therefore shows the ``(u, v)``
distribution of mesh first-hit rays on the bottle surface — useful
as a binning-pipeline smoke check, **not** as a real caustic risk
map.

Run from the repository root:

    python examples/run_contribution_map_demo.py

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
    create_contribution_map,
)
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


def _build_first_hit_selection(
    first_hits, weights: np.ndarray | None = None
) -> HotspotSelection:
    """Synthetic HotspotSelection over first-hit mask.

    This demo does not link detector hotspot rays to mesh first-hit
    rays (no ray history yet). To run the contribution map binning
    pipeline end-to-end we build a HotspotSelection whose selected
    mask is the mesh first-hit mask itself.

    Dummy pixel indices: this demo does not link detector pixels
    to surface hits.
    """
    selected_mask = np.array(first_hits.hit_mask, dtype=bool, copy=True)
    selected_ray_indices = np.flatnonzero(selected_mask).astype(np.int64)
    m = int(selected_ray_indices.size)
    if weights is None:
        selected_weights = np.ones(m, dtype=float)
    else:
        selected_weights = np.asarray(weights, dtype=float)
    selected_pixel_indices = np.full((m, 2), -1, dtype=np.int64)
    pixel_mask = np.zeros((1, 1), dtype=bool)
    return HotspotSelection(
        selected_mask=selected_mask,
        selected_ray_indices=selected_ray_indices,
        selected_pixel_indices=selected_pixel_indices,
        selected_weights=selected_weights,
        pixel_mask=pixel_mask,
        ray_count=int(first_hits.ray_count),
        selected_count=m,
        selection_mode="threshold",
    )


def _print_summary(
    rays,
    first_hits,
    detector_hits,
    accumulation,
    hotspot_selection,
    contribution: ContributionMap,
) -> None:
    print("Contribution map demo")
    print(
        "Synthetic contribution-map smoke check; "
        "not a physical PET-bottle risk map."
    )
    print(
        "Note: detector hotspot rays are reported for context only "
        "and are NOT linked to the contribution map."
    )
    print(f"Initial ray count: {int(rays.ray_count)}")
    print(f"First-hit count: {int(first_hits.hit_mask.sum())}")
    print(f"Detector hit count: {int(detector_hits.hit_mask.sum())}")
    print(f"Hotspot selected count: {int(hotspot_selection.selected_count)}")
    print(f"Contribution total selected: {int(contribution.total_selected)}")
    print(f"Contribution total weight: {float(contribution.total_weight):.4f}")
    nonzero_bins = int((contribution.count_map > 0).sum())
    print(f"Contribution nonzero bins: {nonzero_bins}")
    print(
        f"Contribution normalized max: "
        f"{float(contribution.normalized_map.max()):.4f}"
    )


def _check_invariants(
    rays,
    first_hits,
    hit_uv,
    trace,
    detector_hits,
    accumulation,
    hotspot_selection,
    contribution: ContributionMap,
) -> bool:
    checks: list[bool] = []

    checks.append(int(rays.ray_count) == EXPECTED_RAY_COUNT)
    checks.append(int(first_hits.ray_count) == int(rays.ray_count))
    checks.append(int(hit_uv.ray_count) == int(first_hits.ray_count))
    checks.append(
        int(detector_hits.ray_count) == int(trace.final_rays.ray_count)
    )
    checks.append(
        int(accumulation.total_hits) == int(detector_hits.hit_mask.sum())
    )
    checks.append(
        int(hotspot_selection.selected_count)
        <= int(detector_hits.hit_mask.sum())
    )
    checks.append(
        not bool(hotspot_selection.selected_mask[~detector_hits.hit_mask].any())
    )

    checks.append(contribution.count_map.shape == CONTRIBUTION_RESOLUTION)
    checks.append(contribution.weight_map.shape == CONTRIBUTION_RESOLUTION)
    checks.append(contribution.normalized_map.shape == CONTRIBUTION_RESOLUTION)

    first_hit_count = int(first_hits.hit_mask.sum())
    checks.append(int(contribution.total_selected) <= first_hit_count)
    checks.append(
        bool(
            np.isclose(
                float(contribution.weight_map.sum()),
                float(contribution.total_weight),
            )
        )
    )
    checks.append(
        int(contribution.count_map.sum()) == int(contribution.total_selected)
    )
    checks.append(float(contribution.normalized_map.max()) <= 1.0 + 1e-12)

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

    # Detector hotspot selection — reported for context only; not
    # linked to the contribution map (no ray history yet).
    hotspot_selection = select_hotspot_rays(
        detector_hits,
        accumulation,
        top_percent=TOP_PERCENT,
    )

    # Contribution map input — synthetic HotspotSelection from the
    # mesh first-hit mask. See module docstring for why we cannot
    # use the detector hotspot selection here.
    contribution_selection = _build_first_hit_selection(first_hits)
    contribution = create_contribution_map(
        contribution_selection,
        hit_uv,
        resolution=CONTRIBUTION_RESOLUTION,
    )

    _print_summary(
        rays,
        first_hits,
        detector_hits,
        accumulation,
        hotspot_selection,
        contribution,
    )
    ok = _check_invariants(
        rays,
        first_hits,
        hit_uv,
        trace,
        detector_hits,
        accumulation,
        hotspot_selection,
        contribution,
    )
    print(f"Invariants: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
