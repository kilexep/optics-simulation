"""Detector-linked contribution map demo.

Detector-linked contribution-map smoke check; not a physical
PET-bottle risk map. Wires the foundations together end-to-end:
synthetic z-axis aligned bottle-like mesh -> parallel ray grid ->
mesh first-hit intersection -> per-vertex surface (u, v) ->
hit-to-surface (u, v) -> multi-step trace (with lineage tracking)
-> detector plane intersection -> detector grid accumulation ->
detector hotspot ray selection -> remap detector-space hotspot
ray indices to initial-ray indices via
``trace.final_source_ray_indices`` -> initial-ray-aligned
:class:`HotspotSelection` -> contribution map binning of mesh
first-hit ``(u, v)`` for those initial rays.

Unlike :mod:`run_contribution_map_demo`, the contribution map here
is built from the **detector hotspot rays** (not from the mesh
first-hit mask), made possible by the lineage tracking added to
:class:`MultiStepTraceResult`.

Limitations
-----------
- The mesh is a simple solid cylinder approximation. It does not
  model wall thickness, neck, shoulder, base curvature, or water
  volume.
- Selected weights are a placeholder (1.0 per ray); per-ray
  weights are not yet preserved in ``DetectorAccumulationResult``.
- Ray splitting is not modelled (TIR rays terminate; reflected
  rays are not regenerated). Under this regime
  ``trace.final_source_ray_indices`` is unique, and the demo
  invariants fail loudly if duplicate initial indices appear.

Run from the repository root:

    python examples/run_detector_linked_contribution_map_demo.py

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


def _build_initial_aligned_selection(
    rays_count: int,
    initial_indices: np.ndarray,
    detector_selection: HotspotSelection,
) -> HotspotSelection:
    """Build a HotspotSelection in initial-ray space.

    The selected_pixel_indices forwarded here are detector-space
    bookkeeping; the contribution map itself bins on surface (u, v),
    not on detector pixels. Forwarded for dataclass consistency only.
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
    trace,
    detector_hits,
    hotspot_selection,
    initial_indices: np.ndarray,
    contribution: ContributionMap,
) -> None:
    print("Detector-linked contribution map demo")
    print(
        "Detector-linked contribution-map smoke check; "
        "not a physical PET-bottle risk map."
    )
    print(
        "Selected weights are a placeholder (1.0 per ray); "
        "per-ray weights are not yet preserved in DetectorAccumulationResult."
    )
    print(f"Initial ray count: {int(rays.ray_count)}")
    print(f"Final ray count: {int(trace.final_rays.ray_count)}")
    print(f"Detector hit count: {int(detector_hits.hit_mask.sum())}")
    print(
        f"Hotspot selected final ray count: "
        f"{int(hotspot_selection.selected_count)}"
    )
    print(f"Mapped initial ray count: {int(initial_indices.size)}")
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
    trace,
    detector_hits,
    hotspot_selection,
    initial_indices: np.ndarray,
    initial_aligned_selection: HotspotSelection,
    contribution: ContributionMap,
) -> bool:
    checks: list[bool] = []

    checks.append(int(rays.ray_count) == EXPECTED_RAY_COUNT)
    checks.append(
        trace.final_source_ray_indices.shape
        == (int(trace.final_rays.ray_count),)
    )
    checks.append(trace.final_source_ray_indices.dtype == np.int64)

    final_ray_count = int(trace.final_rays.ray_count)
    final_indices = hotspot_selection.selected_ray_indices
    if final_indices.size > 0:
        checks.append(int(final_indices.min()) >= 0)
        checks.append(int(final_indices.max()) < final_ray_count)

    checks.append(initial_indices.dtype == np.int64)
    checks.append(int(initial_indices.size) == int(hotspot_selection.selected_count))
    if initial_indices.size > 0:
        checks.append(int(initial_indices.min()) >= 0)
        checks.append(int(initial_indices.max()) < int(rays.ray_count))
        checks.append(
            int(np.unique(initial_indices).size) == int(initial_indices.size)
        )

    checks.append(
        int(initial_aligned_selection.ray_count) == int(rays.ray_count)
    )
    checks.append(
        int(initial_aligned_selection.selected_count)
        == int(hotspot_selection.selected_count)
    )

    checks.append(
        int(contribution.total_selected) <= int(hotspot_selection.selected_count)
    )
    checks.append(contribution.count_map.shape == CONTRIBUTION_RESOLUTION)
    checks.append(contribution.weight_map.shape == CONTRIBUTION_RESOLUTION)
    checks.append(contribution.normalized_map.shape == CONTRIBUTION_RESOLUTION)
    checks.append(float(contribution.normalized_map.max()) <= 1.0 + 1e-12)

    checks.append(
        bool(
            np.isclose(
                float(contribution.weight_map.sum()),
                float(contribution.total_weight),
            )
        )
    )

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

    # Lineage mapping: detector-space final ray indices ->
    # initial-ray indices via the multi-step trace's source map.
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

    _print_summary(
        rays,
        trace,
        detector_hits,
        hotspot_selection,
        initial_indices,
        contribution,
    )
    ok = _check_invariants(
        rays,
        trace,
        detector_hits,
        hotspot_selection,
        initial_indices,
        initial_aligned_selection,
        contribution,
    )
    print(f"Invariants: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
