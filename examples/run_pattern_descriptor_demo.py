"""Pattern descriptor demo.

Synthetic Gaussian-descriptor smoke check; not a physical PET-bottle
pattern. Runs the same synthetic pipeline as
``run_gaussian_pattern_demo`` to obtain a :class:`GaussianDimplePattern`,
then exercises the descriptor round-trip:

    pattern
      -> gaussian_pattern_to_descriptor(pattern)
      -> json.dumps(descriptor)
      -> json.loads(json_text)
      -> gaussian_pattern_from_descriptor(descriptor_reloaded)
      -> compare to original (metadata + depth_field allclose)

The descriptor itself does not store ``depth_field``: reconstruction
recomputes it from the dimples and resolution. This demo confirms
that the descriptor is JSON-serializable, that depth_field is
recoverable, and that all metadata survives the round-trip.

This demo performs **no file I/O** and does **no mesh displacement**.
JSON is used in-memory only via ``json.dumps`` / ``json.loads``.

Limitations
-----------
- The mesh is a simple solid cylinder approximation. It does not
  model wall thickness, neck, shoulder, base curvature, or water
  volume.
- Selected weights are placeholder (1.0 per ray); per-ray optical
  power weighting is not yet preserved in
  ``DetectorAccumulationResult``.
- ``risk.total_risk == 0`` is handled by generating an empty
  pattern (``count=0``); the descriptor still round-trips with
  ``"dimples": []``.

Run from the repository root:

    python examples/run_pattern_descriptor_demo.py

The script exits 0 when every invariant holds and 1 otherwise.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from optics_simulation.contribution import (
    build_risk_map,
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
from optics_simulation.pattern import (
    GaussianDimplePattern,
    create_gaussian_dimple_pattern,
    gaussian_pattern_from_descriptor,
    gaussian_pattern_to_descriptor,
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
CONTRIBUTION_RESOLUTION = (32, 64)
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

    Self-contained copy (no cross-import from other example scripts).
    The contribution map bins on surface (u, v); the forwarded
    detector-space pixel indices are bookkeeping only.
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


def _run_synthetic_pipeline() -> tuple[GaussianDimplePattern, bool]:
    """Run the synthetic mesh -> ... -> Gaussian pattern chain."""
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
        detector_hits, accumulation, top_percent=TOP_PERCENT
    )

    final_indices = hotspot_selection.selected_ray_indices
    initial_indices = trace.final_source_ray_indices[final_indices]
    initial_aligned_selection = _build_initial_aligned_selection(
        int(rays.ray_count), initial_indices, hotspot_selection
    )

    contribution = create_contribution_map(
        initial_aligned_selection, hit_uv, resolution=CONTRIBUTION_RESOLUTION
    )
    risk = build_risk_map(
        contribution, threshold=RISK_THRESHOLD, epsilon=RISK_EPSILON
    )

    zero_risk_skipped = float(risk.total_risk) <= 0.0
    count = 0 if zero_risk_skipped else PATTERN_COUNT
    pattern = create_gaussian_dimple_pattern(
        risk,
        count=count,
        amplitude=PATTERN_AMPLITUDE,
        sigma_u=PATTERN_SIGMA_U,
        sigma_v=PATTERN_SIGMA_V,
        max_depth=PATTERN_MAX_DEPTH,
        seed=PATTERN_SEED,
    )
    return pattern, zero_risk_skipped


def _print_summary(
    pattern: GaussianDimplePattern,
    descriptor: dict,
    json_text: str,
    restored: GaussianDimplePattern,
    depth_allclose: bool,
    zero_risk_skipped: bool,
) -> None:
    print("Pattern descriptor demo")
    print(
        "Synthetic Gaussian-descriptor smoke check; "
        "not a physical PET-bottle pattern."
    )
    print(f"Pattern family: {descriptor['pattern_family']}")
    print(f"Coordinate system: {descriptor['coordinate_system']}")
    print(f"Dimple count: {int(descriptor['dimple_count'])}")
    print(f"Resolution: {descriptor['resolution']}")
    print(f"Max depth: {float(descriptor['max_depth']):.4f}")
    print(f"Seed: {descriptor['seed']}")
    print(f"JSON character count: {len(json_text)}")
    print(f"Round-trip dimples: {len(restored.dimples)}")
    print(f"Round-trip depth allclose: {bool(depth_allclose)}")
    if zero_risk_skipped:
        print(
            "Note: risk.total_risk == 0; pattern was generated with "
            "count=0; descriptor still serializes empty dimples list."
        )


def _check_invariants(
    pattern: GaussianDimplePattern,
    descriptor: dict,
    json_text: str,
    restored: GaussianDimplePattern,
    depth_allclose: bool,
) -> bool:
    checks: list[bool] = []

    checks.append(descriptor["pattern_family"] == "gaussian_dimple")
    checks.append(descriptor["coordinate_system"] == "normalized_cylindrical_uv")
    checks.append(descriptor["depth_field_type"] == "normalized_eta")
    checks.append(int(descriptor["dimple_count"]) == len(pattern.dimples))
    checks.append("depth_field" not in descriptor)
    checks.append(isinstance(json_text, str) and len(json_text) > 0)

    checks.append(len(restored.dimples) == len(pattern.dimples))
    checks.append(restored.resolution == pattern.resolution)
    checks.append(float(restored.max_depth) == float(pattern.max_depth))
    checks.append(restored.seed == pattern.seed)
    checks.append(restored.depth_field.shape == pattern.depth_field.shape)
    checks.append(bool(depth_allclose))

    return all(checks)


def main() -> int:
    pattern, zero_risk_skipped = _run_synthetic_pipeline()

    descriptor = gaussian_pattern_to_descriptor(pattern)
    json_text = json.dumps(descriptor)
    descriptor_reloaded = json.loads(json_text)
    restored = gaussian_pattern_from_descriptor(descriptor_reloaded)

    depth_allclose = bool(
        np.allclose(restored.depth_field, pattern.depth_field, atol=1e-12)
    )

    _print_summary(
        pattern, descriptor, json_text, restored, depth_allclose,
        zero_risk_skipped,
    )
    ok = _check_invariants(
        pattern, descriptor, json_text, restored, depth_allclose
    )
    print(f"Invariants: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
