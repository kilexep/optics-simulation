"""Legacy PET-water STL optical smoke demo.

Legacy-style STL optical smoke check; not a physical PET-bottle
validation. Reproduces the geometric / scaling / refraction
structure of the old interactive PET-bottle experiment inside the
current testable optics framework: load an STL, apply legacy
anisotropic target-dimension scaling, build a generated inner
offset water boundary, and run the four-step
``air -> PET -> water -> PET -> air`` :func:`run_multi_mesh_trace`
sequence into a finite detector plane on the opposite side. The
demo runs no thermal simulation, no pattern generation, and no
calibrated physical validation.

Scope (also enforced at runtime)
--------------------------------
- This reproduces the old experiment's geometric / scaling /
  refraction structure inside the current framework.
- It uses target-dimension scaling and a generated inner offset
  mesh.
- It does not prove physical accuracy.
- It does not repair meshes.
- It does not infer real material regions automatically.
- It does not perform thermal simulation.
- It does not prove fire prevention or PET-bottle safety.
- Multi-mesh tracing is a foundation, not a full medium tracker.

Usage::

    python examples/run_legacy_pet_water_stl_optical_smoke_demo.py \\
        --mesh data/raw/stl/pet_bottle.stl \\
        --target-height 225.6 --target-diameter 72.1 \\
        --wall-thickness 0.3

The script exits 0 when every invariant holds and 1 otherwise.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from optics_simulation.geometry import load_mesh
from optics_simulation.metrics import (
    DetectorIrradianceSurrogate,
    compute_relative_irradiance_surrogate,
)
from optics_simulation.optics import (
    LegacyPetWaterTraceSetup,
    MultiMeshTraceResult,
    RayBundle,
    accumulate_detector_hits,
    create_detector_grid,
    create_detector_plane,
    create_legacy_pet_water_trace_setup,
    intersect_detector_plane,
    make_ray_bundle,
    run_multi_mesh_trace,
)


_DEFAULT_MESH_PATH = "data/raw/stl/pet_bottle.stl"

LIGHT_DISTANCE = 200.0
DETECTOR_DISTANCE = 200.0
DETECTOR_WIDTH = 400.0
DETECTOR_HEIGHT = 400.0
DETECTOR_RESOLUTION = (40, 40)
TRACE_EPSILON = 0.1


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Legacy-style STL optical smoke check; not a physical "
            "PET-bottle validation."
        ),
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default=_DEFAULT_MESH_PATH,
        help=(
            "Path to STL or other trimesh-supported mesh file "
            "(default: data/raw/stl/pet_bottle.stl)."
        ),
    )
    parser.add_argument(
        "--target-height", type=float, default=225.6,
        help="Legacy target height in mm (default 225.6).",
    )
    parser.add_argument(
        "--target-diameter", type=float, default=72.1,
        help="Legacy target diameter in mm (default 72.1).",
    )
    parser.add_argument(
        "--wall-thickness", type=float, default=0.3,
        help="Legacy wall thickness in mm (default 0.3).",
    )
    parser.add_argument(
        "--sample-count", type=int, default=25,
        help=(
            "Approximate total ray sample count; the demo uses an "
            "ny x nz grid where ny == nz == ceil(sqrt(sample_count))."
        ),
    )
    parser.add_argument(
        "--angle-degrees", type=float, default=0.0,
        help=(
            "Side-incidence angle around the y-axis in degrees "
            "(default 0)."
        ),
    )
    return parser


def _build_side_light_rays(
    *,
    setup: LegacyPetWaterTraceSetup,
    sample_count: int,
    angle_degrees: float,
) -> tuple[RayBundle, float]:
    bounds = np.asarray(setup.shell_mesh.bounds, dtype=float)
    ref_point = 0.5 * (bounds[0] + bounds[1])

    extent = float(
        max(setup.scale_report.scaled_height,
            setup.scale_report.scaled_xy_extent)
    ) * 1.5
    half_extent = 0.5 * extent

    n_per_axis = max(2, int(math.ceil(math.sqrt(int(sample_count)))))
    ys = np.linspace(-half_extent, +half_extent, n_per_axis)
    zs = np.linspace(-half_extent, +half_extent, n_per_axis)
    yv, zv = np.meshgrid(ys, zs, indexing="xy")

    local_origins = np.column_stack([
        np.full(yv.size, float(LIGHT_DISTANCE)),
        yv.ravel(),
        zv.ravel(),
    ])
    local_direction = np.array([-1.0, 0.0, 0.0], dtype=float)

    angle_rad = math.radians(float(angle_degrees))
    ca, sa = math.cos(angle_rad), math.sin(angle_rad)
    rotation = np.array(
        [
            [ca, 0.0, sa],
            [0.0, 1.0, 0.0],
            [-sa, 0.0, ca],
        ],
        dtype=float,
    )

    world_origins = (
        ref_point.reshape(1, 3) + local_origins @ rotation.T
    )
    world_direction = rotation @ local_direction
    world_directions = np.broadcast_to(
        world_direction.reshape(1, 3), world_origins.shape,
    ).copy()

    rays = make_ray_bundle(world_origins, world_directions)
    source_area = float(extent * extent)
    return rays, source_area


def _intersect_detector(
    *,
    setup: LegacyPetWaterTraceSetup,
    trace: MultiMeshTraceResult,
    angle_degrees: float,
):
    bounds = np.asarray(setup.shell_mesh.bounds, dtype=float)
    ref_point = 0.5 * (bounds[0] + bounds[1])

    angle_rad = math.radians(float(angle_degrees))
    ca, sa = math.cos(angle_rad), math.sin(angle_rad)
    rotation = np.array(
        [
            [ca, 0.0, sa],
            [0.0, 1.0, 0.0],
            [-sa, 0.0, ca],
        ],
        dtype=float,
    )

    detector_local_offset = np.array(
        [-DETECTOR_DISTANCE, 0.0, 0.0], dtype=float,
    )
    detector_center = ref_point + rotation @ detector_local_offset
    detector_normal = rotation @ np.array([1.0, 0.0, 0.0])
    detector_up = np.array([0.0, 0.0, 1.0], dtype=float)

    detector = create_detector_plane(
        center=tuple(detector_center.tolist()),
        normal=tuple(detector_normal.tolist()),
        up=tuple(detector_up.tolist()),
        width=DETECTOR_WIDTH,
        height=DETECTOR_HEIGHT,
    )
    detector_grid = create_detector_grid(
        width=DETECTOR_WIDTH,
        height=DETECTOR_HEIGHT,
        resolution=DETECTOR_RESOLUTION,
    )
    hits = intersect_detector_plane(trace.final_rays, detector)
    accumulation = accumulate_detector_hits(
        hits, detector_grid, weights=trace.final_ray_weights,
    )
    return detector_grid, hits, accumulation


def _print_static_header() -> None:
    print("Legacy PET-water STL optical smoke demo")
    print(
        "Legacy-style STL optical smoke check; not a physical "
        "PET-bottle validation."
    )
    print(
        "Note: this reproduces the old experiment's "
        "geometric/scaling/refraction structure inside the current "
        "framework."
    )
    print(
        "Note: it uses target-dimension scaling and a generated "
        "inner offset mesh."
    )
    print("Note: it does not prove physical accuracy.")
    print("Note: it does not repair meshes.")
    print(
        "Note: it does not infer real material regions "
        "automatically."
    )
    print("Note: it does not perform thermal simulation.")
    print(
        "Note: it does not prove fire prevention or PET-bottle "
        "safety."
    )
    print(
        "Note: multi-mesh tracing is a foundation, not a full "
        "medium tracker."
    )


def _check_invariants(
    *,
    setup: LegacyPetWaterTraceSetup,
    rays: RayBundle,
    trace: MultiMeshTraceResult,
    detector_hits_total: int,
    surrogate: DetectorIrradianceSurrogate,
) -> bool:
    checks: list[bool] = []
    checks.append(len(setup.step_specs) == 4)
    checks.append(int(rays.ray_count) > 0)
    checks.append(int(trace.final_rays.ray_count) >= 0)
    checks.append(int(detector_hits_total) >= 0)
    checks.append(np.isfinite(trace.final_ray_weights).all())
    if int(trace.final_rays.ray_count) > 0:
        idx = trace.final_source_ray_indices
        checks.append(idx.dtype == np.int64)
        checks.append(int(idx.min()) >= 0)
        checks.append(int(idx.max()) < int(rays.ray_count))
    checks.append(
        np.isfinite(surrogate.relative_irradiance_map).all()
    )
    return all(checks)


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    _print_static_header()

    mesh_path = Path(args.mesh)
    print(f"Mesh path: {mesh_path}")
    print(f"Target height: {float(args.target_height):.4f}")
    print(f"Target diameter: {float(args.target_diameter):.4f}")
    print(f"Wall thickness: {float(args.wall_thickness):.4f}")

    mesh = load_mesh(mesh_path)
    setup = create_legacy_pet_water_trace_setup(
        mesh,
        target_height=float(args.target_height),
        target_diameter=float(args.target_diameter),
        wall_thickness=float(args.wall_thickness),
    )

    print(f"Scale xyz: {tuple(float(s) for s in setup.scale_report.scale_xyz)}")
    print(f"Scaled height: {float(setup.scale_report.scaled_height):.4f}")
    print(
        f"Scaled xy extent: "
        f"{float(setup.scale_report.scaled_xy_extent):.4f}"
    )
    print(f"Step count: {int(len(setup.step_specs))}")

    rays, source_area = _build_side_light_rays(
        setup=setup,
        sample_count=int(args.sample_count),
        angle_degrees=float(args.angle_degrees),
    )
    print(f"Initial rays: {int(rays.ray_count)}")

    trace = run_multi_mesh_trace(
        rays, setup.step_specs, epsilon=TRACE_EPSILON,
    )
    print(f"Final rays: {int(trace.final_rays.ray_count)}")
    print(f"Termination: {str(trace.termination_reason)}")

    detector_grid, hits, accumulation = _intersect_detector(
        setup=setup, trace=trace,
        angle_degrees=float(args.angle_degrees),
    )
    print(f"Detector hits: {int(accumulation.total_hits)}")
    print(f"Weight sum: {float(accumulation.total_weight):.6f}")

    if int(rays.ray_count) > 0:
        surrogate = compute_relative_irradiance_surrogate(
            accumulation,
            detector_grid,
            ray_count=int(rays.ray_count),
            source_area=float(source_area),
            incident_irradiance=1.0,
        )
        max_rel_irr = float(
            surrogate.relative_irradiance_map.max()
        )
    else:
        # Defensive branch; ray_count > 0 is asserted via invariants.
        surrogate = None  # type: ignore[assignment]
        max_rel_irr = 0.0

    print(f"Max relative irradiance: {max_rel_irr:.6f}")

    ok = _check_invariants(
        setup=setup,
        rays=rays,
        trace=trace,
        detector_hits_total=int(accumulation.total_hits),
        surrogate=surrogate,
    )
    print(f"Invariants: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
