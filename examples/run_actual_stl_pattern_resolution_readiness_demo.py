"""Actual STL pattern-resolution readiness demo.

Actual STL pattern-resolution readiness diagnostic; not a physical
PET-bottle validation. Loads the supplied STL, target-scales it
via the legacy PET-water trace setup, builds the actual-STL
body-region vertex mask, and computes a pattern-resolution
readiness report against the requested ``--sigma`` / ``--max-depth``
parameters. Then creates a synthetic subdivided copy via
``trimesh.Trimesh.subdivide`` and recomputes the readiness report
on the subdivided shell. Console summary only; no optical or
thermal simulation, no file output.

Old H.max raw detector-bin count is **NOT** used as a primary
metric. Canonical reductions for downstream stages remain the
relative irradiance surrogate, ``C99``, and the per-pixel
thermal-risk metrics; this demo runs none of them.

Scope (also enforced at runtime)
--------------------------------
- This diagnostic checks whether a mesh is dense enough to
  represent vertex-displacement patterns at the requested
  ``sigma_u`` / ``sigma_v`` / ``max_depth`` parameters.
- It does not prove manufacturability.
- It does not repair the mesh.
- It does not optimize patterns.
- It does not perform optical or thermal validation.
- Subdivision is a synthetic numerical refinement for
  diagnostics, not a validated manufacturing surface.
- No fire-prevention or PET-bottle-safety claim is allowed.

Usage::

    python examples/run_actual_stl_pattern_resolution_readiness_demo.py \\
        --mesh data/raw/stl/pet_bottle.stl \\
        --target-height 225.6 --target-diameter 72.1 \\
        --wall-thickness 0.3 --sigma 0.03 --max-depth 0.05 \\
        --subdivision-iterations 1

The script exits 0 when every invariant holds and 1 otherwise.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optics_simulation.geometry import (
    create_actual_bottle_body_vertex_mask,
    create_vertex_surface_coordinates,
    load_mesh,
)
from optics_simulation.optics import (
    create_legacy_pet_water_trace_setup,
)
from optics_simulation.pattern import (
    PatternResolutionReadinessReport,
    compute_pattern_resolution_readiness,
    create_subdivided_mesh_copy_for_patterning,
)


_DEFAULT_MESH_PATH = "data/raw/stl/pet_bottle.stl"
_VALID_LABELS = (
    "pattern_resolution_adequate",
    "pattern_resolution_marginal",
    "pattern_resolution_insufficient",
)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Actual STL pattern-resolution readiness diagnostic; "
            "not a physical PET-bottle validation."
        ),
    )
    parser.add_argument(
        "--mesh", type=str, default=_DEFAULT_MESH_PATH,
    )
    parser.add_argument("--target-height", type=float, default=225.6)
    parser.add_argument("--target-diameter", type=float, default=72.1)
    parser.add_argument("--wall-thickness", type=float, default=0.3)
    parser.add_argument(
        "--inner-offset-mode",
        choices=("auto", "minus_normals", "plus_normals"),
        default="auto",
    )
    parser.add_argument("--sigma", type=float, default=0.03)
    parser.add_argument("--max-depth", type=float, default=0.05)
    parser.add_argument(
        "--subdivision-iterations", type=int, default=1,
    )
    parser.add_argument(
        "--min-vertices-required", type=int, default=200,
    )
    parser.add_argument(
        "--min-sigma-to-spacing-ratio", type=float, default=2.0,
    )
    parser.add_argument(
        "--max-depth-to-edge-ratio", type=float, default=0.25,
    )
    return parser


def _print_static_header() -> None:
    print("Actual STL pattern-resolution readiness demo")
    print(
        "Actual STL pattern-resolution readiness diagnostic; not a "
        "physical PET-bottle validation."
    )
    print(
        "Note: this diagnostic checks whether a mesh is dense "
        "enough to represent vertex-displacement patterns."
    )
    print("Note: it does not prove manufacturability.")
    print("Note: it does not repair the mesh.")
    print("Note: it does not optimize patterns.")
    print(
        "Note: it does not perform optical or thermal validation."
    )
    print(
        "Note: subdivision is a synthetic numerical refinement for "
        "diagnostics, not a validated manufacturing surface."
    )
    print(
        "Note: no fire-prevention or PET-bottle-safety claim is "
        "allowed."
    )


def _format_optional_float(value, fmt: str = "{:.6f}") -> str:
    if value is None:
        return "None"
    return fmt.format(float(value))


def _print_report(prefix: str, report: PatternResolutionReadinessReport) -> None:
    print(
        f"{prefix} vertex count: {int(report.vertex_count)}"
    )
    print(
        f"{prefix} selected vertices: "
        f"{int(report.selected_vertex_count)} / "
        f"{int(report.vertex_count)}"
    )
    print(
        f"{prefix} face count: {int(report.face_count)}"
    )
    print(
        f"{prefix} selected edges: "
        f"{int(report.selected_edge_count)}"
    )
    print(
        f"{prefix} edge median: "
        f"{float(report.edge_length_median):.6f}"
    )
    print(
        f"{prefix} edge p90: "
        f"{float(report.edge_length_p90):.6f}"
    )
    print(
        f"{prefix} UV spacing: "
        f"{_format_optional_float(report.estimated_uv_spacing_median)}"
    )
    if report.sigma_u_to_uv_spacing_ratio is None:
        print(f"{prefix} sigma/UV ratio: None")
    else:
        print(
            f"{prefix} sigma/UV ratio: "
            f"min({float(report.sigma_u_to_uv_spacing_ratio):.4f}, "
            f"{float(report.sigma_v_to_uv_spacing_ratio):.4f})"
        )
    print(
        f"{prefix} depth/edge ratio: "
        f"{float(report.max_depth_to_edge_median_ratio):.4f}"
    )
    print(f"{prefix} readiness: {report.readiness_label}")
    if report.warnings:
        for w in report.warnings:
            print(f"{prefix} warning: {w}")
    else:
        print(f"{prefix} warnings: (none)")


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    _print_static_header()

    print(f"Mesh path: {Path(args.mesh)}")
    print(f"Sigma: {float(args.sigma):.6f}")
    print(f"Max depth: {float(args.max_depth):.6f}")

    mesh = load_mesh(Path(args.mesh))
    setup = create_legacy_pet_water_trace_setup(
        mesh,
        target_height=float(args.target_height),
        target_diameter=float(args.target_diameter),
        wall_thickness=float(args.wall_thickness),
        inner_offset_mode=str(args.inner_offset_mode),
    )

    original_shell = setup.shell_mesh
    original_mask = create_actual_bottle_body_vertex_mask(
        original_shell,
    )
    original_surface = create_vertex_surface_coordinates(
        original_shell,
    )
    original_report = compute_pattern_resolution_readiness(
        original_shell,
        include_mask=original_mask.include_mask,
        surface_map=original_surface,
        sigma_u=float(args.sigma),
        sigma_v=float(args.sigma),
        max_depth=float(args.max_depth),
        min_vertices_required=int(args.min_vertices_required),
        min_sigma_to_spacing_ratio=float(
            args.min_sigma_to_spacing_ratio
        ),
        max_depth_to_edge_ratio=float(
            args.max_depth_to_edge_ratio
        ),
    )
    _print_report("Original", original_report)

    subdivided_shell, sub_report = (
        create_subdivided_mesh_copy_for_patterning(
            original_shell,
            iterations=int(args.subdivision_iterations),
        )
    )
    print(
        f"Subdivision iterations: {int(sub_report.iterations)}"
    )
    print(
        f"Subdivision delta: "
        f"vertices {int(sub_report.original_vertex_count)} -> "
        f"{int(sub_report.subdivided_vertex_count)}, "
        f"faces {int(sub_report.original_face_count)} -> "
        f"{int(sub_report.subdivided_face_count)}"
    )

    sub_mask = create_actual_bottle_body_vertex_mask(
        subdivided_shell,
    )
    sub_surface = create_vertex_surface_coordinates(
        subdivided_shell,
    )
    sub_report_readiness = compute_pattern_resolution_readiness(
        subdivided_shell,
        include_mask=sub_mask.include_mask,
        surface_map=sub_surface,
        sigma_u=float(args.sigma),
        sigma_v=float(args.sigma),
        max_depth=float(args.max_depth),
        min_vertices_required=int(args.min_vertices_required),
        min_sigma_to_spacing_ratio=float(
            args.min_sigma_to_spacing_ratio
        ),
        max_depth_to_edge_ratio=float(
            args.max_depth_to_edge_ratio
        ),
    )
    _print_report("Subdivided", sub_report_readiness)

    checks: list[bool] = []
    checks.append(
        isinstance(original_report, PatternResolutionReadinessReport)
    )
    checks.append(
        isinstance(
            sub_report_readiness, PatternResolutionReadinessReport
        )
    )
    checks.append(
        int(sub_report.subdivided_vertex_count)
        > int(sub_report.original_vertex_count)
    )
    checks.append(
        int(original_report.selected_vertex_count) >= 0
    )
    checks.append(
        int(sub_report_readiness.selected_vertex_count) >= 0
    )
    checks.append(
        original_report.readiness_label in _VALID_LABELS
    )
    checks.append(
        sub_report_readiness.readiness_label in _VALID_LABELS
    )

    ok = all(checks)
    print(f"Invariants: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
