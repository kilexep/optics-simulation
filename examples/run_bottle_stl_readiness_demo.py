"""Bottle STL readiness demo runner.

Bottle STL readiness runner; not a physical PET-bottle validation.
Reads a caller-supplied STL/mesh path with
:func:`load_bottle_mesh_readiness_report` and prints a compact
diagnostic summary. When no ``--mesh`` argument is provided, the
runner falls back to the synthetic shell fixture
(``create_subdivided_synthetic_bottle_shell``) and reports on that
mesh in memory; the fallback writes no files. The synthetic STL
round-trip path lives in the unit tests, not in this demo.

Scope (also enforced at runtime)
--------------------------------
- This runner diagnoses mesh readiness before simulation.
- It does not repair meshes.
- It does not infer material regions automatically.
- It does not perform optical simulation.
- It does not perform thermal simulation.
- It does not prove fire prevention or PET-bottle safety.
- Synthetic STL round-trip is only a file-path smoke check.

Usage::

    python examples/run_bottle_stl_readiness_demo.py
    python examples/run_bottle_stl_readiness_demo.py --mesh path/to/file.stl
    python examples/run_bottle_stl_readiness_demo.py --mesh m.stl \\
        --scale 1.0 \\
        --expected-height-min 100 --expected-height-max 250 \\
        --expected-radius-min 25 --expected-radius-max 50 \\
        --require-watertight

The script exits 0 when every invariant holds and 1 otherwise.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optics_simulation.geometry import (
    BottleMeshReadinessFileReport,
    BottleMeshReadinessReport,
    compute_bottle_mesh_readiness_report,
    create_subdivided_synthetic_bottle_shell,
    load_bottle_mesh_readiness_report,
)


SYNTHETIC_OUTER_RADIUS = 30.0
SYNTHETIC_WALL_THICKNESS = 1.0
SYNTHETIC_HEIGHT = 120.0
SYNTHETIC_SECTIONS = 96
SYNTHETIC_HEIGHT_SEGMENTS = 24

SYNTHETIC_PATH_LABEL = "<synthetic fallback>"


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Bottle STL readiness runner; not a physical PET-bottle "
            "validation."
        ),
    )
    parser.add_argument(
        "--mesh",
        type=str,
        default=None,
        help=(
            "Path to an STL or other trimesh-supported mesh file. "
            "If omitted, runs in synthetic fallback mode."
        ),
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor applied to mesh vertices (default 1.0).",
    )
    parser.add_argument(
        "--expected-height-min",
        type=float,
        default=None,
        help="Optional minimum bound for bounding-box height.",
    )
    parser.add_argument(
        "--expected-height-max",
        type=float,
        default=None,
        help="Optional maximum bound for bounding-box height.",
    )
    parser.add_argument(
        "--expected-radius-min",
        type=float,
        default=None,
        help="Optional minimum bound for estimated cylinder radius.",
    )
    parser.add_argument(
        "--expected-radius-max",
        type=float,
        default=None,
        help="Optional maximum bound for estimated cylinder radius.",
    )
    parser.add_argument(
        "--require-watertight",
        action="store_true",
        help=(
            "If set, a non-watertight mesh produces a blocking issue."
        ),
    )
    parser.add_argument(
        "--require-winding-consistent",
        action="store_true",
        help=(
            "If set, an inconsistent-winding mesh produces a "
            "blocking issue."
        ),
    )
    return parser


def _resolve_range(
    lo: float | None, hi: float | None,
) -> tuple[float, float] | None:
    if lo is None and hi is None:
        return None
    if lo is None or hi is None:
        raise SystemExit(
            "Both min and max must be supplied together for an "
            "expected range, or neither."
        )
    return (float(lo), float(hi))


def _print_static_header() -> None:
    print("Bottle STL readiness demo")
    print(
        "Bottle STL readiness runner; not a physical PET-bottle "
        "validation."
    )
    print(
        "Note: this runner diagnoses mesh readiness before "
        "simulation."
    )
    print("Note: it does not repair meshes.")
    print("Note: it does not infer material regions automatically.")
    print("Note: it does not perform optical simulation.")
    print("Note: it does not perform thermal simulation.")
    print(
        "Note: it does not prove fire prevention or PET-bottle "
        "safety."
    )
    print(
        "Note: synthetic STL round-trip is only a file-path smoke "
        "check."
    )


def _print_readiness_block(
    *,
    external_stl_supplied: bool,
    mesh_path_label: str,
    load_success: bool,
    scale_factor: float,
    readiness_report: BottleMeshReadinessReport,
) -> None:
    fit = readiness_report.cylindrical_fit
    qr = readiness_report.quality_report

    print(f"External STL supplied: {bool(external_stl_supplied)}")
    print(f"Mesh path: {mesh_path_label}")
    print(f"Load success: {bool(load_success)}")
    print(f"Scale factor: {float(scale_factor):.6f}")
    print(f"Vertex count: {int(qr.vertex_count)}")
    print(f"Face count: {int(qr.face_count)}")
    print(f"Height: {float(fit.height):.4f}")
    print(f"Estimated radius: {float(fit.estimated_radius):.4f}")
    print(f"Radius CV: {float(fit.radius_cv):.6f}")
    print(f"Fit quality: {fit.fit_quality_label}")
    print(f"Watertight: {bool(fit.is_watertight)}")
    print(f"Winding consistent: {bool(fit.is_winding_consistent)}")
    print(
        f"Simulation ready: {bool(readiness_report.simulation_ready)}"
    )
    print(
        f"Blocking issues: "
        f"{list(readiness_report.blocking_issues)}"
    )
    print(f"Warnings: {list(readiness_report.warnings)}")


def _check_invariants(
    *,
    external_stl_supplied: bool,
    scale_factor: float,
    readiness_report: BottleMeshReadinessReport,
) -> bool:
    fit = readiness_report.cylindrical_fit
    qr = readiness_report.quality_report

    checks: list[bool] = []
    checks.append(int(qr.vertex_count) > 0)
    checks.append(int(qr.face_count) > 0)
    checks.append(float(fit.height) > 0.0)
    checks.append(float(fit.estimated_radius) > 0.0)
    checks.append(float(scale_factor) > 0.0)
    if not external_stl_supplied:
        checks.append(bool(readiness_report.simulation_ready))
    return all(checks)


def _run_external_stl(
    args: argparse.Namespace,
) -> BottleMeshReadinessFileReport:
    expected_h = _resolve_range(
        args.expected_height_min, args.expected_height_max,
    )
    expected_r = _resolve_range(
        args.expected_radius_min, args.expected_radius_max,
    )
    return load_bottle_mesh_readiness_report(
        args.mesh,
        scale_factor=float(args.scale),
        expected_height_range=expected_h,
        expected_radius_range=expected_r,
        require_watertight=bool(args.require_watertight),
        require_winding_consistent=bool(
            args.require_winding_consistent
        ),
    )


def _run_synthetic_fallback(
    args: argparse.Namespace,
) -> BottleMeshReadinessReport:
    expected_h = _resolve_range(
        args.expected_height_min, args.expected_height_max,
    )
    expected_r = _resolve_range(
        args.expected_radius_min, args.expected_radius_max,
    )
    shell_mesh = create_subdivided_synthetic_bottle_shell(
        outer_radius=SYNTHETIC_OUTER_RADIUS,
        wall_thickness=SYNTHETIC_WALL_THICKNESS,
        height=SYNTHETIC_HEIGHT,
        sections=SYNTHETIC_SECTIONS,
        height_segments=SYNTHETIC_HEIGHT_SEGMENTS,
    )
    return compute_bottle_mesh_readiness_report(
        shell_mesh,
        scale_factor=float(args.scale),
        expected_height_range=expected_h,
        expected_radius_range=expected_r,
        require_watertight=bool(args.require_watertight),
        require_winding_consistent=bool(
            args.require_winding_consistent
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    _print_static_header()

    if args.mesh is not None:
        file_report = _run_external_stl(args)
        _print_readiness_block(
            external_stl_supplied=True,
            mesh_path_label=str(file_report.mesh_path),
            load_success=bool(file_report.load_success),
            scale_factor=float(file_report.scale_factor),
            readiness_report=file_report.readiness_report,
        )
        ok = _check_invariants(
            external_stl_supplied=True,
            scale_factor=float(file_report.scale_factor),
            readiness_report=file_report.readiness_report,
        )
    else:
        readiness_report = _run_synthetic_fallback(args)
        _print_readiness_block(
            external_stl_supplied=False,
            mesh_path_label=SYNTHETIC_PATH_LABEL,
            load_success=True,
            scale_factor=float(args.scale),
            readiness_report=readiness_report,
        )
        ok = _check_invariants(
            external_stl_supplied=False,
            scale_factor=float(args.scale),
            readiness_report=readiness_report,
        )

    print(f"Invariants: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
