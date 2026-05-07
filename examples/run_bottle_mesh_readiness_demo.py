"""Bottle mesh readiness diagnostic demo.

Synthetic/loaded bottle mesh readiness diagnostic; not a physical
PET-bottle validation. Runs
:func:`compute_bottle_mesh_readiness_report` on three synthetic
fixtures - the simple solid cylinder, the subdivided solid cylinder,
and the synthetic shell - and prints a compact summary together with
the resulting blocking issues and warnings. No actual STL file is
required; this demo is a smoke check that the readiness report wires
end-to-end on the same fixtures the rest of the pipeline already
uses.

Scope (also enforced at runtime)
--------------------------------
- This report diagnoses whether a mesh resembles the z-axis bottle
  body assumptions used by the current simulator.
- It does not repair meshes.
- It does not infer material regions automatically.
- It does not validate wall thickness for arbitrary STL.
- It does not prove fire prevention or PET-bottle safety.
- It is a pre-simulation readiness report.
- No STL / OBJ / CAD / STEP export, no file write, no
  visualization, no CSV / Parquet logging is performed.

Run from the repository root::

    python examples/run_bottle_mesh_readiness_demo.py

The script exits 0 when every invariant holds and 1 otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optics_simulation.geometry import (
    BottleMeshReadinessReport,
    compute_bottle_mesh_readiness_report,
    create_subdivided_synthetic_bottle_body,
    create_subdivided_synthetic_bottle_shell,
    create_synthetic_bottle_body,
)


OUTER_RADIUS = 30.0
WALL_THICKNESS = 1.0
HEIGHT = 120.0
SECTIONS = 96
HEIGHT_SEGMENTS = 24

EXPECTED_HEIGHT_RANGE = (HEIGHT * 0.95, HEIGHT * 1.05)
EXPECTED_RADIUS_RANGE = (OUTER_RADIUS * 0.85, OUTER_RADIUS * 1.05)


def _print_static_header() -> None:
    print("Bottle mesh readiness demo")
    print(
        "Synthetic/loaded bottle mesh readiness diagnostic; not a "
        "physical PET-bottle validation."
    )
    print(
        "Note: this report diagnoses whether a mesh resembles the "
        "z-axis bottle-body assumptions used by the current "
        "simulator."
    )
    print("Note: it does not repair meshes.")
    print("Note: it does not infer material regions automatically.")
    print(
        "Note: it does not validate wall thickness for arbitrary "
        "STL."
    )
    print(
        "Note: it does not prove fire prevention or PET-bottle "
        "safety."
    )
    print("Note: it is a pre-simulation readiness report.")
    print(f"Expected height range: {EXPECTED_HEIGHT_RANGE}")
    print(f"Expected radius range: {EXPECTED_RADIUS_RANGE}")


def _print_report(label: str, report: BottleMeshReadinessReport) -> None:
    fit = report.cylindrical_fit
    qr = report.quality_report
    print(f"Case: {label}")
    print(f"Vertex count: {int(qr.vertex_count)}")
    print(f"Face count: {int(qr.face_count)}")
    print(f"Height: {float(fit.height):.4f}")
    print(f"Estimated radius: {float(fit.estimated_radius):.4f}")
    print(f"Radius CV: {float(fit.radius_cv):.6f}")
    print(f"Fit quality: {fit.fit_quality_label}")
    print(f"Watertight: {bool(fit.is_watertight)}")
    print(f"Winding consistent: {bool(fit.is_winding_consistent)}")
    print(f"Simulation ready: {bool(report.simulation_ready)}")
    print(f"Blocking issues: {len(report.blocking_issues)}")
    for issue in report.blocking_issues:
        print(f"  - {issue}")
    print(f"Warnings: {len(report.warnings)}")
    for warning in report.warnings:
        print(f"  - {warning}")


def _check_invariants(
    reports: tuple[BottleMeshReadinessReport, ...],
) -> bool:
    checks: list[bool] = []
    checks.append(len(reports) == 3)
    for report in reports:
        checks.append(int(report.quality_report.vertex_count) > 0)
        checks.append(int(report.quality_report.face_count) > 0)
        checks.append(float(report.cylindrical_fit.height) > 0.0)
        checks.append(
            float(report.cylindrical_fit.estimated_radius) > 0.0
        )

    simple_solid, sub_solid, shell = reports
    checks.append(bool(simple_solid.simulation_ready))
    checks.append(bool(sub_solid.simulation_ready))
    checks.append(
        bool(shell.simulation_ready)
        or len(shell.blocking_issues) == 0
    )
    return all(checks)


def main() -> int:
    _print_static_header()

    simple_mesh = create_synthetic_bottle_body(
        radius=OUTER_RADIUS, height=HEIGHT, sections=SECTIONS,
    )
    sub_mesh = create_subdivided_synthetic_bottle_body(
        radius=OUTER_RADIUS,
        height=HEIGHT,
        sections=SECTIONS,
        height_segments=HEIGHT_SEGMENTS,
    )
    shell_mesh = create_subdivided_synthetic_bottle_shell(
        outer_radius=OUTER_RADIUS,
        wall_thickness=WALL_THICKNESS,
        height=HEIGHT,
        sections=SECTIONS,
        height_segments=HEIGHT_SEGMENTS,
    )

    simple_report = compute_bottle_mesh_readiness_report(
        simple_mesh,
        expected_height_range=EXPECTED_HEIGHT_RANGE,
        expected_radius_range=EXPECTED_RADIUS_RANGE,
    )
    sub_report = compute_bottle_mesh_readiness_report(
        sub_mesh,
        expected_height_range=EXPECTED_HEIGHT_RANGE,
        expected_radius_range=EXPECTED_RADIUS_RANGE,
    )
    shell_report = compute_bottle_mesh_readiness_report(
        shell_mesh,
        expected_height_range=EXPECTED_HEIGHT_RANGE,
        expected_radius_range=EXPECTED_RADIUS_RANGE,
    )

    _print_report("Simple solid cylinder", simple_report)
    _print_report("Subdivided solid cylinder", sub_report)
    _print_report("Synthetic shell", shell_report)

    ok = _check_invariants(
        (simple_report, sub_report, shell_report),
    )
    print(f"Invariants: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
