"""Baseline angle scan demo.

Synthetic slab smoke check of the baseline angle scan foundation:
synthetic box mesh -> per-angle parallel ray grid -> multi-step
trace -> detector plane intersection -> detector grid accumulation
-> optical metrics on a relative irradiance surrogate -> per-angle
console summary + max-C99 angle + invariant check.

This is **not** a reproduction of PET-bottle caustics. The detector
weight map is a unit-less relative irradiance surrogate computed
from counts of synthetic rays through a synthetic slab. The
"max C99 angle" reported below is the angle with the largest C99
within this synthetic setup; it is **not** a PET-bottle risk angle.
A higher C99 means a more concentrated caustic (higher local risk),
not a "better" angle.

Run from the repository root:

    python examples/run_angle_scan_demo.py

The script exits 0 when every invariant holds and 1 otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import trimesh

from optics_simulation.angle_scan import (
    AngleScanResult,
    run_baseline_angle_scan,
)
from optics_simulation.optics import (
    create_detector_grid,
    create_detector_plane,
)


IOR_AIR = 1.00028
IOR_PET = 1.575
HOTSPOT_THRESHOLDS = (2.0, 5.0, 10.0)
TOP_PERCENT_FOR_C99 = 1.0
INCIDENT_REFERENCE = 1.0  # unit-less relative irradiance surrogate
ANGLES_DEGREES = (0.0, 15.0, 30.0, 45.0)
RAY_GRID_NX = 11
RAY_GRID_NY = 11
EXPECTED_RAY_COUNT = RAY_GRID_NX * RAY_GRID_NY
VALID_TERMINATIONS = frozenset(
    {
        "completed_interfaces",
        "no_active_rays",
        "no_interfaces",
        "max_steps_reached",
    }
)


def _print_summary(result: AngleScanResult) -> None:
    print("Baseline angle scan demo")
    print("Synthetic slab smoke check; max C99 angle is not a PET-bottle risk angle.")
    angles_str = ", ".join(str(a) for a in ANGLES_DEGREES)
    print(f"Angles: {angles_str}")
    for entry in result.per_angle:
        m = entry.metrics
        print(
            f"Angle {entry.angle_degrees}: "
            f"rays={entry.ray_count} "
            f"final={entry.final_ray_count} "
            f"hits={entry.detector_hits} "
            f"Cmax={m.cmax:.4f} "
            f"C99={m.c99:.4f} "
            f"Eexceed@2.0={m.eexceed[2.0]:.1f} "
            f"Ahot@2.0={m.ahot[2.0]} "
            f"termination={entry.termination_reason}"
        )
    print(f"Max C99 angle: {result.max_c99_angle}")
    if result.max_c99 is None:
        print("Max C99: None")
    else:
        print(f"Max C99: {result.max_c99:.4f}")


def _check_invariants(result: AngleScanResult) -> bool:
    checks: list[bool] = []

    checks.append(result.angle_count == len(ANGLES_DEGREES))
    checks.append(len(result.per_angle) == result.angle_count)

    angles_in_order = tuple(p.angle_degrees for p in result.per_angle)
    checks.append(angles_in_order == ANGLES_DEGREES)

    for entry in result.per_angle:
        checks.append(entry.ray_count == EXPECTED_RAY_COUNT)
        checks.append(entry.final_ray_count >= 0)
        checks.append(entry.final_ray_count <= entry.ray_count)
        checks.append(entry.detector_hits >= 0)
        checks.append(entry.detector_hits <= entry.final_ray_count)
        checks.append(entry.metrics.cmax >= 0.0)
        checks.append(entry.metrics.c99 >= 0.0)
        checks.append(entry.metrics.peak_value >= 0.0)
        checks.append(2.0 in entry.metrics.eexceed)
        checks.append(2.0 in entry.metrics.ahot)
        checks.append(entry.termination_reason in VALID_TERMINATIONS)

    if result.per_angle:
        max_c99_observed = max(p.metrics.c99 for p in result.per_angle)
        checks.append(result.max_c99 is not None)
        checks.append(result.max_c99_angle is not None)
        checks.append(result.max_c99 == max_c99_observed)

    return all(checks)


def main() -> int:
    box = trimesh.creation.box(extents=(10.0, 10.0, 10.0))

    ray_grid_config = {
        "origin_plane_z": 20.0,
        "x_range": (-7.5, 7.5),
        "y_range": (-7.5, 7.5),
        "nx": RAY_GRID_NX,
        "ny": RAY_GRID_NY,
    }
    interface_sequence = [(IOR_AIR, IOR_PET), (IOR_PET, IOR_AIR)]

    detector = create_detector_plane(
        center=(0.0, 0.0, -15.0),
        normal=(0.0, 0.0, 1.0),
        up=(0.0, 1.0, 0.0),
        width=10.0,
        height=10.0,
    )
    detector_grid = create_detector_grid(
        width=10.0, height=10.0, resolution=(20, 20)
    )

    result = run_baseline_angle_scan(
        mesh=box,
        angles_degrees=ANGLES_DEGREES,
        ray_grid_config=ray_grid_config,
        interface_sequence=interface_sequence,
        detector=detector,
        detector_grid=detector_grid,
        incident_reference=INCIDENT_REFERENCE,
        thresholds=HOTSPOT_THRESHOLDS,
        top_percent=TOP_PERCENT_FOR_C99,
    )

    _print_summary(result)
    ok = _check_invariants(result)
    print(f"Invariants: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
