"""Angle-distance sweep demo.

Synthetic angle-distance optical sweep smoke check; not a
physical PET-bottle validation. Wires together: synthetic
solid-cylinder bottle-like mesh -> per-angle parallel ray grid
-> multi-step trace (run once per angle) -> for each
``detector_z``, detector plane intersection -> detector grid
accumulation -> optical metrics on a relative irradiance
surrogate -> per ``(angle, detector_z)`` console summary +
``Max C99`` ``(angle, detector_z)`` + invariant check.

Limitations
-----------
- The mesh is a simple solid-cylinder bottle-like fixture
  (``create_synthetic_bottle_body``). It does not model wall
  thickness, an inner surface, neck, shoulder, base curvature,
  or water volume.
- The interface sequence treats the cylinder as a single PET
  body (one entry interface, one exit interface); multi-wall
  and fluid volumes are out of scope.
- No STL / OBJ / CAD / STEP export, no file write, no
  visualization, no CSV / Parquet logging, no detector heatmap
  save, no original-vs-displaced comparison, no pattern
  generation, no mesh displacement, no thermal model, no
  optimization, no manufacturability validation, no medium
  auto-tracking, no Open3D backend, no physical irradiance unit
  conversion, no pixel-area normalization.
- The ``Max C99 angle`` and ``Max C99 detector z`` reported here
  are within this synthetic setup only; they are **not** real
  PET-bottle risk angles or risk distances. This demo does not
  claim manufacturability and does not claim fire prevention.

Run from the repository root:

    python examples/run_angle_distance_sweep_demo.py

The script exits 0 when every invariant holds and 1 otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optics_simulation.angle_scan import (
    AngleDistanceScanResult,
    run_angle_distance_sweep,
)
from optics_simulation.geometry import create_synthetic_bottle_body


IOR_AIR = 1.00028
IOR_PET = 1.575
HOTSPOT_THRESHOLDS = (2.0, 5.0, 10.0)
TOP_PERCENT_FOR_C99 = 1.0
INCIDENT_REFERENCE = 1.0  # unit-less relative irradiance surrogate

BOTTLE_RADIUS = 30.0
BOTTLE_HEIGHT = 120.0
BOTTLE_SECTIONS = 96

ANGLES_DEGREES = (0.0, 15.0, 30.0)
DETECTOR_Z_VALUES = (-80.0, -100.0, -120.0)

RAY_GRID_NX = 11
RAY_GRID_NY = 11
EXPECTED_RAY_COUNT = RAY_GRID_NX * RAY_GRID_NY
DETECTOR_WIDTH = 200.0
DETECTOR_HEIGHT = 200.0
DETECTOR_RESOLUTION = (40, 40)
VALID_TERMINATIONS = frozenset(
    {
        "completed_interfaces",
        "no_active_rays",
        "no_interfaces",
        "max_steps_reached",
    }
)


def _print_summary(result: AngleDistanceScanResult) -> None:
    print("Angle-distance sweep demo")
    print(
        "Synthetic angle-distance optical sweep smoke check; "
        "not a physical PET-bottle validation."
    )
    print(
        "Note: Max C99 angle / Max C99 detector z are within this "
        "synthetic setup only; not real PET-bottle risk angles or "
        "distances."
    )
    print(
        "Note: no STL / OBJ / CAD / STEP export, no file write, no "
        "visualization is performed."
    )
    angles_str = ", ".join(str(a) for a in ANGLES_DEGREES)
    distances_str = ", ".join(str(z) for z in DETECTOR_Z_VALUES)
    print(f"Angles: {angles_str}")
    print(f"Detector z values: {distances_str}")
    print(f"Result count: {len(result.per_result)}")

    if result.max_c99_angle is None:
        print("Max C99 angle: None")
    else:
        print(f"Max C99 angle: {float(result.max_c99_angle)}")
    if result.max_c99_detector_z is None:
        print("Max C99 detector z: None")
    else:
        print(f"Max C99 detector z: {float(result.max_c99_detector_z)}")
    if result.max_c99 is None:
        print("Max C99: None")
    else:
        print(f"Max C99: {float(result.max_c99):.4f}")

    print(
        "Per-result legend -> C99: top-1% pixel mean ratio. "
        "Cmax: peak pixel ratio. Detector hits: ray count at detector. "
        "Termination: trace stop reason."
    )
    for entry in result.per_result:
        m = entry.metrics
        print(
            f"Angle {entry.angle_degrees} z={entry.detector_z}: "
            f"C99={float(m.c99):.4f} "
            f"Cmax={float(m.cmax):.4f} "
            f"Detector hits={int(entry.detector_hits)} "
            f"Final rays={int(entry.final_ray_count)} "
            f"Termination={entry.termination_reason}"
        )


def _check_invariants(result: AngleDistanceScanResult) -> bool:
    checks: list[bool] = []

    n_angles = len(ANGLES_DEGREES)
    n_distances = len(DETECTOR_Z_VALUES)

    checks.append(int(result.angle_count) == n_angles)
    checks.append(int(result.detector_count) == n_distances)
    checks.append(len(result.per_result) == n_angles * n_distances)

    expected_pairs = [
        (a, z) for a in ANGLES_DEGREES for z in DETECTOR_Z_VALUES
    ]
    actual_pairs = [
        (p.angle_degrees, p.detector_z) for p in result.per_result
    ]
    checks.append(actual_pairs == expected_pairs)

    for entry in result.per_result:
        m = entry.metrics
        checks.append(int(entry.ray_count) == EXPECTED_RAY_COUNT)
        checks.append(int(entry.final_ray_count) >= 0)
        checks.append(int(entry.final_ray_count) <= int(entry.ray_count))
        checks.append(int(entry.detector_hits) >= 0)
        checks.append(float(m.c99) >= 0.0)
        checks.append(float(m.cmax) >= 0.0)
        checks.append(float(m.peak_value) >= 0.0)
        checks.append(2.0 in m.eexceed)
        checks.append(2.0 in m.ahot)
        checks.append(entry.termination_reason in VALID_TERMINATIONS)

    checks.append(result.max_c99 is not None)
    checks.append(result.max_c99_angle is not None)
    checks.append(result.max_c99_detector_z is not None)

    if result.per_result:
        max_c99_observed = max(p.metrics.c99 for p in result.per_result)
        checks.append(result.max_c99 == max_c99_observed)
        chosen = next(
            (
                p for p in result.per_result
                if p.angle_degrees == result.max_c99_angle
                and p.detector_z == result.max_c99_detector_z
            ),
            None,
        )
        checks.append(chosen is not None)
        if chosen is not None:
            checks.append(float(chosen.metrics.c99) == max_c99_observed)

    return all(checks)


def main() -> int:
    mesh = create_synthetic_bottle_body(
        radius=BOTTLE_RADIUS,
        height=BOTTLE_HEIGHT,
        sections=BOTTLE_SECTIONS,
    )

    ray_grid_config = {
        "origin_plane_z": 200.0,
        "x_range": (-50.0, 50.0),
        "y_range": (-70.0, 70.0),
        "nx": RAY_GRID_NX,
        "ny": RAY_GRID_NY,
    }
    interface_sequence = [(IOR_AIR, IOR_PET), (IOR_PET, IOR_AIR)]

    result = run_angle_distance_sweep(
        mesh=mesh,
        angles_degrees=ANGLES_DEGREES,
        detector_z_values=DETECTOR_Z_VALUES,
        ray_grid_config=ray_grid_config,
        interface_sequence=interface_sequence,
        detector_width=DETECTOR_WIDTH,
        detector_height=DETECTOR_HEIGHT,
        detector_resolution=DETECTOR_RESOLUTION,
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
