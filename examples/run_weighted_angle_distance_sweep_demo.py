"""Weighted angle-distance sweep demo.

Synthetic Fresnel-weighted angle-distance smoke check; not a
physical PET-bottle validation. Runs the angle-distance sweep
twice on the **same** synthetic mesh and detector setup — once
with ``use_power_weights=False`` (the historical ray-count
surrogate) and once with ``use_power_weights=True`` (cumulative
Fresnel transmission applied to detector accumulation) — and
prints a side-by-side per ``(angle, detector_z)`` comparison
plus aggregate "max C99 (angle, z)" pairs for each weighting.

Interpretation notes
--------------------
- ``use_power_weights=True`` applies cumulative Fresnel
  transmission weights to detector accumulation.
- This is **still not pixel-area-normalized irradiance**.
- This does **not** prove fire prevention or physical PET-bottle
  safety.
- Weighted ``C99`` / ``Cmax`` should be interpreted as a
  transmitted-power surrogate, **not calibrated W/m^2**.
- Weighted-minus-unweighted Delta values reflect cumulative
  Fresnel transmittance (``<= 1`` per interface). Delta is
  **not** a fire-risk-reduction claim.
- The ``weighted <= unweighted`` invariant used here is a
  smoke-check invariant for this passive-dielectric fixture
  with identical ray paths and identical detector hits. It is
  **not** a general optical-safety guarantee.

Limitations
-----------
- Synthetic solid-cylinder fixture; not a real PET bottle STL.
- Single PET interface sequence ``[(AIR, PET), (PET, AIR)]``;
  no shell, no wall thickness, no fluid medium.
- Detector pipeline still uses unit pixel area; no irradiance
  unit conversion.
- No STL / OBJ / CAD / STEP export, no file write, no
  visualization, no CSV / Parquet logging, no thermal model,
  no original-vs-displaced comparison, no pattern generation,
  no mesh displacement, no manufacturability validation, no
  optimization.

Run from the repository root:

    python examples/run_weighted_angle_distance_sweep_demo.py

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
INTERFACE_SEQUENCE = [(IOR_AIR, IOR_PET), (IOR_PET, IOR_AIR)]
RAY_GRID_CONFIG = {
    "origin_plane_z": 200.0,
    "x_range": (-50.0, 50.0),
    "y_range": (-70.0, 70.0),
    "nx": RAY_GRID_NX,
    "ny": RAY_GRID_NY,
}
VALID_TERMINATIONS = frozenset(
    {
        "completed_interfaces",
        "no_active_rays",
        "no_interfaces",
        "max_steps_reached",
    }
)
WEIGHTED_LE_TOL = 1e-12


def _run_sweep(mesh, *, use_power_weights: bool) -> AngleDistanceScanResult:
    return run_angle_distance_sweep(
        mesh=mesh,
        angles_degrees=ANGLES_DEGREES,
        detector_z_values=DETECTOR_Z_VALUES,
        ray_grid_config=RAY_GRID_CONFIG,
        interface_sequence=INTERFACE_SEQUENCE,
        detector_width=DETECTOR_WIDTH,
        detector_height=DETECTOR_HEIGHT,
        detector_resolution=DETECTOR_RESOLUTION,
        incident_reference=INCIDENT_REFERENCE,
        thresholds=HOTSPOT_THRESHOLDS,
        top_percent=TOP_PERCENT_FOR_C99,
        use_power_weights=use_power_weights,
    )


def _print_summary(
    unweighted: AngleDistanceScanResult,
    weighted: AngleDistanceScanResult,
) -> None:
    print("Weighted angle-distance sweep demo")
    print(
        "Synthetic Fresnel-weighted angle-distance smoke check; "
        "not a physical PET-bottle validation."
    )
    print(
        "Note: use_power_weights=True applies cumulative Fresnel "
        "transmission weights to detector accumulation."
    )
    print("Note: this is still not pixel-area-normalized irradiance.")
    print(
        "Note: this does not prove fire prevention or physical "
        "PET-bottle safety."
    )
    print(
        "Note: weighted C99/Cmax should be interpreted as a "
        "transmitted-power surrogate, not calibrated W/m^2."
    )
    print(
        "Note: weighted-minus-unweighted Delta reflects cumulative "
        "Fresnel transmittance, not a fire-risk-reduction claim."
    )
    print(
        "Note: no STL / OBJ / CAD / STEP export, no file write, no "
        "visualization is performed."
    )
    print(
        "Mode: side-by-side use_power_weights=False vs "
        "use_power_weights=True"
    )
    print(f"Angles: {', '.join(str(a) for a in ANGLES_DEGREES)}")
    print(
        "Detector z values: "
        + ", ".join(str(z) for z in DETECTOR_Z_VALUES)
    )
    print(f"Result count: {len(unweighted.per_result)}")

    def _fmt_or_none(value, fmt: str) -> str:
        if value is None:
            return "None"
        return format(float(value), fmt)

    print(
        f"Unweighted max C99 angle: {_fmt_or_none(unweighted.max_c99_angle, '')}"
    )
    print(
        f"Unweighted max C99 detector z: "
        f"{_fmt_or_none(unweighted.max_c99_detector_z, '')}"
    )
    print(f"Unweighted max C99: {_fmt_or_none(unweighted.max_c99, '.4f')}")
    print(
        f"Weighted max C99 angle: {_fmt_or_none(weighted.max_c99_angle, '')}"
    )
    print(
        f"Weighted max C99 detector z: "
        f"{_fmt_or_none(weighted.max_c99_detector_z, '')}"
    )
    print(f"Weighted max C99: {_fmt_or_none(weighted.max_c99, '.4f')}")
    print(
        "Per-result legend -> Unweighted: ray-count surrogate. "
        "Weighted: cumulative-Fresnel-T surrogate. "
        "Delta = Weighted - Unweighted."
    )

    for ue, we in zip(unweighted.per_result, weighted.per_result):
        d_c99 = float(we.metrics.c99) - float(ue.metrics.c99)
        d_cmax = float(we.metrics.cmax) - float(ue.metrics.cmax)
        print(
            f"Angle {ue.angle_degrees} z={ue.detector_z}: "
            f"Unweighted C99={float(ue.metrics.c99):.4f} "
            f"Weighted C99={float(we.metrics.c99):.4f} "
            f"Delta C99={d_c99:+.4f} "
            f"Unweighted Cmax={float(ue.metrics.cmax):.4f} "
            f"Weighted Cmax={float(we.metrics.cmax):.4f} "
            f"Delta Cmax={d_cmax:+.4f} "
            f"Unweighted detector hits={int(ue.detector_hits)} "
            f"Weighted detector hits={int(we.detector_hits)} "
            f"termination={ue.termination_reason}"
        )


def _check_invariants(
    unweighted: AngleDistanceScanResult,
    weighted: AngleDistanceScanResult,
) -> bool:
    checks: list[bool] = []

    n_angles = len(ANGLES_DEGREES)
    n_distances = len(DETECTOR_Z_VALUES)
    expected_total = n_angles * n_distances

    checks.append(int(unweighted.angle_count) == n_angles)
    checks.append(int(weighted.angle_count) == n_angles)
    checks.append(int(unweighted.detector_count) == n_distances)
    checks.append(int(weighted.detector_count) == n_distances)
    checks.append(len(unweighted.per_result) == expected_total)
    checks.append(len(weighted.per_result) == expected_total)
    checks.append(
        len(unweighted.per_result) == len(weighted.per_result)
    )

    saw_strict_decrease = False
    for ue, we in zip(unweighted.per_result, weighted.per_result):
        # Identical (angle, z) ordering and geometry between the two
        # scans is the demo's central sanity check: weighting changes
        # accumulation, not ray paths or detector intersections.
        checks.append(ue.angle_degrees == we.angle_degrees)
        checks.append(ue.detector_z == we.detector_z)
        checks.append(int(ue.ray_count) == EXPECTED_RAY_COUNT)
        checks.append(int(we.ray_count) == EXPECTED_RAY_COUNT)
        checks.append(int(ue.final_ray_count) == int(we.final_ray_count))
        checks.append(int(ue.detector_hits) == int(we.detector_hits))
        checks.append(ue.termination_reason == we.termination_reason)
        checks.append(ue.termination_reason in VALID_TERMINATIONS)

        checks.append(int(ue.final_ray_count) >= 0)
        checks.append(int(ue.final_ray_count) <= int(ue.ray_count))
        checks.append(int(ue.detector_hits) >= 0)
        checks.append(int(we.detector_hits) >= 0)

        u_c99 = float(ue.metrics.c99)
        w_c99 = float(we.metrics.c99)
        u_cmax = float(ue.metrics.cmax)
        w_cmax = float(we.metrics.cmax)

        checks.append(u_c99 >= 0.0)
        checks.append(w_c99 >= 0.0)
        checks.append(u_cmax >= 0.0)
        checks.append(w_cmax >= 0.0)
        checks.append(2.0 in ue.metrics.eexceed)
        checks.append(2.0 in we.metrics.eexceed)
        checks.append(2.0 in ue.metrics.ahot)
        checks.append(2.0 in we.metrics.ahot)

        # Passive-dielectric Fresnel bound: cumulative T <= 1, identical
        # geometry, so weighted metric values cannot exceed the
        # unweighted counterparts. This is a fixture-specific smoke
        # invariant, not a general optical-safety claim.
        checks.append(w_c99 <= u_c99 + WEIGHTED_LE_TOL)
        checks.append(w_cmax <= u_cmax + WEIGHTED_LE_TOL)

        if (
            w_c99 < u_c99 - WEIGHTED_LE_TOL
            or w_cmax < u_cmax - WEIGHTED_LE_TOL
        ):
            saw_strict_decrease = True

    # At least one matched entry must show a strict decrease, or the
    # weighting plumbing did not engage at all.
    checks.append(saw_strict_decrease)

    checks.append(unweighted.max_c99 is not None)
    checks.append(unweighted.max_c99_angle is not None)
    checks.append(unweighted.max_c99_detector_z is not None)
    checks.append(weighted.max_c99 is not None)
    checks.append(weighted.max_c99_angle is not None)
    checks.append(weighted.max_c99_detector_z is not None)

    if unweighted.per_result:
        u_max_observed = max(p.metrics.c99 for p in unweighted.per_result)
        checks.append(unweighted.max_c99 == u_max_observed)
    if weighted.per_result:
        w_max_observed = max(p.metrics.c99 for p in weighted.per_result)
        checks.append(weighted.max_c99 == w_max_observed)

    return all(checks)


def main() -> int:
    mesh = create_synthetic_bottle_body(
        radius=BOTTLE_RADIUS,
        height=BOTTLE_HEIGHT,
        sections=BOTTLE_SECTIONS,
    )
    unweighted = _run_sweep(mesh, use_power_weights=False)
    weighted = _run_sweep(mesh, use_power_weights=True)

    _print_summary(unweighted, weighted)
    ok = _check_invariants(unweighted, weighted)
    print(f"Invariants: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
