"""Normalized weighted angle-distance sweep demo.

Synthetic normalized relative-irradiance angle-distance smoke
check; not a physical PET-bottle validation. Runs three
angle-distance sweeps on the **same** synthetic mesh and detector
setup and prints a per ``(angle, detector_z)`` side-by-side
comparison plus aggregate "max C99" for each variant:

  1. raw_unweighted: ``use_power_weights=False``,
     ``use_relative_irradiance=False`` (historical ray-count
     surrogate).
  2. raw_weighted: ``use_power_weights=True``,
     ``use_relative_irradiance=False`` (cumulative Fresnel T
     surrogate on the raw weight map).
  3. normalized_weighted: ``use_power_weights=True``,
     ``use_relative_irradiance=True`` (cumulative Fresnel T
     plus source-plane / ray-count / detector-pixel-area
     normalization).

Interpretation notes (also printed at runtime):
- ``use_power_weights=True`` applies cumulative Fresnel
  transmission weights to detector accumulation.
- ``use_relative_irradiance=True`` uses source-plane area, ray
  count, and detector pixel area to compute a relative
  irradiance surrogate.
- This is **still not a calibrated W/m^2 measurement**.
- This does **not** prove fire prevention or physical
  PET-bottle safety.
- Delta normalized-vs-raw values reflect the normalization
  factor ``source_area / (ray_count * pixel_area)``, not a
  fire-risk-reduction claim.

Limitations
-----------
- Synthetic solid-cylinder fixture; not a real PET bottle STL.
- Single PET interface sequence ``[(AIR, PET), (PET, AIR)]``;
  no shell, no wall thickness, no fluid medium.
- No STL / OBJ / CAD / STEP export, no file write, no
  visualization, no CSV / Parquet logging, no thermal model,
  no original-vs-displaced comparison, no pattern generation,
  no mesh displacement, no manufacturability validation, no
  optimization.

Run from the repository root:

    python examples/run_normalized_weighted_angle_distance_sweep_demo.py

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
INCIDENT_REFERENCE = 1.0
INCIDENT_IRRADIANCE = 1.0

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
SOURCE_AREA = (
    (RAY_GRID_CONFIG["x_range"][1] - RAY_GRID_CONFIG["x_range"][0])
    * (RAY_GRID_CONFIG["y_range"][1] - RAY_GRID_CONFIG["y_range"][0])
)
PIXEL_AREA = (
    (DETECTOR_WIDTH / DETECTOR_RESOLUTION[1])
    * (DETECTOR_HEIGHT / DETECTOR_RESOLUTION[0])
)
INCIDENT_POWER_PER_RAY = (
    INCIDENT_IRRADIANCE * SOURCE_AREA / EXPECTED_RAY_COUNT
)
VALID_TERMINATIONS = frozenset(
    {
        "completed_interfaces",
        "no_active_rays",
        "no_interfaces",
        "max_steps_reached",
    }
)


def _run_sweep(
    mesh,
    *,
    use_power_weights: bool,
    use_relative_irradiance: bool,
) -> AngleDistanceScanResult:
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
        use_relative_irradiance=use_relative_irradiance,
        incident_irradiance=INCIDENT_IRRADIANCE,
    )


def _fmt_or_none(value, fmt: str) -> str:
    if value is None:
        return "None"
    return format(float(value), fmt)


def _print_summary(
    raw_unweighted: AngleDistanceScanResult,
    raw_weighted: AngleDistanceScanResult,
    normalized_weighted: AngleDistanceScanResult,
) -> None:
    print("Normalized weighted angle-distance sweep demo")
    print(
        "Synthetic normalized relative-irradiance angle-distance "
        "smoke check; not a physical PET-bottle validation."
    )
    print(
        "Note: use_power_weights=True applies cumulative Fresnel "
        "transmission weights to detector accumulation."
    )
    print(
        "Note: use_relative_irradiance=True uses source-plane area, "
        "ray count, and detector pixel area to compute a relative "
        "irradiance surrogate; still not a calibrated W/m^2 "
        "measurement."
    )
    print(
        "Note: this does not prove fire prevention or physical "
        "PET-bottle safety."
    )
    print(
        "Note: Delta normalized-vs-raw values reflect the "
        "normalization factor source_area / (ray_count * "
        "pixel_area), not a fire-risk-reduction claim."
    )
    print(
        "Note: no STL / OBJ / CAD / STEP export, no file write, no "
        "visualization is performed."
    )
    print(
        "Mode: side-by-side raw_unweighted vs raw_weighted vs "
        "normalized_weighted (use_power_weights=True, "
        "use_relative_irradiance=True)"
    )
    print(f"Angles: {', '.join(str(a) for a in ANGLES_DEGREES)}")
    print(
        "Detector z values: "
        + ", ".join(str(z) for z in DETECTOR_Z_VALUES)
    )
    print(f"Source area: {float(SOURCE_AREA):.4f}")
    print(f"Pixel area: {float(PIXEL_AREA):.4f}")
    print(f"Incident power per ray: {float(INCIDENT_POWER_PER_RAY):.4f}")
    print(f"Result count: {len(normalized_weighted.per_result)}")

    print(
        f"Raw unweighted max C99: "
        f"{_fmt_or_none(raw_unweighted.max_c99, '.4f')}"
    )
    print(
        f"Raw weighted max C99: "
        f"{_fmt_or_none(raw_weighted.max_c99, '.4f')}"
    )
    print(
        f"Normalized weighted max C99: "
        f"{_fmt_or_none(normalized_weighted.max_c99, '.4f')}"
    )
    print(
        "Per-result legend -> Raw weighted: cumulative-Fresnel-T on "
        "raw weight_map. Normalized weighted: same Fresnel-T weights "
        "after source-area/ray-count/pixel-area normalization. Delta "
        "normalized-vs-raw = Normalized minus Raw (reflects the "
        "normalization factor)."
    )

    for ru, rw, nw in zip(
        raw_unweighted.per_result,
        raw_weighted.per_result,
        normalized_weighted.per_result,
    ):
        d_c99 = float(nw.metrics.c99) - float(rw.metrics.c99)
        d_cmax = float(nw.metrics.cmax) - float(rw.metrics.cmax)
        print(
            f"Angle {rw.angle_degrees} z={rw.detector_z}: "
            f"Raw weighted C99={float(rw.metrics.c99):.4f} "
            f"Normalized weighted C99={float(nw.metrics.c99):.4f} "
            f"Delta normalized-vs-raw C99={d_c99:+.4f} "
            f"Raw weighted Cmax={float(rw.metrics.cmax):.4f} "
            f"Normalized weighted Cmax={float(nw.metrics.cmax):.4f} "
            f"Delta normalized-vs-raw Cmax={d_cmax:+.4f} "
            f"Detector hits={int(rw.detector_hits)} "
            f"termination={rw.termination_reason}"
        )


def _check_invariants(
    raw_unweighted: AngleDistanceScanResult,
    raw_weighted: AngleDistanceScanResult,
    normalized_weighted: AngleDistanceScanResult,
) -> bool:
    checks: list[bool] = []
    n_angles = len(ANGLES_DEGREES)
    n_dist = len(DETECTOR_Z_VALUES)
    expected_total = n_angles * n_dist

    for r in (raw_unweighted, raw_weighted, normalized_weighted):
        checks.append(int(r.angle_count) == n_angles)
        checks.append(int(r.detector_count) == n_dist)
        checks.append(len(r.per_result) == expected_total)

    for ru, rw, nw in zip(
        raw_unweighted.per_result,
        raw_weighted.per_result,
        normalized_weighted.per_result,
    ):
        checks.append(ru.angle_degrees == rw.angle_degrees == nw.angle_degrees)
        checks.append(ru.detector_z == rw.detector_z == nw.detector_z)
        checks.append(int(rw.detector_hits) == int(nw.detector_hits))
        checks.append(int(rw.final_ray_count) == int(nw.final_ray_count))
        checks.append(int(ru.ray_count) == EXPECTED_RAY_COUNT)
        checks.append(int(rw.ray_count) == EXPECTED_RAY_COUNT)
        checks.append(int(nw.ray_count) == EXPECTED_RAY_COUNT)
        checks.append(rw.termination_reason in VALID_TERMINATIONS)
        checks.append(nw.termination_reason in VALID_TERMINATIONS)

        checks.append(ru.irradiance_surrogate is None)
        checks.append(rw.irradiance_surrogate is None)
        s = nw.irradiance_surrogate
        checks.append(s is not None)
        if s is not None:
            import numpy as np
            checks.append(np.isfinite(s.relative_irradiance_map).all())
            checks.append((s.relative_irradiance_map >= 0).all())
            checks.append(float(s.source_area) > 0.0)
            checks.append(float(s.pixel_area) > 0.0)
            checks.append(int(s.ray_count) == EXPECTED_RAY_COUNT)

        checks.append(float(ru.metrics.c99) >= 0.0)
        checks.append(float(rw.metrics.c99) >= 0.0)
        checks.append(float(nw.metrics.c99) >= 0.0)
        checks.append(float(ru.metrics.cmax) >= 0.0)
        checks.append(float(rw.metrics.cmax) >= 0.0)
        checks.append(float(nw.metrics.cmax) >= 0.0)

    return all(checks)


def main() -> int:
    mesh = create_synthetic_bottle_body(
        radius=BOTTLE_RADIUS,
        height=BOTTLE_HEIGHT,
        sections=BOTTLE_SECTIONS,
    )
    raw_unweighted = _run_sweep(
        mesh, use_power_weights=False, use_relative_irradiance=False,
    )
    raw_weighted = _run_sweep(
        mesh, use_power_weights=True, use_relative_irradiance=False,
    )
    normalized_weighted = _run_sweep(
        mesh, use_power_weights=True, use_relative_irradiance=True,
    )

    _print_summary(raw_unweighted, raw_weighted, normalized_weighted)
    ok = _check_invariants(
        raw_unweighted, raw_weighted, normalized_weighted
    )
    print(f"Invariants: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
