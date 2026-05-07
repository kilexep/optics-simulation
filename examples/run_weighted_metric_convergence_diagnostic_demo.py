"""Weighted metric convergence diagnostic demo.

Synthetic weighted metric-convergence smoke check; not a
physical PET-bottle validation. Probes how the optical
caustic-risk metrics (``C99`` / ``Cmax``) of the existing
baseline angle scan respond to three independent setup axes —
detector pixel resolution, ray grid density, and ``top_percent``
— at a fixed ``(mesh, angles, detector_z)`` baseline. Each case
is run **twice**: once with ``use_power_weights=False``
(ray-count surrogate) and once with ``use_power_weights=True``
(cumulative Fresnel transmission applied to detector
accumulation). Per-case Unweighted / Weighted / Delta values for
``C99`` and ``Cmax``, plus matched ``Detector hits`` and
``Final rays``, are reported side-by-side. Aggregate
unweighted-saturation-at-1.0 and weighted-saturation-at-1.0
counts are emitted to make any post-Fresnel residual
quantization visible.

Interpretation notes (also printed at runtime):
- ``use_power_weights=True`` applies cumulative Fresnel
  transmission weights to detector accumulation.
- This is **still not pixel-area-normalized irradiance**.
- This is **still not calibrated W/m^2**.
- This demo diagnoses metric sensitivity, not pattern
  effectiveness.
- It does not recommend any correct detector resolution, ray
  grid, or ``top_percent``.
- It does not prove fire prevention or PET-bottle safety.
- Weighted-minus-unweighted Delta reflects Fresnel weighting
  semantics, not a fire-risk-reduction claim.

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

    python examples/run_weighted_metric_convergence_diagnostic_demo.py

The script exits 0 when every invariant holds and 1 otherwise.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optics_simulation.angle_scan import (
    AngleScanResult,
    run_baseline_angle_scan,
)
from optics_simulation.geometry import create_synthetic_bottle_body
from optics_simulation.optics import (
    create_detector_grid,
    create_detector_plane,
)


IOR_AIR = 1.00028
IOR_PET = 1.575
HOTSPOT_THRESHOLDS = (2.0, 5.0, 10.0)
INCIDENT_REFERENCE = 1.0  # unit-less relative irradiance surrogate

BOTTLE_RADIUS = 30.0
BOTTLE_HEIGHT = 120.0
BOTTLE_SECTIONS = 96

DETECTOR_WIDTH = 200.0
DETECTOR_HEIGHT = 200.0
DETECTOR_Z = -100.0

ANGLES_DEGREES = (0.0, 15.0)
INTERFACE_SEQUENCE = [(IOR_AIR, IOR_PET), (IOR_PET, IOR_AIR)]
RAY_GRID_BASE = {
    "origin_plane_z": 200.0,
    "x_range": (-50.0, 50.0),
    "y_range": (-70.0, 70.0),
}

BASELINE_DETECTOR_RESOLUTION = (40, 40)
BASELINE_RAY_GRID = (11, 11)
BASELINE_TOP_PERCENT = 1.0

DETECTOR_RESOLUTIONS = ((40, 40), (80, 80), (160, 160))
RAY_GRID_SIZES = ((11, 11), (21, 21), (41, 41))
TOP_PERCENT_VALUES = (1.0, 0.5, 0.1)

VALID_TERMINATIONS = frozenset(
    {
        "completed_interfaces",
        "no_active_rays",
        "no_interfaces",
        "max_steps_reached",
    }
)
SATURATION_TOL = 1e-9
WEIGHTED_LE_TOL = 1e-12


@dataclass(frozen=True)
class _CasePair:
    detector_resolution: tuple[int, int]
    ray_grid: tuple[int, int]
    top_percent: float
    unweighted: AngleScanResult
    weighted: AngleScanResult


@dataclass(frozen=True)
class _SweepBlock:
    name: str
    cases: tuple[_CasePair, ...]


def _run_one(
    mesh,
    detector,
    *,
    detector_resolution: tuple[int, int],
    ray_grid: tuple[int, int],
    top_percent: float,
    use_power_weights: bool,
) -> AngleScanResult:
    rgc = dict(RAY_GRID_BASE)
    rgc["nx"] = int(ray_grid[0])
    rgc["ny"] = int(ray_grid[1])
    detector_grid = create_detector_grid(
        width=DETECTOR_WIDTH,
        height=DETECTOR_HEIGHT,
        resolution=detector_resolution,
    )
    return run_baseline_angle_scan(
        mesh=mesh,
        angles_degrees=ANGLES_DEGREES,
        ray_grid_config=rgc,
        interface_sequence=INTERFACE_SEQUENCE,
        detector=detector,
        detector_grid=detector_grid,
        incident_reference=INCIDENT_REFERENCE,
        thresholds=HOTSPOT_THRESHOLDS,
        top_percent=top_percent,
        use_power_weights=use_power_weights,
    )


def _run_pair(
    mesh,
    detector,
    *,
    detector_resolution: tuple[int, int],
    ray_grid: tuple[int, int],
    top_percent: float,
) -> _CasePair:
    unweighted = _run_one(
        mesh, detector,
        detector_resolution=detector_resolution,
        ray_grid=ray_grid,
        top_percent=top_percent,
        use_power_weights=False,
    )
    weighted = _run_one(
        mesh, detector,
        detector_resolution=detector_resolution,
        ray_grid=ray_grid,
        top_percent=top_percent,
        use_power_weights=True,
    )
    return _CasePair(
        detector_resolution=tuple(detector_resolution),
        ray_grid=tuple(ray_grid),
        top_percent=float(top_percent),
        unweighted=unweighted,
        weighted=weighted,
    )


def _build_sweep_blocks(mesh, detector) -> tuple[_SweepBlock, ...]:
    a_cases = tuple(
        _run_pair(
            mesh, detector,
            detector_resolution=res,
            ray_grid=BASELINE_RAY_GRID,
            top_percent=BASELINE_TOP_PERCENT,
        )
        for res in DETECTOR_RESOLUTIONS
    )
    b_cases = tuple(
        _run_pair(
            mesh, detector,
            detector_resolution=BASELINE_DETECTOR_RESOLUTION,
            ray_grid=rg,
            top_percent=BASELINE_TOP_PERCENT,
        )
        for rg in RAY_GRID_SIZES
    )
    c_cases = tuple(
        _run_pair(
            mesh, detector,
            detector_resolution=BASELINE_DETECTOR_RESOLUTION,
            ray_grid=BASELINE_RAY_GRID,
            top_percent=tp,
        )
        for tp in TOP_PERCENT_VALUES
    )
    return (
        _SweepBlock("Sweep A: detector resolution", a_cases),
        _SweepBlock("Sweep B: ray grid density", b_cases),
        _SweepBlock("Sweep C: top percent", c_cases),
    )


def _print_summary(
    sweep_blocks: tuple[_SweepBlock, ...],
) -> tuple[int, int, int]:
    total_entries = sum(
        len(b.cases) * len(ANGLES_DEGREES) for b in sweep_blocks
    )

    print("Weighted metric convergence diagnostic demo")
    print(
        "Synthetic weighted metric-convergence smoke check; "
        "not a physical PET-bottle validation."
    )
    print(
        "Note: use_power_weights=True applies cumulative Fresnel "
        "transmission weights to detector accumulation."
    )
    print(
        "Note: this is still not pixel-area-normalized irradiance "
        "or calibrated W/m^2."
    )
    print(
        "Note: this demo diagnoses metric sensitivity, not pattern "
        "effectiveness."
    )
    print(
        "Note: it does not recommend any correct detector "
        "resolution, ray grid, or top_percent."
    )
    print(
        "Note: it does not prove fire prevention or PET-bottle "
        "safety."
    )
    print(
        "Note: weighted-minus-unweighted Delta reflects Fresnel "
        "weighting semantics, not a fire-risk-reduction claim."
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
    print(f"Detector z: {float(DETECTOR_Z)}")
    print(
        "Detector resolutions: "
        + ", ".join(str(r) for r in DETECTOR_RESOLUTIONS)
    )
    print("Ray grid sizes: " + ", ".join(str(r) for r in RAY_GRID_SIZES))
    print(
        "Top-percent values: "
        + ", ".join(str(v) for v in TOP_PERCENT_VALUES)
    )
    print(f"Result count: {total_entries}")
    print(
        "Per-case legend -> Resolution: detector pixel grid. "
        "Ray grid: incident ray grid. Top percent: C99 top-N% pixel "
        "mean. Unweighted C99: ray-count surrogate. Weighted C99: "
        "cumulative-Fresnel-T surrogate. Delta C99: Weighted minus "
        "Unweighted. Unweighted Cmax / Weighted Cmax / Delta Cmax: "
        "peak-pixel analogues. Detector hits: ray count at detector. "
        "Final rays: rays surviving trace."
    )

    unweighted_saturation = 0
    weighted_saturation = 0
    saw_strict_decrease = False

    for block in sweep_blocks:
        print("")
        print(block.name)
        for ai, angle in enumerate(ANGLES_DEGREES):
            print(f"Angle {angle}:")
            for case in block.cases:
                ue = case.unweighted.per_angle[ai]
                we = case.weighted.per_angle[ai]
                u_c99 = float(ue.metrics.c99)
                w_c99 = float(we.metrics.c99)
                u_cmax = float(ue.metrics.cmax)
                w_cmax = float(we.metrics.cmax)

                if abs(u_c99 - 1.0) <= SATURATION_TOL:
                    unweighted_saturation += 1
                if abs(w_c99 - 1.0) <= SATURATION_TOL:
                    weighted_saturation += 1
                if (
                    w_c99 < u_c99 - WEIGHTED_LE_TOL
                    or w_cmax < u_cmax - WEIGHTED_LE_TOL
                ):
                    saw_strict_decrease = True

                print(
                    f"  Resolution={case.detector_resolution} "
                    f"Ray grid={case.ray_grid} "
                    f"Top percent={case.top_percent} "
                    f"Unweighted C99={u_c99:.4f} "
                    f"Weighted C99={w_c99:.4f} "
                    f"Delta C99={(w_c99 - u_c99):+.4f} "
                    f"Unweighted Cmax={u_cmax:.4f} "
                    f"Weighted Cmax={w_cmax:.4f} "
                    f"Delta Cmax={(w_cmax - u_cmax):+.4f} "
                    f"Detector hits={int(ue.detector_hits)} "
                    f"Final rays={int(ue.final_ray_count)}"
                )

    print("")
    print(f"Unweighted saturation count: {unweighted_saturation}")
    print(f"Weighted saturation count: {weighted_saturation}")

    if unweighted_saturation > 0 and weighted_saturation > 0:
        body = (
            f"{unweighted_saturation} of {total_entries} unweighted "
            f"and {weighted_saturation} of {total_entries} weighted "
            f"cases saturated C99 within {SATURATION_TOL} of 1.0; "
            f"detector resolution / ray grid / top_percent may be too "
            f"coarse for the current ray-count or Fresnel-weighted "
            f"surrogate. C99 saturation at 1.0 is counted separately. "
            f"Cmax > 1 from dense ray grids is a separate quantization "
            f"mode not counted in the C99 saturation count."
        )
    elif unweighted_saturation > 0:
        body = (
            f"{unweighted_saturation} of {total_entries} unweighted "
            f"cases saturated C99 within {SATURATION_TOL} of 1.0; "
            f"Fresnel weighting moved every saturated case off 1.0 "
            f"within tolerance. C99 saturation at 1.0 is counted "
            f"separately. Cmax > 1 from dense ray grids is a separate "
            f"quantization mode not counted in the C99 saturation "
            f"count."
        )
    elif weighted_saturation > 0:
        body = (
            f"0 unweighted, {weighted_saturation} weighted of "
            f"{total_entries} cases saturated C99 within "
            f"{SATURATION_TOL} of 1.0; investigate whether "
            f"incident_reference is appropriate. C99 saturation at "
            f"1.0 is counted separately. Cmax > 1 from dense ray "
            f"grids is a separate quantization mode not counted in "
            f"the C99 saturation count."
        )
    else:
        body = (
            f"0 of {total_entries} unweighted and 0 of {total_entries} "
            f"weighted cases saturated C99 within {SATURATION_TOL} of "
            f"1.0; no C99 saturation detected at this tolerance. C99 "
            f"saturation at 1.0 is counted separately. Cmax > 1 from "
            f"dense ray grids is a separate quantization mode not "
            f"counted in the C99 saturation count."
        )
    print(f"Quantization warning: {body}")

    return total_entries, unweighted_saturation, weighted_saturation, saw_strict_decrease


def _check_invariants(
    sweep_blocks: tuple[_SweepBlock, ...],
    total_entries: int,
    unweighted_saturation: int,
    weighted_saturation: int,
    saw_strict_decrease: bool,
) -> bool:
    checks: list[bool] = []

    expected_total = 18
    checks.append(total_entries == expected_total)
    checks.append(len(sweep_blocks) == 3)
    for block in sweep_blocks:
        checks.append(len(block.cases) == 3)
        for case in block.cases:
            expected_ray_count = (
                int(case.ray_grid[0]) * int(case.ray_grid[1])
            )
            for which in (case.unweighted, case.weighted):
                checks.append(
                    int(which.angle_count) == len(ANGLES_DEGREES)
                )
                checks.append(
                    len(which.per_angle) == len(ANGLES_DEGREES)
                )
                for entry in which.per_angle:
                    m = entry.metrics
                    checks.append(int(entry.ray_count) == expected_ray_count)
                    checks.append(int(entry.final_ray_count) >= 0)
                    checks.append(
                        int(entry.final_ray_count) <= int(entry.ray_count)
                    )
                    checks.append(int(entry.detector_hits) >= 0)
                    checks.append(float(m.c99) >= 0.0)
                    checks.append(float(m.cmax) >= 0.0)
                    checks.append(float(m.peak_value) >= 0.0)
                    checks.append(2.0 in m.eexceed)
                    checks.append(2.0 in m.ahot)
                    checks.append(
                        entry.termination_reason in VALID_TERMINATIONS
                    )

            for ai in range(len(ANGLES_DEGREES)):
                ue = case.unweighted.per_angle[ai]
                we = case.weighted.per_angle[ai]
                # Geometry decoupling: weighting changes accumulation,
                # not ray paths or detector intersections.
                checks.append(int(ue.detector_hits) == int(we.detector_hits))
                checks.append(
                    int(ue.final_ray_count) == int(we.final_ray_count)
                )
                checks.append(int(ue.ray_count) == int(we.ray_count))
                checks.append(
                    ue.termination_reason == we.termination_reason
                )
                # Passive-dielectric Fresnel bound: cumulative T <= 1
                # with identical geometry implies weighted metrics
                # cannot exceed unweighted counterparts. Smoke-check
                # invariant for this fixture; not a general
                # optical-safety claim.
                u_c99 = float(ue.metrics.c99)
                w_c99 = float(we.metrics.c99)
                u_cmax = float(ue.metrics.cmax)
                w_cmax = float(we.metrics.cmax)
                checks.append(w_c99 <= u_c99 + WEIGHTED_LE_TOL)
                checks.append(w_cmax <= u_cmax + WEIGHTED_LE_TOL)

    checks.append(saw_strict_decrease)
    checks.append(int(unweighted_saturation) >= 0)
    checks.append(int(weighted_saturation) >= 0)

    return all(checks)


def main() -> int:
    mesh = create_synthetic_bottle_body(
        radius=BOTTLE_RADIUS,
        height=BOTTLE_HEIGHT,
        sections=BOTTLE_SECTIONS,
    )
    detector = create_detector_plane(
        center=(0.0, 0.0, DETECTOR_Z),
        normal=(0.0, 0.0, 1.0),
        up=(0.0, 1.0, 0.0),
        width=DETECTOR_WIDTH,
        height=DETECTOR_HEIGHT,
    )

    sweep_blocks = _build_sweep_blocks(mesh, detector)
    total_entries, u_sat, w_sat, saw_strict = _print_summary(sweep_blocks)
    ok = _check_invariants(
        sweep_blocks, total_entries, u_sat, w_sat, saw_strict
    )
    print(f"Invariants: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
