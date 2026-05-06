"""Metric convergence diagnostic demo.

Synthetic metric-convergence smoke check; not a physical
PET-bottle validation. Probes how the optical caustic-risk
metrics (``C99`` / ``Cmax``) of the existing baseline angle scan
respond to three independent setup axes — detector pixel
resolution, ray grid density, and ``top_percent`` — at a fixed
``(mesh, angles, detector_z)`` baseline. The aim is to surface
quantization / saturation behavior of the current synthetic
setup, **not** to recommend a "correct" resolution and **not**
to claim any fire-prevention or PET-bottle validation outcome.

This demo probes metric quantization at fixed mesh, angles, and
detector z. It does not recommend any correct detector
resolution, ray grid, or ``top_percent``. It does not claim
improvement, fire prevention, or PET-bottle validation.

Background note (already visible in
``run_angle_distance_sweep_demo`` output): at detector
resolution ``(40, 40)``, ray grid ``(11, 11)``, and
``top_percent=1.0``, ``C99`` saturates at ``1.0`` once
``detector_hits >= 16`` because the top 1% of 1600 pixels is 16
and each ray hits a distinct pixel with weight ``1.0``. The
underlying detector pipeline currently ignores Fresnel
reflectance / transmittance, so ``C99`` and ``Cmax`` are
ray-count surrogates rather than power surrogates. Both factors
limit how meaningfully the present metric can compare any two
caustic configurations. This demo makes that limitation
diagnosable; it does **not** fix it.

Limitations
-----------
- Synthetic solid-cylinder fixture; not a real PET bottle STL.
- Single PET interface sequence ``[(AIR, PET), (PET, AIR)]``;
  no shell, no wall thickness, no fluid medium.
- Detector pipeline weights every detector hit as ``1.0`` —
  Fresnel power weighting is **not** introduced by this demo.
- No STL / OBJ / CAD / STEP export, no file write, no
  visualization, no CSV / Parquet logging, no thermal model,
  no original-vs-displaced comparison, no pattern generation,
  no mesh displacement, no manufacturability validation, no
  optimization.

Run from the repository root:

    python examples/run_metric_convergence_diagnostic_demo.py

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


@dataclass(frozen=True)
class _Case:
    detector_resolution: tuple[int, int]
    ray_grid: tuple[int, int]
    top_percent: float
    scan: AngleScanResult


@dataclass(frozen=True)
class _SweepBlock:
    name: str
    cases: tuple[_Case, ...]


def _run_one_case(
    mesh,
    detector,
    *,
    detector_resolution: tuple[int, int],
    ray_grid: tuple[int, int],
    top_percent: float,
) -> _Case:
    rgc = dict(RAY_GRID_BASE)
    rgc["nx"] = int(ray_grid[0])
    rgc["ny"] = int(ray_grid[1])
    detector_grid = create_detector_grid(
        width=DETECTOR_WIDTH,
        height=DETECTOR_HEIGHT,
        resolution=detector_resolution,
    )
    scan = run_baseline_angle_scan(
        mesh=mesh,
        angles_degrees=ANGLES_DEGREES,
        ray_grid_config=rgc,
        interface_sequence=INTERFACE_SEQUENCE,
        detector=detector,
        detector_grid=detector_grid,
        incident_reference=INCIDENT_REFERENCE,
        thresholds=HOTSPOT_THRESHOLDS,
        top_percent=top_percent,
    )
    return _Case(
        detector_resolution=tuple(detector_resolution),
        ray_grid=tuple(ray_grid),
        top_percent=float(top_percent),
        scan=scan,
    )


def _build_sweep_blocks(mesh, detector) -> tuple[_SweepBlock, ...]:
    a_cases = tuple(
        _run_one_case(
            mesh, detector,
            detector_resolution=res,
            ray_grid=BASELINE_RAY_GRID,
            top_percent=BASELINE_TOP_PERCENT,
        )
        for res in DETECTOR_RESOLUTIONS
    )
    b_cases = tuple(
        _run_one_case(
            mesh, detector,
            detector_resolution=BASELINE_DETECTOR_RESOLUTION,
            ray_grid=rg,
            top_percent=BASELINE_TOP_PERCENT,
        )
        for rg in RAY_GRID_SIZES
    )
    c_cases = tuple(
        _run_one_case(
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


def _print_summary(sweep_blocks: tuple[_SweepBlock, ...]) -> int:
    total_entries = sum(
        len(b.cases) * len(ANGLES_DEGREES) for b in sweep_blocks
    )

    print("Metric convergence diagnostic demo")
    print(
        "Synthetic metric-convergence smoke check; "
        "not a physical PET-bottle validation."
    )
    print(
        "Note: this demo probes metric quantization at fixed mesh, "
        "angles, and detector z; it does not recommend any correct "
        "resolution."
    )
    print(
        "Note: not a physical PET-bottle risk assessment and not a "
        "fire-prevention proof."
    )
    print(
        "Note: no STL / OBJ / CAD / STEP export, no file write, no "
        "visualization is performed."
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
        "Ray grid: incident ray grid. "
        "Top percent: C99 top-N% pixel mean. "
        "C99: top-N% pixel mean ratio. "
        "Cmax: peak pixel ratio. "
        "Detector hits: ray count at detector. "
        "Final rays: rays surviving trace."
    )

    saturated_total = 0
    saturated_per_sweep: dict[str, int] = {}

    for block in sweep_blocks:
        print("")
        print(block.name)
        per_block_sat = 0
        for ai, angle in enumerate(ANGLES_DEGREES):
            print(f"Angle {angle}:")
            previous_c99: float | None = None
            previous_cmax: float | None = None
            for case in block.cases:
                entry = case.scan.per_angle[ai]
                m = entry.metrics
                c99 = float(m.c99)
                cmax = float(m.cmax)

                if abs(c99 - 1.0) <= SATURATION_TOL:
                    saturated_total += 1
                    per_block_sat += 1

                if previous_c99 is None:
                    delta_c99_str = "n/a"
                    delta_cmax_str = "n/a"
                else:
                    delta_c99_str = f"{c99 - previous_c99:+.4f}"
                    delta_cmax_str = f"{cmax - previous_cmax:+.4f}"

                print(
                    f"  Resolution={case.detector_resolution} "
                    f"Ray grid={case.ray_grid} "
                    f"Top percent={case.top_percent} "
                    f"C99={c99:.4f} "
                    f"Cmax={cmax:.4f} "
                    f"Detector hits={int(entry.detector_hits)} "
                    f"Final rays={int(entry.final_ray_count)} "
                    f"Delta C99 vs previous={delta_c99_str} "
                    f"Delta Cmax vs previous={delta_cmax_str}"
                )

                previous_c99 = c99
                previous_cmax = cmax

        saturated_per_sweep[block.name] = per_block_sat

    print("")
    if saturated_total > 0:
        per_sweep_str = ", ".join(
            f"{name.split(':')[0]}: {count}"
            for name, count in saturated_per_sweep.items()
        )
        print(
            f"Quantization warning: {saturated_total} of {total_entries} "
            f"cases saturated C99 within {SATURATION_TOL} of 1.0 "
            f"({per_sweep_str}); detector resolution / ray grid / "
            f"top_percent may be too coarse for the current ray-count "
            f"surrogate."
        )
    else:
        print(
            f"Quantization warning: 0 of {total_entries} cases saturated "
            f"C99 within {SATURATION_TOL} of 1.0; no C99 saturation "
            f"detected at this tolerance."
        )

    return total_entries


def _check_invariants(
    sweep_blocks: tuple[_SweepBlock, ...],
    total_entries: int,
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
            checks.append(int(case.scan.angle_count) == len(ANGLES_DEGREES))
            checks.append(
                len(case.scan.per_angle) == len(ANGLES_DEGREES)
            )
            for entry in case.scan.per_angle:
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
                checks.append(entry.termination_reason in VALID_TERMINATIONS)

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
    total_entries = _print_summary(sweep_blocks)
    ok = _check_invariants(sweep_blocks, total_entries)
    print(f"Invariants: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
