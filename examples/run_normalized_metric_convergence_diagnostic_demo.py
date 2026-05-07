"""Normalized metric convergence diagnostic demo.

Synthetic normalized metric-convergence smoke check; not a
physical PET-bottle validation. Three independent 1-D sweeps
(detector resolution, ray grid density, ``top_percent``) at a
fixed ``(mesh, angles, detector_z)`` baseline, this time running
each case **three** ways:

  1. raw_unweighted: use_power_weights=False,
     use_relative_irradiance=False (ray-count surrogate).
  2. raw_weighted: use_power_weights=True,
     use_relative_irradiance=False (cumulative-Fresnel-T on raw
     weight map).
  3. normalized_weighted: use_power_weights=True,
     use_relative_irradiance=True (cumulative-Fresnel-T plus
     source-plane / ray-count / detector-pixel-area
     normalization).

Per-case Raw weighted / Normalized weighted / Delta values for
``C99`` and ``Cmax`` are reported side-by-side along with matched
``Detector hits`` / ``Final rays``. A ``Quantization warning``
line surfaces normalized-side ``C99`` saturation at 1.0.

Interpretation notes (also printed at runtime):
- ``use_power_weights=True`` applies cumulative Fresnel
  transmission weights to detector accumulation.
- ``use_relative_irradiance=True`` uses source-plane area, ray
  count, and detector pixel area to compute a relative
  irradiance surrogate.
- This is **still not a calibrated W/m^2 measurement**.
- This demo diagnoses metric sensitivity, not pattern
  effectiveness.
- It does not recommend any correct detector resolution, ray
  grid, or ``top_percent``.
- It does not prove fire prevention or PET-bottle safety.
- Delta normalized-vs-raw values reflect the normalization
  factor source_area / (ray_count * pixel_area), not a
  fire-risk-reduction claim.

Out of scope
------------
Same disclaimers as ``run_weighted_metric_convergence_diagnostic_demo``.
No STL / OBJ / CAD / STEP export, no file write, no
visualization, no thermal model, no original-vs-displaced
comparison, no pattern generation, no mesh displacement, no
optimization, no manufacturability validation.

Run from the repository root:

    python examples/run_normalized_metric_convergence_diagnostic_demo.py

The script exits 0 when every invariant holds and 1 otherwise.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

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
INCIDENT_REFERENCE = 1.0
INCIDENT_IRRADIANCE = 1.0

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
class _CaseTriple:
    detector_resolution: tuple[int, int]
    ray_grid: tuple[int, int]
    top_percent: float
    raw_unweighted: AngleScanResult
    raw_weighted: AngleScanResult
    normalized_weighted: AngleScanResult


@dataclass(frozen=True)
class _SweepBlock:
    name: str
    cases: tuple[_CaseTriple, ...]


def _run_one(
    mesh,
    detector,
    *,
    detector_resolution: tuple[int, int],
    ray_grid: tuple[int, int],
    top_percent: float,
    use_power_weights: bool,
    use_relative_irradiance: bool,
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
        use_relative_irradiance=use_relative_irradiance,
        incident_irradiance=INCIDENT_IRRADIANCE,
    )


def _run_triple(
    mesh, detector,
    *,
    detector_resolution: tuple[int, int],
    ray_grid: tuple[int, int],
    top_percent: float,
) -> _CaseTriple:
    raw_unweighted = _run_one(
        mesh, detector,
        detector_resolution=detector_resolution,
        ray_grid=ray_grid,
        top_percent=top_percent,
        use_power_weights=False,
        use_relative_irradiance=False,
    )
    raw_weighted = _run_one(
        mesh, detector,
        detector_resolution=detector_resolution,
        ray_grid=ray_grid,
        top_percent=top_percent,
        use_power_weights=True,
        use_relative_irradiance=False,
    )
    normalized_weighted = _run_one(
        mesh, detector,
        detector_resolution=detector_resolution,
        ray_grid=ray_grid,
        top_percent=top_percent,
        use_power_weights=True,
        use_relative_irradiance=True,
    )
    return _CaseTriple(
        detector_resolution=tuple(detector_resolution),
        ray_grid=tuple(ray_grid),
        top_percent=float(top_percent),
        raw_unweighted=raw_unweighted,
        raw_weighted=raw_weighted,
        normalized_weighted=normalized_weighted,
    )


def _build_sweep_blocks(mesh, detector) -> tuple[_SweepBlock, ...]:
    a_cases = tuple(
        _run_triple(
            mesh, detector,
            detector_resolution=res,
            ray_grid=BASELINE_RAY_GRID,
            top_percent=BASELINE_TOP_PERCENT,
        )
        for res in DETECTOR_RESOLUTIONS
    )
    b_cases = tuple(
        _run_triple(
            mesh, detector,
            detector_resolution=BASELINE_DETECTOR_RESOLUTION,
            ray_grid=rg,
            top_percent=BASELINE_TOP_PERCENT,
        )
        for rg in RAY_GRID_SIZES
    )
    c_cases = tuple(
        _run_triple(
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
) -> tuple[int, int]:
    total_entries = sum(
        len(b.cases) * len(ANGLES_DEGREES) for b in sweep_blocks
    )

    print("Normalized metric convergence diagnostic demo")
    print(
        "Synthetic normalized metric-convergence smoke check; "
        "not a physical PET-bottle validation."
    )
    print(
        "Note: use_power_weights=True applies cumulative Fresnel "
        "transmission weights to detector accumulation."
    )
    print(
        "Note: use_relative_irradiance=True uses source-plane area, "
        "ray count, and detector pixel area to compute a relative "
        "irradiance surrogate."
    )
    print(
        "Note: this is still not a calibrated W/m^2 measurement."
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
        "Note: Delta normalized-vs-raw values reflect the "
        "normalization factor source_area / (ray_count * "
        "pixel_area), not a fire-risk-reduction claim."
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
    print(
        "Ray grid sizes: " + ", ".join(str(r) for r in RAY_GRID_SIZES)
    )
    print(
        "Top-percent values: "
        + ", ".join(str(v) for v in TOP_PERCENT_VALUES)
    )
    print(f"Result count: {total_entries}")
    print(
        "Per-case legend -> Raw weighted: cumulative-Fresnel-T on "
        "raw weight_map. Normalized weighted: same Fresnel-T weights "
        "after source-area / ray-count / pixel-area normalization. "
        "Delta normalized-vs-raw = Normalized minus Raw. Detector "
        "hits / Final rays are identical between raw and normalized "
        "(geometry is unchanged)."
    )

    normalized_saturation = 0
    for block in sweep_blocks:
        print("")
        print(block.name)
        for ai, angle in enumerate(ANGLES_DEGREES):
            print(f"Angle {angle}:")
            for case in block.cases:
                rw = case.raw_weighted.per_angle[ai]
                nw = case.normalized_weighted.per_angle[ai]
                rw_c99 = float(rw.metrics.c99)
                nw_c99 = float(nw.metrics.c99)
                rw_cmax = float(rw.metrics.cmax)
                nw_cmax = float(nw.metrics.cmax)
                if abs(nw_c99 - 1.0) <= SATURATION_TOL:
                    normalized_saturation += 1
                print(
                    f"  Resolution={case.detector_resolution} "
                    f"Ray grid={case.ray_grid} "
                    f"Top percent={case.top_percent} "
                    f"Raw weighted C99={rw_c99:.4f} "
                    f"Normalized weighted C99={nw_c99:.4f} "
                    f"Delta normalized-vs-raw C99="
                    f"{(nw_c99 - rw_c99):+.4f} "
                    f"Raw weighted Cmax={rw_cmax:.4f} "
                    f"Normalized weighted Cmax={nw_cmax:.4f} "
                    f"Delta normalized-vs-raw Cmax="
                    f"{(nw_cmax - rw_cmax):+.4f} "
                    f"Detector hits={int(rw.detector_hits)} "
                    f"Final rays={int(rw.final_ray_count)}"
                )

    print("")
    print(f"Normalized saturation count: {normalized_saturation}")
    if normalized_saturation > 0:
        body = (
            f"{normalized_saturation} of {total_entries} normalized "
            f"weighted cases saturated C99 within {SATURATION_TOL} "
            f"of 1.0; investigate whether source_area / ray_count / "
            f"pixel_area combination produces a 1.0 fixed point. "
            f"Cmax > 1 from dense ray grids is a separate "
            f"quantization mode not counted here."
        )
    else:
        body = (
            f"0 of {total_entries} normalized weighted cases "
            f"saturated C99 within {SATURATION_TOL} of 1.0; "
            f"normalization broke the 1.0 saturation pattern at "
            f"this fixture. Cmax > 1 from dense ray grids is a "
            f"separate quantization mode not counted here."
        )
    print(f"Quantization warning: {body}")

    return total_entries, normalized_saturation


def _check_invariants(
    sweep_blocks: tuple[_SweepBlock, ...],
    total_entries: int,
    normalized_saturation: int,
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
            for ai in range(len(ANGLES_DEGREES)):
                ru = case.raw_unweighted.per_angle[ai]
                rw = case.raw_weighted.per_angle[ai]
                nw = case.normalized_weighted.per_angle[ai]
                # Geometry decoupling: weighting / normalization
                # changes accumulation, not ray paths.
                checks.append(int(rw.detector_hits) == int(nw.detector_hits))
                checks.append(
                    int(rw.final_ray_count) == int(nw.final_ray_count)
                )
                checks.append(int(ru.ray_count) == expected_ray_count)
                checks.append(int(rw.ray_count) == expected_ray_count)
                checks.append(int(nw.ray_count) == expected_ray_count)
                checks.append(rw.termination_reason in VALID_TERMINATIONS)
                checks.append(nw.termination_reason in VALID_TERMINATIONS)

                checks.append(ru.irradiance_surrogate is None)
                checks.append(rw.irradiance_surrogate is None)
                s = nw.irradiance_surrogate
                checks.append(s is not None)
                if s is not None:
                    checks.append(
                        np.isfinite(s.relative_irradiance_map).all()
                    )
                    checks.append(
                        (s.relative_irradiance_map >= 0).all()
                    )

                checks.append(float(ru.metrics.c99) >= 0.0)
                checks.append(float(rw.metrics.c99) >= 0.0)
                checks.append(float(nw.metrics.c99) >= 0.0)
                checks.append(float(ru.metrics.cmax) >= 0.0)
                checks.append(float(rw.metrics.cmax) >= 0.0)
                checks.append(float(nw.metrics.cmax) >= 0.0)

    checks.append(int(normalized_saturation) >= 0)

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
    total_entries, n_sat = _print_summary(sweep_blocks)
    ok = _check_invariants(sweep_blocks, total_entries, n_sat)
    print(f"Invariants: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
