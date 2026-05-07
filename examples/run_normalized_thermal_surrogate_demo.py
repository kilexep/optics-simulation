"""Normalized thermal surrogate demo.

Synthetic normalized optical-to-thermal surrogate smoke check;
not a physical PET-bottle fire-prevention validation. Wires
together: synthetic solid-cylinder bottle-like mesh -> per-angle
parallel ray grid -> multi-step trace with Fresnel cumulative
weighting -> detector accumulation -> source-plane / ray-count /
detector-pixel-area normalization to a relative irradiance
surrogate -> selection of a single representative scalar
``max_relative_irradiance`` -> conversion to a thermal incident
flux surrogate via a ``NOMINAL_INCIDENT_IRRADIANCE_W_M2``
multiplier -> 0-D lumped-capacitance target heating with
optional convective + grey-body radiative loss.

Interpretation notes (also printed at runtime):
- This uses nominal incident irradiance only for scale.
- This is **not calibrated W/m^2 validation**.
- This is **not** ignition validation.
- The threshold is illustrative, **not** a material ignition
  threshold.
- This does **not** prove fire prevention or PET-bottle safety.

The thermal model consumes an optical flux surrogate; the input
is **not calibrated W/m^2** unless the caller provides
calibrated incident irradiance. The demo's
``NOMINAL_INCIDENT_IRRADIANCE_W_M2 = 1000.0`` is an illustrative
unit only — it provides a numerical scale for the heating curve
and does not anchor the result to any calibrated solar flux,
spectrum, or material absorptivity.

Limitations
-----------
- Synthetic solid-cylinder fixture; not a real PET bottle STL.
- Single PET interface sequence ``[(AIR, PET), (PET, AIR)]``;
  no shell, no wall thickness, no fluid medium.
- 0-D lumped-capacitance target; no spatial conduction, no
  multi-layer stack, no moisture, no pyrolysis chemistry, no
  ignition criterion.
- Constant scalar incident flux; no time-varying source, no
  spectral absorptivity, no view factor.
- No STL / OBJ / CAD / STEP export, no file write, no
  visualization, no CSV / Parquet logging.

Run from the repository root:

    python examples/run_normalized_thermal_surrogate_demo.py

The script exits 0 when every invariant holds and 1 otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from optics_simulation.angle_scan import (
    AngleDistanceScanResult,
    PerAngleDistanceResult,
    run_angle_distance_sweep,
)
from optics_simulation.geometry import create_synthetic_bottle_body
from optics_simulation.thermal import (
    LumpedTargetHeatingResult,
    simulate_lumped_target_heating,
)


IOR_AIR = 1.00028
IOR_PET = 1.575
HOTSPOT_THRESHOLDS = (2.0, 5.0, 10.0)
TOP_PERCENT_FOR_C99 = 1.0
INCIDENT_REFERENCE = 1.0
INCIDENT_IRRADIANCE_NORMALIZATION = 1.0  # surrogate, unit-less

BOTTLE_RADIUS = 30.0
BOTTLE_HEIGHT = 120.0
BOTTLE_SECTIONS = 96

ANGLES_DEGREES = (0.0, 15.0, 30.0)
DETECTOR_Z_VALUES = (-80.0, -100.0, -120.0)

RAY_GRID_NX = 11
RAY_GRID_NY = 11
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

# Thermal surrogate scale and material-style constants.
NOMINAL_INCIDENT_IRRADIANCE_W_M2 = 1000.0  # illustrative scale only
DURATION_S = 60.0
DT_S = 0.1
AREAL_HEAT_CAPACITY_J_M2K = 1200.0
ABSORPTIVITY = 0.8
H_CONV_W_M2K = 10.0
EMISSIVITY = 0.9
AMBIENT_TEMP_K = 293.15
THRESHOLD_TEMP_K = 373.15  # illustrative only; not a material ignition point


def _run_normalized_sweep(mesh) -> AngleDistanceScanResult:
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
        use_power_weights=True,
        use_relative_irradiance=True,
        incident_irradiance=INCIDENT_IRRADIANCE_NORMALIZATION,
    )


def _select_max_relative_entry(
    sweep: AngleDistanceScanResult,
) -> tuple[PerAngleDistanceResult, float]:
    best_entry = sweep.per_result[0]
    best_value = -np.inf
    for entry in sweep.per_result:
        s = entry.irradiance_surrogate
        if s is None:
            continue
        local_max = float(s.relative_irradiance_map.max())
        if local_max > best_value:
            best_value = local_max
            best_entry = entry
    return best_entry, max(best_value, 0.0)


def _print_summary(
    sweep: AngleDistanceScanResult,
    selected: PerAngleDistanceResult,
    max_relative_irradiance: float,
    max_relative_c99: float,
    thermal: LumpedTargetHeatingResult,
) -> None:
    print("Normalized thermal surrogate demo")
    print(
        "Synthetic normalized optical-to-thermal surrogate smoke "
        "check; not a physical PET-bottle fire-prevention validation."
    )
    print(
        "Note: this uses nominal incident irradiance only for scale."
    )
    print(
        "Note: this is not calibrated W/m^2 validation."
    )
    print(
        "Note: this is not ignition validation; the lumped-target "
        "model is a simplified target-heating surrogate, not "
        "ignition validation model."
    )
    print(
        "Note: the threshold is illustrative, not a material "
        "ignition threshold."
    )
    print(
        "Note: this does not prove fire prevention or PET-bottle "
        "safety."
    )
    print(
        "Note: no STL / OBJ / CAD / STEP export, no file write, no "
        "visualization is performed."
    )
    print(
        f"Mode: use_power_weights=True, use_relative_irradiance=True"
    )
    print(f"Angles: {', '.join(str(a) for a in ANGLES_DEGREES)}")
    print(
        "Detector z values: "
        + ", ".join(str(z) for z in DETECTOR_Z_VALUES)
    )
    print(f"Result count: {len(sweep.per_result)}")
    print(f"Max relative irradiance: {float(max_relative_irradiance):.4f}")
    print(f"Max relative C99: {float(max_relative_c99):.4f}")
    print(f"Selected angle: {float(selected.angle_degrees)}")
    print(f"Selected detector z: {float(selected.detector_z)}")
    print(
        f"Nominal incident irradiance: "
        f"{float(NOMINAL_INCIDENT_IRRADIANCE_W_M2):.4f}"
    )
    print(
        f"Thermal incident flux surrogate: "
        f"{float(thermal.incident_flux_w_m2):.4f}"
    )
    print(
        f"Absorbed flux surrogate: "
        f"{float(thermal.absorbed_flux_w_m2):.4f}"
    )
    print(
        f"Areal heat capacity: "
        f"{float(thermal.areal_heat_capacity_j_m2k):.4f}"
    )
    print(f"Absorptivity: {float(thermal.absorptivity):.4f}")
    print(f"h conv: {float(thermal.h_conv_w_m2k):.4f}")
    print(f"Emissivity: {float(thermal.emissivity):.4f}")
    print(f"Duration: {float(thermal.duration_s):.4f}")
    print(f"dt: {float(thermal.dt_s):.4f}")
    print(f"Initial temperature: {float(thermal.initial_temp_k):.4f}")
    print(f"Ambient temperature: {float(thermal.ambient_temp_k):.4f}")
    print(f"Max temperature: {float(thermal.max_temperature_k):.4f}")
    print(f"Final temperature: {float(thermal.final_temperature_k):.4f}")
    print(
        f"Max temperature rise: "
        f"{float(thermal.max_temperature_rise_k):+.4f}"
    )
    print(
        f"Final temperature rise: "
        f"{float(thermal.final_temperature_rise_k):+.4f}"
    )
    if thermal.time_to_threshold_s is None:
        print("Time to threshold: None")
    else:
        print(
            f"Time to threshold: "
            f"{float(thermal.time_to_threshold_s):.4f}"
        )


def _check_invariants(
    sweep: AngleDistanceScanResult,
    max_relative_irradiance: float,
    thermal: LumpedTargetHeatingResult,
) -> bool:
    checks: list[bool] = []

    checks.append(len(sweep.per_result) > 0)
    for entry in sweep.per_result:
        checks.append(entry.irradiance_surrogate is not None)
        s = entry.irradiance_surrogate
        if s is not None:
            checks.append(np.isfinite(s.relative_irradiance_map).all())
            checks.append((s.relative_irradiance_map >= 0).all())

    checks.append(float(max_relative_irradiance) >= 0.0)
    checks.append(float(thermal.incident_flux_w_m2) >= 0.0)
    checks.append(float(thermal.absorbed_flux_w_m2) >= 0.0)
    checks.append(np.isfinite(thermal.temperature_k).all())
    checks.append(thermal.temperature_k.size > 0)

    if thermal.incident_flux_w_m2 > 0.0:
        checks.append(
            float(thermal.max_temperature_k)
            >= float(thermal.initial_temp_k) - 1e-12
        )
    checks.append(
        float(thermal.final_temperature_k)
        == float(thermal.temperature_k[-1])
    )
    checks.append(
        float(thermal.max_temperature_k)
        == float(thermal.temperature_k.max())
    )

    if thermal.time_to_threshold_s is not None:
        t = float(thermal.time_to_threshold_s)
        checks.append(0.0 <= t <= float(thermal.duration_s))

    return all(checks)


def main() -> int:
    mesh = create_synthetic_bottle_body(
        radius=BOTTLE_RADIUS,
        height=BOTTLE_HEIGHT,
        sections=BOTTLE_SECTIONS,
    )
    sweep = _run_normalized_sweep(mesh)

    selected, max_relative_irradiance = _select_max_relative_entry(sweep)
    max_relative_c99 = max(
        float(p.metrics.c99) for p in sweep.per_result
    )

    incident_flux = (
        NOMINAL_INCIDENT_IRRADIANCE_W_M2 * float(max_relative_irradiance)
    )
    thermal = simulate_lumped_target_heating(
        incident_flux_w_m2=incident_flux,
        duration_s=DURATION_S,
        dt_s=DT_S,
        areal_heat_capacity_j_m2k=AREAL_HEAT_CAPACITY_J_M2K,
        absorptivity=ABSORPTIVITY,
        h_conv_w_m2k=H_CONV_W_M2K,
        emissivity=EMISSIVITY,
        ambient_temp_k=AMBIENT_TEMP_K,
        threshold_temp_k=THRESHOLD_TEMP_K,
    )

    _print_summary(
        sweep, selected, max_relative_irradiance, max_relative_c99, thermal,
    )
    ok = _check_invariants(sweep, max_relative_irradiance, thermal)
    print(f"Invariants: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
