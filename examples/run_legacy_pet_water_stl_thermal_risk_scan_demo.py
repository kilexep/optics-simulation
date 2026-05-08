"""Legacy PET-water STL optical-to-thermal-risk scan demo.

Actual STL optical-to-thermal risk surrogate scan; not a physical
PET-bottle validation. Loads the supplied STL, scales it to the
legacy ``target_height`` / ``target_diameter`` dimensions, builds
the auto-direction inner offset water boundary, runs the legacy
parity angle / detector-distance optical scan with per-entry
:class:`DetectorIrradianceSurrogate` storage, and then forwards
each entry's relative irradiance map into the per-pixel
lumped-target heating surrogate to produce
:class:`ThermalRiskMetrics`. Console summary only; no
visualization, no heatmap image export, no thermal calibration.

Old H.max (raw detector-bin count) is **NOT** treated as a primary
metric here; the canonical reductions are the relative irradiance
surrogate, ``C99``, and the thermal-risk metrics.

Scope (also enforced at runtime)
--------------------------------
- This uses the actual loaded STL as geometry input.
- The STL is target-dimension normalized and uses a generated
  inner water mesh.
- The optical map is a relative irradiance surrogate.
- The thermal map is a per-pixel lumped target-heating surrogate.
- Threshold values are illustrative unless calibrated by
  experiment.
- This is not pyrolysis.
- This is not CFD.
- This is not fire-prevention validation.
- Do not use old H.max as a primary research metric.
- Do not claim PET-bottle safety.

Usage::

    python examples/run_legacy_pet_water_stl_thermal_risk_scan_demo.py \\
        --mesh data/raw/stl/pet_bottle.stl \\
        --target-height 225.6 --target-diameter 72.1 \\
        --wall-thickness 0.3 --inner-offset-mode auto \\
        --angle-count 72 --angle-step 5 \\
        --detector-count 20 --detector-start 100 --detector-spacing 20 \\
        --sample-count-y 21 --sample-count-z 41 \\
        --detector-resolution 40 \\
        --duration 60 --dt 0.5

The script exits 0 when every invariant holds and 1 otherwise.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from optics_simulation.geometry import load_mesh
from optics_simulation.optics import (
    LegacyOpticalScanEntry,
    LegacyOpticalScanResult,
    LegacyParitySchedule,
    LegacyPetWaterTraceSetup,
    create_legacy_experiment_schedule,
    create_legacy_pet_water_trace_setup,
    run_legacy_pet_water_angle_distance_scan,
    summarize_legacy_optical_scan,
)
from optics_simulation.thermal import (
    LegacyThermalRiskEntry,
    LegacyThermalRiskScanResult,
    compute_thermal_risk_over_legacy_optical_scan,
)


_DEFAULT_MESH_PATH = "data/raw/stl/pet_bottle.stl"


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Actual STL optical-to-thermal risk surrogate scan; "
            "not a physical PET-bottle validation."
        ),
    )
    parser.add_argument(
        "--mesh", type=str, default=_DEFAULT_MESH_PATH,
    )
    parser.add_argument("--target-height", type=float, default=225.6)
    parser.add_argument("--target-diameter", type=float, default=72.1)
    parser.add_argument("--wall-thickness", type=float, default=0.3)
    parser.add_argument(
        "--inner-offset-mode",
        choices=("auto", "minus_normals", "plus_normals"),
        default="auto",
    )
    parser.add_argument("--angle-count", type=int, default=72)
    parser.add_argument("--angle-step", type=float, default=5.0)
    parser.add_argument("--detector-count", type=int, default=20)
    parser.add_argument("--detector-start", type=float, default=100.0)
    parser.add_argument(
        "--detector-spacing", type=float, default=20.0,
    )
    parser.add_argument("--source-width", type=float, default=100.0)
    parser.add_argument("--source-height", type=float, default=250.0)
    parser.add_argument("--source-radius", type=float, default=200.0)
    parser.add_argument("--sample-count-y", type=int, default=21)
    parser.add_argument("--sample-count-z", type=int, default=41)
    parser.add_argument("--detector-size", type=float, default=400.0)
    parser.add_argument(
        "--detector-resolution", type=int, default=40,
    )
    parser.add_argument(
        "--nominal-incident-irradiance",
        type=float, default=1000.0,
    )
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument(
        "--areal-heat-capacity", type=float, default=1200.0,
    )
    parser.add_argument("--absorptivity", type=float, default=0.8)
    parser.add_argument("--h-conv", type=float, default=10.0)
    parser.add_argument("--emissivity", type=float, default=0.9)
    parser.add_argument(
        "--threshold-temp", type=float, default=373.15,
    )
    return parser


def _print_static_header() -> None:
    print("Legacy PET-water STL thermal risk scan demo")
    print(
        "Actual STL optical-to-thermal risk surrogate scan; not a "
        "physical PET-bottle validation."
    )
    print(
        "Note: this uses the actual loaded STL as geometry input."
    )
    print(
        "Note: the STL is target-dimension normalized and uses a "
        "generated inner water mesh."
    )
    print(
        "Note: the optical map is a relative irradiance surrogate."
    )
    print(
        "Note: the thermal map is a per-pixel lumped target-"
        "heating surrogate."
    )
    print(
        "Note: threshold values are illustrative unless calibrated "
        "by experiment."
    )
    print("Note: this is not pyrolysis.")
    print("Note: this is not CFD.")
    print("Note: this is not fire-prevention validation.")
    print(
        "Note: old H.max raw detector-bin count is not used as a "
        "primary research metric."
    )
    print("Note: this does not claim PET-bottle safety.")


def _print_setup_block(
    *,
    args: argparse.Namespace,
    setup: LegacyPetWaterTraceSetup,
    schedule: LegacyParitySchedule,
) -> None:
    print(f"Mesh path: {Path(args.mesh)}")
    print(f"Target height: {float(args.target_height):.4f}")
    print(f"Target diameter: {float(args.target_diameter):.4f}")
    print(f"Wall thickness: {float(args.wall_thickness):.4f}")
    print(
        f"Inner offset mode: "
        f"{str(setup.inner_offset_report.offset_mode)}"
    )
    print(f"Angles: {list(schedule.angles_degrees)}")
    print(f"Detector distances: {list(schedule.detector_distances)}")
    print(
        f"Nominal incident irradiance: "
        f"{float(args.nominal_incident_irradiance):.4f}"
    )
    print(f"Threshold temperature: {float(args.threshold_temp):.4f}")


def _format_optional_float(value, fmt: str = "{:.6f}") -> str:
    if value is None:
        return "None"
    return fmt.format(float(value))


def _format_optional_angle(value) -> str:
    if value is None:
        return "None"
    return f"{float(value):.4f}"


def _print_optical_summary(
    *,
    optical_result: LegacyOpticalScanResult,
) -> None:
    print(f"Result count: {int(len(optical_result.entries))}")
    print(
        f"Max relative irradiance: "
        f"{_format_optional_float(optical_result.max_relative_irradiance)}"
    )
    print(
        f"Max C99: "
        f"{_format_optional_float(optical_result.max_c99)}"
    )


def _print_thermal_summary(
    *,
    thermal_result: LegacyThermalRiskScanResult,
) -> None:
    print(
        f"Max temperature: "
        f"{_format_optional_float(thermal_result.max_temperature_k)}"
    )
    print(
        f"Max temperature angle: "
        f"{_format_optional_angle(thermal_result.max_temperature_angle)}"
    )
    print(
        f"Max temperature distance: "
        f"{_format_optional_angle(thermal_result.max_temperature_distance)}"
    )
    print(
        f"Max top-percent temperature rise: "
        f"{_format_optional_float(thermal_result.max_top_percent_temperature_rise_k)}"
    )
    print(
        f"Max top-percent temperature rise angle: "
        f"{_format_optional_angle(thermal_result.max_top_percent_temperature_rise_angle)}"
    )
    print(
        f"Max top-percent temperature rise distance: "
        f"{_format_optional_angle(thermal_result.max_top_percent_temperature_rise_distance)}"
    )
    if thermal_result.max_threshold_exceeded_count is None:
        print("Max threshold exceeded count: None")
    else:
        print(
            f"Max threshold exceeded count: "
            f"{int(thermal_result.max_threshold_exceeded_count)}"
        )
    print(
        f"Max threshold exceeded count angle: "
        f"{_format_optional_angle(thermal_result.max_threshold_exceeded_count_angle)}"
    )
    print(
        f"Max threshold exceeded count distance: "
        f"{_format_optional_angle(thermal_result.max_threshold_exceeded_count_distance)}"
    )


def _format_thermal_entry_line(
    label: str, optical: LegacyOpticalScanEntry,
    thermal: LegacyThermalRiskEntry,
) -> str:
    tm = thermal.thermal_metrics
    return (
        f"  {label}: angle={float(thermal.angle_degrees):.4f}, "
        f"distance={float(thermal.detector_distance):.4f}, "
        f"detector_hits={int(thermal.detector_hits)}, "
        f"c99={float(thermal.optical_c99):.6f}, "
        f"max_rel_irr={float(thermal.max_relative_irradiance):.6f}, "
        f"max_temp={float(tm.max_temperature_k):.4f}, "
        f"top_percent_rise={float(tm.top_percent_max_temperature_rise_k):.4f}, "
        f"threshold_exceeded={int(tm.threshold_exceeded_count)}"
    )


def _find_thermal_entry(
    *,
    thermal_result: LegacyThermalRiskScanResult,
    angle: float | None,
    distance: float | None,
) -> LegacyThermalRiskEntry | None:
    if angle is None or distance is None:
        return None
    for entry in thermal_result.entries:
        if (
            float(entry.angle_degrees) == float(angle)
            and float(entry.detector_distance) == float(distance)
        ):
            return entry
    return None


def _print_representative_entries(
    *,
    optical_result: LegacyOpticalScanResult,
    thermal_result: LegacyThermalRiskScanResult,
) -> None:
    print("Representative entries")
    if not thermal_result.entries:
        print("  (no entries)")
        return

    seen_keys: set[tuple[float, float]] = set()

    def _emit(label: str, optical_entry, thermal_entry) -> None:
        key = (
            float(thermal_entry.angle_degrees),
            float(thermal_entry.detector_distance),
        )
        if key in seen_keys:
            return
        seen_keys.add(key)
        print(
            _format_thermal_entry_line(
                label, optical_entry, thermal_entry,
            )
        )

    _emit("first", optical_result.entries[0], thermal_result.entries[0])

    candidates = [
        (
            "max_relative_irradiance",
            optical_result.max_relative_irradiance_angle,
            optical_result.max_relative_irradiance_detector_distance,
        ),
        (
            "max_c99",
            optical_result.max_c99_angle,
            optical_result.max_c99_detector_distance,
        ),
        (
            "max_temperature",
            thermal_result.max_temperature_angle,
            thermal_result.max_temperature_distance,
        ),
        (
            "max_threshold_count",
            thermal_result.max_threshold_exceeded_count_angle,
            thermal_result.max_threshold_exceeded_count_distance,
        ),
    ]
    for label, angle, distance in candidates:
        if angle is None or distance is None:
            continue
        thermal_entry = _find_thermal_entry(
            thermal_result=thermal_result,
            angle=angle, distance=distance,
        )
        if thermal_entry is None:
            continue
        # Find matching optical entry by angle/distance.
        for opt_entry in optical_result.entries:
            if (
                float(opt_entry.angle_degrees) == float(angle)
                and float(opt_entry.detector_distance) == float(distance)
            ):
                _emit(label, opt_entry, thermal_entry)
                break


def _check_invariants(
    *,
    args: argparse.Namespace,
    optical_result: LegacyOpticalScanResult,
    thermal_result: LegacyThermalRiskScanResult,
) -> bool:
    import numpy as np

    checks: list[bool] = []
    expected = int(args.angle_count) * int(args.detector_count)
    checks.append(int(len(optical_result.entries)) == expected)
    checks.append(
        int(thermal_result.entry_count) == int(len(optical_result.entries))
    )
    for entry in optical_result.entries:
        checks.append(entry.irradiance_surrogate is not None)
    for thermal_entry in thermal_result.entries:
        tm = thermal_entry.thermal_metrics
        checks.append(np.isfinite(float(tm.max_temperature_k)))
        checks.append(
            np.isfinite(float(tm.top_percent_max_temperature_rise_k))
        )
    if int(thermal_result.entry_count) > 0:
        checks.append(thermal_result.max_temperature_k is not None)
        checks.append(
            thermal_result.max_threshold_exceeded_count is not None
        )
    return all(checks)


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    _print_static_header()

    mesh = load_mesh(Path(args.mesh))
    setup = create_legacy_pet_water_trace_setup(
        mesh,
        target_height=float(args.target_height),
        target_diameter=float(args.target_diameter),
        wall_thickness=float(args.wall_thickness),
        inner_offset_mode=str(args.inner_offset_mode),
    )
    schedule = create_legacy_experiment_schedule(
        angle_count=int(args.angle_count),
        angle_step_degrees=float(args.angle_step),
        detector_count=int(args.detector_count),
        detector_start=float(args.detector_start),
        detector_spacing=float(args.detector_spacing),
        detector_size=float(args.detector_size),
        source_width=float(args.source_width),
        source_height=float(args.source_height),
        source_radius=float(args.source_radius),
        sample_count_y=int(args.sample_count_y),
        sample_count_z=int(args.sample_count_z),
    )

    _print_setup_block(args=args, setup=setup, schedule=schedule)

    optical_result = run_legacy_pet_water_angle_distance_scan(
        setup=setup,
        angles_degrees=schedule.angles_degrees,
        detector_distances=schedule.detector_distances,
        source_width=schedule.source_width,
        source_height=schedule.source_height,
        sample_count_y=schedule.sample_count_y,
        sample_count_z=schedule.sample_count_z,
        detector_size=schedule.detector_size,
        detector_resolution=(
            int(args.detector_resolution),
            int(args.detector_resolution),
        ),
        source_radius=schedule.source_radius,
        epsilon=0.1,
        store_irradiance_surrogate=True,
    )
    summarize_legacy_optical_scan(optical_result, schedule)

    thermal_result = compute_thermal_risk_over_legacy_optical_scan(
        optical_result,
        nominal_incident_irradiance_w_m2=float(
            args.nominal_incident_irradiance
        ),
        duration_s=float(args.duration),
        dt_s=float(args.dt),
        areal_heat_capacity_j_m2k=float(args.areal_heat_capacity),
        absorptivity=float(args.absorptivity),
        h_conv_w_m2k=float(args.h_conv),
        emissivity=float(args.emissivity),
        threshold_temp_k=float(args.threshold_temp),
    )

    _print_optical_summary(optical_result=optical_result)
    _print_thermal_summary(thermal_result=thermal_result)
    _print_representative_entries(
        optical_result=optical_result,
        thermal_result=thermal_result,
    )

    ok = _check_invariants(
        args=args,
        optical_result=optical_result,
        thermal_result=thermal_result,
    )
    print(f"Invariants: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
