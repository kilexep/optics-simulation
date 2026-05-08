"""Actual STL original-vs-patterned PET-water thermal-risk demo.

Actual STL patterned optical-to-thermal risk smoke check; not a
physical PET-bottle validation. Loads the supplied STL, scales it
to the legacy ``target_height`` / ``target_diameter`` dimensions,
builds the auto-direction inner offset water boundary, applies a
synthetic uniform Gaussian dimple pattern to a diagnostic
body-region mask of the scaled outer shell, and runs the legacy
four-step ``air -> PET -> water -> PET -> air`` optical-to-thermal
scan over both the original and the patterned setups. Console-only
summary; the printed deltas are diagnostic comparisons, not
manufacturing or fire-prevention validation.

Old H.max raw detector-bin count is **NOT** used as a primary
metric. Canonical reductions are the relative irradiance
surrogate, ``C99``, and the per-pixel thermal-risk metrics.

Scope (also enforced at runtime)
--------------------------------
- This uses an actual STL as a geometry input.
- The STL is target-dimension normalized.
- Patterning is applied to a diagnostic body-region mask, not a
  verified manufacturing surface.
- Normal-direction handling is geometric robustness, not physical
  validation.
- The generated inner mesh is a legacy-style water boundary, not
  a measured wall thickness.
- Thermal-risk metrics are surrogate metrics.
- Delta values are comparison diagnostics only.
- A negative delta is not required.
- Do not claim fire prevention or PET-bottle safety.

Usage::

    python examples/run_actual_stl_patterned_thermal_risk_demo.py \\
        --mesh data/raw/stl/pet_bottle.stl \\
        --target-height 225.6 --target-diameter 72.1 \\
        --wall-thickness 0.3 --inner-offset-mode auto \\
        --angle-count 36 --angle-step 10 \\
        --detector-count 10 --detector-start 100 --detector-spacing 40 \\
        --sample-count-y 15 --sample-count-z 31 \\
        --detector-resolution 40 \\
        --pattern-count 20 --pattern-sigma 0.03 \\
        --pattern-max-depth 0.05 --pattern-seed 42 \\
        --duration 60 --dt 0.5

The script exits 0 when every invariant holds and 1 otherwise.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from optics_simulation.geometry import load_mesh
from optics_simulation.optics import (
    LegacyOpticalScanResult,
    LegacyPatternedPetWaterSetup,
    LegacyPetWaterTraceSetup,
    create_legacy_experiment_schedule,
    create_legacy_patterned_pet_water_setup,
    run_legacy_pet_water_angle_distance_scan,
)
from optics_simulation.thermal import (
    LegacyThermalRiskScanResult,
    compute_thermal_risk_over_legacy_optical_scan,
)


_DEFAULT_MESH_PATH = "data/raw/stl/pet_bottle.stl"


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Actual STL patterned optical-to-thermal risk smoke "
            "check; not a physical PET-bottle validation."
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
    parser.add_argument("--pattern-count", type=int, default=20)
    parser.add_argument("--pattern-sigma", type=float, default=0.03)
    parser.add_argument(
        "--pattern-max-depth", type=float, default=0.05,
    )
    parser.add_argument("--pattern-seed", type=int, default=42)
    parser.add_argument("--angle-count", type=int, default=36)
    parser.add_argument("--angle-step", type=float, default=10.0)
    parser.add_argument("--detector-count", type=int, default=10)
    parser.add_argument("--detector-start", type=float, default=100.0)
    parser.add_argument(
        "--detector-spacing", type=float, default=40.0,
    )
    parser.add_argument("--source-width", type=float, default=100.0)
    parser.add_argument("--source-height", type=float, default=250.0)
    parser.add_argument("--source-radius", type=float, default=200.0)
    parser.add_argument("--sample-count-y", type=int, default=15)
    parser.add_argument("--sample-count-z", type=int, default=31)
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
    print("Actual STL patterned thermal risk demo")
    print(
        "Actual STL patterned optical-to-thermal risk smoke "
        "check; not a physical PET-bottle validation."
    )
    print("Note: this uses an actual STL as a geometry input.")
    print("Note: the STL is target-dimension normalized.")
    print(
        "Note: patterning is applied to a diagnostic body-region "
        "mask, not a verified manufacturing surface."
    )
    print(
        "Note: normal-direction handling is geometric robustness, "
        "not physical validation."
    )
    print(
        "Note: the generated inner mesh is a legacy-style water "
        "boundary, not a measured wall thickness."
    )
    print("Note: thermal-risk metrics are surrogate metrics.")
    print(
        "Note: delta values are comparison diagnostics only; a "
        "negative delta is not required."
    )
    print(
        "Note: do not claim fire prevention or PET-bottle safety."
    )


def _build_patterned_legacy_setup(
    setup: LegacyPatternedPetWaterSetup,
) -> LegacyPetWaterTraceSetup:
    """Reuse :class:`LegacyPetWaterTraceSetup` so the existing scan runner
    can consume the patterned shell and water meshes without changes.
    """
    original = setup.original_setup
    return LegacyPetWaterTraceSetup(
        shell_mesh=setup.patterned_shell_mesh,
        water_mesh=setup.patterned_water_mesh,
        step_specs=setup.patterned_step_specs,
        scale_report=original.scale_report,
        inner_offset_report=setup.patterned_inner_offset_report,
        target_height=original.target_height,
        target_diameter=original.target_diameter,
        wall_thickness=original.wall_thickness,
        ior_air=original.ior_air,
        ior_pet=original.ior_pet,
        ior_water=original.ior_water,
    )


def _format_optional(value, fmt: str = "{:.6f}") -> str:
    if value is None:
        return "None"
    return fmt.format(float(value))


def _format_optional_int(value) -> str:
    if value is None:
        return "None"
    return str(int(value))


def _print_delta_block(
    *,
    setup: LegacyPatternedPetWaterSetup,
    args: argparse.Namespace,
    original_optical: LegacyOpticalScanResult,
    patterned_optical: LegacyOpticalScanResult,
    original_thermal: LegacyThermalRiskScanResult,
    patterned_thermal: LegacyThermalRiskScanResult,
) -> None:
    body_mask = setup.body_mask_report
    pattern = setup.patterned_mesh_result
    print(f"Mesh path: {Path(args.mesh)}")
    print(f"Target height: {float(args.target_height):.4f}")
    print(f"Target diameter: {float(args.target_diameter):.4f}")
    print(f"Wall thickness: {float(args.wall_thickness):.4f}")
    print(
        f"Body mask selected vertices: "
        f"{int(body_mask.selected_count)} / "
        f"{int(body_mask.vertex_count)}"
    )
    print(
        f"Pattern moved vertices: "
        f"{int(pattern.moved_vertex_count)}"
    )
    print(
        f"Pattern max depth: {float(args.pattern_max_depth):.6f}"
    )
    print(
        f"Selected displacement sign: "
        f"{float(pattern.selected_offset_sign):+.0f}"
    )
    print(
        f"Inward displacement detected: "
        f"{bool(pattern.inward_displacement_detected)}"
    )

    original_max_temp = original_thermal.max_temperature_k
    patterned_max_temp = patterned_thermal.max_temperature_k
    if (
        original_max_temp is not None
        and patterned_max_temp is not None
    ):
        delta_max_temp = float(patterned_max_temp) - float(
            original_max_temp,
        )
    else:
        delta_max_temp = None

    original_thresh = original_thermal.max_threshold_exceeded_count
    patterned_thresh = patterned_thermal.max_threshold_exceeded_count
    if (
        original_thresh is not None
        and patterned_thresh is not None
    ):
        delta_thresh = int(patterned_thresh) - int(original_thresh)
    else:
        delta_thresh = None

    original_max_c99 = original_optical.max_c99
    patterned_max_c99 = patterned_optical.max_c99
    if original_max_c99 is not None and patterned_max_c99 is not None:
        delta_c99 = float(patterned_max_c99) - float(
            original_max_c99,
        )
    else:
        delta_c99 = None

    print(
        f"Original max temperature: "
        f"{_format_optional(original_max_temp)}"
    )
    print(
        f"Patterned max temperature: "
        f"{_format_optional(patterned_max_temp)}"
    )
    if delta_max_temp is None:
        print("Delta max temperature: None")
    else:
        print(f"Delta max temperature: {delta_max_temp:+.6f}")

    print(
        f"Original threshold exceeded count: "
        f"{_format_optional_int(original_thresh)}"
    )
    print(
        f"Patterned threshold exceeded count: "
        f"{_format_optional_int(patterned_thresh)}"
    )
    if delta_thresh is None:
        print("Delta threshold exceeded count: None")
    else:
        print(f"Delta threshold exceeded count: {delta_thresh:+d}")

    print(
        f"Original max C99: {_format_optional(original_max_c99)}"
    )
    print(
        f"Patterned max C99: {_format_optional(patterned_max_c99)}"
    )
    if delta_c99 is None:
        print("Delta max C99: None")
    else:
        print(f"Delta max C99: {delta_c99:+.6f}")


def _check_invariants(
    *,
    setup: LegacyPatternedPetWaterSetup,
    original_optical: LegacyOpticalScanResult,
    patterned_optical: LegacyOpticalScanResult,
    original_thermal: LegacyThermalRiskScanResult,
    patterned_thermal: LegacyThermalRiskScanResult,
) -> bool:
    checks: list[bool] = []
    checks.append(int(setup.body_mask_report.selected_count) > 0)
    checks.append(
        int(setup.patterned_mesh_result.moved_vertex_count) > 0
    )
    checks.append(
        int(len(original_optical.entries))
        == int(len(patterned_optical.entries))
    )
    checks.append(
        int(original_thermal.entry_count)
        == int(patterned_thermal.entry_count)
    )
    if int(len(original_optical.entries)) > 0:
        checks.append(
            np.isfinite(
                float(original_thermal.max_temperature_k or 0.0)
            )
        )
        checks.append(
            np.isfinite(
                float(patterned_thermal.max_temperature_k or 0.0)
            )
        )
    return all(checks)


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    _print_static_header()

    mesh = load_mesh(Path(args.mesh))
    patterned_setup = create_legacy_patterned_pet_water_setup(
        mesh,
        target_height=float(args.target_height),
        target_diameter=float(args.target_diameter),
        wall_thickness=float(args.wall_thickness),
        inner_offset_mode=str(args.inner_offset_mode),
        pattern_count=int(args.pattern_count),
        pattern_sigma_u=float(args.pattern_sigma),
        pattern_sigma_v=float(args.pattern_sigma),
        pattern_max_depth=float(args.pattern_max_depth),
        pattern_seed=int(args.pattern_seed),
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

    scan_kwargs = dict(
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

    original_optical = run_legacy_pet_water_angle_distance_scan(
        setup=patterned_setup.original_setup,
        **scan_kwargs,
    )
    patterned_legacy_setup = _build_patterned_legacy_setup(
        patterned_setup,
    )
    patterned_optical = run_legacy_pet_water_angle_distance_scan(
        setup=patterned_legacy_setup,
        **scan_kwargs,
    )

    thermal_kwargs = dict(
        nominal_incident_irradiance_w_m2=float(
            args.nominal_incident_irradiance,
        ),
        duration_s=float(args.duration),
        dt_s=float(args.dt),
        areal_heat_capacity_j_m2k=float(args.areal_heat_capacity),
        absorptivity=float(args.absorptivity),
        h_conv_w_m2k=float(args.h_conv),
        emissivity=float(args.emissivity),
        threshold_temp_k=float(args.threshold_temp),
    )
    original_thermal = compute_thermal_risk_over_legacy_optical_scan(
        original_optical, **thermal_kwargs,
    )
    patterned_thermal = compute_thermal_risk_over_legacy_optical_scan(
        patterned_optical, **thermal_kwargs,
    )

    _print_delta_block(
        setup=patterned_setup,
        args=args,
        original_optical=original_optical,
        patterned_optical=patterned_optical,
        original_thermal=original_thermal,
        patterned_thermal=patterned_thermal,
    )

    ok = _check_invariants(
        setup=patterned_setup,
        original_optical=original_optical,
        patterned_optical=patterned_optical,
        original_thermal=original_thermal,
        patterned_thermal=patterned_thermal,
    )
    print(f"Invariants: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
