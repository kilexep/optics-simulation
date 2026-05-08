"""Legacy PET-water STL angle / detector-distance scan demo.

Actual STL legacy-style optical scan smoke check; not a physical
PET-bottle validation. Loads the supplied STL, scales it to the
legacy ``target_height`` / ``target_diameter`` dimensions, builds
the auto-direction inner offset water boundary, and runs
:func:`run_legacy_pet_water_angle_distance_scan` over a
caller-supplied set of ``--angles`` and ``--detector-distances``.
The trace is run **once per angle** and every requested detector
distance is evaluated against the surviving rays. Console summary
only; no visualization, no file output, no thermal simulation.

Scope (also enforced at runtime)
--------------------------------
- This scan uses the loaded STL as a geometry input.
- It uses target-dimension scaling and a generated inner offset
  mesh.
- It does not prove physical accuracy.
- It does not repair meshes.
- It does not infer material regions automatically.
- It does not perform thermal simulation.
- It does not prove fire prevention or PET-bottle safety.
- Detector intensity metrics are relative optical surrogates.

Usage::

    python examples/run_legacy_pet_water_stl_angle_scan_demo.py \\
        --mesh data/raw/stl/pet_bottle.stl \\
        --target-height 225.6 --target-diameter 72.1 \\
        --wall-thickness 0.3 --inner-offset-mode auto \\
        --sample-count-y 21 --sample-count-z 41 \\
        --angles "0,15,30,45" \\
        --detector-distances "100,140,180,220,260,300"

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
    LegacyOpticalScanResult,
    LegacyPetWaterTraceSetup,
    create_legacy_pet_water_trace_setup,
    run_legacy_pet_water_angle_distance_scan,
)


_DEFAULT_MESH_PATH = "data/raw/stl/pet_bottle.stl"


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Actual STL legacy-style optical scan smoke check; "
            "not a physical PET-bottle validation."
        ),
    )
    parser.add_argument(
        "--mesh", type=str, default=_DEFAULT_MESH_PATH,
        help="STL/mesh path (default data/raw/stl/pet_bottle.stl).",
    )
    parser.add_argument(
        "--target-height", type=float, default=225.6,
    )
    parser.add_argument(
        "--target-diameter", type=float, default=72.1,
    )
    parser.add_argument(
        "--wall-thickness", type=float, default=0.3,
    )
    parser.add_argument(
        "--inner-offset-mode",
        choices=("auto", "minus_normals", "plus_normals"),
        default="auto",
    )
    parser.add_argument(
        "--sample-count-y", type=int, default=21,
    )
    parser.add_argument(
        "--sample-count-z", type=int, default=41,
    )
    parser.add_argument(
        "--angles", type=str, default="0,15,30,45",
        help="Comma-separated list of angles in degrees.",
    )
    parser.add_argument(
        "--detector-distances", type=str,
        default="100,140,180,220,260,300",
        help="Comma-separated list of detector distances in mm.",
    )
    parser.add_argument(
        "--source-width", type=float, default=100.0,
    )
    parser.add_argument(
        "--source-height", type=float, default=250.0,
    )
    parser.add_argument(
        "--detector-size", type=float, default=400.0,
    )
    parser.add_argument(
        "--detector-resolution", type=int, default=40,
    )
    return parser


def _parse_float_list(text: str, name: str) -> tuple[float, ...]:
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    try:
        return tuple(float(p) for p in parts)
    except ValueError as exc:
        raise SystemExit(
            f"Failed to parse {name}={text!r}: {exc}"
        )


def _print_static_header() -> None:
    print("Legacy PET-water STL angle scan demo")
    print(
        "Actual STL legacy-style optical scan smoke check; not a "
        "physical PET-bottle validation."
    )
    print("Note: this scan uses the loaded STL as a geometry input.")
    print(
        "Note: it uses target-dimension scaling and a generated "
        "inner offset mesh."
    )
    print("Note: it does not prove physical accuracy.")
    print("Note: it does not repair meshes.")
    print(
        "Note: it does not infer material regions automatically."
    )
    print("Note: it does not perform thermal simulation.")
    print(
        "Note: it does not prove fire prevention or PET-bottle "
        "safety."
    )
    print(
        "Note: detector intensity metrics are relative optical "
        "surrogates."
    )


def _print_setup_block(
    *,
    args: argparse.Namespace,
    setup: LegacyPetWaterTraceSetup,
    angles: tuple[float, ...],
    distances: tuple[float, ...],
) -> None:
    print(f"Mesh path: {Path(args.mesh)}")
    print(f"Target height: {float(args.target_height):.4f}")
    print(f"Target diameter: {float(args.target_diameter):.4f}")
    print(f"Wall thickness: {float(args.wall_thickness):.4f}")
    print(
        f"Inner offset mode: "
        f"{str(setup.inner_offset_report.offset_mode)}"
    )
    print(
        f"Inward offset detected: "
        f"{bool(setup.inner_offset_report.inward_offset_detected)}"
    )
    print(
        f"Selected offset sign: "
        f"{float(setup.inner_offset_report.selected_offset_sign):+.0f}"
    )
    print(f"Angles: {list(angles)}")
    print(f"Detector distances: {list(distances)}")
    print(
        f"Source samples: "
        f"{int(args.sample_count_y)} x {int(args.sample_count_z)} "
        f"= {int(args.sample_count_y) * int(args.sample_count_z)}"
    )


def _print_scan_aggregates(
    *,
    result: LegacyOpticalScanResult,
) -> None:
    print(f"Result count: {int(len(result.entries))}")
    if result.max_relative_irradiance is not None:
        print(
            f"Max relative irradiance angle: "
            f"{float(result.max_relative_irradiance_angle):.4f}"
        )
        print(
            f"Max relative irradiance detector distance: "
            f"{float(result.max_relative_irradiance_detector_distance):.4f}"
        )
        print(
            f"Max relative irradiance: "
            f"{float(result.max_relative_irradiance):.6f}"
        )
    else:
        print("Max relative irradiance angle: None")
        print("Max relative irradiance detector distance: None")
        print("Max relative irradiance: None")
    if result.max_c99 is not None:
        print(f"Max C99 angle: {float(result.max_c99_angle):.4f}")
        print(
            f"Max C99 detector distance: "
            f"{float(result.max_c99_detector_distance):.4f}"
        )
        print(f"Max C99: {float(result.max_c99):.6f}")
    else:
        print("Max C99 angle: None")
        print("Max C99 detector distance: None")
        print("Max C99: None")


def _print_per_entry_block(
    *,
    result: LegacyOpticalScanResult,
) -> None:
    last_angle: float | None = None
    for entry in result.entries:
        if last_angle is None or entry.angle_degrees != last_angle:
            print(f"Angle {float(entry.angle_degrees):.4f}")
            last_angle = float(entry.angle_degrees)
        print(
            f"  Detector distance: "
            f"{float(entry.detector_distance):.4f}"
        )
        print(f"    Final rays: {int(entry.final_ray_count)}")
        print(f"    Detector hits: {int(entry.detector_hits)}")
        print(f"    C99: {float(entry.c99):.6f}")
        print(f"    Cmax: {float(entry.cmax):.6f}")
        print(
            f"    Termination: {str(entry.termination_reason)}"
        )


def _check_invariants(
    *,
    setup: LegacyPetWaterTraceSetup,
    result: LegacyOpticalScanResult,
    sample_count_y: int,
    sample_count_z: int,
    angle_count: int,
    distance_count: int,
) -> bool:
    checks: list[bool] = []
    checks.append(len(setup.step_specs) == 4)
    checks.append(int(angle_count * distance_count) == len(result.entries))
    checks.append(int(result.angle_count) == int(angle_count))
    checks.append(int(result.detector_count) == int(distance_count))
    expected_ray_count = int(sample_count_y) * int(sample_count_z)
    for entry in result.entries:
        checks.append(int(entry.ray_count) == expected_ray_count)
        checks.append(int(entry.final_ray_count) >= 0)
        checks.append(int(entry.detector_hits) >= 0)
        checks.append(float(entry.c99) >= 0.0)
        checks.append(float(entry.cmax) >= 0.0)
    if len(result.entries) > 0:
        checks.append(result.max_relative_irradiance is not None)
        checks.append(result.max_c99 is not None)
    return all(checks)


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    angles = _parse_float_list(args.angles, "--angles")
    distances = _parse_float_list(
        args.detector_distances, "--detector-distances",
    )

    _print_static_header()

    mesh = load_mesh(Path(args.mesh))
    setup = create_legacy_pet_water_trace_setup(
        mesh,
        target_height=float(args.target_height),
        target_diameter=float(args.target_diameter),
        wall_thickness=float(args.wall_thickness),
        inner_offset_mode=str(args.inner_offset_mode),
    )
    _print_setup_block(
        args=args, setup=setup,
        angles=angles, distances=distances,
    )

    result = run_legacy_pet_water_angle_distance_scan(
        setup=setup,
        angles_degrees=angles,
        detector_distances=distances,
        source_width=float(args.source_width),
        source_height=float(args.source_height),
        sample_count_y=int(args.sample_count_y),
        sample_count_z=int(args.sample_count_z),
        detector_size=float(args.detector_size),
        detector_resolution=(
            int(args.detector_resolution),
            int(args.detector_resolution),
        ),
    )

    _print_scan_aggregates(result=result)
    _print_per_entry_block(result=result)

    ok = _check_invariants(
        setup=setup,
        result=result,
        sample_count_y=int(args.sample_count_y),
        sample_count_z=int(args.sample_count_z),
        angle_count=len(angles),
        distance_count=len(distances),
    )
    print(f"Invariants: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
