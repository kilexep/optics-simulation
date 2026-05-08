"""Legacy PET-water STL parity-scan demo.

Legacy experiment parity optical scan; not a physical PET-bottle
validation. Reproduces the deterministic angle / detector-distance
sweep schedule from the old interactive matplotlib PET-bottle
experiment as a console-only summary, using the current testable
optics framework. The interactive UI, detector heatmap images,
slider, and any kind of file export are intentionally NOT
reproduced.

Pipeline:

1. Load the supplied STL.
2. Build the legacy four-step PET-water trace setup with the
   normal-direction autodetected inner offset.
3. Build the legacy parity schedule (default: 72 angles at 5
   degree steps, 20 detector distances at 20 mm spacing starting
   from 100 mm).
4. Run the angle / detector-distance scan against the schedule.
5. Summarize totals and print a few representative entries.

Scope (also enforced at runtime)
--------------------------------
- This reproduces the old experiment's sweep schedule inside the
  current framework.
- It does not reproduce the interactive matplotlib visualization.
- It does not prove physical accuracy.
- It does not perform thermal simulation.
- It does not repair meshes.
- It does not prove fire prevention or PET-bottle safety.
- Metrics are relative optical surrogates.

Usage::

    python examples/run_legacy_pet_water_stl_parity_scan_demo.py \\
        --mesh data/raw/stl/pet_bottle.stl \\
        --target-height 225.6 --target-diameter 72.1 \\
        --wall-thickness 0.3 --inner-offset-mode auto \\
        --angle-count 72 --angle-step 5 \\
        --detector-count 20 --detector-start 100 \\
        --detector-spacing 20 \\
        --sample-count-y 21 --sample-count-z 41 \\
        --detector-resolution 40

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
    LegacyParityScanSummary,
    LegacyParitySchedule,
    LegacyPetWaterTraceSetup,
    create_legacy_experiment_schedule,
    create_legacy_pet_water_trace_setup,
    run_legacy_pet_water_angle_distance_scan,
    summarize_legacy_optical_scan,
)


_DEFAULT_MESH_PATH = "data/raw/stl/pet_bottle.stl"


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Legacy experiment parity optical scan; not a physical "
            "PET-bottle validation."
        ),
    )
    parser.add_argument(
        "--mesh", type=str, default=_DEFAULT_MESH_PATH,
        help="STL/mesh path (default data/raw/stl/pet_bottle.stl).",
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
    return parser


def _print_static_header() -> None:
    print("Legacy PET-water STL parity scan demo")
    print(
        "Legacy experiment parity optical scan; not a physical "
        "PET-bottle validation."
    )
    print(
        "Note: this reproduces the old experiment's sweep schedule "
        "inside the current framework."
    )
    print(
        "Note: it does not reproduce the interactive matplotlib "
        "visualization."
    )
    print("Note: it does not prove physical accuracy.")
    print("Note: it does not perform thermal simulation.")
    print("Note: it does not repair meshes.")
    print(
        "Note: it does not prove fire prevention or PET-bottle "
        "safety."
    )
    print("Note: metrics are relative optical surrogates.")


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
    print(
        f"Inward offset detected: "
        f"{bool(setup.inner_offset_report.inward_offset_detected)}"
    )
    print(f"Angle count: {int(schedule.angle_count)}")
    print(f"Angle step: {float(args.angle_step):.4f}")
    print(f"Detector count: {int(schedule.detector_count)}")
    print(f"Detector start: {float(schedule.detector_start):.4f}")
    print(f"Detector spacing: {float(schedule.detector_spacing):.4f}")
    print(
        f"Source samples: "
        f"{int(schedule.sample_count_y)} x "
        f"{int(schedule.sample_count_z)} = "
        f"{int(schedule.sample_count_y) * int(schedule.sample_count_z)}"
    )


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "None"
    return f"{float(value):.6f}"


def _format_optional_angle(value: float | None) -> str:
    if value is None:
        return "None"
    return f"{float(value):.4f}"


def _print_summary_block(
    *,
    summary: LegacyParityScanSummary,
) -> None:
    print(f"Result count: {int(summary.total_entries)}")
    print(f"Total detector hits: {int(summary.total_detector_hits)}")
    print(f"Nonzero entries: {int(summary.nonzero_entry_count)}")
    print(f"Zero-hit entries: {int(summary.zero_hit_entry_count)}")
    print(f"Max detector hits: {int(summary.max_detector_hits)}")
    print(
        f"Max detector hits angle: "
        f"{_format_optional_angle(summary.max_detector_hits_angle)}"
    )
    print(
        f"Max detector hits distance: "
        f"{_format_optional_angle(summary.max_detector_hits_distance)}"
    )
    print(
        f"Max relative irradiance: "
        f"{_format_optional_float(summary.max_relative_irradiance)}"
    )
    print(
        f"Max relative irradiance angle: "
        f"{_format_optional_angle(summary.max_relative_irradiance_angle)}"
    )
    print(
        f"Max relative irradiance distance: "
        f"{_format_optional_angle(summary.max_relative_irradiance_distance)}"
    )
    print(f"Max C99: {_format_optional_float(summary.max_c99)}")
    print(
        f"Max C99 angle: "
        f"{_format_optional_angle(summary.max_c99_angle)}"
    )
    print(
        f"Max C99 distance: "
        f"{_format_optional_angle(summary.max_c99_distance)}"
    )


def _format_entry_line(label: str, entry: LegacyOpticalScanEntry) -> str:
    return (
        f"  {label}: angle={float(entry.angle_degrees):.4f}, "
        f"distance={float(entry.detector_distance):.4f}, "
        f"final_rays={int(entry.final_ray_count)}, "
        f"detector_hits={int(entry.detector_hits)}, "
        f"c99={float(entry.c99):.6f}, "
        f"cmax={float(entry.cmax):.6f}, "
        f"max_rel_irr={float(entry.max_relative_irradiance):.6f}"
    )


def _print_representative_entries(
    *,
    result: LegacyOpticalScanResult,
    summary: LegacyParityScanSummary,
) -> None:
    print("Representative entries")
    if not result.entries:
        print("  (no entries)")
        return

    seen_keys: set[tuple[float, float]] = set()

    def _emit(label: str, entry: LegacyOpticalScanEntry) -> None:
        key = (float(entry.angle_degrees), float(entry.detector_distance))
        if key in seen_keys:
            return
        seen_keys.add(key)
        print(_format_entry_line(label, entry))

    _emit("first", result.entries[0])

    if (
        summary.max_detector_hits_angle is not None
        and summary.max_detector_hits_distance is not None
    ):
        for entry in result.entries:
            if (
                float(entry.angle_degrees)
                == float(summary.max_detector_hits_angle)
                and float(entry.detector_distance)
                == float(summary.max_detector_hits_distance)
            ):
                _emit("max_detector_hits", entry)
                break

    if (
        summary.max_relative_irradiance_angle is not None
        and summary.max_relative_irradiance_distance is not None
    ):
        for entry in result.entries:
            if (
                float(entry.angle_degrees)
                == float(summary.max_relative_irradiance_angle)
                and float(entry.detector_distance)
                == float(summary.max_relative_irradiance_distance)
            ):
                _emit("max_relative_irradiance", entry)
                break

    if (
        summary.max_c99_angle is not None
        and summary.max_c99_distance is not None
    ):
        for entry in result.entries:
            if (
                float(entry.angle_degrees)
                == float(summary.max_c99_angle)
                and float(entry.detector_distance)
                == float(summary.max_c99_distance)
            ):
                _emit("max_c99", entry)
                break


def _check_invariants(
    *,
    args: argparse.Namespace,
    setup: LegacyPetWaterTraceSetup,
    schedule: LegacyParitySchedule,
    result: LegacyOpticalScanResult,
    summary: LegacyParityScanSummary,
) -> bool:
    checks: list[bool] = []
    checks.append(len(setup.step_specs) == 4)
    checks.append(int(schedule.angle_count) == int(args.angle_count))
    checks.append(
        int(schedule.detector_count) == int(args.detector_count)
    )
    checks.append(
        int(len(result.entries))
        == int(schedule.angle_count) * int(schedule.detector_count)
    )
    checks.append(int(summary.total_entries) == int(len(result.entries)))
    checks.append(int(summary.total_detector_hits) >= 0)
    checks.append(
        int(summary.nonzero_entry_count)
        + int(summary.zero_hit_entry_count)
        == int(summary.total_entries)
    )
    if int(summary.total_entries) > 0:
        checks.append(summary.max_relative_irradiance is not None)
        checks.append(summary.max_c99 is not None)
        checks.append(summary.max_detector_hits_angle is not None)
        checks.append(summary.max_detector_hits_distance is not None)
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

    result = run_legacy_pet_water_angle_distance_scan(
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
    )
    summary = summarize_legacy_optical_scan(result, schedule)

    _print_summary_block(summary=summary)
    _print_representative_entries(result=result, summary=summary)

    ok = _check_invariants(
        args=args, setup=setup, schedule=schedule,
        result=result, summary=summary,
    )
    print(f"Invariants: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
