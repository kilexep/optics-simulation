"""Synthetic shell medium-tracking envelope diagnostic demo.

Synthetic shell medium-tracking validity smoke check; not a
physical PET-bottle validation. Reports how the medium-tracking
validation flag from
:func:`run_surface_classified_shell_trace` moves with the side
aperture and the per-step ``epsilon`` chosen by the caller. The
demo runs three named cases on the same synthetic cylindrical
shell fixture and the same direction:

- ``paraxial_epsilon_0p1``: the validated paraxial envelope
  (``y in [-25, 25]``, ``z in [-50, 50]``, ``epsilon = 0.1``).
  Expected to pass for both fill states.
- ``paraxial_epsilon_tiny``: the same paraxial aperture with
  the multi-step default ``epsilon = 1e-6``. May or may not
  pass; faceted-mesh chord deviation can place the next-bounce
  origin within the same outer-face polygon. Reported as a
  diagnostic only, not asserted.
- ``wide_epsilon_0p1``: the wider aperture used by earlier
  shell demos (``y in [-50, 50]``, ``z in [-70, 70]``) with
  ``epsilon = 0.1``. Grazing rays at the outer edge of the
  bottle do not always cross the cavity, so the canonical
  ``outer -> inner -> inner -> outer`` sequence may be
  violated. Reported as a diagnostic only, not asserted.

Scope (also enforced at runtime)
--------------------------------
- The validated shell preset applies only to the synthetic
  cylindrical shell fixture.
- The paraxial side-incidence aperture is a controlled smoke
  test envelope.
- Wide or grazing ray bundles may violate the canonical
  ``outer -> inner -> inner -> outer`` sequence; that violation
  is **diagnostic**, not a code failure.
- This is **not** general STL medium tracking.
- This does **not** prove fire prevention or PET-bottle safety.

Run from the repository root::

    python examples/run_shell_medium_tracking_envelope_demo.py

The script exits 0 when the strict invariants hold and 1
otherwise. The wide-envelope and tiny-epsilon cases are reported
but **not** asserted; only the paraxial-envelope-with-epsilon-0.1
case is required to pass.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from optics_simulation.geometry import (
    create_subdivided_synthetic_bottle_shell,
)
from optics_simulation.optics import (
    RayBundle,
    make_ray_bundle,
    run_surface_classified_shell_trace,
)


OUTER_RADIUS = 30.0
WALL_THICKNESS = 1.0
HEIGHT = 120.0
SECTIONS = 96
HEIGHT_SEGMENTS = 24

ORIGIN_X = 100.0
RAY_NY = 11
RAY_NZ = 11

PARAXIAL_Y_RANGE = (-25.0, 25.0)
PARAXIAL_Z_RANGE = (-50.0, 50.0)
WIDE_Y_RANGE = (-50.0, 50.0)
WIDE_Z_RANGE = (-70.0, 70.0)

EPSILON_PARAXIAL = 0.1
EPSILON_TINY = 1e-6
EPSILON_WIDE = 0.1

RADIAL_TOLERANCE = 0.1
Z_TOLERANCE = 1e-6

FILL_MEDIA = ("air", "water")


@dataclass(frozen=True)
class EnvelopeCase:
    name: str
    label: str
    y_range: tuple[float, float]
    z_range: tuple[float, float]
    epsilon: float
    description: str
    asserted: bool


CASES: tuple[EnvelopeCase, ...] = (
    EnvelopeCase(
        name="paraxial_epsilon_0p1",
        label="Paraxial envelope",
        y_range=PARAXIAL_Y_RANGE,
        z_range=PARAXIAL_Z_RANGE,
        epsilon=EPSILON_PARAXIAL,
        description=(
            "Validated paraxial side-incidence envelope; expected "
            "to pass for both fill states."
        ),
        asserted=True,
    ),
    EnvelopeCase(
        name="paraxial_epsilon_tiny",
        label="Paraxial envelope",
        y_range=PARAXIAL_Y_RANGE,
        z_range=PARAXIAL_Z_RANGE,
        epsilon=EPSILON_TINY,
        description=(
            "Paraxial aperture with the multi-step default "
            "epsilon=1e-6; chord deviation may place the next "
            "origin within the same outer-face polygon. "
            "Diagnostic only."
        ),
        asserted=False,
    ),
    EnvelopeCase(
        name="wide_epsilon_0p1",
        label="Wide envelope",
        y_range=WIDE_Y_RANGE,
        z_range=WIDE_Z_RANGE,
        epsilon=EPSILON_WIDE,
        description=(
            "Wide aperture used by earlier shell demos; grazing "
            "rays may violate the canonical outer-inner-inner-"
            "outer sequence. Diagnostic only."
        ),
        asserted=False,
    ),
)


def _build_side_incidence_rays(
    *,
    origin_x: float,
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    ny: int,
    nz: int,
    direction: tuple[float, float, float] = (-1.0, 0.0, 0.0),
) -> RayBundle:
    ys = np.linspace(y_range[0], y_range[1], ny)
    zs = np.linspace(z_range[0], z_range[1], nz)
    yv, zv = np.meshgrid(ys, zs, indexing="xy")
    origins = np.column_stack([
        np.full(yv.size, float(origin_x)),
        yv.ravel(),
        zv.ravel(),
    ])
    directions = np.broadcast_to(
        np.asarray(direction, dtype=float).reshape(1, 3),
        origins.shape,
    ).copy()
    return make_ray_bundle(origins, directions)


def _print_static_header() -> None:
    print("Shell medium tracking envelope demo")
    print(
        "Synthetic shell medium-tracking validity smoke check; "
        "not a physical PET-bottle validation."
    )
    print(
        "Note: the validated shell preset applies only to the "
        "synthetic cylindrical shell fixture."
    )
    print(
        "Note: the paraxial side-incidence aperture is a "
        "controlled smoke-test envelope."
    )
    print(
        "Note: wide or grazing ray bundles may violate the "
        "canonical outer-inner-inner-outer sequence; that "
        "violation is diagnostic, not a code failure."
    )
    print(
        "Note: this is not general STL medium tracking."
    )
    print(
        "Note: this does not prove fire prevention or PET-bottle "
        "safety."
    )


def _print_case_block(
    case: EnvelopeCase,
    rays: RayBundle,
    air_record: dict,
    water_record: dict,
) -> None:
    print(f"Case: {case.name}")
    print(f"{case.label}: y_range={case.y_range}, z_range={case.z_range}")
    print(f"epsilon: {float(case.epsilon):.6f}")
    print(f"Asserted: {bool(case.asserted)}")
    print(f"Description: {case.description}")
    print(f"Ray count: {int(rays.ray_count)}")
    for fill_label, record in (
        ("air", air_record),
        ("water", water_record),
    ):
        print(f"Fill medium: {fill_label}")
        print(
            f"Validation passed: {bool(record['validation_passed'])}"
        )
        print(
            f"Unexpected surface count: "
            f"{int(record['unexpected_surface_total'])}"
        )
        print(f"Final rays: {int(record['final_ray_count'])}")
        if not record["validation_passed"]:
            print(
                "Envelope diagnostic: medium-tracking validation "
                "did not pass; this is a diagnostic, not a code "
                "failure."
            )
        else:
            print("Envelope diagnostic: medium-tracking validation passed.")


def _check_invariants(
    rays_by_case: dict[str, RayBundle],
    case_records: dict[str, dict[str, dict]],
) -> bool:
    checks: list[bool] = []

    for case in CASES:
        rays = rays_by_case[case.name]
        air = case_records[case.name]["air"]
        water = case_records[case.name]["water"]
        for record in (air, water):
            checks.append(int(record["final_ray_count"]) >= 0)
            checks.append(int(record["unexpected_surface_total"]) >= 0)
            checks.append(
                int(record["final_ray_count"]) <= int(rays.ray_count)
            )
        if case.asserted:
            for record in (air, water):
                checks.append(bool(record["validation_passed"]))
                checks.append(
                    int(record["unexpected_surface_total"]) == 0
                )

    return all(checks)


def main() -> int:
    shell_mesh = create_subdivided_synthetic_bottle_shell(
        outer_radius=OUTER_RADIUS,
        wall_thickness=WALL_THICKNESS,
        height=HEIGHT,
        sections=SECTIONS,
        height_segments=HEIGHT_SEGMENTS,
    )

    _print_static_header()

    rays_by_case: dict[str, RayBundle] = {}
    case_records: dict[str, dict[str, dict]] = {}

    for case in CASES:
        rays = _build_side_incidence_rays(
            origin_x=ORIGIN_X,
            y_range=case.y_range,
            z_range=case.z_range,
            ny=RAY_NY,
            nz=RAY_NZ,
        )
        rays_by_case[case.name] = rays

        air_tracking = run_surface_classified_shell_trace(
            shell_mesh, rays,
            fill_medium="air",
            outer_radius=OUTER_RADIUS,
            wall_thickness=WALL_THICKNESS,
            height=HEIGHT,
            radial_tolerance=RADIAL_TOLERANCE,
            z_tolerance=Z_TOLERANCE,
            epsilon=case.epsilon,
        )
        water_tracking = run_surface_classified_shell_trace(
            shell_mesh, rays,
            fill_medium="water",
            outer_radius=OUTER_RADIUS,
            wall_thickness=WALL_THICKNESS,
            height=HEIGHT,
            radial_tolerance=RADIAL_TOLERANCE,
            z_tolerance=Z_TOLERANCE,
            epsilon=case.epsilon,
        )
        air_record = {
            "validation_passed": bool(air_tracking.validation_passed),
            "unexpected_surface_total": int(
                air_tracking.unexpected_surface_total
            ),
            "final_ray_count": int(air_tracking.final_ray_count),
        }
        water_record = {
            "validation_passed": bool(water_tracking.validation_passed),
            "unexpected_surface_total": int(
                water_tracking.unexpected_surface_total
            ),
            "final_ray_count": int(water_tracking.final_ray_count),
        }
        case_records[case.name] = {
            "air": air_record,
            "water": water_record,
        }

        _print_case_block(case, rays, air_record, water_record)

    ok = _check_invariants(rays_by_case, case_records)
    print(f"Invariants: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
