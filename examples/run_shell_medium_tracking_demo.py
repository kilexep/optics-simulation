"""Surface-classified synthetic shell medium-tracking demo.

Surface-classified synthetic shell medium-tracking smoke check;
not a physical PET-bottle validation. Demonstrates that the
surface-classified wrapper
:func:`run_surface_classified_shell_trace` consumes the same
``interface_sequence`` preset that the manual shell/fill-state
demos consume, and that the per-step transmitted hits land on
the expected synthetic shell surfaces in the expected order.

This demo validates that the wrapper does **not** change
refraction, propagation, or medium semantics. The wrapper's
``trace_result`` is byte-identical to the manual
:func:`run_multi_step_trace` call when both calls are passed the
same ``interface_sequence`` and the same ``epsilon``. The
wrapper additionally classifies every per-step
``intersection.hit_points`` and reports per-step surface-kind
counts and a single ``unexpected_surface_total``.

Why epsilon matters
-------------------
The synthetic cylindrical shell mesh is polygonally faceted (the
default fixture uses ``sections=96``). Each polygon's chord
deviates from the ideal cylinder by approximately
``outer_radius * (pi / sections) ** 2 / 2`` (about ``0.016 mm``
for the default geometry). The default
:func:`run_multi_step_trace` ``epsilon = 1e-6`` is much smaller
than that chord deviation, which can put the next-bounce ray's
new origin within the chord-deviation band of the same outer
face and let the trace pick a neighbouring outer polygon as the
next intersection. This demo passes ``epsilon = 0.1`` (well
above any reasonable chord deviation) to both the manual and the
tracked traces so that the trace cleanly walks
``outer -> inner -> inner -> outer`` and the surface-validation
flag is meaningful.

Scope (also enforced at runtime)
--------------------------------
- This demo is scoped to the synthetic cylindrical shell fixture.
- It is **not** general STL medium tracking.
- It does **not** infer real bottle wall thickness from STL.
- It does **not** model neck, shoulder, base petaloid geometry,
  caps, labels, seams, or manufacturing defects.
- It validates medium-transition consistency for a side-incidence
  synthetic setup only.
- It does **not** change the refraction convention.
- It does **not** add reflected branches.
- It does **not** prove fire prevention or PET-bottle safety.

Run from the repository root::

    python examples/run_shell_medium_tracking_demo.py

The script exits 0 when every invariant holds and 1 otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np

from optics_simulation.geometry import (
    create_subdivided_synthetic_bottle_shell,
)
from optics_simulation.optics import (
    RayBundle,
    create_shell_medium_preset,
    make_ray_bundle,
    run_multi_step_trace,
    run_surface_classified_shell_trace,
)


OUTER_RADIUS = 30.0
WALL_THICKNESS = 1.0
HEIGHT = 120.0
SECTIONS = 96
HEIGHT_SEGMENTS = 24

ORIGIN_X = 100.0
Y_RANGE = (-25.0, 25.0)
Z_RANGE = (-50.0, 50.0)
RAY_NY = 7
RAY_NZ = 7

TRACE_EPSILON = 0.1
RADIAL_TOLERANCE = 0.1
Z_TOLERANCE = 1e-6

FILL_MEDIA = ("air", "water")


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


def _print_static_header(rays: RayBundle, shell_watertight: bool) -> None:
    print("Shell medium tracking demo")
    print(
        "Surface-classified synthetic shell medium-tracking smoke "
        "check; not a physical PET-bottle validation."
    )
    print(
        "Note: this is scoped to the synthetic cylindrical shell "
        "fixture. It is not general STL medium tracking."
    )
    print(
        "Note: it does not model neck, shoulder, base, caps, "
        "labels, seams, or manufacturing defects."
    )
    print(
        "Note: it does not change refraction or propagation, and "
        "it does not add reflected branches."
    )
    print(
        "Note: it does not prove fire prevention or PET-bottle "
        "safety."
    )
    print(
        f"Side-incidence ray grid: origin_x={ORIGIN_X}, "
        f"y_range={Y_RANGE}, z_range={Z_RANGE}, "
        f"ny={RAY_NY}, nz={RAY_NZ}"
    )
    print(f"Ray count: {int(rays.ray_count)}")
    print(f"Shell watertight: {bool(shell_watertight)}")
    print(f"Trace epsilon: {float(TRACE_EPSILON):.4f}")
    print(f"Radial tolerance: {float(RADIAL_TOLERANCE):.4f}")
    print(f"Z tolerance: {float(Z_TOLERANCE):.6f}")


def _print_fill_block(
    fill_medium: str,
    manual_final: int,
    tracked_final: int,
    weights_allclose: bool,
    indices_equal: bool,
    medium_path: tuple[str, ...],
    expected_surfaces: tuple[str, ...],
    validation_passed: bool,
    unexpected_total: int,
    step_lines: list[str],
) -> None:
    print(f"Fill medium: {fill_medium}")
    print(f"Medium path: {' -> '.join(medium_path)}")
    print(
        f"Expected surface sequence: {', '.join(expected_surfaces)}"
    )
    print(f"Manual final rays: {int(manual_final)}")
    print(f"Tracked final rays: {int(tracked_final)}")
    print(f"Final ray weights allclose: {bool(weights_allclose)}")
    print(f"Final source indices equal: {bool(indices_equal)}")
    print(f"Validation passed: {bool(validation_passed)}")
    print(f"Unexpected surface count: {int(unexpected_total)}")
    for line in step_lines:
        print(line)


def _format_step_line(
    step_index: int,
    eta_i: float,
    eta_t: float,
    medium_i: str,
    medium_t: str,
    expected_kind: str,
    ray_count: int,
    hit_count: int,
    transmitted_count: int,
    surface_kind_counts: dict[str, int],
    unexpected: int,
) -> str:
    counts_str = ", ".join(
        f"{k}={v}" for k, v in surface_kind_counts.items()
    )
    return (
        f"Step {step_index}: eta=({eta_i:.5f}->{eta_t:.5f}), "
        f"medium={medium_i}->{medium_t}, expected={expected_kind}, "
        f"rays={ray_count}, hit={hit_count}, "
        f"transmitted={transmitted_count}, "
        f"counts={{{counts_str}}}, unexpected={unexpected}"
    )


def _check_invariants(
    expected_ray_count: int,
    fill_records: list[dict],
    vertices_mutated: bool,
    faces_mutated: bool,
) -> bool:
    checks: list[bool] = []

    for record in fill_records:
        checks.append(int(record["manual_final"]) == int(record["tracked_final"]))
        checks.append(bool(record["weights_allclose"]))
        checks.append(bool(record["indices_equal"]))
        checks.append(bool(record["validation_passed"]))
        checks.append(int(record["unexpected_total"]) == 0)
        checks.append(int(record["step_count"]) == 4)
        checks.append(int(record["initial_ray_count"]) == int(expected_ray_count))

    checks.append(not bool(vertices_mutated))
    checks.append(not bool(faces_mutated))

    return all(checks)


def main() -> int:
    shell_mesh = create_subdivided_synthetic_bottle_shell(
        outer_radius=OUTER_RADIUS,
        wall_thickness=WALL_THICKNESS,
        height=HEIGHT,
        sections=SECTIONS,
        height_segments=HEIGHT_SEGMENTS,
    )
    rays = _build_side_incidence_rays(
        origin_x=ORIGIN_X,
        y_range=Y_RANGE,
        z_range=Z_RANGE,
        ny=RAY_NY,
        nz=RAY_NZ,
    )

    vertices_before = np.array(shell_mesh.vertices, copy=True)
    faces_before = np.array(shell_mesh.faces, copy=True)

    _print_static_header(
        rays, shell_watertight=bool(shell_mesh.is_watertight),
    )

    fill_records: list[dict] = []
    for fill_medium in FILL_MEDIA:
        manual_preset = create_shell_medium_preset(fill_medium=fill_medium)
        manual_trace = run_multi_step_trace(
            shell_mesh, rays, manual_preset.interface_sequence,
            epsilon=TRACE_EPSILON,
        )
        tracked = run_surface_classified_shell_trace(
            shell_mesh, rays,
            fill_medium=fill_medium,
            outer_radius=OUTER_RADIUS,
            wall_thickness=WALL_THICKNESS,
            height=HEIGHT,
            radial_tolerance=RADIAL_TOLERANCE,
            z_tolerance=Z_TOLERANCE,
            epsilon=TRACE_EPSILON,
        )
        weights_allclose = bool(np.allclose(
            manual_trace.final_ray_weights,
            tracked.final_ray_weights,
        ))
        indices_equal = bool(np.array_equal(
            manual_trace.final_source_ray_indices,
            tracked.final_source_ray_indices,
        ))
        step_lines: list[str] = []
        for s in tracked.step_summaries:
            step_lines.append(
                _format_step_line(
                    step_index=s.step_index,
                    eta_i=s.eta_i,
                    eta_t=s.eta_t,
                    medium_i=s.medium_i,
                    medium_t=s.medium_t,
                    expected_kind=s.expected_surface_kind,
                    ray_count=s.ray_count,
                    hit_count=s.hit_count,
                    transmitted_count=s.transmitted_count,
                    surface_kind_counts=s.surface_kind_counts,
                    unexpected=s.unexpected_surface_count,
                )
            )

        _print_fill_block(
            fill_medium,
            manual_final=int(manual_trace.final_rays.ray_count),
            tracked_final=int(tracked.final_ray_count),
            weights_allclose=weights_allclose,
            indices_equal=indices_equal,
            medium_path=tracked.medium_path,
            expected_surfaces=tracked.expected_surface_sequence,
            validation_passed=tracked.validation_passed,
            unexpected_total=tracked.unexpected_surface_total,
            step_lines=step_lines,
        )

        fill_records.append({
            "fill_medium": fill_medium,
            "manual_final": int(manual_trace.final_rays.ray_count),
            "tracked_final": int(tracked.final_ray_count),
            "weights_allclose": weights_allclose,
            "indices_equal": indices_equal,
            "validation_passed": bool(tracked.validation_passed),
            "unexpected_total": int(tracked.unexpected_surface_total),
            "step_count": int(tracked.trace_result.step_count),
            "initial_ray_count": int(tracked.initial_ray_count),
        })

    vertices_mutated = not np.array_equal(
        np.asarray(shell_mesh.vertices), vertices_before
    )
    faces_mutated = not np.array_equal(
        np.asarray(shell_mesh.faces), faces_before
    )
    print(f"Mesh vertices mutated: {bool(vertices_mutated)}")
    print(f"Mesh faces mutated: {bool(faces_mutated)}")

    ok = _check_invariants(
        expected_ray_count=int(RAY_NY * RAY_NZ),
        fill_records=fill_records,
        vertices_mutated=vertices_mutated,
        faces_mutated=faces_mutated,
    )
    print(f"Invariants: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
