"""Detector accumulation demo.

Synthetic mesh smoke check of the optical pipeline foundations built
so far: synthetic box mesh -> parallel ray grid -> multi-step trace
-> detector plane intersection -> detector grid accumulation ->
optical metrics on a relative irradiance surrogate -> console
summary + invariant check.

This is **not** a reproduction of PET-bottle caustics. The detector
weight map is a unit-less relative irradiance surrogate computed from
counts of synthetic rays through a synthetic slab.

Run from the repository root:

    python examples/run_detector_demo.py

The script exits 0 when every invariant holds and 1 otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import trimesh

from optics_simulation.metrics import compute_optical_metrics
from optics_simulation.optics import (
    accumulate_detector_hits,
    create_detector_grid,
    create_detector_plane,
    intersect_detector_plane,
    parallel_ray_grid,
    run_multi_step_trace,
)


IOR_AIR = 1.00028
IOR_PET = 1.575
HOTSPOT_THRESHOLDS = (2.0, 5.0, 10.0)
TOP_PERCENT_FOR_C99 = 1.0
INCIDENT_REFERENCE = 1.0  # unit-less relative irradiance surrogate


def _print_summary(rays, trace, accum, metrics) -> None:
    print("Detector accumulation demo")
    print(f"Initial rays: {rays.ray_count}")
    print(f"Termination: {trace.termination_reason}")
    for i, step in enumerate(trace.steps):
        print(
            f"Step {i}: ray_count={step.ray_count} "
            f"hit={step.hit_count} miss={step.miss_count} "
            f"transmitted={step.transmitted_count} tir={step.tir_count}"
        )
    print(f"Final rays: {trace.final_rays.ray_count}")
    print(f"Detector hits: {accum.total_hits}")
    print(f"Count map sum: {int(accum.count_map.sum())}")
    print(f"Weight map sum: {float(accum.weight_map.sum()):.1f}")
    print(f"Peak value: {metrics.peak_value:.1f}")
    print(f"Cmax: {metrics.cmax:.4f}")
    print(f"C99: {metrics.c99:.4f}")
    print(f"Eexceed@2.0: {metrics.eexceed[2.0]:.1f}")
    print(f"Ahot@2.0: {metrics.ahot[2.0]}")


def _check_invariants(rays, trace, hits, accum, metrics) -> bool:
    checks: list[bool] = []

    checks.append(rays.ray_count == trace.initial_rays.ray_count)

    for step in trace.steps:
        checks.append(step.hit_count + step.miss_count == step.ray_count)
        checks.append(
            step.transmitted_count + step.tir_count + step.miss_count
            == step.ray_count
        )
        checks.append(
            step.transmitted_count == step.propagation.next_rays.ray_count
        )

    for i in range(len(trace.steps) - 1):
        checks.append(
            trace.steps[i + 1].rays is trace.steps[i].propagation.next_rays
        )

    if trace.steps:
        checks.append(
            trace.final_rays is trace.steps[-1].propagation.next_rays
        )
    else:
        checks.append(trace.final_rays is trace.initial_rays)

    checks.append(accum.total_hits == int(hits.hit_mask.sum()))
    checks.append(int(accum.count_map.sum()) == accum.total_hits)
    checks.append(
        bool(np.isclose(accum.weight_map.sum(), float(accum.total_weight)))
    )

    checks.append(metrics.pixel_count == accum.weight_map.size)
    checks.append(
        bool(np.isclose(metrics.total_value, float(accum.weight_map.sum())))
    )
    checks.append(metrics.peak_value >= 0.0)
    checks.append(metrics.c99 >= 0.0)

    return all(checks)


def main() -> int:
    box = trimesh.creation.box(extents=(10.0, 10.0, 10.0))
    rays = parallel_ray_grid(
        origin_plane_z=20.0,
        direction=(0.0, 0.0, -1.0),
        x_range=(-7.5, 7.5),
        y_range=(-7.5, 7.5),
        nx=11,
        ny=11,
    )

    interface_sequence = [(IOR_AIR, IOR_PET), (IOR_PET, IOR_AIR)]
    trace = run_multi_step_trace(box, rays, interface_sequence)

    detector = create_detector_plane(
        center=(0.0, 0.0, -15.0),
        normal=(0.0, 0.0, 1.0),
        up=(0.0, 1.0, 0.0),
        width=10.0,
        height=10.0,
    )
    hits = intersect_detector_plane(trace.final_rays, detector)

    grid = create_detector_grid(width=10.0, height=10.0, resolution=(20, 20))
    accum = accumulate_detector_hits(hits, grid)

    metrics = compute_optical_metrics(
        accum.weight_map,
        incident_reference=INCIDENT_REFERENCE,
        thresholds=HOTSPOT_THRESHOLDS,
        top_percent=TOP_PERCENT_FOR_C99,
    )

    _print_summary(rays, trace, accum, metrics)
    ok = _check_invariants(rays, trace, hits, accum, metrics)
    print(f"Invariants: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
