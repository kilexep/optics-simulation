"""Multi-step ray-trace loop scaffold.

Thin orchestrator that calls :func:`run_single_interface_pipeline`
once per entry of a caller-supplied ``interface_sequence``. No new
physics is introduced: medium-state tracking, reflected-ray
generation, Fresnel power weighting, detector accumulation,
contribution maps, and angle scans are all out of scope.

Convention
----------
- ``eta_i`` is the refractive index of the medium the ray is
  currently in.
- ``eta_t`` is the refractive index of the medium the ray is about
  to enter.
- Each ``(eta_i, eta_t)`` pair in ``interface_sequence`` is passed
  through to ``run_single_interface_pipeline`` (and onward to
  ``refract_direction``) **as supplied**. This module does not
  reinterpret, swap, or auto-track media. Callers that need to
  represent multiple media transitions are responsible for ordering
  the sequence themselves.

Termination
-----------
- ``"no_interfaces"``: ``interface_sequence`` is empty. This takes
  priority over ``max_steps`` semantics — an empty sequence always
  yields ``no_interfaces`` regardless of ``max_steps``.
- ``"max_steps_reached"``: ``max_steps`` capped the loop before all
  interfaces ran (including ``max_steps == 0`` with a non-empty
  sequence).
- ``"no_active_rays"``: a step produced ``next_rays.ray_count == 0``
  (all rays missed or were total-internally-reflected).
- ``"completed_interfaces"``: every entry of ``interface_sequence``
  ran successfully and rays remained active at the end.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import trimesh

from optics_simulation.optics.pipeline import (
    SingleInterfacePipelineResult,
    run_single_interface_pipeline,
)
from optics_simulation.optics.ray import OpticsError, RayBundle


@dataclass(frozen=True)
class MultiStepTraceResult:
    initial_rays: RayBundle
    final_rays: RayBundle
    steps: tuple[SingleInterfacePipelineResult, ...]
    step_count: int
    termination_reason: str


def _validate_interface_sequence(
    interface_sequence: Sequence[tuple[float, float]],
) -> None:
    for index, pair in enumerate(interface_sequence):
        eta_i, eta_t = pair
        if float(eta_i) <= 0.0 or float(eta_t) <= 0.0:
            raise OpticsError(
                f"interface_sequence[{index}] has non-positive refractive index: "
                f"eta_i={eta_i}, eta_t={eta_t}"
            )


def run_multi_step_trace(
    mesh: trimesh.Trimesh,
    rays: RayBundle,
    interface_sequence: Sequence[tuple[float, float]],
    *,
    epsilon: float = 1e-6,
    max_steps: int | None = None,
) -> MultiStepTraceResult:
    """Run ``run_single_interface_pipeline`` once per interface pair.

    See module docstring for the eta convention and termination
    policy. Each step's ``(eta_i, eta_t)`` is taken verbatim from
    ``interface_sequence``; no medium tracking is performed.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise OpticsError(
            f"mesh must be a trimesh.Trimesh; got {type(mesh).__name__}"
        )
    if not isinstance(rays, RayBundle):
        raise OpticsError(
            f"rays must be a RayBundle; got {type(rays).__name__}"
        )
    if epsilon < 0.0:
        raise OpticsError(f"epsilon must be >= 0; got {epsilon}")

    seq_len = len(interface_sequence)
    if max_steps is not None:
        if max_steps < 0:
            raise OpticsError(f"max_steps must be >= 0 or None; got {max_steps}")
        if max_steps > seq_len:
            raise OpticsError(
                f"max_steps ({max_steps}) exceeds interface_sequence length "
                f"({seq_len}); cannot infer additional interfaces without "
                f"medium tracking"
            )

    _validate_interface_sequence(interface_sequence)

    if seq_len == 0:
        return MultiStepTraceResult(
            initial_rays=rays,
            final_rays=rays,
            steps=(),
            step_count=0,
            termination_reason="no_interfaces",
        )

    effective_steps = seq_len if max_steps is None else max_steps

    if effective_steps == 0:
        return MultiStepTraceResult(
            initial_rays=rays,
            final_rays=rays,
            steps=(),
            step_count=0,
            termination_reason="max_steps_reached",
        )

    current_rays = rays
    steps_list: list[SingleInterfacePipelineResult] = []
    terminated_early = False

    for i in range(effective_steps):
        eta_i, eta_t = interface_sequence[i]
        step = run_single_interface_pipeline(
            mesh,
            current_rays,
            float(eta_i),
            float(eta_t),
            epsilon=epsilon,
        )
        steps_list.append(step)
        current_rays = step.propagation.next_rays
        if current_rays.ray_count == 0:
            terminated_early = True
            break

    if terminated_early:
        termination_reason = "no_active_rays"
    elif effective_steps == seq_len:
        termination_reason = "completed_interfaces"
    else:
        termination_reason = "max_steps_reached"

    return MultiStepTraceResult(
        initial_rays=rays,
        final_rays=current_rays,
        steps=tuple(steps_list),
        step_count=len(steps_list),
        termination_reason=termination_reason,
    )
