"""Multi-mesh sequential trace foundation.

Legacy-style STL optical smoke check; **not a physical PET-bottle
validation**. Sibling of :func:`run_multi_step_trace` that allows a
**different** ``trimesh.Trimesh`` per step. Each step still calls
:func:`run_single_interface_pipeline` exactly once with caller-
supplied ``(eta_i, eta_t)``; medium-state tracking, reflected-ray
generation, splitting, axis alignment, and physical / spectral
calibration remain out of scope. This module is the foundation
for the legacy outer-shell / inner-water surface-pair trace and
is **not** a general medium tracker.

Convention
----------
- Each ``MultiMeshTraceStepSpec`` carries one mesh plus the
  caller-authoritative ``(eta_i, eta_t)`` for that interface.
  The (mesh, eta_i, eta_t) tuple is passed verbatim to
  ``run_single_interface_pipeline``; this module does not
  reinterpret, swap, or auto-track media.
- Lineage and Fresnel-transmission weight chaining mirror
  :func:`run_multi_step_trace` exactly so a same-mesh sequence
  reproduces the canonical multi-step result.

Termination
-----------
- ``"no_interfaces"``: ``step_specs`` is empty. Takes priority
  over ``max_steps`` semantics.
- ``"max_steps_reached"``: ``max_steps`` capped the loop before
  every spec ran (including ``max_steps == 0`` with a non-empty
  sequence).
- ``"no_active_rays"``: a step produced
  ``next_rays.ray_count == 0``.
- ``"completed_interfaces"``: every spec ran and rays remained
  active at the end.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import trimesh

from optics_simulation.optics.pipeline import (
    SingleInterfacePipelineResult,
    run_single_interface_pipeline,
)
from optics_simulation.optics.ray import OpticsError, RayBundle


@dataclass(frozen=True)
class MultiMeshTraceStepSpec:
    mesh: trimesh.Trimesh
    eta_i: float
    eta_t: float
    label: str = ""


@dataclass(frozen=True)
class MultiMeshTraceResult:
    initial_rays: RayBundle
    final_rays: RayBundle
    steps: tuple[SingleInterfacePipelineResult, ...]
    step_specs: tuple[MultiMeshTraceStepSpec, ...]
    step_count: int
    termination_reason: str
    final_source_ray_indices: np.ndarray
    final_ray_weights: np.ndarray


def _validate_step_specs(
    step_specs: Sequence[MultiMeshTraceStepSpec],
) -> tuple[MultiMeshTraceStepSpec, ...]:
    try:
        specs_tuple = tuple(step_specs)
    except TypeError as exc:
        raise OpticsError(
            "step_specs must be a sequence of MultiMeshTraceStepSpec; "
            f"got {type(step_specs).__name__}"
        ) from exc

    for index, spec in enumerate(specs_tuple):
        if not isinstance(spec, MultiMeshTraceStepSpec):
            raise OpticsError(
                f"step_specs[{index}] must be a "
                f"MultiMeshTraceStepSpec; got {type(spec).__name__}"
            )
        if not isinstance(spec.mesh, trimesh.Trimesh):
            raise OpticsError(
                f"step_specs[{index}].mesh must be a trimesh.Trimesh; "
                f"got {type(spec.mesh).__name__}"
            )
        eta_i = float(spec.eta_i)
        eta_t = float(spec.eta_t)
        if not (np.isfinite(eta_i) and np.isfinite(eta_t)):
            raise OpticsError(
                f"step_specs[{index}] has non-finite refractive "
                f"index: eta_i={eta_i}, eta_t={eta_t}"
            )
        if eta_i <= 0.0 or eta_t <= 0.0:
            raise OpticsError(
                f"step_specs[{index}] has non-positive refractive "
                f"index: eta_i={eta_i}, eta_t={eta_t}"
            )
    return specs_tuple


def run_multi_mesh_trace(
    rays: RayBundle,
    step_specs: Sequence[MultiMeshTraceStepSpec],
    *,
    epsilon: float = 1e-6,
    max_steps: int | None = None,
) -> MultiMeshTraceResult:
    """Run :func:`run_single_interface_pipeline` once per step spec.

    Legacy-style STL optical smoke check; **not a physical
    PET-bottle validation**. Sibling of
    :func:`run_multi_step_trace` that uses ``step.mesh`` per step
    instead of a single shared mesh. See module docstring for the
    convention and termination policy. Lineage and weight
    chaining match :func:`run_multi_step_trace` so a same-mesh
    sequence is observationally equivalent.
    """
    if not isinstance(rays, RayBundle):
        raise OpticsError(
            f"rays must be a RayBundle; got {type(rays).__name__}"
        )
    if epsilon < 0.0:
        raise OpticsError(f"epsilon must be >= 0; got {epsilon}")

    specs_tuple = _validate_step_specs(step_specs)
    seq_len = len(specs_tuple)

    if max_steps is not None:
        if max_steps < 0:
            raise OpticsError(
                f"max_steps must be >= 0 or None; got {max_steps}"
            )
        if max_steps > seq_len:
            raise OpticsError(
                f"max_steps ({max_steps}) exceeds step_specs length "
                f"({seq_len})"
            )

    current_source_indices = np.arange(rays.ray_count, dtype=np.int64)
    current_weights = np.ones(rays.ray_count, dtype=float)

    if seq_len == 0:
        return MultiMeshTraceResult(
            initial_rays=rays,
            final_rays=rays,
            steps=(),
            step_specs=specs_tuple,
            step_count=0,
            termination_reason="no_interfaces",
            final_source_ray_indices=current_source_indices.copy(),
            final_ray_weights=current_weights.copy(),
        )

    effective_steps = seq_len if max_steps is None else max_steps

    if effective_steps == 0:
        return MultiMeshTraceResult(
            initial_rays=rays,
            final_rays=rays,
            steps=(),
            step_specs=specs_tuple,
            step_count=0,
            termination_reason="max_steps_reached",
            final_source_ray_indices=current_source_indices.copy(),
            final_ray_weights=current_weights.copy(),
        )

    current_rays = rays
    steps_list: list[SingleInterfacePipelineResult] = []
    terminated_early = False

    for i in range(effective_steps):
        spec = specs_tuple[i]
        step = run_single_interface_pipeline(
            spec.mesh,
            current_rays,
            float(spec.eta_i),
            float(spec.eta_t),
            epsilon=epsilon,
        )
        steps_list.append(step)
        source = step.propagation.source_ray_indices
        current_source_indices = current_source_indices[source]
        current_weights = (
            current_weights[source]
            * step.propagation.transmittance[source]
        )
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

    return MultiMeshTraceResult(
        initial_rays=rays,
        final_rays=current_rays,
        steps=tuple(steps_list),
        step_specs=specs_tuple,
        step_count=len(steps_list),
        termination_reason=termination_reason,
        final_source_ray_indices=current_source_indices.copy(),
        final_ray_weights=current_weights.astype(float, copy=True),
    )
