"""Surface-classified synthetic shell medium-tracking foundation.

Surface-classified synthetic shell medium-tracking smoke check;
**not a physical PET-bottle validation**. Wraps
:func:`run_multi_step_trace` over the synthetic cylindrical-shell
fixture (:func:`create_subdivided_synthetic_bottle_shell`) and
verifies that side-incidence rays cross the expected sequence of
shell surfaces under a caller-selected fill-state preset
(:func:`create_shell_medium_preset`).

Why this exists
---------------
The multi-step trace consumes a caller-provided
``interface_sequence`` verbatim and does not auto-track media or
verify that a ray actually hits the surface that the caller's
preset implies. For the synthetic shell side-incidence smoke
setup, the canonical surface order is::

    outer_lateral -> inner_lateral -> inner_lateral -> outer_lateral

This module provides the classifier
(:func:`classify_synthetic_shell_hit_surfaces`), the expected
sequence accessor (:func:`expected_shell_surface_sequence`), and
a thin diagnostic wrapper
(:func:`run_surface_classified_shell_trace`) that runs the same
:func:`run_multi_step_trace` the manual demo runs and reports
per-step surface-kind counts plus a single
``unexpected_surface_total`` count. The wrapper is **diagnostic**:
it does not raise on mismatch by default and it does not change
refraction, propagation, or medium semantics elsewhere.

Scope (also enforced in the demo)
---------------------------------
- This is scoped to the synthetic cylindrical shell fixture.
- It is **not** general STL medium tracking.
- It does **not** infer real bottle wall thickness from STL.
- It does **not** model neck, shoulder, base petaloid geometry,
  caps, labels, seams, or manufacturing defects.
- It validates medium-transition consistency for a side-incidence
  synthetic setup only.
- It does **not** change the refraction convention.
- It does **not** add reflected branches.
- It does **not** prove fire prevention or PET-bottle safety.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from optics_simulation.optics.media_presets import (
    create_shell_medium_preset,
)
from optics_simulation.optics.multi_step import (
    MultiStepTraceResult,
    run_multi_step_trace,
)
from optics_simulation.optics.ray import OpticsError, RayBundle


_SUPPORTED_FILL_MEDIA = ("air", "water")
_EXPECTED_SHELL_SURFACE_SEQUENCE = (
    "outer_lateral",
    "inner_lateral",
    "inner_lateral",
    "outer_lateral",
)


@dataclass(frozen=True)
class ShellHitSurfaceClassification:
    surface_kind: np.ndarray             # (N,) object, str values
    outer_lateral_mask: np.ndarray       # (N,) bool
    inner_lateral_mask: np.ndarray       # (N,) bool
    z_boundary_mask: np.ndarray          # (N,) bool
    miss_mask: np.ndarray                # (N,) bool
    unknown_mask: np.ndarray             # (N,) bool
    ray_count: int
    outer_radius: float
    inner_radius: float
    height: float


@dataclass(frozen=True)
class ShellMediumTrackingStepSummary:
    step_index: int
    eta_i: float
    eta_t: float
    medium_i: str
    medium_t: str
    expected_surface_kind: str
    ray_count: int
    hit_count: int
    transmitted_count: int
    surface_kind_counts: dict[str, int]
    unexpected_surface_count: int


@dataclass(frozen=True)
class ShellMediumTrackingResult:
    fill_medium: str
    preset_name: str
    interface_sequence: tuple[tuple[float, float], ...]
    medium_path: tuple[str, ...]
    expected_surface_sequence: tuple[str, ...]
    initial_ray_count: int
    final_ray_count: int
    final_source_ray_indices: np.ndarray
    final_ray_weights: np.ndarray
    step_summaries: tuple[ShellMediumTrackingStepSummary, ...]
    trace_result: MultiStepTraceResult
    validation_passed: bool
    unexpected_surface_total: int


def _check_finite_positive(value: float, *, name: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise OpticsError(
            f"{name} must be a finite float > 0; got {value!r}"
        ) from exc
    if not np.isfinite(v):
        raise OpticsError(f"{name} must be finite; got {v}")
    if v <= 0.0:
        raise OpticsError(f"{name} must be > 0; got {v}")
    return v


def _check_finite_nonneg(value: float, *, name: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise OpticsError(
            f"{name} must be a finite float >= 0; got {value!r}"
        ) from exc
    if not np.isfinite(v):
        raise OpticsError(f"{name} must be finite; got {v}")
    if v < 0.0:
        raise OpticsError(f"{name} must be >= 0; got {v}")
    return v


def expected_shell_surface_sequence(
    *, fill_medium: str,
) -> tuple[str, ...]:
    """Return the expected synthetic-shell side-incidence surface order.

    Surface-classified synthetic shell medium-tracking smoke
    check; **not a physical PET-bottle validation**. For both
    ``fill_medium="air"`` and ``fill_medium="water"`` the
    side-incidence ray crosses **four** surfaces in the same
    order::

        ("outer_lateral", "inner_lateral",
         "inner_lateral", "outer_lateral")

    The IORs differ (the empty-shell preset has air in the
    cavity; the water-filled preset has water in the cavity), but
    the geometric surface sequence is identical because both
    presets use the same synthetic cylindrical shell fixture.
    This accessor exists for surface-validation only — it is
    **not** automatic medium tracking and **not** general STL
    segmentation.
    """
    if fill_medium not in _SUPPORTED_FILL_MEDIA:
        raise OpticsError(
            f"fill_medium must be one of {_SUPPORTED_FILL_MEDIA}; "
            f"got {fill_medium!r}"
        )
    return _EXPECTED_SHELL_SURFACE_SEQUENCE


def classify_synthetic_shell_hit_surfaces(
    hit_points: np.ndarray,
    hit_mask: np.ndarray,
    *,
    outer_radius: float,
    wall_thickness: float,
    height: float,
    radial_tolerance: float = 1e-6,
    z_tolerance: float = 1e-6,
) -> ShellHitSurfaceClassification:
    """Classify per-ray hit points against the synthetic shell surfaces.

    Surface-classified synthetic shell medium-tracking smoke
    check; **not a physical PET-bottle validation**. Each row of
    ``hit_points`` is classified into one of:

    - ``"miss"``: ``hit_mask[i]`` is ``False``.
    - ``"z_boundary"``: ``abs(abs(z) - height/2) <= z_tolerance``.
      Takes priority over outer / inner lateral.
    - ``"outer_lateral"``: ``abs(r - outer_radius) <= radial_tolerance``
      and not on the z-boundary.
    - ``"inner_lateral"``: ``abs(r - inner_radius) <= radial_tolerance``
      and not on the z-boundary, where
      ``inner_radius = outer_radius - wall_thickness``.
    - ``"unknown"``: hit but matched none of the above.

    The classifier is for the synthetic cylindrical-shell fixture
    only; it does **not** segment arbitrary STL geometry. The
    inputs are read but never mutated.

    Parameters
    ----------
    hit_points
        Dense ``(N, 3)`` array of per-ray hit points (NaN allowed
        on miss rows).
    hit_mask
        Dense ``(N,)`` boolean mask of which rays hit the mesh.
    outer_radius, wall_thickness, height
        Same geometric parameters used to build the synthetic
        shell.
    radial_tolerance, z_tolerance
        Strict-positive radial tolerance and non-negative z
        tolerance used for surface membership.

    Raises
    ------
    OpticsError
        On invalid geometric parameters, malformed input shapes,
        or non-finite hit-points where ``hit_mask`` is ``True``.
    """
    outer_r = _check_finite_positive(outer_radius, name="outer_radius")
    wall_t = _check_finite_positive(
        wall_thickness, name="wall_thickness"
    )
    if wall_t >= outer_r:
        raise OpticsError(
            f"wall_thickness ({wall_t}) must be strictly less than "
            f"outer_radius ({outer_r})"
        )
    height_f = _check_finite_positive(height, name="height")
    radial_tol = _check_finite_positive(
        radial_tolerance, name="radial_tolerance"
    )
    z_tol = _check_finite_nonneg(z_tolerance, name="z_tolerance")

    pts = np.asarray(hit_points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise OpticsError(
            f"hit_points must have shape (N, 3); got {pts.shape}"
        )
    n = int(pts.shape[0])

    mask = np.asarray(hit_mask, dtype=bool)
    if mask.shape != (n,):
        raise OpticsError(
            f"hit_mask must have shape ({n},); got {mask.shape}"
        )

    if mask.any():
        hit_only = pts[mask]
        if not np.isfinite(hit_only).all():
            raise OpticsError(
                "hit_points must be finite where hit_mask is True"
            )

    safe_pts = np.where(mask[:, None], pts, 0.0)
    r = np.sqrt(safe_pts[:, 0] ** 2 + safe_pts[:, 1] ** 2)
    z = safe_pts[:, 2]
    inner_r = outer_r - wall_t
    half_h = 0.5 * height_f

    miss_mask = ~mask
    z_boundary_mask = (
        (np.abs(np.abs(z) - half_h) <= z_tol) & mask
    )
    outer_lateral_mask = (
        (np.abs(r - outer_r) <= radial_tol)
        & mask
        & ~z_boundary_mask
    )
    inner_lateral_mask = (
        (np.abs(r - inner_r) <= radial_tol)
        & mask
        & ~z_boundary_mask
    )
    unknown_mask = (
        mask
        & ~z_boundary_mask
        & ~outer_lateral_mask
        & ~inner_lateral_mask
    )

    surface_kind = np.empty(n, dtype=object)
    surface_kind[miss_mask] = "miss"
    surface_kind[z_boundary_mask] = "z_boundary"
    surface_kind[outer_lateral_mask] = "outer_lateral"
    surface_kind[inner_lateral_mask] = "inner_lateral"
    surface_kind[unknown_mask] = "unknown"

    return ShellHitSurfaceClassification(
        surface_kind=surface_kind.copy(),
        outer_lateral_mask=outer_lateral_mask.astype(bool, copy=True),
        inner_lateral_mask=inner_lateral_mask.astype(bool, copy=True),
        z_boundary_mask=z_boundary_mask.astype(bool, copy=True),
        miss_mask=miss_mask.astype(bool, copy=True),
        unknown_mask=unknown_mask.astype(bool, copy=True),
        ray_count=n,
        outer_radius=float(outer_r),
        inner_radius=float(inner_r),
        height=float(height_f),
    )


def _medium_path_for(fill_medium: str) -> tuple[str, ...]:
    if fill_medium == "air":
        return ("air", "pet", "air", "pet", "air")
    if fill_medium == "water":
        return ("air", "pet", "water", "pet", "air")
    raise OpticsError(
        f"fill_medium must be one of {_SUPPORTED_FILL_MEDIA}; "
        f"got {fill_medium!r}"
    )


def run_surface_classified_shell_trace(
    mesh: trimesh.Trimesh,
    rays: RayBundle,
    *,
    fill_medium: str,
    outer_radius: float,
    wall_thickness: float,
    height: float,
    ior_air: float = 1.00028,
    ior_pet: float = 1.575,
    ior_water: float = 1.333,
    validate_surfaces: bool = True,
    radial_tolerance: float = 1e-6,
    z_tolerance: float = 1e-6,
    epsilon: float = 1e-6,
) -> ShellMediumTrackingResult:
    """Run a synthetic shell side-incidence trace with surface validation.

    Surface-classified synthetic shell medium-tracking smoke
    check; **not a physical PET-bottle validation**. Builds the
    interface sequence via
    :func:`create_shell_medium_preset(fill_medium=fill_medium)`,
    runs :func:`run_multi_step_trace` on ``mesh`` and ``rays``
    verbatim (no refraction or propagation changes), and then
    classifies every per-step ``intersection.hit_points`` against
    the synthetic shell surfaces. For each step the wrapper
    counts how many **transmitted** rays landed on the expected
    surface kind for that step (per
    :func:`expected_shell_surface_sequence`); the per-step counts
    are aggregated into ``unexpected_surface_total`` and the
    overall ``validation_passed`` flag.

    The wrapper is intentionally **diagnostic**: when
    ``validate_surfaces=True`` and one or more transmitted rays
    miss the expected surface kind, the result is returned with
    ``validation_passed=False`` rather than raising. This keeps
    the wrapper safe to use inside screening loops over many
    candidates and ray fixtures. ``validate_surfaces=False``
    forces ``validation_passed=True`` regardless of the count.

    The wrapper does **not** modify
    :mod:`optics_simulation.optics.refraction`,
    :mod:`optics_simulation.optics.propagation`, or
    :mod:`optics_simulation.optics.multi_step`, and does **not**
    add reflected ray branches.

    Parameters
    ----------
    mesh, rays
        Forwarded verbatim to :func:`run_multi_step_trace`.
    fill_medium
        ``"air"`` (empty shell) or ``"water"`` (water-filled
        shell). Any other value raises :class:`OpticsError` from
        :func:`create_shell_medium_preset`.
    outer_radius, wall_thickness, height
        Synthetic shell geometry parameters used by the
        classifier. Must match the mesh that was used to build
        ``mesh`` for the surface validation to be meaningful.
    ior_air, ior_pet, ior_water
        Forwarded to :func:`create_shell_medium_preset`.
    validate_surfaces, radial_tolerance, z_tolerance
        Surface-validation knobs. ``z_tolerance`` may be ``0``;
        the others must be strictly positive.
    epsilon
        Forwarded to :func:`run_multi_step_trace` as the
        next-ray origin offset along the refracted direction. The
        default ``1e-6`` matches the multi-step default, but on a
        polygonally faceted shell the ray's new origin can lie
        within the **chord deviation** of the same outer face
        (``approx outer_radius * (pi / sections) ** 2 / 2``); when
        that happens, neighbouring outer polygons can be hit
        instead of the expected inner cylinder. Callers running
        on the synthetic shell fixture should pass an
        ``epsilon`` larger than the chord deviation (the demo
        uses ``0.1``).

    Raises
    ------
    OpticsError
        On invalid mesh / rays / preset arguments, invalid
        geometric parameters, malformed intermediate intersection
        results, or any downstream :class:`OpticsError` from
        :func:`run_multi_step_trace`.
    """
    if not isinstance(mesh, trimesh.Trimesh):
        raise OpticsError(
            f"mesh must be a trimesh.Trimesh; got {type(mesh).__name__}"
        )
    if not isinstance(rays, RayBundle):
        raise OpticsError(
            f"rays must be a RayBundle; got {type(rays).__name__}"
        )
    preset = create_shell_medium_preset(
        fill_medium=fill_medium,
        ior_air=ior_air,
        ior_pet=ior_pet,
        ior_water=ior_water,
    )
    medium_path = _medium_path_for(fill_medium)
    expected_surfaces = expected_shell_surface_sequence(
        fill_medium=fill_medium
    )

    trace_result = run_multi_step_trace(
        mesh, rays, preset.interface_sequence,
        epsilon=epsilon,
    )

    step_summaries: list[ShellMediumTrackingStepSummary] = []
    unexpected_total = 0

    for step_index, step in enumerate(trace_result.steps):
        classification = classify_synthetic_shell_hit_surfaces(
            step.intersection.hit_points,
            step.intersection.hit_mask,
            outer_radius=outer_radius,
            wall_thickness=wall_thickness,
            height=height,
            radial_tolerance=radial_tolerance,
            z_tolerance=z_tolerance,
        )
        surface_kind_counts: dict[str, int] = {
            "outer_lateral": int(classification.outer_lateral_mask.sum()),
            "inner_lateral": int(classification.inner_lateral_mask.sum()),
            "z_boundary": int(classification.z_boundary_mask.sum()),
            "miss": int(classification.miss_mask.sum()),
            "unknown": int(classification.unknown_mask.sum()),
        }

        if step_index < len(expected_surfaces):
            expected_kind = str(expected_surfaces[step_index])
        else:
            expected_kind = "unknown"

        transmitted_indices = np.asarray(
            step.propagation.source_ray_indices, dtype=np.int64,
        )
        if (
            step_index < len(expected_surfaces)
            and transmitted_indices.size > 0
        ):
            kinds = classification.surface_kind[transmitted_indices]
            unexpected = int(np.sum(kinds != expected_kind))
        else:
            unexpected = 0

        if step_index < len(preset.interface_sequence):
            eta_i, eta_t = preset.interface_sequence[step_index]
        else:
            eta_i, eta_t = float("nan"), float("nan")

        if step_index + 1 < len(medium_path):
            medium_i = medium_path[step_index]
            medium_t = medium_path[step_index + 1]
        else:
            medium_i = "unknown"
            medium_t = "unknown"

        step_summaries.append(
            ShellMediumTrackingStepSummary(
                step_index=int(step_index),
                eta_i=float(eta_i),
                eta_t=float(eta_t),
                medium_i=str(medium_i),
                medium_t=str(medium_t),
                expected_surface_kind=str(expected_kind),
                ray_count=int(step.ray_count),
                hit_count=int(step.hit_count),
                transmitted_count=int(step.transmitted_count),
                surface_kind_counts=dict(surface_kind_counts),
                unexpected_surface_count=int(unexpected),
            )
        )
        unexpected_total += int(unexpected)

    if validate_surfaces:
        validation_passed = unexpected_total == 0
    else:
        validation_passed = True

    return ShellMediumTrackingResult(
        fill_medium=str(fill_medium),
        preset_name=str(preset.name),
        interface_sequence=tuple(preset.interface_sequence),
        medium_path=tuple(medium_path),
        expected_surface_sequence=tuple(expected_surfaces),
        initial_ray_count=int(rays.ray_count),
        final_ray_count=int(trace_result.final_rays.ray_count),
        final_source_ray_indices=(
            trace_result.final_source_ray_indices.copy()
        ),
        final_ray_weights=trace_result.final_ray_weights.copy(),
        step_summaries=tuple(step_summaries),
        trace_result=trace_result,
        validation_passed=bool(validation_passed),
        unexpected_surface_total=int(unexpected_total),
    )
