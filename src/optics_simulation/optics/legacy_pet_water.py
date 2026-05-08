"""Legacy PET-water four-interface trace setup.

Legacy-style STL optical smoke check; **not a physical PET-bottle
validation**. Reproduces the geometric / scaling / refraction
ordering of the old interactive PET-bottle experiment as a single
helper that builds the four-step
``air -> PET -> water -> PET -> air`` :class:`MultiMeshTraceStepSpec`
sequence:

1. ``shell_mesh``: ``air -> PET``
2. ``water_mesh``: ``PET -> water``
3. ``water_mesh``: ``water -> PET``
4. ``shell_mesh``: ``PET -> air``

The helper performs **only** mesh setup. It does **not** run any
optical, thermal, or pattern simulation, does **not** repair or
align meshes, does **not** infer material regions, and does
**not** prove fire prevention or PET-bottle safety. Multi-mesh
tracing is a foundation, not a full medium tracker.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from optics_simulation.geometry.bottle_scaling import (
    BottleTargetScaleReport,
    InnerOffsetMeshReport,
    create_inner_offset_mesh_from_vertex_normals,
    create_target_scaled_mesh_copy,
)
from optics_simulation.optics.multi_mesh_trace import (
    MultiMeshTraceStepSpec,
)
from optics_simulation.optics.ray import OpticsError


_DEFAULT_TARGET_HEIGHT = 225.6
_DEFAULT_TARGET_DIAMETER = 72.1
_DEFAULT_WALL_THICKNESS = 0.3
_DEFAULT_IOR_AIR = 1.0
_DEFAULT_IOR_PET = 1.57
_DEFAULT_IOR_WATER = 1.333


@dataclass(frozen=True)
class LegacyPetWaterTraceSetup:
    shell_mesh: trimesh.Trimesh
    water_mesh: trimesh.Trimesh
    step_specs: tuple[MultiMeshTraceStepSpec, ...]
    scale_report: BottleTargetScaleReport
    inner_offset_report: InnerOffsetMeshReport
    target_height: float
    target_diameter: float
    wall_thickness: float
    ior_air: float
    ior_pet: float
    ior_water: float


def _check_positive_finite(value: float, name: str) -> float:
    v = float(value)
    if not np.isfinite(v) or v <= 0.0:
        raise OpticsError(
            f"{name} must be a finite float > 0; got {v}"
        )
    return v


def create_legacy_pet_water_trace_setup(
    mesh: trimesh.Trimesh,
    *,
    target_height: float = _DEFAULT_TARGET_HEIGHT,
    target_diameter: float = _DEFAULT_TARGET_DIAMETER,
    wall_thickness: float = _DEFAULT_WALL_THICKNESS,
    ior_air: float = _DEFAULT_IOR_AIR,
    ior_pet: float = _DEFAULT_IOR_PET,
    ior_water: float = _DEFAULT_IOR_WATER,
    inner_offset_mode: str = "auto",
) -> LegacyPetWaterTraceSetup:
    """Build the legacy-style four-step PET-water trace setup.

    Legacy STL inner-offset orientation diagnostic; **not a
    physical PET-bottle validation**. Steps:

    1. Apply legacy anisotropic target-dimension scaling via
       :func:`create_target_scaled_mesh_copy` so the bounding box
       matches ``(target_diameter, target_diameter, target_height)``.
    2. Build the inner offset water boundary via
       :func:`create_inner_offset_mesh_from_vertex_normals` with
       ``invert=True`` and the caller-supplied
       ``inner_offset_mode``. The default ``"auto"`` picks the
       offset sign whose median radial distance is smaller, so
       the resulting inner mesh sits inside the source shell
       regardless of whether vertex normals point outward or
       inward.
    3. Emit four :class:`MultiMeshTraceStepSpec` entries with
       caller-supplied refractive indices and the labels
       ``"air_to_pet_outer_shell"``,
       ``"pet_to_water_inner_surface"``,
       ``"water_to_pet_inner_surface"``,
       ``"pet_to_air_outer_shell"``.

    The helper performs **no** ray tracing. The input mesh is read
    but never mutated.

    Raises
    ------
    GeometryError
        On invalid mesh, invalid target dimensions, invalid
        ``wall_thickness``, invalid ``inner_offset_mode``, or
        non-finite vertex normals.
    OpticsError
        On invalid refractive index values.
    """
    ior_air_f = _check_positive_finite(ior_air, "ior_air")
    ior_pet_f = _check_positive_finite(ior_pet, "ior_pet")
    ior_water_f = _check_positive_finite(ior_water, "ior_water")

    shell_mesh, scale_report = create_target_scaled_mesh_copy(
        mesh,
        target_height=target_height,
        target_diameter=target_diameter,
    )
    water_mesh, inner_offset_report = (
        create_inner_offset_mesh_from_vertex_normals(
            shell_mesh,
            thickness=wall_thickness,
            invert=True,
            offset_mode=inner_offset_mode,
        )
    )

    step_specs = (
        MultiMeshTraceStepSpec(
            mesh=shell_mesh,
            eta_i=ior_air_f,
            eta_t=ior_pet_f,
            label="air_to_pet_outer_shell",
        ),
        MultiMeshTraceStepSpec(
            mesh=water_mesh,
            eta_i=ior_pet_f,
            eta_t=ior_water_f,
            label="pet_to_water_inner_surface",
        ),
        MultiMeshTraceStepSpec(
            mesh=water_mesh,
            eta_i=ior_water_f,
            eta_t=ior_pet_f,
            label="water_to_pet_inner_surface",
        ),
        MultiMeshTraceStepSpec(
            mesh=shell_mesh,
            eta_i=ior_pet_f,
            eta_t=ior_air_f,
            label="pet_to_air_outer_shell",
        ),
    )

    return LegacyPetWaterTraceSetup(
        shell_mesh=shell_mesh,
        water_mesh=water_mesh,
        step_specs=step_specs,
        scale_report=scale_report,
        inner_offset_report=inner_offset_report,
        target_height=float(target_height),
        target_diameter=float(target_diameter),
        wall_thickness=float(wall_thickness),
        ior_air=ior_air_f,
        ior_pet=ior_pet_f,
        ior_water=ior_water_f,
    )


def _identity_scale_report_from_shell(
    shell_mesh: trimesh.Trimesh,
) -> BottleTargetScaleReport:
    """Synthesize a no-op scale report for a caller-prepared shell mesh.

    The caller-prepared shell is treated as if it had already been
    scaled by an identity transform; ``original_*`` and
    ``scaled_*`` fields read the same bounds and ``scale_xyz`` is
    ``(1.0, 1.0, 1.0)``. ``target_height`` / ``target_diameter`` are
    derived from the shell's actual bounding-box extents so the
    report stays self-consistent without claiming a particular
    target dimension was applied.
    """
    bounds = np.asarray(shell_mesh.bounds, dtype=float)
    extents = bounds[1] - bounds[0]
    xy_extent = float(max(extents[0], extents[1]))
    z_extent = float(extents[2])
    return BottleTargetScaleReport(
        original_bounds_min=bounds[0].astype(float, copy=True),
        original_bounds_max=bounds[1].astype(float, copy=True),
        scaled_bounds_min=bounds[0].astype(float, copy=True),
        scaled_bounds_max=bounds[1].astype(float, copy=True),
        target_height=z_extent,
        target_diameter=xy_extent,
        original_height=z_extent,
        original_xy_extent=xy_extent,
        scale_xyz=(1.0, 1.0, 1.0),
        scaled_height=z_extent,
        scaled_xy_extent=xy_extent,
    )


def create_legacy_pet_water_trace_setup_from_shell_mesh(
    shell_mesh: trimesh.Trimesh,
    *,
    wall_thickness: float = _DEFAULT_WALL_THICKNESS,
    inner_offset_mode: str = "auto",
    ior_air: float = _DEFAULT_IOR_AIR,
    ior_pet: float = _DEFAULT_IOR_PET,
    ior_water: float = _DEFAULT_IOR_WATER,
) -> LegacyPetWaterTraceSetup:
    """Build a legacy four-step PET-water trace setup from a prepared shell.

    Subdivided actual STL risk-guided pattern smoke check; **not a
    physical PET-bottle validation**. Sibling of
    :func:`create_legacy_pet_water_trace_setup` for the case where
    the caller has already produced a scaled (and possibly
    subdivided or otherwise refined) shell mesh and wants to skip
    the legacy anisotropic target-dimension scaling step. The
    helper:

    1. Treats ``shell_mesh`` as already prepared and does **not**
       scale it.
    2. Builds the inner offset water boundary via
       :func:`create_inner_offset_mesh_from_vertex_normals` with
       ``invert=True`` and the caller-supplied
       ``inner_offset_mode``.
    3. Emits the same four
       :class:`MultiMeshTraceStepSpec` entries as
       :func:`create_legacy_pet_water_trace_setup`.

    The returned ``LegacyPetWaterTraceSetup`` carries an
    identity-scale report (``scale_xyz == (1, 1, 1)`` with
    ``target_*`` values derived from the shell's bounding-box
    extents) so downstream consumers that only read the report's
    bounds keep working. The input mesh is read but never mutated.

    Subdivision (or any other mesh refinement performed before
    calling this helper) is a synthetic numerical refinement for
    diagnostics. It is **not** mesh repair, **not** manufacturing
    validation, and **not** a calibrated physical model. This
    helper performs **no** ray tracing.

    Raises
    ------
    GeometryError
        On invalid mesh, invalid ``wall_thickness``, invalid
        ``inner_offset_mode``, or non-finite vertex normals.
    OpticsError
        On invalid refractive index values.
    """
    ior_air_f = _check_positive_finite(ior_air, "ior_air")
    ior_pet_f = _check_positive_finite(ior_pet, "ior_pet")
    ior_water_f = _check_positive_finite(ior_water, "ior_water")

    if not isinstance(shell_mesh, trimesh.Trimesh):
        raise OpticsError(
            "shell_mesh must be a trimesh.Trimesh; got "
            f"{type(shell_mesh).__name__}"
        )

    water_mesh, inner_offset_report = (
        create_inner_offset_mesh_from_vertex_normals(
            shell_mesh,
            thickness=wall_thickness,
            invert=True,
            offset_mode=inner_offset_mode,
        )
    )

    step_specs = (
        MultiMeshTraceStepSpec(
            mesh=shell_mesh,
            eta_i=ior_air_f,
            eta_t=ior_pet_f,
            label="air_to_pet_outer_shell",
        ),
        MultiMeshTraceStepSpec(
            mesh=water_mesh,
            eta_i=ior_pet_f,
            eta_t=ior_water_f,
            label="pet_to_water_inner_surface",
        ),
        MultiMeshTraceStepSpec(
            mesh=water_mesh,
            eta_i=ior_water_f,
            eta_t=ior_pet_f,
            label="water_to_pet_inner_surface",
        ),
        MultiMeshTraceStepSpec(
            mesh=shell_mesh,
            eta_i=ior_pet_f,
            eta_t=ior_air_f,
            label="pet_to_air_outer_shell",
        ),
    )

    scale_report = _identity_scale_report_from_shell(shell_mesh)

    return LegacyPetWaterTraceSetup(
        shell_mesh=shell_mesh,
        water_mesh=water_mesh,
        step_specs=step_specs,
        scale_report=scale_report,
        inner_offset_report=inner_offset_report,
        target_height=float(scale_report.target_height),
        target_diameter=float(scale_report.target_diameter),
        wall_thickness=float(wall_thickness),
        ior_air=ior_air_f,
        ior_pet=ior_pet_f,
        ior_water=ior_water_f,
    )
