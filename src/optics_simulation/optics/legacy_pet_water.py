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
) -> LegacyPetWaterTraceSetup:
    """Build the legacy-style four-step PET-water trace setup.

    Legacy-style STL optical smoke check; **not a physical
    PET-bottle validation**. Steps:

    1. Apply legacy anisotropic target-dimension scaling via
       :func:`create_target_scaled_mesh_copy` so the bounding box
       matches ``(target_diameter, target_diameter, target_height)``.
    2. Build the inner offset water boundary via
       :func:`create_inner_offset_mesh_from_vertex_normals` with
       ``invert=True``.
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
        ``wall_thickness``, or non-finite vertex normals.
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
