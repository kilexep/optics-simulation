"""Legacy patterned PET-water four-step trace setup for actual STL.

Actual STL patterned optical-to-thermal risk smoke check; **not a
physical PET-bottle validation**. Wraps the legacy four-step
``air -> PET -> water -> PET -> air`` trace setup with an
**outer-shell pattern** applied to the diagnostic body-region
mask of an actual STL bottle. The patterning pipeline is:

1. Build the unpatterned legacy setup
   (:func:`create_legacy_pet_water_trace_setup`).
2. Build a body-region include-mask
   (:func:`create_actual_bottle_body_vertex_mask`).
3. Sample a synthetic uniform Gaussian dimple pattern over a
   uniform :class:`RiskMap` (no contribution-map back-tracking).
4. Compute per-vertex pattern depths via
   :func:`compute_vertex_displacement_amounts` with the
   body-region mask.
5. Apply the depths via
   :func:`create_actual_stl_patterned_mesh_copy` so the moved
   vertices end up radially inward regardless of the STL's vertex
   normal convention.
6. Build the patterned inner offset water mesh via
   :func:`create_inner_offset_mesh_from_vertex_normals` with the
   caller-supplied ``inner_offset_mode``.
7. Emit the patterned four-step :class:`MultiMeshTraceStepSpec`
   sequence.

Limitations
-----------
- The body mask is a **diagnostic heuristic**, not a verified
  manufacturing surface.
- The pattern is a synthetic uniform Gaussian, not optimized
  against a contribution map.
- Normal-direction handling is geometric robustness, not physical
  validation.
- This helper performs **no** ray tracing.
- This is **not** fire-prevention validation, **not** PET-bottle
  safety, and **not** a manufacturable real PET-bottle pattern.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from optics_simulation.geometry.actual_stl_masks import (
    ActualBottleBodyMaskReport,
    create_actual_bottle_body_vertex_mask,
)
from optics_simulation.geometry.bottle_scaling import (
    InnerOffsetMeshReport,
    create_inner_offset_mesh_from_vertex_normals,
)
from optics_simulation.geometry.surface_coordinates import (
    create_vertex_surface_coordinates,
)
from optics_simulation.optics.legacy_pet_water import (
    LegacyPetWaterTraceSetup,
    create_legacy_pet_water_trace_setup,
)
from optics_simulation.optics.multi_mesh_trace import (
    MultiMeshTraceStepSpec,
)
from optics_simulation.pattern.actual_stl_displacement import (
    ActualSTLPatternedMeshResult,
    create_actual_stl_patterned_mesh_copy,
)
from optics_simulation.pattern.gaussian import (
    PatternError,
    create_gaussian_dimple_pattern,
)
from optics_simulation.pattern.vertex_displacement import (
    VertexPatternDisplacement,
    compute_vertex_displacement_amounts,
)


_DEFAULT_TARGET_HEIGHT = 225.6
_DEFAULT_TARGET_DIAMETER = 72.1
_DEFAULT_WALL_THICKNESS = 0.3
_DEFAULT_IOR_AIR = 1.0
_DEFAULT_IOR_PET = 1.57
_DEFAULT_IOR_WATER = 1.333

_PATTERN_RISK_MAP_RESOLUTION = (64, 64)


@dataclass(frozen=True)
class LegacyPatternedPetWaterSetup:
    original_setup: LegacyPetWaterTraceSetup
    patterned_shell_mesh: trimesh.Trimesh
    patterned_water_mesh: trimesh.Trimesh
    patterned_step_specs: tuple[MultiMeshTraceStepSpec, ...]
    body_mask_report: ActualBottleBodyMaskReport
    displacement_result: VertexPatternDisplacement
    patterned_mesh_result: ActualSTLPatternedMeshResult
    patterned_inner_offset_report: InnerOffsetMeshReport


def _build_uniform_risk_map(nv: int, nu: int):
    # Local import to break the
    # geometry / optics / contribution circular import at package
    # init time. ``RiskMap`` is functionally just a uniform
    # distribution wrapper here; the rest of the function does not
    # depend on contribution-package state.
    from optics_simulation.contribution.risk_map import RiskMap

    risk = np.ones((int(nv), int(nu)), dtype=float)
    total = float(risk.sum())
    return RiskMap(
        risk_map=risk.copy(),
        probability_map=(risk / total).astype(float, copy=True),
        active_mask=np.ones((int(nv), int(nu)), dtype=bool),
        total_risk=total,
        active_count=int(nv * nu),
        epsilon=0.0,
        threshold=None,
    )


def create_legacy_patterned_pet_water_setup(
    mesh: trimesh.Trimesh,
    *,
    target_height: float = _DEFAULT_TARGET_HEIGHT,
    target_diameter: float = _DEFAULT_TARGET_DIAMETER,
    wall_thickness: float = _DEFAULT_WALL_THICKNESS,
    inner_offset_mode: str = "auto",
    pattern_count: int = 20,
    pattern_sigma_u: float = 0.03,
    pattern_sigma_v: float = 0.03,
    pattern_max_depth: float = 0.05,
    pattern_seed: int = 42,
    active_threshold: float = 0.01,
    z_min_fraction: float = 0.15,
    z_max_fraction: float = 0.85,
    radial_quantile: float = 0.50,
    ior_air: float = _DEFAULT_IOR_AIR,
    ior_pet: float = _DEFAULT_IOR_PET,
    ior_water: float = _DEFAULT_IOR_WATER,
) -> LegacyPatternedPetWaterSetup:
    """Build a legacy four-step PET-water trace setup with an outer-shell pattern.

    Actual STL patterned optical-to-thermal risk smoke check;
    **not a physical PET-bottle validation**. See module docstring
    for the patterning pipeline. Performs **no** ray tracing.

    Raises
    ------
    GeometryError
        On invalid mesh, invalid target dimensions, invalid wall
        thickness, invalid inner offset mode, or non-finite
        vertex normals.
    PatternError
        On pattern / displacement validation failures.
    OpticsError
        On invalid refractive index values.
    """
    original_setup = create_legacy_pet_water_trace_setup(
        mesh,
        target_height=target_height,
        target_diameter=target_diameter,
        wall_thickness=wall_thickness,
        inner_offset_mode=inner_offset_mode,
        ior_air=ior_air,
        ior_pet=ior_pet,
        ior_water=ior_water,
    )

    scaled_shell = original_setup.shell_mesh
    surface_map = create_vertex_surface_coordinates(scaled_shell)
    body_mask_report = create_actual_bottle_body_vertex_mask(
        scaled_shell,
        z_min_fraction=z_min_fraction,
        z_max_fraction=z_max_fraction,
        radial_quantile=radial_quantile,
    )

    risk_map_obj = _build_uniform_risk_map(
        _PATTERN_RISK_MAP_RESOLUTION[0],
        _PATTERN_RISK_MAP_RESOLUTION[1],
    )
    pattern = create_gaussian_dimple_pattern(
        risk_map_obj,
        count=int(pattern_count),
        sigma_u=float(pattern_sigma_u),
        sigma_v=float(pattern_sigma_v),
        max_depth=float(pattern_max_depth),
        seed=int(pattern_seed),
    )

    displacement_result = compute_vertex_displacement_amounts(
        scaled_shell, surface_map, pattern,
        active_threshold=float(active_threshold),
        include_mask=body_mask_report.include_mask,
    )

    patterned_mesh_result = create_actual_stl_patterned_mesh_copy(
        scaled_shell, displacement_result,
    )
    patterned_shell_mesh = patterned_mesh_result.patterned_mesh

    patterned_water_mesh, patterned_inner_offset_report = (
        create_inner_offset_mesh_from_vertex_normals(
            patterned_shell_mesh,
            thickness=wall_thickness,
            invert=True,
            offset_mode=inner_offset_mode,
        )
    )

    patterned_step_specs = (
        MultiMeshTraceStepSpec(
            mesh=patterned_shell_mesh,
            eta_i=float(ior_air),
            eta_t=float(ior_pet),
            label="air_to_pet_outer_shell",
        ),
        MultiMeshTraceStepSpec(
            mesh=patterned_water_mesh,
            eta_i=float(ior_pet),
            eta_t=float(ior_water),
            label="pet_to_water_inner_surface",
        ),
        MultiMeshTraceStepSpec(
            mesh=patterned_water_mesh,
            eta_i=float(ior_water),
            eta_t=float(ior_pet),
            label="water_to_pet_inner_surface",
        ),
        MultiMeshTraceStepSpec(
            mesh=patterned_shell_mesh,
            eta_i=float(ior_pet),
            eta_t=float(ior_air),
            label="pet_to_air_outer_shell",
        ),
    )

    return LegacyPatternedPetWaterSetup(
        original_setup=original_setup,
        patterned_shell_mesh=patterned_shell_mesh,
        patterned_water_mesh=patterned_water_mesh,
        patterned_step_specs=patterned_step_specs,
        body_mask_report=body_mask_report,
        displacement_result=displacement_result,
        patterned_mesh_result=patterned_mesh_result,
        patterned_inner_offset_report=patterned_inner_offset_report,
    )


def _synthesize_body_mask_report_from_user_mask(
    *,
    scaled_shell,
    body_include_mask: np.ndarray,
) -> ActualBottleBodyMaskReport:
    """Wrap a caller-supplied include mask in an ActualBottleBodyMaskReport.

    Falls back to :func:`create_actual_bottle_body_vertex_mask` for
    the non-mask metadata fields (``z_min`` / ``z_max`` /
    ``radial_min`` / ``radial_max`` / ``radial_median`` /
    ``radial_threshold`` / ``body_z_min`` / ``body_z_max``) so the
    returned report is internally consistent even though the
    caller bypassed the heuristic.
    """
    n = int(len(scaled_shell.vertices))
    arr = np.asarray(body_include_mask)
    if arr.shape != (n,):
        raise PatternError(
            f"body_include_mask must have shape ({n},); got {arr.shape}"
        )
    if arr.dtype == bool:
        mask = arr.astype(bool, copy=True)
    elif np.issubdtype(arr.dtype, np.number):
        if not np.isfinite(arr).all():
            raise PatternError(
                "body_include_mask contains NaN or inf values"
            )
        mask = arr.astype(bool, copy=True)
    else:
        raise PatternError(
            f"body_include_mask must be bool or numeric castable "
            f"to bool; got dtype={arr.dtype}"
        )

    default_report = create_actual_bottle_body_vertex_mask(
        scaled_shell,
    )
    notes = (
        "User-provided body_include_mask used; z and radial "
        "fields are defaults from "
        "create_actual_bottle_body_vertex_mask.",
        "Pattern displacement applied through this mask is "
        "diagnostic, not a verified manufacturing surface.",
    )
    return ActualBottleBodyMaskReport(
        include_mask=mask,
        vertex_count=n,
        selected_count=int(mask.sum()),
        selected_fraction=float(mask.sum()) / float(n),
        z_min=default_report.z_min,
        z_max=default_report.z_max,
        body_z_min=default_report.body_z_min,
        body_z_max=default_report.body_z_max,
        radial_min=default_report.radial_min,
        radial_max=default_report.radial_max,
        radial_median=default_report.radial_median,
        radial_threshold=default_report.radial_threshold,
        notes=notes,
    )


def create_risk_guided_legacy_patterned_pet_water_setup(
    *,
    original_setup: LegacyPetWaterTraceSetup,
    risk_map,
    body_include_mask: np.ndarray,
    pattern_count: int = 20,
    pattern_amplitude: float = 1.0,
    pattern_sigma_u: float = 0.03,
    pattern_sigma_v: float = 0.03,
    pattern_max_depth: float = 0.05,
    pattern_seed: int = 42,
    active_threshold: float = 0.01,
    wall_thickness: float = _DEFAULT_WALL_THICKNESS,
    inner_offset_mode: str = "auto",
    ior_air: float = _DEFAULT_IOR_AIR,
    ior_pet: float = _DEFAULT_IOR_PET,
    ior_water: float = _DEFAULT_IOR_WATER,
) -> LegacyPatternedPetWaterSetup:
    """Build a patterned PET-water setup driven by a caller-supplied risk map.

    Actual STL hotspot-backtracked risk-guided pattern smoke
    check; **not a physical PET-bottle validation**. Reuses
    ``original_setup.shell_mesh`` as the scaled shell and consumes
    a caller-supplied ``risk_map`` and ``body_include_mask``
    (typically produced by
    :func:`build_legacy_hotspot_contribution_map` and
    :func:`create_actual_bottle_body_vertex_mask` respectively),
    sampling Gaussian dimple centers from the risk map and
    applying them only to the body-region vertices.

    The output is the same :class:`LegacyPatternedPetWaterSetup`
    shape as :func:`create_legacy_patterned_pet_water_setup`, so
    the legacy scan runner can consume it through the existing
    :class:`LegacyPetWaterTraceSetup` adapter. Vertex-displacement
    direction is autodetected via
    :func:`create_actual_stl_patterned_mesh_copy`.

    Raises
    ------
    PatternError
        On invalid inputs (mask shape mismatch, non-finite values,
        risk-map / mesh inconsistency, or a pattern_count > 0 with
        a zero-mass risk map).
    GeometryError, OpticsError
        On downstream geometry or optics validation failures.
    """
    if not isinstance(original_setup, LegacyPetWaterTraceSetup):
        raise PatternError(
            "original_setup must be a LegacyPetWaterTraceSetup; "
            f"got {type(original_setup).__name__}"
        )

    scaled_shell = original_setup.shell_mesh
    surface_map = create_vertex_surface_coordinates(scaled_shell)
    body_mask_report = _synthesize_body_mask_report_from_user_mask(
        scaled_shell=scaled_shell,
        body_include_mask=body_include_mask,
    )

    pattern = create_gaussian_dimple_pattern(
        risk_map,
        count=int(pattern_count),
        amplitude=float(pattern_amplitude),
        sigma_u=float(pattern_sigma_u),
        sigma_v=float(pattern_sigma_v),
        max_depth=float(pattern_max_depth),
        seed=int(pattern_seed),
    )

    displacement_result = compute_vertex_displacement_amounts(
        scaled_shell, surface_map, pattern,
        active_threshold=float(active_threshold),
        include_mask=body_mask_report.include_mask,
    )
    patterned_mesh_result = create_actual_stl_patterned_mesh_copy(
        scaled_shell, displacement_result,
    )
    patterned_shell_mesh = patterned_mesh_result.patterned_mesh

    patterned_water_mesh, patterned_inner_offset_report = (
        create_inner_offset_mesh_from_vertex_normals(
            patterned_shell_mesh,
            thickness=float(wall_thickness),
            invert=True,
            offset_mode=str(inner_offset_mode),
        )
    )
    patterned_step_specs = (
        MultiMeshTraceStepSpec(
            mesh=patterned_shell_mesh,
            eta_i=float(ior_air),
            eta_t=float(ior_pet),
            label="air_to_pet_outer_shell",
        ),
        MultiMeshTraceStepSpec(
            mesh=patterned_water_mesh,
            eta_i=float(ior_pet),
            eta_t=float(ior_water),
            label="pet_to_water_inner_surface",
        ),
        MultiMeshTraceStepSpec(
            mesh=patterned_water_mesh,
            eta_i=float(ior_water),
            eta_t=float(ior_pet),
            label="water_to_pet_inner_surface",
        ),
        MultiMeshTraceStepSpec(
            mesh=patterned_shell_mesh,
            eta_i=float(ior_pet),
            eta_t=float(ior_air),
            label="pet_to_air_outer_shell",
        ),
    )
    return LegacyPatternedPetWaterSetup(
        original_setup=original_setup,
        patterned_shell_mesh=patterned_shell_mesh,
        patterned_water_mesh=patterned_water_mesh,
        patterned_step_specs=patterned_step_specs,
        body_mask_report=body_mask_report,
        displacement_result=displacement_result,
        patterned_mesh_result=patterned_mesh_result,
        patterned_inner_offset_report=patterned_inner_offset_report,
    )
