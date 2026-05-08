"""Subdivided actual-STL risk-guided patterned PET-water setup.

Subdivided actual STL risk-guided pattern smoke check; **not a
physical PET-bottle validation**. Sits between
:func:`create_target_scaled_mesh_copy` and
:func:`create_risk_guided_legacy_patterned_pet_water_setup` for
the diagnostic case where the caller wants to apply a
risk-guided Gaussian dimple pattern on a synthetically subdivided
copy of the actual STL instead of the raw scaled shell. The
purpose is to disentangle two effects observed in earlier
risk-guided pattern sweeps:

- Whether the "mixed" tradeoff outcome is driven by the pattern
  parameters themselves.
- Whether it is driven by the underlying STL being too coarse to
  represent a smooth indentation at the requested ``sigma_u`` /
  ``sigma_v`` / ``max_depth``.

Pipeline
--------
1. Anisotropically scale the input STL via
   :func:`create_target_scaled_mesh_copy`.
2. Compute a pattern-resolution readiness diagnostic on the
   scaled shell.
3. Subdivide the scaled shell ``subdivision_iterations`` times
   via :func:`create_subdivided_mesh_copy_for_patterning`.
4. Recompute the body-region include mask and
   pattern-resolution readiness on the subdivided shell.
5. Build an unpatterned legacy four-step setup directly on the
   subdivided shell via
   :func:`create_legacy_pet_water_trace_setup_from_shell_mesh`
   (no further scaling).
6. Build the risk-guided patterned setup on the same subdivided
   shell via
   :func:`create_risk_guided_legacy_patterned_pet_water_setup`,
   using the caller-supplied ``risk_map`` and the subdivided
   body mask.

Limitations
-----------
- Subdivision is a **synthetic numerical refinement** for
  pattern-resolution diagnostics, **not** mesh repair.
- The subdivided shell is **not** a validated manufacturing
  surface.
- The generated pattern is risk-guided but not optimized.
- This module performs **no** ray tracing, **no** thermal
  simulation, and **no** file I/O.
- This is **not** fire-prevention validation, **not** PET-bottle
  safety, and **not** a calibrated physical model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import trimesh

from optics_simulation.contribution.risk_map import RiskMap
from optics_simulation.geometry.actual_stl_masks import (
    ActualBottleBodyMaskReport,
    create_actual_bottle_body_vertex_mask,
)
from optics_simulation.geometry.bottle_scaling import (
    create_target_scaled_mesh_copy,
)
from optics_simulation.geometry.surface_coordinates import (
    create_vertex_surface_coordinates,
)
from optics_simulation.optics.legacy_patterned_pet_water import (
    LegacyPatternedPetWaterSetup,
    create_risk_guided_legacy_patterned_pet_water_setup,
)
from optics_simulation.optics.legacy_pet_water import (
    LegacyPetWaterTraceSetup,
    create_legacy_pet_water_trace_setup_from_shell_mesh,
)
from optics_simulation.optics.ray import OpticsError
from optics_simulation.pattern.gaussian import PatternError
from optics_simulation.pattern.resolution_readiness import (
    MeshSubdivisionReport,
    PatternResolutionReadinessReport,
    compute_pattern_resolution_readiness,
    create_subdivided_mesh_copy_for_patterning,
)


@dataclass(frozen=True)
class SubdividedLegacyPatternSetup:
    base_scaled_shell: trimesh.Trimesh
    subdivided_shell: trimesh.Trimesh
    original_subdivided_setup: LegacyPetWaterTraceSetup
    patterned_subdivided_setup: LegacyPatternedPetWaterSetup
    subdivision_report: MeshSubdivisionReport
    original_readiness: PatternResolutionReadinessReport
    subdivided_readiness: PatternResolutionReadinessReport
    body_mask_report: ActualBottleBodyMaskReport
    risk_map: RiskMap
    moved_vertex_count: int
    max_displacement: float


def create_subdivided_risk_guided_legacy_pattern_setup(
    mesh: trimesh.Trimesh,
    *,
    target_height: float = 225.6,
    target_diameter: float = 72.1,
    wall_thickness: float = 0.3,
    inner_offset_mode: str = "auto",
    subdivision_iterations: int = 1,
    risk_map: RiskMap,
    pattern_count: int = 20,
    pattern_amplitude: float = 1.0,
    pattern_sigma_u: float = 0.03,
    pattern_sigma_v: float = 0.03,
    pattern_max_depth: float = 0.05,
    pattern_seed: int = 42,
    active_threshold: float = 0.01,
    readiness_min_vertices_required: int = 200,
    readiness_min_sigma_to_spacing_ratio: float = 2.0,
    readiness_max_depth_to_edge_ratio: float = 0.25,
) -> SubdividedLegacyPatternSetup:
    """Build a risk-guided patterned PET-water setup on a subdivided STL.

    Subdivided actual STL risk-guided pattern smoke check; **not a
    physical PET-bottle validation**. See module docstring for the
    pipeline. Performs **no** ray tracing.

    Subdivision is a synthetic numerical refinement for
    pattern-resolution diagnostics, not mesh repair, not
    manufacturing validation, and not a calibrated physical model.

    Raises
    ------
    GeometryError
        On invalid mesh, invalid target dimensions, invalid wall
        thickness, invalid inner offset mode, or non-finite
        vertex normals.
    PatternError
        On pattern / displacement validation failures or invalid
        readiness-threshold values.
    OpticsError
        On invalid refractive index values, invalid subdivision
        iterations, or invalid risk map type.
    """
    if not isinstance(risk_map, RiskMap):
        raise OpticsError(
            "risk_map must be a RiskMap; "
            f"got {type(risk_map).__name__}"
        )
    if (
        isinstance(subdivision_iterations, bool)
        or not isinstance(subdivision_iterations, int)
        or int(subdivision_iterations) <= 0
    ):
        raise OpticsError(
            "subdivision_iterations must be a positive int; "
            f"got {subdivision_iterations!r}"
        )

    base_scaled_shell, _ = create_target_scaled_mesh_copy(
        mesh,
        target_height=float(target_height),
        target_diameter=float(target_diameter),
    )

    base_body_mask = create_actual_bottle_body_vertex_mask(
        base_scaled_shell,
    )
    base_surface = create_vertex_surface_coordinates(
        base_scaled_shell,
    )
    original_readiness = compute_pattern_resolution_readiness(
        base_scaled_shell,
        include_mask=base_body_mask.include_mask,
        surface_map=base_surface,
        sigma_u=float(pattern_sigma_u),
        sigma_v=float(pattern_sigma_v),
        max_depth=float(pattern_max_depth),
        min_vertices_required=int(readiness_min_vertices_required),
        min_sigma_to_spacing_ratio=float(
            readiness_min_sigma_to_spacing_ratio
        ),
        max_depth_to_edge_ratio=float(
            readiness_max_depth_to_edge_ratio
        ),
    )

    subdivided_shell, subdivision_report = (
        create_subdivided_mesh_copy_for_patterning(
            base_scaled_shell,
            iterations=int(subdivision_iterations),
        )
    )

    sub_body_mask = create_actual_bottle_body_vertex_mask(
        subdivided_shell,
    )
    sub_surface = create_vertex_surface_coordinates(
        subdivided_shell,
    )
    subdivided_readiness = compute_pattern_resolution_readiness(
        subdivided_shell,
        include_mask=sub_body_mask.include_mask,
        surface_map=sub_surface,
        sigma_u=float(pattern_sigma_u),
        sigma_v=float(pattern_sigma_v),
        max_depth=float(pattern_max_depth),
        min_vertices_required=int(readiness_min_vertices_required),
        min_sigma_to_spacing_ratio=float(
            readiness_min_sigma_to_spacing_ratio
        ),
        max_depth_to_edge_ratio=float(
            readiness_max_depth_to_edge_ratio
        ),
    )

    original_subdivided_setup = (
        create_legacy_pet_water_trace_setup_from_shell_mesh(
            subdivided_shell,
            wall_thickness=float(wall_thickness),
            inner_offset_mode=str(inner_offset_mode),
        )
    )

    patterned_subdivided_setup = (
        create_risk_guided_legacy_patterned_pet_water_setup(
            original_setup=original_subdivided_setup,
            risk_map=risk_map,
            body_include_mask=sub_body_mask.include_mask,
            pattern_count=int(pattern_count),
            pattern_amplitude=float(pattern_amplitude),
            pattern_sigma_u=float(pattern_sigma_u),
            pattern_sigma_v=float(pattern_sigma_v),
            pattern_max_depth=float(pattern_max_depth),
            pattern_seed=int(pattern_seed),
            active_threshold=float(active_threshold),
            wall_thickness=float(wall_thickness),
            inner_offset_mode=str(inner_offset_mode),
        )
    )

    moved = patterned_subdivided_setup.patterned_mesh_result
    return SubdividedLegacyPatternSetup(
        base_scaled_shell=base_scaled_shell,
        subdivided_shell=subdivided_shell,
        original_subdivided_setup=original_subdivided_setup,
        patterned_subdivided_setup=patterned_subdivided_setup,
        subdivision_report=subdivision_report,
        original_readiness=original_readiness,
        subdivided_readiness=subdivided_readiness,
        body_mask_report=sub_body_mask,
        risk_map=risk_map,
        moved_vertex_count=int(moved.moved_vertex_count),
        max_displacement=float(moved.max_displacement),
    )
