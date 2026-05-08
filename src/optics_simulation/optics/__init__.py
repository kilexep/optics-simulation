"""Optics package: ray sources, intersection, Snell/Fresnel, propagation, pipeline, multi-step, detector, accumulation."""
from optics_simulation.optics.detector import (
    DetectorHitResult,
    DetectorPlane,
    create_detector_plane,
    intersect_detector_plane,
)
from optics_simulation.optics.detector_accumulation import (
    DetectorAccumulationResult,
    DetectorGrid,
    accumulate_detector_hits,
    create_detector_grid,
)
from optics_simulation.optics.intersection import (
    IntersectionResult,
    intersect_rays,
)
from optics_simulation.optics.legacy_pet_water import (
    LegacyPetWaterTraceSetup,
    create_legacy_pet_water_trace_setup,
)
from optics_simulation.optics.legacy_scan import (
    LegacyOpticalScanEntry,
    LegacyOpticalScanResult,
    LegacyParityScanSummary,
    LegacyParitySchedule,
    LegacySourcePlaneConfig,
    create_legacy_experiment_schedule,
    create_oriented_parallel_ray_grid,
    run_legacy_pet_water_angle_distance_scan,
    summarize_legacy_optical_scan,
)
from optics_simulation.optics.media_presets import (
    ShellMediumPreset,
    create_shell_medium_preset,
)
from optics_simulation.optics.media_tracking import (
    ShellHitSurfaceClassification,
    ShellMediumTrackingResult,
    ShellMediumTrackingStepSummary,
    classify_synthetic_shell_hit_surfaces,
    expected_shell_surface_sequence,
    run_surface_classified_shell_trace,
)
from optics_simulation.optics.multi_mesh_trace import (
    MultiMeshTraceResult,
    MultiMeshTraceStepSpec,
    run_multi_mesh_trace,
)
from optics_simulation.optics.multi_step import (
    MultiStepTraceResult,
    run_multi_step_trace,
)
from optics_simulation.optics.pipeline import (
    SingleInterfacePipelineResult,
    run_single_interface_pipeline,
)
from optics_simulation.optics.propagation import (
    PropagationResult,
    propagate_through_interface,
)
from optics_simulation.optics.ray import (
    OpticsError,
    RayBundle,
    make_ray_bundle,
    parallel_ray_grid,
)
from optics_simulation.optics.refraction import (
    RefractionResult,
    fresnel_unpolarized,
    normalize_vector,
    refract_direction,
)


# Lazy attribute access for symbols whose eager import would
# trigger a circular geometry / optics / pattern / contribution
# import chain at package init time. The legacy patterned setup
# transitively imports ``contribution.risk_map`` (via
# ``pattern.gaussian``), which imports
# ``geometry.hit_coordinates`` -- which is the very module
# whose load causes ``optics/__init__.py`` to be loaded in the
# first place.
_LAZY_LEGACY_PATTERNED = {
    "LegacyPatternedPetWaterSetup",
    "create_legacy_patterned_pet_water_setup",
    "create_risk_guided_legacy_patterned_pet_water_setup",
}

_LAZY_LEGACY_DETAILED_TRACE = {
    "LegacyDetailedTraceResult",
    "run_legacy_pet_water_detailed_trace",
}

_LAZY_LEGACY_RISK_GUIDED_SWEEP = {
    "RiskGuidedPatternCandidateSpec",
    "RiskGuidedPatternSweepEntry",
    "RiskGuidedPatternSweepResult",
    "run_actual_stl_risk_guided_pattern_parameter_sweep",
}


def __getattr__(name: str):
    if name in _LAZY_LEGACY_PATTERNED:
        from optics_simulation.optics.legacy_patterned_pet_water import (
            LegacyPatternedPetWaterSetup,
            create_legacy_patterned_pet_water_setup,
            create_risk_guided_legacy_patterned_pet_water_setup,
        )
        return {
            "LegacyPatternedPetWaterSetup": LegacyPatternedPetWaterSetup,
            "create_legacy_patterned_pet_water_setup": (
                create_legacy_patterned_pet_water_setup
            ),
            "create_risk_guided_legacy_patterned_pet_water_setup": (
                create_risk_guided_legacy_patterned_pet_water_setup
            ),
        }[name]
    if name in _LAZY_LEGACY_DETAILED_TRACE:
        from optics_simulation.optics.legacy_detailed_trace import (
            LegacyDetailedTraceResult,
            run_legacy_pet_water_detailed_trace,
        )
        return {
            "LegacyDetailedTraceResult": LegacyDetailedTraceResult,
            "run_legacy_pet_water_detailed_trace": (
                run_legacy_pet_water_detailed_trace
            ),
        }[name]
    if name in _LAZY_LEGACY_RISK_GUIDED_SWEEP:
        from optics_simulation.optics.legacy_risk_guided_sweep import (
            RiskGuidedPatternCandidateSpec,
            RiskGuidedPatternSweepEntry,
            RiskGuidedPatternSweepResult,
            run_actual_stl_risk_guided_pattern_parameter_sweep,
        )
        return {
            "RiskGuidedPatternCandidateSpec": (
                RiskGuidedPatternCandidateSpec
            ),
            "RiskGuidedPatternSweepEntry": (
                RiskGuidedPatternSweepEntry
            ),
            "RiskGuidedPatternSweepResult": (
                RiskGuidedPatternSweepResult
            ),
            "run_actual_stl_risk_guided_pattern_parameter_sweep": (
                run_actual_stl_risk_guided_pattern_parameter_sweep
            ),
        }[name]
    raise AttributeError(
        f"module 'optics_simulation.optics' has no attribute {name!r}"
    )

__all__ = [
    "DetectorAccumulationResult",
    "DetectorGrid",
    "DetectorHitResult",
    "DetectorPlane",
    "IntersectionResult",
    "LegacyDetailedTraceResult",
    "LegacyOpticalScanEntry",
    "LegacyOpticalScanResult",
    "LegacyParityScanSummary",
    "LegacyParitySchedule",
    "LegacyPatternedPetWaterSetup",
    "LegacyPetWaterTraceSetup",
    "LegacySourcePlaneConfig",
    "MultiMeshTraceResult",
    "MultiMeshTraceStepSpec",
    "MultiStepTraceResult",
    "OpticsError",
    "PropagationResult",
    "RayBundle",
    "RefractionResult",
    "RiskGuidedPatternCandidateSpec",
    "RiskGuidedPatternSweepEntry",
    "RiskGuidedPatternSweepResult",
    "ShellHitSurfaceClassification",
    "ShellMediumPreset",
    "ShellMediumTrackingResult",
    "ShellMediumTrackingStepSummary",
    "SingleInterfacePipelineResult",
    "accumulate_detector_hits",
    "classify_synthetic_shell_hit_surfaces",
    "create_detector_grid",
    "create_detector_plane",
    "create_legacy_experiment_schedule",
    "create_legacy_patterned_pet_water_setup",
    "create_legacy_pet_water_trace_setup",
    "create_oriented_parallel_ray_grid",
    "create_risk_guided_legacy_patterned_pet_water_setup",
    "create_shell_medium_preset",
    "expected_shell_surface_sequence",
    "fresnel_unpolarized",
    "intersect_detector_plane",
    "intersect_rays",
    "make_ray_bundle",
    "normalize_vector",
    "parallel_ray_grid",
    "propagate_through_interface",
    "refract_direction",
    "run_actual_stl_risk_guided_pattern_parameter_sweep",
    "run_legacy_pet_water_angle_distance_scan",
    "run_legacy_pet_water_detailed_trace",
    "run_multi_mesh_trace",
    "run_multi_step_trace",
    "run_single_interface_pipeline",
    "run_surface_classified_shell_trace",
    "summarize_legacy_optical_scan",
]
