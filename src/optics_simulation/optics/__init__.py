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

__all__ = [
    "DetectorAccumulationResult",
    "DetectorGrid",
    "DetectorHitResult",
    "DetectorPlane",
    "IntersectionResult",
    "MultiStepTraceResult",
    "OpticsError",
    "PropagationResult",
    "RayBundle",
    "RefractionResult",
    "ShellHitSurfaceClassification",
    "ShellMediumPreset",
    "ShellMediumTrackingResult",
    "ShellMediumTrackingStepSummary",
    "SingleInterfacePipelineResult",
    "accumulate_detector_hits",
    "classify_synthetic_shell_hit_surfaces",
    "create_detector_grid",
    "create_detector_plane",
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
    "run_multi_step_trace",
    "run_single_interface_pipeline",
    "run_surface_classified_shell_trace",
]
