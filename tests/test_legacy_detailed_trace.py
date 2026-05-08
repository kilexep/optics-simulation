import pytest

from optics_simulation.geometry import (
    HitSurfaceCoordinates,
    SurfaceCoordinateMap,
    create_subdivided_synthetic_bottle_body,
)
from optics_simulation.metrics import (
    DetectorIrradianceSurrogate,
    OpticalMetrics,
)
from optics_simulation.optics import (
    DetectorAccumulationResult,
    DetectorHitResult,
    LegacyDetailedTraceResult,
    MultiMeshTraceResult,
    OpticsError,
    RayBundle,
    create_legacy_pet_water_trace_setup,
    run_legacy_pet_water_detailed_trace,
)


def _setup():
    mesh = create_subdivided_synthetic_bottle_body(
        radius=30.0, height=120.0,
        sections=32, height_segments=8,
    )
    return create_legacy_pet_water_trace_setup(
        mesh,
        target_height=225.6, target_diameter=72.1,
        wall_thickness=0.3, inner_offset_mode="auto",
    )


def _common_kwargs():
    return dict(
        angle_degrees=0.0,
        detector_distance=120.0,
        source_width=80.0,
        source_height=240.0,
        source_radius=200.0,
        sample_count_y=4,
        sample_count_z=4,
        detector_size=400.0,
        detector_resolution=(20, 20),
    )


def test_returns_legacy_detailed_trace_result() -> None:
    setup = _setup()
    result = run_legacy_pet_water_detailed_trace(
        setup=setup, **_common_kwargs(),
    )
    assert isinstance(result, LegacyDetailedTraceResult)
    assert isinstance(result.rays, RayBundle)


def test_result_has_trace_result() -> None:
    setup = _setup()
    result = run_legacy_pet_water_detailed_trace(
        setup=setup, **_common_kwargs(),
    )
    assert isinstance(result.trace_result, MultiMeshTraceResult)
    assert int(result.trace_result.step_count) >= 0


def test_result_has_detector_and_metrics() -> None:
    setup = _setup()
    result = run_legacy_pet_water_detailed_trace(
        setup=setup, **_common_kwargs(),
    )
    assert isinstance(result.detector_hits, DetectorHitResult)
    assert isinstance(result.accumulation, DetectorAccumulationResult)
    assert isinstance(
        result.irradiance_surrogate, DetectorIrradianceSurrogate
    )
    assert isinstance(result.optical_metrics, OpticalMetrics)


def test_first_shell_hit_coordinates_match_initial_ray_count() -> None:
    setup = _setup()
    result = run_legacy_pet_water_detailed_trace(
        setup=setup, **_common_kwargs(),
    )
    assert isinstance(
        result.first_shell_hit_coordinates, HitSurfaceCoordinates
    )
    assert int(result.first_shell_hit_coordinates.ray_count) == int(
        result.rays.ray_count
    )
    assert isinstance(result.shell_surface_map, SurfaceCoordinateMap)


def test_source_area_equals_width_times_height() -> None:
    setup = _setup()
    kwargs = _common_kwargs()
    result = run_legacy_pet_water_detailed_trace(
        setup=setup, **kwargs,
    )
    assert result.source_area == pytest.approx(
        float(kwargs["source_width"]) * float(kwargs["source_height"]),
        abs=1e-9,
    )


def test_no_file_output(tmp_path) -> None:
    setup = _setup()
    before = sorted(tmp_path.iterdir())
    run_legacy_pet_water_detailed_trace(
        setup=setup, **_common_kwargs(),
    )
    after = sorted(tmp_path.iterdir())
    assert before == after


def test_invalid_setup_raises_optics_error() -> None:
    with pytest.raises(OpticsError, match="LegacyPetWaterTraceSetup"):
        run_legacy_pet_water_detailed_trace(
            setup="not a setup",  # type: ignore[arg-type]
            **_common_kwargs(),
        )


def test_invalid_detector_distance_raises() -> None:
    setup = _setup()
    bad_kwargs = _common_kwargs()
    bad_kwargs["detector_distance"] = 0.0
    with pytest.raises(OpticsError, match="detector_distance"):
        run_legacy_pet_water_detailed_trace(
            setup=setup, **bad_kwargs,
        )
