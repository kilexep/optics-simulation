import numpy as np
import pytest

from optics_simulation.contribution import (
    ContributionMap,
    LegacyHotspotContributionResult,
    RiskMap,
    build_legacy_hotspot_contribution_map,
)
from optics_simulation.geometry import (
    create_subdivided_synthetic_bottle_body,
)
from optics_simulation.metrics import HotspotSelection, MetricsError
from optics_simulation.optics import (
    create_legacy_pet_water_trace_setup,
    run_legacy_pet_water_detailed_trace,
)


def _detailed():
    mesh = create_subdivided_synthetic_bottle_body(
        radius=30.0, height=120.0,
        sections=32, height_segments=8,
    )
    setup = create_legacy_pet_water_trace_setup(
        mesh,
        target_height=225.6, target_diameter=72.1,
        wall_thickness=0.3, inner_offset_mode="auto",
    )
    return run_legacy_pet_water_detailed_trace(
        setup=setup,
        angle_degrees=0.0,
        detector_distance=120.0,
        source_width=80.0,
        source_height=240.0,
        source_radius=200.0,
        sample_count_y=6,
        sample_count_z=6,
        detector_size=400.0,
        detector_resolution=(20, 20),
    )


def test_returns_dataclass() -> None:
    detailed = _detailed()
    result = build_legacy_hotspot_contribution_map(
        detailed, top_percent=20.0,
    )
    assert isinstance(result, LegacyHotspotContributionResult)
    assert isinstance(result.hotspot_selection, HotspotSelection)
    assert isinstance(
        result.initial_aligned_selection, HotspotSelection,
    )
    assert isinstance(result.contribution_map, ContributionMap)
    assert isinstance(result.risk_map, RiskMap)


def test_hotspot_selected_count_nonneg() -> None:
    detailed = _detailed()
    result = build_legacy_hotspot_contribution_map(
        detailed, top_percent=20.0,
    )
    assert int(result.hotspot_selection.selected_count) >= 0


def test_mapped_initial_count_matches_selected_final_count() -> None:
    detailed = _detailed()
    result = build_legacy_hotspot_contribution_map(
        detailed, top_percent=20.0,
    )
    assert int(result.mapped_initial_ray_count) == int(
        result.selected_final_ray_count
    )
    assert (
        result.initial_ray_indices.shape[0]
        == result.mapped_initial_ray_count
    )
    if result.initial_ray_indices.size > 0:
        assert int(result.initial_ray_indices.min()) >= 0
        assert (
            int(result.initial_ray_indices.max())
            < int(detailed.rays.ray_count)
        )


def test_contribution_map_shape_matches_resolution() -> None:
    detailed = _detailed()
    res = (16, 32)
    result = build_legacy_hotspot_contribution_map(
        detailed, top_percent=20.0, contribution_resolution=res,
    )
    assert result.contribution_map.weight_map.shape == res
    assert result.contribution_map.count_map.shape == res
    assert result.contribution_resolution == res


def test_risk_map_shape_matches_resolution() -> None:
    detailed = _detailed()
    res = (16, 32)
    result = build_legacy_hotspot_contribution_map(
        detailed, top_percent=20.0, contribution_resolution=res,
    )
    assert result.risk_map.risk_map.shape == res
    assert result.risk_map.probability_map.shape == res
    assert result.risk_map.active_mask.shape == res


def test_no_file_output(tmp_path) -> None:
    detailed = _detailed()
    before = sorted(tmp_path.iterdir())
    build_legacy_hotspot_contribution_map(
        detailed, top_percent=20.0,
    )
    after = sorted(tmp_path.iterdir())
    assert before == after


def test_invalid_top_percent_raises() -> None:
    detailed = _detailed()
    for bad in (0.0, -1.0, 101.0, float("nan")):
        with pytest.raises(MetricsError, match="top_percent"):
            build_legacy_hotspot_contribution_map(
                detailed, top_percent=bad,
            )


def test_initial_aligned_selection_count_matches_initial_indices() -> None:
    detailed = _detailed()
    result = build_legacy_hotspot_contribution_map(
        detailed, top_percent=20.0,
    )
    aligned = result.initial_aligned_selection
    assert int(aligned.ray_count) == int(detailed.rays.ray_count)
    assert int(aligned.selected_count) == int(
        result.initial_ray_indices.shape[0]
    )
    if aligned.selected_count > 0:
        # selected_mask must be True at the initial indices.
        np.testing.assert_array_equal(
            aligned.selected_mask[result.initial_ray_indices],
            np.ones(result.initial_ray_indices.shape[0], dtype=bool),
        )
