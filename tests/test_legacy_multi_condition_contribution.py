import pytest

from optics_simulation.contribution import (
    ContributionMap,
    LegacyConditionSelection,
    LegacyHotspotContributionResult,
    LegacyMultiConditionContributionResult,
    RiskMap,
    build_multi_condition_legacy_hotspot_contribution_map,
    select_legacy_worst_conditions,
)
from optics_simulation.geometry import (
    create_subdivided_synthetic_bottle_body,
)
from optics_simulation.optics import (
    LegacyOpticalScanResult,
    OpticsError,
    create_legacy_pet_water_trace_setup,
    run_legacy_pet_water_angle_distance_scan,
)
from optics_simulation.thermal import (
    LegacyThermalRiskScanResult,
    compute_thermal_risk_over_legacy_optical_scan,
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


def _smoke_optical_thermal():
    setup = _setup()
    optical = run_legacy_pet_water_angle_distance_scan(
        setup=setup,
        angles_degrees=(0.0, 15.0),
        detector_distances=(100.0, 140.0),
        source_width=80.0,
        source_height=240.0,
        sample_count_y=5,
        sample_count_z=5,
        detector_size=400.0,
        detector_resolution=(20, 20),
        epsilon=0.1,
        store_irradiance_surrogate=True,
    )
    thermal = compute_thermal_risk_over_legacy_optical_scan(
        optical,
        nominal_incident_irradiance_w_m2=1000.0,
        duration_s=5.0,
        dt_s=1.0,
        areal_heat_capacity_j_m2k=1200.0,
        absorptivity=0.8,
        h_conv_w_m2k=10.0,
        emissivity=0.9,
        threshold_temp_k=373.15,
    )
    return setup, optical, thermal


def _empty_optical() -> LegacyOpticalScanResult:
    return LegacyOpticalScanResult(
        entries=(),
        angle_count=0,
        detector_count=0,
        max_relative_irradiance_angle=None,
        max_relative_irradiance_detector_distance=None,
        max_relative_irradiance=None,
        max_c99_angle=None,
        max_c99_detector_distance=None,
        max_c99=None,
    )


def _empty_thermal() -> LegacyThermalRiskScanResult:
    return LegacyThermalRiskScanResult(
        entries=(),
        entry_count=0,
        max_temperature_k=None,
        max_temperature_angle=None,
        max_temperature_distance=None,
        max_top_percent_temperature_rise_k=None,
        max_top_percent_temperature_rise_angle=None,
        max_top_percent_temperature_rise_distance=None,
        max_threshold_exceeded_count=None,
        max_threshold_exceeded_count_angle=None,
        max_threshold_exceeded_count_distance=None,
        nominal_incident_irradiance_w_m2=1000.0,
    )


def test_select_returns_tuple() -> None:
    _, optical, thermal = _smoke_optical_thermal()
    selections = select_legacy_worst_conditions(
        optical, thermal, max_conditions=3,
    )
    assert isinstance(selections, tuple)
    for s in selections:
        assert isinstance(s, LegacyConditionSelection)


def test_select_empty_results_returns_empty_tuple() -> None:
    selections = select_legacy_worst_conditions(
        _empty_optical(), _empty_thermal(), max_conditions=3,
    )
    assert selections == ()


def test_select_unique_angle_distance_pairs() -> None:
    _, optical, thermal = _smoke_optical_thermal()
    selections = select_legacy_worst_conditions(
        optical, thermal, max_conditions=3,
    )
    pairs = [
        (float(s.angle_degrees), float(s.detector_distance))
        for s in selections
    ]
    assert len(pairs) == len(set(pairs))


def test_select_max_conditions_limits_output() -> None:
    _, optical, thermal = _smoke_optical_thermal()
    selections = select_legacy_worst_conditions(
        optical, thermal, max_conditions=1,
    )
    assert len(selections) <= 1


def test_select_reasons_preserved() -> None:
    _, optical, thermal = _smoke_optical_thermal()
    selections = select_legacy_worst_conditions(
        optical, thermal,
        max_conditions=3,
        include_reasons=("max_temperature", "max_c99"),
    )
    for s in selections:
        assert s.reason in ("max_temperature", "max_c99")


def test_select_invalid_max_conditions_raises() -> None:
    _, optical, thermal = _smoke_optical_thermal()
    with pytest.raises(OpticsError, match="max_conditions"):
        select_legacy_worst_conditions(
            optical, thermal, max_conditions=0,
        )


def test_select_unknown_reason_raises() -> None:
    _, optical, thermal = _smoke_optical_thermal()
    with pytest.raises(OpticsError, match="include_reasons"):
        select_legacy_worst_conditions(
            optical, thermal,
            max_conditions=3,
            include_reasons=("not_a_reason",),
        )


def test_build_returns_result() -> None:
    setup, optical, thermal = _smoke_optical_thermal()
    selections = select_legacy_worst_conditions(
        optical, thermal, max_conditions=2,
    )
    result = build_multi_condition_legacy_hotspot_contribution_map(
        setup=setup,
        conditions=selections,
        source_width=80.0,
        source_height=240.0,
        source_radius=200.0,
        sample_count_y=5,
        sample_count_z=5,
        detector_size=400.0,
        detector_resolution=(20, 20),
        epsilon=0.1,
        hotspot_top_percent=20.0,
        contribution_resolution=(16, 32),
    )
    assert isinstance(result, LegacyMultiConditionContributionResult)
    assert isinstance(result.aggregate_contribution_map, ContributionMap)
    assert isinstance(result.aggregate_risk_map, RiskMap)
    for r in result.per_condition_results:
        assert isinstance(r, LegacyHotspotContributionResult)


def test_build_condition_count_matches_selected() -> None:
    setup, optical, thermal = _smoke_optical_thermal()
    selections = select_legacy_worst_conditions(
        optical, thermal, max_conditions=2,
    )
    result = build_multi_condition_legacy_hotspot_contribution_map(
        setup=setup,
        conditions=selections,
        source_width=80.0,
        source_height=240.0,
        source_radius=200.0,
        sample_count_y=5,
        sample_count_z=5,
        detector_size=400.0,
        detector_resolution=(20, 20),
        epsilon=0.1,
        hotspot_top_percent=20.0,
        contribution_resolution=(16, 32),
    )
    assert int(result.condition_count) == int(len(selections))
    assert len(result.selected_conditions) == int(len(selections))
    assert len(result.per_condition_results) == int(len(selections))


def test_build_aggregate_contribution_shape_matches_resolution() -> None:
    setup, optical, thermal = _smoke_optical_thermal()
    selections = select_legacy_worst_conditions(
        optical, thermal, max_conditions=2,
    )
    res = (16, 32)
    result = build_multi_condition_legacy_hotspot_contribution_map(
        setup=setup,
        conditions=selections,
        source_width=80.0,
        source_height=240.0,
        source_radius=200.0,
        sample_count_y=5,
        sample_count_z=5,
        detector_size=400.0,
        detector_resolution=(20, 20),
        epsilon=0.1,
        hotspot_top_percent=20.0,
        contribution_resolution=res,
    )
    assert result.aggregate_contribution_map.weight_map.shape == res
    assert result.aggregate_contribution_map.count_map.shape == res
    assert result.contribution_resolution == res


def test_build_aggregate_risk_shape_matches_resolution() -> None:
    setup, optical, thermal = _smoke_optical_thermal()
    selections = select_legacy_worst_conditions(
        optical, thermal, max_conditions=2,
    )
    res = (16, 32)
    result = build_multi_condition_legacy_hotspot_contribution_map(
        setup=setup,
        conditions=selections,
        source_width=80.0,
        source_height=240.0,
        source_radius=200.0,
        sample_count_y=5,
        sample_count_z=5,
        detector_size=400.0,
        detector_resolution=(20, 20),
        epsilon=0.1,
        hotspot_top_percent=20.0,
        contribution_resolution=res,
    )
    assert result.aggregate_risk_map.risk_map.shape == res
    assert result.aggregate_risk_map.probability_map.shape == res
    assert result.aggregate_risk_map.active_mask.shape == res


def test_build_aggregate_nonzero_bins_nonneg() -> None:
    setup, optical, thermal = _smoke_optical_thermal()
    selections = select_legacy_worst_conditions(
        optical, thermal, max_conditions=2,
    )
    result = build_multi_condition_legacy_hotspot_contribution_map(
        setup=setup,
        conditions=selections,
        source_width=80.0,
        source_height=240.0,
        source_radius=200.0,
        sample_count_y=5,
        sample_count_z=5,
        detector_size=400.0,
        detector_resolution=(20, 20),
        epsilon=0.1,
        hotspot_top_percent=20.0,
        contribution_resolution=(16, 32),
    )
    assert int(result.aggregate_nonzero_bins) >= 0


def test_build_no_file_output(tmp_path) -> None:
    setup, optical, thermal = _smoke_optical_thermal()
    selections = select_legacy_worst_conditions(
        optical, thermal, max_conditions=2,
    )
    before = sorted(tmp_path.iterdir())
    build_multi_condition_legacy_hotspot_contribution_map(
        setup=setup,
        conditions=selections,
        source_width=80.0,
        source_height=240.0,
        source_radius=200.0,
        sample_count_y=5,
        sample_count_z=5,
        detector_size=400.0,
        detector_resolution=(20, 20),
        epsilon=0.1,
        hotspot_top_percent=20.0,
        contribution_resolution=(16, 32),
    )
    after = sorted(tmp_path.iterdir())
    assert before == after


def test_build_invalid_setup_raises() -> None:
    with pytest.raises(OpticsError, match="LegacyPetWaterTraceSetup"):
        build_multi_condition_legacy_hotspot_contribution_map(
            setup="not a setup",  # type: ignore[arg-type]
            conditions=(),
            source_width=80.0,
            source_height=240.0,
            source_radius=200.0,
            sample_count_y=5,
            sample_count_z=5,
            detector_size=400.0,
            detector_resolution=(20, 20),
        )


def test_build_empty_conditions_returns_zero_aggregate() -> None:
    setup = _setup()
    res = (16, 32)
    result = build_multi_condition_legacy_hotspot_contribution_map(
        setup=setup,
        conditions=(),
        source_width=80.0,
        source_height=240.0,
        source_radius=200.0,
        sample_count_y=5,
        sample_count_z=5,
        detector_size=400.0,
        detector_resolution=(20, 20),
        epsilon=0.1,
        hotspot_top_percent=20.0,
        contribution_resolution=res,
    )
    assert int(result.condition_count) == 0
    assert result.aggregate_contribution_map.weight_map.shape == res
    assert int(result.aggregate_nonzero_bins) == 0
