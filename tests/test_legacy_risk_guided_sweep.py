import numpy as np
import pytest

from optics_simulation.contribution import (
    build_multi_condition_legacy_hotspot_contribution_map,
    select_legacy_worst_conditions,
)
from optics_simulation.geometry import (
    create_actual_bottle_body_vertex_mask,
    create_subdivided_synthetic_bottle_body,
)
from optics_simulation.optics import (
    OpticsError,
    RiskGuidedPatternCandidateSpec,
    RiskGuidedPatternSweepEntry,
    RiskGuidedPatternSweepResult,
    create_legacy_pet_water_trace_setup,
    run_actual_stl_risk_guided_pattern_parameter_sweep,
    run_legacy_pet_water_angle_distance_scan,
)
from optics_simulation.thermal import (
    compute_thermal_risk_over_legacy_optical_scan,
)


def _smoke_fixture():
    mesh = create_subdivided_synthetic_bottle_body(
        radius=30.0, height=120.0,
        sections=32, height_segments=8,
    )
    setup = create_legacy_pet_water_trace_setup(
        mesh,
        target_height=225.6, target_diameter=72.1,
        wall_thickness=0.3, inner_offset_mode="auto",
    )
    angles = (0.0, 15.0)
    distances = (100.0, 140.0)
    optical = run_legacy_pet_water_angle_distance_scan(
        setup=setup,
        angles_degrees=angles,
        detector_distances=distances,
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
    selections = select_legacy_worst_conditions(
        optical, thermal, max_conditions=2,
    )
    multi = build_multi_condition_legacy_hotspot_contribution_map(
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
    body_mask = create_actual_bottle_body_vertex_mask(
        setup.shell_mesh,
    )
    return setup, thermal, multi, body_mask, angles, distances


def _candidates() -> tuple[RiskGuidedPatternCandidateSpec, ...]:
    return (
        RiskGuidedPatternCandidateSpec(
            name="cand_a",
            pattern_count=5, sigma_u=0.05, sigma_v=0.05,
            max_depth=0.02, seed=1,
        ),
        RiskGuidedPatternCandidateSpec(
            name="cand_b",
            pattern_count=5, sigma_u=0.03, sigma_v=0.03,
            max_depth=0.05, seed=2,
        ),
    )


def _run_sweep(
    setup, thermal, multi, body_mask, angles, distances,
    candidates,
):
    return run_actual_stl_risk_guided_pattern_parameter_sweep(
        original_setup=setup,
        baseline_thermal_scan=thermal,
        aggregate_risk_map=multi.aggregate_risk_map,
        body_include_mask=body_mask.include_mask,
        candidates=candidates,
        angles_degrees=angles,
        detector_distances=distances,
        source_width=80.0,
        source_height=240.0,
        source_radius=200.0,
        sample_count_y=5,
        sample_count_z=5,
        detector_size=400.0,
        detector_resolution=(20, 20),
        duration_s=5.0,
        dt_s=1.0,
    )


def test_candidate_spec_fields_accessible() -> None:
    cand = RiskGuidedPatternCandidateSpec(
        name="x",
        pattern_count=10, sigma_u=0.04, sigma_v=0.04,
        max_depth=0.05, seed=7, amplitude=0.8,
    )
    assert cand.name == "x"
    assert int(cand.pattern_count) == 10
    assert float(cand.sigma_u) == 0.04
    assert float(cand.sigma_v) == 0.04
    assert float(cand.max_depth) == 0.05
    assert int(cand.seed) == 7
    assert float(cand.amplitude) == 0.8


def test_sweep_returns_result() -> None:
    fixture = _smoke_fixture()
    sweep = _run_sweep(*fixture, _candidates())
    assert isinstance(sweep, RiskGuidedPatternSweepResult)
    for e in sweep.entries:
        assert isinstance(e, RiskGuidedPatternSweepEntry)


def test_sweep_candidate_count_matches_input() -> None:
    fixture = _smoke_fixture()
    cands = _candidates()
    sweep = _run_sweep(*fixture, cands)
    assert int(sweep.candidate_count) == len(cands)
    assert len(sweep.entries) == len(cands)


def test_sweep_entries_have_nonneg_moved_count() -> None:
    fixture = _smoke_fixture()
    sweep = _run_sweep(*fixture, _candidates())
    for e in sweep.entries:
        assert int(e.moved_vertex_count) >= 0


def test_sweep_entries_preserve_candidate_order() -> None:
    fixture = _smoke_fixture()
    cands = _candidates()
    sweep = _run_sweep(*fixture, cands)
    names = [e.candidate.name for e in sweep.entries]
    assert names == [c.name for c in cands]


def test_sweep_best_by_max_temperature_populated() -> None:
    fixture = _smoke_fixture()
    sweep = _run_sweep(*fixture, _candidates())
    assert sweep.best_candidate_by_max_temperature_delta is not None
    assert sweep.best_delta_max_temperature_k is not None


def test_sweep_best_by_threshold_count_populated() -> None:
    fixture = _smoke_fixture()
    sweep = _run_sweep(*fixture, _candidates())
    assert (
        sweep.best_candidate_by_threshold_count_delta is not None
    )
    assert sweep.best_delta_threshold_count is not None


def test_sweep_empty_candidates_raises() -> None:
    fixture = _smoke_fixture()
    setup, thermal, multi, body_mask, angles, distances = fixture
    with pytest.raises(OpticsError, match="candidates"):
        run_actual_stl_risk_guided_pattern_parameter_sweep(
            original_setup=setup,
            baseline_thermal_scan=thermal,
            aggregate_risk_map=multi.aggregate_risk_map,
            body_include_mask=body_mask.include_mask,
            candidates=(),
            angles_degrees=angles,
            detector_distances=distances,
            source_width=80.0,
            source_height=240.0,
            source_radius=200.0,
            sample_count_y=5,
            sample_count_z=5,
            detector_size=400.0,
            detector_resolution=(20, 20),
            duration_s=5.0,
            dt_s=1.0,
        )


def test_sweep_no_file_output(tmp_path) -> None:
    fixture = _smoke_fixture()
    before = sorted(tmp_path.iterdir())
    _run_sweep(*fixture, _candidates())
    after = sorted(tmp_path.iterdir())
    assert before == after


def test_sweep_invalid_candidate_spec_raises() -> None:
    fixture = _smoke_fixture()
    setup, thermal, multi, body_mask, angles, distances = fixture
    bad = (
        RiskGuidedPatternCandidateSpec(
            name="bad",
            pattern_count=0,  # invalid
            sigma_u=0.04, sigma_v=0.04,
            max_depth=0.05, seed=1,
        ),
    )
    with pytest.raises(OpticsError, match="pattern_count"):
        run_actual_stl_risk_guided_pattern_parameter_sweep(
            original_setup=setup,
            baseline_thermal_scan=thermal,
            aggregate_risk_map=multi.aggregate_risk_map,
            body_include_mask=body_mask.include_mask,
            candidates=bad,
            angles_degrees=angles,
            detector_distances=distances,
            source_width=80.0,
            source_height=240.0,
            source_radius=200.0,
            sample_count_y=5,
            sample_count_z=5,
            detector_size=400.0,
            detector_resolution=(20, 20),
            duration_s=5.0,
            dt_s=1.0,
        )


def test_sweep_duplicate_candidate_names_raise() -> None:
    fixture = _smoke_fixture()
    setup, thermal, multi, body_mask, angles, distances = fixture
    dups = (
        RiskGuidedPatternCandidateSpec(
            name="dup",
            pattern_count=5, sigma_u=0.04, sigma_v=0.04,
            max_depth=0.02, seed=1,
        ),
        RiskGuidedPatternCandidateSpec(
            name="dup",
            pattern_count=5, sigma_u=0.05, sigma_v=0.05,
            max_depth=0.05, seed=2,
        ),
    )
    with pytest.raises(OpticsError, match="duplicated"):
        run_actual_stl_risk_guided_pattern_parameter_sweep(
            original_setup=setup,
            baseline_thermal_scan=thermal,
            aggregate_risk_map=multi.aggregate_risk_map,
            body_include_mask=body_mask.include_mask,
            candidates=dups,
            angles_degrees=angles,
            detector_distances=distances,
            source_width=80.0,
            source_height=240.0,
            source_radius=200.0,
            sample_count_y=5,
            sample_count_z=5,
            detector_size=400.0,
            detector_resolution=(20, 20),
            duration_s=5.0,
            dt_s=1.0,
        )
