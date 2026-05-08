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
    RingOffsetSweepCandidateSpec,
    RingOffsetSweepEntry,
    RingOffsetSweepResult,
    create_legacy_pet_water_trace_setup,
    rank_ring_offset_sweep_by_composite_score,
    rank_ring_offset_sweep_pareto_indices,
    run_actual_stl_ring_offset_sweep,
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


def _candidates() -> tuple[RingOffsetSweepCandidateSpec, ...]:
    return (
        RingOffsetSweepCandidateSpec(
            name="ring_1_3", inner_radius_px=1, outer_radius_px=3,
        ),
        RingOffsetSweepCandidateSpec(
            name="ring_2_5", inner_radius_px=2, outer_radius_px=5,
        ),
    )


def _run_sweep(setup, thermal, multi, body_mask, angles, distances,
               candidates):
    return run_actual_stl_ring_offset_sweep(
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
        pattern_count=5,
        pattern_sigma_u=0.05, pattern_sigma_v=0.05,
        pattern_max_depth=0.05,
        pattern_seed=1,
    )


def test_candidate_spec_fields_accessible() -> None:
    spec = RingOffsetSweepCandidateSpec(
        name="x", inner_radius_px=1, outer_radius_px=3,
        risk_epsilon=0.02,
    )
    assert spec.name == "x"
    assert int(spec.inner_radius_px) == 1
    assert int(spec.outer_radius_px) == 3
    assert float(spec.risk_epsilon) == 0.02


def test_sweep_returns_result() -> None:
    fixture = _smoke_fixture()
    sweep = _run_sweep(*fixture, _candidates())
    assert isinstance(sweep, RingOffsetSweepResult)
    for e in sweep.entries:
        assert isinstance(e, RingOffsetSweepEntry)


def test_sweep_candidate_count_matches_input() -> None:
    fixture = _smoke_fixture()
    cands = _candidates()
    sweep = _run_sweep(*fixture, cands)
    assert int(sweep.candidate_count) == len(cands)
    assert len(sweep.entries) == len(cands)
    assert [e.candidate.name for e in sweep.entries] == [
        c.name for c in cands
    ]


def test_sweep_entries_have_consistent_pass_count() -> None:
    fixture = _smoke_fixture()
    sweep = _run_sweep(*fixture, _candidates())
    for e in sweep.entries:
        assert (
            int(e.guardrail_pass_count)
            + int(e.guardrail_fail_count)
            == int(e.total_condition_count)
        )
        if int(e.total_condition_count) > 0:
            expected_ratio = (
                float(e.guardrail_pass_count)
                / float(e.total_condition_count)
            )
            assert float(e.pass_ratio) == pytest.approx(
                expected_ratio,
            )


def test_sweep_pareto_names_subset_of_entries() -> None:
    fixture = _smoke_fixture()
    sweep = _run_sweep(*fixture, _candidates())
    entry_names = {e.candidate.name for e in sweep.entries}
    assert set(sweep.pareto_candidate_names).issubset(entry_names)
    assert len(sweep.pareto_candidate_names) > 0


def test_sweep_best_pointers_populated() -> None:
    fixture = _smoke_fixture()
    sweep = _run_sweep(*fixture, _candidates())
    assert sweep.best_by_max_temperature_delta is not None
    assert sweep.best_delta_max_temperature_k is not None
    assert sweep.best_by_threshold_count_delta is not None
    assert sweep.best_delta_threshold_count is not None
    assert sweep.best_by_pass_ratio is not None
    assert sweep.best_pass_ratio is not None
    assert sweep.best_by_composite_score is not None
    assert sweep.best_composite_score is not None


def test_sweep_invalid_candidate_outer_le_inner_raises() -> None:
    fixture = _smoke_fixture()
    setup, thermal, multi, body_mask, angles, distances = fixture
    bad = (
        RingOffsetSweepCandidateSpec(
            name="bad", inner_radius_px=3, outer_radius_px=3,
        ),
    )
    with pytest.raises(OpticsError, match="outer_radius_px"):
        _run_sweep(setup, thermal, multi, body_mask, angles,
                   distances, bad)


def test_sweep_invalid_candidate_negative_inner_raises() -> None:
    fixture = _smoke_fixture()
    setup, thermal, multi, body_mask, angles, distances = fixture
    bad = (
        RingOffsetSweepCandidateSpec(
            name="neg", inner_radius_px=-1, outer_radius_px=3,
        ),
    )
    with pytest.raises(OpticsError, match="inner_radius_px"):
        _run_sweep(setup, thermal, multi, body_mask, angles,
                   distances, bad)


def test_sweep_duplicate_candidate_names_raise() -> None:
    fixture = _smoke_fixture()
    setup, thermal, multi, body_mask, angles, distances = fixture
    dups = (
        RingOffsetSweepCandidateSpec(
            name="d", inner_radius_px=1, outer_radius_px=3,
        ),
        RingOffsetSweepCandidateSpec(
            name="d", inner_radius_px=2, outer_radius_px=5,
        ),
    )
    with pytest.raises(OpticsError, match="duplicated"):
        _run_sweep(setup, thermal, multi, body_mask, angles,
                   distances, dups)


def test_sweep_empty_candidates_raises() -> None:
    fixture = _smoke_fixture()
    setup, thermal, multi, body_mask, angles, distances = fixture
    with pytest.raises(OpticsError, match="candidates"):
        _run_sweep(setup, thermal, multi, body_mask, angles,
                   distances, ())


def test_sweep_no_file_output(tmp_path) -> None:
    fixture = _smoke_fixture()
    before = sorted(tmp_path.iterdir())
    _run_sweep(*fixture, _candidates())
    after = sorted(tmp_path.iterdir())
    assert before == after


# ---------------------------------------------------------------
# Ranking-only tests (no scan: hand-constructed entries)
# ---------------------------------------------------------------


def _entry(
    name: str, *,
    delta_t: float, delta_count: int, pass_ratio: float,
    new_bin_fraction: float = 0.0,
) -> RingOffsetSweepEntry:
    spec = RingOffsetSweepCandidateSpec(
        name=name, inner_radius_px=1, outer_radius_px=3,
    )
    total = 100
    pass_count = int(round(pass_ratio * total))
    return RingOffsetSweepEntry(
        candidate=spec,
        source_risk_active_count=10,
        ring_risk_active_count=20,
        moved_vertex_count=10,
        max_displacement=0.05,
        delta_max_temperature_k=float(delta_t),
        delta_threshold_count=int(delta_count),
        candidate_only_bins=int(new_bin_fraction * 100),
        new_bin_fraction=float(new_bin_fraction),
        guardrail_pass_count=pass_count,
        guardrail_fail_count=total - pass_count,
        total_condition_count=total,
        pass_ratio=float(pass_count) / float(total),
        tradeoff_label="mixed",
    )


def test_pareto_filter_excludes_dominated_candidate() -> None:
    # B dominates A on all three axes (lower Δ, lower count,
    # higher pass_ratio); A should be filtered out.
    a = _entry("A", delta_t=10.0, delta_count=2, pass_ratio=0.3)
    b = _entry("B", delta_t=5.0, delta_count=0, pass_ratio=0.7)
    indices = rank_ring_offset_sweep_pareto_indices([a, b])
    assert indices == (1,)


def test_pareto_filter_keeps_tradeoff_pair() -> None:
    # A is best on Δ_max_temp, B is best on pass_ratio. Neither
    # dominates the other.
    a = _entry("A", delta_t=1.0, delta_count=0, pass_ratio=0.3)
    b = _entry("B", delta_t=10.0, delta_count=0, pass_ratio=0.9)
    indices = rank_ring_offset_sweep_pareto_indices([a, b])
    assert set(indices) == {0, 1}


def test_pareto_filter_empty_input() -> None:
    assert rank_ring_offset_sweep_pareto_indices([]) == ()


def test_composite_score_orders_lower_better() -> None:
    a = _entry("A", delta_t=10.0, delta_count=2, pass_ratio=0.3)
    b = _entry("B", delta_t=5.0, delta_count=0, pass_ratio=0.7)
    c = _entry("C", delta_t=1.0, delta_count=-1, pass_ratio=0.9)
    order = rank_ring_offset_sweep_by_composite_score([a, b, c])
    # C is best on every axis -> first.
    assert order[0] == 2
    # A is worst on every axis -> last.
    assert order[-1] == 0


def test_composite_score_invalid_weight_raises() -> None:
    a = _entry("A", delta_t=1.0, delta_count=0, pass_ratio=0.5)
    with pytest.raises(OpticsError, match="composite weights"):
        rank_ring_offset_sweep_by_composite_score(
            [a], weight_max_temperature=-1.0,
        )


def test_composite_score_empty_input() -> None:
    assert rank_ring_offset_sweep_by_composite_score([]) == ()
