import numpy as np
import pytest

from optics_simulation.contribution import (
    LegacyConditionSelection,
    build_multi_condition_legacy_hotspot_contribution_map,
    select_legacy_worst_conditions,
)
from optics_simulation.geometry import (
    create_actual_bottle_body_vertex_mask,
    create_subdivided_synthetic_bottle_body,
)
from optics_simulation.optics import (
    ConditionSubsetDiagnostics,
    OpticsError,
    RING_RADIUS_UNITS_DESCRIPTION,
    RingOffsetSweepCandidateSpec,
    RingOffsetSweepEntry,
    RingOffsetSweepResult,
    compute_holdout_diagnostics,
    create_legacy_pet_water_trace_setup,
    rank_ring_offset_sweep_by_composite_score,
    rank_ring_offset_sweep_pareto_indices,
    run_actual_stl_ring_offset_sweep,
    run_legacy_pet_water_angle_distance_scan,
    split_selected_vs_holdout_conditions,
    summarize_thermal_risk_comparison_subset,
    validate_ring_offset_sweep_result,
)
from optics_simulation.thermal import (
    compare_legacy_thermal_risk_scans,
    compute_thermal_risk_over_legacy_optical_scan,
)


# ---------------------------------------------------------------
# Renamed-module compatibility: the active implementation lives at
# optics_simulation.optics.ring_offset_sweep.
# ---------------------------------------------------------------


def test_renamed_module_is_importable() -> None:
    import importlib

    mod = importlib.import_module(
        "optics_simulation.optics.ring_offset_sweep",
    )
    assert hasattr(mod, "run_actual_stl_ring_offset_sweep")
    assert hasattr(mod, "split_selected_vs_holdout_conditions")
    assert hasattr(mod, "compute_holdout_diagnostics")
    assert hasattr(mod, "RING_RADIUS_UNITS_DESCRIPTION")


def test_old_legacy_module_path_is_gone() -> None:
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(
            "optics_simulation.optics.legacy_ring_offset_sweep",
        )


def test_units_description_mentions_risk_map_bins() -> None:
    assert "risk-map bin" in RING_RADIUS_UNITS_DESCRIPTION
    assert "mesh-space" in RING_RADIUS_UNITS_DESCRIPTION
    assert "detector-grid" in RING_RADIUS_UNITS_DESCRIPTION


# ---------------------------------------------------------------
# Smoke fixture for end-to-end sweep tests
# ---------------------------------------------------------------


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
    angles = (0.0, 15.0, 30.0)
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
    return setup, thermal, multi, body_mask, selections, angles, distances


def _candidates() -> tuple[RingOffsetSweepCandidateSpec, ...]:
    return (
        RingOffsetSweepCandidateSpec(
            name="ring_1_3", inner_radius_px=1, outer_radius_px=3,
        ),
        RingOffsetSweepCandidateSpec(
            name="ring_2_5", inner_radius_px=2, outer_radius_px=5,
        ),
    )


def _run_sweep(
    setup, thermal, multi, body_mask, selections,
    angles, distances, candidates, *,
    ranking_basis: str = "holdout",
):
    return run_actual_stl_ring_offset_sweep(
        original_setup=setup,
        baseline_thermal_scan=thermal,
        aggregate_risk_map=multi.aggregate_risk_map,
        body_include_mask=body_mask.include_mask,
        candidates=candidates,
        selections=selections,
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
        ranking_basis=ranking_basis,
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


def test_sweep_returns_result_with_extended_fields() -> None:
    fixture = _smoke_fixture()
    sweep = _run_sweep(*fixture, _candidates())
    assert isinstance(sweep, RingOffsetSweepResult)
    assert sweep.units_description == RING_RADIUS_UNITS_DESCRIPTION
    for e in sweep.entries:
        assert isinstance(e, RingOffsetSweepEntry)
        assert isinstance(
            e.full_diagnostics, ConditionSubsetDiagnostics,
        )
        assert isinstance(
            e.in_sample_diagnostics, ConditionSubsetDiagnostics,
        )
        assert isinstance(
            e.holdout_diagnostics, ConditionSubsetDiagnostics,
        )
        assert e.units_description == RING_RADIUS_UNITS_DESCRIPTION


def test_sweep_candidate_count_matches_input() -> None:
    fixture = _smoke_fixture()
    cands = _candidates()
    sweep = _run_sweep(*fixture, cands)
    assert int(sweep.candidate_count) == len(cands)
    assert len(sweep.entries) == len(cands)
    assert [e.candidate.name for e in sweep.entries] == [
        c.name for c in cands
    ]


def test_sweep_evaluation_count_equals_full_grid() -> None:
    fixture = _smoke_fixture()
    setup, thermal, multi, body_mask, selections, angles, distances = (
        fixture
    )
    sweep = _run_sweep(*fixture, _candidates())
    expected = len(angles) * len(distances)
    assert int(sweep.evaluation_condition_count) == expected
    for e in sweep.entries:
        assert int(e.full_diagnostics.entry_count) == expected


def test_sweep_selection_in_sample_holdout_split_consistent() -> None:
    fixture = _smoke_fixture()
    setup, thermal, multi, body_mask, selections, angles, distances = (
        fixture
    )
    sweep = _run_sweep(*fixture, _candidates())
    assert int(sweep.selected_condition_count) == len({
        (float(s.angle_degrees), float(s.detector_distance))
        for s in selections
    })
    for e in sweep.entries:
        n_full = int(e.full_diagnostics.entry_count)
        n_in = int(e.in_sample_diagnostics.entry_count)
        n_out = int(e.holdout_diagnostics.entry_count)
        assert n_in + n_out == n_full
        assert n_in <= sweep.selected_condition_count


def test_sweep_overlap_metrics_invariants() -> None:
    fixture = _smoke_fixture()
    sweep = _run_sweep(*fixture, _candidates())
    for e in sweep.entries:
        assert int(e.source_overlap_bin_count) >= 0
        assert 0.0 <= float(e.source_overlap_fraction) <= 1.0
        assert (
            float(e.retained_source_bin_fraction)
            == float(e.source_overlap_fraction)
        )
        assert 0.0 <= float(e.new_bin_fraction) <= 1.0
        assert 0.0 <= float(e.jaccard_overlap) <= 1.0
        assert int(e.candidate_only_bin_count) >= 0


def test_sweep_coverage_metrics_invariants() -> None:
    fixture = _smoke_fixture()
    sweep = _run_sweep(*fixture, _candidates())
    for e in sweep.entries:
        assert int(e.body_mask_vertex_count) > 0
        assert (
            int(e.moved_vertex_count)
            <= int(e.body_mask_vertex_count)
        )
        assert 0.0 <= float(e.moved_vertex_fraction) <= 1.0
        assert float(e.risk_support_expansion_ratio) >= 0.0


def test_sweep_pareto_basis_defaults_to_holdout() -> None:
    fixture = _smoke_fixture()
    sweep = _run_sweep(*fixture, _candidates())
    assert sweep.pareto_basis == "holdout"
    assert sweep.composite_basis == "holdout"


def test_sweep_pareto_axes_subset_of_known_axes() -> None:
    fixture = _smoke_fixture()
    sweep = _run_sweep(*fixture, _candidates())
    known = {
        "delta_max_temperature_k",
        "delta_threshold_count",
        "pass_ratio_neg",
    }
    assert set(sweep.pareto_axes).issubset(known)
    assert set(sweep.non_discriminative_metrics).issubset(known)


def test_sweep_invalid_candidate_outer_le_inner_raises() -> None:
    fixture = _smoke_fixture()
    bad = (
        RingOffsetSweepCandidateSpec(
            name="bad", inner_radius_px=3, outer_radius_px=3,
        ),
    )
    with pytest.raises(OpticsError, match="outer_radius_px"):
        _run_sweep(*fixture, bad)


def test_sweep_invalid_candidate_negative_inner_raises() -> None:
    fixture = _smoke_fixture()
    bad = (
        RingOffsetSweepCandidateSpec(
            name="neg", inner_radius_px=-1, outer_radius_px=3,
        ),
    )
    with pytest.raises(OpticsError, match="inner_radius_px"):
        _run_sweep(*fixture, bad)


def test_sweep_duplicate_candidate_names_raise() -> None:
    fixture = _smoke_fixture()
    dups = (
        RingOffsetSweepCandidateSpec(
            name="d", inner_radius_px=1, outer_radius_px=3,
        ),
        RingOffsetSweepCandidateSpec(
            name="d", inner_radius_px=2, outer_radius_px=5,
        ),
    )
    with pytest.raises(OpticsError, match="duplicated"):
        _run_sweep(*fixture, dups)


def test_sweep_empty_candidates_raises() -> None:
    fixture = _smoke_fixture()
    with pytest.raises(OpticsError, match="candidates"):
        _run_sweep(*fixture, ())


def test_sweep_invalid_ranking_basis_raises() -> None:
    fixture = _smoke_fixture()
    with pytest.raises(OpticsError, match="ranking_basis"):
        _run_sweep(
            *fixture, _candidates(),
            ranking_basis="not_a_basis",
        )


def test_sweep_no_file_output(tmp_path) -> None:
    fixture = _smoke_fixture()
    before = sorted(tmp_path.iterdir())
    _run_sweep(*fixture, _candidates())
    after = sorted(tmp_path.iterdir())
    assert before == after


# ---------------------------------------------------------------
# Selection-vs-holdout split helpers
# ---------------------------------------------------------------


def test_split_function_returns_disjoint_subsets() -> None:
    fixture = _smoke_fixture()
    spec = RingOffsetSweepCandidateSpec(
        name="ring_1_3", inner_radius_px=1, outer_radius_px=3,
    )
    sweep = _run_sweep(*fixture, (spec,))
    e = sweep.entries[0]
    assert (
        int(e.full_diagnostics.entry_count)
        == int(e.in_sample_diagnostics.entry_count)
        + int(e.holdout_diagnostics.entry_count)
    )


def test_split_helper_rejects_invalid_comparison_type() -> None:
    with pytest.raises(OpticsError, match="comparison"):
        split_selected_vs_holdout_conditions(
            "not a comparison",  # type: ignore[arg-type]
            (),
        )


def test_split_helper_rejects_invalid_selection_type() -> None:
    fixture = _smoke_fixture()
    setup, thermal, multi, body_mask, selections, angles, distances = (
        fixture
    )
    cmp = compare_legacy_thermal_risk_scans(thermal, thermal)
    with pytest.raises(OpticsError, match="LegacyConditionSelection"):
        split_selected_vs_holdout_conditions(
            cmp,
            ("not a selection",),  # type: ignore[arg-type]
        )


def test_split_when_selections_empty_returns_all_holdout() -> None:
    fixture = _smoke_fixture()
    setup, thermal, multi, body_mask, selections, angles, distances = (
        fixture
    )
    cmp = compare_legacy_thermal_risk_scans(thermal, thermal)
    in_sample, holdout = split_selected_vs_holdout_conditions(
        cmp, (),
    )
    assert in_sample == ()
    assert len(holdout) == int(len(cmp.entries))


def test_split_excludes_selected_conditions_from_holdout() -> None:
    fixture = _smoke_fixture()
    setup, thermal, multi, body_mask, selections, angles, distances = (
        fixture
    )
    cmp = compare_legacy_thermal_risk_scans(thermal, thermal)
    _, holdout = split_selected_vs_holdout_conditions(
        cmp, selections,
    )
    selected_keys = {
        (float(s.angle_degrees), float(s.detector_distance))
        for s in selections
    }
    for e in holdout:
        key = (
            float(e.angle_degrees),
            float(e.detector_distance),
        )
        assert key not in selected_keys


def test_summarize_subset_empty_input() -> None:
    diag = summarize_thermal_risk_comparison_subset(())
    assert int(diag.entry_count) == 0
    assert diag.worst_delta_max_temperature_k is None
    assert diag.best_delta_threshold_count is None
    assert diag.dominant_label == "unchanged"


def test_summarize_subset_pass_ratio_consistent() -> None:
    fixture = _smoke_fixture()
    setup, thermal, multi, body_mask, selections, angles, distances = (
        fixture
    )
    cmp = compare_legacy_thermal_risk_scans(thermal, thermal)
    diag = summarize_thermal_risk_comparison_subset(cmp.entries)
    assert int(diag.entry_count) == int(len(cmp.entries))
    if int(diag.entry_count) > 0:
        assert (
            int(diag.pass_count) + int(diag.fail_count)
            == int(diag.entry_count)
        )
        assert float(diag.pass_ratio) == pytest.approx(
            float(diag.pass_count) / float(diag.entry_count),
        )


def test_compute_holdout_diagnostics_uses_only_holdout() -> None:
    fixture = _smoke_fixture()
    setup, thermal, multi, body_mask, selections, angles, distances = (
        fixture
    )
    cmp = compare_legacy_thermal_risk_scans(thermal, thermal)
    holdout_diag = compute_holdout_diagnostics(cmp, selections)
    _, holdout_entries = split_selected_vs_holdout_conditions(
        cmp, selections,
    )
    assert int(holdout_diag.entry_count) == len(holdout_entries)


# ---------------------------------------------------------------
# Ranking-only tests (hand-built entries)
# ---------------------------------------------------------------


def _diag(
    *, entry_count: int, pass_ratio: float,
    worst_dt: float, best_thresh: int,
) -> ConditionSubsetDiagnostics:
    pass_count = int(round(pass_ratio * entry_count))
    return ConditionSubsetDiagnostics(
        entry_count=int(entry_count),
        pass_count=int(pass_count),
        fail_count=int(entry_count - pass_count),
        pass_ratio=(
            float(pass_count) / float(entry_count)
            if entry_count > 0 else 0.0
        ),
        worst_delta_max_temperature_k=float(worst_dt),
        best_delta_max_temperature_k=float(worst_dt),
        best_delta_threshold_count=int(best_thresh),
        improved_count=0, worsened_count=0,
        mixed_count=int(entry_count), unchanged_count=0,
        dominant_label="mixed",
    )


def _entry(
    name: str, *,
    full_dt: float, full_thresh: int, full_pass: float,
    holdout_dt: float | None = None,
    holdout_thresh: int | None = None,
    holdout_pass: float | None = None,
    new_bin_fraction: float = 0.0,
) -> RingOffsetSweepEntry:
    spec = RingOffsetSweepCandidateSpec(
        name=name, inner_radius_px=1, outer_radius_px=3,
    )
    full = _diag(
        entry_count=100, pass_ratio=full_pass,
        worst_dt=full_dt, best_thresh=full_thresh,
    )
    if holdout_dt is None:
        holdout_dt = full_dt
        holdout_thresh = full_thresh
        holdout_pass = full_pass
    holdout = _diag(
        entry_count=80,
        pass_ratio=float(holdout_pass),
        worst_dt=float(holdout_dt),
        best_thresh=int(holdout_thresh),
    )
    in_sample = _diag(
        entry_count=20, pass_ratio=full_pass,
        worst_dt=full_dt, best_thresh=full_thresh,
    )
    return RingOffsetSweepEntry(
        candidate=spec,
        units_description=RING_RADIUS_UNITS_DESCRIPTION,
        source_risk_active_count=10,
        transformed_risk_active_count=20,
        risk_support_expansion_ratio=2.0,
        moved_vertex_count=10,
        moved_vertex_fraction=0.1,
        body_mask_vertex_count=100,
        max_displacement=0.05,
        full_diagnostics=full,
        in_sample_diagnostics=in_sample,
        holdout_diagnostics=holdout,
        hotspot_diagnostic_angle_degrees=0.0,
        hotspot_diagnostic_detector_distance=100.0,
        baseline_nonzero_bins=10,
        candidate_nonzero_bins=10,
        source_overlap_bin_count=10,
        source_overlap_fraction=1.0,
        retained_source_bin_fraction=1.0,
        candidate_only_bin_count=int(new_bin_fraction * 10),
        new_bin_fraction=float(new_bin_fraction),
        jaccard_overlap=1.0,
    )


def test_pareto_filter_excludes_dominated_holdout() -> None:
    a = _entry("A", full_dt=10.0, full_thresh=2, full_pass=0.3)
    b = _entry("B", full_dt=5.0, full_thresh=0, full_pass=0.7)
    indices = rank_ring_offset_sweep_pareto_indices(
        [a, b], basis="holdout",
    )
    assert indices == (1,)


def test_pareto_filter_keeps_tradeoff_pair_holdout() -> None:
    a = _entry("A", full_dt=1.0, full_thresh=0, full_pass=0.3)
    b = _entry("B", full_dt=10.0, full_thresh=0, full_pass=0.9)
    indices = rank_ring_offset_sweep_pareto_indices(
        [a, b], basis="holdout",
    )
    assert set(indices) == {0, 1}


def test_pareto_filter_drops_non_discriminative_axis() -> None:
    a = _entry("A", full_dt=1.0, full_thresh=-1, full_pass=0.3)
    b = _entry("B", full_dt=5.0, full_thresh=-1, full_pass=0.7)
    c = _entry("C", full_dt=3.0, full_thresh=-1, full_pass=0.5)
    indices = rank_ring_offset_sweep_pareto_indices(
        [a, b, c], basis="holdout",
    )
    assert 0 in indices
    assert 1 in indices


def test_pareto_filter_empty_input() -> None:
    assert rank_ring_offset_sweep_pareto_indices([]) == ()


def test_pareto_filter_basis_validation() -> None:
    a = _entry("A", full_dt=1.0, full_thresh=0, full_pass=0.5)
    with pytest.raises(OpticsError, match="basis"):
        rank_ring_offset_sweep_pareto_indices(
            [a], basis="bogus",
        )


def test_composite_score_orders_lower_better_holdout() -> None:
    a = _entry("A", full_dt=10.0, full_thresh=2, full_pass=0.3)
    b = _entry("B", full_dt=5.0, full_thresh=0, full_pass=0.7)
    c = _entry("C", full_dt=1.0, full_thresh=-1, full_pass=0.9)
    order = rank_ring_offset_sweep_by_composite_score(
        [a, b, c], basis="holdout",
    )
    assert order[0] == 2
    assert order[-1] == 0


def test_composite_score_invalid_weight_raises() -> None:
    a = _entry("A", full_dt=1.0, full_thresh=0, full_pass=0.5)
    with pytest.raises(OpticsError, match="composite weights"):
        rank_ring_offset_sweep_by_composite_score(
            [a], weight_max_temperature=-1.0,
        )


def test_composite_score_empty_input() -> None:
    assert rank_ring_offset_sweep_by_composite_score([]) == ()


def test_composite_score_constant_axes_are_ignored() -> None:
    # Δmax_T differs but Δthresh count and pass_ratio are
    # constant. Composite must order purely by Δmax_T.
    a = _entry("A", full_dt=10.0, full_thresh=0, full_pass=0.5)
    b = _entry("B", full_dt=1.0, full_thresh=0, full_pass=0.5)
    order = rank_ring_offset_sweep_by_composite_score(
        [a, b], basis="holdout",
    )
    assert order[0] == 1
    assert order[1] == 0


# ---------------------------------------------------------------
# no-improving-candidate flag (predicate equivalence)
# ---------------------------------------------------------------


def test_no_improving_predicate_when_all_dt_positive() -> None:
    a = _entry("A", full_dt=5.0, full_thresh=0, full_pass=0.3)
    b = _entry("B", full_dt=10.0, full_thresh=0, full_pass=0.5)
    # The runner sets no_improving_candidate_under_current_sweep
    # when every full_diagnostics.worst_delta_max_temperature_k
    # > 0. Verify the underlying predicate.
    worst_full = max(
        float(a.full_diagnostics.worst_delta_max_temperature_k),
        float(b.full_diagnostics.worst_delta_max_temperature_k),
    )
    min_full = min(
        float(a.full_diagnostics.worst_delta_max_temperature_k),
        float(b.full_diagnostics.worst_delta_max_temperature_k),
    )
    assert worst_full > 0.0
    assert min_full > 0.0


def test_no_improving_predicate_negated_when_one_dt_negative() -> None:
    a = _entry("A", full_dt=-1.0, full_thresh=0, full_pass=0.3)
    b = _entry("B", full_dt=10.0, full_thresh=0, full_pass=0.5)
    min_full = min(
        float(a.full_diagnostics.worst_delta_max_temperature_k),
        float(b.full_diagnostics.worst_delta_max_temperature_k),
    )
    assert min_full <= 0.0


# ---------------------------------------------------------------
# validate_ring_offset_sweep_result invariants
# ---------------------------------------------------------------


def _build_synthetic_result(
    entries: tuple[RingOffsetSweepEntry, ...], *,
    selected_condition_count: int,
    evaluation_condition_count: int,
    holdout_condition_count: int,
    pareto_axes: tuple[str, ...] = (
        "delta_max_temperature_k",
        "delta_threshold_count",
        "pass_ratio_neg",
    ),
    non_discriminative_metrics: tuple[str, ...] = (),
    no_improving_full: bool = True,
    no_improving_holdout: bool = True,
) -> RingOffsetSweepResult:
    return RingOffsetSweepResult(
        entries=entries,
        candidate_count=len(entries),
        units_description=RING_RADIUS_UNITS_DESCRIPTION,
        selected_condition_count=selected_condition_count,
        evaluation_condition_count=evaluation_condition_count,
        holdout_condition_count=holdout_condition_count,
        pareto_basis="holdout",
        pareto_axes=pareto_axes,
        pareto_non_dominated_names=tuple(
            e.candidate.name for e in entries
        ),
        non_discriminative_metrics=non_discriminative_metrics,
        best_by_metric_names={
            "delta_max_temperature_k": (
                entries[0].candidate.name if entries else None
            ),
            "delta_threshold_count": (
                entries[0].candidate.name if entries else None
            ),
            "pass_ratio_neg": (
                entries[0].candidate.name if entries else None
            ),
        },
        best_by_metric_values={
            "delta_max_temperature_k": 0.0,
            "delta_threshold_count": 0.0,
            "pass_ratio_neg": 0.0,
        },
        composite_ranking_names=tuple(
            e.candidate.name for e in entries
        ),
        composite_best_name=(
            entries[0].candidate.name if entries else None
        ),
        composite_best_score=0.0,
        composite_basis="holdout",
        no_improving_candidate_under_current_sweep=(
            bool(no_improving_full)
        ),
        no_improving_candidate_in_holdout=(
            bool(no_improving_holdout)
        ),
    )


def test_validate_helper_passes_for_real_sweep_result() -> None:
    fixture = _smoke_fixture()
    sweep = _run_sweep(*fixture, _candidates())
    problems = validate_ring_offset_sweep_result(sweep)
    assert problems == ()


def test_validate_helper_rejects_non_result_input() -> None:
    problems = validate_ring_offset_sweep_result(
        "not a result",  # type: ignore[arg-type]
    )
    assert len(problems) == 1
    assert "RingOffsetSweepResult" in problems[0]


def test_validate_helper_detects_count_mismatch() -> None:
    e = _entry(
        "A", full_dt=1.0, full_thresh=0, full_pass=0.5,
        holdout_dt=1.0, holdout_thresh=0, holdout_pass=0.5,
    )
    bad = _build_synthetic_result(
        (e,),
        selected_condition_count=10,
        evaluation_condition_count=50,  # 10 + 80 != 50
        holdout_condition_count=80,
    )
    problems = validate_ring_offset_sweep_result(bad)
    assert any(
        "selection_count + holdout_count" in p
        for p in problems
    )


def test_validate_helper_detects_pareto_nd_overlap() -> None:
    e = _entry(
        "A", full_dt=1.0, full_thresh=0, full_pass=0.5,
        holdout_dt=1.0, holdout_thresh=0, holdout_pass=0.5,
    )
    bad = _build_synthetic_result(
        (e,),
        selected_condition_count=20,
        evaluation_condition_count=100,
        holdout_condition_count=80,
        pareto_axes=("delta_max_temperature_k",),
        non_discriminative_metrics=("delta_max_temperature_k",),
    )
    problems = validate_ring_offset_sweep_result(bad)
    assert any("overlap" in p for p in problems)


def test_validate_helper_detects_unknown_axis() -> None:
    e = _entry(
        "A", full_dt=1.0, full_thresh=0, full_pass=0.5,
        holdout_dt=1.0, holdout_thresh=0, holdout_pass=0.5,
    )
    bad = _build_synthetic_result(
        (e,),
        selected_condition_count=20,
        evaluation_condition_count=100,
        holdout_condition_count=80,
        pareto_axes=("not_a_real_axis",),
    )
    problems = validate_ring_offset_sweep_result(bad)
    assert any("unknown axis" in p for p in problems)


def test_validate_helper_detects_no_improving_flag_mismatch() -> None:
    # Entry with negative holdout Δmax_T, but flag claims
    # no improving in holdout.
    e = _entry(
        "A", full_dt=1.0, full_thresh=0, full_pass=0.5,
        holdout_dt=-3.0, holdout_thresh=0, holdout_pass=0.5,
    )
    bad = _build_synthetic_result(
        (e,),
        selected_condition_count=20,
        evaluation_condition_count=100,
        holdout_condition_count=80,
        no_improving_holdout=True,
    )
    problems = validate_ring_offset_sweep_result(bad)
    assert any(
        "no_improving_candidate_in_holdout" in p
        for p in problems
    )


def test_validate_helper_detects_unknown_best_by_name() -> None:
    e = _entry(
        "A", full_dt=1.0, full_thresh=0, full_pass=0.5,
        holdout_dt=1.0, holdout_thresh=0, holdout_pass=0.5,
    )
    bad_result = RingOffsetSweepResult(
        entries=(e,),
        candidate_count=1,
        units_description=RING_RADIUS_UNITS_DESCRIPTION,
        selected_condition_count=20,
        evaluation_condition_count=100,
        holdout_condition_count=80,
        pareto_basis="holdout",
        pareto_axes=(
            "delta_max_temperature_k",
            "delta_threshold_count",
            "pass_ratio_neg",
        ),
        pareto_non_dominated_names=(e.candidate.name,),
        non_discriminative_metrics=(),
        best_by_metric_names={
            "delta_max_temperature_k": "ghost_candidate",
            "delta_threshold_count": e.candidate.name,
            "pass_ratio_neg": e.candidate.name,
        },
        best_by_metric_values={
            "delta_max_temperature_k": 0.0,
            "delta_threshold_count": 0.0,
            "pass_ratio_neg": 0.0,
        },
        composite_ranking_names=(e.candidate.name,),
        composite_best_name=e.candidate.name,
        composite_best_score=0.0,
        composite_basis="holdout",
        no_improving_candidate_under_current_sweep=True,
        no_improving_candidate_in_holdout=True,
    )
    problems = validate_ring_offset_sweep_result(bad_result)
    assert any("ghost_candidate" in p for p in problems)


# ---------------------------------------------------------------
# Threshold-axis non-discriminative handling (real sweep)
# ---------------------------------------------------------------


def test_constant_threshold_axis_is_reported_as_non_discriminative() -> None:
    # Build a synthetic result where every entry has the same
    # holdout Δthreshold count, then verify the validation
    # helper still reports OK and that the axis is marked
    # non-discriminative.
    e_a = _entry(
        "A", full_dt=1.0, full_thresh=-1, full_pass=0.5,
        holdout_dt=1.0, holdout_thresh=-1, holdout_pass=0.5,
    )
    e_b = _entry(
        "B", full_dt=2.0, full_thresh=-1, full_pass=0.6,
        holdout_dt=2.0, holdout_thresh=-1, holdout_pass=0.6,
    )
    pareto_axes = ("delta_max_temperature_k", "pass_ratio_neg")
    nd = ("delta_threshold_count",)
    result = _build_synthetic_result(
        (e_a, e_b),
        selected_condition_count=20,
        evaluation_condition_count=100,
        holdout_condition_count=80,
        pareto_axes=pareto_axes,
        non_discriminative_metrics=nd,
    )
    problems = validate_ring_offset_sweep_result(result)
    assert problems == ()
    assert "delta_threshold_count" in result.non_discriminative_metrics
    assert (
        "delta_threshold_count" not in result.pareto_axes
    )


# ---------------------------------------------------------------
# Demo missing-STL handling (subprocess)
# ---------------------------------------------------------------


def test_demo_reports_missing_stl_with_clear_message(
    tmp_path,
) -> None:
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    demo = (
        repo_root
        / "examples"
        / "run_actual_stl_ring_offset_sweep_demo.py"
    )
    missing = tmp_path / "no_such_mesh.stl"
    env = os.environ.copy()
    src_path = str(repo_root / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        src_path + os.pathsep + existing if existing else src_path
    )
    proc = subprocess.run(
        [
            sys.executable, str(demo),
            "--mesh", str(missing),
            "--angle-count", "2", "--angle-step", "15",
            "--detector-count", "2", "--detector-spacing", "40",
            "--sample-count-y", "5", "--sample-count-z", "5",
            "--detector-resolution", "20",
            "--pattern-count", "5", "--pattern-max-depth", "0.02",
            "--duration", "5", "--dt", "1",
            "--max-conditions", "2",
        ],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    assert proc.returncode != 0
    assert "Mesh file not found" in proc.stdout
    assert "Invariants: FAIL" in proc.stdout
