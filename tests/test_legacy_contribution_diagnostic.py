import numpy as np
import pytest

from optics_simulation.contribution import (
    ContributionError,
    ContributionMap,
    ContributionMapOverlapDiagnostic,
    LegacyHotspotContributionResult,
    PatternInducedHotspotDiagnostic,
    build_pattern_induced_hotspot_diagnostic,
    compare_legacy_contribution_maps,
)
from optics_simulation.geometry import (
    create_subdivided_synthetic_bottle_body,
)
from optics_simulation.optics import (
    LegacyPatternedPetWaterSetup,
    LegacyPetWaterTraceSetup,
    create_legacy_patterned_pet_water_setup,
    create_legacy_pet_water_trace_setup,
    run_legacy_pet_water_angle_distance_scan,
)
from optics_simulation.thermal import (
    compare_legacy_thermal_risk_scans,
    compute_thermal_risk_over_legacy_optical_scan,
)


def _empty_map(nv: int = 4, nu: int = 6) -> ContributionMap:
    return ContributionMap(
        count_map=np.zeros((nv, nu), dtype=np.int64),
        weight_map=np.zeros((nv, nu), dtype=float),
        normalized_map=np.zeros((nv, nu), dtype=float),
        resolution=(nv, nu),
        total_selected=0,
        total_weight=0.0,
        selected_ray_indices=np.empty(0, dtype=np.int64),
        u_bin_indices=np.empty(0, dtype=np.int64),
        v_bin_indices=np.empty(0, dtype=np.int64),
    )


def _filled_map(weights, nv: int = 4, nu: int = 6) -> ContributionMap:
    arr = np.asarray(weights, dtype=float).reshape(nv, nu)
    counts = (arr > 0.0).astype(np.int64)
    norm = arr / arr.max() if arr.max() > 0.0 else np.zeros_like(arr)
    return ContributionMap(
        count_map=counts,
        weight_map=arr.astype(float, copy=True),
        normalized_map=norm.astype(float, copy=True),
        resolution=(nv, nu),
        total_selected=int(counts.sum()),
        total_weight=float(arr.sum()),
        selected_ray_indices=np.empty(0, dtype=np.int64),
        u_bin_indices=np.empty(0, dtype=np.int64),
        v_bin_indices=np.empty(0, dtype=np.int64),
    )


def test_compare_returns_diagnostic() -> None:
    base = _empty_map()
    cand = _empty_map()
    result = compare_legacy_contribution_maps(base, cand)
    assert isinstance(result, ContributionMapOverlapDiagnostic)


def test_compare_overlap_counts_correct() -> None:
    nv, nu = 4, 6
    b = np.zeros((nv, nu), dtype=float)
    c = np.zeros((nv, nu), dtype=float)
    # Shared bin at (0, 0).
    b[0, 0] = 1.0
    c[0, 0] = 2.0
    # Baseline-only bin at (1, 1).
    b[1, 1] = 1.5
    # Candidate-only bins at (2, 2), (3, 3).
    c[2, 2] = 3.0
    c[3, 3] = 4.0
    base = _filled_map(b, nv=nv, nu=nu)
    cand = _filled_map(c, nv=nv, nu=nu)
    result = compare_legacy_contribution_maps(base, cand)
    assert int(result.baseline_nonzero_bins) == 2
    assert int(result.candidate_nonzero_bins) == 3
    assert int(result.shared_nonzero_bins) == 1
    assert int(result.baseline_only_bins) == 1
    assert int(result.candidate_only_bins) == 2


def test_candidate_new_bin_fraction_handles_empty_candidate() -> None:
    nv, nu = 4, 6
    b = np.zeros((nv, nu), dtype=float)
    b[0, 0] = 1.0
    c = np.zeros((nv, nu), dtype=float)
    base = _filled_map(b, nv=nv, nu=nu)
    cand = _filled_map(c, nv=nv, nu=nu)
    result = compare_legacy_contribution_maps(base, cand)
    assert float(result.candidate_new_bin_fraction) == 0.0
    assert float(result.overlap_fraction_of_candidate) == 0.0


def test_candidate_new_bin_fraction_full_new() -> None:
    nv, nu = 4, 6
    b = np.zeros((nv, nu), dtype=float)
    c = np.zeros((nv, nu), dtype=float)
    c[0, 0] = 1.0
    c[1, 2] = 2.0
    c[3, 5] = 3.0
    base = _filled_map(b, nv=nv, nu=nu)
    cand = _filled_map(c, nv=nv, nu=nu)
    result = compare_legacy_contribution_maps(base, cand)
    assert int(result.candidate_only_bins) == 3
    assert float(result.candidate_new_bin_fraction) == pytest.approx(
        1.0,
    )


def test_compare_weight_sums_correct() -> None:
    nv, nu = 4, 6
    b = np.zeros((nv, nu), dtype=float)
    c = np.zeros((nv, nu), dtype=float)
    b[0, 0] = 1.0
    c[0, 0] = 2.5
    c[1, 1] = 3.0
    base = _filled_map(b, nv=nv, nu=nu)
    cand = _filled_map(c, nv=nv, nu=nu)
    result = compare_legacy_contribution_maps(base, cand)
    assert float(result.shared_candidate_weight) == pytest.approx(2.5)
    assert float(result.candidate_only_weight) == pytest.approx(3.0)
    assert float(result.candidate_total_weight) == pytest.approx(5.5)
    assert float(result.baseline_total_weight) == pytest.approx(1.0)


def test_compare_shape_mismatch_raises() -> None:
    base = _empty_map(nv=4, nu=6)
    cand = _empty_map(nv=4, nu=8)
    with pytest.raises(ContributionError, match="shape"):
        compare_legacy_contribution_maps(base, cand)


def test_compare_invalid_input_type_raises() -> None:
    cand = _empty_map()
    with pytest.raises(ContributionError, match="ContributionMap"):
        compare_legacy_contribution_maps(
            "not a map",  # type: ignore[arg-type]
            cand,
        )


def _smoke_setups():
    mesh = create_subdivided_synthetic_bottle_body(
        radius=30.0, height=120.0,
        sections=32, height_segments=8,
    )
    baseline = create_legacy_pet_water_trace_setup(
        mesh,
        target_height=225.6, target_diameter=72.1,
        wall_thickness=0.3, inner_offset_mode="auto",
    )
    patterned = create_legacy_patterned_pet_water_setup(
        mesh,
        target_height=225.6, target_diameter=72.1,
        wall_thickness=0.3, inner_offset_mode="auto",
        pattern_count=10,
        pattern_sigma_u=0.05, pattern_sigma_v=0.05,
        pattern_max_depth=0.05,
        pattern_seed=1,
    )
    return baseline, patterned


def _patterned_to_legacy(
    setup: LegacyPatternedPetWaterSetup,
) -> LegacyPetWaterTraceSetup:
    original = setup.original_setup
    return LegacyPetWaterTraceSetup(
        shell_mesh=setup.patterned_shell_mesh,
        water_mesh=setup.patterned_water_mesh,
        step_specs=setup.patterned_step_specs,
        scale_report=original.scale_report,
        inner_offset_report=setup.patterned_inner_offset_report,
        target_height=original.target_height,
        target_diameter=original.target_diameter,
        wall_thickness=original.wall_thickness,
        ior_air=original.ior_air,
        ior_pet=original.ior_pet,
        ior_water=original.ior_water,
    )


def _comparison_fixture():
    baseline, patterned = _smoke_setups()
    candidate_legacy = _patterned_to_legacy(patterned)
    angles = (0.0, 15.0)
    distances = (100.0, 140.0)
    scan_kwargs = dict(
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
    thermal_kwargs = dict(
        nominal_incident_irradiance_w_m2=1000.0,
        duration_s=5.0,
        dt_s=1.0,
        areal_heat_capacity_j_m2k=1200.0,
        absorptivity=0.8,
        h_conv_w_m2k=10.0,
        emissivity=0.9,
        threshold_temp_k=373.15,
    )
    base_optical = run_legacy_pet_water_angle_distance_scan(
        setup=baseline, **scan_kwargs,
    )
    cand_optical = run_legacy_pet_water_angle_distance_scan(
        setup=candidate_legacy, **scan_kwargs,
    )
    base_thermal = compute_thermal_risk_over_legacy_optical_scan(
        base_optical, **thermal_kwargs,
    )
    cand_thermal = compute_thermal_risk_over_legacy_optical_scan(
        cand_optical, **thermal_kwargs,
    )
    comparison = compare_legacy_thermal_risk_scans(
        base_thermal, cand_thermal,
    )
    return baseline, candidate_legacy, comparison


def test_diagnostic_returns_pattern_induced_hotspot_diagnostic() -> None:
    baseline, candidate_legacy, comparison = _comparison_fixture()
    diag = build_pattern_induced_hotspot_diagnostic(
        baseline_setup=baseline,
        candidate_setup=candidate_legacy,
        comparison_result=comparison,
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
    assert isinstance(diag, PatternInducedHotspotDiagnostic)
    assert isinstance(
        diag.baseline_contribution, LegacyHotspotContributionResult,
    )
    assert isinstance(
        diag.candidate_contribution, LegacyHotspotContributionResult,
    )
    assert isinstance(diag.overlap, ContributionMapOverlapDiagnostic)


def test_diagnostic_selects_entry_within_comparison() -> None:
    baseline, candidate_legacy, comparison = _comparison_fixture()
    diag = build_pattern_induced_hotspot_diagnostic(
        baseline_setup=baseline,
        candidate_setup=candidate_legacy,
        comparison_result=comparison,
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
    # The selected (angle, distance) must match one of the
    # comparison entries.
    keys = {
        (
            float(e.angle_degrees),
            float(e.detector_distance),
        )
        for e in comparison.entries
    }
    assert (
        float(diag.worsened_angle_degrees),
        float(diag.worsened_detector_distance),
    ) in keys


def test_diagnostic_resolution_matches_request() -> None:
    baseline, candidate_legacy, comparison = _comparison_fixture()
    res = (16, 32)
    diag = build_pattern_induced_hotspot_diagnostic(
        baseline_setup=baseline,
        candidate_setup=candidate_legacy,
        comparison_result=comparison,
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
    assert (
        diag.baseline_contribution.contribution_map.weight_map.shape
        == res
    )
    assert (
        diag.candidate_contribution.contribution_map.weight_map.shape
        == res
    )


def test_diagnostic_no_file_output(tmp_path) -> None:
    baseline, candidate_legacy, comparison = _comparison_fixture()
    before = sorted(tmp_path.iterdir())
    build_pattern_induced_hotspot_diagnostic(
        baseline_setup=baseline,
        candidate_setup=candidate_legacy,
        comparison_result=comparison,
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
