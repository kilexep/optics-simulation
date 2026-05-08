import numpy as np
import pytest

from optics_simulation.thermal import (
    LegacyThermalRiskComparisonEntry,
    LegacyThermalRiskComparisonResult,
    LegacyThermalRiskEntry,
    LegacyThermalRiskScanResult,
    ThermalError,
    ThermalRiskGuardrailConfig,
    ThermalRiskMetrics,
    compare_legacy_thermal_risk_scans,
)


def _make_metrics(
    *,
    max_temp: float,
    max_rise: float,
    final_temp: float = 295.0,
    final_rise: float = 1.0,
    mean_max_temp: float = 300.0,
    mean_max_rise: float = 5.0,
    mean_final_temp: float = 295.0,
    top_pct_rise: float = 10.0,
    threshold_count: int = 0,
    threshold_fraction: float = 0.0,
    threshold_area: float | None = None,
    pixel_count: int = 1600,
    pixel_area: float | None = None,
) -> ThermalRiskMetrics:
    return ThermalRiskMetrics(
        max_temperature_k=float(max_temp),
        max_temperature_rise_k=float(max_rise),
        final_temperature_k=float(final_temp),
        final_temperature_rise_k=float(final_rise),
        mean_max_temperature_k=float(mean_max_temp),
        mean_max_temperature_rise_k=float(mean_max_rise),
        mean_final_temperature_k=float(mean_final_temp),
        top_percent_max_temperature_rise_k=float(top_pct_rise),
        threshold_temp_k=373.15,
        threshold_exceeded_count=int(threshold_count),
        threshold_exceeded_fraction=float(threshold_fraction),
        threshold_exceeded_area=threshold_area,
        mean_time_to_threshold_s=None,
        min_time_to_threshold_s=None,
        max_time_to_threshold_s=None,
        exceedance_duration_sum_s=None,
        exceedance_degree_seconds_sum_ks=None,
        pixel_count=int(pixel_count),
        pixel_area=pixel_area,
        top_percent=1.0,
    )


def _make_entry(
    *, angle: float, distance: float, metrics: ThermalRiskMetrics,
) -> LegacyThermalRiskEntry:
    return LegacyThermalRiskEntry(
        angle_degrees=float(angle),
        detector_distance=float(distance),
        optical_c99=1.0,
        optical_cmax=1.5,
        max_relative_irradiance=2.0,
        detector_hits=10,
        final_ray_count=10,
        incident_flux_map_max_w_m2=1000.0,
        thermal_metrics=metrics,
    )


def _make_scan(
    entries: list[LegacyThermalRiskEntry],
) -> LegacyThermalRiskScanResult:
    return LegacyThermalRiskScanResult(
        entries=tuple(entries),
        entry_count=len(entries),
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


def _make_simple_pair():
    baseline_entries = [
        _make_entry(
            angle=0.0, distance=100.0,
            metrics=_make_metrics(
                max_temp=400.0, max_rise=100.0,
                threshold_count=10, top_pct_rise=50.0,
            ),
        ),
        _make_entry(
            angle=15.0, distance=100.0,
            metrics=_make_metrics(
                max_temp=410.0, max_rise=110.0,
                threshold_count=12, top_pct_rise=55.0,
            ),
        ),
    ]
    candidate_entries = [
        _make_entry(
            angle=0.0, distance=100.0,
            metrics=_make_metrics(
                max_temp=395.0, max_rise=95.0,
                threshold_count=8, top_pct_rise=48.0,
            ),
        ),
        _make_entry(
            angle=15.0, distance=100.0,
            metrics=_make_metrics(
                max_temp=415.0, max_rise=115.0,
                threshold_count=14, top_pct_rise=58.0,
            ),
        ),
    ]
    return _make_scan(baseline_entries), _make_scan(candidate_entries)


def test_returns_legacy_thermal_risk_comparison_result() -> None:
    baseline, candidate = _make_simple_pair()
    result = compare_legacy_thermal_risk_scans(baseline, candidate)
    assert isinstance(result, LegacyThermalRiskComparisonResult)
    assert result.metric_type == "legacy_thermal_risk_comparison"
    for entry in result.entries:
        assert isinstance(entry, LegacyThermalRiskComparisonEntry)


def test_empty_scans_return_empty_result() -> None:
    empty = _make_scan([])
    result = compare_legacy_thermal_risk_scans(empty, empty)
    assert result.entries == ()
    assert result.entry_count == 0
    assert result.pass_count == 0
    assert result.worst_delta_max_temperature_k is None
    assert result.best_delta_max_temperature_k is None
    assert result.best_delta_threshold_count is None


def test_mismatched_entry_count_raises() -> None:
    baseline = _make_scan(
        [
            _make_entry(
                angle=0.0, distance=100.0,
                metrics=_make_metrics(
                    max_temp=400.0, max_rise=100.0,
                ),
            ),
        ]
    )
    candidate = _make_scan(
        [
            _make_entry(
                angle=0.0, distance=100.0,
                metrics=_make_metrics(
                    max_temp=400.0, max_rise=100.0,
                ),
            ),
            _make_entry(
                angle=15.0, distance=100.0,
                metrics=_make_metrics(
                    max_temp=400.0, max_rise=100.0,
                ),
            ),
        ]
    )
    with pytest.raises(ThermalError, match="entry_count"):
        compare_legacy_thermal_risk_scans(baseline, candidate)


def test_mismatched_angle_order_raises() -> None:
    baseline = _make_scan(
        [
            _make_entry(
                angle=0.0, distance=100.0,
                metrics=_make_metrics(
                    max_temp=400.0, max_rise=100.0,
                ),
            ),
        ]
    )
    candidate = _make_scan(
        [
            _make_entry(
                angle=15.0, distance=100.0,
                metrics=_make_metrics(
                    max_temp=400.0, max_rise=100.0,
                ),
            ),
        ]
    )
    with pytest.raises(ThermalError, match="ordering mismatch"):
        compare_legacy_thermal_risk_scans(baseline, candidate)


def test_pass_fail_counts_with_default_strict_guardrails() -> None:
    baseline, candidate = _make_simple_pair()
    result = compare_legacy_thermal_risk_scans(baseline, candidate)
    # Entry 0: candidate cooler / fewer threshold pixels => passes.
    # Entry 1: candidate hotter / more threshold pixels => fails.
    assert result.entry_count == 2
    assert result.pass_count + result.fail_count == result.entry_count
    assert result.pass_count == 1
    assert result.fail_count == 1


def test_tradeoff_label_counts_are_correct() -> None:
    baseline, candidate = _make_simple_pair()
    result = compare_legacy_thermal_risk_scans(baseline, candidate)
    total_label = (
        result.improved_count
        + result.worsened_count
        + result.unchanged_count
        + result.mixed_count
    )
    assert total_label == result.entry_count
    # Entry 0: all key deltas <= -tol => improved.
    # Entry 1: all key deltas >= +tol => worsened.
    assert result.improved_count == 1
    assert result.worsened_count == 1
    assert result.mixed_count == 0
    assert result.unchanged_count == 0


def test_worst_delta_max_temperature_aggregate_correct() -> None:
    baseline, candidate = _make_simple_pair()
    result = compare_legacy_thermal_risk_scans(baseline, candidate)
    deltas = [
        float(e.comparison.delta_max_temperature_k)
        for e in result.entries
    ]
    expected_worst = max(deltas)
    assert result.worst_delta_max_temperature_k == pytest.approx(
        expected_worst, abs=1e-12,
    )


def test_best_delta_max_temperature_aggregate_correct() -> None:
    baseline, candidate = _make_simple_pair()
    result = compare_legacy_thermal_risk_scans(baseline, candidate)
    deltas = [
        float(e.comparison.delta_max_temperature_k)
        for e in result.entries
    ]
    expected_best = min(deltas)
    assert result.best_delta_max_temperature_k == pytest.approx(
        expected_best, abs=1e-12,
    )


def test_best_delta_threshold_count_aggregate_correct() -> None:
    baseline, candidate = _make_simple_pair()
    result = compare_legacy_thermal_risk_scans(baseline, candidate)
    deltas = [
        int(e.comparison.delta_threshold_exceeded_count)
        for e in result.entries
    ]
    expected_best = min(deltas)
    assert result.best_delta_threshold_count == int(expected_best)


def test_custom_guardrails_affect_pass_fail() -> None:
    baseline, candidate = _make_simple_pair()
    relaxed = ThermalRiskGuardrailConfig(
        max_temperature_increase_limit_k=10.0,
        top_percent_rise_increase_limit_k=10.0,
        threshold_count_increase_limit=10,
        threshold_area_increase_limit=10.0,
        tolerance=1e-9,
    )
    result = compare_legacy_thermal_risk_scans(
        baseline, candidate, guardrails=relaxed,
    )
    # With a 10K / 10-count slack, the second entry's +5K, +2 count
    # delta should now pass.
    assert result.pass_count == 2
    assert result.fail_count == 0


def test_invalid_baseline_type_raises() -> None:
    _, candidate = _make_simple_pair()
    with pytest.raises(
        ThermalError, match="LegacyThermalRiskScanResult",
    ):
        compare_legacy_thermal_risk_scans(
            "not a scan", candidate,  # type: ignore[arg-type]
        )


def test_invalid_candidate_type_raises() -> None:
    baseline, _ = _make_simple_pair()
    with pytest.raises(
        ThermalError, match="LegacyThermalRiskScanResult",
    ):
        compare_legacy_thermal_risk_scans(
            baseline, "not a scan",  # type: ignore[arg-type]
        )


def test_no_file_output(tmp_path) -> None:
    baseline, candidate = _make_simple_pair()
    before = sorted(tmp_path.iterdir())
    compare_legacy_thermal_risk_scans(baseline, candidate)
    after = sorted(tmp_path.iterdir())
    assert before == after
