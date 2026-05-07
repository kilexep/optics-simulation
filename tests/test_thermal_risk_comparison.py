import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from optics_simulation.thermal import (
    AngleDistanceThermalRiskComparisonResult,
    AngleDistanceThermalRiskScanResult,
    PerAngleDistanceThermalRiskComparison,
    PerAngleDistanceThermalRiskResult,
    ThermalError,
    ThermalRiskGuardrailConfig,
    ThermalRiskMetricComparison,
    ThermalRiskMetrics,
    compare_angle_distance_thermal_risk_scans,
    compare_thermal_risk_metrics,
)


def _baseline_metrics(
    *,
    max_temperature_k: float = 400.0,
    max_temperature_rise_k: float = 100.0,
    top_percent_max_temperature_rise_k: float = 90.0,
    threshold_exceeded_count: int = 50,
    threshold_exceeded_fraction: float = 0.05,
    threshold_exceeded_area: float | None = 100.0,
    mean_max_temperature_rise_k: float = 30.0,
    mean_final_temperature_k: float = 320.0,
) -> ThermalRiskMetrics:
    return ThermalRiskMetrics(
        max_temperature_k=float(max_temperature_k),
        max_temperature_rise_k=float(max_temperature_rise_k),
        final_temperature_k=float(max_temperature_k),
        final_temperature_rise_k=float(max_temperature_rise_k),
        mean_max_temperature_k=350.0,
        mean_max_temperature_rise_k=float(mean_max_temperature_rise_k),
        mean_final_temperature_k=float(mean_final_temperature_k),
        top_percent_max_temperature_rise_k=float(
            top_percent_max_temperature_rise_k
        ),
        threshold_temp_k=373.15,
        threshold_exceeded_count=int(threshold_exceeded_count),
        threshold_exceeded_fraction=float(threshold_exceeded_fraction),
        threshold_exceeded_area=(
            None if threshold_exceeded_area is None
            else float(threshold_exceeded_area)
        ),
        mean_time_to_threshold_s=10.0,
        min_time_to_threshold_s=5.0,
        max_time_to_threshold_s=15.0,
        exceedance_duration_sum_s=None,
        exceedance_degree_seconds_sum_ks=None,
        pixel_count=1000,
        pixel_area=1.0,
        top_percent=1.0,
    )


def _candidate_with(
    base: ThermalRiskMetrics,
    *,
    delta_max: float = 0.0,
    delta_top: float = 0.0,
    delta_count: int = 0,
    delta_area: float | None = 0.0,
) -> ThermalRiskMetrics:
    new_count = int(base.threshold_exceeded_count) + int(delta_count)
    new_fraction = (
        float(new_count) / float(base.pixel_count)
        if base.pixel_count > 0 else 0.0
    )
    if (
        base.threshold_exceeded_area is None
        or delta_area is None
    ):
        new_area: float | None = None
    else:
        new_area = float(base.threshold_exceeded_area) + float(delta_area)
    return replace(
        base,
        max_temperature_k=float(base.max_temperature_k) + float(delta_max),
        max_temperature_rise_k=(
            float(base.max_temperature_rise_k) + float(delta_max)
        ),
        final_temperature_k=(
            float(base.final_temperature_k) + float(delta_max)
        ),
        final_temperature_rise_k=(
            float(base.final_temperature_rise_k) + float(delta_max)
        ),
        top_percent_max_temperature_rise_k=(
            float(base.top_percent_max_temperature_rise_k)
            + float(delta_top)
        ),
        threshold_exceeded_count=new_count,
        threshold_exceeded_fraction=new_fraction,
        threshold_exceeded_area=new_area,
    )


def test_compare_returns_thermal_risk_metric_comparison() -> None:
    b = _baseline_metrics()
    c = _baseline_metrics()
    out = compare_thermal_risk_metrics(b, c)
    assert isinstance(out, ThermalRiskMetricComparison)
    assert out.metric_type == "thermal_risk_metric_comparison"


def test_unchanged_metrics_label_is_unchanged() -> None:
    b = _baseline_metrics()
    out = compare_thermal_risk_metrics(b, b)
    assert out.tradeoff_label == "unchanged"
    assert out.passes_guardrails is True
    assert out.guardrail_violations == ()


def test_lower_all_key_metrics_label_is_improved() -> None:
    b = _baseline_metrics()
    c = _candidate_with(
        b, delta_max=-5.0, delta_top=-3.0, delta_count=-2,
        delta_area=-10.0,
    )
    out = compare_thermal_risk_metrics(b, c)
    assert out.tradeoff_label == "improved"
    assert out.passes_guardrails is True


def test_higher_all_key_metrics_label_is_worsened() -> None:
    b = _baseline_metrics()
    c = _candidate_with(
        b, delta_max=+5.0, delta_top=+3.0, delta_count=+2,
        delta_area=+10.0,
    )
    out = compare_thermal_risk_metrics(b, c)
    assert out.tradeoff_label == "worsened"
    assert out.passes_guardrails is False


def test_mixed_deltas_label_is_mixed() -> None:
    b = _baseline_metrics()
    c = _candidate_with(
        b, delta_max=+5.0, delta_top=+3.0, delta_count=-2,
        delta_area=-10.0,
    )
    out = compare_thermal_risk_metrics(b, c)
    assert out.tradeoff_label == "mixed"


def test_max_temperature_increase_violates_default_guardrail() -> None:
    b = _baseline_metrics()
    c = _candidate_with(b, delta_max=+0.5)
    out = compare_thermal_risk_metrics(b, c)
    assert out.passes_guardrails is False
    assert "max_temperature_k" in out.guardrail_violations


def test_top_percent_rise_increase_violates_default_guardrail() -> None:
    b = _baseline_metrics()
    c = _candidate_with(b, delta_top=+0.5)
    out = compare_thermal_risk_metrics(b, c)
    assert out.passes_guardrails is False
    assert "top_percent_max_temperature_rise_k" in out.guardrail_violations


def test_threshold_count_increase_violates_default_guardrail() -> None:
    b = _baseline_metrics()
    c = _candidate_with(b, delta_count=+1)
    out = compare_thermal_risk_metrics(b, c)
    assert out.passes_guardrails is False
    assert "threshold_exceeded_count" in out.guardrail_violations


def test_threshold_area_increase_violates_default_guardrail_when_available() -> None:
    b = _baseline_metrics(threshold_exceeded_area=100.0)
    c = _candidate_with(b, delta_area=+0.25)
    out = compare_thermal_risk_metrics(b, c)
    assert out.passes_guardrails is False
    assert "threshold_exceeded_area" in out.guardrail_violations


def test_threshold_area_guardrail_skipped_when_area_none() -> None:
    b = _baseline_metrics(threshold_exceeded_area=None)
    c = _candidate_with(b, delta_max=0.0, delta_top=0.0, delta_count=0)
    out = compare_thermal_risk_metrics(b, c)
    assert out.delta_threshold_exceeded_area is None
    assert "threshold_exceeded_area" not in out.guardrail_violations


def test_custom_guardrail_limits_allow_small_increases() -> None:
    b = _baseline_metrics()
    c = _candidate_with(
        b, delta_max=+0.5, delta_top=+0.4, delta_count=+1,
        delta_area=+1.0,
    )
    cfg = ThermalRiskGuardrailConfig(
        max_temperature_increase_limit_k=1.0,
        top_percent_rise_increase_limit_k=1.0,
        threshold_count_increase_limit=2,
        threshold_area_increase_limit=2.0,
    )
    out = compare_thermal_risk_metrics(b, c, guardrails=cfg)
    assert out.passes_guardrails is True
    assert out.guardrail_violations == ()


def test_invalid_baseline_or_candidate_type_raises() -> None:
    b = _baseline_metrics()
    with pytest.raises(ThermalError, match="baseline"):
        compare_thermal_risk_metrics(object(), b)  # type: ignore[arg-type]
    with pytest.raises(ThermalError, match="candidate"):
        compare_thermal_risk_metrics(b, object())  # type: ignore[arg-type]


def test_invalid_guardrail_values_raise() -> None:
    b = _baseline_metrics()
    bad_max = ThermalRiskGuardrailConfig(
        max_temperature_increase_limit_k=-1.0
    )
    with pytest.raises(
        ThermalError, match="max_temperature_increase_limit_k"
    ):
        compare_thermal_risk_metrics(b, b, guardrails=bad_max)
    bad_top = ThermalRiskGuardrailConfig(
        top_percent_rise_increase_limit_k=float("inf")
    )
    with pytest.raises(
        ThermalError, match="top_percent_rise_increase_limit_k"
    ):
        compare_thermal_risk_metrics(b, b, guardrails=bad_top)
    bad_count = ThermalRiskGuardrailConfig(
        threshold_count_increase_limit=-1
    )
    with pytest.raises(
        ThermalError, match="threshold_count_increase_limit"
    ):
        compare_thermal_risk_metrics(b, b, guardrails=bad_count)
    bad_area = ThermalRiskGuardrailConfig(
        threshold_area_increase_limit=-0.1
    )
    with pytest.raises(
        ThermalError, match="threshold_area_increase_limit"
    ):
        compare_thermal_risk_metrics(b, b, guardrails=bad_area)
    bad_tol = ThermalRiskGuardrailConfig(tolerance=-1.0)
    with pytest.raises(ThermalError, match="tolerance"):
        compare_thermal_risk_metrics(b, b, guardrails=bad_tol)
    with pytest.raises(ThermalError, match="ThermalRiskGuardrailConfig"):
        compare_thermal_risk_metrics(
            b, b, guardrails=object(),  # type: ignore[arg-type]
        )


def _build_per_entry(
    *, angle: float, z: float, metrics: ThermalRiskMetrics,
) -> PerAngleDistanceThermalRiskResult:
    return PerAngleDistanceThermalRiskResult(
        angle_degrees=float(angle),
        detector_z=float(z),
        optical_c99=0.5,
        optical_cmax=0.7,
        detector_hits=10,
        final_ray_count=10,
        termination_reason="completed_interfaces",
        thermal_metrics=metrics,
    )


def _build_scan(
    entries: list[PerAngleDistanceThermalRiskResult],
) -> AngleDistanceThermalRiskScanResult:
    if not entries:
        return AngleDistanceThermalRiskScanResult(
            per_result=(),
            result_count=0,
            max_temperature_angle=None,
            max_temperature_detector_z=None,
            max_temperature_k=None,
            max_temperature_rise_k=None,
            max_top_percent_temperature_rise_angle=None,
            max_top_percent_temperature_rise_detector_z=None,
            max_top_percent_temperature_rise_k=None,
            max_threshold_exceeded_count_angle=None,
            max_threshold_exceeded_count_detector_z=None,
            max_threshold_exceeded_count=None,
            top_percent=1.0,
        )
    return AngleDistanceThermalRiskScanResult(
        per_result=tuple(entries),
        result_count=len(entries),
        max_temperature_angle=float(entries[0].angle_degrees),
        max_temperature_detector_z=float(entries[0].detector_z),
        max_temperature_k=float(
            entries[0].thermal_metrics.max_temperature_k
        ),
        max_temperature_rise_k=float(
            entries[0].thermal_metrics.max_temperature_rise_k
        ),
        max_top_percent_temperature_rise_angle=float(
            entries[0].angle_degrees
        ),
        max_top_percent_temperature_rise_detector_z=float(
            entries[0].detector_z
        ),
        max_top_percent_temperature_rise_k=float(
            entries[0]
            .thermal_metrics.top_percent_max_temperature_rise_k
        ),
        max_threshold_exceeded_count_angle=float(
            entries[0].angle_degrees
        ),
        max_threshold_exceeded_count_detector_z=float(
            entries[0].detector_z
        ),
        max_threshold_exceeded_count=int(
            entries[0].thermal_metrics.threshold_exceeded_count
        ),
        top_percent=1.0,
    )


def test_scan_comparison_preserves_ordering() -> None:
    b = _baseline_metrics()
    c1 = _candidate_with(b, delta_max=+1.0)
    c2 = _candidate_with(b, delta_count=+5)
    baseline_scan = _build_scan([
        _build_per_entry(angle=0.0, z=-100.0, metrics=b),
        _build_per_entry(angle=10.0, z=-100.0, metrics=b),
    ])
    candidate_scan = _build_scan([
        _build_per_entry(angle=0.0, z=-100.0, metrics=c1),
        _build_per_entry(angle=10.0, z=-100.0, metrics=c2),
    ])
    out = compare_angle_distance_thermal_risk_scans(
        baseline_scan, candidate_scan,
    )
    assert isinstance(out, AngleDistanceThermalRiskComparisonResult)
    assert out.result_count == 2
    pairs = [(p.angle_degrees, p.detector_z) for p in out.per_result]
    assert pairs == [(0.0, -100.0), (10.0, -100.0)]
    for p in out.per_result:
        assert isinstance(
            p, PerAngleDistanceThermalRiskComparison,
        )


def test_scan_comparison_counts_pass_fail() -> None:
    b = _baseline_metrics()
    pass_c = _baseline_metrics()
    fail_c = _candidate_with(b, delta_max=+5.0)
    baseline_scan = _build_scan([
        _build_per_entry(angle=0.0, z=-100.0, metrics=b),
        _build_per_entry(angle=10.0, z=-100.0, metrics=b),
    ])
    candidate_scan = _build_scan([
        _build_per_entry(angle=0.0, z=-100.0, metrics=pass_c),
        _build_per_entry(angle=10.0, z=-100.0, metrics=fail_c),
    ])
    out = compare_angle_distance_thermal_risk_scans(
        baseline_scan, candidate_scan,
    )
    assert out.pass_count == 1
    assert out.fail_count == 1
    assert out.pass_count + out.fail_count == out.result_count


def test_scan_comparison_counts_tradeoff_labels() -> None:
    b = _baseline_metrics()
    improved = _candidate_with(
        b, delta_max=-1.0, delta_top=-1.0, delta_count=-1, delta_area=-1.0,
    )
    worsened = _candidate_with(
        b, delta_max=+1.0, delta_top=+1.0, delta_count=+1, delta_area=+1.0,
    )
    mixed = _candidate_with(
        b, delta_max=+1.0, delta_count=-1, delta_area=-1.0,
    )
    unchanged = _baseline_metrics()
    baseline_scan = _build_scan([
        _build_per_entry(angle=a, z=-100.0, metrics=b)
        for a in (0.0, 10.0, 20.0, 30.0)
    ])
    candidate_scan = _build_scan([
        _build_per_entry(angle=0.0, z=-100.0, metrics=improved),
        _build_per_entry(angle=10.0, z=-100.0, metrics=worsened),
        _build_per_entry(angle=20.0, z=-100.0, metrics=mixed),
        _build_per_entry(angle=30.0, z=-100.0, metrics=unchanged),
    ])
    out = compare_angle_distance_thermal_risk_scans(
        baseline_scan, candidate_scan,
    )
    assert out.improved_count == 1
    assert out.worsened_count == 1
    assert out.mixed_count == 1
    assert out.unchanged_count == 1


def test_scan_comparison_worst_delta_max_temperature_aggregate() -> None:
    b = _baseline_metrics()
    c1 = _candidate_with(b, delta_max=+1.0)
    c2 = _candidate_with(b, delta_max=+5.0)
    c3 = _candidate_with(b, delta_max=+3.0)
    baseline_scan = _build_scan([
        _build_per_entry(angle=0.0, z=-100.0, metrics=b),
        _build_per_entry(angle=10.0, z=-100.0, metrics=b),
        _build_per_entry(angle=20.0, z=-100.0, metrics=b),
    ])
    candidate_scan = _build_scan([
        _build_per_entry(angle=0.0, z=-100.0, metrics=c1),
        _build_per_entry(angle=10.0, z=-100.0, metrics=c2),
        _build_per_entry(angle=20.0, z=-100.0, metrics=c3),
    ])
    out = compare_angle_distance_thermal_risk_scans(
        baseline_scan, candidate_scan,
    )
    assert out.worst_delta_max_temperature_k == pytest.approx(5.0)
    assert out.worst_delta_max_temperature_angle == 10.0
    assert out.worst_delta_max_temperature_detector_z == -100.0


def test_scan_comparison_best_delta_threshold_count_aggregate() -> None:
    b = _baseline_metrics()
    c1 = _candidate_with(b, delta_count=+1)
    c2 = _candidate_with(b, delta_count=-3)
    c3 = _candidate_with(b, delta_count=-1)
    baseline_scan = _build_scan([
        _build_per_entry(angle=0.0, z=-100.0, metrics=b),
        _build_per_entry(angle=10.0, z=-100.0, metrics=b),
        _build_per_entry(angle=20.0, z=-100.0, metrics=b),
    ])
    candidate_scan = _build_scan([
        _build_per_entry(angle=0.0, z=-100.0, metrics=c1),
        _build_per_entry(angle=10.0, z=-100.0, metrics=c2),
        _build_per_entry(angle=20.0, z=-100.0, metrics=c3),
    ])
    out = compare_angle_distance_thermal_risk_scans(
        baseline_scan, candidate_scan,
    )
    assert out.best_delta_threshold_count == -3
    assert out.best_delta_threshold_count_angle == 10.0
    assert out.best_delta_threshold_count_detector_z == -100.0


def test_scan_comparison_empty_inputs_return_empty_result() -> None:
    empty = _build_scan([])
    out = compare_angle_distance_thermal_risk_scans(empty, empty)
    assert out.result_count == 0
    assert out.per_result == ()
    assert out.pass_count == 0
    assert out.fail_count == 0
    assert out.mixed_count == 0
    assert out.improved_count == 0
    assert out.worsened_count == 0
    assert out.unchanged_count == 0
    assert out.worst_delta_max_temperature_k is None
    assert out.worst_delta_max_temperature_angle is None
    assert out.worst_delta_max_temperature_detector_z is None
    assert out.best_delta_threshold_count is None
    assert out.best_delta_threshold_count_angle is None
    assert out.best_delta_threshold_count_detector_z is None


def test_scan_comparison_mismatched_ordering_raises() -> None:
    b = _baseline_metrics()
    baseline_scan = _build_scan([
        _build_per_entry(angle=0.0, z=-100.0, metrics=b),
        _build_per_entry(angle=10.0, z=-100.0, metrics=b),
    ])
    candidate_scan = _build_scan([
        _build_per_entry(angle=0.0, z=-100.0, metrics=b),
        _build_per_entry(angle=20.0, z=-100.0, metrics=b),
    ])
    with pytest.raises(ThermalError, match="ordering mismatch"):
        compare_angle_distance_thermal_risk_scans(
            baseline_scan, candidate_scan,
        )

    short_candidate = _build_scan([
        _build_per_entry(angle=0.0, z=-100.0, metrics=b),
    ])
    with pytest.raises(ThermalError, match="result_count"):
        compare_angle_distance_thermal_risk_scans(
            baseline_scan, short_candidate,
        )


def test_scan_comparison_invalid_input_type_raises() -> None:
    b = _baseline_metrics()
    scan = _build_scan([
        _build_per_entry(angle=0.0, z=-100.0, metrics=b),
    ])
    with pytest.raises(ThermalError, match="baseline_scan"):
        compare_angle_distance_thermal_risk_scans(
            object(), scan,  # type: ignore[arg-type]
        )
    with pytest.raises(ThermalError, match="candidate_scan"):
        compare_angle_distance_thermal_risk_scans(
            scan, object(),  # type: ignore[arg-type]
        )


def test_no_file_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    b = _baseline_metrics()
    before = set(os.listdir(tmp_path))
    compare_thermal_risk_metrics(b, b)
    scan = _build_scan([
        _build_per_entry(angle=0.0, z=-100.0, metrics=b),
    ])
    compare_angle_distance_thermal_risk_scans(scan, scan)
    after = set(os.listdir(tmp_path))
    assert before == after
