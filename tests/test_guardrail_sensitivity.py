import os
from dataclasses import replace
from pathlib import Path

import pytest

from optics_simulation.thermal import (
    AngleDistanceThermalRiskComparisonResult,
    AngleDistanceThermalRiskScanResult,
    GuardrailScenario,
    GuardrailSensitivityEntry,
    GuardrailSensitivityResult,
    PerAngleDistanceThermalRiskResult,
    ThermalError,
    ThermalRiskGuardrailConfig,
    ThermalRiskMetrics,
    evaluate_guardrail_sensitivity,
    evaluate_scan_guardrail_sensitivity,
)


def _baseline_metrics(
    *,
    max_temperature_k: float = 400.0,
    max_temperature_rise_k: float = 100.0,
    top_percent_max_temperature_rise_k: float = 90.0,
    threshold_exceeded_count: int = 50,
    threshold_exceeded_area: float | None = 100.0,
) -> ThermalRiskMetrics:
    return ThermalRiskMetrics(
        max_temperature_k=float(max_temperature_k),
        max_temperature_rise_k=float(max_temperature_rise_k),
        final_temperature_k=float(max_temperature_k),
        final_temperature_rise_k=float(max_temperature_rise_k),
        mean_max_temperature_k=350.0,
        mean_max_temperature_rise_k=30.0,
        mean_final_temperature_k=320.0,
        top_percent_max_temperature_rise_k=float(
            top_percent_max_temperature_rise_k
        ),
        threshold_temp_k=373.15,
        threshold_exceeded_count=int(threshold_exceeded_count),
        threshold_exceeded_fraction=(
            float(threshold_exceeded_count) / 1000.0
        ),
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
    new_fraction = float(new_count) / float(base.pixel_count)
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


def _strict_scenario() -> GuardrailScenario:
    return GuardrailScenario(
        name="strict",
        guardrails=ThermalRiskGuardrailConfig(),
        description="default strict synthetic guardrail",
    )


def _measurement_scenario() -> GuardrailScenario:
    return GuardrailScenario(
        name="measurement_0p1K",
        guardrails=ThermalRiskGuardrailConfig(
            max_temperature_increase_limit_k=0.1,
            top_percent_rise_increase_limit_k=0.1,
            threshold_count_increase_limit=0,
            threshold_area_increase_limit=0.0,
            tolerance=1e-9,
        ),
        description="measurement-tolerance guardrail (synthetic)",
    )


def _relaxed_scenario() -> GuardrailScenario:
    return GuardrailScenario(
        name="relaxed_1K",
        guardrails=ThermalRiskGuardrailConfig(
            max_temperature_increase_limit_k=1.0,
            top_percent_rise_increase_limit_k=1.0,
            threshold_count_increase_limit=0,
            threshold_area_increase_limit=0.0,
            tolerance=1e-9,
        ),
        description="relaxed 1K guardrail (synthetic)",
    )


def test_evaluate_returns_guardrail_sensitivity_result() -> None:
    b = _baseline_metrics()
    out = evaluate_guardrail_sensitivity(
        b, b, [_strict_scenario(), _measurement_scenario()],
    )
    assert isinstance(out, GuardrailSensitivityResult)
    assert out.metric_type == "guardrail_sensitivity_diagnostic"
    for e in out.entries:
        assert isinstance(e, GuardrailSensitivityEntry)


def test_empty_scenarios_returns_empty_result() -> None:
    b = _baseline_metrics()
    out = evaluate_guardrail_sensitivity(b, b, [])
    assert out.scenario_count == 0
    assert out.entries == ()
    assert out.pass_count == 0
    assert out.fail_count == 0
    assert out.mixed_count == 0
    assert out.improved_count == 0
    assert out.worsened_count == 0
    assert out.unchanged_count == 0


def test_scenario_order_preserved() -> None:
    b = _baseline_metrics()
    scenarios = [
        _strict_scenario(),
        _measurement_scenario(),
        _relaxed_scenario(),
    ]
    out = evaluate_guardrail_sensitivity(b, b, scenarios)
    names = [e.scenario_name for e in out.entries]
    assert names == ["strict", "measurement_0p1K", "relaxed_1K"]


def test_pass_fail_counts_consistent() -> None:
    b = _baseline_metrics()
    c = _candidate_with(b, delta_max=+0.05)  # passes measurement, fails strict
    out = evaluate_guardrail_sensitivity(
        b, c,
        [_strict_scenario(), _measurement_scenario(), _relaxed_scenario()],
    )
    assert out.pass_count + out.fail_count == out.scenario_count
    assert out.pass_count == 2  # measurement + relaxed
    assert out.fail_count == 1  # strict


def test_label_counts_consistent() -> None:
    b = _baseline_metrics()
    c = _candidate_with(b, delta_max=+0.05, delta_count=-1)  # mixed
    out = evaluate_guardrail_sensitivity(
        b, c,
        [_strict_scenario(), _measurement_scenario(), _relaxed_scenario()],
    )
    total_labels = (
        out.mixed_count + out.improved_count
        + out.worsened_count + out.unchanged_count
    )
    assert total_labels == out.scenario_count
    assert out.mixed_count == 3


def test_guardrail_violations_propagated() -> None:
    b = _baseline_metrics()
    c = _candidate_with(b, delta_max=+0.5)
    out = evaluate_guardrail_sensitivity(
        b, c,
        [_strict_scenario(), _relaxed_scenario()],
    )
    strict_entry = out.entries[0]
    relaxed_entry = out.entries[1]
    assert "max_temperature_k" in strict_entry.guardrail_violations
    assert relaxed_entry.guardrail_violations == ()


def test_relaxed_passes_when_strict_fails() -> None:
    b = _baseline_metrics()
    c = _candidate_with(b, delta_max=+0.5)
    out = evaluate_guardrail_sensitivity(
        b, c,
        [_strict_scenario(), _relaxed_scenario()],
    )
    assert out.entries[0].passes_guardrails is False
    assert out.entries[1].passes_guardrails is True


def test_invalid_baseline_type_raises() -> None:
    b = _baseline_metrics()
    with pytest.raises(ThermalError, match="baseline"):
        evaluate_guardrail_sensitivity(
            object(), b, [_strict_scenario()],  # type: ignore[arg-type]
        )


def test_invalid_candidate_type_raises() -> None:
    b = _baseline_metrics()
    with pytest.raises(ThermalError, match="candidate"):
        evaluate_guardrail_sensitivity(
            b, object(), [_strict_scenario()],  # type: ignore[arg-type]
        )


def test_invalid_scenario_type_raises() -> None:
    b = _baseline_metrics()
    with pytest.raises(ThermalError, match="GuardrailScenario"):
        evaluate_guardrail_sensitivity(
            b, b, [object()],  # type: ignore[list-item]
        )

    bad_inner = GuardrailScenario(
        name="broken",
        guardrails=object(),  # type: ignore[arg-type]
        description="invalid",
    )
    with pytest.raises(ThermalError, match="ThermalRiskGuardrailConfig"):
        evaluate_guardrail_sensitivity(b, b, [bad_inner])

    with pytest.raises(ThermalError, match="scenarios"):
        evaluate_guardrail_sensitivity(
            b, b, object(),  # type: ignore[arg-type]
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
    return AngleDistanceThermalRiskScanResult(
        per_result=tuple(entries),
        result_count=len(entries),
        max_temperature_angle=(
            float(entries[0].angle_degrees) if entries else None
        ),
        max_temperature_detector_z=(
            float(entries[0].detector_z) if entries else None
        ),
        max_temperature_k=(
            float(entries[0].thermal_metrics.max_temperature_k)
            if entries else None
        ),
        max_temperature_rise_k=(
            float(entries[0].thermal_metrics.max_temperature_rise_k)
            if entries else None
        ),
        max_top_percent_temperature_rise_angle=(
            float(entries[0].angle_degrees) if entries else None
        ),
        max_top_percent_temperature_rise_detector_z=(
            float(entries[0].detector_z) if entries else None
        ),
        max_top_percent_temperature_rise_k=(
            float(
                entries[0]
                .thermal_metrics.top_percent_max_temperature_rise_k
            )
            if entries else None
        ),
        max_threshold_exceeded_count_angle=(
            float(entries[0].angle_degrees) if entries else None
        ),
        max_threshold_exceeded_count_detector_z=(
            float(entries[0].detector_z) if entries else None
        ),
        max_threshold_exceeded_count=(
            int(entries[0].thermal_metrics.threshold_exceeded_count)
            if entries else None
        ),
        top_percent=1.0,
    )


def test_evaluate_scan_preserves_scenario_order() -> None:
    b = _baseline_metrics()
    c = _candidate_with(b, delta_max=+0.05)
    baseline_scan = _build_scan([
        _build_per_entry(angle=0.0, z=-100.0, metrics=b),
    ])
    candidate_scan = _build_scan([
        _build_per_entry(angle=0.0, z=-100.0, metrics=c),
    ])
    pairs = evaluate_scan_guardrail_sensitivity(
        baseline_scan, candidate_scan,
        [_strict_scenario(), _measurement_scenario(), _relaxed_scenario()],
    )
    assert tuple(name for name, _ in pairs) == (
        "strict", "measurement_0p1K", "relaxed_1K",
    )


def test_evaluate_scan_returns_comparison_per_scenario() -> None:
    b = _baseline_metrics()
    c = _candidate_with(b, delta_max=+0.05)
    baseline_scan = _build_scan([
        _build_per_entry(angle=0.0, z=-100.0, metrics=b),
        _build_per_entry(angle=10.0, z=-100.0, metrics=b),
    ])
    candidate_scan = _build_scan([
        _build_per_entry(angle=0.0, z=-100.0, metrics=c),
        _build_per_entry(angle=10.0, z=-100.0, metrics=c),
    ])
    pairs = evaluate_scan_guardrail_sensitivity(
        baseline_scan, candidate_scan,
        [_strict_scenario(), _relaxed_scenario()],
    )
    assert len(pairs) == 2
    for name, comparison in pairs:
        assert isinstance(
            comparison, AngleDistanceThermalRiskComparisonResult,
        )
        assert comparison.result_count == 2

    strict_pair = pairs[0]
    relaxed_pair = pairs[1]
    assert strict_pair[1].fail_count == 2
    assert relaxed_pair[1].pass_count == 2

    empty = evaluate_scan_guardrail_sensitivity(
        baseline_scan, candidate_scan, [],
    )
    assert empty == ()


def test_evaluate_scan_invalid_input_types_raise() -> None:
    b = _baseline_metrics()
    scan = _build_scan([
        _build_per_entry(angle=0.0, z=-100.0, metrics=b),
    ])
    with pytest.raises(ThermalError, match="baseline_scan"):
        evaluate_scan_guardrail_sensitivity(
            object(), scan, [_strict_scenario()],  # type: ignore[arg-type]
        )
    with pytest.raises(ThermalError, match="candidate_scan"):
        evaluate_scan_guardrail_sensitivity(
            scan, object(), [_strict_scenario()],  # type: ignore[arg-type]
        )


def test_no_file_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    b = _baseline_metrics()
    before = set(os.listdir(tmp_path))
    evaluate_guardrail_sensitivity(
        b, b, [_strict_scenario(), _relaxed_scenario()],
    )
    scan = _build_scan([
        _build_per_entry(angle=0.0, z=-100.0, metrics=b),
    ])
    evaluate_scan_guardrail_sensitivity(
        scan, scan, [_strict_scenario()],
    )
    after = set(os.listdir(tmp_path))
    assert before == after
